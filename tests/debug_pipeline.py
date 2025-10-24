#!/usr/bin/env python3
"""调试脚本：查看 DINOv3 + SAM2 pipeline 的中间结果"""

import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.prompt_generator import PromptConfig, generate_prompts_from_heatmap
from src.data_loader import load_image
from src.utils import to_torch_dtype

print("=" * 70)
print("DINOv3 + SAM2 Pipeline 调试")
print("=" * 70)

# 加载配置
config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/model_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

prompt_config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/prompt_config.yaml"
with open(prompt_config_path, 'r') as f:
    prompt_cfg_dict = yaml.safe_load(f)

# 加载图像
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"
sample = load_image(image_path)

print(f"\n图像信息:")
print(f"  路径: {image_path}")
print(f"  形状: {sample.image.shape}")
print(f"  dtype: {sample.image.dtype}")
print(f"  值范围: [{sample.image.min()}, {sample.image.max()}]")

# 初始化模型
print("\n" + "=" * 70)
print("步骤 1: 初始化 DINOv3")
print("=" * 70)

dinov3_cfg = Dinov3Config(**config["dinov3"])
device = torch.device(config["device"])
dtype = to_torch_dtype(config["dtype"])

extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
print("✅ DINOv3 加载成功")

# 提取特征和 attention
print("\n" + "=" * 70)
print("步骤 2: 提取 DINOv3 attention")
print("=" * 70)

feats = extractor.extract(sample.image)
print(f"特征形状: {feats['features'].shape}")
print(f"Attention 形状: {feats['attention'].shape if feats['attention'] is not None else None}")

# 生成 heatmap
heatmap = extractor.attention_to_heatmap(
    feats["attention"], 
    (sample.image.shape[1], sample.image.shape[0])
)
print(f"Heatmap 形状: {heatmap.shape}")
print(f"Heatmap 值范围: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

# 保存 heatmap 可视化
heatmap_vis = (heatmap * 255).astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/debug")
output_dir.mkdir(exist_ok=True, parents=True)
cv2.imwrite(str(output_dir / "heatmap.png"), heatmap_colored)
print(f"✅ Heatmap 已保存到: {output_dir / 'heatmap.png'}")

# 生成 prompts
print("\n" + "=" * 70)
print("步骤 3: 生成 Prompts")
print("=" * 70)

prompt_params = {}
if "attention" in prompt_cfg_dict:
    prompt_params.update(prompt_cfg_dict["attention"])
if "points" in prompt_cfg_dict:
    prompt_params.update(prompt_cfg_dict["points"])
prompt_config = PromptConfig(**prompt_params)

boxes, points, labels = generate_prompts_from_heatmap(heatmap, prompt_config)

print(f"生成的 box 数量: {len(boxes)}")
if boxes:
    print(f"前 5 个 boxes:")
    for i, box in enumerate(boxes[:5]):
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        print(f"  Box {i}: [{x0}, {y0}, {x1}, {y1}], area={area}")

# 在图像上绘制 boxes
img_with_boxes = sample.image.copy()
for i, box in enumerate(boxes):
    x0, y0, x1, y1 = box
    cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(img_with_boxes, f"{i}", (x0, y0-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imwrite(str(output_dir / "boxes.png"), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
print(f"✅ Boxes 已保存到: {output_dir / 'boxes.png'}")

if not boxes:
    print("❌ 没有生成任何 box，无法继续")
    sys.exit(1)

# 初始化 SAM2
print("\n" + "=" * 70)
print("步骤 4: 运行 SAM2 分割")
print("=" * 70)

sam2_cfg = Sam2Config(**config["sam2"])
segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
print("✅ SAM2 加载成功")

# 执行分割
masks = segmenter.segment_batched(
    sample.image,
    boxes,
    points=points if points else None,
    labels=labels if labels else None,
    batch_size=config["pipeline"].get("max_prompts_per_batch"),
)

print(f"\n生成的 mask 数量: {len(masks)}")

# 分析每个 mask
print("\nMask 分析:")
area_threshold = config["pipeline"]["area_threshold"]
print(f"配置的面积阈值: {area_threshold}")

valid_masks = []
for i, mask in enumerate(masks):
    area = float(mask.astype(np.uint8).sum())
    passed = area >= area_threshold
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Mask {i}: area={area:.0f} {status}")
    if passed:
        valid_masks.append((i, mask))

print(f"\n通过阈值的 mask 数量: {len(valid_masks)}/{len(masks)}")

if not valid_masks:
    print("\n❌ 没有 mask 通过面积阈值!")
    print(f"   最大 mask 面积: {max(m.sum() for m in masks):.0f}")
    print(f"   配置的阈值: {area_threshold}")
    print(f"\n建议:")
    print(f"   1. 降低 area_threshold (当前: {area_threshold})")
    print(f"   2. 调整 prompt_config 中的 percentile (当前: {prompt_config.percentile})")
    print(f"   3. 降低 min_component_area (当前: {prompt_config.min_component_area})")
else:
    print(f"\n✅ 有 {len(valid_masks)} 个 mask 通过阈值")
    # 保存 mask 可视化
    for i, (idx, mask) in enumerate(valid_masks[:3]):
        mask_vis = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / f"mask_{idx}.png"), mask_vis)
    print(f"✅ 前 3 个有效 mask 已保存到: {output_dir}/mask_*.png")

# 保存所有 mask 的组合可视化
combined = np.zeros_like(sample.image)
for i, mask in enumerate(masks):
    color = np.array([
        (i * 50) % 255,
        (i * 80) % 255,
        (i * 120) % 255
    ], dtype=np.uint8)
    combined[mask.astype(bool)] = color

cv2.imwrite(str(output_dir / "all_masks.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
print(f"\n✅ 所有 mask 的可视化已保存到: {output_dir / 'all_masks.png'}")

print("\n" + "=" * 70)
print("调试完成！请查看输出目录: outputs/debug/")
print("=" * 70)