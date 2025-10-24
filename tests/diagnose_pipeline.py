#!/usr/bin/env python3
"""诊断 DINOv3 → SAM2 Pipeline - 简化版"""

import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.prompt_generator import PromptConfig, generate_prompts_from_heatmap
from src.data_loader import load_image
from src.utils import to_torch_dtype

print("=" * 70)
print("诊断 DINOv3 → SAM2 Pipeline")
print("=" * 70)

# 加载图像
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"
sample = load_image(image_path)

print(f"\n图像: {sample.image.shape}")

# 加载配置
config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/model_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

prompt_config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/prompt_config.yaml"
with open(prompt_config_path, 'r') as f:
    prompt_cfg_dict = yaml.safe_load(f)

output_dir = Path("outputs/pipeline_diagnosis")
output_dir.mkdir(exist_ok=True, parents=True)

# 初始化模型
dinov3_cfg = Dinov3Config(**config["dinov3"])
device = torch.device(config["device"])
dtype = to_torch_dtype(config["dtype"])

extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
print("✅ DINOv3 加载成功")

sam2_cfg = Sam2Config(**config["sam2"])
segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
print("✅ SAM2 加载成功\n")

# ============================================================
# 步骤 1: 读取已生成的好的热图
# ============================================================
print("=" * 70)
print("步骤 1: 使用预先生成的好的热图")
print("=" * 70)

# 从之前的测试读取物体性或组合热图
good_heatmap_path = "outputs/dinov3_simple_viz/02_objectness_overlay.png"
if not Path(good_heatmap_path).exists():
    print(f"错误: 请先运行 test_dinov3_simple.py 生成热图")
    print(f"预期路径: {good_heatmap_path}")
    sys.exit(1)

# 读取并转换回热图
good_overlay = cv2.imread(good_heatmap_path)
# 我们需要重新生成热图，所以直接使用简化的方法

print("重新生成物体性热图...")

# 简化版物体性计算
with torch.no_grad():
    # 提取特征（使用 extract 可以正确处理 dtype）
    feats = extractor.extract(sample.image)
    
    # 为了获取 patch tokens，我们需要绕过 monkey patch
    # 暂时移除我们的 forward hook
    if hasattr(extractor.model, 'blocks') and len(extractor.model.blocks) > 0:
        last_block = extractor.model.blocks[-1]
        if hasattr(last_block, 'attn'):
            # 保存被修改的 forward
            modified_forward = last_block.attn.forward
            
            # 临时恢复为能输出特征的版本
            # 重新运行一次以获取完整特征
            
            # 简化：直接使用中间层特征
            intermediate = extractor.model.get_intermediate_layers(
                extractor._prepare(sample.image),
                n=1,
                return_class_token=False
            )
            
            patch_features = intermediate[0]  # [B, N, D]
            patch_features = patch_features[0]  # [N, D]
            
            # 恢复 forward
            # last_block.attn.forward = modified_forward

H = W = int(np.sqrt(len(patch_features)))
print(f"✅ 获取 patch 特征: {H}x{W}")

# 计算物体性
import torch.nn.functional as F
patch_features_norm = F.normalize(patch_features, dim=-1)
similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())

K = 20
topk_sim, _ = torch.topk(similarity_matrix, k=K+1, dim=1, largest=True)
avg_similarity = topk_sim[:, 1:].mean(dim=1)
objectness = (1 - avg_similarity).cpu().numpy()

# 重塑并调整大小
objectness_2d = objectness[:H*W].reshape(H, W)
objectness_norm = (objectness_2d - objectness_2d.min()) / \
                 (objectness_2d.max() - objectness_2d.min() + 1e-8)

good_heatmap = cv2.resize(
    objectness_norm,
    (sample.image.shape[1], sample.image.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

print(f"✅ 生成物体性热图")
print(f"   范围: [{good_heatmap.min():.3f}, {good_heatmap.max():.3f}]")

# 保存好的热图
heatmap_vis = (good_heatmap * 255).astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
cv2.imwrite(str(output_dir / "good_heatmap.png"), heatmap_colored)

overlay = cv2.addWeighted(
    cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.6,
    heatmap_colored, 0.4, 0
)
cv2.imwrite(str(output_dir / "good_heatmap_overlay.png"), overlay)

# ============================================================
# 步骤 2: 使用当前 pipeline 的热图（attention 方法）
# ============================================================
print("\n" + "=" * 70)
print("步骤 2: 当前 Pipeline 的热图（Attention）")
print("=" * 70)

# 重新提取以使用 attention
feats = extractor.extract(sample.image)
current_heatmap = extractor.attention_to_heatmap(
    feats["attention"],
    (sample.image.shape[1], sample.image.shape[0])
)

print(f"当前 attention 热图:")
print(f"   范围: [{current_heatmap.min():.3f}, {current_heatmap.max():.3f}]")

# 保存当前热图
heatmap_vis = (current_heatmap * 255).astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
cv2.imwrite(str(output_dir / "current_heatmap.png"), heatmap_colored)

overlay = cv2.addWeighted(
    cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.6,
    heatmap_colored, 0.4, 0
)
cv2.imwrite(str(output_dir / "current_heatmap_overlay.png"), overlay)

# ============================================================
# 步骤 3: 比较两种热图生成的 prompts
# ============================================================
print("\n" + "=" * 70)
print("步骤 3: 比较生成的 Prompts")
print("=" * 70)

prompt_params = {}
if "attention" in prompt_cfg_dict:
    prompt_params.update(prompt_cfg_dict["attention"])
if "points" in prompt_cfg_dict:
    prompt_params.update(prompt_cfg_dict["points"])
prompt_config = PromptConfig(**prompt_params)

print(f"Prompt 配置:")
print(f"  percentile: {prompt_config.percentile}")
print(f"  min_component_area: {prompt_config.min_component_area}")

# 从好的热图生成 prompts
good_boxes, good_points, good_labels = generate_prompts_from_heatmap(
    good_heatmap, prompt_config
)

print(f"\n从物体性热图生成:")
print(f"  Box 数量: {len(good_boxes)}")
if good_boxes:
    for i, box in enumerate(good_boxes[:5]):
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        print(f"    Box {i}: [{x0}, {y0}, {x1}, {y1}], area={area}")

# 从当前热图生成 prompts
current_boxes, current_points, current_labels = generate_prompts_from_heatmap(
    current_heatmap, prompt_config
)

print(f"\n从 attention 热图生成:")
print(f"  Box 数量: {len(current_boxes)}")
if current_boxes:
    for i, box in enumerate(current_boxes[:5]):
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        print(f"    Box {i}: [{x0}, {y0}, {x1}, {y1}], area={area}")

# 可视化 boxes
def draw_boxes(image, boxes, color, name):
    img_with_boxes = image.copy()
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img_with_boxes, f"{i}", (x0, y0-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(
        str(output_dir / f"{name}_boxes.png"),
        cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    )
    return img_with_boxes

if good_boxes:
    draw_boxes(sample.image, good_boxes, (0, 255, 0), "good")
    print(f"  ✅ 保存物体性方法的 boxes")
    
if current_boxes:
    draw_boxes(sample.image, current_boxes, (255, 0, 0), "current")
    print(f"  ✅ 保存 attention 方法的 boxes")

# ============================================================
# 步骤 4: 用两种 prompts 运行 SAM2 并比较结果
# ============================================================
print("\n" + "=" * 70)
print("步骤 4: SAM2 分割对比")
print("=" * 70)

def run_sam2_and_visualize(boxes, points, labels, name):
    if not boxes:
        print(f"  {name}: 没有 boxes，跳过")
        return []
    
    masks = segmenter.segment_batched(
        sample.image,
        boxes,
        points=points if points else None,
        labels=labels if labels else None,
        batch_size=32,
    )
    
    print(f"\n{name} SAM2 结果:")
    print(f"  生成 {len(masks)} 个 masks")
    
    area_threshold = config["pipeline"]["area_threshold"]
    valid_count = 0
    total_area = 0
    for i, mask in enumerate(masks):
        area = mask.sum()
        total_area += area
        passed = area >= area_threshold
        if passed:
            valid_count += 1
        status = "✅" if passed else "❌"
        if i < 10:  # 只打印前 10 个
            print(f"    Mask {i}: area={area:.0f} {status}")
    
    print(f"  通过阈值: {valid_count}/{len(masks)}")
    print(f"  平均面积: {total_area/len(masks):.0f}")
    
    # 可视化所有 masks
    combined = np.zeros_like(sample.image)
    for i, mask in enumerate(masks):
        color = np.array([
            (i * 50) % 255,
            (i * 80) % 255,
            (i * 120) % 255
        ], dtype=np.uint8)
        combined[mask.astype(bool)] = color
    
    cv2.imwrite(
        str(output_dir / f"{name}_masks.png"),
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    )
    
    return masks

good_masks = run_sam2_and_visualize(good_boxes, good_points, good_labels, "objectness")
current_masks = run_sam2_and_visualize(current_boxes, current_points, current_labels, "attention")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)

print(f"\n1. 热图质量:")
print(f"   物体性方法: [{good_heatmap.min():.3f}, {good_heatmap.max():.3f}]")
print(f"   Attention 方法: [{current_heatmap.min():.3f}, {current_heatmap.max():.3f}]")

print(f"\n2. Prompt 生成:")
print(f"   物体性方法: {len(good_boxes)} boxes")
print(f"   Attention 方法: {len(current_boxes)} boxes")

print(f"\n3. SAM2 分割:")
print(f"   物体性方法: {len(good_masks)} masks")
print(f"   Attention 方法: {len(current_masks)} masks")

print(f"\n4. 问题分析:")
if len(good_boxes) > len(current_boxes) * 1.5:
    print(f"   ⚠️ Attention 热图质量差，生成的 boxes 太少")
    print(f"   ⚠️ 主要问题：Attention 方法不适合物体检测")
    print(f"   ✅ 解决方案：替换为物体性方法")
elif len(good_boxes) > 0 and len(current_boxes) > 0:
    good_avg = sum(m.sum() for m in good_masks) / len(good_masks) if good_masks else 0
    current_avg = sum(m.sum() for m in current_masks) / len(current_masks) if current_masks else 0
    print(f"   两种方法生成的 boxes 数量相近")
    print(f"   物体性方法平均 mask 面积: {good_avg:.0f}")
    print(f"   Attention 方法平均 mask 面积: {current_avg:.0f}")
    if current_avg < good_avg * 0.5:
        print(f"   ⚠️ Attention 方法的 boxes 质量较差")
        print(f"   ✅ 解决方案：替换为物体性方法")

print(f"\n5. 推荐:")
print(f"   ✅ 使用物体性方法替换 attention_to_heatmap")
print(f"   ✅ 或者降低面积阈值以适应 attention 方法")

print(f"\n所有结果保存在: {output_dir}/")
print("\n关键对比文件:")
print("  - good_heatmap_overlay.png vs current_heatmap_overlay.png")
print("  - good_boxes.png vs current_boxes.png")  
print("  - objectness_masks.png vs attention_masks.png")