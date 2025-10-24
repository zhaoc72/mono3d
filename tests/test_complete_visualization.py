#!/usr/bin/env python3
"""
完整的可视化测试脚本 - 验证每个中间步骤
"""

import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.inference_pipeline import ZeroShotSegmentationPipeline, PipelineConfig
from src.prompt_generator import ClusterConfig, PromptConfig
from src.data_loader import load_image
from src.utils import to_torch_dtype

print("=" * 80)
print("完整可视化测试 - DINOv3 + SAM2 分割")
print("=" * 80)

# ==================== 配置 ====================
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/complete_visualization_test")
output_dir.mkdir(exist_ok=True, parents=True)

print(f"\n输入图像: {image_path}")
print(f"输出目录: {output_dir}")

# ==================== 加载图像 ====================
print("\n" + "=" * 80)
print("步骤 1: 加载图像")
print("=" * 80)

sample = load_image(image_path)
image = sample.image
print(f"✅ 图像尺寸: {image.shape}")
print(f"   高度: {image.shape[0]}")
print(f"   宽度: {image.shape[1]}")
print(f"   通道: {image.shape[2]}")

# 保存原图
cv2.imwrite(
    str(output_dir / "step1_original.jpg"),
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
)

# ==================== 初始化模型 ====================
print("\n" + "=" * 80)
print("步骤 2: 初始化 DINOv3")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

print(f"设备: {device}")
print(f"精度: {dtype}")

dinov3_cfg = Dinov3Config(
    repo_or_dir="/media/pc/D/zhaochen/mono3d/dinov3",
    model_name="dinov3_vith16plus",
    checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vith16plus.pth",
    use_torch_hub=True,
    torchhub_source="local",
    output_layers=[4, 8, 12],
    layer_weights=[0.2, 0.3, 0.5],
    fusion_method="weighted_concat",
    enable_objectness=True
)

extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
print("✅ DINOv3 初始化成功")
print(f"   融合层: {dinov3_cfg.output_layers}")
print(f"   层权重: {dinov3_cfg.layer_weights}")
print(f"   融合方法: {dinov3_cfg.fusion_method}")

# ==================== 提取特征 ====================
print("\n" + "=" * 80)
print("步骤 3: 提取 DINOv3 特征")
print("=" * 80)

feats = extractor.extract_features(image)

patch_map = feats["patch_map"]
if hasattr(patch_map, "detach"):
    patch_map = patch_map.detach().cpu().numpy()

grid_h, grid_w, feat_dim = patch_map.shape

print(f"✅ 特征提取成功")
print(f"   Patch grid: {grid_h}x{grid_w}")
print(f"   特征维度: {feat_dim}")
print(f"   期望维度: {1280 * len(dinov3_cfg.output_layers)} (3层 × 1280)")

# 可视化 Objectness Map
if feats.get('objectness_map') is not None:
    objectness = feats['objectness_map']
    print(f"   Objectness Map: {objectness.shape}")
    print(f"   范围: [{objectness.min():.3f}, {objectness.max():.3f}]")
    
    # 上采样到原图尺寸
    objectness_upsampled = cv2.resize(
        objectness,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    
    # 保存热图
    obj_vis = (objectness_upsampled * 255).astype(np.uint8)
    obj_colored = cv2.applyColorMap(obj_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / "step3a_objectness.jpg"), obj_colored)
    
    # 叠加到原图
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
        obj_colored, 0.4, 0
    )
    cv2.imwrite(str(output_dir / "step3b_objectness_overlay.jpg"), overlay)
    print(f"   ✅ Objectness 可视化已保存")

# ==================== 聚类 ====================
print("\n" + "=" * 80)
print("步骤 4: 特征聚类")
print("=" * 80)

cluster_cfg = ClusterConfig(
    num_clusters=6,
    min_region_area=400,
    use_objectness_filter=True,
    objectness_threshold=0.3
)

from src.prompt_generator import kmeans_cluster, labels_to_regions

# K-Means 聚类
features = patch_map.reshape(-1, feat_dim)
labels, centroids = kmeans_cluster(features, cluster_cfg)
label_map = labels.reshape(grid_h, grid_w)

print(f"✅ 聚类完成")
print(f"   聚类数: {cluster_cfg.num_clusters}")
print(f"   标签图: {label_map.shape}")
print(f"   唯一标签: {np.unique(label_map)}")

# 可视化聚类结果
label_vis = (label_map * 255 / label_map.max()).astype(np.uint8)
label_colored = cv2.applyColorMap(label_vis, cv2.COLORMAP_JET)
label_colored_upsampled = cv2.resize(
    label_colored,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)
cv2.imwrite(str(output_dir / "step4a_clusters.jpg"), label_colored_upsampled)

overlay = cv2.addWeighted(
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
    label_colored_upsampled, 0.4, 0
)
cv2.imwrite(str(output_dir / "step4b_clusters_overlay.jpg"), overlay)
print(f"   ✅ 聚类可视化已保存")

# ==================== 生成候选区域 ====================
print("\n" + "=" * 80)
print("步骤 5: 生成候选区域")
print("=" * 80)

proposals = labels_to_regions(
    label_map,
    image.shape[:2],
    cluster_cfg,
    objectness_map=feats.get('objectness_map')
)

print(f"✅ 候选区域生成完成")
print(f"   总候选数: {len(proposals)}")

if proposals:
    print(f"\n   前 5 个候选区域:")
    for i, prop in enumerate(proposals[:5]):
        x0, y0, x1, y1 = prop.bbox
        area = (x1 - x0) * (y1 - y0)
        print(f"     {i}: bbox=[{x0:4d},{y0:4d},{x1:4d},{y1:4d}], "
              f"area={area:6.0f}, objectness={prop.objectness:.3f}")
    
    # 可视化候选框
    img_with_boxes = image.copy()
    for i, prop in enumerate(proposals):
        x0, y0, x1, y1 = prop.bbox
        cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            img_with_boxes, f"{i}", (x0, y0-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    
    cv2.imwrite(
        str(output_dir / "step5_proposals.jpg"),
        cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    )
    print(f"   ✅ 候选框可视化已保存")
else:
    print("   ⚠️ 没有生成任何候选区域!")

# ==================== 生成 SAM2 Prompts ====================
print("\n" + "=" * 80)
print("步骤 6: 生成 SAM2 Prompts")
print("=" * 80)

prompt_cfg = PromptConfig(
    include_boxes=True,
    include_points=True,
    point_strategy="centroid"
)

from src.prompt_generator import expand_region_instances, proposals_to_prompts

instance_proposals = expand_region_instances(
    proposals,
    prompt_cfg,
    cluster_cfg,
    patch_map,
    image.shape[:2]
)

boxes, points, labels_list = proposals_to_prompts(
    instance_proposals,
    prompt_cfg,
    patch_map=patch_map,
    image_shape=image.shape[:2],
    cluster_config=cluster_cfg
)

print(f"✅ Prompts 生成完成")
print(f"   实例数: {len(instance_proposals)}")
print(f"   Boxes: {len(boxes)}")
print(f"   Points: {len([p for p in points if p])}")

# 可视化 boxes + points
if boxes:
    img_with_prompts = image.copy()
    for i, (box, pts) in enumerate(zip(boxes, points)):
        x0, y0, x1, y1 = box
        cv2.rectangle(img_with_prompts, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        if pts:
            for px, py in pts:
                cv2.circle(img_with_prompts, (px, py), 5, (255, 0, 0), -1)
    
    cv2.imwrite(
        str(output_dir / "step6_prompts.jpg"),
        cv2.cvtColor(img_with_prompts, cv2.COLOR_RGB2BGR)
    )
    print(f"   ✅ Prompts 可视化已保存")

# ==================== 初始化 SAM2 ====================
print("\n" + "=" * 80)
print("步骤 7: 初始化 SAM2")
print("=" * 80)

sam2_cfg = Sam2Config(
    backend="official",
    checkpoint_path="/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt",
    model_config="sam2.1/sam2.1_hiera_l"
)

segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
print("✅ SAM2 初始化成功")

# ==================== 执行分割 ====================
print("\n" + "=" * 80)
print("步骤 8: 执行 SAM2 分割")
print("=" * 80)

masks = segmenter.segment_batched(
    image,
    boxes,
    points=points if points else None,
    labels=labels_list if labels_list else None,
    batch_size=32
)

print(f"✅ 分割完成")
print(f"   生成掩码: {len(masks)}")

# 分析掩码
area_threshold = 100
valid_masks = []
total_area = 0

print(f"\n   掩码详情 (面积阈值: {area_threshold}):")
print(f"   {'索引':<6} {'面积':<10} {'状态':<6} {'占比%':<8}")
print("   " + "-" * 40)

for i, mask in enumerate(masks):
    area = int(mask.astype(np.uint8).sum())
    total_area += area
    
    passed = area >= area_threshold
    if passed:
        valid_masks.append(mask)
    
    ratio = area / (image.shape[0] * image.shape[1]) * 100
    status = "✅" if passed else "❌"
    
    if i < 10:  # 只打印前10个
        print(f"   {i:<6} {area:<10} {status:<6} {ratio:>6.2f}%")

if len(masks) > 10:
    print(f"   ... (还有 {len(masks) - 10} 个)")

print(f"\n   总结:")
print(f"     - 总掩码: {len(masks)}")
print(f"     - 有效掩码: {len(valid_masks)} ({len(valid_masks)/len(masks)*100:.1f}%)")
print(f"     - 平均面积: {total_area/len(masks):.0f}")

# 可视化所有掩码
if masks:
    combined = np.zeros_like(image)
    for i, mask in enumerate(valid_masks if valid_masks else masks):
        color = np.array([
            (i * 50) % 255,
            (i * 80 + 60) % 255,
            (i * 120 + 30) % 255
        ], dtype=np.uint8)
        combined[mask.astype(bool)] = color
    
    cv2.imwrite(
        str(output_dir / "step8a_masks.jpg"),
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    )
    
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
    )
    cv2.imwrite(str(output_dir / "step8b_masks_overlay.jpg"), overlay)
    
    print(f"   ✅ 掩码可视化已保存")

# ==================== 创建完整对比图 ====================
print("\n" + "=" * 80)
print("步骤 9: 创建完整对比图")
print("=" * 80)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('DINOv3 + SAM2 分割流程', fontsize=16, fontweight='bold')

# 1. 原图
axes[0, 0].imshow(image)
axes[0, 0].set_title('1. 原始图像', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# 2. Objectness
if feats.get('objectness_map') is not None:
    axes[0, 1].imshow(objectness_upsampled, cmap='hot')
    axes[0, 1].set_title('2. Objectness Map', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
else:
    axes[0, 1].axis('off')

# 3. 聚类
label_rgb = cv2.cvtColor(label_colored_upsampled, cv2.COLOR_BGR2RGB)
axes[0, 2].imshow(label_rgb)
axes[0, 2].set_title('3. 特征聚类', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# 4. 候选区域
if proposals:
    axes[0, 3].imshow(img_with_boxes)
    axes[0, 3].set_title(f'4. 候选区域 ({len(proposals)}个)', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
else:
    axes[0, 3].axis('off')

# 5. Prompts
if boxes:
    axes[1, 0].imshow(img_with_prompts)
    axes[1, 0].set_title(f'5. SAM2 Prompts', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
else:
    axes[1, 0].axis('off')

# 6. 分割掩码
if masks:
    axes[1, 1].imshow(combined)
    axes[1, 1].set_title(f'6. 分割掩码 ({len(valid_masks)}个)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
else:
    axes[1, 1].axis('off')

# 7. 叠加结果
if masks:
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(overlay_rgb)
    axes[1, 2].set_title('7. 最终结果', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
else:
    axes[1, 2].axis('off')

# 8. 统计信息
axes[1, 3].axis('off')
stats_text = f"""
统计信息:

图像尺寸: {image.shape[1]}×{image.shape[0]}
特征维度: {feat_dim}

聚类数: {cluster_cfg.num_clusters}
候选区域: {len(proposals)}
实例数: {len(instance_proposals)}

总掩码: {len(masks)}
有效掩码: {len(valid_masks)}
有效率: {len(valid_masks)/len(masks)*100:.1f}%
"""
axes[1, 3].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
axes[1, 3].set_title('8. 统计信息', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "step9_complete_comparison.jpg", dpi=150, bbox_inches='tight')
plt.close()

print("✅ 完整对比图已保存")

# ==================== 生成 README ====================
readme_content = f"""# DINOv3 + SAM2 分割可视化结果

## 测试图像
- 路径: {image_path}
- 尺寸: {image.shape[1]}×{image.shape[0]}

## 处理流程

### step1_original.jpg
原始输入图像

### step3a_objectness.jpg & step3b_objectness_overlay.jpg
DINOv3 Objectness Map - 显示每个区域是否为物体的置信度

### step4a_clusters.jpg & step4b_clusters_overlay.jpg
K-Means 特征聚类结果 ({cluster_cfg.num_clusters}个聚类)

### step5_proposals.jpg
候选区域 (绿色框) - 共 {len(proposals)} 个

### step6_prompts.jpg
SAM2 Prompts - 绿色框=bounding boxes, 红点=正样本点

### step8a_masks.jpg & step8b_masks_overlay.jpg
最终分割掩码 - 共 {len(valid_masks)} 个有效掩码

### step9_complete_comparison.jpg
完整流程对比图

## 关键参数
- DINOv3 融合层: {dinov3_cfg.output_layers}
- 层权重: {dinov3_cfg.layer_weights}
- 融合方法: {dinov3_cfg.fusion_method}
- 聚类数: {cluster_cfg.num_clusters}
- Objectness 阈值: {cluster_cfg.objectness_threshold}
- 面积阈值: {area_threshold}

## 结果统计
- 候选区域: {len(proposals)}
- 生成掩码: {len(masks)}
- 有效掩码: {len(valid_masks)}
- 有效率: {len(valid_masks)/len(masks)*100:.1f}%
"""

with open(output_dir / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

# ==================== 完成 ====================
print("\n" + "=" * 80)
print("✅ 测试完成!")
print("=" * 80)

print(f"\n📁 所有结果已保存到: {output_dir}/")
print("\n生成的文件:")
print("  step1_original.jpg           - 原始图像")
print("  step3a_objectness.jpg        - Objectness 热图")
print("  step3b_objectness_overlay.jpg - Objectness 叠加")
print("  step4a_clusters.jpg          - 聚类结果")
print("  step4b_clusters_overlay.jpg  - 聚类叠加")
print("  step5_proposals.jpg          - 候选区域")
print("  step6_prompts.jpg            - SAM2 Prompts")
print("  step8a_masks.jpg             - 分割掩码")
print("  step8b_masks_overlay.jpg     - 掩码叠加")
print("  step9_complete_comparison.jpg - 完整对比")
print("  README.md                    - 详细说明")

print("\n" + "=" * 80)