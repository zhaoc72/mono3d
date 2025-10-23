#!/usr/bin/env python3
"""
完整的Zero-shot分割测试脚本 - 修复版
测试单张图像并生成详细的可视化结果
"""

import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path("/media/pc/D/zhaochen/mono3d/mono3d")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.prompt_generator import (
    PromptConfig, ClusterConfig,
    kmeans_cluster, labels_to_regions, proposals_to_prompts
)
from src.data_loader import load_image
from src.utils import to_torch_dtype

def visualize_results(image, masks, proposals, heatmap, output_dir, prefix=""):
    """创建详细的可视化结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. 保存原图
    cv2.imwrite(
        str(output_dir / f"{prefix}01_original.jpg"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    
    # 2. 保存热图
    if heatmap is not None:
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_vis = (heatmap_norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        
        # 热图单独
        cv2.imwrite(
            str(output_dir / f"{prefix}02_heatmap.jpg"),
            heatmap_colored
        )
        
        # 热图叠加
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
            heatmap_colored, 0.4, 0
        )
        cv2.imwrite(
            str(output_dir / f"{prefix}03_heatmap_overlay.jpg"),
            overlay
        )
    
    # 3. 可视化候选区域（bounding boxes）
    if proposals:
        img_boxes = image.copy()
        for i, prop in enumerate(proposals):
            x0, y0, x1, y1 = prop.bbox
            cv2.rectangle(img_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                img_boxes, f"{i}", (x0, y0-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            # 绘制质心
            cx, cy = prop.centroid
            cv2.circle(img_boxes, (cx, cy), 5, (255, 0, 0), -1)
        
        cv2.imwrite(
            str(output_dir / f"{prefix}04_proposals.jpg"),
            cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR)
        )
    
    # 4. 可视化所有分割掩码
    if masks:
        # 彩色叠加
        combined = np.zeros_like(image)
        for i, mask in enumerate(masks):
            color = np.array([
                (i * 50) % 255,
                (i * 80 + 60) % 255,
                (i * 120 + 30) % 255
            ], dtype=np.uint8)
            combined[mask.astype(bool)] = color
        
        # 掩码单独
        cv2.imwrite(
            str(output_dir / f"{prefix}05_masks.jpg"),
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        )
        
        # 掩码叠加
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
        )
        cv2.imwrite(
            str(output_dir / f"{prefix}06_masks_overlay.jpg"),
            overlay
        )
        
        # 保存前3个单独掩码
        for i, mask in enumerate(masks[:3]):
            mask_vis = (mask.astype(np.uint8) * 255)
            cv2.imwrite(
                str(output_dir / f"{prefix}07_mask_{i:02d}.jpg"),
                mask_vis
            )
    
    print(f"✅ 可视化结果已保存到: {output_dir}/")


def main():
    print("=" * 80)
    print("Zero-shot Instance Segmentation 测试")
    print("DINOv3 + SAM2 无监督分割")
    print("=" * 80)
    
    # ===================== 配置 =====================
    image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    prompt_config_path = PROJECT_ROOT / "configs" / "prompt_config.yaml"
    output_dir = PROJECT_ROOT / "outputs" / "complete_test"
    
    print(f"\n📋 配置信息:")
    print(f"  图像: {image_path}")
    print(f"  输出: {output_dir}")
    
    # ===================== 加载图像 =====================
    print(f"\n{'='*80}")
    print("步骤 1: 加载图像")
    print("=" * 80)
    
    sample = load_image(image_path)
    image = sample.image
    print(f"✅ 图像加载成功")
    print(f"   大小: {image.shape[1]}x{image.shape[0]}")
    print(f"   通道: {image.shape[2]}")
    
    # ===================== 加载配置 =====================
    print(f"\n{'='*80}")
    print("步骤 2: 加载配置")
    print("=" * 80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(prompt_config_path, 'r') as f:
        prompt_cfg_dict = yaml.safe_load(f)
    
    device = torch.device(config["device"])
    dtype = to_torch_dtype(config["dtype"])
    
    print(f"✅ 配置加载成功")
    print(f"   设备: {device}")
    print(f"   精度: {dtype}")
    
    # ===================== 初始化 DINOv3 =====================
    print(f"\n{'='*80}")
    print("步骤 3: 初始化 DINOv3")
    print("=" * 80)
    
    dinov3_cfg = Dinov3Config(**config["dinov3"])
    extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
    
    print(f"✅ DINOv3 初始化成功")
    print(f"   模型: {dinov3_cfg.model_name}")
    print(f"   权重: {dinov3_cfg.checkpoint_path}")
    
    # ===================== 提取特征 =====================
    print(f"\n{'='*80}")
    print("步骤 4: 提取 DINOv3 特征")
    print("=" * 80)
    
    feats = extractor.extract_features(image)
    patch_map = feats["patch_map"]
    
    if hasattr(patch_map, "detach"):
        patch_map = patch_map.detach().cpu().numpy()
    
    print(f"✅ 特征提取成功")
    print(f"   Patch map: {patch_map.shape}")
    print(f"   特征维度: {patch_map.shape[-1]}")
    
    # 生成热图
    attention_map = feats.get("attention_map")
    if attention_map is not None:
        print(f"   Attention map: {attention_map.shape}")
        print(f"   范围: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
    
    # ===================== 特征聚类 =====================
    print(f"\n{'='*80}")
    print("步骤 5: 特征聚类 (K-Means)")
    print("=" * 80)
    
    cluster_cfg = ClusterConfig(**config["pipeline"]["cluster"])
    
    # 重塑特征用于聚类
    features = patch_map.reshape(-1, patch_map.shape[-1])
    labels, centroids = kmeans_cluster(features, cluster_cfg)
    
    grid_size = int(np.sqrt(len(labels)))
    label_map = labels.reshape(grid_size, grid_size)
    
    print(f"✅ 聚类完成")
    print(f"   聚类数: {cluster_cfg.num_clusters}")
    print(f"   标签图: {label_map.shape}")
    print(f"   唯一标签: {np.unique(label_map)}")
    
    # ===================== 生成候选区域 =====================
    print(f"\n{'='*80}")
    print("步骤 6: 生成候选区域")
    print("=" * 80)
    
    proposals = labels_to_regions(label_map, image.shape[:2], cluster_cfg)
    
    print(f"✅ 候选区域生成完成")
    print(f"   候选数: {len(proposals)}")
    
    if proposals:
        print(f"\n   前 5 个候选区域:")
        for i, prop in enumerate(proposals[:5]):
            x0, y0, x1, y1 = prop.bbox
            area = (x1 - x0) * (y1 - y0)
            print(f"     {i}: bbox=[{x0:4d},{y0:4d},{x1:4d},{y1:4d}], "
                  f"area={area:6.0f}, centroid={prop.centroid}")
    
    if not proposals:
        print("❌ 未生成任何候选区域！")
        print("   建议: 降低 min_region_area 参数")
        return
    
    # ===================== 转换为 SAM2 Prompts =====================
    print(f"\n{'='*80}")
    print("步骤 7: 生成 SAM2 Prompts")
    print("=" * 80)
    
    # PromptConfig 只接受这3个参数
    prompt_config = PromptConfig(
        include_boxes=True,
        include_points=True,
        point_strategy="centroid"
    )
    
    boxes, points, labels_list = proposals_to_prompts(proposals, prompt_config)
    
    print(f"✅ Prompts 生成完成")
    print(f"   Boxes: {len(boxes)}")
    print(f"   Points: {len([p for p in points if p])}")
    print(f"   Prompt配置:")
    print(f"     - 包含框: {prompt_config.include_boxes}")
    print(f"     - 包含点: {prompt_config.include_points}")
    print(f"     - 点策略: {prompt_config.point_strategy}")
    
    # ===================== 初始化 SAM2 =====================
    print(f"\n{'='*80}")
    print("步骤 8: 初始化 SAM2")
    print("=" * 80)
    
    sam2_cfg = Sam2Config(**config["sam2"])
    segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
    
    print(f"✅ SAM2 初始化成功")
    print(f"   后端: {sam2_cfg.backend}")
    print(f"   权重: {sam2_cfg.checkpoint_path}")
    print(f"   配置: {sam2_cfg.model_config}")
    
    # ===================== 执行分割 =====================
    print(f"\n{'='*80}")
    print("步骤 9: 执行 SAM2 分割")
    print("=" * 80)
    
    masks = segmenter.segment_batched(
        image,
        boxes,
        points=points if points else None,
        labels=labels_list if labels_list else None,
        batch_size=config["pipeline"].get("max_prompts_per_batch", 32)
    )
    
    print(f"✅ 分割完成")
    print(f"   生成掩码: {len(masks)}")
    
    # ===================== 分析结果 =====================
    print(f"\n{'='*80}")
    print("步骤 10: 分析分割结果")
    print("=" * 80)
    
    area_threshold = config["pipeline"]["area_threshold"]
    
    valid_masks = []
    total_area = 0
    max_area = 0
    
    print(f"\n   面积阈值: {area_threshold}")
    print(f"\n   {'索引':<6} {'面积':<10} {'状态':<6} {'占比':<8}")
    print("   " + "-" * 40)
    
    for i, mask in enumerate(masks):
        area = int(mask.astype(np.uint8).sum())
        total_area += area
        max_area = max(max_area, area)
        
        passed = area >= area_threshold
        if passed:
            valid_masks.append(mask)
        
        ratio = area / (image.shape[0] * image.shape[1]) * 100
        status = "✅" if passed else "❌"
        
        print(f"   {i:<6} {area:<10} {status:<6} {ratio:>6.2f}%")
    
    print(f"\n   总结:")
    print(f"     - 总掩码: {len(masks)}")
    print(f"     - 有效掩码: {len(valid_masks)} ({len(valid_masks)/len(masks)*100:.1f}%)")
    print(f"     - 平均面积: {total_area/len(masks):.0f}")
    print(f"     - 最大面积: {max_area}")
    
    # ===================== 可视化 =====================
    print(f"\n{'='*80}")
    print("步骤 11: 生成可视化结果")
    print("=" * 80)
    
    # 上采样 attention map 用于可视化
    heatmap = None
    if attention_map is not None:
        heatmap = cv2.resize(
            attention_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
    
    visualize_results(
        image,
        valid_masks if valid_masks else masks,
        proposals,
        heatmap,
        output_dir,
        prefix=""
    )
    
    # ===================== 完成 =====================
    print(f"\n{'='*80}")
    print("✅ 测试完成！")
    print("=" * 80)
    
    print(f"\n📊 最终统计:")
    print(f"   ├─ 输入图像: {image.shape[1]}x{image.shape[0]}")
    print(f"   ├─ 特征维度: {patch_map.shape[-1]}")
    print(f"   ├─ 聚类数: {cluster_cfg.num_clusters}")
    print(f"   ├─ 候选区域: {len(proposals)}")
    print(f"   ├─ 生成掩码: {len(masks)}")
    print(f"   └─ 有效掩码: {len(valid_masks)}")
    
    print(f"\n📁 输出文件:")
    print(f"   {output_dir}/")
    print(f"   ├─ 01_original.jpg       : 原始图像")
    print(f"   ├─ 02_heatmap.jpg        : DINOv3 attention 热图")
    print(f"   ├─ 03_heatmap_overlay.jpg: 热图叠加")
    print(f"   ├─ 04_proposals.jpg      : 候选区域（框+质心）")
    print(f"   ├─ 05_masks.jpg          : 所有分割掩码")
    print(f"   ├─ 06_masks_overlay.jpg  : 掩码叠加到原图")
    print(f"   └─ 07_mask_XX.jpg        : 单独的掩码")
    
    if len(valid_masks) < len(masks) * 0.5:
        print(f"\n⚠️  警告: 只有 {len(valid_masks)}/{len(masks)} 个掩码通过阈值")
        print(f"   建议:")
        print(f"   1. 降低 area_threshold (当前: {area_threshold})")
        print(f"   2. 降低 min_region_area (当前: {cluster_cfg.min_region_area})")
        print(f"   3. 考虑使用 objectness 方法替代 attention 方法")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)