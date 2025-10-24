#!/usr/bin/env python3
"""测试完整的改进 Pipeline"""

import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.inference_pipeline import ZeroShotSegmentationPipeline, PipelineConfig
from src.prompt_generator import ClusterConfig, PromptConfig
from src.superpixel_helper import SuperpixelConfig, visualize_superpixels
from src.graph_clustering import GraphClusterConfig
from src.density_clustering import DensityClusterConfig
from src.crf_refinement import CRFConfig
from src.data_loader import load_image
from src.utils import to_torch_dtype

print("=" * 80)
print("完整 Pipeline 测试")
print("=" * 80)

# 加载配置
config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/model_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 加载图像
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
sample = load_image(image_path)

print(f"\n图像: {sample.image.shape}")

# 初始化配置
device = torch.device(config["device"])
dtype = to_torch_dtype(config["dtype"])

dinov3_cfg = Dinov3Config(**config["dinov3"])
sam2_cfg = Sam2Config(**config["sam2"])

# Pipeline 配置
pipeline_dict = config["pipeline"]

cluster_cfg = ClusterConfig(**pipeline_dict["cluster"])
prompt_cfg = PromptConfig(**pipeline_dict["prompt"])

superpixel_cfg = None
if pipeline_dict.get("use_superpixels", False):
    superpixel_cfg = SuperpixelConfig(**pipeline_dict.get("superpixel", {}))

graph_cluster_cfg = None
if pipeline_dict.get("use_graph_clustering", False):
    graph_cluster_cfg = GraphClusterConfig(**pipeline_dict.get("graph_cluster", {}))

density_cluster_cfg = None
if pipeline_dict.get("use_density_clustering", False):
    density_cluster_cfg = DensityClusterConfig(**pipeline_dict.get("density_cluster", {}))

crf_cfg = CRFConfig(**pipeline_dict.get("crf", {}))

pipeline_cfg = PipelineConfig(
    cluster=cluster_cfg,
    prompt=prompt_cfg,
    use_superpixels=pipeline_dict.get("use_superpixels", False),
    superpixel=superpixel_cfg or SuperpixelConfig(),
    use_graph_clustering=pipeline_dict.get("use_graph_clustering", False),
    graph_cluster=graph_cluster_cfg or GraphClusterConfig(),
    use_density_clustering=pipeline_dict.get("use_density_clustering", False),
    density_cluster=density_cluster_cfg or DensityClusterConfig(),
    crf=crf_cfg
)

print("\n" + "=" * 80)
print("配置信息")
print("=" * 80)
print(f"DINOv3:")
print(f"  多层: {dinov3_cfg.output_layers}")
print(f"  权重: {dinov3_cfg.layer_weights}")
print(f"  融合: {dinov3_cfg.fusion_method}")
print(f"  对象性: {dinov3_cfg.enable_objectness}")

print(f"\n高级聚类:")
print(f"  超像素: {pipeline_cfg.use_superpixels}")
if pipeline_cfg.use_superpixels:
    print(f"    方法: {pipeline_cfg.superpixel.method}")
    print(f"    数量: {pipeline_cfg.superpixel.n_segments}")

print(f"  图聚类: {pipeline_cfg.use_graph_clustering}")
if pipeline_cfg.use_graph_clustering:
    print(f"    方法: {pipeline_cfg.graph_cluster.method}")

print(f"  密度聚类: {pipeline_cfg.use_density_clustering}")
if pipeline_cfg.use_density_clustering:
    print(f"    方法: {pipeline_cfg.density_cluster.method}")

print(f"\nCRF 细化: {pipeline_cfg.crf.enable}")

# 初始化 Pipeline
print("\n" + "=" * 80)
print("初始化 Pipeline")
print("=" * 80)

pipeline = ZeroShotSegmentationPipeline(
    dinov3_cfg,
    sam2_cfg,
    pipeline_cfg,
    device=str(device),
    dtype=dtype
)

print("✅ Pipeline 初始化完成")

# 运行
print("\n" + "=" * 80)
print("运行推理")
print("=" * 80)

nms_config = {
    "enable_nms": pipeline_dict.get("enable_nms", True),
    "iou_threshold": pipeline_dict.get("iou_threshold", 0.6),
    "objectness_weight": pipeline_dict.get("objectness_weight", 0.5),
    "confidence_weight": pipeline_dict.get("confidence_weight", 0.3),
    "area_weight": pipeline_dict.get("area_weight", 0.2)
}

result = pipeline.run(sample.image, nms_config=nms_config)

print(f"\n结果:")
print(f"  候选区域: {len(result.proposals)}")
print(f"  最终掩码: {len(result.masks)}")

# 保存结果
output_dir = Path("outputs/complete_pipeline_test")
output_dir.mkdir(exist_ok=True, parents=True)

# 1. 保存对象性图
if result.objectness_map is not None:
    obj_map = result.objectness_map
    obj_vis = (obj_map * 255).astype(np.uint8)
    obj_colored = cv2.applyColorMap(obj_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / "objectness_map.png"), obj_colored)
    
    overlay = cv2.addWeighted(
        cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.6,
        obj_colored, 0.4, 0
    )
    cv2.imwrite(str(output_dir / "objectness_overlay.png"), overlay)

# 2. 保存超像素
if result.superpixel_labels is not None:
    visualize_superpixels(
        sample.image,
        result.superpixel_labels,
        str(output_dir / "superpixels.png")
    )

# 3. 保存掩码
if result.masks:
    combined = np.zeros_like(sample.image)
    for i, mask in enumerate(result.masks):
        color = np.array([
            (i * 50) % 255,
            (i * 80 + 60) % 255,
            (i * 120 + 30) % 255
        ], dtype=np.uint8)
        combined[mask.astype(bool)] = color
    
    cv2.imwrite(
        str(output_dir / "masks.png"),
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    )
    
    overlay = cv2.addWeighted(
        cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.5,
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
    )
    cv2.imwrite(str(output_dir / "masks_overlay.png"), overlay)

# 4. 打印详情
print(f"\n候选区域详情:")
for i, prop in enumerate(result.proposals[:5]):
    print(f"  区域 {i}:")
    print(f"    对象性: {prop.objectness:.3f}")
    print(f"    得分: {prop.score:.1f}")
    print(f"    框: {prop.bbox}")

print(f"\n✅ 所有结果保存在: {output_dir}/")
print("=" * 80)