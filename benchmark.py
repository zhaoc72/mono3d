#!/usr/bin/env python3
"""
性能基准测试 - 对比不同配置的速度和质量
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.inference_pipeline import ZeroShotSegmentationPipeline, PipelineConfig
from src.prompt_generator import ClusterConfig, PromptConfig
from src.superpixel_helper import SuperpixelConfig
from src.data_loader import load_image
from src.utils import to_torch_dtype


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    total_time: float
    dinov3_time: float
    clustering_time: float
    sam2_time: float
    num_proposals: int
    num_masks: int
    avg_mask_area: float


def run_benchmark(
    config_name: str,
    pipeline_cfg: PipelineConfig,
    image: np.ndarray,
    device: torch.device,
    dtype: torch.dtype
) -> BenchmarkResult:
    """运行单个配置的基准测试"""
    
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"{'='*60}")
    
    # 初始化
    dinov3_cfg = Dinov3Config(
        repo_or_dir="/media/pc/D/zhaochen/mono3d/dinov3",
        model_name="dinov3_vith16plus",
        checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vith16plus.pth",
        use_torch_hub=True,
        torchhub_source="local"
    )
    
    sam2_cfg = Sam2Config(
        backend="official",
        checkpoint_path="/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt",
        model_config="sam2.1/sam2.1_hiera_l"
    )
    
    pipeline = ZeroShotSegmentationPipeline(
        dinov3_cfg,
        sam2_cfg,
        pipeline_cfg,
        device=str(device),
        dtype=dtype
    )
    
    # 预热
    _ = pipeline.run(image)
    
    # 计时
    times = {}
    
    # DINOv3
    start = time.time()
    feats = pipeline.extractor.extract_features(image)
    times['dinov3'] = time.time() - start
    
    # 聚类
    start = time.time()
    patch_map = feats["patch_map"]
    if hasattr(patch_map, "detach"):
        patch_map = patch_map.detach().cpu().numpy()
    
    if pipeline_cfg.use_superpixels:
        label_map, _, _ = pipeline._cluster_with_superpixels(image, patch_map)
    else:
        label_map, _ = pipeline._cluster_basic(patch_map)
    times['clustering'] = time.time() - start
    
    # 完整推理
    start = time.time()
    result = pipeline.run(image)
    times['total'] = time.time() - start
    
    # SAM2 时间（估算）
    times['sam2'] = times['total'] - times['dinov3'] - times['clustering']
    
    # 统计
    avg_area = 0.0
    if result.masks:
        areas = [mask.sum() for mask in result.masks]
        avg_area = float(np.mean(areas))
    
    print(f"  DINOv3: {times['dinov3']:.3f}s")
    print(f"  聚类:   {times['clustering']:.3f}s")
    print(f"  SAM2:   {times['sam2']:.3f}s")
    print(f"  总计:   {times['total']:.3f}s")
    print(f"  候选:   {len(result.proposals)}")
    print(f"  掩码:   {len(result.masks)}")
    print(f"  平均面积: {avg_area:.0f}")
    
    return BenchmarkResult(
        name=config_name,
        total_time=times['total'],
        dinov3_time=times['dinov3'],
        clustering_time=times['clustering'],
        sam2_time=times['sam2'],
        num_proposals=len(result.proposals),
        num_masks=len(result.masks),
        avg_mask_area=avg_area
    )


def main():
    """主函数"""
    print("=" * 80)
    print("性能基准测试")
    print("=" * 80)
    
    # 加载测试图像
    image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
    sample = load_image(image_path)
    image = sample.image
    
    print(f"\n测试图像: {image.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"设备: {device}")
    print(f"精度: {dtype}")
    
    # 定义测试配置
    configs = []
    
    # 1. 最小配置（速度优先）
    configs.append((
        "最小配置（速度优先）",
        PipelineConfig(
            cluster=ClusterConfig(
                num_clusters=4,
                min_region_area=200
            ),
            prompt=PromptConfig()
        )
    ))
    
    # 2. 标准配置
    configs.append((
        "标准配置",
        PipelineConfig(
            cluster=ClusterConfig(
                num_clusters=6,
                min_region_area=400,
                use_objectness_filter=True
            ),
            prompt=PromptConfig()
        )
    ))
    
    # 3. 高质量配置
    configs.append((
        "高质量配置（对象性）",
        PipelineConfig(
            cluster=ClusterConfig(
                num_clusters=8,
                min_region_area=400,
                use_objectness_filter=True,
                objectness_threshold=0.3
            ),
            prompt=PromptConfig()
        )
    ))
    
    # 4. 超像素配置
    configs.append((
        "超像素配置",
        PipelineConfig(
            cluster=ClusterConfig(
                num_clusters=6,
                min_region_area=400
            ),
            prompt=PromptConfig(),
            use_superpixels=True,
            superpixel=SuperpixelConfig(
                method="slic",
                n_segments=800
            )
        )
    ))
    
    # 运行基准测试
    results: List[BenchmarkResult] = []
    
    for config_name, pipeline_cfg in configs:
        try:
            result = run_benchmark(
                config_name,
                pipeline_cfg,
                image,
                device,
                dtype
            )
            results.append(result)
        except Exception as e:
            print(f"❌ {config_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "=" * 80)
    print("基准测试总结")
    print("=" * 80)
    
    if not results:
        print("没有成功的测试结果")
        return 1
    
    # 表头
    print(f"\n{'配置':<25} {'总时间':<10} {'DINOv3':<10} {'聚类':<10} {'SAM2':<10} {'掩码':<8}")
    print("-" * 90)
    
    # 数据行
    for r in results:
        print(f"{r.name:<25} {r.total_time:<10.3f} {r.dinov3_time:<10.3f} "
              f"{r.clustering_time:<10.3f} {r.sam2_time:<10.3f} {r.num_masks:<8}")
    
    # 分析
    print("\n" + "=" * 80)
    print("性能分析")
    print("=" * 80)
    
    fastest = min(results, key=lambda x: x.total_time)
    most_masks = max(results, key=lambda x: x.num_masks)
    
    print(f"\n🏆 最快配置: {fastest.name}")
    print(f"   时间: {fastest.total_time:.3f}s")
    
    print(f"\n🎯 最多掩码: {most_masks.name}")
    print(f"   掩码数: {most_masks.num_masks}")
    
    # 速度对比
    baseline = results[0]
    print(f"\n⚡ 速度对比（相对于{baseline.name}）:")
    for r in results[1:]:
        speedup = baseline.total_time / r.total_time
        print(f"   {r.name}: {speedup:.2f}x")
    
    # 推荐
    print("\n" + "=" * 80)
    print("推荐配置")
    print("=" * 80)
    
    print("\n1. 速度优先 → 最小配置")
    print("   - 适用场景: 实时处理，低延迟需求")
    print("   - 权衡: 掩码质量可能较低")
    
    print("\n2. 平衡选择 → 标准配置")
    print("   - 适用场景: 大多数应用")
    print("   - 权衡: 速度与质量均衡")
    
    print("\n3. 质量优先 → 高质量配置或超像素配置")
    print("   - 适用场景: 离线处理，精度要求高")
    print("   - 权衡: 处理时间较长")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())