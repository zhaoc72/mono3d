#!/usr/bin/env python3
"""
端到端测试脚本 - 验证完整 Pipeline
测试所有核心功能的集成
"""

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
from src.superpixel_helper import SuperpixelConfig
from src.graph_clustering import GraphClusterConfig
from src.density_clustering import DensityClusterConfig
from src.crf_refinement import CRFConfig
from src.data_loader import load_image
from src.utils import to_torch_dtype, ensure_directory

print("=" * 80)
print("端到端 Pipeline 测试")
print("=" * 80)


def test_basic_pipeline():
    """测试基础 Pipeline（最简配置）"""
    print("\n" + "=" * 80)
    print("测试 1: 基础 Pipeline")
    print("=" * 80)
    
    try:
        # 加载图像
        image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
        sample = load_image(image_path)
        print(f"✅ 图像加载: {sample.image.shape}")
        
        # 配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        dinov3_cfg = Dinov3Config(
            repo_or_dir="/media/pc/D/zhaochen/mono3d/dinov3",
            model_name="dinov3_vith16plus",
            checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vith16plus.pth",
            use_torch_hub=True,
            torchhub_source="local",
            enable_objectness=True
        )
        
        sam2_cfg = Sam2Config(
            backend="official",
            checkpoint_path="/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt",
            model_config="sam2.1/sam2.1_hiera_l"
        )
        
        cluster_cfg = ClusterConfig(
            num_clusters=6,
            min_region_area=400,
            use_objectness_filter=True
        )
        
        prompt_cfg = PromptConfig(
            include_boxes=True,
            include_points=True
        )
        
        pipeline_cfg = PipelineConfig(
            cluster=cluster_cfg,
            prompt=prompt_cfg
        )
        
        # 初始化 Pipeline
        print("初始化 Pipeline...")
        pipeline = ZeroShotSegmentationPipeline(
            dinov3_cfg,
            sam2_cfg,
            pipeline_cfg,
            device=str(device),
            dtype=dtype
        )
        print("✅ Pipeline 初始化成功")
        
        # 运行推理
        print("运行推理...")
        nms_config = {
            "enable_nms": True,
            "iou_threshold": 0.6,
            "objectness_weight": 0.5
        }
        
        result = pipeline.run(sample.image, nms_config=nms_config)
        print(f"✅ 推理完成")
        print(f"   候选区域: {len(result.proposals)}")
        print(f"   最终掩码: {len(result.masks)}")
        
        # 保存结果
        output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/e2e_test_basic")
        ensure_directory(output_dir)
        
        if result.masks:
            combined = np.zeros_like(sample.image)
            for i, mask in enumerate(result.masks):
                color = np.array([
                    (i * 50) % 255,
                    (i * 80 + 60) % 255,
                    (i * 120 + 30) % 255
                ], dtype=np.uint8)
                combined[mask.astype(bool)] = color
            
            overlay = cv2.addWeighted(
                cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.5,
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
            )
            cv2.imwrite(str(output_dir / "result.png"), overlay)
            print(f"✅ 结果已保存: {output_dir}/result.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_pipeline():
    """测试高级 Pipeline（启用超像素 + 图聚类）"""
    print("\n" + "=" * 80)
    print("测试 2: 高级 Pipeline（超像素 + 图聚类）")
    print("=" * 80)
    
    try:
        # 加载图像
        image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
        sample = load_image(image_path)
        print(f"✅ 图像加载: {sample.image.shape}")
        
        # 配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
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
        
        sam2_cfg = Sam2Config(
            backend="official",
            checkpoint_path="/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt",
            model_config="sam2.1/sam2.1_hiera_l"
        )
        
        cluster_cfg = ClusterConfig(
            num_clusters=6,
            min_region_area=400,
            use_objectness_filter=True
        )
        
        prompt_cfg = PromptConfig(
            include_boxes=True,
            include_points=True
        )
        
        superpixel_cfg = SuperpixelConfig(
            method="slic",
            n_segments=1000,
            compactness=10.0
        )
        
        graph_cluster_cfg = GraphClusterConfig(
            method="spectral",
            n_clusters=6
        )
        
        pipeline_cfg = PipelineConfig(
            cluster=cluster_cfg,
            prompt=prompt_cfg,
            use_superpixels=True,
            superpixel=superpixel_cfg,
            use_graph_clustering=True,
            graph_cluster=graph_cluster_cfg
        )
        
        # 初始化 Pipeline
        print("初始化高级 Pipeline...")
        pipeline = ZeroShotSegmentationPipeline(
            dinov3_cfg,
            sam2_cfg,
            pipeline_cfg,
            device=str(device),
            dtype=dtype
        )
        print("✅ Pipeline 初始化成功")
        
        # 运行推理
        print("运行推理...")
        nms_config = {
            "enable_nms": True,
            "iou_threshold": 0.6,
            "objectness_weight": 0.5
        }
        
        result = pipeline.run(sample.image, nms_config=nms_config)
        print(f"✅ 推理完成")
        print(f"   候选区域: {len(result.proposals)}")
        print(f"   最终掩码: {len(result.masks)}")
        
        # 保存结果
        output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/e2e_test_advanced")
        ensure_directory(output_dir)
        
        if result.masks:
            combined = np.zeros_like(sample.image)
            for i, mask in enumerate(result.masks):
                color = np.array([
                    (i * 50) % 255,
                    (i * 80 + 60) % 255,
                    (i * 120 + 30) % 255
                ], dtype=np.uint8)
                combined[mask.astype(bool)] = color
            
            overlay = cv2.addWeighted(
                cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.5,
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
            )
            cv2.imwrite(str(output_dir / "result.png"), overlay)
            
            # 保存超像素可视化
            if result.superpixel_labels is not None:
                from src.superpixel_helper import visualize_superpixels
                visualize_superpixels(
                    sample.image,
                    result.superpixel_labels,
                    str(output_dir / "superpixels.png")
                )
                print(f"✅ 超像素已保存")
            
            print(f"✅ 结果已保存: {output_dir}/result.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multilayer_fusion():
    """测试多层特征融合"""
    print("\n" + "=" * 80)
    print("测试 3: 多层特征融合")
    print("=" * 80)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        dinov3_cfg = Dinov3Config(
            repo_or_dir="/media/pc/D/zhaochen/mono3d/dinov3",
            model_name="dinov3_vith16plus",
            checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vith16plus.pth",
            use_torch_hub=True,
            torchhub_source="local",
            output_layers=[4, 8, 12],
            layer_weights=[0.2, 0.3, 0.5],
            fusion_method="weighted_concat"
        )
        
        extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
        print("✅ DINOv3 初始化成功")
        
        # 测试图像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 提取特征
        print("提取多层特征...")
        feats = extractor.extract_features(image)
        
        print(f"✅ 特征提取成功")
        print(f"   Patch map: {feats['patch_map'].shape}")
        print(f"   Grid size: {feats['grid_size']}")
        
        if feats.get('objectness_map') is not None:
            print(f"   Objectness map: {feats['objectness_map'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始端到端测试")
    print("=" * 80)
    
    results = {}
    
    # 测试 1: 基础 Pipeline
    results['basic'] = test_basic_pipeline()
    
    # 测试 2: 高级 Pipeline
    results['advanced'] = test_advanced_pipeline()
    
    # 测试 3: 多层融合
    results['multilayer'] = test_multilayer_fusion()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<15}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())