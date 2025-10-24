#!/usr/bin/env python3
"""
端到端测试脚本 - 验证完整 Pipeline
测试所有核心功能的集成，包含多层特征融合可视化
"""

import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def visualize_multilayer_features(image, feats, config, output_dir):
    """可视化多层特征融合结果
    
    Args:
        image: 原始图像 (H, W, 3)
        feats: 特征字典
        config: DINOv3 配置
        output_dir: 输出目录
    """
    patch_map = feats['patch_map']  # (H, W, D)
    grid_h, grid_w, total_dim = patch_map.shape
    
    # 1. 保存原始图像
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title("Original Image", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "0_original.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ 原始图像")
    
    # 2. 如果有 objectness map,先可视化
    if feats.get('objectness_map') is not None:
        objectness = feats['objectness_map']
        
        plt.figure(figsize=(10, 6))
        plt.imshow(objectness, cmap='hot')
        plt.colorbar(label='Objectness Score', fraction=0.046)
        plt.title("Objectness Map", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "1_objectness.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Objectness Map")
    
    # 3. 可视化每一层的特征 (使用 PCA 降维到 3 维用于 RGB 可视化)
    layer_dim = 1280
    num_layers = len(config.output_layers)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 6))
    if num_layers == 1:
        axes = [axes]
    
    for i, layer_idx in enumerate(config.output_layers):
        # 提取该层的特征
        start_idx = i * layer_dim
        end_idx = start_idx + layer_dim
        layer_feats = patch_map[:, :, start_idx:end_idx]  # (H, W, 1280)
        
        # PCA 降维到 3D
        layer_feats_flat = layer_feats.reshape(-1, layer_dim)
        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(layer_feats_flat)
        
        # 归一化到 [0, 1]
        pca_feats = (pca_feats - pca_feats.min(axis=0)) / (pca_feats.max(axis=0) - pca_feats.min(axis=0) + 1e-8)
        pca_map = pca_feats.reshape(grid_h, grid_w, 3)
        
        # 上采样到原图大小
        pca_map_upsampled = cv2.resize(
            pca_map, 
            (image.shape[1], image.shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        axes[i].imshow(pca_map_upsampled)
        axes[i].set_title(f"Layer {layer_idx}\n(weight={config.layer_weights[i]})", 
                         fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "2_individual_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 各层特征 (Layers {config.output_layers})")
    
    # 4. 可视化融合后的特征
    fused_feats = patch_map.reshape(-1, total_dim)
    pca = PCA(n_components=3)
    pca_fused = pca.fit_transform(fused_feats)
    
    # 归一化
    pca_fused = (pca_fused - pca_fused.min(axis=0)) / (pca_fused.max(axis=0) - pca_fused.min(axis=0) + 1e-8)
    pca_fused_map = pca_fused.reshape(grid_h, grid_w, 3)
    
    # 上采样
    pca_fused_upsampled = cv2.resize(
        pca_fused_map,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pca_fused_upsampled)
    plt.title(f"Fused Features ({config.fusion_method})\nDim: {total_dim} = {num_layers} × 1280", 
             fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "3_fused_features.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 融合特征")
    
    # 5. 创建完整对比图
    fig = plt.figure(figsize=(20, 12))
    
    # 原图
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Objectness (如果有)
    if feats.get('objectness_map') is not None:
        ax2 = plt.subplot(2, 3, 2)
        im = ax2.imshow(feats['objectness_map'], cmap='hot')
        ax2.set_title("Objectness Map", fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # 各层特征
    for i, layer_idx in enumerate(config.output_layers):
        ax = plt.subplot(2, 3, 3 + i)
        
        start_idx = i * layer_dim
        end_idx = start_idx + layer_dim
        layer_feats = patch_map[:, :, start_idx:end_idx]
        
        layer_feats_flat = layer_feats.reshape(-1, layer_dim)
        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(layer_feats_flat)
        pca_feats = (pca_feats - pca_feats.min(axis=0)) / (pca_feats.max(axis=0) - pca_feats.min(axis=0) + 1e-8)
        pca_map = pca_feats.reshape(grid_h, grid_w, 3)
        pca_map_upsampled = cv2.resize(pca_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        ax.imshow(pca_map_upsampled)
        ax.set_title(f"Layer {layer_idx} (w={config.layer_weights[i]})", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # 融合结果
    ax_fused = plt.subplot(2, 3, 6)
    ax_fused.imshow(pca_fused_upsampled)
    ax_fused.set_title(f"Fused Features\n{config.fusion_method} | Dim={total_dim}", 
                       fontsize=12, fontweight='bold')
    ax_fused.axis('off')
    
    plt.suptitle("Multi-Layer Feature Fusion Visualization", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "4_complete_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 完整对比图")
    
    # 6. 特征统计信息
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 各层特征的方差解释率
    ax1 = axes[0, 0]
    variance_ratios = []
    for i in range(num_layers):
        start_idx = i * layer_dim
        end_idx = start_idx + layer_dim
        layer_feats = patch_map[:, :, start_idx:end_idx].reshape(-1, layer_dim)
        pca_temp = PCA(n_components=10)
        pca_temp.fit(layer_feats)
        variance_ratios.append(pca_temp.explained_variance_ratio_[:5])
    
    x = np.arange(5)
    width = 0.25
    for i, (layer_idx, ratios) in enumerate(zip(config.output_layers, variance_ratios)):
        ax1.bar(x + i*width, ratios, width, label=f'Layer {layer_idx}')
    
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Feature Variance by Layer (Top 5 PCs)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'PC{i+1}' for i in range(5)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 融合特征的方差解释率
    ax2 = axes[0, 1]
    pca_fused_full = PCA(n_components=min(20, total_dim))
    pca_fused_full.fit(fused_feats)
    n_components = len(pca_fused_full.explained_variance_ratio_)
    ax2.plot(range(1, n_components + 1), pca_fused_full.explained_variance_ratio_, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax2.set_title(f'Fused Features Variance (Top {n_components} PCs)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 特征维度分布
    ax3 = axes[1, 0]
    layer_labels = [f'Layer {idx}' for idx in config.output_layers]
    layer_dims = [layer_dim] * num_layers
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    bars = ax3.bar(layer_labels, layer_dims, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.axhline(y=total_dim, color='r', linestyle='--', linewidth=2, label=f'Total: {total_dim}')
    ax3.set_ylabel('Feature Dimension', fontsize=12)
    ax3.set_title('Feature Dimensions by Layer', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在每个柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 权重分布
    ax4 = axes[1, 1]
    wedges, texts, autotexts = ax4.pie(
        config.layer_weights, 
        labels=layer_labels, 
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    ax4.set_title('Layer Weights Distribution', fontsize=14, fontweight='bold')
    
    plt.suptitle('Feature Statistics & Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "5_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 统计分析图")
    
    # 7. 生成 README 说明文件
    readme_content = f"""# 多层特征融合可视化结果

## 配置信息
- **模型**: {config.model_name}
- **融合层**: {config.output_layers}
- **层权重**: {config.layer_weights}
- **融合方法**: {config.fusion_method}
- **总特征维度**: {total_dim} ({num_layers} × 1280)

## 文件说明

### 0_original.png
原始输入图像

### 1_objectness.png
Objectness Map - 显示模型对每个区域是否为物体的置信度

### 2_individual_layers.png
各层独立特征可视化 (使用PCA降维到RGB)
- Layer {config.output_layers[0]}: 低层特征,关注边缘和纹理
- Layer {config.output_layers[1] if len(config.output_layers) > 1 else 'N/A'}: 中层特征,关注局部模式
- Layer {config.output_layers[2] if len(config.output_layers) > 2 else 'N/A'}: 高层特征,关注语义信息

### 3_fused_features.png
融合后的特征 - 整合了多层信息的综合表示

### 4_complete_comparison.png
完整对比图 - 展示从原图到各层特征再到融合特征的完整流程

### 5_statistics.png
统计分析图,包含:
- 各层特征方差分析 (前5个主成分)
- 融合特征方差分析 (前20个主成分)
- 各层特征维度分布
- 层权重分布

## 关键发现
1. 特征维度从单层的1280扩展到{total_dim}
2. 不同层捕获不同级别的视觉特征
3. 权重融合平衡了各层的贡献

## 生成时间
{Path(output_dir).name}
"""
    
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"   ✓ README 说明文件")


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
        
        # 加载真实图像用于可视化
        image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
        sample = load_image(image_path)
        image = sample.image
        print(f"✅ 图像加载: {image.shape}")
        
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
        
        # 提取特征
        print("提取多层特征...")
        feats = extractor.extract_features(image)
        
        print(f"✅ 特征提取成功")
        print(f"   Patch map: {feats['patch_map'].shape}")
        print(f"   Grid size: {feats['grid_size']}")
        
        # 详细的融合验证
        expected_dim = 1280 * len(dinov3_cfg.output_layers)
        actual_dim = feats['patch_map'].shape[-1]
        
        print(f"\n📊 多层融合验证:")
        print(f"   配置层数: {len(dinov3_cfg.output_layers)} (layers {dinov3_cfg.output_layers})")
        print(f"   单层特征维度: 1280")
        print(f"   期望融合维度: {expected_dim}")
        print(f"   实际融合维度: {actual_dim}")
        print(f"   融合方法: {dinov3_cfg.fusion_method}")
        print(f"   层权重: {dinov3_cfg.layer_weights}")
        
        if actual_dim == expected_dim:
            print(f"   ✅ 维度匹配! 多层融合成功!")
        else:
            print(f"   ⚠️  维度不匹配")
        
        if feats.get('objectness_map') is not None:
            print(f"   Objectness map: {feats['objectness_map'].shape}")
        
        # 可视化多层特征
        print("\n🎨 生成可视化...")
        output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/e2e_test_multilayer")
        ensure_directory(output_dir)
        
        visualize_multilayer_features(
            image, 
            feats, 
            dinov3_cfg,
            output_dir
        )
        
        print(f"\n✅ 所有可视化已保存到: {output_dir}")
        print(f"   包含 7 个文件:")
        print(f"   - 0_original.png: 原始图像")
        print(f"   - 1_objectness.png: Objectness Map")
        print(f"   - 2_individual_layers.png: 各层独立特征")
        print(f"   - 3_fused_features.png: 融合特征")
        print(f"   - 4_complete_comparison.png: 完整对比")
        print(f"   - 5_statistics.png: 统计分析")
        print(f"   - README.md: 详细说明")
        
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