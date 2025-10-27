#!/usr/bin/env python3
"""
修复版 DINOv3 可视化脚本
专门支持 ViT-7B/16 模型和正确的路径配置
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml

def load_yaml(path: str):
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """将浮点数组归一化到 uint8 范围 [0, 255]"""
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=np.uint8)
    
    clipped = np.zeros_like(data, dtype=np.float16)
    clipped[finite] = data[finite]
    minimum = float(clipped[finite].min())
    maximum = float(clipped[finite].max())
    
    if maximum - minimum < 1e-6:
        return np.zeros_like(clipped, dtype=np.uint8)
    
    normalized = (clipped - minimum) / (maximum - minimum)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)

def save_heatmap(map_2d: np.ndarray, output_path: Path, base_image: np.ndarray = None):
    """保存热力图可视化"""
    heatmap_uint8 = normalize_to_uint8(map_2d)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    
    # 保存独立热力图
    cv2.imwrite(str(output_path.with_suffix('.png')), heatmap_color)
    print(f"   ✅ Saved: {output_path.name}")
    
    # 如果提供了基础图像，保存叠加图
    if base_image is not None:
        base_bgr = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
        # 调整热力图大小以匹配图像
        if heatmap_color.shape[:2] != base_bgr.shape[:2]:
            heatmap_color = cv2.resize(
                heatmap_color,
                (base_bgr.shape[1], base_bgr.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
        overlay = cv2.addWeighted(base_bgr, 0.6, heatmap_color, 0.4, 0.0)
        overlay_path = output_path.with_name(output_path.stem + '_overlay.png')
        cv2.imwrite(str(overlay_path), overlay)
        print(f"   ✅ Saved: {overlay_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description="可视化 DINOv3 backbone 和 adapter 输出"
    )
    parser.add_argument("--input", required=True, help="输入图像路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--config", required=True, help="配置 YAML 文件")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument(
        "--detection-threshold", type=float, default=0.05,
        help="检测分数阈值 (默认: 0.05)"
    )
    parser.add_argument(
        "--skip-adapters", action="store_true",
        help="跳过 adapter，只运行 backbone"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print("=" * 70)
    print("DINOv3 Visualization (ViT-7B/16)")
    print("=" * 70)
    print(f"\n📖 Loading config: {args.config}")
    
    config = load_yaml(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载图像
    print(f"\n🖼️  Loading image: {args.input}")
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise ValueError(f"Failed to load image from {args.input}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"   Image shape: {image_rgb.shape}")
    
    # 导入必需模块
    print(f"\n📦 Importing modules...")
    sys.path.insert(0, str(Path(__file__).parent))
    
    from src.config import Dinov3BackboneConfig
    from src.dinov3_feature import Dinov3Backbone
    
    # 设置设备和数据类型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"   Device: {device}")
    print(f"   Dtype: {dtype}")
    
    # 解析 DINOv3 backbone 配置
    print(f"\n⚙️  Parsing configuration...")
    model_section = config.get('model', config)
    backbone_config_dict = model_section.get('backbone', {})
    
    if not backbone_config_dict:
        raise ValueError("Could not find 'model.backbone' section in config")
    
    print(f"   Backbone config keys: {list(backbone_config_dict.keys())}")
    
    # 初始化 DINOv3 backbone
    print(f"\n============================================================")
    print(f"🧠 Step 1: Initializing DINOv3 Backbone")
    print(f"============================================================")
    
    repo_path = backbone_config_dict.get('repo_path')
    checkpoint_path = backbone_config_dict.get('checkpoint_path')
    
    print(f"   Repo path: {repo_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Model name: {backbone_config_dict.get('model_name', 'dinov3_vit7b16')}")
    
    # 检查文件是否存在
    if repo_path and not Path(repo_path).exists():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_path}")
    if checkpoint_path and not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    dinov3_config = Dinov3BackboneConfig(
        repo_path=repo_path,
        model_name=backbone_config_dict.get('model_name', 'dinov3_vit7b16'),
        checkpoint_path=checkpoint_path,
        image_size=backbone_config_dict.get('image_size', 518),
        output_layers=tuple(backbone_config_dict.get('output_layers', [9, 19, 29, 39])),
        enable_objectness=backbone_config_dict.get('enable_objectness', True),
        enable_pca=backbone_config_dict.get('enable_pca', True),
        pca_dim=backbone_config_dict.get('pca_dim', 32),
        patch_size=backbone_config_dict.get('patch_size', 16),
    )
    
    backbone = Dinov3Backbone(dinov3_config, device=device, dtype=dtype)
    print(f"   ✅ Backbone initialized successfully")
    
    # 提取特征
    print(f"\n🔄 Extracting DINOv3 features...")
    features = backbone.extract_features(image_rgb)
    
    patch_tokens = features['patch_tokens']
    patch_tokens_raw = features.get('patch_tokens_raw', patch_tokens)
    grid_size = features['grid_size']
    processed_shape = features['processed_image_shape']
    objectness_map = features.get('objectness_map')
    attention_map = features.get('attention_map')
    
    print(f"   ✅ Feature extraction complete:")
    print(f"      - Grid size: {grid_size}")
    print(f"      - Processed shape: {processed_shape}")
    print(f"      - Patch tokens: {patch_tokens.shape}")
    print(f"      - Patch tokens raw: {patch_tokens_raw.shape}")
    print(f"      - Objectness map: {objectness_map.shape if objectness_map is not None else 'None'}")
    print(f"      - Attention map: {attention_map.shape if attention_map is not None else 'None'}")
    
    # 保存 objectness map
    if objectness_map is not None:
        print(f"\n💾 Saving objectness map...")
        save_heatmap(objectness_map, output_dir / "objectness", image_rgb)
    else:
        print(f"\n⚠️  No objectness map available")
    
    # 保存 attention map
    if attention_map is not None:
        print(f"\n💾 Saving attention map...")
        save_heatmap(attention_map, output_dir / "attention", image_rgb)
    else:
        print(f"\n⚠️  No attention map available")
    
    # 如果指定跳过 adapters，则在此结束
    if args.skip_adapters:
        print(f"\n⏭️  Skipping adapters as requested")
        print(f"\n✅ Visualization complete!")
        print(f"   Output directory: {output_dir}")
        return
    
    # 导入 adapter 模块
    from src.adapters.detection import build_detection_adapter, DetectionAdapterConfig
    from src.adapters.segmentation import build_segmentation_adapter, SegmentationAdapterConfig
    
    processed_hw = (processed_shape[1], processed_shape[0])
    
    # Detection Adapter
    detection_config_dict = model_section.get('detection_adapter', {})
    if detection_config_dict:
        print(f"\n============================================================")
        print(f"🎯 Step 2: Running Detection Adapter")
        print(f"============================================================")
        
        checkpoint_path = detection_config_dict.get('checkpoint_path', '')
        print(f"   Checkpoint: {checkpoint_path or '<random initialization>'}")
        print(f"   Threshold: {args.detection_threshold}")
        
        detection_config = DetectionAdapterConfig(
            checkpoint_path=checkpoint_path,
            feature_dim=patch_tokens_raw.shape[-1],
            num_classes=detection_config_dict.get('num_classes', 91),
            score_threshold=args.detection_threshold
        )
        
        detection_adapter = build_detection_adapter(
            detection_config,
            device=str(device),
            torch_dtype=dtype
        )
        
        print(f"   🔄 Running detection...")
        detection = detection_adapter.predict(
            patch_tokens_raw,
            image_size=processed_hw,
            grid_size=grid_size
        )
        
        print(f"   ✅ Detection complete: {len(detection.boxes)} boxes")
        if len(detection.boxes) > 0:
            print(f"      Score range: [{detection.scores.min():.3f}, {detection.scores.max():.3f}]")
    
    # Segmentation Adapter
    segmentation_config_dict = model_section.get('segmentation_adapter', {})
    if segmentation_config_dict:
        print(f"\n============================================================")
        print(f"🖼️  Step 3: Running Segmentation Adapter")
        print(f"============================================================")
        
        checkpoint_path = segmentation_config_dict.get('checkpoint_path', '')
        print(f"   Checkpoint: {checkpoint_path or '<random initialization>'}")
        
        segmentation_config = SegmentationAdapterConfig(
            checkpoint_path=checkpoint_path,
            feature_dim=patch_tokens_raw.shape[-1],
            num_classes=segmentation_config_dict.get('num_classes', 150),
        )
        
        segmentation_adapter = build_segmentation_adapter(
            segmentation_config,
            device=str(device),
            torch_dtype=dtype
        )
        
        print(f"   🔄 Running segmentation...")
        segmentation = segmentation_adapter.predict(
            patch_tokens_raw,
            image_size=processed_hw,
            grid_size=grid_size
        )
        
        print(f"   ✅ Segmentation complete")
        print(f"      Logits shape: {segmentation.logits.shape}")
        print(f"      Logits range: [{segmentation.logits.min():.3f}, {segmentation.logits.max():.3f}]")
    
    # 保存元数据
    import json
    metadata = {
        "image": args.input,
        "image_shape": list(image_rgb.shape),
        "processed_shape": list(processed_shape),
        "grid_size": list(grid_size),
        "detection_threshold": args.detection_threshold,
        "feature_dim": patch_tokens.shape[-1],
        "feature_dim_raw": patch_tokens_raw.shape[-1],
    }
    
    if 'detection' in locals():
        metadata["num_detections"] = len(detection.boxes)
    if 'segmentation' in locals():
        metadata["num_segmentation_classes"] = segmentation.logits.shape[0]
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print(f"✅ Visualization Complete!")
    print(f"=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📄 Generated files:")
    if objectness_map is not None:
        print(f"   - objectness.png / objectness_overlay.png")
    if attention_map is not None:
        print(f"   - attention.png / attention_overlay.png")
    if 'detection' in locals():
        print(f"   - Detection: {len(detection.boxes)} boxes")
    if 'segmentation' in locals():
        print(f"   - Segmentation: {segmentation.logits.shape[0]} classes")
    print(f"   - metadata.json")

if __name__ == "__main__":
    main()