#!/usr/bin/env python3
"""
Fixed DINOv3 visualization that correctly parses the config structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import yaml


def load_yaml(path: str):
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize float array to uint8 range [0, 255]."""
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=np.uint8)
    
    clipped = np.zeros_like(data, dtype=np.float32)
    clipped[finite] = data[finite]
    minimum = float(clipped[finite].min())
    maximum = float(clipped[finite].max())
    
    if maximum - minimum < 1e-6:
        return np.zeros_like(clipped, dtype=np.uint8)
    
    normalized = (clipped - minimum) / (maximum - minimum)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def save_heatmap(map_2d: np.ndarray, output_path: Path, base_image: Optional[np.ndarray] = None):
    """Save heatmap visualization."""
    heatmap_uint8 = normalize_to_uint8(map_2d)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    
    # Save standalone heatmap
    cv2.imwrite(str(output_path.with_suffix('.png')), heatmap_color)
    
    # Save overlay if base image provided
    if base_image is not None:
        base_bgr = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
        # Resize heatmap to match image size if needed
        if heatmap_color.shape[:2] != base_bgr.shape[:2]:
            heatmap_color = cv2.resize(
                heatmap_color,
                (base_bgr.shape[1], base_bgr.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
        overlay = cv2.addWeighted(base_bgr, 0.6, heatmap_color, 0.4, 0.0)
        cv2.imwrite(str(output_path.with_name(output_path.stem + '_overlay.png')), overlay)


def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list,
    output_path: Path,
    processed_shape: Tuple[int, int]
):
    """Visualize detection results."""
    if len(boxes) == 0:
        print("⚠️ No detections to visualize (try lowering score_threshold in config)")
        # Still save the original image
        base_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), base_bgr)
        return
    
    base_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    processed_h, processed_w = processed_shape
    
    scale_x = width / max(1, processed_w)
    scale_y = height / max(1, processed_h)
    
    # Color palette
    colors = [
        (240, 86, 60), (67, 160, 71), (66, 133, 244), (171, 71, 188),
        (255, 202, 40), (38, 198, 218), (255, 112, 67), (124, 179, 66),
    ]
    
    for idx, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        color = colors[idx % len(colors)]
        
        # Scale box to original image size
        x1 = int(np.clip(box[0] * scale_x, 0, width - 1))
        y1 = int(np.clip(box[1] * scale_y, 0, height - 1))
        x2 = int(np.clip(box[2] * scale_x, 0, width - 1))
        y2 = int(np.clip(box[3] * scale_y, 0, height - 1))
        
        cv2.rectangle(base_bgr, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        class_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))
        label = f"{class_name}:{float(score):.2f}"
        cv2.putText(
            base_bgr, label, (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    
    cv2.imwrite(str(output_path), base_bgr)
    print(f"✅ Saved detection visualization to {output_path}")


def visualize_segmentation(
    image: np.ndarray,
    logits: np.ndarray,
    class_names: list,
    output_path: Path
):
    """Visualize segmentation results."""
    # Apply softmax to get probabilities
    logits_shifted = logits - logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / (exp_logits.sum(axis=0, keepdims=True) + 1e-8)
    
    # Get class predictions
    class_map = probs.argmax(axis=0).astype(np.int32)
    
    # Get confidence
    confidence = probs.max(axis=0)
    
    print(f"   Segmentation stats:")
    print(f"   - Unique classes: {len(np.unique(class_map))}")
    print(f"   - Mean confidence: {confidence.mean():.3f}")
    print(f"   - Top 5 classes: {np.bincount(class_map.flatten()).argsort()[-5:][::-1].tolist()}")
    
    # Create color palette
    np.random.seed(12345)
    num_classes = logits.shape[0]
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)  # Background as black
    
    # Create color map
    color_map = palette[class_map % len(palette)].astype(np.uint8)
    color_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    
    # Resize to match image size if needed
    if color_map.shape[:2] != image.shape[:2]:
        color_bgr = cv2.resize(
            color_bgr,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Save standalone segmentation map
    cv2.imwrite(str(output_path), color_bgr)
    
    # Save overlay
    base_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.6, color_bgr, 0.4, 0.0)
    cv2.imwrite(str(output_path.with_name(output_path.stem + '_overlay.png')), overlay)
    
    print(f"✅ Saved segmentation visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 backbone and adapter outputs")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--detection-threshold", type=float, default=0.1, 
                       help="Detection score threshold (default: 0.1, lower = more detections)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading image from {args.input}")
    # Load image
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise ValueError(f"Failed to load image from {args.input}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {image_rgb.shape}")
    
    # Import required modules
    print("Importing DINOv3 modules...")
    from src.config import Dinov3BackboneConfig
    from src.dinov3_feature import Dinov3Backbone
    from src.adapters.detection import build_detection_adapter, DetectionAdapterConfig
    from src.adapters.segmentation import build_segmentation_adapter, SegmentationAdapterConfig
    
    # Setup device and dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"Using device: {device}")
    
    # Parse DINOv3 backbone config - handle both flat and nested structures
    print("Parsing configuration...")
    model_section = config.get('model', config)
    backbone_config_dict = model_section.get('backbone', {})
    
    if not backbone_config_dict:
        raise ValueError("Could not find 'model.backbone' section in config")
    
    print(f"Backbone config keys: {list(backbone_config_dict.keys())}")
    
    # Initialize DINOv3 backbone
    print("Initializing DINOv3 backbone...")
    dinov3_config = Dinov3BackboneConfig(
        repo_path=backbone_config_dict.get('repo_path'),
        model_name=backbone_config_dict.get('model_name', 'dinov3_vitl16'),
        checkpoint_path=backbone_config_dict.get('checkpoint_path'),
        image_size=backbone_config_dict.get('image_size', 518),
        output_layers=tuple(backbone_config_dict.get('output_layers', [4, 8, 12])),
        enable_objectness=True,  # Force enable for visualization
        enable_pca=backbone_config_dict.get('enable_pca', True),
        pca_dim=backbone_config_dict.get('pca_dim', 32),
    )
    backbone = Dinov3Backbone(dinov3_config, device=device, dtype=dtype)
    
    # Extract features
    print("Extracting DINOv3 features...")
    features = backbone.extract_features(image_rgb)
    
    patch_tokens = features['patch_tokens']
    patch_tokens_raw = features.get('patch_tokens_raw', patch_tokens)
    grid_size = features['grid_size']
    processed_shape = features['processed_image_shape']
    objectness_map = features.get('objectness_map')
    attention_map = features.get('attention_map')
    
    print(f"Feature extraction complete:")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Processed shape: {processed_shape}")
    print(f"  - Patch tokens shape: {patch_tokens.shape}")
    print(f"  - Patch tokens raw shape: {patch_tokens_raw.shape}")
    
    # Save objectness map
    if objectness_map is not None:
        print("Saving objectness map...")
        save_heatmap(
            objectness_map,
            output_dir / "objectness",
            image_rgb
        )
    else:
        print("⚠️ No objectness map available")
    
    # Save attention map
    if attention_map is not None:
        print("Saving attention map...")
        save_heatmap(
            attention_map,
            output_dir / "attention",
            image_rgb
        )
    else:
        print("⚠️ No attention map available")
    
    # Parse detection adapter config
    detection_config_dict = model_section.get('detection_adapter', {})
    if not detection_config_dict:
        print("⚠️ No detection adapter config found, skipping detection")
    else:
        # Initialize detection adapter with lower threshold
        print(f"Initializing detection adapter (threshold: {args.detection_threshold})...")
        detection_config = DetectionAdapterConfig(
            checkpoint_path=detection_config_dict.get('checkpoint_path', ''),
            feature_dim=patch_tokens_raw.shape[-1],  # Use actual feature dim
            num_classes=detection_config_dict.get('num_classes', 91),
            score_threshold=args.detection_threshold
        )
        detection_adapter = build_detection_adapter(
            detection_config,
            device=str(device),
            torch_dtype=dtype
        )
        
        # Run detection
        print("Running detection...")
        processed_hw = (processed_shape[1], processed_shape[0])
        detection = detection_adapter.predict(
            patch_tokens_raw,
            image_size=processed_hw,
            grid_size=grid_size
        )
        
        print(f"Detection complete: {len(detection.boxes)} boxes found")
        
        if len(detection.boxes) > 0:
            print(f"  Detection scores range: [{detection.scores.min():.3f}, {detection.scores.max():.3f}]")
            print(f"  Top 5 scores: {sorted(detection.scores, reverse=True)[:5]}")
        
        # Visualize detections
        print("Visualizing detections...")
        visualize_detections(
            image_rgb,
            detection.boxes,
            detection.scores,
            detection.class_ids,
            detection.class_names or [],
            output_dir / "detections.png",
            processed_shape
        )
    
    # Parse segmentation adapter config
    segmentation_config_dict = model_section.get('segmentation_adapter', {})
    if not segmentation_config_dict:
        print("⚠️ No segmentation adapter config found, skipping segmentation")
    else:
        # Initialize segmentation adapter
        print("Initializing segmentation adapter...")
        segmentation_config = SegmentationAdapterConfig(
            checkpoint_path=segmentation_config_dict.get('checkpoint_path', ''),
            feature_dim=patch_tokens_raw.shape[-1],  # Use actual feature dim
            num_classes=segmentation_config_dict.get('num_classes', 150),
        )
        segmentation_adapter = build_segmentation_adapter(
            segmentation_config,
            device=str(device),
            torch_dtype=dtype
        )
        
        # Run segmentation
        print("Running segmentation...")
        segmentation = segmentation_adapter.predict(
            patch_tokens_raw,
            image_size=processed_hw,
            grid_size=grid_size
        )
        
        print(f"Segmentation complete: {segmentation.logits.shape[0]} classes")
        print(f"  Logits shape: {segmentation.logits.shape}")
        print(f"  Logits range: [{segmentation.logits.min():.3f}, {segmentation.logits.max():.3f}]")
        
        # Visualize segmentation
        print("Visualizing segmentation...")
        visualize_segmentation(
            image_rgb,
            segmentation.logits,
            segmentation.class_names or [],
            output_dir / "segmentation.png"
        )
    
    # Save metadata
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
        metadata.update({
            "num_detections": len(detection.boxes),
        })
    
    if 'segmentation' in locals():
        metadata.update({
            "num_segmentation_classes": segmentation.logits.shape[0],
        })
    
    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Visualization complete! Results saved to {output_dir}")
    if objectness_map is not None:
        print(f"  - objectness.png / objectness_overlay.png")
    if attention_map is not None:
        print(f"  - attention.png / attention_overlay.png")
    if 'detection' in locals():
        print(f"  - detections.png ({len(detection.boxes)} boxes)")
    if 'segmentation' in locals():
        print(f"  - segmentation.png / segmentation_overlay.png")
    print(f"  - metadata.json")
    
    if 'detection' in locals() and len(detection.boxes) == 0:
        print(f"\n💡 Tip: No detections found. Try:")
        print(f"     python {Path(__file__).name} --detection-threshold 0.05 ...")


if __name__ == "__main__":
    main()