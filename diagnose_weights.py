#!/usr/bin/env python3
"""
Diagnostic script to check DINOv3 adapter weights.
This helps identify why weights are not loading correctly.
"""

import sys
from pathlib import Path
import torch

def check_checkpoint(checkpoint_path: str, name: str = "Checkpoint"):
    """Check a checkpoint file and print its structure."""
    print(f"\n{'='*60}")
    print(f"Checking {name}: {checkpoint_path}")
    print('='*60)
    
    if not Path(checkpoint_path).exists():
        print(f"❌ File does not exist!")
        return
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        # Check structure
        if isinstance(checkpoint, dict):
            print(f"\n📦 Checkpoint is a dictionary with keys:")
            for key in checkpoint.keys():
                print(f"  - {key}")
            
            # Check for common weight keys
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"\n🔑 Found 'model' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\n🔑 Found 'state_dict' key")
            else:
                state_dict = checkpoint
                print(f"\n🔑 Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            print(f"\n📦 Checkpoint is a state_dict directly")
        
        # Analyze state dict
        if isinstance(state_dict, dict):
            print(f"\n📊 State dict statistics:")
            print(f"  - Total parameters: {len(state_dict)}")
            
            # Show first 10 keys
            keys = list(state_dict.keys())
            print(f"\n  First 10 parameter names:")
            for i, key in enumerate(keys[:10]):
                tensor = state_dict[key]
                if isinstance(tensor, torch.Tensor):
                    print(f"    {i+1}. {key}: {tuple(tensor.shape)}")
                else:
                    print(f"    {i+1}. {key}: {type(tensor)}")
            
            if len(keys) > 10:
                print(f"    ... and {len(keys) - 10} more")
            
            # Check for common patterns
            print(f"\n  Key patterns:")
            patterns = {
                'backbone': sum(1 for k in keys if 'backbone' in k.lower()),
                'encoder': sum(1 for k in keys if 'encoder' in k.lower()),
                'decoder': sum(1 for k in keys if 'decoder' in k.lower()),
                'head': sum(1 for k in keys if 'head' in k.lower()),
                'linear': sum(1 for k in keys if 'linear' in k.lower()),
                'conv': sum(1 for k in keys if 'conv' in k.lower()),
                'norm': sum(1 for k in keys if 'norm' in k.lower()),
                'attn': sum(1 for k in keys if 'attn' in k.lower()),
            }
            
            for pattern, count in patterns.items():
                if count > 0:
                    print(f"    - '{pattern}': {count} parameters")
            
            # Check first tensor shape for dimension info
            first_tensor_key = None
            for key in keys:
                if isinstance(state_dict[key], torch.Tensor):
                    first_tensor_key = key
                    break
            
            if first_tensor_key:
                first_tensor = state_dict[first_tensor_key]
                print(f"\n  Sample tensor ({first_tensor_key}):")
                print(f"    - Shape: {tuple(first_tensor.shape)}")
                print(f"    - Dtype: {first_tensor.dtype}")
                print(f"    - Device: {first_tensor.device}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🔍 DINOv3 Adapter Weight Diagnostic Tool")
    
    # Default paths
    backbone_path = "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth"
    detection_path = "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_coco.pth"
    segmentation_path = "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_ade20k.pth"
    
    # Check backbone
    check_checkpoint(backbone_path, "DINOv3 Backbone (ViT-L/16)")
    
    # Check detection adapter
    check_checkpoint(detection_path, "Detection Adapter (ViT-7B COCO)")
    
    # Check segmentation adapter
    check_checkpoint(segmentation_path, "Segmentation Adapter (ViT-7B ADE20K)")
    
    print(f"\n{'='*60}")
    print("💡 Analysis Complete")
    print('='*60)
    
    print("\n📝 Common Issues:")
    print("  1. If checkpoint doesn't exist: Check file paths")
    print("  2. If keys don't match: Adapters might be for full ViT-7B, not just heads")
    print("  3. If weights are for ViT-7B: They include the full backbone + head")
    print("     → Solution: Use random initialization for heads (already implemented)")
    
    print("\n🔧 Recommendations:")
    print("  1. ViT-L/16 backbone (300M) can work with random-initialized heads")
    print("  2. Lower detection threshold: --detection-threshold 0.05")
    print("  3. If you have separate head weights, update checkpoint paths")
    
    print("\n🚀 Quick Fix:")
    print("  python visualize_dinov3_v2.py \\")
    print("    --input /path/to/image.jpg \\")
    print("    --output outputs/result \\")
    print("    --config configs/model_config.yaml \\")
    print("    --detection-threshold 0.05")


if __name__ == "__main__":
    main()