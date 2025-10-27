#!/usr/bin/env python3
"""
Enhanced diagnostic script to check DINOv3 adapter weights.
This version includes better error handling and more detailed inspection.
"""

import sys
import os
from pathlib import Path
import torch
import argparse
from datetime import datetime


class TeeOutput:
    """Redirect output to both console and file."""
    
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_output_logging(output_file=None, auto_save=False):
    """
    Setup output logging to file.
    
    Args:
        output_file: Specific output file path
        auto_save: If True, auto-generate filename with timestamp
    
    Returns:
        TeeOutput instance or None
    """
    if output_file:
        log_path = Path(output_file)
    elif auto_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"diagnostic_{timestamp}.log")
    else:
        return None
    
    # Create parent directories if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Saving diagnostic output to: {log_path.absolute()}")
    print(f"{'='*70}\n")
    
    tee = TeeOutput(log_path)
    sys.stdout = tee
    sys.stderr = tee
    
    return tee


def check_checkpoint(checkpoint_path: str, name: str = "Checkpoint"):
    """Check a checkpoint file and print its structure with enhanced details."""
    print(f"\n{'='*70}")
    print(f"🔍 Checking {name}")
    print(f"📁 Path: {checkpoint_path}")
    print('='*70)
    
    # Check if file exists
    path_obj = Path(checkpoint_path)
    if not path_obj.exists():
        print(f"❌ File does not exist!")
        print(f"   Searched at: {path_obj.absolute()}")
        return False
    
    # Check file size
    file_size_mb = path_obj.stat().st_size / (1024 * 1024)
    print(f"✅ File exists ({file_size_mb:.2f} MB)")
    
    try:
        # Load checkpoint
        print(f"📥 Loading checkpoint (this may take a moment)...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded successfully")
        
        # Determine checkpoint structure
        state_dict = None
        checkpoint_type = None
        
        if isinstance(checkpoint, dict):
            print(f"\n📦 Checkpoint is a dictionary with {len(checkpoint)} top-level keys:")
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, dict):
                    print(f"  - '{key}': dict with {len(value)} items")
                elif isinstance(value, torch.Tensor):
                    print(f"  - '{key}': Tensor {tuple(value.shape)}")
                else:
                    print(f"  - '{key}': {type(value).__name__}")
            
            # Try to extract state_dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                checkpoint_type = "model"
                print(f"\n🔑 Found 'model' key with {len(state_dict)} parameters")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                checkpoint_type = "state_dict"
                print(f"\n🔑 Found 'state_dict' key with {len(state_dict)} parameters")
            elif 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
                checkpoint_type = "teacher"
                print(f"\n🔑 Found 'teacher' key with {len(state_dict)} parameters")
            elif 'student' in checkpoint:
                state_dict = checkpoint['student']
                checkpoint_type = "student"
                print(f"\n🔑 Found 'student' key with {len(state_dict)} parameters")
            else:
                # Assume checkpoint itself is the state_dict
                state_dict = checkpoint
                checkpoint_type = "direct"
                print(f"\n🔑 Using checkpoint directly as state_dict ({len(state_dict)} items)")
        else:
            # Not a dictionary, might be direct state_dict
            if hasattr(checkpoint, 'items'):
                state_dict = checkpoint
                checkpoint_type = "direct"
                print(f"\n📦 Checkpoint is directly a state_dict with {len(state_dict)} parameters")
            else:
                print(f"\n❌ Unexpected checkpoint type: {type(checkpoint)}")
                return False
        
        # Analyze state dict
        if state_dict and isinstance(state_dict, dict):
            print(f"\n📊 State dict analysis:")
            print(f"  ├─ Total parameters: {len(state_dict)}")
            
            # Analyze parameter types
            tensor_count = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
            non_tensor_count = len(state_dict) - tensor_count
            print(f"  ├─ Tensors: {tensor_count}")
            if non_tensor_count > 0:
                print(f"  ├─ Non-tensors: {non_tensor_count}")
            
            # Calculate total parameters
            total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
            print(f"  └─ Total parameter count: {total_params:,} ({total_params/1e6:.1f}M)")
            
            # Show parameter keys
            keys = list(state_dict.keys())
            print(f"\n  📝 First 15 parameter names:")
            for i, key in enumerate(keys[:15]):
                tensor = state_dict[key]
                if isinstance(tensor, torch.Tensor):
                    print(f"    {i+1:2d}. {key}: {tuple(tensor.shape)}")
                else:
                    print(f"    {i+1:2d}. {key}: {type(tensor).__name__}")
            
            if len(keys) > 15:
                print(f"    ... and {len(keys) - 15} more parameters")
            
            # Check for common patterns
            print(f"\n  🔍 Key pattern analysis:")
            patterns = {
                'backbone': [k for k in keys if 'backbone' in k.lower()],
                'encoder': [k for k in keys if 'encoder' in k.lower()],
                'decoder': [k for k in keys if 'decoder' in k.lower()],
                'head': [k for k in keys if 'head' in k.lower()],
                'linear': [k for k in keys if 'linear' in k.lower()],
                'classifier': [k for k in keys if 'classifier' in k.lower() or 'class_embed' in k.lower()],
                'bbox': [k for k in keys if 'bbox' in k.lower()],
                'conv': [k for k in keys if 'conv' in k.lower()],
                'norm': [k for k in keys if 'norm' in k.lower()],
                'attn': [k for k in keys if 'attn' in k.lower()],
                'blocks': [k for k in keys if 'blocks' in k.lower() or 'block' in k.lower()],
                'patch_embed': [k for k in keys if 'patch_embed' in k.lower()],
            }
            
            for pattern, matched_keys in patterns.items():
                if matched_keys:
                    print(f"    - '{pattern}': {len(matched_keys)} parameters")
                    if len(matched_keys) <= 3:
                        for k in matched_keys:
                            print(f"        • {k}")
            
            # Analyze model architecture
            print(f"\n  🏗️  Architecture hints:")
            
            # Check for ViT architecture
            if any('blocks' in k for k in keys):
                num_blocks = len(set(k.split('.')[k.split('.').index('blocks')+1] 
                                   for k in keys if 'blocks' in k and len(k.split('.')) > k.split('.').index('blocks')+1))
                print(f"    - Vision Transformer with ~{num_blocks} blocks detected")
            
            # Check patch embedding
            patch_embed_keys = [k for k in keys if 'patch_embed' in k.lower()]
            if patch_embed_keys:
                print(f"    - Patch embedding layer found")
                for k in patch_embed_keys[:2]:
                    if isinstance(state_dict[k], torch.Tensor):
                        print(f"        • {k}: {tuple(state_dict[k].shape)}")
            
            # Check for detection/segmentation heads
            if any('class_embed' in k or 'classifier' in k for k in keys):
                print(f"    - Classification head detected")
            if any('bbox' in k for k in keys):
                print(f"    - Bounding box regression head detected")
            
            # Sample tensor statistics
            print(f"\n  📈 Sample tensor statistics:")
            sample_keys = [k for k in keys if isinstance(state_dict[k], torch.Tensor)][:3]
            for key in sample_keys:
                tensor = state_dict[key]
                print(f"    • {key}:")
                print(f"        Shape: {tuple(tensor.shape)}")
                print(f"        Dtype: {tensor.dtype}")
                print(f"        Mean: {tensor.float().mean():.6f}, Std: {tensor.float().std():.6f}")
                print(f"        Min: {tensor.min():.6f}, Max: {tensor.max():.6f}")
            
            return True
        else:
            print(f"\n❌ Could not extract valid state_dict")
            return False
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False


def suggest_fixes(results: dict):
    """Provide actionable suggestions based on diagnostic results."""
    print(f"\n{'='*70}")
    print("💡 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print('='*70)
    
    all_exist = all(results.values())
    
    if not all_exist:
        print("\n❌ Missing checkpoints detected:")
        for name, exists in results.items():
            if not exists:
                print(f"   - {name}")
        print("\n📝 Next steps:")
        print("   1. Verify the checkpoint paths in your config file")
        print("   2. Download missing checkpoints from the official sources")
        print("   3. Update paths in configs/model_config.yaml")
    
    print("\n🔧 Configuration tips:")
    print("   1. DINOv3 backbone checkpoint:")
    print("      - ViT-L/16: ~300M parameters")
    print("      - Should contain 'blocks', 'patch_embed', etc.")
    print("      - Download from: https://github.com/facebookresearch/dinov2")
    
    print("\n   2. Detection adapter:")
    print("      - If checkpoint is full ViT-7B model: Use random initialization instead")
    print("      - Set checkpoint_path: '' in config to use random weights")
    print("      - Random init is fine for zero-shot tasks")
    
    print("\n   3. Segmentation adapter:")
    print("      - Similar to detection adapter")
    print("      - Random initialization works for exploratory tasks")
    
    print("\n🚀 Quick fix for testing:")
    print("   Edit configs/model_config.yaml:")
    print("   ```yaml")
    print("   detection_adapter:")
    print("     checkpoint_path: ''  # Use random initialization")
    print("   segmentation_adapter:")
    print("     checkpoint_path: ''  # Use random initialization")
    print("   ```")
    
    print("\n⚡ Performance tips:")
    print("   - Lower detection threshold: score_threshold: 0.05")
    print("   - Increase max_prompts: max_prompts: 100")
    print("   - Enable objectness: enable_objectness: true")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose DINOv3 and adapter checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python diagnose_weights_fixed.py
  python diagnose_weights_fixed.py --backbone /path/to/dinov3.pth
  python diagnose_weights_fixed.py --config configs/model_config.yaml
  python diagnose_weights_fixed.py --config configs/model_config.yaml --save-output diagnostic.log
        """
    )
    
    parser.add_argument(
        "--backbone",
        default="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth",
        help="Path to DINOv3 backbone checkpoint"
    )
    parser.add_argument(
        "--detection",
        default="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_coco.pth",
        help="Path to detection adapter checkpoint"
    )
    parser.add_argument(
        "--segmentation",
        default="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_ade20k.pth",
        help="Path to segmentation adapter checkpoint"
    )
    parser.add_argument(
        "--config",
        help="Load checkpoint paths from config YAML file"
    )
    parser.add_argument(
        "--save-output",
        "--output",
        "-o",
        metavar="FILE",
        help="Save diagnostic output to file (e.g., diagnostic.log)"
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Automatically save output to diagnostic_YYYYMMDD_HHMMSS.log"
    )
    
    return parser.parse_args()


def main():
    """Main diagnostic routine."""
    args = parse_args()
    
    # Setup output logging
    tee = setup_output_logging(args.save_output, args.auto_save)
    
    try:
        print("🔍 DINOv3 Adapter Weight Diagnostic Tool")
        print("=" * 70)
        
        # Load paths from config if provided
        if args.config:
            try:
                import yaml
                print(f"\n📖 Loading config from: {args.config}")
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
                
                backbone_path = config.get('dinov3', {}).get('checkpoint_path', args.backbone)
                detection_path = config.get('detection_adapter', {}).get('checkpoint_path', args.detection)
                segmentation_path = config.get('segmentation_adapter', {}).get('checkpoint_path', args.segmentation)
                
                print(f"✅ Config loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load config ({e}), using command-line arguments")
                backbone_path = args.backbone
                detection_path = args.detection
                segmentation_path = args.segmentation
        else:
            backbone_path = args.backbone
            detection_path = args.detection
            segmentation_path = args.segmentation
        
        # Check each checkpoint
        results = {
            "backbone": False,
            "detection": False,
            "segmentation": False
        }
        
        results["backbone"] = check_checkpoint(backbone_path, "DINOv3 Backbone (ViT-L/16)")
        results["detection"] = check_checkpoint(detection_path, "Detection Adapter (ViT-7B COCO)")
        results["segmentation"] = check_checkpoint(segmentation_path, "Segmentation Adapter (ViT-7B ADE20K)")
        
        # Provide suggestions
        suggest_fixes(results)
        
        print(f"\n{'='*70}")
        print("✅ Diagnostic complete!")
        print('='*70)
        
        return_code = 0 if all(results.values()) else 1
        
    finally:
        # Close log file if it was opened
        if tee:
            print(f"\n📝 Diagnostic output saved to: {tee.log.name}")
            tee.close()
            sys.stdout = tee.terminal
            sys.stderr = tee.terminal
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())