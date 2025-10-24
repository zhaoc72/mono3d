#!/usr/bin/env python3
"""测试 SAM2 模型加载"""

import sys
from pathlib import Path

def test_sam2_loading():
    """测试不同的 SAM2 加载方法"""
    
    print("=" * 70)
    print("SAM2 模型加载测试")
    print("=" * 70)
    
    # 配置
    checkpoint_path = "/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt"
    config_name = "sam2.1/sam2.1_hiera_l"
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Config: {config_name}")
    
    # 方法 1: 使用 initialize_config_dir
    print("\n" + "=" * 70)
    print("方法 1: 使用 Hydra initialize_config_dir")
    print("=" * 70)
    
    try:
        import torch
        import sam2
        from pathlib import Path
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        # Find SAM2 config directory
        sam2_package_path = Path(sam2.__file__).parent
        config_dir = sam2_package_path / "configs"
        
        print(f"SAM2 包路径: {sam2_package_path}")
        print(f"配置目录: {config_dir}")
        print(f"配置目录存在: {config_dir.exists()}")
        
        if not config_dir.exists():
            print("❌ 配置目录不存在！")
        else:
            # Clear existing Hydra instance
            if GlobalHydra.instance().is_initialized():
                print("清理现有 Hydra 实例...")
                GlobalHydra.instance().clear()
            
            # Initialize with SAM2's config directory
            print(f"初始化 Hydra (config_dir={config_dir})...")
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                print(f"调用 build_sam2('{config_name}', ...)...")
                sam2_model = build_sam2(
                    config_name, 
                    checkpoint_path, 
                    device="cpu"
                )
                print("✅ 方法 1 成功！")
                print(f"模型类型: {type(sam2_model)}")
                return True
                
    except Exception as e:
        print(f"❌ 方法 1 失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法 2: 手动加载配置
    print("\n" + "=" * 70)
    print("方法 2: 手动加载 YAML 配置")
    print("=" * 70)
    
    try:
        import torch
        import yaml
        import sam2
        from pathlib import Path
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        
        # Find config file
        sam2_package_path = Path(sam2.__file__).parent
        config_path = sam2_package_path / "configs" / f"{config_name}.yaml"
        
        print(f"配置文件路径: {config_path}")
        print(f"配置文件存在: {config_path.exists()}")
        
        if not config_path.exists():
            print("❌ 配置文件不存在！")
        else:
            # Load YAML
            print("加载 YAML 配置...")
            with open(config_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            
            print("配置内容预览:")
            print(yaml.dump(cfg_dict, default_flow_style=False)[:500])
            
            # Convert to OmegaConf
            print("\n转换为 OmegaConf...")
            cfg = OmegaConf.create(cfg_dict)
            
            # Instantiate model
            print("实例化模型...")
            sam2_model = instantiate(cfg, _recursive_=False)
            
            # Load checkpoint
            print(f"加载 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle different formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            print(f"State dict keys 数量: {len(state_dict)}")
            
            # Load state dict
            print("加载 state dict...")
            missing, unexpected = sam2_model.load_state_dict(state_dict, strict=False)
            
            if missing:
                print(f"Missing keys: {len(missing)} (显示前5个)")
                for k in list(missing)[:5]:
                    print(f"  - {k}")
            
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} (显示前5个)")
                for k in list(unexpected)[:5]:
                    print(f"  - {k}")
            
            print("✅ 方法 2 成功！")
            print(f"模型类型: {type(sam2_model)}")
            return True
            
    except Exception as e:
        print(f"❌ 方法 2 失败: {e}")
        import traceback
        traceback.print_exc()
    
    return False


if __name__ == "__main__":
    success = test_sam2_loading()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ 至少一种方法成功！可以使用更新后的 sam2_segmenter.py")
    else:
        print("❌ 所有方法都失败了")
        print("\n可能的问题:")
        print("1. SAM2 安装不完整")
        print("2. Checkpoint 文件损坏或路径错误")
        print("3. 配置文件路径不匹配")
        print("\n建议:")
        print("1. 重新安装 SAM2:")
        print("   cd /media/pc/D/zhaochen/mono3d/sam2")
        print("   pip install -e .")
        print("2. 验证 checkpoint 文件:")
        print("   ls -lh /media/pc/D/zhaochen/mono3d/sam2/checkpoints/")
    print("=" * 70)
    
    sys.exit(0 if success else 1)