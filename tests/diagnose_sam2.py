#!/usr/bin/env python3
"""诊断脚本：查找正确的 SAM2 配置文件"""

import os
import sys
from pathlib import Path

def find_sam2_configs(sam2_root):
    """查找 SAM2 配置文件"""
    sam2_path = Path(sam2_root)
    
    print("=" * 60)
    print("SAM2 配置文件诊断")
    print("=" * 60)
    
    # 1. 检查 SAM2 根目录
    print(f"\n1. SAM2 根目录: {sam2_path}")
    print(f"   存在: {sam2_path.exists()}")
    
    if not sam2_path.exists():
        print("   ❌ 路径不存在！")
        return
    
    # 2. 查找所有 yaml 配置文件
    print("\n2. 查找所有 YAML 配置文件:")
    yaml_files = list(sam2_path.rglob("*.yaml"))
    
    if not yaml_files:
        print("   ❌ 未找到任何 YAML 文件")
        return
    
    print(f"   找到 {len(yaml_files)} 个 YAML 文件:\n")
    
    config_files = []
    for yaml_file in sorted(yaml_files):
        rel_path = yaml_file.relative_to(sam2_path)
        print(f"   - {rel_path}")
        
        # 查找包含 "hiera" 的配置文件
        if "hiera" in str(yaml_file).lower():
            config_files.append((yaml_file, rel_path))
    
    # 3. 分析 hiera 配置文件
    if config_files:
        print("\n3. Hiera 模型配置文件:")
        for full_path, rel_path in config_files:
            print(f"\n   📄 {rel_path}")
            print(f"      完整路径: {full_path}")
            
            # 确定 Hydra 配置名称
            # 从 configs/ 目录开始的相对路径
            try:
                configs_idx = str(rel_path).index("configs/")
                hydra_path = str(rel_path)[configs_idx + 8:]  # 跳过 "configs/"
                if hydra_path.endswith(".yaml"):
                    hydra_path = hydra_path[:-5]  # 移除 .yaml
                print(f"      Hydra 名称: '{hydra_path}'")
            except ValueError:
                # 如果没有 configs/ 目录，使用文件名
                hydra_name = rel_path.stem
                print(f"      Hydra 名称: '{hydra_name}'")
    
    # 4. 检查 sam2 包的 configs 目录
    print("\n4. 检查 sam2 包内的 configs 目录:")
    
    # 可能的 configs 目录位置
    possible_configs = [
        sam2_path / "sam2" / "configs",
        sam2_path / "configs",
    ]
    
    for config_dir in possible_configs:
        if config_dir.exists():
            print(f"\n   ✅ 找到: {config_dir}")
            subdirs = [d for d in config_dir.iterdir() if d.is_dir()]
            if subdirs:
                print(f"   子目录:")
                for subdir in sorted(subdirs):
                    print(f"      - {subdir.name}")
                    yaml_in_subdir = list(subdir.glob("*.yaml"))
                    for yf in sorted(yaml_in_subdir):
                        print(f"         • {yf.name}")
        else:
            print(f"   ❌ 不存在: {config_dir}")
    
    # 5. 推荐配置
    print("\n5. 推荐的 model_config.yaml 配置:")
    print("\n" + "=" * 60)
    
    if config_files:
        print("根据找到的配置文件，建议使用以下配置之一：\n")
        
        for full_path, rel_path in config_files:
            if "sam2.1" in str(rel_path) and "hiera_l" in str(rel_path):
                print("✅ 推荐（SAM 2.1 Large）:")
                print(f"""
sam2:
  backend: "official"
  checkpoint_path: "{sam2_path}/checkpoints/sam2.1_hiera_large.pt"
  model_config: "sam2.1/sam2.1_hiera_l"
""")
            
            try:
                configs_idx = str(rel_path).index("configs/")
                hydra_path = str(rel_path)[configs_idx + 8:]
                if hydra_path.endswith(".yaml"):
                    hydra_path = hydra_path[:-5]
                
                if "hiera_l" in hydra_path.lower():
                    print(f"备选方案:")
                    print(f"""
sam2:
  backend: "official"
  checkpoint_path: "<你的checkpoint路径>"
  model_config: "{hydra_path}"
""")
            except ValueError:
                pass
    
    print("=" * 60)
    
    # 6. 测试导入
    print("\n6. 测试 SAM2 导入:")
    try:
        sys.path.insert(0, str(sam2_path))
        from sam2.build_sam import build_sam2
        print("   ✅ SAM2 导入成功")
        
        # 尝试获取配置搜索路径
        try:
            import hydra
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            print("\n   Hydra 信息:")
            print(f"   Hydra 版本: {hydra.__version__}")
        except Exception as e:
            print(f"   ⚠️  Hydra 检查失败: {e}")
            
    except ImportError as e:
        print(f"   ❌ SAM2 导入失败: {e}")
        print(f"   请确保已安装 SAM2: pip install -e {sam2_path}")


if __name__ == "__main__":
    # 默认路径
    default_path = "/media/pc/D/zhaochen/mono3d/sam2"
    
    if len(sys.argv) > 1:
        sam2_root = sys.argv[1]
    else:
        sam2_root = default_path
    
    print(f"使用 SAM2 路径: {sam2_root}")
    print("(可以通过命令行参数指定其他路径: python diagnose_sam2.py <path>)\n")
    
    find_sam2_configs(sam2_root)