#!/usr/bin/env python3
"""
诊断 CLI 运行问题
"""
import sys
import subprocess
from pathlib import Path

print("=" * 80)
print("CLI 诊断")
print("=" * 80)

# 项目目录
project_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d")
print(f"\n项目目录: {project_dir}")
print(f"存在: {project_dir.exists()}")

if not project_dir.exists():
    print("❌ 项目目录不存在")
    sys.exit(1)

# 检查 src/__main__.py
main_file = project_dir / "src" / "__main__.py"
print(f"\n检查 src/__main__.py: {main_file.exists()}")

if not main_file.exists():
    print("❌ src/__main__.py 不存在")
    print("\n需要复制修复文件:")
    print(f"  cp /mnt/user-data/outputs/src_main_fixed.py {main_file}")
    sys.exit(1)

# 检查文件内容
print("\n检查文件前 20 行...")
try:
    with open(main_file, 'r') as f:
        lines = f.readlines()[:20]
        for i, line in enumerate(lines, 1):
            print(f"  {i:2d}: {line.rstrip()}")
except Exception as e:
    print(f"❌ 无法读取文件: {e}")
    sys.exit(1)

# 尝试导入
print("\n测试导入...")
sys.path.insert(0, str(project_dir))

try:
    from src import __main__ as main_module
    print("✓ 成功导入 src.__main__")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查 main 函数
if hasattr(main_module, 'main'):
    print("✓ main 函数存在")
else:
    print("❌ main 函数不存在")
    sys.exit(1)

# 测试参数解析
print("\n测试参数解析...")
test_args = [
    'image',
    '--input', '/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg',
    '--output', 'outputs/single_image',
    '--config', 'configs/model_config.yaml'
]

# 保存原始 argv
original_argv = sys.argv.copy()
sys.argv = ['src.inference_pipeline'] + test_args

try:
    if hasattr(main_module, 'parse_args'):
        args = main_module.parse_args()
        print("✓ 参数解析成功")
        print(f"  mode: {args.mode}")
        print(f"  input: {args.input}")
        print(f"  output: {args.output}")
        print(f"  config: {args.config}")
    else:
        print("⚠️  parse_args 函数不存在")
except SystemExit as e:
    print(f"❌ 参数解析失败: SystemExit({e.code})")
except Exception as e:
    print(f"❌ 参数解析失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    sys.argv = original_argv

# 运行实际命令
print("\n" + "=" * 80)
print("运行实际 CLI 命令...")
print("=" * 80)

cmd = [
    sys.executable, '-m', 'src.inference_pipeline',
    'image',
    '--input', '/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg',
    '--output', 'outputs/single_image',
    '--config', 'configs/model_config.yaml'
]

print(f"\n命令: {' '.join(cmd)}")
print(f"工作目录: {project_dir}\n")

try:
    result = subprocess.run(
        cmd,
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        timeout=120
    )
    
    print("=" * 80)
    print("标准输出:")
    print("=" * 80)
    if result.stdout:
        print(result.stdout)
    else:
        print("(无输出)")
    
    print("\n" + "=" * 80)
    print("标准错误:")
    print("=" * 80)
    if result.stderr:
        print(result.stderr)
    else:
        print("(无错误)")
    
    print("\n" + "=" * 80)
    print(f"返回码: {result.returncode}")
    print("=" * 80)
    
    if result.returncode == 0:
        print("\n✅ 命令执行成功")
        
        # 检查输出
        output_dir = project_dir / "outputs" / "single_image"
        if output_dir.exists():
            print(f"\n输出目录存在: {output_dir}")
            
            # 列出文件
            import os
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(str(output_dir), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
        else:
            print(f"\n⚠️  输出目录不存在: {output_dir}")
    else:
        print(f"\n❌ 命令失败，返回码: {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("\n❌ 命令超时 (120秒)")
except Exception as e:
    print(f"\n❌ 运行失败: {e}")
    import traceback
    traceback.print_exc()