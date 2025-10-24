#!/usr/bin/env python3
"""
一键修复脚本 - 自动诊断并修复问题
"""
import sys
import shutil
from pathlib import Path

print("=" * 80)
print("mono3d 一键修复脚本")
print("=" * 80)

# 项目根目录
PROJECT_ROOT = Path("/media/pc/D/zhaochen/mono3d/mono3d")

print(f"\n当前项目目录: {PROJECT_ROOT}")
print(f"目录存在: {PROJECT_ROOT.exists()}")

if not PROJECT_ROOT.exists():
    print(f"\n❌ 项目目录不存在!")
    print(f"请确保路径正确: {PROJECT_ROOT}")
    sys.exit(1)

# 1. 备份原文件
print(f"\n1. 备份原文件...")
main_file = PROJECT_ROOT / "src" / "__main__.py"
if main_file.exists():
    backup_file = main_file.with_suffix(".py.backup")
    shutil.copy(main_file, backup_file)
    print(f"   ✓ 已备份到: {backup_file}")
else:
    print(f"   ⚠️  原文件不存在: {main_file}")

# 2. 复制修复后的文件
print(f"\n2. 复制修复后的文件...")
fixed_file = Path("/mnt/user-data/outputs/src_main_fixed.py")

if fixed_file.exists():
    shutil.copy(fixed_file, main_file)
    print(f"   ✓ 已复制修复文件到: {main_file}")
else:
    print(f"   ❌ 修复文件不存在: {fixed_file}")
    sys.exit(1)

# 3. 复制诊断脚本
print(f"\n3. 复制诊断脚本...")
diagnose_src = Path("/mnt/user-data/outputs/diagnose_single_image.py")
diagnose_dst = PROJECT_ROOT / "diagnose_single_image.py"

if diagnose_src.exists():
    shutil.copy(diagnose_src, diagnose_dst)
    print(f"   ✓ 已复制到: {diagnose_dst}")
else:
    print(f"   ⚠️  诊断脚本不存在")

# 4. 验证关键文件
print(f"\n4. 验证关键文件...")
key_files = [
    PROJECT_ROOT / "configs" / "model_config.yaml",
    PROJECT_ROOT / "configs" / "prompt_config.yaml",
    Path("/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vith16plus.pth"),
    Path("/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt"),
]

all_exist = True
for f in key_files:
    exists = f.exists()
    status = "✓" if exists else "❌"
    print(f"   {status} {f.name}: {exists}")
    if not exists:
        all_exist = False

if not all_exist:
    print(f"\n⚠️  部分关键文件不存在,可能导致运行失败")

# 5. 创建输出目录
print(f"\n5. 创建输出目录...")
output_dir = PROJECT_ROOT / "outputs" / "single_image_test"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"   ✓ 输出目录: {output_dir}")

# 6. 生成测试命令
print(f"\n6. 生成测试命令...")
commands = f"""
# 切换到项目目录
cd {PROJECT_ROOT}

# 方法1: 运行诊断脚本 (推荐)
python diagnose_single_image.py

# 方法2: 使用修复后的 CLI
python -m src.inference_pipeline image \\
  --input /media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg \\
  --output outputs/single_image_test \\
  --config configs/model_config.yaml \\
  --prompt-config configs/prompt_config.yaml

# 检查输出
ls -la outputs/single_image_test/
"""

script_file = PROJECT_ROOT / "run_test.sh"
with open(script_file, 'w') as f:
    f.write(commands)

print(f"   ✓ 测试脚本已保存: {script_file}")

print(f"\n" + "=" * 80)
print(f"✅ 修复完成!")
print(f"=" * 80)

print(f"\n下一步:")
print(f"1. cd {PROJECT_ROOT}")
print(f"2. python diagnose_single_image.py")
print(f"   或")
print(f"   bash run_test.sh")

print(f"\n如果仍有问题,请查看: /mnt/user-data/outputs/TROUBLESHOOTING.md")