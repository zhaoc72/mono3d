#!/usr/bin/env python3
"""
创建对比网格图 - 一目了然看到三种方法的差异

输出一个3x4的网格图:
       原图    | 热图    | Boxes  | 分割结果
方法1 objectness
方法2 combined
方法3 attention
"""

import cv2
import numpy as np
from pathlib import Path


def create_comparison_grid(output_dir, grid_path):
    """创建3x4对比网格图"""
    
    output_dir = Path(output_dir)
    
    # 检查所有必需的文件是否存在
    required_files = {
        'original': 'original.jpg',
        'obj_heat': 'objectness_heatmap.jpg',
        'obj_boxes': 'objectness_boxes.jpg',
        'obj_masks': 'objectness_masks.jpg',
        'comb_heat': 'combined_heatmap.jpg',
        'comb_boxes': 'combined_boxes.jpg',
        'comb_masks': 'combined_masks.jpg',
        'att_heat': 'attention_heatmap.jpg',
        'att_boxes': 'attention_boxes.jpg',
        'att_masks': 'attention_masks.jpg',
    }
    
    images = {}
    missing = []
    
    for key, filename in required_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            images[key] = cv2.imread(str(filepath))
        else:
            missing.append(filename)
    
    if missing:
        print(f"⚠️  缺少文件: {', '.join(missing)}")
        print("请先运行 quick_test.py")
        return False
    
    # 获取图像尺寸并调整
    target_h, target_w = 400, 400  # 每个小图的目标大小
    
    def resize_img(img):
        return cv2.resize(img, (target_w, target_h))
    
    # 调整所有图像大小
    for key in images:
        images[key] = resize_img(images[key])
    
    # 创建网格
    # 布局: 4列 x 3行
    # 列: 原图 | 热图 | Boxes | 分割结果
    # 行: Objectness | Combined | Attention
    
    # 添加标题栏
    header_h = 60
    col_labels = ['原图', '热图', 'Boxes', '分割结果']
    row_labels = ['方法1: Objectness', '方法2: Combined', '方法3: Attention (当前)']
    
    # 创建白色背景
    grid_h = header_h + target_h * 3 + 80  # 额外空间用于行标签
    grid_w = target_w * 4 + 200  # 额外空间用于列标签
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # 添加列标题
    for i, label in enumerate(col_labels):
        x = 150 + i * target_w + target_w // 2
        cv2.putText(
            grid, label, (x - 50, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
    
    # 放置图像
    rows_data = [
        ('obj', row_labels[0]),
        ('comb', row_labels[1]),
        ('att', row_labels[2]),
    ]
    
    for row_idx, (prefix, row_label) in enumerate(rows_data):
        y_offset = header_h + row_idx * target_h + 40
        
        # 添加行标签
        cv2.putText(
            grid, row_label, (10, y_offset + target_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        
        # 放置图像
        cols = [
            images['original'] if row_idx == 0 else None,  # 原图只在第一行显示
            images[f'{prefix}_heat'],
            images[f'{prefix}_boxes'],
            images[f'{prefix}_masks'],
        ]
        
        for col_idx, img in enumerate(cols):
            if img is not None:
                x_offset = 150 + col_idx * target_w
                grid[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = img
        
        # 如果不是第一行，在原图列显示一个指示
        if row_idx > 0:
            x_offset = 150
            text = "同上"
            cv2.putText(
                grid, text,
                (x_offset + target_w // 2 - 30, y_offset + target_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2
            )
    
    # 添加底部说明
    y_note = grid_h - 20
    note = "绿色框=检测框 | 彩色区域=分割结果"
    cv2.putText(
        grid, note, (150, y_note),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
    )
    
    # 保存
    cv2.imwrite(str(grid_path), grid)
    print(f"✅ 对比网格图已保存: {grid_path}")
    print(f"   尺寸: {grid.shape[1]}x{grid.shape[0]}")
    
    return True


def create_side_by_side(output_dir, side_path):
    """创建并排对比图 - 只显示最终分割结果"""
    
    output_dir = Path(output_dir)
    
    # 加载三个分割结果
    methods = ['objectness', 'combined', 'attention']
    images = []
    
    for method in methods:
        filepath = output_dir / f"{method}_masks.jpg"
        if filepath.exists():
            img = cv2.imread(str(filepath))
            images.append(img)
        else:
            print(f"⚠️  缺少文件: {filepath}")
            return False
    
    if len(images) != 3:
        print("无法创建并排对比图")
        return False
    
    # 调整大小
    target_h, target_w = 600, 600
    images = [cv2.resize(img, (target_w, target_h)) for img in images]
    
    # 创建并排图像
    margin = 20
    label_h = 60
    
    canvas_h = target_h + label_h + margin * 2
    canvas_w = target_w * 3 + margin * 4
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    labels = ['方法1: Objectness', '方法2: Combined', '方法3: Attention']
    
    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = margin + i * (target_w + margin)
        y_offset = label_h + margin
        
        # 放置图像
        canvas[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = img
        
        # 添加标签
        text_x = x_offset + target_w // 2 - 100
        cv2.putText(
            canvas, label, (text_x, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        
        # 添加边框
        cv2.rectangle(
            canvas,
            (x_offset, y_offset),
            (x_offset + target_w, y_offset + target_h),
            (200, 200, 200), 2
        )
    
    # 保存
    cv2.imwrite(str(side_path), canvas)
    print(f"✅ 并排对比图已保存: {side_path}")
    
    return True


if __name__ == "__main__":
    import sys
    
    output_dir = "/media/pc/D/zhaochen/mono3d/mono3d/outputs/quick_comparison"
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        print("请先运行 quick_test.py")
        sys.exit(1)
    
    print("=" * 80)
    print("创建可视化对比图")
    print("=" * 80)
    
    # 创建网格对比图
    print("\n1. 创建网格对比图...")
    grid_path = output_dir / "comparison_grid.jpg"
    success1 = create_comparison_grid(output_dir, grid_path)
    
    # 创建并排对比图
    print("\n2. 创建并排对比图...")
    side_path = output_dir / "comparison_side_by_side.jpg"
    success2 = create_side_by_side(output_dir, side_path)
    
    if success1 or success2:
        print("\n" + "=" * 80)
        print("✅ 完成！")
        print("=" * 80)
        print(f"\n查看结果:")
        if success1:
            print(f"  - {grid_path}")
        if success2:
            print(f"  - {side_path}")
    else:
        print("\n❌ 创建失败，请先运行 quick_test.py")