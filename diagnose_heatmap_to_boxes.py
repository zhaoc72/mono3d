#!/usr/bin/env python3
"""
Heatmap to Boxes 诊断工具
详细可视化从热图到候选框的每一步转换过程
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/claude')
from improved_dinov3_extractor import ImprovedDINOv3Extractor


def visualize_heatmap_to_boxes_pipeline(
    heatmap: np.ndarray,
    image: np.ndarray,
    output_dir: Path,
    method_name: str,
    percentile: float = 65.0,
    smoothing_kernel: int = 11,
    min_component_area: int = 4000
):
    """
    可视化从 heatmap 到 boxes 的完整流程
    
    流程：
    1. 原始热图
    2. 归一化
    3. 平滑（高斯模糊）
    4. 阈值化（二值化）
    5. 形态学操作（闭运算+开运算）
    6. 连通组件分析
    7. 生成边界框
    """
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print(f"详细分析: {method_name}")
    print(f"{'='*80}")
    
    # 用于存储每一步的结果
    steps = {}
    
    # ==================== 步骤 1: 原始热图 ====================
    print(f"\n步骤 1: 原始热图")
    print(f"  范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"  均值: {heatmap.mean():.4f}")
    print(f"  标准差: {heatmap.std():.4f}")
    
    steps['01_original_heatmap'] = heatmap.copy()
    
    # ==================== 步骤 2: 归一化 ====================
    print(f"\n步骤 2: 归一化到 [0, 1]")
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    print(f"  新范围: [{heatmap_norm.min():.4f}, {heatmap_norm.max():.4f}]")
    
    steps['02_normalized'] = heatmap_norm.copy()
    
    # ==================== 步骤 3: 平滑 ====================
    print(f"\n步骤 3: 高斯平滑")
    if smoothing_kernel > 0:
        kernel_size = smoothing_kernel if smoothing_kernel % 2 == 1 else smoothing_kernel + 1
        heatmap_smooth = cv2.GaussianBlur(
            heatmap_norm, 
            (kernel_size, kernel_size), 
            0
        )
        print(f"  核大小: {kernel_size}x{kernel_size}")
        print(f"  平滑后范围: [{heatmap_smooth.min():.4f}, {heatmap_smooth.max():.4f}]")
    else:
        heatmap_smooth = heatmap_norm.copy()
        print(f"  跳过平滑")
    
    steps['03_smoothed'] = heatmap_smooth.copy()
    
    # ==================== 步骤 4: 阈值化 ====================
    print(f"\n步骤 4: 阈值化")
    threshold = np.percentile(heatmap_smooth, percentile)
    print(f"  百分位: {percentile}%")
    print(f"  阈值: {threshold:.4f}")
    
    binary_mask = (heatmap_smooth > threshold).astype(np.uint8)
    num_foreground_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    foreground_ratio = num_foreground_pixels / total_pixels * 100
    
    print(f"  前景像素: {num_foreground_pixels} ({foreground_ratio:.2f}%)")
    print(f"  背景像素: {total_pixels - num_foreground_pixels} ({100-foreground_ratio:.2f}%)")
    
    steps['04_thresholded'] = binary_mask.copy()
    
    # ==================== 步骤 5: 形态学操作 ====================
    print(f"\n步骤 5: 形态学操作")
    
    # 5.1 闭运算（先膨胀后腐蚀，填充小孔）
    kernel = np.ones((3, 3), np.uint8)
    after_close = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    print(f"  闭运算 (iterations=2): 填充小孔")
    print(f"    前景像素变化: {binary_mask.sum()} → {after_close.sum()}")
    
    steps['05_after_close'] = after_close.copy()
    
    # 5.2 开运算（先腐蚀后膨胀，去除小噪点）
    after_open = cv2.morphologyEx(after_close, cv2.MORPH_OPEN, kernel, iterations=1)
    print(f"  开运算 (iterations=1): 去除噪点")
    print(f"    前景像素变化: {after_close.sum()} → {after_open.sum()}")
    
    steps['06_after_open'] = after_open.copy()
    
    # ==================== 步骤 6: 连通组件分析 ====================
    print(f"\n步骤 6: 连通组件分析")
    labeled_array, num_features = ndimage.label(after_open)
    print(f"  发现连通组件: {num_features} 个")
    
    # 分析每个连通组件
    component_stats = []
    for label_id in range(1, num_features + 1):
        component_mask = (labeled_array == label_id)
        area = component_mask.sum()
        
        coords = np.where(component_mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = width / height if height > 0 else 0
        
        component_stats.append({
            'id': label_id,
            'area': area,
            'bbox': (x_min, y_min, x_max, y_max),
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'passed': area >= min_component_area
        })
    
    # 按面积排序
    component_stats.sort(key=lambda x: x['area'], reverse=True)
    
    print(f"\n  连通组件详情 (按面积排序):")
    print(f"  {'ID':<4} {'面积':<10} {'宽×高':<12} {'长宽比':<8} {'状态':<6}")
    print(f"  {'-'*50}")
    
    for stat in component_stats:
        status = "✅ PASS" if stat['passed'] else "❌ FAIL"
        print(f"  {stat['id']:<4} {stat['area']:<10} "
              f"{stat['width']}×{stat['height']:<10} "
              f"{stat['aspect_ratio']:<8.2f} {status}")
    
    # 创建连通组件可视化（不同组件不同颜色）
    component_vis = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
    for label_id in range(1, num_features + 1):
        color = np.array([
            (label_id * 50) % 255,
            (label_id * 80 + 60) % 255,
            (label_id * 120 + 30) % 255
        ], dtype=np.uint8)
        component_vis[labeled_array == label_id] = color
    
    steps['07_components'] = component_vis
    
    # ==================== 步骤 7: 过滤并生成 Boxes ====================
    print(f"\n步骤 7: 过滤并生成边界框")
    print(f"  最小面积阈值: {min_component_area}")
    
    valid_components = [s for s in component_stats if s['passed']]
    print(f"  有效组件: {len(valid_components)} / {num_features}")
    
    if len(valid_components) == 0:
        print(f"  ⚠️  没有组件通过面积阈值！")
        print(f"  建议: 降低 min_component_area (当前: {min_component_area})")
    
    # 在图像上绘制 boxes
    img_with_boxes = image.copy()
    for i, stat in enumerate(valid_components):
        x_min, y_min, x_max, y_max = stat['bbox']
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, f"#{i}", (x_min, y_min-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    steps['08_final_boxes'] = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    
    # ==================== 保存所有步骤的可视化 ====================
    print(f"\n保存可视化结果...")
    
    # 保存每一步
    for step_name, step_data in steps.items():
        if step_name == '07_components' or step_name == '08_final_boxes':
            # 已经是 BGR 或 RGB
            if step_name == '07_components':
                cv2.imwrite(str(output_dir / f"{step_name}.jpg"), 
                           cv2.cvtColor(step_data, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(output_dir / f"{step_name}.jpg"), step_data)
        else:
            # 灰度图，转为彩色可视化
            if step_data.dtype == np.uint8:
                # 二值图
                vis = cv2.cvtColor(step_data * 255, cv2.COLOR_GRAY2BGR)
            else:
                # 浮点图，转为 0-255
                vis = (step_data * 255).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            
            cv2.imwrite(str(output_dir / f"{step_name}.jpg"), vis)
    
    # ==================== 创建流程对比图 ====================
    print(f"创建流程对比图...")
    
    # 准备8张图
    images_to_show = [
        ('01_original_heatmap', '1. 原始热图'),
        ('02_normalized', '2. 归一化'),
        ('03_smoothed', '3. 高斯平滑'),
        ('04_thresholded', '4. 阈值化'),
        ('05_after_close', '5. 闭运算'),
        ('06_after_open', '6. 开运算'),
        ('07_components', '7. 连通组件'),
        ('08_final_boxes', '8. 最终Boxes')
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Heatmap → Boxes 流程: {method_name}', fontsize=16, fontweight='bold')
    
    for idx, (step_name, title) in enumerate(images_to_show):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        step_data = steps[step_name]
        
        if step_name in ['07_components', '08_final_boxes']:
            # RGB 图像
            ax.imshow(step_data)
        elif step_data.dtype == np.uint8:
            # 二值图
            ax.imshow(step_data, cmap='gray', vmin=0, vmax=1)
        else:
            # 浮点热图
            im = ax.imshow(step_data, cmap='jet', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_overview.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== 保存参数和统计 ====================
    with open(output_dir / 'pipeline_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Heatmap → Boxes 流程分析: {method_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("--- 输入参数 ---\n")
        f.write(f"百分位阈值: {percentile}%\n")
        f.write(f"平滑核大小: {smoothing_kernel}\n")
        f.write(f"最小面积: {min_component_area}\n\n")
        
        f.write("--- 热图统计 ---\n")
        f.write(f"范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]\n")
        f.write(f"均值: {heatmap.mean():.4f}\n")
        f.write(f"标准差: {heatmap.std():.4f}\n")
        f.write(f"中位数: {np.median(heatmap):.4f}\n\n")
        
        f.write("--- 阈值化统计 ---\n")
        f.write(f"阈值: {threshold:.4f}\n")
        f.write(f"前景像素: {num_foreground_pixels} ({foreground_ratio:.2f}%)\n")
        f.write(f"背景像素: {total_pixels - num_foreground_pixels} ({100-foreground_ratio:.2f}%)\n\n")
        
        f.write("--- 连通组件分析 ---\n")
        f.write(f"总组件数: {num_features}\n")
        f.write(f"有效组件数: {len(valid_components)}\n")
        f.write(f"通过率: {len(valid_components)/num_features*100:.1f}%\n\n")
        
        f.write("连通组件详情:\n")
        for stat in component_stats:
            status = "PASS" if stat['passed'] else "FAIL"
            f.write(f"  组件 {stat['id']}: 面积={stat['area']}, "
                   f"大小={stat['width']}×{stat['height']}, "
                   f"长宽比={stat['aspect_ratio']:.2f}, {status}\n")
    
    print(f"✅ 完成！输出目录: {output_dir}/")
    
    return len(valid_components), component_stats


def test_different_parameters(
    heatmap: np.ndarray,
    image: np.ndarray,
    method_name: str,
    output_base_dir: Path
):
    """
    测试不同参数组合的效果
    """
    print(f"\n{'='*80}")
    print(f"参数对比测试: {method_name}")
    print(f"{'='*80}")
    
    # 测试不同的参数组合
    param_sets = [
        # (percentile, smoothing, min_area, name)
        (50, 11, 2000, "宽松-低阈值"),
        (65, 11, 4000, "默认参数"),
        (75, 11, 6000, "严格-高阈值"),
        (65, 5, 4000, "弱平滑"),
        (65, 15, 4000, "强平滑"),
        (65, 11, 1000, "小面积"),
    ]
    
    results = []
    
    for percentile, smoothing, min_area, param_name in param_sets:
        print(f"\n测试: {param_name}")
        print(f"  percentile={percentile}, smoothing={smoothing}, min_area={min_area}")
        
        output_dir = output_base_dir / method_name / f"params_{param_name.replace('-', '_')}"
        
        num_boxes, stats = visualize_heatmap_to_boxes_pipeline(
            heatmap, image, output_dir, f"{method_name} - {param_name}",
            percentile=percentile,
            smoothing_kernel=smoothing,
            min_component_area=min_area
        )
        
        results.append({
            'name': param_name,
            'percentile': percentile,
            'smoothing': smoothing,
            'min_area': min_area,
            'num_boxes': num_boxes,
            'stats': stats
        })
        
        print(f"  → 生成 {num_boxes} 个 boxes")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"参数对比总结")
    print(f"{'='*80}\n")
    
    print(f"{'参数组合':<20} {'百分位':<8} {'平滑':<6} {'最小面积':<10} {'Boxes数':<8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<20} {r['percentile']:<8} {r['smoothing']:<6} "
              f"{r['min_area']:<10} {r['num_boxes']:<8}")
    
    # 找出生成最多boxes的参数
    best = max(results, key=lambda x: x['num_boxes'])
    print(f"\n🏆 生成最多boxes的参数: {best['name']}")
    print(f"   生成了 {best['num_boxes']} 个 boxes")
    
    return results


def main():
    print("=" * 80)
    print("Heatmap → Boxes 详细诊断工具")
    print("=" * 80)
    
    # 配置
    image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
    output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/heatmap_boxes_diagnosis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n📋 配置:")
    print(f"  图像: {Path(image_path).name}")
    print(f"  设备: {device}")
    print(f"  输出: {output_dir}")
    
    # 加载图像
    print(f"\n📷 加载图像...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = (image.shape[1], image.shape[0])
    print(f"   ✅ {image_size[0]}x{image_size[1]}")
    
    # 加载 DINOv3
    print(f"\n🔧 加载 DINOv3...")
    dinov3_model = torch.hub.load(
        '/media/pc/D/zhaochen/mono3d/dinov3',
        'dinov3_vith16plus',
        source='local',
        trust_repo=True,
        pretrained=False
    )
    state = torch.load(
        '/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vith16plus.pth',
        map_location='cpu'
    )
    dinov3_model.load_state_dict(state, strict=False)
    dinov3_model = dinov3_model.to(device).eval()
    print(f"   ✅ DINOv3 就绪")
    
    # 测试 objectness 和 combined 方法
    for method in ['objectness', 'combined']:
        print(f"\n{'='*80}")
        print(f"分析方法: {method.upper()}")
        print(f"{'='*80}")
        
        # 生成热图
        extractor = ImprovedDINOv3Extractor(dinov3_model, device, heatmap_method=method)
        heatmap = extractor.generate_heatmap(image, image_size)
        
        # 1. 使用默认参数分析
        default_dir = output_dir / f"{method}_default"
        print(f"\n使用默认参数:")
        num_boxes, stats = visualize_heatmap_to_boxes_pipeline(
            heatmap, image, default_dir, method,
            percentile=65,
            smoothing_kernel=11,
            min_component_area=4000
        )
        
        # 2. 测试不同参数组合
        print(f"\n{'='*80}")
        print(f"测试不同参数组合")
        print(f"{'='*80}")
        test_different_parameters(heatmap, image, method, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ 诊断完成！")
    print(f"{'='*80}")
    
    print(f"\n📁 输出结构:")
    print(f"   {output_dir}/")
    print(f"   ├── objectness_default/")
    print(f"   │   ├── pipeline_overview.jpg      ← 8步流程对比图")
    print(f"   │   ├── pipeline_analysis.txt      ← 详细统计")
    print(f"   │   ├── 01_original_heatmap.jpg    ← 原始热图")
    print(f"   │   ├── 02_normalized.jpg          ← 归一化")
    print(f"   │   ├── 03_smoothed.jpg            ← 平滑")
    print(f"   │   ├── 04_thresholded.jpg         ← 阈值化")
    print(f"   │   ├── 05_after_close.jpg         ← 闭运算")
    print(f"   │   ├── 06_after_open.jpg          ← 开运算")
    print(f"   │   ├── 07_components.jpg          ← 连通组件")
    print(f"   │   └── 08_final_boxes.jpg         ← 最终boxes")
    print(f"   ├── objectness/")
    print(f"   │   ├── params_宽松_低阈值/")
    print(f"   │   ├── params_默认参数/")
    print(f"   │   ├── params_严格_高阈值/")
    print(f"   │   ├── params_弱平滑/")
    print(f"   │   ├── params_强平滑/")
    print(f"   │   └── params_小面积/")
    print(f"   ├── combined_default/")
    print(f"   └── combined/")
    
    print(f"\n💡 下一步:")
    print(f"   1. 查看 pipeline_overview.jpg 了解转换流程")
    print(f"   2. 阅读 pipeline_analysis.txt 查看详细统计")
    print(f"   3. 对比不同参数组合的效果")
    print(f"   4. 根据效果调整参数")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()