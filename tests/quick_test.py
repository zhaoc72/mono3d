#!/usr/bin/env python3
"""
独立的可视化对比脚本
不依赖项目内部模块，可以直接运行

对比三种热图生成方法：
1. objectness (物体性)
2. combined (组合)  
3. attention (注意力)
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import ndimage
from sklearn.cluster import KMeans
import torch.nn.functional as F

# 添加改进提取器路径
sys.path.insert(0, '/home/claude')
from improved_dinov3_extractor import ImprovedDINOv3Extractor


@dataclass
class BoundingBox:
    """边界框"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]
    
    @property
    def area(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    @property
    def centroid(self) -> Tuple[int, int]:
        cx = (self.x_min + self.x_max) // 2
        cy = (self.y_min + self.y_max) // 2
        return (cx, cy)


def heatmap_to_boxes(
    heatmap: np.ndarray,
    percentile: float = 65.0,
    smoothing_kernel: int = 10,
    min_component_area: int = 4000
) -> List[BoundingBox]:
    """从热图生成候选框
    
    Args:
        heatmap: 热图 (H, W)
        percentile: 阈值百分位
        smoothing_kernel: 平滑核大小
        min_component_area: 最小连通区域面积
    
    Returns:
        候选框列表
    """
    # 1. 归一化
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 2. 平滑
    if smoothing_kernel > 0:
        # 确保核大小是奇数
        kernel_size = smoothing_kernel if smoothing_kernel % 2 == 1 else smoothing_kernel + 1
        heatmap_norm = cv2.GaussianBlur(
            heatmap_norm, 
            (kernel_size, kernel_size), 
            0
        )
    
    # 3. 阈值化
    threshold = np.percentile(heatmap_norm, percentile)
    binary_mask = (heatmap_norm > threshold).astype(np.uint8)
    
    # 4. 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 5. 连通组件分析
    labeled_array, num_features = ndimage.label(binary_mask)
    
    boxes = []
    for label_id in range(1, num_features + 1):
        component_mask = (labeled_array == label_id)
        area = component_mask.sum()
        
        if area < min_component_area:
            continue
        
        # 获取边界框
        coords = np.where(component_mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        boxes.append(BoundingBox(
            x_min=int(x_min),
            y_min=int(y_min),
            x_max=int(x_max),
            y_max=int(y_max)
        ))
    
    return boxes


def load_sam2_model(device: torch.device):
    """加载 SAM2 模型"""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        checkpoint = "/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        return predictor
    except Exception as e:
        print(f"⚠️  SAM2 加载失败: {e}")
        print(f"   将使用模拟分割结果")
        return None


def segment_with_sam2(
    predictor,
    image: np.ndarray,
    boxes: List[BoundingBox],
    device: torch.device
) -> List[np.ndarray]:
    """使用 SAM2 进行分割"""
    if predictor is None:
        # 模拟分割结果
        print("   使用模拟分割...")
        masks = []
        for box in boxes:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[box.y_min:box.y_max, box.x_min:box.x_max] = 1
            masks.append(mask)
        return masks
    
    predictor.set_image(image)
    
    masks = []
    for box in boxes:
        box_array = np.array([box.to_list()])
        
        mask, score, logit = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_array,
            multimask_output=False
        )
        
        masks.append(mask[0].astype(np.uint8))
    
    return masks


def save_heatmap(heatmap, image, output_dir):
    """保存热图可视化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 归一化
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 1. 纯热图
    heatmap_vis = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / "01_heatmap.jpg"), heatmap_colored)
    
    # 2. 热图叠加
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
        heatmap_colored, 0.4, 0
    )
    cv2.imwrite(str(output_dir / "02_heatmap_overlay.jpg"), overlay)
    
    # 3. 统计
    with open(output_dir / "heatmap_stats.txt", 'w') as f:
        f.write("=== 热图统计 ===\n")
        f.write(f"最小值: {heatmap.min():.4f}\n")
        f.write(f"最大值: {heatmap.max():.4f}\n")
        f.write(f"均值:   {heatmap.mean():.4f}\n")
        f.write(f"标准差: {heatmap.std():.4f}\n")
        f.write(f"中位数: {np.median(heatmap):.4f}\n")
    
    return heatmap_colored, {
        'min': float(heatmap.min()),
        'max': float(heatmap.max()),
        'mean': float(heatmap.mean()),
        'std': float(heatmap.std())
    }


def save_boxes(image, boxes, output_dir):
    """保存候选框可视化"""
    output_dir = Path(output_dir)
    
    if not boxes:
        img_blank = image.copy()
        cv2.putText(img_blank, "No boxes", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / "03_boxes.jpg"),
                   cv2.cvtColor(img_blank, cv2.COLOR_RGB2BGR))
        return
    
    # 简洁版
    img_simple = image.copy()
    for box in boxes:
        cv2.rectangle(img_simple, 
                     (box.x_min, box.y_min), 
                     (box.x_max, box.y_max), 
                     (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "03_boxes.jpg"),
               cv2.cvtColor(img_simple, cv2.COLOR_RGB2BGR))
    
    # 详细版
    img_detail = image.copy()
    for i, box in enumerate(boxes):
        cv2.rectangle(img_detail,
                     (box.x_min, box.y_min),
                     (box.x_max, box.y_max),
                     (0, 255, 0), 2)
        cv2.putText(img_detail, f"#{i}", 
                   (box.x_min, box.y_min-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_detail, f"{box.area:.0f}px",
                   (box.x_min, box.y_max+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    cv2.imwrite(str(output_dir / "04_boxes_detail.jpg"),
               cv2.cvtColor(img_detail, cv2.COLOR_RGB2BGR))
    
    # 统计
    with open(output_dir / "boxes_stats.txt", 'w') as f:
        f.write("=== 候选框统计 ===\n")
        f.write(f"总数: {len(boxes)}\n\n")
        f.write("详细信息:\n")
        for i, box in enumerate(boxes):
            f.write(f"Box {i}: [{box.x_min:4d}, {box.y_min:4d}, "
                   f"{box.x_max:4d}, {box.y_max:4d}], area={box.area:7.0f}\n")


def save_masks(image, masks, area_threshold, output_dir):
    """保存分割掩码可视化"""
    output_dir = Path(output_dir)
    
    if not masks:
        img_blank = image.copy()
        cv2.putText(img_blank, "No masks", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / "05_masks.jpg"),
                   cv2.cvtColor(img_blank, cv2.COLOR_RGB2BGR))
        return []
    
    # 彩色组合
    combined = np.zeros_like(image)
    valid_masks = []
    
    for i, mask in enumerate(masks):
        area = mask.sum()
        if area >= area_threshold:
            valid_masks.append((i, mask, area))
            color = np.array([
                (i * 50) % 255,
                (i * 80 + 60) % 255,
                (i * 120 + 30) % 255
            ], dtype=np.uint8)
            combined[mask.astype(bool)] = color
    
    cv2.imwrite(str(output_dir / "05_masks.jpg"),
               cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    # 叠加
    if len(valid_masks) > 0:
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
        )
        cv2.imwrite(str(output_dir / "06_masks_overlay.jpg"), overlay)
    
    # 单独掩码
    individual_dir = output_dir / "07_individual_masks"
    individual_dir.mkdir(exist_ok=True)
    for idx, (i, mask, area) in enumerate(valid_masks[:5]):
        mask_vis = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(individual_dir / f"mask_{i:02d}_area_{area:.0f}.jpg"), 
                   mask_vis)
    
    # 统计
    with open(output_dir / "masks_stats.txt", 'w') as f:
        f.write("=== 分割掩码统计 ===\n")
        f.write(f"总掩码数: {len(masks)}\n")
        f.write(f"有效掩码数: {len(valid_masks)}\n")
        if len(masks) > 0:
            f.write(f"有效率: {len(valid_masks)/len(masks)*100:.1f}%\n")
        f.write(f"面积阈值: {area_threshold}\n\n")
        f.write("所有掩码详情:\n")
        for i, mask in enumerate(masks):
            area = mask.sum()
            passed = "✅" if area >= area_threshold else "❌"
            ratio = area / (image.shape[0] * image.shape[1]) * 100
            f.write(f"Mask {i}: area={area:8.0f}, ratio={ratio:5.2f}%, {passed}\n")
    
    return valid_masks


def create_comparison_grid(methods_data, output_path):
    """创建对比网格图"""
    print(f"\n创建对比网格图...")
    
    cell_h, cell_w = 300, 400
    margin = 10
    label_h = 40
    
    method_labels = {
        'objectness': '物体性 (推荐)',
        'combined': '组合 (推荐)',
        'attention': 'Attention (当前)'
    }
    
    col_labels = ['原图', '热图', 'Boxes', '掩码']
    
    n_methods = len(methods_data)
    n_cols = len(col_labels)
    
    canvas_h = label_h + n_methods * (cell_h + margin) + margin
    canvas_w = margin + n_cols * (cell_w + margin)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 列标题
    for col_idx, label in enumerate(col_labels):
        x = margin + col_idx * (cell_w + margin) + cell_w // 2 - 30
        cv2.putText(canvas, label, (x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 每个方法的结果
    for method_idx, (method_name, data) in enumerate(methods_data.items()):
        y_offset = label_h + method_idx * (cell_h + margin)
        
        images = [
            data.get('image'),
            data.get('heatmap'),
            data.get('boxes'),
            data.get('masks')
        ]
        
        for col_idx, img in enumerate(images):
            if img is None:
                continue
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            img_resized = cv2.resize(img, (cell_w, cell_h))
            
            x_offset = margin + col_idx * (cell_w + margin)
            canvas[y_offset:y_offset+cell_h, x_offset:x_offset+cell_w] = img_resized
        
        # 方法标签
        label_text = method_labels.get(method_name, method_name)
        cv2.putText(canvas, label_text, (5, y_offset + cell_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(str(output_path), canvas)
    print(f"✅ 对比网格图: {output_path}")


def create_summary_report(results, output_path):
    """创建文字对比报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DINOv3 热图方法对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'方法':<20} {'Boxes':<10} {'总Masks':<12} {'有效Masks':<12} {'有效率':<10}\n")
        f.write("-" * 80 + "\n")
        
        for method, stats in results.items():
            valid_rate = (stats['valid_masks'] / stats['total_masks'] * 100) if stats['total_masks'] > 0 else 0
            f.write(f"{method:<20} {stats['boxes']:<10} {stats['total_masks']:<12} "
                   f"{stats['valid_masks']:<12} {valid_rate:>8.1f}%\n")
        
        best = max(results.items(), key=lambda x: x[1]['valid_masks'])
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"🏆 推荐方法: {best[0].upper()}\n")
        f.write(f"   有效掩码数: {best[1]['valid_masks']}\n")
        f.write("=" * 80 + "\n")


def main():
    print("=" * 80)
    print("DINOv3 热图方法可视化对比测试")
    print("=" * 80)
    
    # 配置
    image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
    output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/quick_comparison_visual")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    area_threshold = 300
    
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
    
    cv2.imwrite(str(output_dir / "00_original.jpg"),
               cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # 加载模型
    print(f"\n🔧 加载模型...")
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
    print(f"   ✅ DINOv3")
    
    sam2_predictor = load_sam2_model(device)
    if sam2_predictor:
        print(f"   ✅ SAM2")
    
    # 测试三种方法
    methods = ['objectness', 'combined', 'attention']
    results = {}
    methods_data = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"📊 测试方法: {method.upper()}")
        print(f"{'='*80}")
        
        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True)
        
        # 创建提取器
        extractor = ImprovedDINOv3Extractor(dinov3_model, device, heatmap_method=method)
        
        # 生成热图
        print(f"   生成热图...")
        heatmap = extractor.generate_heatmap(image, image_size)
        print(f"   ├─ 范围: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"   ├─ 均值: {heatmap.mean():.3f}")
        print(f"   └─ 标准差: {heatmap.std():.3f}")
        
        heatmap_colored, heatmap_stats = save_heatmap(heatmap, image, method_dir)
        
        # 生成候选框
        print(f"   生成候选框...")
        boxes = heatmap_to_boxes(heatmap, percentile=95, smoothing_kernel=11, min_component_area=4000)
        print(f"   └─ {len(boxes)}个boxes")
        
        save_boxes(image, boxes, method_dir)
        
        if not boxes:
            print(f"   ⚠️  跳过SAM2")
            results[method] = {
                'boxes': 0, 'total_masks': 0, 'valid_masks': 0,
                'avg_area': 0, 'max_area': 0,
                'heatmap_min': heatmap_stats['min'],
                'heatmap_max': heatmap_stats['max'],
                'heatmap_mean': heatmap_stats['mean'],
                'heatmap_std': heatmap_stats['std']
            }
            methods_data[method] = {
                'image': cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                'heatmap': heatmap_colored,
                'boxes': cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                'masks': cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            }
            continue
        
        # SAM2分割
        print(f"   运行SAM2...")
        masks = segment_with_sam2(sam2_predictor, image, boxes, device)
        print(f"   └─ {len(masks)}个masks")
        
        valid_masks = save_masks(image, masks, area_threshold, method_dir)
        print(f"   ✅ {len(valid_masks)}个有效masks")
        
        # 统计
        total_area = sum(m[2] for m in valid_masks) if valid_masks else 0
        max_area = max((m[2] for m in valid_masks), default=0)
        avg_area = total_area / len(valid_masks) if valid_masks else 0
        
        results[method] = {
            'boxes': len(boxes),
            'total_masks': len(masks),
            'valid_masks': len(valid_masks),
            'avg_area': avg_area,
            'max_area': max_area,
            'heatmap_min': heatmap_stats['min'],
            'heatmap_max': heatmap_stats['max'],
            'heatmap_mean': heatmap_stats['mean'],
            'heatmap_std': heatmap_stats['std']
        }
        
        # 准备对比数据
        boxes_img = cv2.imread(str(method_dir / "04_boxes_detail.jpg"))
        masks_img = cv2.imread(str(method_dir / "06_masks_overlay.jpg"))
        
        methods_data[method] = {
            'image': cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            'heatmap': heatmap_colored,
            'boxes': boxes_img if boxes_img is not None else cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            'masks': masks_img if masks_img is not None else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        }
    
    # 生成对比
    print(f"\n{'='*80}")
    print("📊 生成对比报告")
    print(f"{'='*80}")
    
    create_comparison_grid(methods_data, output_dir / "comparison_grid.jpg")
    create_summary_report(results, output_dir / "comparison_summary.txt")
    
    # 打印总结
    print(f"\n{'='*80}")
    print("📈 结果对比")
    print(f"{'='*80}\n")
    
    print(f"{'方法':<15} {'Boxes':<10} {'总Masks':<12} {'有效Masks':<12} {'有效率':<10}")
    print("-" * 65)
    
    for method in methods:
        r = results[method]
        valid_rate = (r['valid_masks'] / r['total_masks'] * 100) if r['total_masks'] > 0 else 0
        print(f"{method:<15} {r['boxes']:<10} {r['total_masks']:<12} "
              f"{r['valid_masks']:<12} {valid_rate:>8.1f}%")
    
    best = max(results.items(), key=lambda x: x[1]['valid_masks'])
    
    print(f"\n{'='*80}")
    print(f"🏆 推荐方法: {best[0].upper()}")
    print(f"   有效Masks: {best[1]['valid_masks']}")
    print(f"{'='*80}")
    
    print(f"\n💾 输出位置: {output_dir}/")
    print(f"   ├── comparison_grid.jpg")
    print(f"   ├── comparison_summary.txt")
    print(f"   ├── objectness/")
    print(f"   ├── combined/")
    print(f"   └── attention/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()