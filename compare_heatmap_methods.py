#!/usr/bin/env python3
"""
对比不同的热图生成方法对SAM2分割效果的影响

测试方法:
1. 物体性评分方法 (Objectness)
2. 组合方法 (Combined: objectness + anomaly + contrast)
3. 当前的Attention方法 (作为baseline)
"""

import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.prompt_generator import PromptConfig, generate_prompts_from_heatmap
from src.data_loader import load_image
from src.utils import to_torch_dtype

print("=" * 80)
print("DINOv3热图方法 + SAM2分割效果对比")
print("=" * 80)


class HeatmapGenerator:
    """不同的热图生成方法"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def extract_features(self, image):
        """提取DINOv3特征"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features_dict = self.model.forward_features(image_tensor)
            patch_features = features_dict['x_norm_patchtokens'][0]  # [N, D]
        
        H = W = int(np.sqrt(len(patch_features)))
        return patch_features, H, W
    
    def method_objectness(self, image):
        """方法1: 物体性评分"""
        patch_features, H, W = self.extract_features(image)
        
        # 归一化特征
        patch_features_norm = F.normalize(patch_features, dim=-1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())
        
        # 物体性 = 1 - 与邻近patches的平均相似度
        K = 20
        topk_sim, _ = torch.topk(similarity_matrix, k=K+1, dim=1, largest=True)
        avg_similarity = topk_sim[:, 1:].mean(dim=1)
        objectness = (1 - avg_similarity).cpu().numpy()
        
        # 重塑为2D
        objectness_2d = objectness[:H*W].reshape(H, W)
        
        return objectness_2d, "Objectness"
    
    def method_combined(self, image):
        """方法2: 组合方法"""
        patch_features, H, W = self.extract_features(image)
        
        # 1. 物体性
        patch_features_norm = F.normalize(patch_features, dim=-1)
        similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())
        K = 20
        topk_sim, _ = torch.topk(similarity_matrix, k=K+1, dim=1, largest=True)
        avg_similarity = topk_sim[:, 1:].mean(dim=1)
        objectness = (1 - avg_similarity).cpu().numpy()
        objectness_2d = objectness[:H*W].reshape(H, W)
        
        # 2. 异常度
        mean_feature = patch_features.mean(dim=0, keepdim=True)
        distances = torch.norm(patch_features - mean_feature, dim=-1).cpu().numpy()
        anomaly_2d = distances[:H*W].reshape(H, W)
        
        # 3. 局部对比度
        feature_map = patch_features.reshape(H, W, -1)
        contrast = torch.zeros(H, W, device=self.device)
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                i_start = max(0, -di)
                i_end = H + min(0, -di)
                j_start = max(0, -dj)
                j_end = W + min(0, -dj)
                
                center = feature_map[i_start:i_end, j_start:j_end, :]
                neighbor = feature_map[i_start+di:i_end+di, j_start+dj:j_end+dj, :]
                
                diff = torch.norm(center - neighbor, dim=-1)
                contrast[i_start:i_end, j_start:j_end] += diff
        
        contrast = contrast.cpu().numpy()
        
        # 归一化并组合
        objectness_norm = (objectness_2d - objectness_2d.min()) / (objectness_2d.max() - objectness_2d.min() + 1e-8)
        anomaly_norm = (anomaly_2d - anomaly_2d.min()) / (anomaly_2d.max() - anomaly_2d.min() + 1e-8)
        contrast_norm = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)
        
        combined = (
            objectness_norm * 0.4 +
            anomaly_norm * 0.3 +
            contrast_norm * 0.3
        )
        
        return combined, "Combined"
    
    def method_attention(self, image, extractor):
        """方法3: 当前的Attention方法 (baseline)"""
        feats = extractor.extract(image)
        attention = feats["attention"]
        
        if attention is None:
            raise ValueError("Failed to extract attention")
        
        # 使用现有的attention_to_heatmap方法
        heatmap = extractor.attention_to_heatmap(
            attention,
            (image.shape[1], image.shape[0])
        )
        
        return heatmap, "Attention (Current)"


def resize_heatmap(heatmap, target_size):
    """调整热图大小到目标尺寸"""
    return cv2.resize(
        heatmap,
        target_size,
        interpolation=cv2.INTER_CUBIC
    )


def save_heatmap_visualization(heatmap, image, name, output_dir):
    """保存热图可视化"""
    # 归一化
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 热图
    heatmap_vis = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{name}_heatmap.png"), heatmap_colored)
    
    # 叠加
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
        heatmap_colored, 0.4, 0
    )
    cv2.imwrite(str(output_dir / f"{name}_overlay.png"), overlay)


def visualize_boxes(image, boxes, name, output_dir):
    """可视化bounding boxes"""
    img_with_boxes = image.copy()
    
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            img_with_boxes, f"{i}", (x0, y0-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    cv2.imwrite(
        str(output_dir / f"{name}_boxes.png"),
        cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    )


def visualize_masks(image, masks, name, output_dir):
    """可视化分割masks"""
    if not masks:
        print(f"  ⚠️ {name}: 没有生成任何mask")
        return
    
    # 单独保存每个mask
    combined = np.zeros_like(image)
    
    for i, mask in enumerate(masks):
        # 为每个mask分配不同的颜色
        color = np.array([
            (i * 50) % 255,
            (i * 80) % 255,
            (i * 120) % 255
        ], dtype=np.uint8)
        
        combined[mask.astype(bool)] = color
        
        # 保存前3个mask
        if i < 3:
            mask_vis = (mask.astype(np.uint8) * 255)
            cv2.imwrite(str(output_dir / f"{name}_mask_{i}.png"), mask_vis)
    
    # 保存组合的可视化
    cv2.imwrite(
        str(output_dir / f"{name}_all_masks.png"),
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    )
    
    # 叠加到原图
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.4, 0
    )
    cv2.imwrite(str(output_dir / f"{name}_overlay.png"), overlay)


def analyze_results(method_name, heatmap, boxes, masks, area_threshold):
    """分析结果"""
    print(f"\n{'='*60}")
    print(f"{method_name}")
    print(f"{'='*60}")
    
    print(f"热图统计:")
    print(f"  范围: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"  均值: {heatmap.mean():.3f}")
    print(f"  标准差: {heatmap.std():.3f}")
    
    print(f"\nPrompts生成:")
    print(f"  Box数量: {len(boxes)}")
    
    if boxes:
        print(f"  前5个boxes:")
        for i, box in enumerate(boxes[:5]):
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0)
            print(f"    Box {i}: [{x0:4d}, {y0:4d}, {x1:4d}, {y1:4d}], area={area:6.0f}")
    
    print(f"\nSAM2分割结果:")
    print(f"  生成的mask数量: {len(masks)}")
    
    if masks:
        valid_count = 0
        total_area = 0
        max_area = 0
        
        for i, mask in enumerate(masks):
            area = mask.sum()
            total_area += area
            max_area = max(max_area, area)
            
            passed = area >= area_threshold
            if passed:
                valid_count += 1
            
            status = "✅" if passed else "❌"
            if i < 10:  # 只显示前10个
                print(f"    Mask {i}: area={area:8.0f} {status}")
        
        print(f"\n  通过阈值的mask: {valid_count}/{len(masks)}")
        print(f"  平均mask面积: {total_area/len(masks):.0f}")
        print(f"  最大mask面积: {max_area:.0f}")
        
        return {
            'total_masks': len(masks),
            'valid_masks': valid_count,
            'avg_area': total_area / len(masks),
            'max_area': max_area
        }
    else:
        return {
            'total_masks': 0,
            'valid_masks': 0,
            'avg_area': 0,
            'max_area': 0
        }


def main():
    # 配置
    image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"
    config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/model_config.yaml"
    prompt_config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/prompt_config.yaml"
    
    output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/heatmap_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载图像
    print(f"\n加载图像: {image_path}")
    sample = load_image(image_path)
    print(f"图像形状: {sample.image.shape}")
    
    # 保存原图
    cv2.imwrite(
        str(output_dir / "original.png"),
        cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
    )
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(prompt_config_path, 'r') as f:
        prompt_cfg_dict = yaml.safe_load(f)
    
    device = torch.device(config["device"])
    dtype = to_torch_dtype(config["dtype"])
    
    # 初始化模型
    print("\n初始化模型...")
    dinov3_cfg = Dinov3Config(**config["dinov3"])
    extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
    print("✅ DINOv3加载成功")
    
    sam2_cfg = Sam2Config(**config["sam2"])
    segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
    print("✅ SAM2加载成功")
    
    # 初始化热图生成器
    heatmap_generator = HeatmapGenerator(extractor.model, device)
    
    # Prompt配置
    prompt_params = {}
    if "attention" in prompt_cfg_dict:
        prompt_params.update(prompt_cfg_dict["attention"])
    if "points" in prompt_cfg_dict:
        prompt_params.update(prompt_cfg_dict["points"])
    prompt_config = PromptConfig(**prompt_params)
    
    area_threshold = config["pipeline"]["area_threshold"]
    
    print(f"\nPrompt配置:")
    print(f"  percentile: {prompt_config.percentile}")
    print(f"  min_component_area: {prompt_config.min_component_area}")
    print(f"  area_threshold: {area_threshold}")
    
    # 测试三种方法
    methods = [
        ("objectness", heatmap_generator.method_objectness, sample.image),
        ("combined", heatmap_generator.method_combined, sample.image),
        ("attention", heatmap_generator.method_attention, sample.image, extractor),
    ]
    
    results_summary = {}
    
    for method_info in methods:
        if len(method_info) == 3:
            method_id, method_func, image = method_info
            heatmap_2d, method_name = method_func(image)
        else:
            method_id, method_func, image, extra_arg = method_info
            heatmap_2d, method_name = method_func(image, extra_arg)
        
        print(f"\n{'='*80}")
        print(f"测试方法: {method_name}")
        print(f"{'='*80}")
        
        # 调整热图大小
        heatmap = resize_heatmap(heatmap_2d, (sample.image.shape[1], sample.image.shape[0]))
        
        # 保存热图可视化
        save_heatmap_visualization(heatmap, sample.image, method_id, output_dir)
        
        # 生成prompts
        boxes, points, labels = generate_prompts_from_heatmap(heatmap, prompt_config)
        
        # 可视化boxes
        if boxes:
            visualize_boxes(sample.image, boxes, method_id, output_dir)
        
        # 运行SAM2
        if boxes:
            masks = segmenter.segment_batched(
                sample.image,
                boxes,
                points=points if points else None,
                labels=labels if labels else None,
                batch_size=32,
            )
            
            # 可视化masks
            visualize_masks(sample.image, masks, method_id, output_dir)
        else:
            masks = []
        
        # 分析结果
        stats = analyze_results(method_name, heatmap, boxes, masks, area_threshold)
        results_summary[method_name] = stats
    
    # 打印总结
    print(f"\n{'='*80}")
    print("总结对比")
    print(f"{'='*80}")
    
    print(f"\n{'方法':<25} {'总Masks':<10} {'有效Masks':<12} {'平均面积':<12} {'最大面积':<12}")
    print("-" * 80)
    
    for method_name, stats in results_summary.items():
        print(f"{method_name:<25} {stats['total_masks']:<10} {stats['valid_masks']:<12} "
              f"{stats['avg_area']:<12.0f} {stats['max_area']:<12.0f}")
    
    # 找出最佳方法
    best_method = max(results_summary.items(), key=lambda x: x[1]['valid_masks'])
    
    print(f"\n{'='*80}")
    print(f"🏆 最佳方法: {best_method[0]}")
    print(f"   有效Masks数量: {best_method[1]['valid_masks']}")
    print(f"{'='*80}")
    
    print(f"\n所有结果保存在: {output_dir}/")
    print("\n关键文件:")
    print("  热图对比:")
    print("    - objectness_overlay.png")
    print("    - combined_overlay.png")
    print("    - attention_overlay.png")
    print("\n  Boxes对比:")
    print("    - objectness_boxes.png")
    print("    - combined_boxes.png")
    print("    - attention_boxes.png")
    print("\n  分割结果对比:")
    print("    - objectness_overlay.png")
    print("    - combined_overlay.png")
    print("    - attention_overlay.png")


if __name__ == "__main__":
    main()