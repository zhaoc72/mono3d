#!/usr/bin/env python3
"""
快速测试脚本 - 对比三种方法的效果

这个脚本会快速展示三种方法的:
1. 热图质量
2. 生成的prompts数量
3. SAM2分割结果
"""

import sys
sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

import yaml
import torch
import numpy as np
import cv2
from pathlib import Path

from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.prompt_generator import PromptConfig, generate_prompts_from_heatmap
from src.data_loader import load_image
from src.utils import to_torch_dtype

# 导入改进的提取器
sys.path.insert(0, '/home/claude')
from improved_dinov3_extractor import ImprovedDINOv3Extractor


def quick_test():
    print("=" * 80)
    print("DINOv3 热图方法快速对比测试")
    print("=" * 80)
    
    # 配置路径
    image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
    config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/model_config.yaml"
    prompt_config_path = "/media/pc/D/zhaochen/mono3d/mono3d/configs/prompt_config.yaml"
    
    output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/quick_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载图像
    print(f"\n📷 加载图像: {Path(image_path).name}")
    sample = load_image(image_path)
    image_size = (sample.image.shape[1], sample.image.shape[0])
    print(f"   大小: {image_size[0]}x{image_size[1]}")
    
    # 保存原图
    cv2.imwrite(
        str(output_dir / "original.jpg"),
        cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
    )
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(prompt_config_path, 'r') as f:
        prompt_cfg_dict = yaml.safe_load(f)
    
    device = torch.device(config["device"])
    dtype = to_torch_dtype(config["dtype"])
    
    # 加载DINOv3模型
    print("\n🔧 加载DINOv3模型...")
    model = torch.hub.load(
        config["dinov3"]["repo_or_dir"],
        config["dinov3"]["model_name"],
        source="local",
        trust_repo=True,
        pretrained=False
    )
    
    state = torch.load(config["dinov3"]["checkpoint_path"], map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    print("   ✅ DINOv3就绪")
    
    # 加载SAM2
    print("\n🔧 加载SAM2模型...")
    sam2_cfg = Sam2Config(**config["sam2"])
    segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
    print("   ✅ SAM2就绪")
    
    # Prompt配置
    prompt_params = {}
    if "attention" in prompt_cfg_dict:
        prompt_params.update(prompt_cfg_dict["attention"])
    if "points" in prompt_cfg_dict:
        prompt_params.update(prompt_cfg_dict["points"])
    prompt_config = PromptConfig(**prompt_params)
    area_threshold = config["pipeline"]["area_threshold"]
    
    # 测试三种方法
    methods = ['objectness', 'combined', 'attention']
    results = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"📊 测试方法: {method.upper()}")
        print(f"{'='*80}")
        
        # 创建提取器
        extractor = ImprovedDINOv3Extractor(model, device, heatmap_method=method)
        
        # 生成热图
        print(f"   生成热图...")
        heatmap = extractor.generate_heatmap(sample.image, image_size)
        
        print(f"   ├─ 范围: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"   ├─ 均值: {heatmap.mean():.3f}")
        print(f"   └─ 标准差: {heatmap.std():.3f}")
        
        # 保存热图
        heatmap_vis = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(
            cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.6,
            heatmap_colored, 0.4, 0
        )
        cv2.imwrite(str(output_dir / f"{method}_heatmap.jpg"), overlay)
        
        # 生成prompts
        print(f"   生成prompts...")
        boxes, points, labels = generate_prompts_from_heatmap(heatmap, prompt_config)
        
        print(f"   └─ 生成{len(boxes)}个boxes")
        
        if not boxes:
            print(f"   ⚠️  没有生成任何box，跳过SAM2")
            results[method] = {'boxes': 0, 'masks': 0, 'valid_masks': 0}
            continue
        
        # 可视化boxes
        img_boxes = sample.image.copy()
        for i, box in enumerate(boxes[:10]):  # 只画前10个
            x0, y0, x1, y1 = box
            cv2.rectangle(img_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imwrite(
            str(output_dir / f"{method}_boxes.jpg"),
            cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR)
        )
        
        # 运行SAM2
        print(f"   运行SAM2分割...")
        masks = segmenter.segment_batched(
            sample.image, boxes,
            points=points if points else None,
            labels=labels if labels else None,
            batch_size=32
        )
        
        # 统计有效masks
        valid_count = sum(1 for m in masks if m.sum() >= area_threshold)
        
        print(f"   ├─ 生成{len(masks)}个masks")
        print(f"   └─ {valid_count}个有效masks (≥{area_threshold}px)")
        
        # 可视化masks
        if masks:
            combined = np.zeros_like(sample.image)
            for i, mask in enumerate(masks):
                if mask.sum() >= area_threshold:  # 只显示有效masks
                    color = np.array([
                        (i * 50) % 255,
                        (i * 80) % 255,
                        (i * 120) % 255
                    ], dtype=np.uint8)
                    combined[mask.astype(bool)] = color
            
            overlay = cv2.addWeighted(
                cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR), 0.6,
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.4, 0
            )
            cv2.imwrite(str(output_dir / f"{method}_masks.jpg"), overlay)
        
        results[method] = {
            'boxes': len(boxes),
            'masks': len(masks),
            'valid_masks': valid_count
        }
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("📈 结果对比")
    print(f"{'='*80}")
    
    print(f"\n{'方法':<15} {'Boxes':<10} {'总Masks':<12} {'有效Masks':<12}")
    print("-" * 50)
    
    for method in methods:
        r = results[method]
        print(f"{method:<15} {r['boxes']:<10} {r['masks']:<12} {r['valid_masks']:<12}")
    
    # 找出最佳方法
    best = max(results.items(), key=lambda x: x[1]['valid_masks'])
    
    print(f"\n{'='*80}")
    print(f"🏆 推荐方法: {best[0].upper()}")
    print(f"   有效Masks: {best[1]['valid_masks']}")
    print(f"{'='*80}")
    
    print(f"\n💾 结果保存在: {output_dir}/")
    print("\n查看文件:")
    for method in methods:
        print(f"   {method}_heatmap.jpg - 热图")
        print(f"   {method}_boxes.jpg - Boxes")
        print(f"   {method}_masks.jpg - 分割结果")
    
    # 给出建议
    print(f"\n{'='*80}")
    print("💡 使用建议")
    print(f"{'='*80}")
    
    if best[0] == 'objectness':
        print("\n物体性方法效果最好！")
        print("在 dinov3_feature.py 中，替换 attention_to_heatmap 方法为物体性方法。")
    elif best[0] == 'combined':
        print("\n组合方法效果最好！")
        print("在 dinov3_feature.py 中，替换 attention_to_heatmap 方法为组合方法。")
    else:
        print("\n当前attention方法已经不错。")
        print("但可以尝试物体性或组合方法看是否有改进。")


if __name__ == "__main__":
    quick_test()