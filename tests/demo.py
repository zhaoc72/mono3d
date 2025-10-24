#!/usr/bin/env python3
"""
交互式演示 - 可视化不同配置的效果
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.inference_pipeline import ZeroShotSegmentationPipeline, PipelineConfig
from src.prompt_generator import ClusterConfig, PromptConfig
from src.superpixel_helper import SuperpixelConfig, visualize_superpixels
from src.data_loader import load_image
from src.utils import to_torch_dtype, ensure_directory


def create_comparison_grid(results: dict, output_path: Path):
    """创建对比网格图"""
    
    # 提取图像
    images = {}
    for name, data in results.items():
        images[name] = data['overlay']
    
    if not images:
        return
    
    # 网格参数
    n_configs = len(images)
    cell_h, cell_w = 400, 400
    margin = 20
    label_h = 60
    
    # 计算布局
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    canvas_h = label_h + n_rows * (cell_h + margin) + margin
    canvas_w = margin + n_cols * (cell_w + margin)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 标题
    title = "Zero-Shot Instance Segmentation Comparison"
    cv2.putText(
        canvas, title,
        (canvas_w // 2 - 300, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
    )
    
    # 放置图像
    for idx, (name, img) in enumerate(images.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        y_offset = label_h + row * (cell_h + margin) + margin
        x_offset = margin + col * (cell_w + margin)
        
        # 调整大小
        img_resized = cv2.resize(img, (cell_w, cell_h))
        
        # 放置
        canvas[y_offset:y_offset+cell_h, x_offset:x_offset+cell_w] = img_resized
        
        # 标签
        label_y = y_offset - 10
        cv2.putText(
            canvas, name,
            (x_offset, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        
        # 统计
        info = f"Masks: {results[name]['num_masks']}"
        cv2.putText(
            canvas, info,
            (x_offset, y_offset + cell_h + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1
        )
    
    # 保存
    cv2.imwrite(str(output_path), canvas)
    print(f"✅ 对比图已保存: {output_path}")


def run_demo():
    """运行交互式演示"""
    
    print("=" * 80)
    print("零样本实例分割 - 交互式演示")
    print("=" * 80)
    
    # 加载图像
    image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
    sample = load_image(image_path)
    image = sample.image
    
    print(f"\n图像: {image.shape}")
    
    output_dir = Path("/media/pc/D/zhaochen/mono3d/mono3d/outputs/interactive_demo")
    ensure_directory(output_dir)
    
    # 保存原图
    cv2.imwrite(
        str(output_dir / "original.jpg"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"设备: {device}")
    
    # 定义配置
    configs = {
        "Basic": PipelineConfig(
            cluster=ClusterConfig(num_clusters=6, min_region_area=400),
            prompt=PromptConfig()
        ),
        "With Objectness": PipelineConfig(
            cluster=ClusterConfig(
                num_clusters=6,
                min_region_area=400,
                use_objectness_filter=True,
                objectness_threshold=0.3
            ),
            prompt=PromptConfig()
        ),
        "With Superpixels": PipelineConfig(
            cluster=ClusterConfig(num_clusters=6, min_region_area=400),
            prompt=PromptConfig(),
            use_superpixels=True,
            superpixel=SuperpixelConfig(method="slic", n_segments=1000)
        ),
    }
    
    # 共享模型配置
    dinov3_cfg = Dinov3Config(
        repo_or_dir="/media/pc/D/zhaochen/mono3d/dinov3",
        model_name="dinov3_vith16plus",
        checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vith16plus.pth",
        use_torch_hub=True,
        torchhub_source="local",
        enable_objectness=True
    )
    
    sam2_cfg = Sam2Config(
        backend="official",
        checkpoint_path="/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt",
        model_config="sam2.1/sam2.1_hiera_l"
    )
    
    # 运行所有配置
    results = {}
    
    for config_name, pipeline_cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"运行: {config_name}")
        print(f"{'='*60}")
        
        try:
            # 初始化
            pipeline = ZeroShotSegmentationPipeline(
                dinov3_cfg,
                sam2_cfg,
                pipeline_cfg,
                device=str(device),
                dtype=dtype
            )
            
            # 推理
            result = pipeline.run(image)
            
            print(f"  候选: {len(result.proposals)}")
            print(f"  掩码: {len(result.masks)}")
            
            # 可视化
            if result.masks:
                combined = np.zeros_like(image)
                for i, mask in enumerate(result.masks):
                    color = np.array([
                        (i * 50) % 255,
                        (i * 80 + 60) % 255,
                        (i * 120 + 30) % 255
                    ], dtype=np.uint8)
                    combined[mask.astype(bool)] = color
                
                overlay = cv2.addWeighted(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
                    cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
                )
                
                # 保存
                config_dir = output_dir / config_name.lower().replace(" ", "_")
                ensure_directory(config_dir)
                
                cv2.imwrite(str(config_dir / "overlay.jpg"), overlay)
                
                # 保存对象性图
                if result.objectness_map is not None:
                    obj_vis = (result.objectness_map * 255).astype(np.uint8)
                    obj_colored = cv2.applyColorMap(obj_vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(config_dir / "objectness.jpg"), obj_colored)
                
                # 保存超像素
                if result.superpixel_labels is not None:
                    visualize_superpixels(
                        image,
                        result.superpixel_labels,
                        str(config_dir / "superpixels.jpg")
                    )
                
                results[config_name] = {
                    'overlay': overlay,
                    'num_masks': len(result.masks),
                    'num_proposals': len(result.proposals)
                }
                
                print(f"  ✅ 结果已保存: {config_dir}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 创建对比图
    if results:
        print(f"\n{'='*60}")
        print("创建对比网格图...")
        create_comparison_grid(results, output_dir / "comparison.jpg")
    
    # 总结
    print(f"\n{'='*80}")
    print("演示完成")
    print(f"{'='*80}")
    
    print(f"\n📁 输出目录: {output_dir}/")
    print("\n生成的文件:")
    print("  - comparison.jpg  : 对比网格图")
    print("  - original.jpg    : 原始图像")
    
    for name in results.keys():
        dir_name = name.lower().replace(" ", "_")
        print(f"  - {dir_name}/")
        print(f"    ├── overlay.jpg      : 分割叠加")
        print(f"    ├── objectness.jpg   : 对象性图（如有）")
        print(f"    └── superpixels.jpg  : 超像素（如有）")
    
    print(f"\n💡 提示:")
    print("  1. 查看 comparison.jpg 快速对比不同方法")
    print("  2. 查看各方法的详细输出了解差异")
    print("  3. 根据需求选择合适的配置")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_demo())
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)