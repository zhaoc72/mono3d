#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 添加这一行

"""
使用官方DINOv3代码的测试脚本
支持ViT-7B/16 + 多GPU模型并行
"""

import argparse
import sys
from pathlib import Path
import json

import cv2
import numpy as np
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.official_pipeline import OfficialDINOv3Pipeline
from src.utils import setup_logging, LOGGER


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_image(image_path: str) -> np.ndarray:
    """加载图像"""
    LOGGER.info(f"📸 Loading image: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    LOGGER.info(f"   Image shape: {rgb.shape}")
    return rgb


def save_detection_visualization(
    detection_result: dict,
    image: np.ndarray,
    output_dir: Path,
):
    """保存检测可视化"""
    LOGGER.info("💾 Saving detection visualization...")
    
    vis = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    # 颜色调色板
    colors = [
        (240, 86, 60), (67, 160, 71), (66, 133, 244),
        (171, 71, 188), (255, 202, 40), (38, 198, 218),
    ]
    
    boxes = detection_result['boxes']
    scores = detection_result['scores']
    labels = detection_result['labels']
    
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[idx % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # 添加标签
        text = f"Class {int(label)}: {score:.2f}"
        cv2.putText(
            vis, text, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    
    output_path = output_dir / "detections.png"
    cv2.imwrite(str(output_path), vis)
    LOGGER.info(f"   ✅ Saved: {output_path}")


def save_segmentation_visualization(
    segmentation_result: dict,
    image: np.ndarray,
    output_dir: Path,
):
    """保存分割可视化"""
    LOGGER.info("💾 Saving segmentation visualization...")
    
    class_map = segmentation_result['class_map']
    
    # 创建随机颜色调色板
    np.random.seed(42)
    num_classes = segmentation_result['num_classes']
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # 背景为黑色
    
    # 生成彩色分割图
    color_map = palette[class_map]
    
    # 调整大小匹配原图
    if color_map.shape[:2] != image.shape[:2]:
        color_map = cv2.resize(
            color_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # 保存分割图
    seg_path = output_dir / "segmentation_map.png"
    cv2.imwrite(str(seg_path), cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
    LOGGER.info(f"   ✅ Saved: {seg_path}")
    
    # 保存叠加图
    base_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    color_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.6, color_bgr, 0.4, 0.0)
    
    overlay_path = output_dir / "segmentation_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    LOGGER.info(f"   ✅ Saved: {overlay_path}")


def save_metadata(
    result: dict,
    image_path: str,
    output_dir: Path,
):
    """保存元数据"""
    metadata = {
        "image": image_path,
        "image_shape": list(result['image_shape']),
        "processed_size": list(result['processed_size']),
        "num_detections": result['detection']['num_detections'],
        "num_classes": result['segmentation']['num_classes'],
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    LOGGER.info(f"💾 Metadata saved")


def main():
    parser = argparse.ArgumentParser(
        description="使用官方DINOv3代码进行零样本实例分割"
    )
    parser.add_argument(
        "--image",
        default="/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg",
        help="输入图像路径"
    )
    parser.add_argument(
        "--config",
        default="configs/official_model_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output",
        default="outputs/official_test",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    LOGGER.info("=" * 70)
    LOGGER.info("🚀 Official DINOv3 Zero-Shot Instance Segmentation")
    LOGGER.info("=" * 70)
    
    # 加载配置
    config_path = PROJECT_ROOT / args.config
    LOGGER.info(f"\n📖 Loading config: {config_path}")
    config = load_config(str(config_path))
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"📁 Output directory: {output_dir}")
    
    # 加载图像
    image = load_image(args.image)
    
    # 初始化管道
    LOGGER.info("\n🔧 Initializing pipeline...")
    pipeline = OfficialDINOv3Pipeline(config, device=args.device)
    
    # 运行推理
    result = pipeline.run(image)
    
    # 保存结果
    LOGGER.info("\n💾 Saving results...")
    save_detection_visualization(result['detection'], image, output_dir)
    save_segmentation_visualization(result['segmentation'], image, output_dir)
    save_metadata(result, args.image, output_dir)
    
    # 打印总结
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("✅ Test Complete!")
    LOGGER.info("=" * 70)
    LOGGER.info(f"\n📊 Results Summary:")
    LOGGER.info(f"   - Detections: {result['detection']['num_detections']}")
    LOGGER.info(f"   - Segmentation classes: {result['segmentation']['num_classes']}")
    LOGGER.info(f"   - Output directory: {output_dir}")
    LOGGER.info(f"\n📄 Generated files:")
    LOGGER.info(f"   - detections.png")
    LOGGER.info(f"   - segmentation_map.png")
    LOGGER.info(f"   - segmentation_overlay.png")
    LOGGER.info(f"   - metadata.json")


if __name__ == "__main__":
    main()