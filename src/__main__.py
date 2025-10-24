#!/usr/bin/env python3
"""Command-line interface for the zero-shot segmentation pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .data_loader import load_image, load_video_frames, stream_directory_images
from .datasets.vkitti import VkittiFilter, iter_vkitti_frames
from .dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from .inference_pipeline import PipelineConfig, ZeroShotSegmentationPipeline
from .prompt_generator import ClusterConfig, PromptConfig
from .sam2_segmenter import SAM2Segmenter, Sam2Config
from .superpixel_helper import SuperpixelConfig
from .graph_clustering import GraphClusterConfig
from .density_clustering import DensityClusterConfig
from .crf_refinement import CRFConfig
from .utils import ensure_directory, load_yaml, save_mask, setup_logging, to_torch_dtype, LOGGER


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-shot instance segmentation using DINOv3 + SAM2"
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Inference mode")
    
    # Image mode
    image_parser = subparsers.add_parser("image", help="Process a single image")
    image_parser.add_argument("--input", required=True, help="Path to input image")
    image_parser.add_argument("--output", required=True, help="Output directory")
    image_parser.add_argument("--config", required=True, help="Model config YAML")
    image_parser.add_argument("--prompt-config", help="Prompt config YAML")
    image_parser.add_argument("--visualize-intermediate", action="store_true", 
                             help="Save intermediate visualization results")
    
    # Directory mode
    dir_parser = subparsers.add_parser("directory", help="Process a directory of images")
    dir_parser.add_argument("--input", required=True, help="Input directory")
    dir_parser.add_argument("--output", required=True, help="Output directory")
    dir_parser.add_argument("--config", required=True, help="Model config YAML")
    dir_parser.add_argument("--prompt-config", help="Prompt config YAML")
    
    # Video mode
    video_parser = subparsers.add_parser("video", help="Process a video file")
    video_parser.add_argument("--input", required=True, help="Path to video file")
    video_parser.add_argument("--output", required=True, help="Output directory")
    video_parser.add_argument("--config", required=True, help="Model config YAML")
    video_parser.add_argument("--prompt-config", help="Prompt config YAML")
    video_parser.add_argument("--frame-skip", type=int, default=1, help="Frame sampling interval")
    
    # VKITTI mode
    vkitti_parser = subparsers.add_parser("vkitti", help="Process Virtual KITTI 2 dataset")
    vkitti_parser.add_argument("--input", required=True, help="VKITTI2 root directory")
    vkitti_parser.add_argument("--output", required=True, help="Output directory")
    vkitti_parser.add_argument("--config", required=True, help="Model config YAML")
    vkitti_parser.add_argument("--prompt-config", help="Prompt config YAML")
    vkitti_parser.add_argument("--vkitti-scenes", nargs="+", help="Scene names to process")
    vkitti_parser.add_argument("--vkitti-clones", nargs="+", help="Clone names to process")
    vkitti_parser.add_argument("--vkitti-camera", default="Camera_0", help="Camera name")
    vkitti_parser.add_argument("--vkitti-limit", type=int, help="Max frames to process")
    
    return parser.parse_args()


def load_configs(config_path: str, prompt_config_path: Optional[str] = None) -> dict:
    """Load configuration files."""
    config = load_yaml(config_path)
    
    # prompt_config.yaml 中的配置不直接用于初始化
    # 它们是运行时参数，在这里只是加载并存储到 config 中
    if prompt_config_path:
        prompt_config = load_yaml(prompt_config_path)
        # 将 prompt_config 存储为单独的键，不合并到 prompt 中
        config["prompt_config_yaml"] = prompt_config
    
    return config


def build_pipeline(config: dict) -> ZeroShotSegmentationPipeline:
    """Build the segmentation pipeline from configuration."""
    device = config.get("device", "cuda")
    dtype = to_torch_dtype(config.get("dtype", "float32"))
    
    LOGGER.info(f"Using device: {device}, dtype: {dtype}")
    
    # DINOv3 config
    LOGGER.info("Initializing DINOv3...")
    dinov3_cfg = Dinov3Config(**config["dinov3"])
    
    # SAM2 config
    LOGGER.info("Initializing SAM2...")
    sam2_cfg = Sam2Config(**config["sam2"])
    
    # Pipeline config
    pipeline_dict = config.get("pipeline", {})
    
    cluster_cfg = ClusterConfig(**pipeline_dict.get("cluster", {}))
    
    # PromptConfig - 只使用它支持的参数
    prompt_dict = pipeline_dict.get("prompt", {})
    prompt_cfg = PromptConfig(
        include_boxes=prompt_dict.get("include_boxes", True),
        include_points=prompt_dict.get("include_points", True),
        point_strategy=prompt_dict.get("point_strategy", "density"),
        max_points_per_region=prompt_dict.get("max_points_per_region", 5),
        density_noise_handling=prompt_dict.get("density_noise_handling", "nearest"),
    )
    
    superpixel_cfg = SuperpixelConfig(**pipeline_dict.get("superpixel", {}))
    graph_cluster_cfg = GraphClusterConfig(**pipeline_dict.get("graph_cluster", {}))
    density_cluster_cfg = DensityClusterConfig(**pipeline_dict.get("density_cluster", {}))
    crf_cfg = CRFConfig(**pipeline_dict.get("crf", {}))
    
    pipeline_cfg = PipelineConfig(
        cluster=cluster_cfg,
        prompt=prompt_cfg,
        use_superpixels=pipeline_dict.get("use_superpixels", False),
        superpixel=superpixel_cfg,
        use_graph_clustering=pipeline_dict.get("use_graph_clustering", False),
        graph_cluster=graph_cluster_cfg,
        use_density_clustering=pipeline_dict.get("use_density_clustering", False),
        density_cluster=density_cluster_cfg,
        crf=crf_cfg,
    )
    
    # Initialize models
    LOGGER.info("Loading models...")
    extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
    LOGGER.info("✓ DINOv3 loaded")
    
    segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
    LOGGER.info("✓ SAM2 loaded")
    
    return ZeroShotSegmentationPipeline(
        dinov3_cfg,
        sam2_cfg,
        pipeline_cfg,
        device=device,
        dtype=dtype,
        extractor=extractor,
        segmenter=segmenter,
    )


def visualize_intermediate_results(
    image: np.ndarray,
    result,
    output_dir: Path,
    stem: str
):
    """可视化中间过程结果"""
    
    viz_dir = output_dir / "intermediate" / stem
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Saving intermediate visualizations to {viz_dir}")
    
    # 1. 保存原图
    cv2.imwrite(
        str(viz_dir / "01_original.jpg"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    
    # 2. 可视化 Attention Map (如果有)
    if result.attention_map is not None:
        attention_norm = (result.attention_map * 255).astype(np.uint8)
        attention_colored = cv2.applyColorMap(attention_norm, cv2.COLORMAP_JET)
        
        cv2.imwrite(str(viz_dir / "02_attention_map.jpg"), attention_colored)
        
        # 叠加到原图
        attention_resized = cv2.resize(attention_colored, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
            attention_resized, 0.4, 0
        )
        cv2.imwrite(str(viz_dir / "02_attention_overlay.jpg"), overlay)
    
    # 3. 可视化 Objectness Map (如果有)
    if result.objectness_map is not None:
        objectness_norm = (result.objectness_map * 255).astype(np.uint8)
        objectness_colored = cv2.applyColorMap(objectness_norm, cv2.COLORMAP_JET)
        
        cv2.imwrite(str(viz_dir / "03_objectness_map.jpg"), objectness_colored)
        
        # 叠加到原图
        objectness_resized = cv2.resize(objectness_colored, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
            objectness_resized, 0.4, 0
        )
        cv2.imwrite(str(viz_dir / "03_objectness_overlay.jpg"), overlay)
    
    # 4. 可视化聚类 Label Map
    if result.label_map is not None:
        # 为每个标签分配颜色
        unique_labels = np.unique(result.label_map)
        colored_labels = np.zeros((*result.label_map.shape, 3), dtype=np.uint8)
        
        for label in unique_labels:
            color = np.array([
                (int(label) * 50) % 255,
                (int(label) * 80 + 60) % 255,
                (int(label) * 120 + 30) % 255
            ], dtype=np.uint8)
            colored_labels[result.label_map == label] = color
        
        # 调整大小
        colored_labels_resized = cv2.resize(
            colored_labels,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        cv2.imwrite(
            str(viz_dir / "04_cluster_labels.jpg"),
            cv2.cvtColor(colored_labels_resized, cv2.COLOR_RGB2BGR)
        )
        
        # 叠加
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
            cv2.cvtColor(colored_labels_resized, cv2.COLOR_RGB2BGR), 0.5, 0
        )
        cv2.imwrite(str(viz_dir / "04_cluster_overlay.jpg"), overlay)
    
    # 5. 可视化 Proposals (Bounding Boxes)
    if result.proposals:
        img_with_boxes = image.copy()
        
        for i, proposal in enumerate(result.proposals):
            x0, y0, x1, y1 = proposal.bbox
            
            # 根据 objectness 分配颜色（红色=低，绿色=高）
            obj_score = proposal.objectness
            color_r = int(255 * (1 - obj_score))
            color_g = int(255 * obj_score)
            color = (color_g, color_r, 0)
            
            cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), color, 2)
            
            # 显示编号和 objectness
            label = f"#{i} obj:{obj_score:.2f}"
            cv2.putText(
                img_with_boxes, label, (x0, y0-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        cv2.imwrite(
            str(viz_dir / "05_proposals_boxes.jpg"),
            cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        )
    
    # 6. 可视化 Prompts (Boxes + Points)
    if result.prompts:
        img_with_prompts = image.copy()
        
        boxes = result.prompts.get('boxes', [])
        points = result.prompts.get('points', [])
        labels = result.prompts.get('labels', [])
        
        # 画 boxes
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            cv2.rectangle(img_with_prompts, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                img_with_prompts, f"#{i}", (x0, y0-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        # 画 points
        for i, (prompt_points, prompt_labels) in enumerate(zip(points, labels)):
            for point, label in zip(prompt_points, prompt_labels):
                x, y = point
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(img_with_prompts, (x, y), 5, color, -1)
        
        cv2.imwrite(
            str(viz_dir / "06_prompts.jpg"),
            cv2.cvtColor(img_with_prompts, cv2.COLOR_RGB2BGR)
        )
    
    # 7. 可视化最终 Masks (分开显示)
    if result.masks:
        # 彩色组合
        combined = np.zeros_like(image)
        
        for i, mask in enumerate(result.masks):
            color = np.array([
                (i * 50) % 255,
                (i * 80 + 60) % 255,
                (i * 120 + 30) % 255
            ], dtype=np.uint8)
            combined[mask.astype(bool)] = color
        
        cv2.imwrite(
            str(viz_dir / "07_final_masks.jpg"),
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        )
        
        # 叠加
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
        )
        cv2.imwrite(str(viz_dir / "07_final_overlay.jpg"), overlay)
        
        # 单独保存前5个mask
        individual_dir = viz_dir / "individual_masks"
        individual_dir.mkdir(exist_ok=True)
        
        for i, mask in enumerate(result.masks[:5]):
            mask_vis = (mask.astype(np.uint8) * 255)
            cv2.imwrite(str(individual_dir / f"mask_{i:02d}.jpg"), mask_vis)
    
    # 8. 创建流程总结
    summary_text = f"""
=== 中间过程可视化总结 ===

1. 原图: 01_original.jpg

2. DINOv3 特征:
   - Attention Map: 02_attention_map.jpg
   - Attention Overlay: 02_attention_overlay.jpg
   - Objectness Map: 03_objectness_map.jpg
   - Objectness Overlay: 03_objectness_overlay.jpg

3. 聚类结果:
   - Cluster Labels: 04_cluster_labels.jpg
   - Cluster Overlay: 04_cluster_overlay.jpg
   - 聚类数量: {len(np.unique(result.label_map)) if result.label_map is not None else 0}

4. 候选区域:
   - Proposals: 05_proposals_boxes.jpg
   - 候选数量: {len(result.proposals)}
   - Objectness 范围: [{min(p.objectness for p in result.proposals):.3f}, {max(p.objectness for p in result.proposals):.3f}]

5. SAM2 Prompts:
   - Prompts: 06_prompts.jpg
   - Box 数量: {len(result.prompts.get('boxes', []))}
   - Point 数量: {sum(len(p) for p in result.prompts.get('points', []))}

6. 最终分割:
   - Final Masks: 07_final_masks.jpg
   - Final Overlay: 07_final_overlay.jpg
   - Mask 数量: {len(result.masks)}
   - Individual Masks: individual_masks/ (前5个)

=== 统计信息 ===

Proposals:
"""
    
    for i, proposal in enumerate(result.proposals[:10]):
        summary_text += f"  #{i}: bbox={proposal.bbox}, obj={proposal.objectness:.3f}, score={proposal.score:.1f}\n"
    
    summary_text += f"\nMasks:\n"
    
    for i, mask in enumerate(result.masks[:10]):
        area = mask.sum()
        ratio = area / (image.shape[0] * image.shape[1]) * 100
        summary_text += f"  #{i}: area={area:8.0f} px ({ratio:5.2f}%)\n"
    
    with open(viz_dir / "README.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    LOGGER.info(f"✓ Intermediate visualizations saved to {viz_dir}")


def process_and_save(
    pipeline: ZeroShotSegmentationPipeline,
    image: np.ndarray,
    output_dir: Path,
    stem: str,
    config: dict,
    save_viz: bool = True,
    visualize_intermediate: bool = False,
) -> int:
    """Process an image and save results."""
    # NMS config
    pipeline_dict = config.get("pipeline", {})
    nms_config = {
        "enable_nms": pipeline_dict.get("enable_nms", True),
        "iou_threshold": pipeline_dict.get("iou_threshold", 0.6),
        "objectness_weight": pipeline_dict.get("objectness_weight", 0.5),
        "confidence_weight": pipeline_dict.get("confidence_weight", 0.3),
        "area_weight": pipeline_dict.get("area_weight", 0.2),
    }
    
    LOGGER.info(f"Processing {stem}...")
    
    # Run pipeline
    result = pipeline.run(image, nms_config=nms_config)
    
    LOGGER.info(f"Generated {len(result.masks)} masks")
    
    # 可视化中间过程
    if visualize_intermediate:
        visualize_intermediate_results(image, result, output_dir, stem)
    
    # Save masks
    mask_dir = output_dir / "masks" / stem
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    mask_format = pipeline_dict.get("output_mask_format", "png")
    area_threshold = pipeline_dict.get("area_threshold", 100)
    
    valid_count = 0
    for idx, mask in enumerate(result.masks):
        area = int(mask.astype(np.uint8).sum())
        if area >= area_threshold:
            mask_path = mask_dir / f"mask_{idx:03d}.{mask_format}"
            save_mask(mask_path, mask, format_hint=mask_format)
            valid_count += 1
    
    LOGGER.info(f"Saved {valid_count} valid masks (threshold: {area_threshold})")
    
    # Save visualization
    if save_viz and pipeline_dict.get("save_visualization", True):
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create overlay
        combined = np.zeros_like(image)
        for idx, mask in enumerate(result.masks):
            color = np.array([
                (idx * 50) % 255,
                (idx * 80 + 60) % 255,
                (idx * 120 + 30) % 255,
            ], dtype=np.uint8)
            combined[mask.astype(bool)] = color
        
        alpha = pipeline_dict.get("visualization_alpha", 0.5)
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            1 - alpha,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
            alpha,
            0,
        )
        
        viz_path = viz_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(viz_path), overlay)
        LOGGER.info(f"Saved visualization: {viz_path}")
    
    return valid_count


def main() -> int:
    """Main entry point."""
    try:
        args = parse_args()
        
        if args.mode is None:
            print("Error: Please specify a mode (image, directory, video, or vkitti)")
            return 1
        
        setup_logging()
        LOGGER.info("=" * 80)
        LOGGER.info("mono3d Zero-Shot Segmentation Pipeline")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Mode: {args.mode}")
        LOGGER.info(f"Input: {args.input}")
        LOGGER.info(f"Output: {args.output}")
        
        # Load configuration
        LOGGER.info(f"Loading config from: {args.config}")
        config = load_configs(args.config, getattr(args, "prompt_config", None))
        LOGGER.info("✓ Configuration loaded")
        
        # Build pipeline
        LOGGER.info("Building pipeline...")
        pipeline = build_pipeline(config)
        LOGGER.info("✓ Pipeline ready")
        
        # Create output directory
        output_dir = ensure_directory(args.output)
        LOGGER.info(f"Output directory: {output_dir}")
        
        # Check if intermediate visualization is requested
        visualize_intermediate = getattr(args, "visualize_intermediate", False)
        if visualize_intermediate:
            LOGGER.info("✓ Intermediate visualization enabled")
        
        processed = 0
        
        if args.mode == "image":
            # Single image
            LOGGER.info(f"Processing image: {args.input}")
            sample = load_image(args.input)
            stem = Path(args.input).stem
            
            valid_count = process_and_save(
                pipeline, sample.image, output_dir, stem, config,
                visualize_intermediate=visualize_intermediate
            )
            
            LOGGER.info(f"✓ Saved {valid_count} masks to {output_dir / 'masks' / stem}")
            processed = 1
            
        elif args.mode == "directory":
            # Directory of images
            LOGGER.info(f"Processing directory: {args.input}")
            
            for sample in stream_directory_images(args.input):
                stem = Path(sample.path).stem
                LOGGER.info(f"Processing: {stem}")
                
                valid_count = process_and_save(
                    pipeline, sample.image, output_dir, stem, config
                )
                
                LOGGER.info(f"  → {valid_count} valid masks")
                processed += 1
            
        elif args.mode == "video":
            # Video file
            LOGGER.info(f"Processing video: {args.input}")
            
            frames = load_video_frames(
                args.input,
                frame_skip=args.frame_skip,
            )
            
            for idx, sample in enumerate(frames):
                stem = f"frame_{idx:06d}"
                LOGGER.info(f"Processing: {stem}")
                
                valid_count = process_and_save(
                    pipeline, sample.image, output_dir, stem, config
                )
                
                LOGGER.info(f"  → {valid_count} valid masks")
                processed += 1
            
        elif args.mode == "vkitti":
            # VKITTI dataset
            LOGGER.info(f"Processing VKITTI2: {args.input}")
            
            vkitti_filter = VkittiFilter(
                scenes=args.vkitti_scenes,
                clones=args.vkitti_clones,
                camera=args.vkitti_camera,
                limit=args.vkitti_limit,
            )
            
            for sample in iter_vkitti_frames(args.input, filter=vkitti_filter):
                meta = sample.metadata
                stem = f"{meta['scene']}_{meta['clone']}_{meta['frame']}"
                
                LOGGER.info(f"Processing: {stem}")
                
                valid_count = process_and_save(
                    pipeline, sample.image, output_dir, stem, config
                )
                
                LOGGER.info(f"  → {valid_count} valid masks")
                processed += 1
        
        LOGGER.info("=" * 80)
        LOGGER.info(f"✓ Processing complete: {processed} images")
        LOGGER.info(f"✓ Output directory: {output_dir}")
        LOGGER.info("=" * 80)
        
        return 0
    
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


# 关键修改：添加模块执行入口
if __name__ == "__main__":
    sys.exit(main())