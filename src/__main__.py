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
from .inference_pipeline import (
    AutoSuperpixelConfig,
    PipelineConfig,
    ProposalRefineConfig,
    ZeroShotSegmentationPipeline,
)
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
    
    image_parser = subparsers.add_parser("image", help="Process a single image")
    image_parser.add_argument("--input", required=True, help="Path to input image")
    image_parser.add_argument("--output", required=True, help="Output directory")
    image_parser.add_argument("--config", required=True, help="Model config YAML")
    image_parser.add_argument("--prompt-config", help="Prompt config YAML")
    image_parser.add_argument("--visualize-intermediate", action="store_true", 
                             help="Save intermediate visualization results")
    
    dir_parser = subparsers.add_parser("directory", help="Process a directory of images")
    dir_parser.add_argument("--input", required=True, help="Input directory")
    dir_parser.add_argument("--output", required=True, help="Output directory")
    dir_parser.add_argument("--config", required=True, help="Model config YAML")
    dir_parser.add_argument("--prompt-config", help="Prompt config YAML")
    
    video_parser = subparsers.add_parser("video", help="Process a video file")
    video_parser.add_argument("--input", required=True, help="Path to video file")
    video_parser.add_argument("--output", required=True, help="Output directory")
    video_parser.add_argument("--config", required=True, help="Model config YAML")
    video_parser.add_argument("--prompt-config", help="Prompt config YAML")
    video_parser.add_argument("--frame-skip", type=int, default=1, help="Frame sampling interval")
    
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
    
    if prompt_config_path:
        prompt_config = load_yaml(prompt_config_path)
        config["prompt_config_yaml"] = prompt_config
    
    return config


def build_pipeline(config: dict) -> ZeroShotSegmentationPipeline:
    """Build the segmentation pipeline from configuration."""
    device = config.get("device", "cuda")
    dtype = to_torch_dtype(config.get("dtype", "float32"))
    
    LOGGER.info(f"Using device: {device}, dtype: {dtype}")
    
    LOGGER.info("Initializing DINOv3...")
    dinov3_cfg = Dinov3Config(**config["dinov3"])
    
    LOGGER.info("Initializing SAM2...")
    sam2_cfg = Sam2Config(**config["sam2"])
    
    pipeline_dict = config.get("pipeline", {})
    
    # 读取 prompt_config_yaml（如果有）
    prompt_config_yaml = config.get("prompt_config_yaml", {})
    points_config = prompt_config_yaml.get("points", {})
    
    # ClusterConfig - 支持自适应聚类
    cluster_dict = pipeline_dict.get("cluster", {})
    cluster_cfg = ClusterConfig(
        # 聚类方法
        clustering_method=cluster_dict.get("clustering_method", "adaptive"),

        # Fixed方法参数
        num_clusters=cluster_dict.get("num_clusters", 6),
        
        # Adaptive方法参数
        min_clusters=cluster_dict.get("min_clusters", 3),
        max_clusters=cluster_dict.get("max_clusters", 20),
        cluster_selection_method=cluster_dict.get("cluster_selection_method", "elbow"),
        
        # 通用参数
        max_iterations=cluster_dict.get("max_iterations", 30),
        random_state=cluster_dict.get("random_state", 0),
        min_region_area=cluster_dict.get("min_region_area", 200),
        max_regions=cluster_dict.get("max_regions", 20),
        min_instance_area=cluster_dict.get("min_instance_area", 200),
        enable_instance_split=cluster_dict.get("enable_instance_split", True),
        max_instances_per_cluster=cluster_dict.get("max_instances_per_cluster", 8),

        use_objectness_filter=cluster_dict.get("use_objectness_filter", False),
        objectness_threshold=cluster_dict.get("objectness_threshold", 0.3),
        objectness_weight=cluster_dict.get("objectness_weight", 0.5),
        apply_objectness_mask=cluster_dict.get("apply_objectness_mask", False),
        objectness_mask_threshold=cluster_dict.get("objectness_mask_threshold"),
        objectness_min_keep_patches=cluster_dict.get("objectness_min_keep_patches", 64),
        objectness_percentile=cluster_dict.get("objectness_percentile"),
        objectness_min_std=cluster_dict.get("objectness_min_std", 0.0),
        use_connected_components=cluster_dict.get("use_connected_components", True),
        min_component_area=cluster_dict.get("min_component_area", 150),
        foreground_only=cluster_dict.get("foreground_only", True),
        max_background_ratio=cluster_dict.get("max_background_ratio", 0.85),
        merge_small_clusters=cluster_dict.get("merge_small_clusters", True),
        min_cluster_size=cluster_dict.get("min_cluster_size", 80),
        merge_similar_clusters=cluster_dict.get("merge_similar_clusters", True),
        similarity_threshold=cluster_dict.get("similarity_threshold", 0.92),
        min_similarity_cluster_size=cluster_dict.get("min_similarity_cluster_size", 0),
        merge_edge_background=cluster_dict.get("merge_edge_background", False),
        edge_background_area_ratio=cluster_dict.get("edge_background_area_ratio", 0.12),
        edge_touch_ratio=cluster_dict.get("edge_touch_ratio", 0.25),
    )
    
    LOGGER.info(f"Clustering method: {cluster_cfg.clustering_method}")
    if cluster_cfg.clustering_method == "adaptive":
        LOGGER.info(f"  Adaptive range: {cluster_cfg.min_clusters}-{cluster_cfg.max_clusters}")
        LOGGER.info(f"  Selection method: {cluster_cfg.cluster_selection_method}")
    elif cluster_cfg.clustering_method == "fixed":
        LOGGER.info(f"  Fixed k: {cluster_cfg.num_clusters}")

    # PromptConfig - 合并两个配置文件
    prompt_dict = pipeline_dict.get("prompt", {})

    # 优先使用 prompt_config.yaml 中的配置
    point_strategy = points_config.get("point_strategy", prompt_dict.get("point_strategy", "density"))
    max_points = points_config.get("positive_points_per_box", prompt_dict.get("max_points_per_region", 5))

    prompt_density_settings = {}
    if isinstance(prompt_dict.get("density_cluster"), dict):
        prompt_density_settings.update(prompt_dict["density_cluster"])
    if isinstance(points_config.get("density_cluster"), dict):
        prompt_density_settings.update(points_config["density_cluster"])
    if "method" not in prompt_density_settings:
        prompt_density_settings.setdefault("method", "meanshift")
    if prompt_density_settings:
        prompt_density_cfg = DensityClusterConfig(**prompt_density_settings)
    else:
        prompt_density_cfg = DensityClusterConfig(method="meanshift")

    prompt_cfg = PromptConfig(
        include_boxes=prompt_dict.get("include_boxes", True),
        include_points=prompt_dict.get("include_points", True),
        include_masks=prompt_dict.get("include_masks", True),
        include_heatmaps=prompt_dict.get("include_heatmaps", False),
        heatmap_weight=prompt_dict.get("heatmap_weight", 0.4),
        point_strategy=point_strategy,
        max_points_per_region=max_points,
        min_positive_points=prompt_dict.get("min_positive_points", 3),
        fallback_point_strategy=prompt_dict.get("fallback_point_strategy", "grid"),
        density_noise_handling=prompt_dict.get("density_noise_handling", "nearest"),
        grid_points_per_side=prompt_dict.get("grid_points_per_side", 3),
        log_top_k=prompt_dict.get("log_top_k", 5),
        mask_gaussian_sigma=prompt_dict.get("mask_gaussian_sigma", 0.0),
        density_cluster=prompt_density_cfg,
    )
    
    LOGGER.info(f"Prompt strategy: {prompt_cfg.point_strategy}, max_points: {prompt_cfg.max_points_per_region}")
    
    superpixel_cfg = SuperpixelConfig(**pipeline_dict.get("superpixel", {}))
    auto_superpixel_cfg = AutoSuperpixelConfig(**pipeline_dict.get("auto_superpixel", {}))
    graph_cluster_cfg = GraphClusterConfig(**pipeline_dict.get("graph_cluster", {}))
    density_cluster_cfg = DensityClusterConfig(**pipeline_dict.get("density_cluster", {}))
    crf_cfg = CRFConfig(**pipeline_dict.get("crf", {}))

    proposal_refine_cfg = ProposalRefineConfig(**pipeline_dict.get("proposal_refine", {}))

    pipeline_cfg = PipelineConfig(
        cluster=cluster_cfg,
        prompt=prompt_cfg,
        use_superpixels=pipeline_dict.get("use_superpixels", False),
        superpixel=superpixel_cfg,
        auto_superpixel=auto_superpixel_cfg,
        use_graph_clustering=pipeline_dict.get("use_graph_clustering", False),
        graph_cluster=graph_cluster_cfg,
        use_density_clustering=pipeline_dict.get("use_density_clustering", False),
        density_cluster=density_cluster_cfg,
        crf=crf_cfg,
        proposal_refine=proposal_refine_cfg,
    )
    
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
        
        objectness_resized = cv2.resize(objectness_colored, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.6,
            objectness_resized, 0.4, 0
        )
        cv2.imwrite(str(viz_dir / "03_objectness_overlay.jpg"), overlay)
    
    # 4. 可视化聚类 Label Map
    if result.label_map is not None:
        unique_labels = np.unique(result.label_map)
        colored_labels = np.zeros((*result.label_map.shape, 3), dtype=np.uint8)
        
        for label in unique_labels:
            color = np.array([
                (int(label) * 50) % 255,
                (int(label) * 80 + 60) % 255,
                (int(label) * 120 + 30) % 255
            ], dtype=np.uint8)
            colored_labels[result.label_map == label] = color
        
        colored_labels_resized = cv2.resize(
            colored_labels,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        cv2.imwrite(
            str(viz_dir / "04_cluster_labels.jpg"),
            cv2.cvtColor(colored_labels_resized, cv2.COLOR_RGB2BGR)
        )
        
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
            
            obj_score = proposal.objectness
            color_r = int(255 * (1 - obj_score))
            color_g = int(255 * obj_score)
            color = (color_g, color_r, 0)
            
            cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), color, 2)
            
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
        
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            cv2.rectangle(img_with_prompts, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                img_with_prompts, f"#{i}", (x0, y0-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        for i, (prompt_points, prompt_labels) in enumerate(zip(points, labels)):
            for point, label in zip(prompt_points, prompt_labels):
                x, y = point
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(img_with_prompts, (x, y), 5, color, -1)
        
        cv2.imwrite(
            str(viz_dir / "06_prompts.jpg"),
            cv2.cvtColor(img_with_prompts, cv2.COLOR_RGB2BGR)
        )
    
    # 7. 可视化最终 Masks
    if result.masks:
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
        
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5,
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), 0.5, 0
        )
        cv2.imwrite(str(viz_dir / "07_final_overlay.jpg"), overlay)
        
        individual_dir = viz_dir / "individual_masks"
        individual_dir.mkdir(exist_ok=True)
        
        for i, mask in enumerate(result.masks[:10]):
            mask_vis = (mask.astype(np.uint8) * 255)
            cv2.imwrite(str(individual_dir / f"mask_{i:02d}.jpg"), mask_vis)
    
    # 8. 创建流程总结
    summary_text = f"""
=== 中间过程可视化总结 ===

1. 原图: 01_original.jpg

2. DINOv3 特征:
   - Attention Map: 02_attention_map.jpg
   - Objectness Map: 03_objectness_map.jpg

3. 聚类结果:
   - Cluster Labels: 04_cluster_labels.jpg
   - 聚类数量: {len(np.unique(result.label_map)) if result.label_map is not None else 0}

4. 候选区域:
   - Proposals: 05_proposals_boxes.jpg
   - 候选数量: {len(result.proposals)}

5. SAM2 Prompts:
   - Prompts: 06_prompts.jpg
   - Box 数量: {len(result.prompts.get('boxes', []))}
   - Point 数量: {sum(len(p) for p in result.prompts.get('points', []))}

6. 最终分割:
   - Final Masks: 07_final_masks.jpg
   - Mask 数量: {len(result.masks)}

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


def postprocess_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    后处理单个mask
    
    1. 闭运算填充小孔
    2. 去除小连通域
    """
    try:
        from scipy.ndimage import label as connected_components
    except ImportError:
        # 如果没有scipy，只做形态学操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_closed = cv2.morphologyEx(
            mask.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        )
        return mask_closed.astype(np.uint8)
    
    # 1. 闭运算
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(
        mask.astype(np.uint8), 
        cv2.MORPH_CLOSE, 
        kernel
    )
    
    # 2. 去除小连通域，只保留最大的
    labeled, num = connected_components(mask_closed)
    if num > 1:
        sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
        if sizes:
            max_component = np.argmax(sizes) + 1
            mask_closed = (labeled == max_component).astype(np.uint8)
    
    return mask_closed


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
    pipeline_dict = config.get("pipeline", {})
    nms_config = {
        "enable_nms": pipeline_dict.get("enable_nms", True),
        "iou_threshold": pipeline_dict.get("iou_threshold", 0.6),
        "objectness_weight": pipeline_dict.get("objectness_weight", 0.5),
        "confidence_weight": pipeline_dict.get("confidence_weight", 0.3),
        "area_weight": pipeline_dict.get("area_weight", 0.2),
        "min_mask_objectness": pipeline_dict.get("min_mask_objectness"),
        "min_mask_area": pipeline_dict.get("min_mask_area"),
    }
    
    LOGGER.info(f"Processing {stem}...")
    
    # Run pipeline
    result = pipeline.run(image, nms_config=nms_config)
    
    LOGGER.info(f"Generated {len(result.masks)} masks")
    
    # 后处理masks
    kernel_size = pipeline_dict.get("kernel_size", 7)
    processed_masks = []
    for mask in result.masks:
        processed_mask = postprocess_mask(mask, kernel_size=kernel_size)
        processed_masks.append(processed_mask)
    
    result.masks = processed_masks
    LOGGER.info(f"Applied morphological post-processing (kernel_size={kernel_size})")
    
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
        
        LOGGER.info(f"Loading config from: {args.config}")
        config = load_configs(args.config, getattr(args, "prompt_config", None))
        LOGGER.info("✓ Configuration loaded")
        
        LOGGER.info("Building pipeline...")
        pipeline = build_pipeline(config)
        LOGGER.info("✓ Pipeline ready")
        
        output_dir = ensure_directory(args.output)
        LOGGER.info(f"Output directory: {output_dir}")
        
        visualize_intermediate = getattr(args, "visualize_intermediate", False)
        if visualize_intermediate:
            LOGGER.info("✓ Intermediate visualization enabled")
        
        processed = 0
        
        if args.mode == "image":
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


if __name__ == "__main__":
    sys.exit(main())