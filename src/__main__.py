#!/usr/bin/env python3
"""Command-line interface for the zero-shot segmentation pipeline."""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data_loader import load_image, load_video_frames, stream_directory_images
from .datasets.vkitti import VkittiFilter, iter_vkitti_frames
from .dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from .class_aware_pipeline import (
    ClassAwarePipelineResult,
    ClassAwarePromptPipeline,
    PromptFusionConfig,
    PromptPostProcessConfig,
)
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


def _maybe_extend_sys_path(paths: Sequence[str]) -> None:
    for raw_path in paths:
        if not raw_path:
            continue
        expanded = os.path.expanduser(str(raw_path))
        if expanded in sys.path:
            continue
        if os.path.isdir(expanded) or (os.path.isfile(expanded) and expanded.endswith(".py")):
            sys.path.insert(0, expanded)


def _invoke_factory(factory: Any, kwargs: Dict[str, Any]) -> Any:
    if not callable(factory):
        raise TypeError(f"Adapter factory {factory!r} is not callable")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return factory(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return factory(**filtered)


def _instantiate_adapter(
    adapter_cfg: Dict[str, Any],
    device: str,
    dtype: Any,
    extra_paths: Sequence[str] = (),
) -> Any:
    if not adapter_cfg:
        raise ValueError("Adapter configuration must be provided when class-aware mode is enabled")
    target = adapter_cfg.get("target")
    if not target:
        raise ValueError("Adapter configuration requires a 'target' callable path")

    python_paths = adapter_cfg.get("python_paths") or []
    if isinstance(python_paths, str):
        python_paths = [python_paths]
    _maybe_extend_sys_path(list(extra_paths) + list(python_paths))

    module_path, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"Invalid adapter target '{target}'. Use 'module.submodule:callable'")

    LOGGER.info("Loading adapter factory %s", target)
    module = importlib.import_module(module_path)
    factory: Any = module
    for part in attr.split("."):
        factory = getattr(factory, part)

    kwargs = dict(adapter_cfg.get("kwargs", {}))
    if "checkpoint_path" in adapter_cfg and "checkpoint_path" not in kwargs:
        kwargs["checkpoint_path"] = adapter_cfg["checkpoint_path"]

    dtype_name = getattr(dtype, "__str__", lambda: str(dtype))()
    defaults: Dict[str, Any] = {
        "device": device,
        "torch_dtype": dtype,
        "dtype": dtype,
        "dtype_str": dtype_name,
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)

    try:
        adapter = _invoke_factory(factory, kwargs)
    except Exception as exc:  # pragma: no cover - informative logging
        raise RuntimeError(f"Failed to instantiate adapter '{target}': {exc}") from exc

    return adapter


def build_pipeline(config: dict) -> ZeroShotSegmentationPipeline | ClassAwarePromptPipeline:
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

    class_aware_cfg: Dict[str, Any] = config.get("class_aware", {}) or {}
    detection_cfg = class_aware_cfg.get("detection_adapter")
    segmentation_cfg = class_aware_cfg.get("segmentation_adapter")

    enable_flag = class_aware_cfg.get("enable")
    if enable_flag is None:
        enable_flag = bool(detection_cfg and segmentation_cfg)

    if enable_flag:
        LOGGER.info("Class-aware fusion enabled; initializing adapters...")
        if detection_cfg is None or segmentation_cfg is None:
            raise ValueError(
                "Class-aware pipeline requires detection_adapter and segmentation_adapter configuration"
            )

        repo_path = config.get("dinov3", {}).get("repo_or_dir")
        extra_paths: List[str] = []
        if repo_path:
            extra_paths.append(repo_path)

        detection_adapter = _instantiate_adapter(detection_cfg, device, dtype, extra_paths)
        segmentation_adapter = _instantiate_adapter(segmentation_cfg, device, dtype, extra_paths)

        LOGGER.info(
            "DINOv3 backbone features + detection adapter + segmentation adapter → class-aware prompts"
        )
        LOGGER.info("No fine-tuning required: all modules run in zero-shot mode with official checkpoints")

        fusion_cfg = PromptFusionConfig(**class_aware_cfg.get("prompt_fusion", {}))
        post_cfg = PromptPostProcessConfig(**class_aware_cfg.get("prompt_postprocess", {}))
        foreground_ids = class_aware_cfg.get("foreground_class_ids")
        background_ids = class_aware_cfg.get("background_class_ids")

        LOGGER.info(
            "✓ Class-aware zero-shot prompt pipeline ready (foreground classes: %s)",
            foreground_ids if foreground_ids is not None else "auto",
        )

        return ClassAwarePromptPipeline(
            dinov3_cfg,
            sam2_cfg,
            fusion_cfg,
            post_cfg,
            detection_adapter,
            segmentation_adapter,
            foreground_class_ids=foreground_ids,
            background_class_ids=background_ids,
            device=device,
            dtype=dtype,
            extractor=extractor,
            segmenter=segmenter,
        )

    return ZeroShotSegmentationPipeline(
        dinov3_cfg,
        sam2_cfg,
        pipeline_cfg,
        device=device,
        dtype=dtype,
        extractor=extractor,
        segmenter=segmenter,
    )


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(array.min()) if array.size else 0.0
    max_val = float(array.max()) if array.size else 0.0
    if max_val - min_val < 1e-6:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - min_val) / (max_val - min_val)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(channel.min()) if channel.size else 0.0
    max_val = float(channel.max()) if channel.size else 0.0
    if max_val - min_val < 1e-6:
        return np.zeros_like(channel, dtype=np.float32)
    return (channel - min_val) / (max_val - min_val)


def _feature_map_to_rgb(feature_map: Optional[np.ndarray], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if feature_map is None:
        return None
    feature = np.asarray(feature_map, dtype=np.float32)
    if feature.ndim == 2:
        feature = feature[..., None]
    if feature.ndim != 3:
        return None
    channels = feature.shape[-1]
    if channels < 3:
        feature = np.repeat(feature, 3, axis=-1)
    else:
        feature = feature[..., :3]
    normalized = np.stack([
        _normalize_channel(feature[..., idx]) for idx in range(3)
    ], axis=-1)
    rgb = (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)
    height, width = image_shape
    if rgb.shape[0] != height or rgb.shape[1] != width:
        if cv2 is not None:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
        else:  # pragma: no cover - minimal fallback
            scale_y = max(1, int(np.round(height / max(rgb.shape[0], 1))))
            scale_x = max(1, int(np.round(width / max(rgb.shape[1], 1))))
            rgb = np.kron(rgb, np.ones((scale_y, scale_x, 1), dtype=np.uint8))
            rgb = rgb[:height, :width, :]
    return rgb


def _resize_map_to_image(
    array: np.ndarray,
    image_shape: Tuple[int, int],
    nearest: bool = False,
) -> np.ndarray:
    height, width = image_shape
    if array.shape == (height, width):
        return array
    if cv2 is not None:
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
        return cv2.resize(array, (width, height), interpolation=interpolation)
    # pragma: no cover - minimal fallback when OpenCV is unavailable
    scale_y = max(1, int(np.round(height / max(array.shape[0], 1))))
    scale_x = max(1, int(np.round(width / max(array.shape[1], 1))))
    resized = np.kron(array, np.ones((scale_y, scale_x), dtype=array.dtype))
    return resized[:height, :width]


def _class_color(class_id: int) -> Tuple[int, int, int]:
    seed = int(class_id)
    return (
        (seed * 97 + 37) % 255,
        (seed * 57 + 19) % 255,
        (seed * 139 + 73) % 255,
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

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    image_shape = (height, width)
    is_class_aware = isinstance(result, ClassAwarePipelineResult)

    summary_lines: List[str] = []
    summary_lines.append("=== Input ===")
    summary_lines.append(f"Image size: {width}x{height}")

    cv2.imwrite(str(viz_dir / "01_original.jpg"), image_bgr)

    # Feature map
    feature_map = getattr(result, "patch_map", None)
    feature_rgb = _feature_map_to_rgb(feature_map, image_shape)
    if feature_rgb is not None:
        cv2.imwrite(str(viz_dir / "02_feature_map.jpg"), cv2.cvtColor(feature_rgb, cv2.COLOR_RGB2BGR))

    # Attention map
    attention_map = getattr(result, "attention_map", None)
    if attention_map is not None:
        attn_resized = _resize_map_to_image(attention_map, image_shape)
        attn_norm = _normalize_to_uint8(attn_resized)
        attn_color = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(viz_dir / "03_attention_map.jpg"), attn_color)
        attention_overlay = cv2.addWeighted(image_bgr, 0.6, attn_color, 0.4, 0)
        cv2.imwrite(str(viz_dir / "03_attention_overlay.jpg"), attention_overlay)

    # Objectness map
    objectness_map = getattr(result, "objectness_map", None)
    if objectness_map is not None:
        obj_resized = _resize_map_to_image(objectness_map, image_shape)
        obj_norm = _normalize_to_uint8(obj_resized)
        obj_color = cv2.applyColorMap(obj_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(viz_dir / "04_objectness_map.jpg"), obj_color)
        obj_overlay = cv2.addWeighted(image_bgr, 0.6, obj_color, 0.4, 0)
        cv2.imwrite(str(viz_dir / "04_objectness_overlay.jpg"), obj_overlay)

    # Pipeline-specific visualizations
    if is_class_aware:
        summary_lines.append("\n=== Class-aware detection ===")
        detection = result.detection
        class_names = list(result.segmentation.class_names) if result.segmentation.class_names else []
        det_canvas = image_bgr.copy()
        if detection.boxes.size:
            for idx, (box, class_id, score) in enumerate(
                zip(detection.boxes, detection.class_ids, detection.scores)
            ):
                x1, y1, x2, y2 = map(int, box)
                color = _class_color(int(class_id))
                cv2.rectangle(det_canvas, (x1, y1), (x2, y2), color, 2)
                name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                label = f"#{idx} {name}:{score:.2f}"
                cv2.putText(
                    det_canvas,
                    label,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                summary_lines.append(
                    f"  #{idx}: {name} score={score:.3f} box=({x1},{y1},{x2},{y2})"
                )
        else:
            summary_lines.append("  (no detections)")
        cv2.imwrite(str(viz_dir / "05_detection_boxes.jpg"), det_canvas)

        # Segmentation logits overview
        seg_probs = result.segmentation_probs
        if seg_probs.size:
            seg_canvas = image_bgr.copy()
            class_map = np.argmax(seg_probs, axis=0).astype(np.int32)
            class_map_resized = _resize_map_to_image(class_map, image_shape, nearest=True)
            color_map = np.zeros((height, width, 3), dtype=np.uint8)
            for class_id in np.unique(class_map_resized):
                color_map[class_map_resized == class_id] = _class_color(int(class_id))
            seg_overlay = cv2.addWeighted(seg_canvas, 0.6, color_map, 0.4, 0)
            cv2.imwrite(str(viz_dir / "06_segmentation_argmax.jpg"), seg_overlay)

            flat = seg_probs.reshape(seg_probs.shape[0], -1)
            class_scores = flat.mean(axis=1)
            order = list(np.argsort(class_scores)[::-1])
            summary_lines.append("\n=== Segmentation confidence (mean per class) ===")
            for class_idx in order[: min(5, len(order))]:
                mean_score = float(class_scores[class_idx])
                name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                summary_lines.append(f"  {name}: {mean_score:.3f}")

            for rank, class_idx in enumerate(order[: min(3, len(order))], start=1):
                heat = _normalize_to_uint8(seg_probs[class_idx])
                heat_resized = _resize_map_to_image(heat, image_shape)
                heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
                heat_overlay = cv2.addWeighted(image_bgr, 0.6, heat_color, 0.4, 0)
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                slug = re.sub(r"[^0-9A-Za-z_-]+", "_", class_name).strip("_") or f"class_{class_idx}"
                cv2.imwrite(
                    str(viz_dir / f"06_segmentation_heat_{rank}_{slug}.jpg"),
                    heat_overlay,
                )

        # Prompts
        if result.prompts:
            prompt_canvas = image_bgr.copy()
            summary_lines.append("\n=== Prompts ===")
            for idx, prompt in enumerate(result.prompts):
                color = _class_color(prompt.class_id)
                x1, y1, x2, y2 = prompt.box
                cv2.rectangle(prompt_canvas, (x1, y1), (x2, y2), color, 2)
                cv2.circle(prompt_canvas, prompt.point, 5, color, -1)
                label = f"{prompt.class_name}:{prompt.score:.2f}"
                cv2.putText(
                    prompt_canvas,
                    label,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                if idx < 10:
                    summary_lines.append(
                        f"  #{idx}: {prompt.class_name} score={prompt.score:.3f} point={prompt.point}"
                    )
            cv2.imwrite(str(viz_dir / "07_prompts.jpg"), prompt_canvas)

        # Final masks
        instance_canvas = image_bgr.copy()
        combined = np.zeros_like(image_bgr)
        summary_lines.append("\n=== Instances ===")
        for idx, instance in enumerate(result.instances):
            mask_bool = instance.mask.astype(bool)
            color = _class_color(instance.class_id)
            combined[mask_bool] = color
            x1, y1, x2, y2 = instance.bbox
            cv2.rectangle(instance_canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                instance_canvas,
                f"{instance.class_name}:{instance.score:.2f}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            if idx < 10:
                area = int(instance.mask.astype(np.uint8).sum())
                summary_lines.append(
                    f"  #{idx}: {instance.class_name} area={area} bbox={instance.bbox}"
                )
        instance_overlay = cv2.addWeighted(instance_canvas, 0.6, combined, 0.4, 0)
        cv2.imwrite(str(viz_dir / "08_final_masks.jpg"), instance_overlay)

    else:
        # Default clustering pipeline summaries
        if result.label_map is not None:
            summary_lines.append("\n=== Clusters ===")
            unique_labels = np.unique(result.label_map)
            colored_labels = np.zeros((*result.label_map.shape, 3), dtype=np.uint8)
            for label in unique_labels:
                color = np.array([
                    (int(label) * 50) % 255,
                    (int(label) * 80 + 60) % 255,
                    (int(label) * 120 + 30) % 255,
                ], dtype=np.uint8)
                colored_labels[result.label_map == label] = color
            colored_labels_resized = cv2.resize(
                colored_labels,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            cluster_overlay = cv2.addWeighted(
                image_bgr,
                0.5,
                cv2.cvtColor(colored_labels_resized, cv2.COLOR_RGB2BGR),
                0.5,
                0,
            )
            cv2.imwrite(str(viz_dir / "05_cluster_overlay.jpg"), cluster_overlay)
            summary_lines.append(f"Clusters: {len(unique_labels)}")

        if result.proposals:
            proposal_canvas = image_bgr.copy()
            summary_lines.append("\n=== Proposals ===")
            for idx, proposal in enumerate(result.proposals):
                x0, y0, x1, y1 = proposal.bbox
                color = (0, 255, 0)
                cv2.rectangle(proposal_canvas, (x0, y0), (x1, y1), color, 2)
                label = f"#{idx} obj:{proposal.objectness:.2f}"
                cv2.putText(
                    proposal_canvas,
                    label,
                    (x0, max(y0 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                if idx < 10:
                    summary_lines.append(
                        f"  #{idx}: objectness={proposal.objectness:.3f} bbox={proposal.bbox}"
                    )
            cv2.imwrite(str(viz_dir / "06_proposals_boxes.jpg"), proposal_canvas)

        prompts_dict = getattr(result, "prompts", {}) or {}
        boxes = prompts_dict.get("boxes", [])
        points = prompts_dict.get("points", [])
        labels = prompts_dict.get("labels", [])
        if boxes or points:
            prompt_canvas = image_bgr.copy()
            summary_lines.append("\n=== Prompts ===")
            for idx, box in enumerate(boxes):
                x0, y0, x1, y1 = box
                cv2.rectangle(prompt_canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    prompt_canvas,
                    f"#{idx}",
                    (x0, max(y0 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            for prompt_points, prompt_labels in zip(points, labels):
                for point, label_val in zip(prompt_points, prompt_labels):
                    color = (0, 255, 0) if label_val == 1 else (0, 0, 255)
                    cv2.circle(prompt_canvas, tuple(point), 5, color, -1)
            cv2.imwrite(str(viz_dir / "07_prompts.jpg"), prompt_canvas)

        if processed_masks := getattr(result, "masks", []):
            summary_lines.append("\n=== Masks ===")
            combined = np.zeros_like(image_bgr)
            for idx, mask in enumerate(processed_masks):
                mask_bool = mask.astype(bool)
                color = np.array([
                    (idx * 50) % 255,
                    (idx * 80 + 60) % 255,
                    (idx * 120 + 30) % 255,
                ], dtype=np.uint8)
                combined[mask_bool] = color
                if idx < 10:
                    area = int(mask.astype(np.uint8).sum())
                    summary_lines.append(f"  #{idx}: area={area}")
            mask_overlay = cv2.addWeighted(image_bgr, 0.6, combined, 0.4, 0)
            cv2.imwrite(str(viz_dir / "08_final_masks.jpg"), mask_overlay)

    summary_path = viz_dir / "README.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

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
    pipeline: ZeroShotSegmentationPipeline | ClassAwarePromptPipeline,
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
    
    is_class_aware = isinstance(pipeline, ClassAwarePromptPipeline)

    if is_class_aware:
        result = pipeline.run(image)  # type: ignore[assignment]
    else:
        result = pipeline.run(image, nms_config=nms_config)

    if is_class_aware:
        mask_sources = [instance.mask for instance in result.instances]
        LOGGER.info(f"Generated {len(mask_sources)} class-aware instances")
    else:
        LOGGER.info(f"Generated {len(result.masks)} masks")

    # 后处理masks
    kernel_size = pipeline_dict.get("kernel_size", 7)
    processed_masks = []
    if is_class_aware:
        for instance in result.instances:
            processed_mask = postprocess_mask(instance.mask, kernel_size=kernel_size)
            instance.mask = processed_mask
            bbox_mask = np.argwhere(processed_mask > 0)
            if bbox_mask.size > 0:
                ys = bbox_mask[:, 0]
                xs = bbox_mask[:, 1]
                instance.bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            processed_masks.append(processed_mask)
    else:
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
    if is_class_aware:
        for idx, instance in enumerate(result.instances):
            mask = instance.mask.astype(np.uint8)
            area = int(mask.sum())
            if area < area_threshold:
                continue
            class_slug = re.sub(r"[^0-9A-Za-z_-]+", "_", instance.class_name).strip("_")
            if not class_slug:
                class_slug = f"class_{instance.class_id}"
            mask_path = mask_dir / f"mask_{idx:03d}_{class_slug}.{mask_format}"
            save_mask(mask_path, mask, format_hint=mask_format)
            valid_count += 1
    else:
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
        if is_class_aware:
            for idx, instance in enumerate(result.instances):
                color = np.array([
                    (instance.class_id * 67 + 40) % 255,
                    (idx * 80 + 60) % 255,
                    (instance.class_id * 97 + 30) % 255,
                ], dtype=np.uint8)
                combined[instance.mask.astype(bool)] = color
        else:
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