#!/usr/bin/env python3
"""Command-line entry point for the simplified DINOv3 + SAM2 pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .adapters.detection import build_detection_adapter
from .adapters.segmentation import build_segmentation_adapter
from .config import load_pipeline_config
from .data_loader import load_image, stream_directory_images
from .dinov3_feature import Dinov3Backbone
from .pipeline import ForegroundSegmentationPipeline
from .pipeline_types import (
    DetectionOutput,
    InstancePrediction,
    PipelineResult,
    SegmentationOutput,
)
from .utils import LOGGER, ensure_directory, save_json, save_mask, setup_logging


_PALETTE: List[Tuple[int, int, int]] = [
    (240, 86, 60),
    (67, 160, 71),
    (66, 133, 244),
    (171, 71, 188),
    (255, 202, 40),
    (38, 198, 218),
    (255, 112, 67),
    (124, 179, 66),
]


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=np.uint8)
    clipped = np.zeros_like(data, dtype=np.float32)
    clipped[finite] = data[finite]
    minimum = float(clipped[finite].min())
    maximum = float(clipped[finite].max())
    if maximum - minimum < 1e-6:
        return np.zeros_like(clipped, dtype=np.uint8)
    normalized = (clipped - minimum) / (maximum - minimum)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def _resize_map(map_2d: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if map_2d.shape == shape:
        return map_2d
    height, width = shape
    return cv2.resize(map_2d, (width, height), interpolation=cv2.INTER_CUBIC)


def _segmentation_foreground(segmentation_probs: np.ndarray) -> np.ndarray:
    if segmentation_probs.ndim != 3:
        raise ValueError("Expected segmentation probabilities with shape (C, H, W)")
    if segmentation_probs.shape[0] == 1:
        return segmentation_probs[0]
    return segmentation_probs.max(axis=0)


def _save_heatmap_images(
    map_2d: np.ndarray,
    output_dir: Path,
    stem: str,
    base_image_rgb: Optional[np.ndarray],
) -> None:
    heatmap_uint8 = _normalize_to_uint8(map_2d)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(output_dir / f"{stem}_heatmap.png"), heatmap_color)

    if base_image_rgb is not None:
        base_bgr = cv2.cvtColor(base_image_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(base_bgr, 0.6, heatmap_color, 0.4, 0.0)
        cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)


def _scale_box(
    box: Tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1 = int(round(box[0] * scale_x))
    y1 = int(round(box[1] * scale_y))
    x2 = int(round(box[2] * scale_x))
    y2 = int(round(box[3] * scale_y))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def _detection_to_payload(detection: DetectionOutput) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    class_names = list(detection.class_names or [])
    for box, score, cls_id in zip(
        detection.boxes.tolist(),
        detection.scores.tolist(),
        detection.class_ids.tolist(),
    ):
        record: Dict[str, object] = {
            "box": [float(x) for x in box],
            "score": float(score),
            "class_id": int(cls_id),
        }
        if class_names and 0 <= int(cls_id) < len(class_names):
            record["class_name"] = class_names[int(cls_id)]
        payload.append(record)
    return payload


def _compute_segmentation_probabilities(segmentation: SegmentationOutput) -> np.ndarray:
    logits = segmentation.logits.astype(np.float32)
    activation = segmentation.activation.lower()
    if activation == "softmax":
        logits = logits - logits.max(axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        denom = exp_logits.sum(axis=0, keepdims=True) + 1e-8
        return exp_logits / denom
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-logits))
    raise ValueError(f"Unsupported segmentation activation: {segmentation.activation}")


def _build_segmentation_palette(num_classes: int) -> np.ndarray:
    if num_classes <= 0:
        return np.zeros((1, 3), dtype=np.uint8)
    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    return palette


def _save_detection_visualization(
    detection: DetectionOutput,
    output_dir: Path,
    image_rgb: Optional[np.ndarray] = None,
    processed_shape: Optional[Tuple[int, int]] = None,
) -> None:
    if detection.boxes.size == 0:
        return

    if image_rgb is None or processed_shape is None:
        LOGGER.warning(
            "Skipping detection visualization because image data or processed"
            " shape was not provided."
        )
        return

    base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image_rgb.shape[:2]
    processed_h, processed_w = processed_shape
    scale_x = width / max(1, processed_w)
    scale_y = height / max(1, processed_h)

    class_names = list(detection.class_names or [])

    for idx, (box, score, cls_id) in enumerate(
        zip(detection.boxes, detection.scores, detection.class_ids)
    ):
        color = _PALETTE[idx % len(_PALETTE)]
        scaled = _scale_box(tuple(box), scale_x, scale_y, width, height)
        cv2.rectangle(base_bgr, (scaled[0], scaled[1]), (scaled[2], scaled[3]), color, 2)
        class_name = None
        if class_names and int(cls_id) < len(class_names):
            class_name = class_names[int(cls_id)]
        label = class_name or str(int(cls_id))
        text = f"{label}:{float(score):.2f}"
        cv2.putText(
            base_bgr,
            text,
            (scaled[0], max(15, scaled[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_dir / "detections.png"), base_bgr)


def _save_detection_outputs(
    detection: DetectionOutput,
    output_dir: Path,
    image_rgb: Optional[np.ndarray],
    processed_shape: Optional[Tuple[int, int]],
) -> None:
    ensure_directory(output_dir)
    payload = _detection_to_payload(detection)
    save_json(output_dir / "detections.json", {"detections": payload})
    if image_rgb is not None and processed_shape is not None:
        _save_detection_visualization(detection, output_dir, image_rgb, processed_shape)


def _save_segmentation_outputs(
    segmentation: SegmentationOutput,
    output_dir: Path,
    image_rgb: Optional[np.ndarray],
    processed_shape: Optional[Tuple[int, int]],
) -> None:
    ensure_directory(output_dir)
    np.save(output_dir / "segmentation_logits.npy", segmentation.logits.astype(np.float32))
    probs = _compute_segmentation_probabilities(segmentation)
    np.save(output_dir / "segmentation_probs.npy", probs.astype(np.float32))
    label_map = probs.argmax(axis=0).astype(np.int32)
    np.save(output_dir / "segmentation_argmax.npy", label_map)

    palette = _build_segmentation_palette(segmentation.logits.shape[0])
    color_map = palette[label_map % len(palette)].astype(np.uint8)
    color_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "segmentation_map.png"), color_bgr)

    if image_rgb is not None:
        base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if color_map.shape[:2] != base_bgr.shape[:2]:
            color_map = cv2.resize(
                color_map,
                (base_bgr.shape[1], base_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            color_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(base_bgr, 0.6, color_bgr, 0.4, 0.0)
        cv2.imwrite(str(output_dir / "segmentation_overlay.png"), overlay)


def _save_adapter_metadata(
    output_dir: Path,
    image_path: str,
    detection: DetectionOutput,
    segmentation: SegmentationOutput,
    features: Dict[str, object],
) -> None:
    patch_tokens = features.get("patch_tokens")
    patch_tokens_raw = features.get("patch_tokens_raw")
    feature_dim = 0
    reduced_dim = 0
    if isinstance(patch_tokens, np.ndarray) and patch_tokens.ndim == 2:
        reduced_dim = int(patch_tokens.shape[-1])
    if isinstance(patch_tokens_raw, np.ndarray) and patch_tokens_raw.ndim == 2:
        feature_dim = int(patch_tokens_raw.shape[-1])
    elif reduced_dim:
        feature_dim = reduced_dim

    metadata: Dict[str, object] = {
        "image": image_path,
        "grid_size": [int(x) for x in features.get("grid_size", (0, 0))],
        "processed_image_shape": [int(x) for x in features.get("processed_image_shape", (0, 0))],
        "feature_dim": feature_dim,
        "reduced_feature_dim": reduced_dim,
        "num_detections": int(detection.boxes.shape[0]),
        "num_segmentation_classes": int(segmentation.logits.shape[0]),
    }
    if detection.class_names:
        metadata["detection_class_names"] = list(detection.class_names)
    if segmentation.class_names:
        metadata["segmentation_class_names"] = list(segmentation.class_names)
    save_json(output_dir / "metadata.json", metadata)


def _save_prompt_visualization(result: PipelineResult, output_dir: Path) -> None:
    if not result.prompts:
        return

    image_rgb = result.original_image
    base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = result.original_shape
    processed_h, processed_w = result.processed_shape
    scale_x = width / max(1, processed_w)
    scale_y = height / max(1, processed_h)

    for idx, prompt in enumerate(result.prompts):
        color = _PALETTE[idx % len(_PALETTE)]
        if prompt.box != (0, 0, 0, 0):
            scaled_box = _scale_box(prompt.box, scale_x, scale_y, width, height)
            cv2.rectangle(base_bgr, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), color, 1)
        if prompt.point != (0, 0):
            x = int(round(prompt.point[0] * scale_x))
            y = int(round(prompt.point[1] * scale_y))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            cv2.circle(base_bgr, (x, y), 3, color, -1)
        cv2.putText(
            base_bgr,
            str(idx),
            (10, 20 + 15 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_dir / "prompts.png"), base_bgr)


def _save_instance_overlay(instances: List[InstancePrediction], output_dir: Path, base_image: np.ndarray) -> None:
    if not instances:
        return

    base_bgr = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(base_bgr)

    for idx, instance in enumerate(instances):
        mask = instance.mask.astype(bool)
        if mask.shape[:2] != base_bgr.shape[:2]:
            resized = _resize_map(mask.astype(np.float32), base_bgr.shape[:2]) >= 0.5
            mask = resized
        color = _PALETTE[idx % len(_PALETTE)]
        overlay[mask] = color

    blended = cv2.addWeighted(base_bgr, 0.6, overlay, 0.4, 0.0)
    cv2.imwrite(str(output_dir / "instances_overlay.png"), blended)


def _save_instances(instances: List[InstancePrediction], output_dir: Path) -> None:
    masks_dir = ensure_directory(output_dir / "masks")
    records = []
    for idx, instance in enumerate(instances):
        mask_path = masks_dir / f"instance_{idx:03d}.png"
        save_mask(mask_path, instance.mask)
        records.append(
            {
                "mask": mask_path.name,
                "bbox": instance.bbox,
                "class_id": int(instance.class_id),
                "class_name": instance.class_name,
                "score": float(instance.score),
            }
        )
    save_json(output_dir / "instances.json", {"instances": records})


def _save_intermediate(result: PipelineResult, output_dir: Path) -> None:
    debug_dir = ensure_directory(output_dir / "intermediate")

    if result.fusion_debug:
        np.save(debug_dir / "segmentation_probs.npy", result.fusion_debug.segmentation_probs)
        if result.fusion_debug.objectness_resized is not None:
            np.save(debug_dir / "objectness.npy", result.fusion_debug.objectness_resized)

    detection_payload = _detection_to_payload(result.detection)
    save_json(debug_dir / "detections.json", {"detections": detection_payload})

    if result.objectness_map is not None:
        np.save(debug_dir / "objectness_tokens.npy", result.objectness_map)
    if result.attention_map is not None:
        np.save(debug_dir / "attention.npy", result.attention_map)
    if result.patch_map is not None:
        np.save(debug_dir / "patch_map.npy", result.patch_map)

    base_image = getattr(result, "original_image", None)
    if base_image is None:
        return

    _save_detection_visualization(result.detection, debug_dir, base_image, result.processed_shape)
    _save_prompt_visualization(result, debug_dir)
    _save_instance_overlay(result.instances, debug_dir, base_image)

    seg_probs = None
    if result.fusion_debug is not None:
        seg_probs = result.fusion_debug.segmentation_probs
    else:
        seg_probs = _compute_segmentation_probabilities(result.segmentation)

    if seg_probs is not None:
        foreground = _segmentation_foreground(seg_probs)
        resized_foreground = _resize_map(foreground, result.original_shape)
        _save_heatmap_images(resized_foreground, debug_dir, "segmentation", base_image)

    if result.fusion_debug and result.fusion_debug.objectness_resized is not None:
        objectness = _resize_map(result.fusion_debug.objectness_resized, result.original_shape)
        _save_heatmap_images(objectness, debug_dir, "objectness", base_image)


def _run_on_image(pipeline: ForegroundSegmentationPipeline, image_path: str) -> PipelineResult:
    sample = load_image(image_path)
    return pipeline.run(sample.image)


def _run_on_directory(pipeline: ForegroundSegmentationPipeline, directory: str) -> List[PipelineResult]:
    results: List[PipelineResult] = []
    for sample in stream_directory_images(directory):
        results.append(pipeline.run(sample.image))
    return results


def _initialize_adapter_components(config) -> Tuple[Dinov3Backbone, Any, Any]:
    dtype = ForegroundSegmentationPipeline._resolve_dtype(config.dinov3.dtype)
    device = config.dinov3.device
    backbone = Dinov3Backbone(config.dinov3, device=device, dtype=dtype)
    detection_adapter = build_detection_adapter(
        config.detection_adapter,
        device=device,
        torch_dtype=dtype,
    )
    segmentation_adapter = build_segmentation_adapter(
        config.segmentation_adapter,
        device=device,
        torch_dtype=dtype,
    )
    return backbone, detection_adapter, segmentation_adapter


def _unique_stem(base: str, used: Dict[str, int]) -> str:
    stem = base or "frame"
    count = used.get(stem, 0)
    used[stem] = count + 1
    if count == 0:
        return stem
    return f"{stem}_{count:02d}"


def _process_adapter_sample(
    sample,
    backbone: Dinov3Backbone,
    detection_adapter,
    segmentation_adapter,
    output_dir: Path,
):
    features = backbone.extract_features(sample.image)
    processed_raw = features.get("processed_image_shape")
    if processed_raw is None:
        processed_shape = sample.image.shape[:2]
    else:
        processed_shape = (int(processed_raw[0]), int(processed_raw[1]))
    grid_size = features.get("grid_size")
    if grid_size is None:
        raise ValueError("Backbone did not return grid_size for adapter inference")
    grid_size_tuple = (int(grid_size[0]), int(grid_size[1]))
    processed_hw = (processed_shape[1], processed_shape[0])

    patch_tokens_raw = features.get("patch_tokens_raw")
    if isinstance(patch_tokens_raw, np.ndarray):
        adapter_tokens = patch_tokens_raw
    else:
        adapter_tokens = features["patch_tokens"]

    detection = detection_adapter.predict(
        adapter_tokens,
        image_size=processed_hw,
        grid_size=grid_size_tuple,
    )
    segmentation = segmentation_adapter.predict(
        adapter_tokens,
        image_size=processed_hw,
        grid_size=grid_size_tuple,
    )

    LOGGER.info("Saving adapter outputs for %s", sample.path)
    ensure_directory(output_dir)
    _save_detection_outputs(detection, output_dir, sample.image, processed_shape)
    _save_segmentation_outputs(segmentation, output_dir, sample.image, processed_shape)
    _save_adapter_metadata(output_dir, sample.path, detection, segmentation, features)

    return detection, segmentation, features


def _run_adapters_mode(config, input_path: str, output_dir: Path) -> None:
    backbone, detection_adapter, segmentation_adapter = _initialize_adapter_components(config)
    target = Path(input_path)
    if target.is_dir():
        used: Dict[str, int] = {}
        summary: List[Dict[str, object]] = []
        for index, sample in enumerate(stream_directory_images(str(target))):
            stem = Path(sample.path).stem or f"frame_{index:04d}"
            unique = _unique_stem(stem, used)
            sample_dir = ensure_directory(output_dir / unique)
            detection, segmentation, features = _process_adapter_sample(
                sample,
                backbone,
                detection_adapter,
                segmentation_adapter,
                sample_dir,
            )
            summary.append(
                {
                    "name": unique,
                    "source": sample.path,
                    "num_detections": int(detection.boxes.shape[0]),
                    "grid_size": [int(x) for x in features.get("grid_size", (0, 0))],
                }
            )
        if summary:
            save_json(output_dir / "summary.json", {"frames": summary})
    else:
        sample = load_image(str(target))
        _process_adapter_sample(
            sample,
            backbone,
            detection_adapter,
            segmentation_adapter,
            output_dir,
        )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Zero-shot foreground instance segmentation")
    parser.add_argument("mode", choices=["image", "directory", "adapters"], help="Inference mode")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Pipeline configuration YAML")
    parser.add_argument(
        "--visualize-intermediate",
        action="store_true",
        help="Persist intermediate fusion artifacts",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")

    args = parser.parse_args(list(argv) if argv is not None else None)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level)

    config = load_pipeline_config(args.config)
    output_dir = ensure_directory(args.output)

    if args.mode in {"image", "directory"}:
        pipeline = ForegroundSegmentationPipeline(config)
        if args.mode == "image":
            result = _run_on_image(pipeline, args.input)
            _save_instances(result.instances, output_dir)
            if args.visualize_intermediate:
                _save_intermediate(result, output_dir)
        else:
            results = _run_on_directory(pipeline, args.input)
            aggregated = []
            for idx, result in enumerate(results):
                instance_dir = ensure_directory(output_dir / f"frame_{idx:04d}")
                _save_instances(result.instances, instance_dir)
                if args.visualize_intermediate:
                    _save_intermediate(result, instance_dir)
                aggregated.append(
                    {
                        "frame_index": idx,
                        "num_instances": len(result.instances),
                    }
                )
            save_json(output_dir / "summary.json", {"frames": aggregated})
    elif args.mode == "adapters":
        _run_adapters_mode(config, args.input, output_dir)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

