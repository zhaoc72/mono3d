#!/usr/bin/env python3
"""Command-line entry point for the simplified DINOv3 + SAM2 pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import load_pipeline_config
from .data_loader import load_image, stream_directory_images
from .pipeline import ForegroundSegmentationPipeline
from .pipeline_types import InstancePrediction, PipelineResult
from .utils import ensure_directory, save_json, save_mask, setup_logging


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


def _save_detection_visualization(result: PipelineResult, output_dir: Path) -> None:
    if result.detection.boxes.size == 0:
        return

    image_rgb = result.original_image
    base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = result.original_shape
    processed_h, processed_w = result.processed_shape
    scale_x = width / max(1, processed_w)
    scale_y = height / max(1, processed_h)

    for idx, (box, score, cls_id) in enumerate(
        zip(result.detection.boxes, result.detection.scores, result.detection.class_ids)
    ):
        color = _PALETTE[idx % len(_PALETTE)]
        scaled = _scale_box(tuple(box), scale_x, scale_y, width, height)
        cv2.rectangle(base_bgr, (scaled[0], scaled[1]), (scaled[2], scaled[3]), color, 2)
        class_name = None
        if result.detection.class_names and int(cls_id) < len(result.detection.class_names):
            class_name = result.detection.class_names[int(cls_id)]
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

    detection_payload = []
    for box, score, cls_id in zip(
        result.detection.boxes.tolist(),
        result.detection.scores.tolist(),
        result.detection.class_ids.tolist(),
    ):
        detection_payload.append(
            {
                "box": [float(x) for x in box],
                "score": float(score),
                "class_id": int(cls_id),
            }
        )
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

    _save_detection_visualization(result, debug_dir)
    _save_prompt_visualization(result, debug_dir)
    _save_instance_overlay(result.instances, debug_dir, base_image)

    seg_probs = None
    if result.fusion_debug is not None:
        seg_probs = result.fusion_debug.segmentation_probs
    else:
        logits = result.segmentation.logits.astype(np.float32)
        if result.segmentation.activation.lower() == "softmax":
            logits = logits - logits.max(axis=0, keepdims=True)
            exp_logits = np.exp(logits)
            denom = exp_logits.sum(axis=0, keepdims=True) + 1e-8
            seg_probs = exp_logits / denom
        else:
            seg_probs = 1.0 / (1.0 + np.exp(-logits))

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


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Zero-shot foreground instance segmentation")
    parser.add_argument("mode", choices=["image", "directory"], help="Inference mode")
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
    pipeline = ForegroundSegmentationPipeline(config)

    output_dir = ensure_directory(args.output)

    if args.mode == "image":
        result = _run_on_image(pipeline, args.input)
        _save_instances(result.instances, output_dir)
        if args.visualize_intermediate:
            _save_intermediate(result, output_dir)
    elif args.mode == "directory":
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
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

