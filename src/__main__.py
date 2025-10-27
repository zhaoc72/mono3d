#!/usr/bin/env python3
"""Command-line entry point for the simplified DINOv3 + SAM2 pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .config import load_pipeline_config
from .data_loader import load_image, stream_directory_images
from .pipeline import ForegroundSegmentationPipeline
from .pipeline_types import InstancePrediction, PipelineResult
from .utils import ensure_directory, save_json, save_mask, setup_logging


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

