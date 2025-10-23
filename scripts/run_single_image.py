#!/usr/bin/env python3
"""Run the zero-shot segmentation pipeline on a single RGB image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_image
from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.inference_pipeline import PipelineConfig, ZeroShotSegmentationPipeline
from src.prompt_generator import ClusterConfig, PromptConfig
from src.sam2_segmenter import SAM2Segmenter, Sam2Config
from src.utils import ensure_directory, load_yaml, save_json, save_mask, to_torch_dtype


def _build_pipeline(
    config: dict,
    cluster_cfg: ClusterConfig,
    prompt_cfg: PromptConfig,
    device_override: str | None,
    dtype_override: str | None,
) -> ZeroShotSegmentationPipeline:
    device = device_override or config.get("device", "cuda")
    dtype_str = dtype_override or config.get("dtype", "float32")
    dtype = to_torch_dtype(dtype_str)

    dinov3_cfg = Dinov3Config(**config["dinov3"])
    sam2_cfg = Sam2Config(**config["sam2"])
    pipeline_cfg = PipelineConfig(cluster=cluster_cfg, prompt=prompt_cfg)

    extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
    segmenter = SAM2Segmenter(sam2_cfg, device, dtype)
    return ZeroShotSegmentationPipeline(
        dinov3_cfg,
        sam2_cfg,
        pipeline_cfg,
        device=device,
        dtype=dtype,
        extractor=extractor,
        segmenter=segmenter,
    )


def _create_overlay(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    if not masks:
        return image
    color_layer = np.zeros_like(image)
    for idx, mask in enumerate(masks):
        color = np.array(
            [
                (37 * (idx + 1)) % 255,
                (91 * (idx + 3)) % 255,
                (53 * (idx + 5)) % 255,
            ],
            dtype=np.uint8,
        )
        color_layer[mask.astype(bool)] = color
    blended = cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0)
    return blended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the input RGB image")
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Model configuration YAML relative to the repository root",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/single_image",
        help="Directory where masks and diagnostics will be written",
    )
    parser.add_argument(
        "--dinov3-checkpoint",
        help="Override path to the DINOv3 checkpoint if different from the config file",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        help="Override path to the SAM2 checkpoint if different from the config file",
    )
    parser.add_argument("--device", help="Device to run inference on (defaults to config value)")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for the models (defaults to config value)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha used when creating the coloured overlay",
    )
    parser.add_argument(
        "--area-threshold",
        type=int,
        help="Minimum number of pixels required for a mask to be kept",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = (ROOT / args.config).resolve()
    config = load_yaml(config_path)

    if args.dinov3_checkpoint:
        config.setdefault("dinov3", {})["checkpoint_path"] = args.dinov3_checkpoint
    if args.sam2_checkpoint:
        config.setdefault("sam2", {})["checkpoint_path"] = args.sam2_checkpoint

    pipeline_section = config.get("pipeline", {})
    cluster_cfg = ClusterConfig(**pipeline_section.get("cluster", {}))
    prompt_cfg = PromptConfig(**pipeline_section.get("prompt", {}))

    pipeline = _build_pipeline(config, cluster_cfg, prompt_cfg, args.device, args.dtype)

    sample = load_image(args.image)
    result = pipeline.run(sample.image)

    output_dir = ensure_directory(args.output_dir)

    mask_format = pipeline_section.get("output_mask_format", "png")
    area_threshold = args.area_threshold
    if area_threshold is None:
        area_threshold = int(pipeline_section.get("area_threshold", 0) or 0)

    kept_masks: List[np.ndarray] = []
    mask_metadata = []
    for idx, mask in enumerate(result.masks):
        area = int(mask.astype(np.uint8).sum())
        passed = area_threshold <= 0 or area >= area_threshold
        mask_metadata.append({"index": idx, "area": area, "passed": passed})
        if passed:
            kept_masks.append(mask)
            mask_path = output_dir / f"mask_{idx:02d}.{mask_format}"
            save_mask(mask_path, mask, format_hint=mask_format)

    overlay = _create_overlay(sample.image, kept_masks, alpha=args.alpha)
    overlay_path = output_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if result.attention_map is not None:
        attn = result.attention_map
        attn_min, attn_max = float(attn.min()), float(attn.max())
        attn_norm = (attn - attn_min) / (attn_max - attn_min + 1e-6)
        attn_vis = (attn_norm * 255).astype(np.uint8)
        attn_color = cv2.applyColorMap(attn_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / "attention_map.png"), attn_color)

    metadata = {
        "image": str(Path(args.image).resolve()),
        "config": str(config_path),
        "device": args.device or config.get("device", "cuda"),
        "dtype": args.dtype or config.get("dtype", "float32"),
        "area_threshold": area_threshold,
        "num_masks": len(result.masks),
        "kept_masks": len(kept_masks),
        "mask_details": mask_metadata,
    }
    save_json(output_dir / "summary.json", metadata)

    print("Inference complete")
    print(f"  Image: {metadata['image']}")
    print(f"  Output directory: {output_dir}")
    print(f"  Total masks: {metadata['num_masks']}")
    print(f"  Masks kept (area >= {area_threshold}): {metadata['kept_masks']}")
    if kept_masks:
        print(f"  First mask saved to: {output_dir / f'mask_00.{mask_format}'}")
    else:
        print("  No masks passed the area threshold")


if __name__ == "__main__":
    main()