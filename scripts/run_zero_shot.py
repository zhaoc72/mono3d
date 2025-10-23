#!/usr/bin/env python3
"""Run the zero-shot DINOv3 + SAM2 pipeline on a single image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dinov3_feature import Dinov3Config
from src.inference_pipeline import PipelineConfig, ZeroShotSegmentationPipeline
from src.prompt_generator import ClusterConfig, PromptConfig
from src.sam2_segmenter import Sam2Config
from src.utils import ensure_directory, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the RGB image to segment")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/zero_shot"), help="Where to save masks and metadata")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device to run inference on")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"], help="Computation dtype")
    parser.add_argument("--dinov3-repo", default="facebookresearch/dinov2", help="torch.hub repo or local path for DINOv3")
    parser.add_argument("--dinov3-model", default="dinov2_vitl14", help="Model name to load from the repo")
    parser.add_argument("--dinov3-checkpoint", type=Path, default=None, help="Optional checkpoint path for DINOv3")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True, help="Path to SAM2 checkpoint weights")
    parser.add_argument("--sam2-config", type=str, required=True, help="SAM2 model config name (e.g. sam2_hiera_l1)"
    )
    parser.add_argument("--num-clusters", type=int, default=6, help="Number of KMeans clusters for patch embeddings")
    parser.add_argument("--min-region-area", type=int, default=800, help="Minimum region size (in pixels) to keep")
    parser.add_argument("--max-regions", type=int, default=5, help="Maximum number of prompts to send to SAM2")
    parser.add_argument("--seed", type=int, default=0, help="Random seed controlling clustering determinism")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> ZeroShotSegmentationPipeline:
    dinov3_cfg = Dinov3Config(
        repo_or_dir=str(args.dinov3_repo),
        model_name=args.dinov3_model,
        checkpoint_path=str(args.dinov3_checkpoint) if args.dinov3_checkpoint else None,
    )
    sam2_cfg = Sam2Config(
        checkpoint_path=str(args.sam2_checkpoint),
        model_config=args.sam2_config,
    )
    cluster_cfg = ClusterConfig(
        num_clusters=args.num_clusters,
        random_state=args.seed,
        min_region_area=args.min_region_area,
        max_regions=args.max_regions,
    )
    prompt_cfg = PromptConfig()
    pipeline_cfg = PipelineConfig(cluster=cluster_cfg, prompt=prompt_cfg)
    dtype = getattr(torch, args.dtype)
    return ZeroShotSegmentationPipeline(dinov3_cfg, sam2_cfg, pipeline_cfg, device=args.device, dtype=dtype)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    image.save(path)


def save_outputs(output_dir: Path, result, image_path: Path) -> Dict[str, Any]:
    ensure_directory(output_dir)
    metadata: Dict[str, Any] = {
        "image": str(image_path),
        "num_masks": len(result.masks),
        "prompts": result.prompts,
        "clusters": {
            "centroids": result.clusters.centroids.tolist(),
            "label_map_shape": list(result.label_map.shape),
        },
    }
    for idx, mask in enumerate(result.masks):
        mask_path = output_dir / f"mask_{idx:03d}.png"
        write_mask_png(mask_path, mask)
    if result.attention_map is not None:
        attn_path = output_dir / "attention_map.npy"
        np.save(attn_path, result.attention_map.astype(np.float32))
        metadata["attention_map"] = str(attn_path)
    label_path = output_dir / "label_map.npy"
    np.save(label_path, result.label_map.astype(np.int32))
    metadata["label_map"] = str(label_path)
    meta_path = output_dir / "metadata.json"
    save_json(meta_path, metadata)
    return metadata


def main() -> None:
    args = parse_args()
    setup_logging()
    image = load_rgb_image(args.image)
    pipeline = build_pipeline(args)
    result = pipeline.run(image)
    metadata = save_outputs(Path(args.output_dir), result, args.image)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()