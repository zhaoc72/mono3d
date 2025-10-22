"""End-to-end pipeline orchestrating DINOv3 proposals and SAM2 masks."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .data_loader import ImageSample, load_image, load_video_frames
from .dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from .mask_postprocess import apply_postprocessing, select_largest_mask, select_masks_by_area
from .prompt_generator import PromptConfig, generate_prompts_from_heatmap
from .reconstruction_input import MaskArtifact, ReconstructionConfig, export_mask
from .sam2_segmenter import SAM2Segmenter, Sam2Config
from .utils import LOGGER, ensure_directory, load_yaml, setup_logging, to_torch_dtype
from .datasets.vkitti import VkittiFilter, iter_vkitti_frames


def build_configs(args: argparse.Namespace) -> Dict[str, object]:
    config = load_yaml(args.config)
    prompt_cfg = load_yaml(args.prompt_config) if args.prompt_config else {}
    dinov3_cfg = Dinov3Config(**config["dinov3"])
    sam2_cfg = Sam2Config(**config["sam2"])
    prompt_params: Dict[str, object] = {}
    if "attention" in prompt_cfg:
        prompt_params.update(prompt_cfg["attention"])
    if "points" in prompt_cfg:
        prompt_params.update(prompt_cfg["points"])
    prompt = PromptConfig(**prompt_params) if prompt_params else PromptConfig()
    reconstruction_cfg = ReconstructionConfig(**config.get("reconstruction", {}))
    return {
        "device": config["device"],
        "dtype": to_torch_dtype(config["dtype"]),
        "dinov3": dinov3_cfg,
        "sam2": sam2_cfg,
        "prompt": prompt,
        "pipeline": config["pipeline"],
        "reconstruction": reconstruction_cfg,
    }


def prepare_output_dirs(output_dir: str, save_visualization: bool) -> Dict[str, Path]:
    base = ensure_directory(output_dir)
    recon = ensure_directory(base / "recon")
    viz = ensure_directory(base / "viz") if save_visualization else None
    return {"base": base, "recon": recon, "viz": viz}


def visualize_mask(sample: ImageSample, mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = sample.image.copy()
    colored = np.zeros_like(overlay)
    colored[..., 1] = 255
    mask_u8 = mask.astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
    overlay[mask_u8 == 0] = sample.image[mask_u8 == 0]
    return overlay


def process_sample(
    extractor: DINOv3FeatureExtractor,
    segmenter: SAM2Segmenter,
    sample: ImageSample,
    configs: Dict[str, object],
    output_paths: Dict[str, Path],
    sample_index: int,
) -> Optional[Path]:
    pipeline_cfg = configs["pipeline"]
    prompt_cfg: PromptConfig = configs["prompt"]
    reconstruction_cfg: ReconstructionConfig = configs["reconstruction"]

    feats = extractor.extract(sample.image)
    heatmap = extractor.attention_to_heatmap(feats["attention"], (sample.image.shape[1], sample.image.shape[0]))
    boxes, points, labels = generate_prompts_from_heatmap(heatmap, prompt_cfg)
    if not boxes:
        LOGGER.warning("No proposals generated for %s", sample.path)
        return None

    masks = segmenter.segment_batched(
        sample.image,
        boxes,
        points=points if points else None,
        labels=labels if labels else None,
        batch_size=pipeline_cfg.get("max_prompts_per_batch"),
    )
    filtered = select_masks_by_area(masks, pipeline_cfg["area_threshold"])
    if not filtered:
        LOGGER.warning("No masks passed area threshold for %s", sample.path)
        return None

    best_mask = select_largest_mask(filtered)
    processed = apply_postprocessing(best_mask, pipeline_cfg["kernel_size"], pipeline_cfg["dilation_radius"])

    recon_cfg = replace(reconstruction_cfg, output_directory=str(output_paths["recon"]))
    artifact_metadata = {"source": sample.path, "index": sample_index}
    artifact_metadata.update(sample.metadata)
    artifact = MaskArtifact(
        mask=processed,
        metadata=artifact_metadata,
        stem=f"sample_{sample_index:05d}",
    )
    mask_path = export_mask(artifact, recon_cfg)

    if pipeline_cfg.get("save_visualization", False):
        import cv2

        overlay = visualize_mask(sample, processed.astype(np.uint8), pipeline_cfg.get("visualization_alpha", 0.5))
        cv2.imwrite(str(output_paths["viz"] / f"{artifact.stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return mask_path


def run_image_mode(args: argparse.Namespace, configs: Dict[str, object]) -> None:
    sample = load_image(args.input, target_size=configs["pipeline"].get("input_size"))
    extractor = DINOv3FeatureExtractor(configs["dinov3"], configs["device"], configs["dtype"])
    segmenter = SAM2Segmenter(configs["sam2"], configs["device"], configs["dtype"])
    outputs = prepare_output_dirs(args.output, configs["pipeline"].get("save_visualization", False))
    process_sample(extractor, segmenter, sample, configs, outputs, 0)


def run_directory_mode(args: argparse.Namespace, configs: Dict[str, object]) -> None:
    from .data_loader import stream_directory_images

    extractor = DINOv3FeatureExtractor(configs["dinov3"], configs["device"], configs["dtype"])
    segmenter = SAM2Segmenter(configs["sam2"], configs["device"], configs["dtype"])
    outputs = prepare_output_dirs(args.output, configs["pipeline"].get("save_visualization", False))

    for idx, sample in enumerate(stream_directory_images(args.input, target_size=configs["pipeline"].get("input_size"))):
        process_sample(extractor, segmenter, sample, configs, outputs, idx)


def run_video_mode(args: argparse.Namespace, configs: Dict[str, object]) -> None:
    extractor = DINOv3FeatureExtractor(configs["dinov3"], configs["device"], configs["dtype"])
    segmenter = SAM2Segmenter(configs["sam2"], configs["device"], configs["dtype"])
    outputs = prepare_output_dirs(args.output, configs["pipeline"].get("save_visualization", False))

    frames = load_video_frames(
        args.input,
        frame_skip=configs["pipeline"].get("frame_skip", 1),
        target_size=configs["pipeline"].get("input_size"),
    )
    for idx, sample in enumerate(frames):
        process_sample(extractor, segmenter, sample, configs, outputs, idx)


def run_vkitti_mode(args: argparse.Namespace, configs: Dict[str, object]) -> None:
    filter = VkittiFilter(
        scenes=args.vkitti_scenes,
        clones=args.vkitti_clones,
        camera=args.vkitti_camera,
        limit=args.vkitti_limit,
    )
    extractor = DINOv3FeatureExtractor(configs["dinov3"], configs["device"], configs["dtype"])
    segmenter = SAM2Segmenter(configs["sam2"], configs["device"], configs["dtype"])
    outputs = prepare_output_dirs(args.output, configs["pipeline"].get("save_visualization", False))

    for idx, sample in enumerate(
        iter_vkitti_frames(
            args.input,
            filter=filter,
            target_size=configs["pipeline"].get("input_size"),
        )
    ):
        process_sample(extractor, segmenter, sample, configs, outputs, idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 + SAM2 zero-shot instance segmentation pipeline")
    parser.add_argument("mode", choices=["image", "video", "directory", "vkitti"], help="Inference mode")
    parser.add_argument("--input", required=True, help="Input path (image, video, or directory)")
    parser.add_argument("--output", required=True, help="Directory to store outputs")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Path to model configuration YAML")
    parser.add_argument("--prompt-config", dest="prompt_config", default=None, help="Optional prompt configuration YAML")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--vkitti-camera", default="Camera_0", help="Camera stream to use for VKITTI mode")
    parser.add_argument(
        "--vkitti-scenes",
        nargs="*",
        default=None,
        help="Optional list of scene names to process in VKITTI mode",
    )
    parser.add_argument(
        "--vkitti-clones",
        nargs="*",
        default=None,
        help="Optional list of clone names to process in VKITTI mode",
    )
    parser.add_argument(
        "--vkitti-limit",
        type=int,
        default=None,
        help="Optional maximum number of frames to process in VKITTI mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    configs = build_configs(args)

    LOGGER.info("Running pipeline on %s in %s mode", args.input, args.mode)
    if args.mode == "image":
        run_image_mode(args, configs)
    elif args.mode == "video":
        run_video_mode(args, configs)
    elif args.mode == "directory":
        run_directory_mode(args, configs)
    else:
        run_vkitti_mode(args, configs)


if __name__ == "__main__":
    main()
