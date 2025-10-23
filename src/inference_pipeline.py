"""Zero-shot segmentation pipeline combining DINOv3 proposals with SAM2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from .prompt_generator import (
    ClusterConfig,
    PromptConfig,
    RegionProposal,
    kmeans_cluster,
    labels_to_regions,
    proposals_to_prompts,
)
from .sam2_segmenter import SAM2Segmenter, Sam2Config
from .utils import LOGGER


@dataclass
class PipelineConfig:
    """High level knobs for the zero-shot segmentation routine."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)


@dataclass
class PipelineResult:
    """Rich return type that exposes intermediate artefacts for inspection."""

    masks: List[np.ndarray]
    proposals: List[RegionProposal]
    label_map: np.ndarray
    attention_map: Optional[np.ndarray]
    prompts: Dict[str, List]
    cluster_centroids: Optional[np.ndarray]


class ZeroShotSegmentationPipeline:
    """Orchestrates DINOv3 feature extraction, clustering and SAM2 inference."""

    def __init__(
        self,
        dinov3_config: Dinov3Config,
        sam2_config: Sam2Config,
        pipeline_config: PipelineConfig,
        device: str = "cuda",
        dtype: torch.dtype | str = torch.float32,
        extractor: Optional[DINOv3FeatureExtractor] = None,
        segmenter: Optional[SAM2Segmenter] = None,
    ) -> None:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.extractor = extractor or DINOv3FeatureExtractor(dinov3_config, device, torch_dtype)
        self.segmenter = segmenter or SAM2Segmenter(sam2_config, device, torch_dtype)
        self.config = pipeline_config

    def _cluster(self, patch_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        features = patch_map.reshape(-1, patch_map.shape[-1])
        labels, centroids = kmeans_cluster(features, self.config.cluster)
        grid_size = int(np.sqrt(len(labels)))
        label_map = labels.reshape(grid_size, grid_size)
        return label_map, centroids

    def run(self, image: np.ndarray) -> PipelineResult:
        feats = self.extractor.extract_features(image)
        patch_map = feats["patch_map"]
        if hasattr(patch_map, "detach"):
            patch_map = patch_map.detach().cpu().numpy()
        label_map, centroids = self._cluster(patch_map)
        proposals = labels_to_regions(label_map, image.shape[:2], self.config.cluster)
        if not proposals:
            LOGGER.warning("No proposals survived filtering; returning empty result")
            return PipelineResult(
                [],
                [],
                label_map,
                feats.get("attention_map"),
                {"boxes": [], "points": [], "labels": []},
                centroids,
            )

        boxes, points, labels = proposals_to_prompts(proposals, self.config.prompt)
        masks = self.segmenter.segment_batched(image, boxes, points=points, labels=labels)
        prompts: Dict[str, List] = {"boxes": boxes, "points": points, "labels": labels}
        return PipelineResult(masks, proposals, label_map, feats.get("attention_map"), prompts, centroids)


def build_pipeline(
    dinov3_cfg: Dinov3Config,
    sam2_cfg: Sam2Config,
    pipeline_cfg: Optional[PipelineConfig] = None,
    device: str = "cuda",
    dtype: str = "float32",
    extractor: Optional[DINOv3FeatureExtractor] = None,
    segmenter: Optional[SAM2Segmenter] = None,
) -> ZeroShotSegmentationPipeline:
    pipeline_cfg = pipeline_cfg or PipelineConfig()
    return ZeroShotSegmentationPipeline(
        dinov3_cfg,
        sam2_cfg,
        pipeline_cfg,
        device=device,
        dtype=dtype,
        extractor=extractor,
        segmenter=segmenter,
    )
