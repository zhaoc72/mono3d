"""End-to-end zero-shot foreground instance segmentation pipeline."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import PipelineConfig
from .dinov3_feature import Dinov3Backbone
from .adapters.detection import build_detection_adapter
from .adapters.segmentation import build_segmentation_adapter
from .foreground_fusion import FusionDebugInfo, fuse_foreground_regions
from .instance_grouping import split_candidates
from .pipeline_types import (
    InstancePrediction,
    PipelineResult,
    PromptDescription,
)
from .prompt_builder import build_prompts
from .sam2_segmenter import SAM2Segmenter
from .utils import LOGGER


class ForegroundSegmentationPipeline:
    """Implements the DINOv3 + SAM2 zero-shot pipeline described in the design doc."""

    def __init__(
        self,
        config: PipelineConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.config = config
        device = device or config.dinov3.device
        dtype = dtype or self._resolve_dtype(config.dinov3.dtype)
        self.device = torch.device(device)
        self.dtype = dtype

        self.backbone = Dinov3Backbone(config.dinov3, device=self.device, dtype=self.dtype)
        self.detection_adapter = build_detection_adapter(
            config.detection_adapter,
            device=device,
            torch_dtype=self.dtype,
        )
        self.segmentation_adapter = build_segmentation_adapter(
            config.segmentation_adapter,
            device=device,
            torch_dtype=self.dtype,
        )
        self.sam2 = SAM2Segmenter(config.sam2, device=self.device, dtype=self.dtype)
        self.post_cfg = config.postprocess

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        lookup = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return lookup.get(name.lower(), torch.float32)

    def _prepare_sam_prompts(self, prompts: List[PromptDescription]):
        boxes = [prompt.box for prompt in prompts] if prompts else None
        points = [[prompt.point] for prompt in prompts] if prompts else None
        labels = [[1] for _ in prompts] if prompts else None
        mask_inputs = [prompt.mask_seed for prompt in prompts] if prompts else None

        if boxes is not None and all(box == (0, 0, 0, 0) for box in boxes):
            boxes = None
        if points is not None and all(point == [(0, 0)] for point in points):
            points = None
            labels = None
        if mask_inputs is not None and not any(mask is not None for mask in mask_inputs):
            mask_inputs = None
        return boxes, points, labels, mask_inputs

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.post_cfg.enable:
            return mask
        result = mask.astype(np.uint8)
        if self.post_cfg.closing_kernel > 1:
            k = self.post_cfg.closing_kernel
            if k % 2 == 0:
                k += 1
            if k > 1:
                kernel = np.ones((k, k), dtype=np.uint8)
                if cv2 := self._try_cv2():  # type: ignore
                    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        if self.post_cfg.opening_kernel > 1:
            k = self.post_cfg.opening_kernel
            if k % 2 == 0:
                k += 1
            if k > 1:
                kernel = np.ones((k, k), dtype=np.uint8)
                if cv2 := self._try_cv2():  # type: ignore
                    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        return (result > 0).astype(np.uint8)

    @staticmethod
    def _try_cv2():  # pragma: no cover - optional dependency
        try:
            import cv2  # type: ignore

            return cv2
        except ImportError:
            return None

    def run(self, image: np.ndarray) -> PipelineResult:
        """Run zero-shot foreground instance segmentation."""

        LOGGER.info("Extracting DINOv3 features...")
        features = self.backbone.extract_features(image)
        patch_tokens = features["patch_tokens"]
        grid_size = features["grid_size"]
        processed_shape = features["processed_image_shape"]
        processed_hw = (processed_shape[1], processed_shape[0])

        detection = self.detection_adapter.predict(
            patch_tokens,
            image_size=processed_hw,
            grid_size=grid_size,
        )
        segmentation = self.segmentation_adapter.predict(
            patch_tokens,
            image_size=processed_hw,
            grid_size=grid_size,
        )

        LOGGER.info("Fusing detection, segmentation, and objectness cues...")
        candidates, fusion_debug = fuse_foreground_regions(
            detection,
            segmentation,
            features.get("objectness_map"),
            processed_shape=processed_shape,
            fusion_cfg=self.config.fusion,
        )

        LOGGER.info("Grouping foreground candidates into instances...")
        grouped_candidates = split_candidates(candidates, self.config.instance_grouping)

        LOGGER.info("Generating SAM2 prompts (%d candidates)...", len(grouped_candidates))
        prompts = build_prompts(grouped_candidates, self.config.prompts)

        boxes, points, labels, mask_inputs = self._prepare_sam_prompts(prompts)
        LOGGER.info("Running SAM2 for %d prompts", len(prompts))
        if prompts:
            masks = self.sam2.segment(
                image,
                boxes=boxes,
                points=points,
                labels=labels,
                mask_inputs=mask_inputs,
            )
        else:
            masks = []

        instances: List[InstancePrediction] = []
        for prompt, mask in zip(prompts, masks):
            refined_mask = self._postprocess_mask(mask)
            if refined_mask.sum() < self.post_cfg.min_instance_area:
                continue
            ys, xs = np.where(refined_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            instances.append(
                InstancePrediction(
                    mask=refined_mask,
                    bbox=bbox,
                    class_id=prompt.class_id,
                    class_name=prompt.class_name,
                    score=prompt.score,
                    prompt=prompt,
                )
            )

        LOGGER.info("Produced %d instances", len(instances))

        return PipelineResult(
            instances=instances,
            prompts=prompts,
            detection=detection,
            segmentation=segmentation,
            objectness_map=features.get("objectness_map"),
            processed_shape=processed_shape,
            original_shape=image.shape[:2],
            original_image=image.copy(),
            attention_map=features.get("attention_map"),
            patch_map=features.get("patch_map"),
            fusion_debug=fusion_debug,
        )

