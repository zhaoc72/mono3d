"""Fuse detection, segmentation, and objectness cues into foreground seeds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - graceful degradation
    cv2 = None  # type: ignore

from .config import ForegroundFusionConfig
from .pipeline_types import CandidateRegion, DetectionOutput, SegmentationOutput
from .utils import LOGGER


@dataclass
class FusionDebugInfo:
    """Intermediate artifacts from the fusion step."""

    segmentation_probs: np.ndarray
    objectness_resized: Optional[np.ndarray]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _activation_to_probs(segmentation: SegmentationOutput) -> np.ndarray:
    logits = segmentation.logits.astype(np.float32)
    if segmentation.activation.lower() == "softmax":
        logits = logits - logits.max(axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        denom = exp_logits.sum(axis=0, keepdims=True) + 1e-8
        return exp_logits / denom
    return _sigmoid(logits)


def _resize_tensor(data: np.ndarray, size: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    if data.shape[-2:] == size:
        return data
    tensor = torch.from_numpy(data).unsqueeze(0)
    align_corners = mode in {"bilinear", "bicubic"}
    resized = torch.nn.functional.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=align_corners,
    )
    return resized.squeeze(0).cpu().numpy()


def _match_segmentation_channel(
    seg_output: SegmentationOutput,
    class_id: int,
    class_name: Optional[str],
) -> Optional[int]:
    if seg_output.class_names:
        if class_name and class_name in seg_output.class_names:
            return int(seg_output.class_names.index(class_name))
    if 0 <= class_id < seg_output.logits.shape[0]:
        return int(class_id)
    return None


def _apply_box(mask: np.ndarray, box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
    x1 = max(0, min(mask.shape[1] - 1, x1))
    y1 = max(0, min(mask.shape[0] - 1, y1))
    x2 = max(0, min(mask.shape[1], x2))
    y2 = max(0, min(mask.shape[0], y2))
    region = np.zeros_like(mask, dtype=np.uint8)
    region[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return region


def fuse_foreground_regions(
    detection: DetectionOutput,
    segmentation: SegmentationOutput,
    objectness_map: Optional[np.ndarray],
    processed_shape: Tuple[int, int],
    fusion_cfg: ForegroundFusionConfig,
) -> Tuple[List[CandidateRegion], FusionDebugInfo]:
    """Produce candidate foreground regions by intersecting adapter outputs."""

    height, width = processed_shape
    probs = _activation_to_probs(segmentation)
    probs_resized = _resize_tensor(probs, processed_shape, mode="bilinear")

    if objectness_map is not None:
        objectness_resized = _resize_tensor(objectness_map.astype(np.float32), processed_shape, mode="bilinear")
    else:
        objectness_resized = None

    candidates: List[CandidateRegion] = []
    for idx, (box, score, class_id) in enumerate(
        zip(detection.boxes, detection.scores, detection.class_ids)
    ):
        if score < fusion_cfg.detection_score_threshold:
            continue

        class_name = None
        if detection.class_names and int(class_id) < len(detection.class_names):
            class_name = detection.class_names[int(class_id)]
        seg_channel = _match_segmentation_channel(segmentation, int(class_id), class_name)
        if seg_channel is None:
            LOGGER.debug("Skipping detection %d: unable to match segmentation channel", idx)
            continue

        class_probs = probs_resized[seg_channel]
        mask = class_probs >= fusion_cfg.segmentation_threshold

        if objectness_resized is not None:
            mask &= objectness_resized >= fusion_cfg.objectness_threshold

        if not mask.any():
            continue

        mask = _apply_box(mask, box)
        area = int(mask.sum())
        if area < fusion_cfg.min_instance_area:
            continue

        if fusion_cfg.dilation_kernel > 1 and cv2 is not None:
            kernel_size = fusion_cfg.dilation_kernel
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = mask.astype(bool)

        score_components = [float(score)]
        score_components.append(float(class_probs[mask].mean()) if mask.any() else 0.0)
        if objectness_resized is not None and mask.any():
            score_components.append(float(objectness_resized[mask].mean()))
        fused_score = float(np.mean(score_components))

        bbox = (
            int(max(0, min(width - 1, box[0]))),
            int(max(0, min(height - 1, box[1]))),
            int(max(0, min(width - 1, box[2]))),
            int(max(0, min(height - 1, box[3]))),
        )

        candidates.append(
            CandidateRegion(
                mask=mask.astype(np.uint8),
                bbox=bbox,
                class_id=int(class_id),
                class_name=class_name or str(int(class_id)),
                score=fused_score,
                detection_index=idx,
            )
        )

    debug = FusionDebugInfo(segmentation_probs=probs_resized, objectness_resized=objectness_resized)
    return candidates, debug

