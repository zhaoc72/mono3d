"""Core dataclasses shared across the foreground segmentation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DetectionOutput:
    """Lightweight container for detection adapter predictions."""

    boxes: np.ndarray  # (N, 4) in xyxy processed-image coordinates
    class_ids: np.ndarray  # (N,)
    scores: np.ndarray  # (N,)
    class_names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:  # pragma: no cover - defensive validation
        if self.boxes.ndim != 2 or self.boxes.shape[-1] != 4:
            raise ValueError("DetectionOutput.boxes must have shape (N, 4)")
        n = self.boxes.shape[0]
        if self.class_ids.shape[0] != n or self.scores.shape[0] != n:
            raise ValueError("DetectionOutput tensors must share the same leading dimension")


@dataclass
class SegmentationOutput:
    """Container for semantic segmentation adapter outputs."""

    logits: np.ndarray  # (C, H, W)
    class_names: Optional[Sequence[str]] = None
    activation: str = "sigmoid"

    def __post_init__(self) -> None:  # pragma: no cover - defensive validation
        if self.logits.ndim != 3:
            raise ValueError("SegmentationOutput.logits must be 3D (C, H, W)")
        if self.activation.lower() not in {"sigmoid", "softmax"}:
            raise ValueError("SegmentationOutput.activation must be 'sigmoid' or 'softmax'")


@dataclass
class CandidateRegion:
    """Foreground region hypothesis prior to SAM2 refinement."""

    mask: np.ndarray  # binary mask at processed resolution
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    score: float
    detection_index: Optional[int] = None


@dataclass
class PromptDescription:
    """Prompt fed into SAM2."""

    box: Tuple[int, int, int, int]
    point: Tuple[int, int]
    class_id: int
    class_name: str
    score: float
    mask_seed: Optional[np.ndarray] = None


@dataclass
class InstancePrediction:
    """Final SAM2-refined instance."""

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    score: float
    prompt: PromptDescription


@dataclass
class PipelineResult:
    """Full pipeline output for a single image."""

    instances: List[InstancePrediction]
    prompts: List[PromptDescription]
    detection: DetectionOutput
    segmentation: SegmentationOutput
    objectness_map: Optional[np.ndarray]
    processed_shape: Tuple[int, int]
    original_shape: Tuple[int, int]
    attention_map: Optional[np.ndarray] = None
    patch_map: Optional[np.ndarray] = None
    fusion_debug: Optional[object] = None

