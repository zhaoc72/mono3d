"""Adapter modules for the simplified pipeline."""

from .detection import (
    DetectionAdapter,
    DetectionAdapterConfig,
    build_coco_adapter,
    build_detection_adapter,
)
from .segmentation import (
    SegmentationAdapter,
    SegmentationAdapterConfig,
    build_ade20k_adapter,
    build_segmentation_adapter,
)

__all__ = [
    "DetectionAdapter",
    "DetectionAdapterConfig",
    "build_coco_adapter",
    "build_detection_adapter",
    "SegmentationAdapter",
    "SegmentationAdapterConfig",
    "build_ade20k_adapter",
    "build_segmentation_adapter",
]

