"""Adapter modules for class-aware segmentation."""

from .detection import DetectionAdapter, build_coco_adapter
from .segmentation import SegmentationAdapter, build_ade20k_adapter

__all__ = [
    "DetectionAdapter",
    "build_coco_adapter",
    "SegmentationAdapter",
    "build_ade20k_adapter",
]