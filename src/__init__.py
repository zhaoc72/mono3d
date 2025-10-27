"""Zero-shot instance segmentation pipelines and utilities."""

from .class_aware_pipeline import (
    ClassAwareInstance,
    ClassAwarePipelineResult,
    ClassAwarePrompt,
    ClassAwarePromptPipeline,
    DetectionAdapter,
    DetectionOutput,
    PromptFusionConfig,
    PromptPostProcessConfig,
    SegmentationAdapter,
    SegmentationOutput,
)

__all__ = [
    "ClassAwareInstance",
    "ClassAwarePipelineResult",
    "ClassAwarePrompt",
    "ClassAwarePromptPipeline",
    "DetectionAdapter",
    "DetectionOutput",
    "PromptFusionConfig",
    "PromptPostProcessConfig",
    "SegmentationAdapter",
    "SegmentationOutput",
]
