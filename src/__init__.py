"""Public API for the simplified mono3d zero-shot pipeline."""
from .config import (
    Dinov3BackboneConfig,
    ForegroundFusionConfig,
    InstanceGroupingConfig,
    PipelineConfig,
    PostProcessConfig,
    PromptStrategyConfig,
)
from .pipeline import ForegroundSegmentationPipeline
from .pipeline_types import (
    CandidateRegion,
    DetectionOutput,
    InstancePrediction,
    PipelineResult,
    PromptDescription,
    SegmentationOutput,
)

__all__ = [
    "Dinov3BackboneConfig",
    "ForegroundFusionConfig",
    "InstanceGroupingConfig",
    "PipelineConfig",
    "PostProcessConfig",
    "PromptStrategyConfig",
    "ForegroundSegmentationPipeline",
    "CandidateRegion",
    "DetectionOutput",
    "InstancePrediction",
    "PipelineResult",
    "PromptDescription",
    "SegmentationOutput",
]

