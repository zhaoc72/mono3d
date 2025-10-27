"""Configuration dataclasses for the foreground instance segmentation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from .adapters.detection import DetectionAdapterConfig
from .adapters.segmentation import SegmentationAdapterConfig
from .sam2_segmenter import Sam2Config


@dataclass
class Dinov3BackboneConfig:
    """How to load features from the official DINOv3 repository."""

    repo_path: Optional[str] = None
    model_name: str = "dinov3_vitl16"
    checkpoint_path: Optional[str] = None
    image_size: int = 518
    output_layers: Sequence[int] = (4, 8, 12)
    layer_weights: Optional[Sequence[float]] = None
    fusion_method: str = "weighted_sum"
    enable_objectness: bool = True
    enable_pca: bool = True
    pca_dim: int = 32
    append_positional_features: bool = True
    positional_feature_scale: float = 0.1
    objectness_smoothing_kernel: int = 3
    objectness_contrast_gamma: float = 1.0
    patch_size: Optional[int] = None
    dtype: str = "float32"
    device: str = "cuda"


@dataclass
class ForegroundFusionConfig:
    """Thresholds for producing foreground seeds from adapter outputs."""

    objectness_threshold: float = 0.45
    segmentation_threshold: float = 0.35
    detection_score_threshold: float = 0.25
    min_instance_area: int = 60
    dilation_kernel: int = 3
    nms_iou_threshold: float = 0.6


@dataclass
class PromptStrategyConfig:
    """How prompts are generated for SAM2."""

    include_points: bool = True
    include_boxes: bool = True
    include_masks: bool = True
    max_points_per_region: int = 3
    grid_points_per_side: int = 0


@dataclass
class InstanceGroupingConfig:
    """Controls how fused masks are split into individual instances."""

    method: str = "connected_components"
    min_area: int = 60
    use_density_clustering: bool = False


@dataclass
class PostProcessConfig:
    """Morphological refinements after SAM2 mask prediction."""

    enable: bool = True
    closing_kernel: int = 5
    opening_kernel: int = 3
    min_instance_area: int = 60


@dataclass
class PipelineConfig:
    """Top-level configuration describing the entire pipeline."""

    dinov3: Dinov3BackboneConfig
    detection_adapter: DetectionAdapterConfig
    segmentation_adapter: SegmentationAdapterConfig
    fusion: ForegroundFusionConfig = field(default_factory=ForegroundFusionConfig)
    prompts: PromptStrategyConfig = field(default_factory=PromptStrategyConfig)
    instance_grouping: InstanceGroupingConfig = field(default_factory=InstanceGroupingConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    sam2: Sam2Config = field(default_factory=Sam2Config)


def _coerce_mapping(data: Mapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = dict(data.get(key, {})) if key in data and data[key] is not None else {}
    if not isinstance(value, MutableMapping):  # pragma: no cover - defensive
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return value


def _parse_dtype(name: str) -> str:
    name = name.lower()
    if name not in {"float32", "float16", "bfloat16"}:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported dtype '{name}' in DINOv3 config")
    return name


def make_pipeline_config(config_dict: Mapping[str, Any]) -> PipelineConfig:
    """Construct :class:`PipelineConfig` from a raw mapping."""

    dinov3_dict = _coerce_mapping(config_dict, "dinov3")
    adapter_det_dict = _coerce_mapping(config_dict, "detection_adapter")
    adapter_seg_dict = _coerce_mapping(config_dict, "segmentation_adapter")
    fusion_dict = _coerce_mapping(config_dict, "fusion")
    prompts_dict = _coerce_mapping(config_dict, "prompts")
    grouping_dict = _coerce_mapping(config_dict, "instance_grouping")
    post_dict = _coerce_mapping(config_dict, "postprocess")
    sam2_dict = _coerce_mapping(config_dict, "sam2")

    if "dtype" in dinov3_dict:
        dinov3_dict["dtype"] = _parse_dtype(str(dinov3_dict["dtype"]))

    dinov3_cfg = Dinov3BackboneConfig(**dinov3_dict)
    detection_cfg = DetectionAdapterConfig(**adapter_det_dict)
    segmentation_cfg = SegmentationAdapterConfig(**adapter_seg_dict)
    fusion_cfg = ForegroundFusionConfig(**fusion_dict)
    prompt_cfg = PromptStrategyConfig(**prompts_dict)
    grouping_cfg = InstanceGroupingConfig(**grouping_dict)
    post_cfg = PostProcessConfig(**post_dict)
    sam2_cfg = Sam2Config(**sam2_dict)

    return PipelineConfig(
        dinov3=dinov3_cfg,
        detection_adapter=detection_cfg,
        segmentation_adapter=segmentation_cfg,
        fusion=fusion_cfg,
        prompts=prompt_cfg,
        instance_grouping=grouping_cfg,
        postprocess=post_cfg,
        sam2=sam2_cfg,
    )


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file."""

    from .utils import load_yaml

    raw = load_yaml(path)
    if not isinstance(raw, Mapping):  # pragma: no cover - defensive
        raise TypeError("Pipeline configuration YAML must contain a mapping at the root")
    return make_pipeline_config(raw)

