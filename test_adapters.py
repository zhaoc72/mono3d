"""Quick smoke test for the refactored pipeline."""

from src.config import (
    Dinov3BackboneConfig,
    ForegroundFusionConfig,
    InstanceGroupingConfig,
    PipelineConfig,
    PostProcessConfig,
    PromptStrategyConfig,
)
from src.adapters.detection import DetectionAdapterConfig
from src.adapters.segmentation import SegmentationAdapterConfig
from src.pipeline import ForegroundSegmentationPipeline
from src.sam2_segmenter import Sam2Config
from src.utils import LOGGER, setup_logging


def build_test_pipeline() -> ForegroundSegmentationPipeline:
    dinov3_cfg = Dinov3BackboneConfig(
        repo_path="/media/pc/D/zhaochen/mono3d/dinov3",
        model_name="dinov3_vitl16",
        checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth",
        image_size=448,
        output_layers=(4, 8, 12),
        enable_objectness=True,
        enable_pca=True,
        pca_dim=32,
    )

    detection_cfg = DetectionAdapterConfig(
        checkpoint_path="",
        feature_dim=1024,
        num_classes=91,
        score_threshold=0.2,
    )

    segmentation_cfg = SegmentationAdapterConfig(
        checkpoint_path="",
        feature_dim=1024,
        num_classes=150,
    )

    pipeline_cfg = PipelineConfig(
        dinov3=dinov3_cfg,
        detection_adapter=detection_cfg,
        segmentation_adapter=segmentation_cfg,
        fusion=ForegroundFusionConfig(),
        prompts=PromptStrategyConfig(),
        instance_grouping=InstanceGroupingConfig(),
        postprocess=PostProcessConfig(),
        sam2=Sam2Config(
            backend="official",
            checkpoint_path="/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt",
            model_config="sam2.1/sam2.1_hiera_l",
        ),
    )

    return ForegroundSegmentationPipeline(pipeline_cfg)


def main() -> None:
    setup_logging()
    pipeline = build_test_pipeline()
    LOGGER.info("Pipeline constructed successfully: %s", pipeline)


if __name__ == "__main__":
    main()

