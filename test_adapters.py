#!/usr/bin/env python3
"""Quick test script for class-aware pipeline with adapters."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_image
from src.class_aware_pipeline import (
    ClassAwarePromptPipeline,
    PromptFusionConfig,
    PromptPostProcessConfig,
)
from src.dinov3_feature import Dinov3Config
from src.sam2_segmenter import Sam2Config
from src.utils import setup_logging, LOGGER

import torch

def test_class_aware_pipeline():
    """Test the class-aware pipeline with simplified adapters."""
    
    setup_logging()
    LOGGER.info("=" * 60)
    LOGGER.info("Testing Class-Aware Pipeline with Adapters")
    LOGGER.info("=" * 60)
    
    # Load test image
    test_image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
    LOGGER.info(f"Loading test image: {test_image_path}")
    sample = load_image(test_image_path)
    image = sample.image
    
    # Configure DINOv3
    dinov3_config = Dinov3Config(
        repo_or_dir="/media/pc/D/zhaochen/mono3d/dinov3",
        model_name="dinov3_vitl16",
        use_torch_hub=True,
        torchhub_source="local",
        checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth",
        image_size=448,
        patch_size=16,
        output_layers=[4, 8, 12],
        layer_weights=[0.2, 0.3, 0.5],
        fusion_method="weighted_sum",
        enable_objectness=True,
    )
    
    # Configure SAM2
    sam2_config = Sam2Config(
        backend="official",
        checkpoint_path="/media/pc/D/zhaochen/mono3d/sam2/checkpoints/sam2.1_hiera_large.pt",
        model_config="sam2.1/sam2.1_hiera_l",
    )
    
    # Configure fusion
    fusion_config = PromptFusionConfig(
        objectness_threshold=0.45,
        segmentation_threshold=0.4,
        class_probability_threshold=0.35,
        detection_score_threshold=0.25,
    )
    
    # Configure postprocess
    postprocess_config = PromptPostProcessConfig(
        enable=True,
        closing_kernel=5,
        opening_kernel=3,
        min_instance_area=60,
    )
    
    # Build adapters
    LOGGER.info("Building adapters...")
    from src.adapters.detection import build_coco_adapter
    from src.adapters.segmentation import build_ade20k_adapter
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    detection_adapter = build_coco_adapter(
        checkpoint_path="",  # Empty for testing
        device=device,
        torch_dtype=dtype,
        feature_dim=1024,
    )
    
    segmentation_adapter = build_ade20k_adapter(
        checkpoint_path="",  # Empty for testing
        device=device,
        torch_dtype=dtype,
        feature_dim=1024,
    )
    
    LOGGER.info("✓ Adapters built successfully")
    
    # Build pipeline
    LOGGER.info("Building class-aware pipeline...")
    pipeline = ClassAwarePromptPipeline(
        dinov3_config=dinov3_config,
        sam2_config=sam2_config,
        fusion_config=fusion_config,
        postprocess_config=postprocess_config,
        detection_adapter=detection_adapter,
        segmentation_adapter=segmentation_adapter,
        foreground_class_ids=None,
        background_class_ids=[0],
        device=device,
        dtype=dtype,
    )
    
    LOGGER.info("✓ Pipeline built successfully")
    
    # Run inference
    LOGGER.info("Running inference...")
    result = pipeline.run(image)
    
    # Print results
    LOGGER.info("=" * 60)
    LOGGER.info("Results:")
    LOGGER.info(f"  Number of instances: {len(result.instances)}")
    LOGGER.info(f"  Number of prompts: {len(result.prompts)}")
    LOGGER.info(f"  Detection boxes: {result.detection.boxes.shape[0]}")
    LOGGER.info(f"  Processed shape: {result.processed_shape}")
    LOGGER.info(f"  Original shape: {result.original_shape}")
    
    if result.instances:
        LOGGER.info("\n  Top instances:")
        for idx, instance in enumerate(result.instances[:5]):
            LOGGER.info(
                f"    #{idx}: {instance.class_name} "
                f"(score={instance.score:.3f}, area={instance.mask.sum()})"
            )
    
    LOGGER.info("=" * 60)
    LOGGER.info("✓ Test completed successfully!")
    
    return result


if __name__ == "__main__":
    result = test_class_aware_pipeline()