"""Unit tests exercising the class-aware prompt pipeline with lightweight stubs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.class_aware_pipeline import (
    ClassAwarePromptPipeline,
    DetectionAdapter,
    DetectionOutput,
    PromptFusionConfig,
    PromptPostProcessConfig,
    SegmentationAdapter,
    SegmentationOutput,
)
from src.dinov3_feature import Dinov3Config
from src.sam2_segmenter import Sam2Config


class _FakeExtractor:
    def extract_features(self, image: np.ndarray) -> dict:
        grid_size = (2, 2)
        patch_tokens = np.zeros((4, 8), dtype=np.float32)
        patch_map = patch_tokens.reshape(grid_size[0], grid_size[1], -1)
        objectness = np.array([[0.9, 0.85], [0.2, 0.1]], dtype=np.float32)
        return {
            "patch_tokens": patch_tokens,
            "cls_token": np.zeros(8, dtype=np.float32),
            "grid_size": grid_size,
            "patch_map": patch_map,
            "attention_map": None,
            "objectness_map": objectness,
            "processed_image_shape": image.shape[:2],
        }


class _FakeDetectionAdapter:
    def predict(
        self,
        patch_tokens: np.ndarray,
        image_size: tuple[int, int],
        grid_size: tuple[int, int],
    ) -> DetectionOutput:
        boxes = np.array([[0, 0, 12, 12], [4, 4, 15, 15]], dtype=np.float32)
        class_ids = np.array([1, 0], dtype=np.int32)
        scores = np.array([0.92, 0.5], dtype=np.float32)
        return DetectionOutput(boxes=boxes, class_ids=class_ids, scores=scores, class_names=["bg", "car"])


class _FakeSegmentationAdapter:
    def predict(
        self,
        patch_tokens: np.ndarray,
        image_size: tuple[int, int],
        grid_size: tuple[int, int],
    ) -> SegmentationOutput:
        h, w = image_size[1], image_size[0]
        logits = np.full((3, h, w), -6.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[0:10, 0:10] = 6.0
        logits[1] = mask
        return SegmentationOutput(logits=logits, class_names=["bg", "car", "person"], activation="sigmoid")


class _FakeSegmenter:
    def segment(
        self,
        image: np.ndarray,
        boxes=None,
        points=None,
        labels=None,
        mask_inputs=None,
    ):
        outputs = []
        if mask_inputs is not None:
            for mask in mask_inputs:
                outputs.append(np.asarray(mask, dtype=np.uint8))
        else:
            for box in boxes or []:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = box
                mask[y1 : y2 + 1, x1 : x2 + 1] = 1
                outputs.append(mask)
        return outputs


def test_pipeline_generates_foreground_prompts():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    extractor = _FakeExtractor()
    detection_adapter: DetectionAdapter = _FakeDetectionAdapter()
    segmentation_adapter: SegmentationAdapter = _FakeSegmentationAdapter()
    segmenter = _FakeSegmenter()

    fusion_cfg = PromptFusionConfig(
        objectness_threshold=0.3,
        segmentation_threshold=0.3,
        class_probability_threshold=0.3,
        detection_score_threshold=0.3,
        min_component_area=4,
        min_prompt_area=4,
        max_prompts_per_class=5,
    )
    post_cfg = PromptPostProcessConfig(enable=False, min_instance_area=1)

    pipeline = ClassAwarePromptPipeline(
        Dinov3Config(),
        Sam2Config(),
        fusion_cfg,
        post_cfg,
        detection_adapter,
        segmentation_adapter,
        foreground_class_ids=[1, 2],
        background_class_ids=[0],
        extractor=extractor,
        segmenter=segmenter,
        device="cpu",
        dtype="float32",
    )

    result = pipeline.run(image)

    assert len(result.prompts) == 1, "Only one foreground prompt should remain"
    assert len(result.instances) == 1, "Exactly one instance expected"
    assert result.patch_map is not None
    assert result.segmentation_probs.shape[0] == 3
    assert result.detection.boxes.shape[0] == 2
    assert result.detection_processed is not None

    instance = result.instances[0]
    assert instance.class_name == "car"
    assert instance.mask.sum() > 0
    x1, y1, x2, y2 = instance.bbox
    assert x1 == 0 and y1 == 0
    assert x2 >= 9 and y2 >= 9
