from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dinov3_feature import Dinov3Config
from src.inference_pipeline import PipelineConfig, ZeroShotSegmentationPipeline
from src.prompt_generator import ClusterConfig, PromptConfig
from src.sam2_segmenter import Sam2Config


class StubExtractor:
    def __init__(self, grid_size: int = 4) -> None:
        coords = np.stack(np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), indexing="ij"), axis=-1)
        self.patch_map = torch.from_numpy(coords.astype(np.float32))
        self.attention_map = (coords[..., 0] + coords[..., 1]) / 2

    def extract_features(self, image):
        return {
            "patch_map": self.patch_map,
            "cls_token": torch.zeros(1),
            "grid_size": (self.patch_map.shape[0], self.patch_map.shape[1]),
            "attention_map": self.attention_map,
        }


@dataclass
class StubCall:
    boxes: list
    points: list
    labels: list
    masks: list


class StubSegmenter:
    def __init__(self) -> None:
        self.calls: list[StubCall] = []

    def segment_batched(self, image, boxes=None, points=None, labels=None, mask_inputs=None):
        height, width = image.shape[:2]
        result = []
        if boxes is None:
            boxes = [[0, 0, width, height] for _ in range(len(points or []))]
        for box in boxes:
            if box is None:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                x0, y0, x1, y1 = box
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 1
            result.append(mask)
        self.calls.append(StubCall(boxes, points or [], labels or [], mask_inputs or []))
        return result


def test_zero_shot_pipeline_returns_intermediate_artifacts():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    pipeline_cfg = PipelineConfig(
        cluster=ClusterConfig(num_clusters=2, min_region_area=10, random_state=0),
        prompt=PromptConfig(include_boxes=True, include_points=True),
    )
    pipeline = ZeroShotSegmentationPipeline(
        Dinov3Config(),
        Sam2Config(),
        pipeline_cfg,
        device="cpu",
        dtype=torch.float32,
        extractor=StubExtractor(),
        segmenter=StubSegmenter(),
    )
    result = pipeline.run(image)
    assert result.label_map.shape == (4, 4)
    assert result.prompts["boxes"], "expected generated boxes"
    assert len(result.masks) == len(result.proposals)
    assert result.cluster_centroids is not None
    assert result.attention_map is not None
    recorded = pipeline.segmenter.calls[-1]
    assert recorded.boxes == result.prompts["boxes"]
    assert recorded.points == result.prompts["points"]
    assert recorded.masks == result.prompts["mask_inputs"]
