"""Tests for the debug hooks added to the inference pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

cv2_module = sys.modules.get("cv2")
if cv2_module is None:
    def _resize(arr, size, interpolation=None):
        if arr.ndim == 3:
            return np.resize(arr, (size[1], size[0], arr.shape[2]))
        return np.resize(arr, (size[1], size[0]))

    cv2_module = types.SimpleNamespace(
        INTER_CUBIC=1,
        INTER_AREA=1,
        COLOR_BGR2RGB=1,
        COLOR_RGB2BGR=2,
        IMREAD_COLOR=1,
        MORPH_CLOSE=1,
        resize=_resize,
        cvtColor=lambda arr, code: arr,
        imread=lambda path, flag=0: np.zeros((4, 4, 3), dtype=np.uint8),
        addWeighted=lambda src1, alpha, src2, beta, gamma: src1,
        applyColorMap=lambda src, colormap: src,
        dilate=lambda arr, kernel: arr,
        morphologyEx=lambda arr, op, kernel: arr,
    )
    sys.modules["cv2"] = cv2_module
else:
    if not hasattr(cv2_module, "resize"):
        def _resize(arr, size, interpolation=None):
            if arr.ndim == 3:
                return np.resize(arr, (size[1], size[0], arr.shape[2]))
            return np.resize(arr, (size[1], size[0]))

        cv2_module.resize = _resize
    for attr, value in {
        "INTER_CUBIC": 1,
        "INTER_AREA": 1,
        "COLOR_BGR2RGB": 1,
        "COLOR_RGB2BGR": 2,
        "IMREAD_COLOR": 1,
        "MORPH_CLOSE": 1,
        "cvtColor": lambda arr, code: arr,
        "imread": lambda path, flag=0: np.zeros((4, 4, 3), dtype=np.uint8),
        "addWeighted": lambda src1, alpha, src2, beta, gamma: src1,
        "applyColorMap": lambda src, colormap: src,
        "dilate": lambda arr, kernel: arr,
        "morphologyEx": lambda arr, op, kernel: arr,
    }.items():
        if not hasattr(cv2_module, attr):
            setattr(cv2_module, attr, value)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import ImageSample
from src.inference_pipeline import process_sample
from src.prompt_generator import PromptConfig
from src.reconstruction_input import ReconstructionConfig


class DummyExtractor:
    def __init__(self) -> None:
        self.attn = torch.ones(1, 2, 5, 5)

    def extract(self, image: np.ndarray):
        return {"attention": self.attn, "all_attentions": [self.attn for _ in range(2)]}

    def attention_to_heatmap(self, attention, image_size, aggregation=None):
        width, height = image_size
        return np.ones((height, width), dtype=np.float32)


class DummySegmenter:
    def segment_batched(self, image, boxes, points=None, labels=None, batch_size=None):
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        return [mask]


@pytest.fixture
def sample_image() -> ImageSample:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    return ImageSample(path="synthetic.png", image=image, metadata={})


def test_process_sample_collects_debug(monkeypatch, tmp_path, sample_image):
    extractor = DummyExtractor()
    segmenter = DummySegmenter()

    def fake_generate_prompts(heatmap, config: PromptConfig):
        box = [0, 0, heatmap.shape[1] - 1, heatmap.shape[0] - 1]
        return [box], [[(2, 2)]], [[1]]

    monkeypatch.setattr("src.inference_pipeline.generate_prompts_from_heatmap", fake_generate_prompts)

    pipeline_cfg = {
        "area_threshold": 1,
        "kernel_size": 1,
        "dilation_radius": 1,
        "attention": {"apply_rollout": False, "token_pooling": "mean"},
        "debug": {"store_attentions": True, "store_masks": True},
    }
    prompt_cfg = PromptConfig(normalize=False, smoothing_kernel=1, min_component_area=1)
    reconstruction_cfg = ReconstructionConfig(output_directory=str(tmp_path / "default_recon"))

    configs = {
        "pipeline": pipeline_cfg,
        "prompt": prompt_cfg,
        "reconstruction": reconstruction_cfg,
    }

    output_paths = {
        "base": tmp_path,
        "recon": tmp_path / "recon",
        "viz": None,
    }
    output_paths["recon"].mkdir(parents=True, exist_ok=True)

    debug_records = []
    result = process_sample(extractor, segmenter, sample_image, configs, output_paths, 0, debug_records)

    assert result is not None
    assert debug_records, "Expected debug records to be collected"
    debug_entry = debug_records[0]
    assert debug_entry["heatmap"].shape == sample_image.image.shape[:2]
    assert "attentions" in debug_entry
    assert "masks" in debug_entry and debug_entry["masks"][0].shape == sample_image.image.shape[:2]
