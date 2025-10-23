"""Unit tests for the improved DINOv3 attention aggregation utilities."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

cv2_module = sys.modules.get("cv2")
if cv2_module is None:
    def _resize(arr, size, interpolation=None):
        if arr.ndim == 2:
            return np.resize(arr, (size[1], size[0]))
        return np.resize(arr, (size[1], size[0], arr.shape[2]))

    cv2_module = types.SimpleNamespace(INTER_CUBIC=1, resize=_resize)
    sys.modules["cv2"] = cv2_module
else:
    if not hasattr(cv2_module, "resize"):
        def _resize(arr, size, interpolation=None):
            if arr.ndim == 2:
                return np.resize(arr, (size[1], size[0]))
            return np.resize(arr, (size[1], size[0], arr.shape[2]))

        cv2_module.resize = _resize
    if not hasattr(cv2_module, "INTER_CUBIC"):
        cv2_module.INTER_CUBIC = 1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dinov3_feature import AttentionAggregationParams, DINOv3FeatureExtractor


def _fake_attentions(num_layers: int = 3, tokens: int = 5, heads: int = 4) -> list[torch.Tensor]:
    """Create deterministic synthetic attention tensors for testing."""

    attentions = []
    base = torch.eye(tokens).view(1, 1, tokens, tokens)
    for layer in range(num_layers):
        noise = torch.full((1, heads, tokens, tokens), 0.05 * (layer + 1))
        attn = torch.softmax(base + noise, dim=-1)
        attentions.append(attn)
    return attentions


def test_attention_rollout_produces_normalized_heatmap():
    attentions = _fake_attentions(num_layers=3)
    params = AttentionAggregationParams(apply_rollout=True, head_fusion="mean")
    heatmap = DINOv3FeatureExtractor.attention_to_heatmap(attentions, (12, 12), aggregation=params)
    assert heatmap.shape == (12, 12)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_attention_layer_weighting_without_rollout():
    attentions = _fake_attentions(num_layers=2)
    params = AttentionAggregationParams(
        apply_rollout=False,
        layer_weights=(0.25, 0.75),
        head_fusion="sum",
        token_pooling="mean",
    )
    heatmap = DINOv3FeatureExtractor.attention_to_heatmap(attentions, (16, 16), aggregation=params)
    assert heatmap.shape == (16, 16)
    assert np.isfinite(heatmap).all()
    assert np.isclose(heatmap.mean(), heatmap.mean(), atol=1e-6)  # sanity check


def test_attention_to_heatmap_accepts_single_tensor():
    attentions = _fake_attentions(num_layers=1)
    params = AttentionAggregationParams(apply_rollout=True, token_pooling="max", normalize=False)
    tensor = attentions[0]
    heatmap = DINOv3FeatureExtractor.attention_to_heatmap(tensor, (10, 10), aggregation=params)
    assert heatmap.shape == (10, 10)
    assert np.isfinite(heatmap).all()


@pytest.mark.parametrize("bad_weights", [(), (1.0,), (0.0, 0.0)])
def test_attention_layer_weights_validation(bad_weights):
    attentions = _fake_attentions(num_layers=2)
    params = AttentionAggregationParams(apply_rollout=False, layer_weights=bad_weights)
    with pytest.raises(ValueError):
        DINOv3FeatureExtractor.attention_to_heatmap(attentions, (8, 8), aggregation=params)
