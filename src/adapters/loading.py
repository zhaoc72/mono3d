"""Utilities for loading pretrained adapter checkpoints with flexible key matching."""
from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import torch

from ..utils import LOGGER

_KNOWN_PREFIXES: Tuple[str, ...] = (
    "module.",
    "model.",
    "state_dict.",
    "ema.",
)


def _strip_known_prefix(key: str) -> str:
    for prefix in _KNOWN_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _build_fuzzy_matches(
    model_state: Dict[str, torch.Tensor],
    checkpoint_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return parameters from ``checkpoint_state`` aligned to ``model_state`` keys.

    The loader first tries exact key matches after removing common prefixes. When
    that fails it falls back to suffix-based matching, preferring longer suffixes
    to avoid ambiguous assignments.
    """

    sanitized = {_strip_known_prefix(k): v for k, v in checkpoint_state.items()}

    matched: Dict[str, torch.Tensor] = {}
    used_checkpoint_keys: set[str] = set()

    # Pass 1: exact key matches.
    for model_key, model_tensor in model_state.items():
        ckpt_tensor = sanitized.get(model_key)
        if ckpt_tensor is None:
            continue
        if ckpt_tensor.shape != model_tensor.shape:
            continue
        matched[model_key] = ckpt_tensor
        used_checkpoint_keys.add(model_key)

    if len(matched) == len(model_state):
        return matched

    # Pass 2: progressively shorter suffix matching.
    suffix_candidates: Dict[str, Iterable[str]] = {}
    for model_key in model_state.keys():
        if model_key in matched:
            continue
        parts = model_key.split(".")
        suffixes = [model_key]
        if len(parts) > 1:
            suffixes.append(".".join(parts[1:]))
        if len(parts) > 2:
            suffixes.append(".".join(parts[-2:]))
        suffixes.append(parts[-1])
        suffix_candidates[model_key] = suffixes

    for model_key, suffixes in suffix_candidates.items():
        model_tensor = model_state[model_key]
        for suffix in suffixes:
            best_match = None
            for ckpt_key, ckpt_tensor in sanitized.items():
                if ckpt_key in used_checkpoint_keys:
                    continue
                if ckpt_tensor.shape != model_tensor.shape:
                    continue
                if not ckpt_key.endswith(suffix):
                    continue
                best_match = ckpt_key
                break
            if best_match is not None:
                matched[model_key] = sanitized[best_match]
                used_checkpoint_keys.add(best_match)
                break

    return matched


def load_adapter_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    adapter_name: str,
) -> int:
    """Load pretrained weights into ``model`` from ``checkpoint_path``.

    Returns the number of tensors successfully loaded. The loader attempts to be
    permissive with key naming, accommodating checkpoints that embed the adapter
    under deeper prefixes than the lightweight runtime model uses.
    """

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        LOGGER.info(
            "%s adapter: no checkpoint provided, using randomly initialized weights",
            adapter_name,
        )
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "%s adapter: failed to load checkpoint %s (%s)",
            adapter_name,
            checkpoint_path,
            exc,
        )
        LOGGER.info("%s adapter: using randomly initialized weights", adapter_name)
        return 0

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:  # pragma: no cover - unexpected structure
        state_dict = checkpoint

    model_state = model.state_dict()
    compatible = _build_fuzzy_matches(model_state, state_dict)

    if not compatible:
        LOGGER.warning(
            "%s adapter: no compatible parameters found in checkpoint %s",
            adapter_name,
            checkpoint_path,
        )
        return 0

    missing = set(model_state.keys()) - set(compatible.keys())
    if missing:
        LOGGER.info(
            "%s adapter: loading %d tensors (%d unmatched)",
            adapter_name,
            len(compatible),
            len(missing),
        )
    else:
        LOGGER.info(
            "%s adapter: loading all %d tensors",
            adapter_name,
            len(compatible),
        )

    model.load_state_dict(compatible, strict=False)
    return len(compatible)
