"""Generate SAM2 prompts from fused candidate regions."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import PromptStrategyConfig
from .pipeline_types import CandidateRegion, PromptDescription


def _centroid(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0
    cy = int(np.mean(ys))
    cx = int(np.mean(xs))
    return cx, cy


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def build_prompts(
    candidates: List[CandidateRegion],
    prompt_cfg: PromptStrategyConfig,
) -> List[PromptDescription]:
    """Convert grouped candidate regions into SAM2 prompt descriptions."""

    prompts: List[PromptDescription] = []
    for region in candidates:
        mask = region.mask.astype(bool)
        if not mask.any():
            continue

        box = region.bbox
        if box == (0, 0, 0, 0):
            box = _bbox(mask)

        point = _centroid(mask)

        mask_seed = region.mask.astype(np.uint8) if prompt_cfg.include_masks else None

        prompts.append(
            PromptDescription(
                box=box if prompt_cfg.include_boxes else (0, 0, 0, 0),
                point=point if prompt_cfg.include_points else (0, 0),
                class_id=region.class_id,
                class_name=region.class_name,
                score=region.score,
                mask_seed=mask_seed,
            )
        )

    return prompts

