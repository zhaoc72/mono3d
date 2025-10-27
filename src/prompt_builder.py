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

    proto_prompts: List[Tuple[PromptDescription, Tuple[int, int, int, int]]] = []

    for region in candidates:
        mask = region.mask.astype(bool)
        if not mask.any():
            continue

        candidate_box = region.bbox
        if candidate_box == (0, 0, 0, 0):
            candidate_box = _bbox(mask)

        if prompt_cfg.min_score > 0.0 and region.score < prompt_cfg.min_score:
            continue

        point = _centroid(mask)

        mask_seed = region.mask.astype(np.uint8) if prompt_cfg.include_masks else None

        proto_prompts.append(
            (
                PromptDescription(
                    box=candidate_box if prompt_cfg.include_boxes else (0, 0, 0, 0),
                    point=point if prompt_cfg.include_points else (0, 0),
                    class_id=region.class_id,
                    class_name=region.class_name,
                    score=region.score,
                    mask_seed=mask_seed,
                ),
                candidate_box,
            )
        )

    if prompt_cfg.nms_iou_threshold > 0 and proto_prompts:
        proto_prompts = _apply_nms(proto_prompts, prompt_cfg.nms_iou_threshold)

    if prompt_cfg.max_prompts_per_class > 0:
        proto_prompts = _limit_per_class(proto_prompts, prompt_cfg.max_prompts_per_class)

    proto_prompts.sort(key=lambda item: item[0].score, reverse=True)

    if prompt_cfg.max_prompts > 0:
        proto_prompts = proto_prompts[: prompt_cfg.max_prompts]

    prompts = [prompt for prompt, _ in proto_prompts]
    return prompts


def _apply_nms(
    proto_prompts: List[Tuple[PromptDescription, Tuple[int, int, int, int]]],
    iou_threshold: float,
) -> List[Tuple[PromptDescription, Tuple[int, int, int, int]]]:
    filtered: List[Tuple[PromptDescription, Tuple[int, int, int, int]]] = []
    for prompt, box in sorted(proto_prompts, key=lambda item: item[0].score, reverse=True):
        if box == (0, 0, 0, 0):
            filtered.append((prompt, box))
            continue
        should_keep = True
        for kept_prompt, kept_box in filtered:
            if kept_box == (0, 0, 0, 0):
                continue
            if _compute_iou(box, kept_box) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            filtered.append((prompt, box))
    return filtered


def _limit_per_class(
    proto_prompts: List[Tuple[PromptDescription, Tuple[int, int, int, int]]],
    max_per_class: int,
) -> List[Tuple[PromptDescription, Tuple[int, int, int, int]]]:
    kept: List[Tuple[PromptDescription, Tuple[int, int, int, int]]] = []
    counts: dict[int, int] = {}
    for prompt, box in sorted(proto_prompts, key=lambda item: item[0].score, reverse=True):
        current = counts.get(prompt.class_id, 0)
        if current >= max_per_class:
            continue
        counts[prompt.class_id] = current + 1
        kept.append((prompt, box))
    return kept


def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)

    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom

