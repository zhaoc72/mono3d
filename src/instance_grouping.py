"""Split fused foreground regions into individual instances."""
from __future__ import annotations

from collections import deque
from typing import Iterable, List

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - fallback path
    cv2 = None  # type: ignore

from .config import InstanceGroupingConfig
from .pipeline_types import CandidateRegion


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _connected_components(mask: np.ndarray) -> Iterable[np.ndarray]:
    mask_uint8 = mask.astype(np.uint8)
    if cv2 is not None:
        num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
        for label in range(1, num_labels):
            component = labels == label
            if component.any():
                yield component
        return

    visited = np.zeros_like(mask_uint8, dtype=bool)
    h, w = mask_uint8.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not mask_uint8[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component = np.zeros_like(mask_uint8, dtype=bool)
            component[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask_uint8[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        component[ny, nx] = True
                        queue.append((ny, nx))
            yield component


def split_candidates(
    candidates: List[CandidateRegion],
    grouping_cfg: InstanceGroupingConfig,
) -> List[CandidateRegion]:
    """Apply instance grouping to produce refined candidate regions."""

    results: List[CandidateRegion] = []
    for region in candidates:
        mask = region.mask.astype(bool)
        if not mask.any():
            continue

        components = list(_connected_components(mask))
        if len(components) == 0:
            continue

        for component in components:
            area = int(component.sum())
            if area < grouping_cfg.min_area:
                continue
            bbox = _bbox_from_mask(component)
            results.append(
                CandidateRegion(
                    mask=component.astype(np.uint8),
                    bbox=bbox,
                    class_id=region.class_id,
                    class_name=region.class_name,
                    score=region.score,
                    detection_index=region.detection_index,
                )
            )

    return results

