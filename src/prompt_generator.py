"""Prompt generation utilities from DINOv3 attention maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class PromptConfig:
    normalize: bool = True
    threshold_strategy: str = "percentile"
    threshold: float = 0.5
    percentile: float = 85.0
    smoothing_kernel: int = 5
    min_component_area: int = 2000
    max_components: int = 5
    positive_points_per_box: int = 1
    negative_points_per_box: int = 1
    negative_offset: int = 5
    point_strategy: str = "peak"


def smooth_heatmap(heatmap: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return heatmap
    kernel = (kernel_size, kernel_size)
    return cv2.GaussianBlur(heatmap, kernel, 0)


def threshold_heatmap(heatmap: np.ndarray, config: PromptConfig) -> np.ndarray:
    if config.threshold_strategy == "percentile":
        thresh_value = np.percentile(heatmap, config.percentile)
    elif config.threshold_strategy == "value":
        thresh_value = config.threshold
    else:
        raise ValueError(f"Unsupported threshold strategy: {config.threshold_strategy}")
    mask = (heatmap >= thresh_value).astype(np.uint8)
    return mask


def find_components(binary_map: np.ndarray, min_area: int, max_components: int) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            contour_list.append(cnt)
    contour_list.sort(key=cv2.contourArea, reverse=True)
    return contour_list[:max_components]


def contour_to_box(contour: np.ndarray) -> List[int]:
    x, y, w, h = cv2.boundingRect(contour)
    return [int(x), int(y), int(x + w), int(y + h)]


def sample_positive_points(box: Sequence[int], count: int) -> List[Tuple[int, int]]:
    x0, y0, x1, y1 = box
    cx = int((x0 + x1) / 2)
    cy = int((y0 + y1) / 2)
    points = [(cx, cy)]
    if count <= 1:
        return points
    xs = np.linspace(x0, x1, num=count + 2, dtype=int)[1:-1]
    ys = np.linspace(y0, y1, num=count + 2, dtype=int)[1:-1]
    for x in xs:
        for y in ys:
            if len(points) >= count:
                break
            points.append((int(x), int(y)))
        if len(points) >= count:
            break
    return points[:count]


def sample_negative_points(box: Sequence[int], count: int, offset: int, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    if count <= 0:
        return []
    height, width = image_shape[:2]
    x0, y0, x1, y1 = box
    candidates = [
        (max(x0 - offset, 0), max(y0 - offset, 0)),
        (min(x1 + offset, width - 1), max(y0 - offset, 0)),
        (max(x0 - offset, 0), min(y1 + offset, height - 1)),
        (min(x1 + offset, width - 1), min(y1 + offset, height - 1)),
    ]
    unique = []
    for pt in candidates:
        if pt not in unique:
            unique.append(pt)
        if len(unique) >= count:
            break
    return unique


def generate_prompts_from_heatmap(heatmap: np.ndarray, config: PromptConfig) -> Tuple[List[List[int]], List[List[Tuple[int, int]]], List[List[int]]]:
    if config.normalize:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    smoothed = smooth_heatmap(heatmap, config.smoothing_kernel)
    binary = threshold_heatmap(smoothed, config)
    contours = find_components(binary, config.min_component_area, config.max_components)
    boxes: List[List[int]] = []
    positive_points: List[List[Tuple[int, int]]] = []
    labels: List[List[int]] = []

    for contour in contours:
        box = contour_to_box(contour)
        boxes.append(box)
        pos_points = sample_positive_points(box, config.positive_points_per_box)
        neg_points = sample_negative_points(box, config.negative_points_per_box, config.negative_offset, heatmap.shape)
        positive_points.append(pos_points + neg_points)
        labels.append([1] * len(pos_points) + [0] * len(neg_points))

    return boxes, positive_points, labels
