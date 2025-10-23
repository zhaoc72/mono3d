"""Convert clustered DINOv3 features into SAM2 friendly prompts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class ClusterConfig:
    """Configuration controlling unsupervised clustering of patch embeddings."""

    num_clusters: int = 6
    max_iterations: int = 30
    random_state: int = 0
    min_region_area: int = 800
    max_regions: int = 5


@dataclass
class PromptConfig:
    """Configuration for converting regions into SAM2 prompts."""

    include_boxes: bool = True
    include_points: bool = True
    point_strategy: str = "centroid"


@dataclass
class RegionProposal:
    """A binary region with metadata used to create SAM2 prompts."""

    label: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    score: float


def _initialise_centroids(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k == 1:
        return features.mean(axis=0, keepdims=True)
    indices = rng.choice(len(features), size=k, replace=False)
    return features[indices]


def _assign_clusters(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    return distances.argmin(axis=1)


def _update_centroids(features: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    new_centroids = centroids.copy()
    for idx in range(centroids.shape[0]):
        mask = labels == idx
        if np.any(mask):
            new_centroids[idx] = features[mask].mean(axis=0)
    return new_centroids


def kmeans_cluster(features: np.ndarray, config: ClusterConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Simple numpy k-means implementation for deterministic unit tests."""

    k = min(config.num_clusters, len(features))
    if k == 0:
        raise ValueError("No features provided for clustering")
    rng = np.random.default_rng(config.random_state)
    centroids = _initialise_centroids(features, k, rng)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(config.max_iterations):
        labels = _assign_clusters(features, centroids)
        new_centroids = _update_centroids(features, labels, centroids)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def _resize_label_map(label_map: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(label_map.astype(np.int32), mode="I")
    resized = image.resize((width, height), resample=Image.NEAREST)
    return np.array(resized, dtype=np.int32)


def labels_to_regions(
    label_map: np.ndarray,
    image_shape: Tuple[int, int],
    config: ClusterConfig,
) -> List[RegionProposal]:
    height, width = image_shape
    proposals: List[RegionProposal] = []
    upscale = _resize_label_map(label_map, width, height)
    for label in np.unique(upscale):
        mask_bool = upscale == int(label)
        area = int(mask_bool.sum())
        if area < config.min_region_area:
            continue
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        cx = int(np.round(xs.mean()))
        cy = int(np.round(ys.mean()))
        proposals.append(
            RegionProposal(
                label=int(label),
                mask=mask_bool.astype(np.uint8),
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                centroid=(cx, cy),
                score=float(area),
            )
        )
    proposals.sort(key=lambda item: item.score, reverse=True)
    return proposals[: config.max_regions]


def proposals_to_prompts(
    proposals: Sequence[RegionProposal],
    config: PromptConfig,
) -> Tuple[List[List[int]], List[List[Tuple[int, int]]], List[List[int]]]:
    boxes: List[List[int]] = []
    points: List[List[Tuple[int, int]]] = []
    labels: List[List[int]] = []
    for proposal in proposals:
        prompt_points: List[Tuple[int, int]] = []
        point_labels: List[int] = []
        if config.include_boxes:
            boxes.append(list(proposal.bbox))
        if config.include_points:
            prompt_points.append(proposal.centroid)
            point_labels.append(1)
        if prompt_points:
            points.append(prompt_points)
            labels.append(point_labels)
        else:
            points.append([])
            labels.append([])
    return boxes, points, labels
