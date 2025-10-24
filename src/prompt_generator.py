"""Convert clustered DINOv3 features into SAM2 friendly prompts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .density_clustering import DensityClusterConfig, DensityClusterer, handle_noise_points


@dataclass
class ClusterConfig:
    """Configuration controlling unsupervised clustering of patch embeddings."""

    num_clusters: int = 6
    max_iterations: int = 30
    random_state: int = 0
    min_region_area: int = 800
    max_regions: int = 5
    min_instance_area: int = 400

    # 新增：对象性筛选配置
    use_objectness_filter: bool = True  # 是否使用对象性过滤
    objectness_threshold: float = 0.3   # 对象性阈值（0-1）
    objectness_weight: float = 0.5      # 对象性权重（与面积组合打分）


@dataclass
class PromptConfig:
    """Configuration for converting regions into SAM2 prompts."""

    include_boxes: bool = True
    include_points: bool = True
    point_strategy: str = "density"  # centroid, density
    max_points_per_region: int = 5
    density_noise_handling: str = "nearest"
    density_cluster: DensityClusterConfig = field(
        default_factory=lambda: DensityClusterConfig(method="meanshift")
    )


@dataclass
class RegionProposal:
    """A binary region with metadata used to create SAM2 prompts."""

    label: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    score: float
    objectness: float = 0.0  # 新增：对象性评分
    patch_coords: Optional[np.ndarray] = None
    seed_points: List[Tuple[int, int]] = field(default_factory=list)
    semantic_label: Optional[int] = None
    instance_id: Optional[int] = None


@dataclass
class InstanceSeed:
    """Intermediate representation for density-clustered instance seeds."""

    point: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    patch_coords: np.ndarray
    label: int
    area: int


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


def _patch_coords_to_mask(
    patch_coords: np.ndarray,
    grid_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert patch coordinates back to an image resolution mask."""

    grid_h, grid_w = grid_shape
    height, width = image_shape
    if len(patch_coords) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    scale_x = width / float(grid_w)
    scale_y = height / float(grid_h)

    mask = np.zeros((height, width), dtype=np.uint8)
    for py, px in patch_coords:
        y0 = int(np.floor(py * scale_y))
        y1 = int(np.ceil((py + 1) * scale_y))
        x0 = int(np.floor(px * scale_x))
        x1 = int(np.ceil((px + 1) * scale_x))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        mask[y0:y1, x0:x1] = 1

    return mask


def _density_cluster_instances(
    proposal: RegionProposal,
    patch_map: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    prompt_config: PromptConfig,
    cluster_config: ClusterConfig,
) -> List[InstanceSeed]:
    """Split a semantic region into instance seeds via density clustering."""

    if patch_map is None or proposal.patch_coords is None or len(proposal.patch_coords) == 0:
        return []

    features = patch_map[proposal.patch_coords[:, 0], proposal.patch_coords[:, 1]]
    if features.ndim != 2 or len(features) < 2:
        return []

    clusterer = DensityClusterer(prompt_config.density_cluster)
    try:
        labels = clusterer.cluster(features)
    except Exception:
        return []

    labels = handle_noise_points(labels, features, method=prompt_config.density_noise_handling)
    unique_labels = [lab for lab in np.unique(labels) if lab >= 0]
    if not unique_labels:
        return []

    grid_h, grid_w = patch_map.shape[:2]
    height, width = image_shape
    region_mask = proposal.mask.astype(bool)

    seeds: List[InstanceSeed] = []
    for lab in unique_labels:
        mask = labels == lab
        coords = proposal.patch_coords[mask]
        if len(coords) == 0:
            continue

        coarse_mask = _patch_coords_to_mask(coords, (grid_h, grid_w), (height, width))
        if region_mask is not None:
            coarse_mask = np.logical_and(coarse_mask, region_mask)

        area = int(coarse_mask.sum())
        if area < cluster_config.min_instance_area:
            continue

        ys, xs = np.nonzero(coarse_mask)
        if len(xs) == 0:
            continue

        centroid = (int(np.round(xs.mean())), int(np.round(ys.mean())))
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))

        seeds.append(
            InstanceSeed(
                point=centroid,
                bbox=bbox,
                mask=coarse_mask.astype(np.uint8),
                patch_coords=coords,
                label=int(lab),
                area=area,
            )
        )

    seeds.sort(key=lambda seed: seed.area, reverse=True)
    if prompt_config.max_points_per_region > 0:
        seeds = seeds[: prompt_config.max_points_per_region]

    return seeds


def labels_to_regions(
    label_map: np.ndarray,
    image_shape: Tuple[int, int],
    config: ClusterConfig,
    objectness_map: Optional[np.ndarray] = None,  # 新增参数
) -> List[RegionProposal]:
    """
    将聚类标签转换为候选区域
    
    Args:
        label_map: [H_grid, W_grid] 聚类标签
        image_shape: (height, width) 原图尺寸
        config: 聚类配置
        objectness_map: [H_grid, W_grid] 对象性评分图（可选）
        
    Returns:
        候选区域列表
    """
    height, width = image_shape
    proposals: List[RegionProposal] = []
    upscale = _resize_label_map(label_map, width, height)
    
    # 上采样对象性图（如果提供）
    objectness_upscale = None
    if objectness_map is not None:
        objectness_pil = Image.fromarray((objectness_map * 255).astype(np.uint8), mode="L")
        objectness_resized = objectness_pil.resize((width, height), resample=Image.BILINEAR)
        objectness_upscale = np.array(objectness_resized, dtype=np.float32) / 255.0
    
    for label in np.unique(upscale):
        mask_bool = upscale == int(label)
        area = int(mask_bool.sum())
        
        # 面积过滤
        if area < config.min_region_area:
            continue
        
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            continue
        
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        cx = int(np.round(xs.mean()))
        cy = int(np.round(ys.mean()))
        
        # 计算区域的对象性评分
        objectness_score = 0.0
        if objectness_upscale is not None:
            region_objectness = objectness_upscale[mask_bool]
            objectness_score = float(region_objectness.mean())
            
            # 对象性过滤
            if config.use_objectness_filter and objectness_score < config.objectness_threshold:
                continue
        
        # 综合评分：面积 + 对象性
        if objectness_upscale is not None:
            combined_score = (
                (1 - config.objectness_weight) * float(area) +
                config.objectness_weight * objectness_score * 10000  # 缩放到与面积可比
            )
        else:
            combined_score = float(area)
        
        patch_mask = label_map == int(label)
        patch_coords = None
        if np.any(patch_mask):
            patch_coords = np.column_stack(np.nonzero(patch_mask))

        proposals.append(
            RegionProposal(
                label=int(label),
                mask=mask_bool.astype(np.uint8),
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                centroid=(cx, cy),
                score=combined_score,
                objectness=objectness_score,
                patch_coords=patch_coords,
                semantic_label=int(label),
            )
        )
    
    # 按综合得分排序
    proposals.sort(key=lambda item: item.score, reverse=True)
    return proposals[: config.max_regions]


def expand_region_instances(
    proposals: Sequence[RegionProposal],
    prompt_config: PromptConfig,
    cluster_config: ClusterConfig,
    patch_map: Optional[np.ndarray],
    image_shape: Tuple[int, int],
) -> List[RegionProposal]:
    """Explode semantic proposals into instance-level proposals using density clustering."""

    expanded: List[RegionProposal] = []
    for proposal in proposals:
        seeds: List[InstanceSeed] = []
        if prompt_config.include_points and prompt_config.point_strategy == "density":
            seeds = _density_cluster_instances(
                proposal,
                patch_map,
                image_shape,
                prompt_config,
                cluster_config,
            )

        if not seeds:
            fallback_seed = InstanceSeed(
                point=proposal.centroid,
                bbox=proposal.bbox,
                mask=proposal.mask.astype(np.uint8),
                patch_coords=proposal.patch_coords
                if proposal.patch_coords is not None
                else np.empty((0, 2), dtype=np.int32),
                label=-1,
                area=int(proposal.mask.sum()),
            )
            seeds = [fallback_seed]

        base_area = max(int(proposal.mask.sum()), 1)
        for idx, seed in enumerate(seeds):
            instance_score = proposal.score * (seed.area / base_area)
            expanded.append(
                RegionProposal(
                    label=proposal.label,
                    mask=seed.mask.astype(np.uint8),
                    bbox=seed.bbox,
                    centroid=seed.point,
                    score=instance_score,
                    objectness=proposal.objectness,
                    patch_coords=seed.patch_coords,
                    seed_points=[seed.point],
                    semantic_label=proposal.semantic_label,
                    instance_id=idx,
                )
            )

    return expanded


def _density_seed_points(
    proposal: RegionProposal,
    patch_map: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    config: PromptConfig,
    cluster_config: ClusterConfig,
) -> List[Tuple[int, int]]:
    """Generate multiple positive points using density clustering inside a proposal."""

    seeds = _density_cluster_instances(
        proposal,
        patch_map,
        image_shape,
        config,
        cluster_config,
    )

    if not seeds:
        return [proposal.centroid]

    unique_points: List[Tuple[int, int]] = []
    seen = set()
    for seed in seeds:
        if seed.point in seen:
            continue
        seen.add(seed.point)
        unique_points.append(seed.point)

    return unique_points or [proposal.centroid]


def _select_points(
    proposal: RegionProposal,
    config: PromptConfig,
    patch_map: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    cluster_config: Optional[ClusterConfig] = None,
) -> List[Tuple[int, int]]:
    if not config.include_points:
        return []

    if config.point_strategy == "density":
        if not proposal.seed_points:
            proposal.seed_points = _density_seed_points(
                proposal,
                patch_map,
                image_shape,
                config,
                cluster_config or ClusterConfig(),
            )
        return proposal.seed_points

    # 默认使用质心
    if not proposal.seed_points:
        proposal.seed_points = [proposal.centroid]
    return proposal.seed_points


def proposals_to_prompts(
    proposals: Sequence[RegionProposal],
    config: PromptConfig,
    patch_map: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    cluster_config: Optional[ClusterConfig] = None,
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
            region_points = _select_points(
                proposal,
                config,
                patch_map,
                image_shape if image_shape is not None else proposal.mask.shape,
                cluster_config,
            )
            prompt_points.extend(region_points)
            point_labels.extend([1] * len(region_points))
        if prompt_points:
            points.append(prompt_points)
            labels.append(point_labels)
        else:
            points.append([])
            labels.append([])
    return boxes, points, labels