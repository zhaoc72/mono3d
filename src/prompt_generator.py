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

    # 聚类数策略
    clustering_method: str = "adaptive"  # fixed, adaptive, hdbscan, meanshift
    
    # Fixed 方法参数
    num_clusters: int = 6  # 仅用于 fixed 方法
    
    # Adaptive 方法参数
    min_clusters: int = 3
    max_clusters: int = 20
    cluster_selection_method: str = "elbow"  # elbow, silhouette, gap
    
    # 通用参数
    max_iterations: int = 30
    random_state: int = 0
    min_region_area: int = 800
    max_regions: int = 20
    min_instance_area: int = 400

    # 对象性筛选配置
    use_objectness_filter: bool = True
    objectness_threshold: float = 0.3
    objectness_weight: float = 0.5
    
    # 连通域配置
    use_connected_components: bool = True
    min_component_area: int = 200


@dataclass
class PromptConfig:
    """Configuration for converting regions into SAM2 prompts."""

    include_boxes: bool = True
    include_points: bool = True
    point_strategy: str = "density"  # centroid, density, grid
    max_points_per_region: int = 5
    density_noise_handling: str = "nearest"
    density_cluster: DensityClusterConfig = field(
        default_factory=lambda: DensityClusterConfig(method="meanshift")
    )
    
    # grid点策略配置
    grid_points_per_side: int = 3


@dataclass
class RegionProposal:
    """A binary region with metadata used to create SAM2 prompts."""

    label: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    score: float
    objectness: float = 0.0
    patch_coords: Optional[np.ndarray] = None
    seed_points: List[Tuple[int, int]] = field(default_factory=list)
    semantic_label: Optional[int] = None
    instance_id: Optional[int] = None
    component_id: Optional[int] = None


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


def find_optimal_clusters_elbow(
    features: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int = 0
) -> int:
    """
    使用肘部法则确定最优聚类数
    
    计算不同k值的WCSS (Within-Cluster Sum of Squares)，
    找到"肘部"位置作为最优k
    """
    wcss_list = []
    k_range = range(min_k, max_k + 1)
    
    for k in k_range:
        temp_config = ClusterConfig(num_clusters=k, random_state=random_state)
        labels, centroids = kmeans_cluster(features, temp_config)
        
        # 计算WCSS
        wcss = 0.0
        for i in range(k):
            cluster_points = features[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[i]) ** 2)
        
        wcss_list.append(wcss)
    
    # 找肘部：计算每个点到首尾连线的距离
    wcss_array = np.array(wcss_list)
    
    # 归一化
    wcss_normalized = (wcss_array - wcss_array.min()) / (wcss_array.max() - wcss_array.min() + 1e-8)
    
    # 计算每个点到首尾连线的距离
    n_points = len(wcss_normalized)
    if n_points < 2:
        return min_k
    
    line_start = np.array([0, wcss_normalized[0]])
    line_end = np.array([n_points - 1, wcss_normalized[-1]])
    
    distances = []
    for i in range(n_points):
        point = np.array([i, wcss_normalized[i]])
        # 点到直线距离
        distance = np.abs(np.cross(line_end - line_start, point - line_start)) / (np.linalg.norm(line_end - line_start) + 1e-8)
        distances.append(distance)
    
    # 肘部 = 距离最大的点
    elbow_idx = np.argmax(distances)
    optimal_k = k_range[elbow_idx]
    
    return optimal_k


def find_optimal_clusters_silhouette(
    features: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int = 0
) -> int:
    """
    使用轮廓系数确定最优聚类数
    
    轮廓系数越高，聚类效果越好
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        # 降级到elbow方法
        return find_optimal_clusters_elbow(features, min_k, max_k, random_state)
    
    silhouette_scores = []
    k_range = range(max(2, min_k), max_k + 1)  # silhouette需要至少2个cluster
    
    for k in k_range:
        temp_config = ClusterConfig(num_clusters=k, random_state=random_state)
        labels, _ = kmeans_cluster(features, temp_config)
        
        # 计算轮廓系数
        if len(np.unique(labels)) > 1:
            score = silhouette_score(features, labels, sample_size=min(1000, len(features)))
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)
    
    # 选择轮廓系数最高的k
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[optimal_idx]
    
    return optimal_k


def adaptive_kmeans_cluster(
    features: np.ndarray,
    config: ClusterConfig
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    自适应确定聚类数的KMeans
    
    Returns:
        (labels, centroids, optimal_k)
    """
    min_k = max(2, config.min_clusters)
    max_k = min(config.max_clusters, len(features) // 10)  # 确保每个cluster至少10个点
    
    if max_k < min_k:
        max_k = min_k
    
    # 根据选择方法确定最优k
    if config.cluster_selection_method == "elbow":
        optimal_k = find_optimal_clusters_elbow(features, min_k, max_k, config.random_state)
    elif config.cluster_selection_method == "silhouette":
        optimal_k = find_optimal_clusters_silhouette(features, min_k, max_k, config.random_state)
    elif config.cluster_selection_method == "gap":
        # Gap statistic (更复杂，这里简化为elbow)
        optimal_k = find_optimal_clusters_elbow(features, min_k, max_k, config.random_state)
    else:
        optimal_k = find_optimal_clusters_elbow(features, min_k, max_k, config.random_state)
    
    # 使用最优k进行聚类
    temp_config = ClusterConfig(
        num_clusters=optimal_k,
        max_iterations=config.max_iterations,
        random_state=config.random_state
    )
    labels, centroids = kmeans_cluster(features, temp_config)
    
    return labels, centroids, optimal_k


def hdbscan_cluster(
    features: np.ndarray,
    config: ClusterConfig
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    使用HDBSCAN自动确定聚类数
    
    HDBSCAN不需要预先指定聚类数
    """
    try:
        import hdbscan
    except ImportError:
        from .utils import LOGGER
        LOGGER.warning("HDBSCAN not available, falling back to adaptive KMeans")
        return adaptive_kmeans_cluster(features, config)
    
    # HDBSCAN聚类
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(10, len(features) // 100),
        min_samples=5,
        cluster_selection_epsilon=0.0,
        metric='euclidean',
        core_dist_n_jobs=-1
    )
    
    labels = clusterer.fit_predict(features)
    
    # HDBSCAN可能产生噪声点（label=-1），处理它们
    if -1 in labels:
        # 将噪声点分配到最近的cluster
        labels = handle_noise_points(labels, features, method="nearest")
    
    # 计算质心
    unique_labels = np.unique(labels)
    optimal_k = len(unique_labels)
    
    centroids = np.zeros((optimal_k, features.shape[1]))
    for i, label in enumerate(unique_labels):
        cluster_points = features[labels == label]
        centroids[i] = cluster_points.mean(axis=0)
    
    # 重新映射标签为0-k
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_mapping[l] for l in labels])
    
    return labels, centroids, optimal_k


def meanshift_cluster(
    features: np.ndarray,
    config: ClusterConfig
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    使用MeanShift自动确定聚类数
    
    MeanShift不需要预先指定聚类数
    """
    try:
        from sklearn.cluster import MeanShift, estimate_bandwidth
    except ImportError:
        from .utils import LOGGER
        LOGGER.warning("MeanShift not available, falling back to adaptive KMeans")
        return adaptive_kmeans_cluster(features, config)
    
    # 估计bandwidth
    bandwidth = estimate_bandwidth(
        features,
        quantile=0.2,
        n_samples=min(500, len(features))
    )
    
    # 如果bandwidth太小，使用默认值
    if bandwidth < 1e-6:
        bandwidth = None
    
    # MeanShift聚类
    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        min_bin_freq=5
    )
    
    labels = ms.fit_predict(features)
    centroids = ms.cluster_centers_
    optimal_k = len(centroids)
    
    return labels, centroids, optimal_k


def smart_cluster(
    features: np.ndarray,
    config: ClusterConfig
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    智能聚类：根据配置选择合适的聚类方法
    
    Returns:
        (labels, centroids, num_clusters)
    """
    from .utils import LOGGER
    
    if config.clustering_method == "fixed":
        # 固定聚类数
        labels, centroids = kmeans_cluster(features, config)
        optimal_k = config.num_clusters
        LOGGER.info(f"Fixed clustering: k={optimal_k}")
        
    elif config.clustering_method == "adaptive":
        # 自适应KMeans
        labels, centroids, optimal_k = adaptive_kmeans_cluster(features, config)
        LOGGER.info(f"Adaptive clustering: optimal k={optimal_k} (range: {config.min_clusters}-{config.max_clusters})")
        
    elif config.clustering_method == "hdbscan":
        # HDBSCAN
        labels, centroids, optimal_k = hdbscan_cluster(features, config)
        LOGGER.info(f"HDBSCAN clustering: found k={optimal_k} clusters")
        
    elif config.clustering_method == "meanshift":
        # MeanShift
        labels, centroids, optimal_k = meanshift_cluster(features, config)
        LOGGER.info(f"MeanShift clustering: found k={optimal_k} clusters")
        
    else:
        LOGGER.warning(f"Unknown clustering method: {config.clustering_method}, using adaptive")
        labels, centroids, optimal_k = adaptive_kmeans_cluster(features, config)
    
    return labels, centroids, optimal_k


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


def _extract_connected_components(
    mask: np.ndarray,
    min_area: int = 200
) -> List[Tuple[np.ndarray, int]]:
    """
    将mask分解为连通域
    
    Returns:
        List of (component_mask, area) tuples
    """
    try:
        from scipy.ndimage import label as connected_components
    except ImportError:
        # 如果没有scipy，返回原mask
        return [(mask, int(mask.sum()))]
    
    labeled, num_components = connected_components(mask)
    
    components = []
    for component_id in range(1, num_components + 1):
        component_mask = (labeled == component_id).astype(np.uint8)
        area = int(component_mask.sum())
        
        if area >= min_area:
            components.append((component_mask, area))
    
    # 按面积降序排序
    components.sort(key=lambda x: x[1], reverse=True)
    
    return components


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
    objectness_map: Optional[np.ndarray] = None,
) -> List[RegionProposal]:
    """
    将聚类标签转换为候选区域，支持连通域分离
    """
    height, width = image_shape
    proposals: List[RegionProposal] = []
    upscale = _resize_label_map(label_map, width, height)
    
    # 上采样对象性图
    objectness_upscale = None
    if objectness_map is not None:
        objectness_pil = Image.fromarray((objectness_map * 255).astype(np.uint8), mode="L")
        objectness_resized = objectness_pil.resize((width, height), resample=Image.BILINEAR)
        objectness_upscale = np.array(objectness_resized, dtype=np.float32) / 255.0
    
    for label in np.unique(upscale):
        cluster_mask = upscale == int(label)
        
        # 连通域分析
        if config.use_connected_components:
            components = _extract_connected_components(
                cluster_mask,
                min_area=config.min_component_area
            )
        else:
            # 不分离连通域，整体作为一个候选
            area = int(cluster_mask.sum())
            if area >= config.min_region_area:
                components = [(cluster_mask, area)]
            else:
                components = []
        
        # 为每个连通域创建proposal
        for component_id, (mask_bool, area) in enumerate(components):
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
            
            # 计算对象性评分
            objectness_score = 0.0
            if objectness_upscale is not None:
                region_objectness = objectness_upscale[mask_bool]
                objectness_score = float(region_objectness.mean())
                
                # 对象性过滤
                if config.use_objectness_filter and objectness_score < config.objectness_threshold:
                    continue
            
            # 综合评分
            if objectness_upscale is not None:
                combined_score = (
                    (1 - config.objectness_weight) * float(area) +
                    config.objectness_weight * objectness_score * 10000
                )
            else:
                combined_score = float(area)
            
            # 记录patch坐标
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
                    component_id=component_id,
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
                    component_id=proposal.component_id,
                )
            )

    return expanded


def _generate_grid_points(
    bbox: Tuple[int, int, int, int],
    mask: np.ndarray,
    points_per_side: int = 3
) -> List[Tuple[int, int]]:
    """
    在bbox内生成网格点，只保留mask内的点
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    
    points = []
    for i in range(points_per_side):
        for j in range(points_per_side):
            # 在grid cell中心采样
            x = x0 + int((j + 0.5) * width / points_per_side)
            y = y0 + int((i + 0.5) * height / points_per_side)
            
            # 检查是否在mask内
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:
                    points.append((x, y))
    
    return points


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
    
    elif config.point_strategy == "grid":
        # 网格点策略
        return _generate_grid_points(
            proposal.bbox,
            proposal.mask,
            points_per_side=config.grid_points_per_side
        )
    
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