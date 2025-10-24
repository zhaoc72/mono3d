"""Zero-shot segmentation pipeline combining DINOv3 proposals with SAM2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import cv2
import numpy as np
import torch

from .dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from .prompt_generator import (
    ClusterConfig,
    PromptConfig,
    RegionProposal,
    kmeans_cluster,
    labels_to_regions,
    expand_region_instances,
    proposals_to_prompts,
    smart_cluster,
)
from .sam2_segmenter import SAM2Segmenter, Sam2Config
from .mask_postprocess import nms_with_objectness, filter_by_combined_score
from .superpixel_helper import SuperpixelGenerator, SuperpixelConfig
from .graph_clustering import GraphClusterer, GraphClusterConfig, merge_small_clusters
from .density_clustering import DensityClusterer, DensityClusterConfig, handle_noise_points
from .crf_refinement import CRFRefiner, CRFConfig, bilateral_filter_mask
from .utils import LOGGER



@dataclass
class AutoSuperpixelConfig:
    """Heuristics for automatically enabling superpixel-guided clustering."""

    enable: bool = False
    min_proposals: int = 6
    min_clusters: int = 6
    max_mean_area_ratio: float = 0.25
    max_single_area_ratio: float = 0.6
    log_top_k: int = 5


@dataclass
@dataclass
class ProposalRefineConfig:
    """Morphological refinement applied to semantic proposals before prompting."""

    enable: bool = False
    closing_kernel: int = 5
    opening_kernel: int = 3
    blur_sigma: float = 1.2
    blur_threshold: float = 0.4
    min_keep_area: int = 120


@dataclass
class PipelineConfig:
    """High level knobs for the zero-shot segmentation routine."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    # 新增：高级聚类选项
    use_superpixels: bool = False
    superpixel: SuperpixelConfig = field(default_factory=SuperpixelConfig)
    auto_superpixel: AutoSuperpixelConfig = field(default_factory=AutoSuperpixelConfig)

    use_graph_clustering: bool = False
    graph_cluster: GraphClusterConfig = field(default_factory=GraphClusterConfig)

    use_density_clustering: bool = False
    density_cluster: DensityClusterConfig = field(default_factory=DensityClusterConfig)

    # CRF 细化
    crf: CRFConfig = field(default_factory=CRFConfig)

    # 语义区域平滑
    proposal_refine: ProposalRefineConfig = field(default_factory=ProposalRefineConfig)


@dataclass
class PipelineResult:
    """Rich return type that exposes intermediate artefacts for inspection."""

    masks: List[np.ndarray]
    proposals: List[RegionProposal]
    label_map: np.ndarray
    attention_map: Optional[np.ndarray]
    objectness_map: Optional[np.ndarray]
    prompts: Dict[str, List]
    cluster_centroids: Optional[np.ndarray]
    superpixel_labels: Optional[np.ndarray] = None  # 新增
    prompt_descriptions: List[Dict[str, Any]] = field(default_factory=list)


class ZeroShotSegmentationPipeline:
    """Orchestrates DINOv3 feature extraction, clustering and SAM2 inference."""

    def __init__(
        self,
        dinov3_config: Dinov3Config,
        sam2_config: Sam2Config,
        pipeline_config: PipelineConfig,
        device: str = "cuda",
        dtype: torch.dtype | str = torch.float32,
        extractor: Optional[DINOv3FeatureExtractor] = None,
        segmenter: Optional[SAM2Segmenter] = None,
    ) -> None:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.extractor = extractor or DINOv3FeatureExtractor(dinov3_config, device, torch_dtype)
        self.segmenter = segmenter or SAM2Segmenter(sam2_config, device, torch_dtype)
        self.config = pipeline_config
        self.auto_superpixel_cfg = self.config.auto_superpixel

        # 初始化可选组件
        self.superpixel_gen = None
        if self.config.use_superpixels or self.auto_superpixel_cfg.enable:
            self.superpixel_gen = SuperpixelGenerator(self.config.superpixel)

        self.graph_clusterer = None
        if self.config.use_graph_clustering:
            self.graph_clusterer = GraphClusterer(self.config.graph_cluster)
        
        self.density_clusterer = None
        if self.config.use_density_clustering:
            self.density_clusterer = DensityClusterer(self.config.density_cluster)

        self.crf_refiner = CRFRefiner(self.config.crf)
        self.proposal_refine_cfg = self.config.proposal_refine

    @staticmethod
    def _build_kernel(size: int) -> Optional[np.ndarray]:
        """Create an odd-sized elliptical kernel for morphology operations."""

        if size <= 1:
            return None
        size = int(size)
        if size % 2 == 0:
            size += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        """Filter out connected components smaller than ``min_area``."""

        if min_area <= 0:
            return (mask > 0).astype(np.uint8)

        mask_uint8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        if num_labels <= 1:
            return mask_uint8

        kept = np.zeros_like(mask_uint8)
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                kept[labels == label_idx] = 1
        return kept

    @staticmethod
    def _gaussian_kernel_size(sigma: float) -> int:
        if sigma <= 0:
            return 0
        radius = max(1, int(np.ceil(3 * float(sigma))))
        return 2 * radius + 1

    def _refine_region_proposals(
        self,
        proposals: Sequence[RegionProposal],
    ) -> List[RegionProposal]:
        """Smooth proposal masks to reduce blocky edges before prompting."""

        cfg = self.proposal_refine_cfg
        if not cfg.enable or not proposals:
            return list(proposals)

        closing_kernel = self._build_kernel(cfg.closing_kernel)
        opening_kernel = self._build_kernel(cfg.opening_kernel)
        gaussian_kernel = self._gaussian_kernel_size(cfg.blur_sigma)

        refined: List[RegionProposal] = []
        for proposal in proposals:
            original_mask = proposal.mask.astype(np.uint8)
            mask = original_mask.copy()

            if closing_kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
            if opening_kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)

            if gaussian_kernel > 0:
                blurred = cv2.GaussianBlur(mask.astype(np.float32), (gaussian_kernel, gaussian_kernel), cfg.blur_sigma)
                mask = (blurred >= float(cfg.blur_threshold)).astype(np.uint8)

            if cfg.min_keep_area > 0:
                mask = self._remove_small_components(mask, cfg.min_keep_area)

            area = int(mask.sum())
            if area == 0:
                continue

            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue

            original_area = max(int(original_mask.sum()), 1)
            area_scale = area / float(original_area)
            new_score = float(proposal.score) * area_scale

            refined.append(
                RegionProposal(
                    label=proposal.label,
                    mask=mask.astype(np.uint8),
                    bbox=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                    centroid=(int(np.round(xs.mean())), int(np.round(ys.mean()))),
                    score=new_score,
                    objectness=proposal.objectness,
                    patch_coords=proposal.patch_coords,
                    seed_points=list(proposal.seed_points),
                    semantic_label=proposal.semantic_label,
                    instance_id=proposal.instance_id,
                    component_id=proposal.component_id,
                )
            )

        if not refined:
            LOGGER.warning("Proposal refinement removed all regions; falling back to raw proposals")
            return list(proposals)

        refined.sort(key=lambda item: item.score, reverse=True)
        if self.config.cluster.max_regions > 0:
            refined = refined[: self.config.cluster.max_regions]

        LOGGER.info(
            "Proposal refinement smoothed %d regions -> %d (closing=%d, opening=%d, sigma=%.2f)",
            len(proposals),
            len(refined),
            cfg.closing_kernel,
            cfg.opening_kernel,
            cfg.blur_sigma,
        )

        return refined

    def _ensure_superpixel_generator(self) -> bool:
        """Lazy initialisation of the optional superpixel generator."""
        if self.superpixel_gen is not None:
            return True
        if not (self.config.use_superpixels or self.auto_superpixel_cfg.enable):
            return False
        try:
            self.superpixel_gen = SuperpixelGenerator(self.config.superpixel)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to initialise superpixel generator: %s", exc)
            self.superpixel_gen = None
            return False
        return True

    @staticmethod
    def _reindex_labels(labels: np.ndarray) -> np.ndarray:
        """Remap non-negative labels to a compact 0..N-1 range while preserving background."""

        positive = [int(label) for label in np.unique(labels) if int(label) >= 0]
        if not positive:
            return labels

        mapping = {label: idx for idx, label in enumerate(sorted(positive))}
        remapped = labels.copy()
        for label, idx in mapping.items():
            remapped[labels == label] = idx
        return remapped

    def _objectness_threshold(self) -> float:
        """Resolve the threshold used for foreground masking."""

        cfg = self.config.cluster
        threshold = getattr(cfg, "objectness_mask_threshold", None)
        if threshold is None:
            threshold = cfg.objectness_threshold
        return float(threshold)

    def _build_patch_objectness_mask(
        self,
        objectness_map: Optional[np.ndarray],
        patch_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Construct a foreground mask on the patch grid using DINO objectness."""

        cfg = self.config.cluster
        if objectness_map is None or not getattr(cfg, "apply_objectness_mask", False):
            return None

        if objectness_map.shape != patch_shape:
            resized = cv2.resize(
                objectness_map.astype(np.float32),
                (patch_shape[1], patch_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            objectness = np.clip(resized, 0.0, 1.0)
        else:
            objectness = np.clip(objectness_map.astype(np.float32), 0.0, 1.0)

        threshold = self._objectness_threshold()
        mask = objectness >= threshold

        min_keep = int(getattr(cfg, "objectness_min_keep_patches", 0))
        min_keep = max(1, min_keep) if mask.size > 0 else 0

        keep_count = int(mask.sum())
        if keep_count < min_keep and mask.size > 0:
            flat = objectness.reshape(-1)
            top_k = min(min_keep, flat.size)
            if top_k <= 0:
                return None
            indices = np.argpartition(flat, flat.size - top_k)[-top_k:]
            new_mask = np.zeros_like(flat, dtype=bool)
            new_mask[indices] = True
            mask = new_mask.reshape(objectness.shape)

        if not mask.any():
            LOGGER.warning("Objectness mask suppressed all patches; falling back to full grid")
            return None

        return mask

    @staticmethod
    def _compute_label_adjacency(label_map: np.ndarray) -> Dict[int, Set[int]]:
        """Construct an adjacency graph between neighbouring clusters."""

        adjacency: Dict[int, Set[int]] = {int(label): set() for label in np.unique(label_map)}
        if label_map.size == 0:
            return adjacency

        # vertical neighbours
        if label_map.shape[0] > 1:
            top = label_map[:-1, :]
            bottom = label_map[1:, :]
            diff = top != bottom
            if np.any(diff):
                a = top[diff].ravel()
                b = bottom[diff].ravel()
                for l1, l2 in zip(a.tolist(), b.tolist()):
                    adjacency[int(l1)].add(int(l2))
                    adjacency[int(l2)].add(int(l1))

        # horizontal neighbours
        if label_map.shape[1] > 1:
            left = label_map[:, :-1]
            right = label_map[:, 1:]
            diff = left != right
            if np.any(diff):
                a = left[diff].ravel()
                b = right[diff].ravel()
                for l1, l2 in zip(a.tolist(), b.tolist()):
                    adjacency[int(l1)].add(int(l2))
                    adjacency[int(l2)].add(int(l1))

        return adjacency

    def _merge_similar_adjacent(
        self,
        flat_labels: np.ndarray,
        grid_shape: Tuple[int, int],
        patch_map: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Merge adjacent clusters whose mean features are highly similar."""

        cfg = self.config.cluster
        valid_mask = flat_labels >= 0
        valid_labels = (
            np.unique(flat_labels[valid_mask]) if np.any(valid_mask) else np.array([], dtype=np.int32)
        )
        if valid_labels.size <= 1:
            return flat_labels, False

        features = patch_map.reshape(-1, patch_map.shape[-1])
        counts = {int(label): int(np.count_nonzero(flat_labels == label)) for label in valid_labels}
        centroids: Dict[int, np.ndarray] = {}
        for label in valid_labels:
            mask = flat_labels == label
            if not np.any(mask):
                continue
            centroid = features[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[int(label)] = centroid

        adjacency = self._compute_label_adjacency(flat_labels.reshape(grid_shape))

        parent: Dict[int, int] = {int(label): int(label) for label in valid_labels}

        def find(label: int) -> int:
            while parent[label] != label:
                parent[label] = parent[parent[label]]
                label = parent[label]
            return label

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a == root_b:
                return
            parent[root_b] = root_a

        merged = False
        threshold = max(0.0, min(1.0, cfg.similarity_threshold))
        min_size = max(0, cfg.min_similarity_cluster_size)

        for label, neighbours in adjacency.items():
            if label < 0:
                continue
            centroid_a = centroids.get(label)
            if centroid_a is None:
                continue
            for neighbour in neighbours:
                if neighbour < 0:
                    continue
                if neighbour <= label:
                    continue
                centroid_b = centroids.get(neighbour)
                if centroid_b is None:
                    continue
                similarity = float(np.dot(centroid_a, centroid_b))
                if similarity < threshold:
                    continue
                if min_size > 0 and counts[label] > min_size and counts[neighbour] > min_size:
                    continue
                union(label, neighbour)
                merged = True

        if not merged:
            return flat_labels, False

        new_labels = flat_labels.copy()
        for label in valid_labels:
            root = find(int(label))
            if root != label:
                new_labels[flat_labels == label] = root

        new_labels = self._reindex_labels(new_labels)
        return new_labels, True

    def _compute_centroids(
        self,
        label_map: np.ndarray,
        patch_map: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Recompute cluster centroids after any label remapping."""

        flat_labels = label_map.reshape(-1)
        features = patch_map.reshape(-1, patch_map.shape[-1])
        unique = [int(label) for label in np.unique(flat_labels) if int(label) >= 0]
        centroids: List[np.ndarray] = []
        for label in unique:
            mask = flat_labels == label
            if not np.any(mask):
                continue
            centroid = features[mask].mean(axis=0)
            centroids.append(centroid.astype(np.float32))

        if centroids:
            centroid_array = np.stack(centroids, axis=0)
        else:
            centroid_array = np.zeros((0, features.shape[-1]), dtype=np.float32)

        return centroid_array, len(unique)

    def _post_process_label_map(
        self,
        label_map: np.ndarray,
        patch_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Apply optional cluster merging heuristics and recompute centroids."""

        cfg = self.config.cluster
        modified = False
        current_map = label_map

        flat = current_map.reshape(-1)
        features = patch_map.reshape(-1, patch_map.shape[-1])
        valid_mask = flat >= 0

        if cfg.merge_small_clusters and cfg.min_cluster_size > 0 and np.any(valid_mask):
            merged_valid = merge_small_clusters(
                flat[valid_mask],
                features[valid_mask],
                cfg.min_cluster_size,
            )
            if not np.array_equal(merged_valid, flat[valid_mask]):
                flat = flat.copy()
                flat[valid_mask] = merged_valid
                current_map = flat.reshape(current_map.shape)
                current_map = self._reindex_labels(current_map)
                modified = True
                valid_mask = current_map.reshape(-1) >= 0

        if cfg.merge_similar_clusters and current_map.size > 0 and np.any(valid_mask):
            flat = current_map.reshape(-1)
            merged_flat, changed = self._merge_similar_adjacent(flat, current_map.shape, patch_map)
            if changed:
                current_map = merged_flat.reshape(current_map.shape)
                valid_mask = current_map.reshape(-1) >= 0
                modified = True

        centroids, num_clusters = self._compute_centroids(current_map, patch_map)
        return current_map if modified else label_map, centroids, num_clusters

    def _cluster_basic(
        self,
        patch_map: np.ndarray,
        objectness_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """基础聚类（支持自适应）"""
        grid_h, grid_w, _ = patch_map.shape
        features = patch_map.reshape(-1, patch_map.shape[-1])

        labels: np.ndarray
        centroids: np.ndarray
        num_clusters: int

        if objectness_mask is not None:
            mask_flat = objectness_mask.reshape(-1)
            keep_indices = np.flatnonzero(mask_flat)
            if keep_indices.size > 0:
                LOGGER.info(
                    "Applying objectness foreground mask: keeping %d/%d patches",
                    keep_indices.size,
                    mask_flat.size,
                )
                fg_features = features[keep_indices]
                labels_fg, centroids, num_clusters = smart_cluster(fg_features, self.config.cluster)
                full_labels = np.full(features.shape[0], -1, dtype=np.int32)
                full_labels[keep_indices] = labels_fg
                labels = full_labels
            else:
                LOGGER.warning(
                    "Objectness mask kept no patches; clustering the full grid instead"
                )
                labels, centroids, num_clusters = smart_cluster(features, self.config.cluster)
        else:
            # 使用智能聚类
            labels, centroids, num_clusters = smart_cluster(features, self.config.cluster)

        label_map = labels.reshape(grid_h, grid_w)
        label_map_post, centroids_post, num_clusters_post = self._post_process_label_map(
            label_map,
            patch_map,
        )

        if label_map_post is label_map:
            valid_clusters = [int(label) for label in np.unique(label_map) if int(label) >= 0]
            return label_map, centroids, len(valid_clusters)
        return label_map_post, centroids_post, num_clusters_post
    
    def _cluster_with_superpixels(
        self,
        image: np.ndarray,
        patch_map: np.ndarray,
        objectness_map: Optional[np.ndarray] = None,
        patch_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用超像素辅助的聚类"""
        # 生成超像素
        superpixel_labels = self.superpixel_gen.generate_superpixels(image)

        # 将 patch 特征聚合到超像素
        patch_features = patch_map.reshape(-1, patch_map.shape[-1])
        superpixel_features, superpixel_ids = self.superpixel_gen.aggregate_features_by_superpixels(
            patch_features,
            superpixel_labels,
            image.shape[:2]
        )

        valid_superpixels: Optional[np.ndarray] = None
        if getattr(self.config.cluster, "apply_objectness_mask", False):
            height, width = image.shape[:2]
            scores = None
            if objectness_map is not None:
                resized = cv2.resize(
                    objectness_map.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized = np.clip(resized, 0.0, 1.0)
                scores = np.zeros(len(superpixel_ids), dtype=np.float32)
                for idx, sp_id in enumerate(superpixel_ids):
                    mask = superpixel_labels == sp_id
                    scores[idx] = float(resized[mask].mean())
            elif patch_mask is not None:
                resized_mask = cv2.resize(
                    patch_mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                scores = np.zeros(len(superpixel_ids), dtype=np.float32)
                for idx, sp_id in enumerate(superpixel_ids):
                    mask = superpixel_labels == sp_id
                    scores[idx] = float(resized_mask[mask].mean())

            if scores is not None and len(scores) > 0:
                threshold = self._objectness_threshold()
                valid_superpixels = scores >= threshold

                total_patches = float(patch_map.shape[0] * patch_map.shape[1])
                min_keep_patches = float(getattr(self.config.cluster, "objectness_min_keep_patches", 0))
                if total_patches > 0 and min_keep_patches > 0:
                    ratio = min(1.0, max(0.0, min_keep_patches / total_patches))
                    min_keep_superpixels = int(np.ceil(ratio * len(superpixel_ids)))
                else:
                    min_keep_superpixels = 0
                min_keep_superpixels = min(len(superpixel_ids), max(1, min_keep_superpixels)) if len(superpixel_ids) > 0 else 0

                keep_count = int(valid_superpixels.sum()) if valid_superpixels is not None else 0
                if valid_superpixels is not None and keep_count < min_keep_superpixels and len(superpixel_ids) > 0:
                    LOGGER.info(
                        "Objectness kept %d/%d superpixels (<%d); boosting by top scores",
                        keep_count,
                        len(superpixel_ids),
                        min_keep_superpixels,
                    )
                    top_k = min(len(superpixel_ids), max(1, min_keep_superpixels))
                    top_indices = np.argsort(scores)[-top_k:]
                    boosted = np.zeros_like(valid_superpixels)
                    boosted[top_indices] = True
                    valid_superpixels = boosted

                if valid_superpixels is not None and not valid_superpixels.any():
                    LOGGER.warning(
                        "Objectness mask removed all superpixels; clustering without filtering"
                    )
                    valid_superpixels = None

        if self.config.use_graph_clustering and valid_superpixels is not None:
            LOGGER.warning(
                "Graph clustering does not support foreground masking; ignoring objectness filter for this path"
            )
            valid_superpixels = None

        # 对超像素特征聚类
        if self.config.use_graph_clustering:
            # 构建超像素邻接图
            edges, edge_weights = self.superpixel_gen.create_adjacency_graph(superpixel_labels)

            # 图聚类
            cluster_input = superpixel_features
            if valid_superpixels is not None:
                cluster_input = superpixel_features[valid_superpixels]
                edges = edges  # graph clustering currently expects full graph; skip filtering
                LOGGER.warning(
                    "Graph clustering currently ignores objectness mask; consider disabling graph clustering"
                )
            cluster_labels = self.graph_clusterer.cluster(
                cluster_input,
                edges,
                edge_weights
            )
            if valid_superpixels is not None and valid_superpixels.any():
                full_labels = np.full(len(superpixel_ids), -1, dtype=np.int32)
                full_labels[valid_superpixels] = cluster_labels
                cluster_labels = full_labels
        else:
            if valid_superpixels is not None and valid_superpixels.any():
                LOGGER.info(
                    "Applying objectness foreground mask: keeping %d/%d superpixels",
                    int(valid_superpixels.sum()),
                    len(superpixel_ids),
                )
                features_for_clustering = superpixel_features[valid_superpixels]
            else:
                features_for_clustering = superpixel_features

            if features_for_clustering.size == 0:
                LOGGER.warning("No superpixel features available for clustering; using zeros")
                cluster_labels = np.full(len(superpixel_ids), -1, dtype=np.int32)
            elif self.config.use_density_clustering:
                # 密度聚类
                cluster_subset = self.density_clusterer.cluster(features_for_clustering)
                cluster_subset = handle_noise_points(cluster_subset, features_for_clustering)
                cluster_labels = np.full(len(superpixel_ids), -1, dtype=np.int32)
                if valid_superpixels is not None and valid_superpixels.any():
                    cluster_labels[valid_superpixels] = cluster_subset
                else:
                    cluster_labels = cluster_subset
            else:
                # 使用智能聚类
                cluster_subset, _, _ = smart_cluster(features_for_clustering, self.config.cluster)
                cluster_labels = np.full(len(superpixel_ids), -1, dtype=np.int32)
                if valid_superpixels is not None and valid_superpixels.any():
                    cluster_labels[valid_superpixels] = cluster_subset
                else:
                    cluster_labels = cluster_subset

        # 将聚类标签映射回像素级
        height, width = image.shape[:2]
        pixel_labels = np.full((height, width), -1, dtype=np.int32)

        for sp_id, cluster_id in zip(superpixel_ids, cluster_labels):
            if int(cluster_id) < 0:
                continue
            pixel_labels[superpixel_labels == sp_id] = cluster_id

        # 下采样到 patch grid 用于后续处理
        grid_h, grid_w, _ = patch_map.shape
        from PIL import Image
        label_pil = Image.fromarray(pixel_labels.astype(np.int32), mode="I")
        label_map = np.array(
            label_pil.resize((grid_w, grid_h), resample=Image.NEAREST), dtype=np.int32
        )

        label_map_post, centroids_post, num_clusters = self._post_process_label_map(
            label_map,
            patch_map,
        )

        return label_map_post, centroids_post, superpixel_labels

    def _should_refine_with_superpixels(
        self,
        proposals: List[RegionProposal],
        num_clusters: int,
        image_shape: Tuple[int, int],
    ) -> bool:
        """Decide whether automatic superpixel refinement should kick in."""
        cfg = self.auto_superpixel_cfg
        if not cfg.enable:
            return False

        if not proposals:
            LOGGER.info("Auto superpixel check: no proposals available -> trigger refine")
            return True

        height, width = image_shape
        image_area = max(1, height * width)
        areas = np.array([float(p.mask.sum()) if p.mask is not None else 0.0 for p in proposals])
        mean_area_ratio = float(areas.mean() / image_area)
        max_area_ratio = float(areas.max() / image_area)

        triggers = {
            "proposal_count": len(proposals) < cfg.min_proposals,
            "cluster_count": num_clusters < cfg.min_clusters,
            "mean_area": mean_area_ratio > cfg.max_mean_area_ratio,
            "max_area": max_area_ratio > cfg.max_single_area_ratio,
        }

        if any(triggers.values()):
            trigger_reasons = ", ".join(key for key, value in triggers.items() if value)
            LOGGER.info(
                "Auto superpixel refine triggered due to: %s (proposals=%d, clusters=%d, "
                "mean_area_ratio=%.3f, max_area_ratio=%.3f)",
                trigger_reasons,
                len(proposals),
                num_clusters,
                mean_area_ratio,
                max_area_ratio,
            )
            return True

        return False

    @staticmethod
    def _describe_prompts(
        proposals: Sequence[RegionProposal],
        boxes: Sequence[Sequence[int]],
        points: Sequence[Sequence[Tuple[int, int]]],
        labels: Sequence[Sequence[int]],
        mask_inputs: Optional[Sequence[np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """Create a structured summary of prompts for downstream inspection."""

        descriptions: List[Dict[str, Any]] = []
        for idx, proposal in enumerate(proposals):
            box = list(boxes[idx]) if idx < len(boxes) else None
            pts = list(points[idx]) if idx < len(points) else []
            lbls = list(labels[idx]) if idx < len(labels) else []
            mask_prompt = None
            if mask_inputs is not None and idx < len(mask_inputs):
                mask_prompt = mask_inputs[idx]
            mask_strength = float(mask_prompt.sum()) if mask_prompt is not None else 0.0
            area = int(proposal.mask.sum()) if proposal.mask is not None else 0
            descriptions.append(
                {
                    "proposal_index": idx,
                    "cluster_label": proposal.label,
                    "component_id": proposal.component_id,
                    "bbox": box,
                    "positive_points": pts,
                    "labels": lbls,
                    "positive_count": len(pts),
                    "objectness": float(proposal.objectness),
                    "area": area,
                    "mask_prompt_strength": mask_strength,
                    "score": float(proposal.score),
                }
            )
        return descriptions

    def _log_prompt_descriptions(
        self,
        descriptions: List[Dict[str, Any]],
        stage: str,
    ) -> None:
        """Emit concise logs describing the prompts passed to SAM2."""

        if not descriptions:
            LOGGER.info("No prompts generated at %s stage", stage)
            return

        top_k_source = (
            self.config.prompt.log_top_k
            if self.config.prompt.log_top_k > 0
            else self.auto_superpixel_cfg.log_top_k
        )
        top_k = min(len(descriptions), top_k_source)
        for idx in range(top_k):
            entry = descriptions[idx]
            LOGGER.info(
                "%s prompt #%d: box=%s, +points=%s (count=%d, cluster=%s, component=%s, "
                "objectness=%.3f, area=%d, mask_strength=%.1f)",
                stage.title(),
                idx,
                entry.get("bbox"),
                entry.get("positive_points"),
                entry.get("positive_count", 0),
                entry.get("cluster_label"),
                entry.get("component_id"),
                entry.get("objectness", 0.0),
                entry.get("area", 0),
                entry.get("mask_prompt_strength", 0.0),
            )

        if len(descriptions) > top_k:
            LOGGER.info(
                "... %d additional prompts omitted from %s stage logs ...",
                len(descriptions) - top_k,
                stage,
            )

    @staticmethod
    def _resize_guidance_map(
        guidance: Optional[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if guidance is None:
            return None
        height, width = image_shape
        guidance = np.asarray(guidance, dtype=np.float32)
        if guidance.ndim != 2:
            return None
        resized = cv2.resize(guidance, (width, height), interpolation=cv2.INTER_CUBIC)
        min_val = float(resized.min())
        max_val = float(resized.max())
        if max_val > min_val:
            resized = (resized - min_val) / (max_val - min_val)
        else:
            resized = np.zeros_like(resized, dtype=np.float32)
        return resized

    def _prepare_mask_prompts(
        self,
        proposals: Sequence[RegionProposal],
        prompt_config: PromptConfig,
        image_shape: Tuple[int, int],
        attention_map: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        mask_inputs: List[np.ndarray] = []
        if not proposals:
            return mask_inputs

        height, width = image_shape
        attention_resized = attention_map
        if prompt_config.include_heatmaps and attention_map is not None:
            attention_resized = self._resize_guidance_map(attention_map, image_shape)

        for proposal in proposals:
            if proposal.mask is None:
                mask_inputs.append(np.zeros((height, width), dtype=np.float32))
                continue
            mask = proposal.mask.astype(np.float32)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            combined = mask
            if prompt_config.include_heatmaps and attention_resized is not None:
                region_heatmap = attention_resized * (mask > 0)
                if region_heatmap.max() > 0:
                    region_heatmap = region_heatmap / (region_heatmap.max() + 1e-6)
                weight = np.clip(prompt_config.heatmap_weight, 0.0, 1.0)
                combined = (1.0 - weight) * combined + weight * region_heatmap

            if prompt_config.mask_gaussian_sigma > 0.0:
                sigma = float(prompt_config.mask_gaussian_sigma)
                kernel = max(1, int(4 * sigma + 1))
                if kernel % 2 == 0:
                    kernel += 1
                combined = cv2.GaussianBlur(combined, (kernel, kernel), sigma)

            max_val = combined.max()
            if max_val > 0:
                combined = combined / max_val
            mask_inputs.append(combined.astype(np.float32))

        return mask_inputs

    def _apply_nms(
        self,
        masks: List[np.ndarray],
        proposals: List[RegionProposal],
        nms_config: Dict
    ) -> Tuple[List[np.ndarray], List[RegionProposal]]:
        """应用 NMS 去重，同时根据对象性/面积过滤背景掩码"""
        if not masks or not proposals:
            return masks, proposals

        objectness_scores = [p.objectness for p in proposals]
        min_obj = float(nms_config.get("min_mask_objectness", 0.0) or 0.0)
        min_area = int(nms_config.get("min_mask_area", 0) or 0)

        if min_obj > 0.0 or min_area > 0:
            filtered_masks, keep_indices = filter_by_combined_score(
                masks,
                objectness_scores,
                min_objectness=min_obj,
                min_area=min_area,
            )

            if not keep_indices:
                LOGGER.warning(
                    "Mask filtering removed all proposals (min_obj=%.2f, min_area=%d)",
                    min_obj,
                    min_area,
                )
                return [], []

            if len(keep_indices) != len(masks):
                LOGGER.info(
                    "Filtered masks by foreground constraints: %d -> %d", len(masks), len(keep_indices)
                )

            masks = filtered_masks
            proposals = [proposals[i] for i in keep_indices]
            objectness_scores = [objectness_scores[i] for i in keep_indices]

        if not nms_config.get("enable_nms", False):
            return masks, proposals

        keep_masks, keep_indices = nms_with_objectness(
            masks,
            objectness_scores,
            sam2_confidence=None,
            iou_threshold=nms_config.get("iou_threshold", 0.6),
            objectness_weight=nms_config.get("objectness_weight", 0.5),
            confidence_weight=nms_config.get("confidence_weight", 0.3),
            area_weight=nms_config.get("area_weight", 0.2)
        )

        keep_proposals = [proposals[i] for i in keep_indices]

        LOGGER.info(f"NMS: {len(masks)} masks -> {len(keep_masks)} masks")

        return keep_masks, keep_proposals
    
    def _refine_masks(
        self,
        image: np.ndarray,
        masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """细化掩码边界"""
        if self.config.crf.enable:
            # CRF 细化
            refined = self.crf_refiner.refine_masks_batch(image, masks)
            LOGGER.info("Applied CRF refinement")
        else:
            # 简单的双边滤波
            refined = []
            for mask in masks:
                refined_mask = bilateral_filter_mask(mask, image)
                refined.append(refined_mask)
        
        return refined

    def run(
        self, 
        image: np.ndarray, 
        nms_config: Optional[Dict] = None
    ) -> PipelineResult:
        """
        运行完整的分割 pipeline
        """
        # 提取多层融合特征
        LOGGER.info("Extracting DINOv3 features...")
        feats = self.extractor.extract_features(image)
        patch_map = feats["patch_map"]
        if hasattr(patch_map, "detach"):
            patch_map = patch_map.detach().cpu().numpy()

        attention_map = feats.get("attention_map")
        attention_resized = self._resize_guidance_map(attention_map, image.shape[:2])
        objectness_map = feats.get("objectness_map")
        patch_mask = self._build_patch_objectness_mask(objectness_map, patch_map.shape[:2])

        # 聚类
        LOGGER.info("Clustering features...")
        superpixel_labels = None

        if self.config.use_superpixels:
            label_map, centroids, superpixel_labels = self._cluster_with_superpixels(
                image,
                patch_map,
                objectness_map=objectness_map,
                patch_mask=patch_mask,
            )
            valid_labels = [int(label) for label in np.unique(label_map) if int(label) >= 0]
            num_clusters = len(valid_labels)
        else:
            label_map, centroids, num_clusters = self._cluster_basic(
                patch_map,
                objectness_mask=patch_mask,
            )

        LOGGER.info(f"Clustering result: {num_clusters} clusters")

        # 生成候选区域
        LOGGER.info("Generating region proposals...")
        semantic_proposals = labels_to_regions(
            label_map,
            image.shape[:2],
            self.config.cluster,
            objectness_map=objectness_map,
            patch_shape=patch_map.shape[:2],
        )

        # 自动超像素细化
        if (
            not self.config.use_superpixels
            and self.auto_superpixel_cfg.enable
            and self._ensure_superpixel_generator()
        ):
            try:
                should_refine = self._should_refine_with_superpixels(
                    semantic_proposals,
                    num_clusters,
                    image.shape[:2],
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Auto superpixel heuristic failed: %s", exc)
                should_refine = False

            if should_refine:
                try:
                    refined_label_map, refined_centroids, refined_superpixels = self._cluster_with_superpixels(
                        image,
                        patch_map,
                        objectness_map=objectness_map,
                        patch_mask=patch_mask,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.warning("Auto superpixel refinement failed: %s", exc)
                else:
                    refined_proposals = labels_to_regions(
                        refined_label_map,
                        image.shape[:2],
                        self.config.cluster,
                        objectness_map=objectness_map,
                        patch_shape=patch_map.shape[:2],
                    )

                    if refined_proposals:
                        LOGGER.info(
                            "Auto superpixel refinement accepted: %d -> %d proposals",
                            len(semantic_proposals),
                            len(refined_proposals),
                        )
                        label_map = refined_label_map
                        centroids = refined_centroids
                        semantic_proposals = refined_proposals
                        superpixel_labels = refined_superpixels
                        valid_labels = [
                            int(label)
                            for label in np.unique(refined_label_map)
                            if int(label) >= 0
                        ]
                        num_clusters = len(valid_labels)
                    else:
                        LOGGER.warning(
                            "Auto superpixel refinement produced no proposals; keeping original clustering"
                        )

        if semantic_proposals and self.proposal_refine_cfg.enable:
            semantic_proposals = self._refine_region_proposals(semantic_proposals)

        if not semantic_proposals:
            LOGGER.warning("No proposals survived filtering; returning empty result")
            return PipelineResult(
                [],
                [],
                label_map,
                attention_resized,
                objectness_map,
                {"boxes": [], "points": [], "labels": [], "mask_inputs": []},
                centroids,
                superpixel_labels,
                [],
            )

        proposals = expand_region_instances(
            semantic_proposals,
            self.config.prompt,
            self.config.cluster,
            patch_map,
            image.shape[:2],
        )

        if not proposals:
            LOGGER.warning("Instance expansion returned no proposals; falling back to semantic regions")
            proposals = semantic_proposals

        # 生成 SAM2 prompts
        LOGGER.info(f"Converting {len(proposals)} proposals to prompts...")
        boxes_raw, points_raw, labels_raw = proposals_to_prompts(
            proposals,
            self.config.prompt,
            patch_map=patch_map,
            image_shape=image.shape[:2],
            cluster_config=self.config.cluster,
        )

        mask_inputs: List[np.ndarray] = []
        if self.config.prompt.include_masks:
            mask_inputs = self._prepare_mask_prompts(
                proposals,
                self.config.prompt,
                image.shape[:2],
                attention_map if self.config.prompt.include_heatmaps else None,
            )

        boxes_for_sam = boxes_raw if self.config.prompt.include_boxes else None
        points_for_sam = points_raw if self.config.prompt.include_points else None
        labels_for_sam = labels_raw if self.config.prompt.include_points else None
        mask_inputs_for_sam = mask_inputs if self.config.prompt.include_masks else None

        initial_descriptions = self._describe_prompts(
            proposals,
            boxes_raw,
            points_raw,
            labels_raw,
            mask_inputs if mask_inputs else None,
        )
        self._log_prompt_descriptions(initial_descriptions, stage="initial")

        # SAM2 分割
        LOGGER.info("Running SAM2 segmentation...")
        masks = self.segmenter.segment_batched(
            image,
            boxes=boxes_for_sam,
            points=points_for_sam,
            labels=labels_for_sam,
            mask_inputs=mask_inputs_for_sam,
        )

        for proposal, mask in zip(proposals, masks):
            binary_mask = mask.astype(np.uint8)
            proposal.mask = binary_mask
            ys, xs = np.nonzero(binary_mask)
            if len(xs) == 0:
                continue
            proposal.bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            proposal.centroid = (int(np.round(xs.mean())), int(np.round(ys.mean())))

        # NMS 去重
        if nms_config is None:
            nms_config = {}
        
        masks, proposals = self._apply_nms(masks, proposals, nms_config)
        
        # CRF 细化（可选）
        if self.config.crf.enable and masks:
            LOGGER.info("Refining mask boundaries...")
            masks = self._refine_masks(image, masks)
        
        # 更新 prompts
        boxes_final, points_final, labels_final = proposals_to_prompts(
            proposals,
            self.config.prompt,
            patch_map=patch_map,
            image_shape=image.shape[:2],
            cluster_config=self.config.cluster,
        )

        mask_inputs_final: List[np.ndarray] = []
        if self.config.prompt.include_masks:
            mask_inputs_final = self._prepare_mask_prompts(
                proposals,
                self.config.prompt,
                image.shape[:2],
                attention_map if self.config.prompt.include_heatmaps else None,
            )

        prompts: Dict[str, List] = {
            "boxes": boxes_final if self.config.prompt.include_boxes else [],
            "points": points_final if self.config.prompt.include_points else [],
            "labels": labels_final if self.config.prompt.include_points else [],
            "mask_inputs": mask_inputs_final if self.config.prompt.include_masks else [],
        }

        if not self.config.prompt.include_boxes:
            prompts["candidate_boxes"] = boxes_final
        if not self.config.prompt.include_points:
            prompts["candidate_points"] = points_final

        final_descriptions = self._describe_prompts(
            proposals,
            boxes_final,
            points_final,
            labels_final,
            mask_inputs_final if mask_inputs_final else None,
        )
        self._log_prompt_descriptions(final_descriptions, stage="final")

        LOGGER.info(f"Pipeline complete: {len(masks)} final masks")

        return PipelineResult(
            masks,
            proposals,
            label_map,
            attention_resized,
            objectness_map,
            prompts,
            centroids,
            superpixel_labels,
            final_descriptions,
        )


def build_pipeline(
    dinov3_cfg: Dinov3Config,
    sam2_cfg: Sam2Config,
    pipeline_cfg: Optional[PipelineConfig] = None,
    device: str = "cuda",
    dtype: str = "float32",
    extractor: Optional[DINOv3FeatureExtractor] = None,
    segmenter: Optional[SAM2Segmenter] = None,
) -> ZeroShotSegmentationPipeline:
    pipeline_cfg = pipeline_cfg or PipelineConfig()
    return ZeroShotSegmentationPipeline(
        dinov3_cfg,
        sam2_cfg,
        pipeline_cfg,
        device=device,
        dtype=dtype,
        extractor=extractor,
        segmenter=segmenter,
    )