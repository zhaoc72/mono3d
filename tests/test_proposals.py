import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_generator import (
    ClusterConfig,
    PromptConfig,
    kmeans_cluster,
    labels_to_regions,
    expand_region_instances,
    proposals_to_prompts,
)


def test_kmeans_cluster_is_deterministic():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(16, 3)).astype(np.float32)
    config = ClusterConfig(num_clusters=3, random_state=123, max_iterations=50)
    labels_first, centroids_first = kmeans_cluster(features, config)
    labels_second, centroids_second = kmeans_cluster(features, config)
    assert np.array_equal(labels_first, labels_second)
    assert np.allclose(centroids_first, centroids_second)


def test_labels_to_regions_and_prompts(monkeypatch):
    label_map = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    config = ClusterConfig(min_region_area=4, max_regions=3)
    proposals = labels_to_regions(label_map, (8, 8), config, patch_shape=label_map.shape)
    assert len(proposals) == 3
    assert proposals[0].mask.shape == (8, 8)

    prompt_cfg = PromptConfig(point_strategy="centroid")
    instance_props = expand_region_instances(
        proposals,
        prompt_cfg,
        config,
        patch_map=None,
        image_shape=(8, 8),
    )

    boxes, points, labels = proposals_to_prompts(
        instance_props,
        prompt_cfg,
        image_shape=(8, 8),
        cluster_config=config,
    )
    assert len(boxes) == len(points) == len(labels) == len(instance_props)
    for proposal, box, point_list in zip(instance_props, boxes, points):
        x0, y0, x1, y1 = box
        assert x0 < x1 and y0 < y1
        if point_list:
            cx, cy = point_list[0]
            assert proposal.mask[cy, cx] == 1
