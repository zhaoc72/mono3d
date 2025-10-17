"""Pytest configuration and fixtures"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Ensure the project package (``mono3d``) is importable when tests are invoked
# directly from the repository root without installing the package.  Pytest's
# discovery does not automatically add the ``src`` layout to ``sys.path`` so we
# append it manually here before importing any project modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def device():
    """Get device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def batch_size():
    """Default batch size for tests"""
    return 2


@pytest.fixture
def image_size():
    """Default image size for tests"""
    return (256, 256)


@pytest.fixture
def dummy_image(batch_size, image_size):
    """Create dummy image tensor"""
    return torch.randn(batch_size, 3, *image_size)


@pytest.fixture
def dummy_pointcloud(batch_size):
    """Create dummy point cloud"""
    num_points = 1024
    return torch.randn(batch_size, num_points, 3)


@pytest.fixture
def dummy_mesh():
    """Create dummy mesh"""
    vertices = torch.randn(100, 3)
    faces = torch.randint(0, 100, (150, 3))
    return {'vertices': vertices, 'faces': faces}


@pytest.fixture
def test_config():
    """Create test configuration"""
    config = OmegaConf.create({
        'project_name': 'test',
        'seed': 42,
        'device': 'cpu',
        'data': {
            'name': 'pix3d',
            'batch_size': 2,
            'image': {
                'size': [256, 256],
            },
        },
        'model': {
            'gaussian': {
                'num_gaussians': 100,
                'sh_degree': 2,
            },
        },
        'training': {
            'epochs': 2,
            'optimizer': {
                'type': 'adam',
                'lr': 1e-4,
                'weight_decay': 0.0,
            },
        },
    })
    return config


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
