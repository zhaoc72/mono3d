"""Tests for utils module"""

import pytest
import torch
import numpy as np
from pathlib import Path

from mono3d.utils.geometry import (
    transform_points,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)
from mono3d.utils.metrics import (
    compute_chamfer_distance,
    compute_iou,
    compute_fscore,
)
from mono3d.utils.io import (
    save_pointcloud,
    load_pointcloud,
    save_mesh,
    load_mesh,
)


class TestGeometry:
    """Test geometry utilities"""
    
    def test_transform_points(self):
        """Test point transformation"""
        points = torch.randn(100, 3)
        transform = torch.eye(4)
        
        transformed = transform_points(points, transform)
        
        assert transformed.shape == points.shape
        assert torch.allclose(transformed, points, atol=1e-5)
    
    def test_quaternion_conversion(self):
        """Test rotation matrix to quaternion conversion"""
        R = torch.eye(3)
        
        quat = rotation_matrix_to_quaternion(R)
        R_reconstructed = quaternion_to_rotation_matrix(quat)
        
        assert torch.allclose(R, R_reconstructed, atol=1e-5)


class TestMetrics:
    """Test metric calculations"""
    
    def test_chamfer_distance(self, dummy_pointcloud):
        """Test Chamfer distance computation"""
        pred = dummy_pointcloud
        target = dummy_pointcloud + torch.randn_like(dummy_pointcloud) * 0.01
        
        cd = compute_chamfer_distance(pred[0], target[0])
        
        assert cd >= 0
        assert isinstance(cd, float)
    
    def test_fscore(self, dummy_pointcloud):
        """Test F-score computation"""
        pred = dummy_pointcloud[0]
        target = dummy_pointcloud[0] + torch.randn_like(dummy_pointcloud[0]) * 0.01
        
        results = compute_fscore(pred, target, threshold=0.01)
        
        assert 'precision' in results
        assert 'recall' in results
        assert 'fscore' in results
        assert 0 <= results['fscore'] <= 1


class TestIO:
    """Test IO operations"""
    
    def test_save_load_pointcloud(self, tmp_output_dir):
        """Test point cloud save and load"""
        points = np.random.rand(100, 3)
        colors = np.random.rand(100, 3)
        
        filepath = tmp_output_dir / "test.ply"
        
        # Save
        save_pointcloud(points, filepath, colors=colors)
        assert filepath.exists()
        
        # Load
        loaded = load_pointcloud(filepath)
        
        assert 'points' in loaded
        assert 'colors' in loaded
        assert loaded['points'].shape == points.shape
        assert loaded['colors'].shape == colors.shape
    
    def test_save_load_mesh(self, tmp_output_dir, dummy_mesh):
        """Test mesh save and load"""
        vertices = dummy_mesh['vertices'].numpy()
        faces = dummy_mesh['faces'].numpy()
        
        filepath = tmp_output_dir / "test.obj"
        
        # Save
        save_mesh(vertices, faces, filepath)
        assert filepath.exists()
        
        # Load
        loaded = load_mesh(filepath)
        
        assert 'vertices' in loaded
        assert 'faces' in loaded
        assert loaded['vertices'].shape == vertices.shape


class TestVisualization:
    """Test visualization utilities"""
    
    def test_visualize_pointcloud(self, tmp_output_dir):
        """Test point cloud visualization"""
        from mono3d.utils.visualization import visualize_pointcloud
        
        points = np.random.rand(100, 3)
        colors = np.random.rand(100, 3)
        
        save_path = tmp_output_dir / "pointcloud_vis.png"
        
        visualize_pointcloud(
            points,
            colors=colors,
            save_path=save_path,
            show=False
        )
        
        assert save_path.exists()