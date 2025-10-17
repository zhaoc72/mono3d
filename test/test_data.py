"""Tests for data module"""

import pytest
import torch
import numpy as np
from pathlib import Path

from mono3d.data import (
    Pix3DDataset,
    build_dataset,
    build_dataloader,
)
from mono3d.data.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Compose,
)
from mono3d.data.utils import (
    CameraParams,
    depth_to_pointcloud,
)


class TestTransforms:
    """Test data transforms"""
    
    def test_to_tensor(self):
        """Test ToTensor transform"""
        from PIL import Image
        
        # Create dummy PIL image
        img = Image.new('RGB', (256, 256))
        sample = {'image': img}
        
        transform = ToTensor()
        result = transform(sample)
        
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 256, 256)
    
    def test_resize(self, image_size):
        """Test Resize transform"""
        image = torch.randn(3, 512, 512)
        sample = {'image': image}
        
        transform = Resize(image_size)
        result = transform(sample)
        
        assert result['image'].shape == (3, *image_size)
    
    def test_compose(self):
        """Test Compose transform"""
        transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(),
        ])
        
        from PIL import Image
        img = Image.new('RGB', (512, 512))
        sample = {'image': img}
        
        result = transforms(sample)
        
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 256, 256)


class TestCameraParams:
    """Test camera parameters"""
    
    def test_camera_creation(self):
        """Test creating camera parameters"""
        camera = CameraParams(
            fx=525.0, fy=525.0,
            cx=320.0, cy=240.0,
            R=np.eye(3),
            t=np.zeros(3),
            width=640, height=480
        )
        
        assert camera.fx == 525.0
        assert camera.width == 640
    
    def test_intrinsic_matrix(self):
        """Test intrinsic matrix"""
        camera = CameraParams(
            fx=525.0, fy=525.0,
            cx=320.0, cy=240.0,
            R=np.eye(3),
            t=np.zeros(3),
            width=640, height=480
        )
        
        K = camera.get_intrinsic_matrix()
        
        assert K.shape == (3, 3)
        assert K[0, 0] == 525.0
        assert K[1, 1] == 525.0
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0
    
    def test_projection(self):
        """Test 3D to 2D projection"""
        camera = CameraParams(
            fx=525.0, fy=525.0,
            cx=320.0, cy=240.0,
            R=np.eye(3),
            t=np.zeros(3),
            width=640, height=480
        )
        
        # Project a point
        point_3d = np.array([[0, 0, 1]])  # 1 meter in front
        point_2d = camera.project(point_3d)
        
        assert point_2d.shape == (1, 2)
        # Should project to center
        assert np.abs(point_2d[0, 0] - 320.0) < 1.0
        assert np.abs(point_2d[0, 1] - 240.0) < 1.0


class TestDepthProcessing:
    """Test depth processing utilities"""
    
    def test_depth_to_pointcloud(self):
        """Test depth to point cloud conversion"""
        # Create dummy depth map
        depth = np.random.rand(480, 640) * 5.0  # 0-5 meters
        
        camera = CameraParams(
            fx=525.0, fy=525.0,
            cx=320.0, cy=240.0,
            R=np.eye(3),
            t=np.zeros(3),
            width=640, height=480
        )
        
        pc = depth_to_pointcloud(depth, camera)
        
        assert 'points' in pc
        assert pc['points'].shape[1] == 3
        assert pc['points'].shape[0] > 0


@pytest.mark.slow
class TestDatasets:
    """Test dataset classes"""
    
    @pytest.mark.skipif(
        not Path("data/Pix3D").exists(),
        reason="Pix3D dataset not available"
    )
    def test_pix3d_dataset(self):
        """Test Pix3D dataset"""
        dataset = Pix3DDataset(
            root="data/Pix3D",
            split="train"
        )
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'category' in sample