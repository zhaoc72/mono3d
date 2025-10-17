"""Tests for model module"""

import pytest
import torch

from mono3d.models import (
    GaussianModel,
    ShapeVAE,
    DINOv3,
    GroundingDINODetector,
)
from mono3d.models.networks import MLP, PointNetEncoder
from mono3d.models.losses import (
    ColorLoss,
    DepthLoss,
    chamfer_distance,
)


class TestNetworks:
    """Test basic network components"""
    
    def test_mlp(self):
        """Test MLP network"""
        mlp = MLP(
            input_dim=256,
            hidden_dims=[512, 256],
            output_dim=128
        )
        
        x = torch.randn(4, 256)
        output = mlp(x)

        assert output.shape == (4, 128)

class TestDetector:
    """Tests for the Grounding DINO detector wrapper."""

    def test_fallback_detection(self):
        detector = GroundingDINODetector()
        image = torch.rand(1, 3, 64, 64)

        detections = detector(image, text_prompts="chair")

        assert "boxes" in detections
        assert detections["boxes"].shape == (1, 1, 4)
        assert detections["scores"].shape == (1, 1)
        assert detections["labels"][0][0] == "chair"
    
    def test_pointnet_encoder(self, dummy_pointcloud):
        """Test PointNet encoder"""
        encoder = PointNetEncoder(
            input_dim=3,
            hidden_dims=[64, 128],
            output_dim=256
        )
        
        output = encoder(dummy_pointcloud)
        
        assert output.shape == (dummy_pointcloud.shape[0], 256)


class TestGaussianModel:
    """Test Gaussian Splatting model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = GaussianModel(
            num_gaussians=100,
            sh_degree=2
        )
        
        assert model.num_gaussians == 100
        assert model.sh_degree == 2
    
    def test_from_pointcloud(self, dummy_pointcloud):
        """Test initialization from point cloud"""
        model = GaussianModel(num_gaussians=100)
        
        points = dummy_pointcloud[0]  # (N, 3)
        model.initialize_from_pointcloud(points)
        
        assert model.xyz.shape[0] == 100
    
    def test_render(self):
        """Test rendering"""
        model = GaussianModel(num_gaussians=100)
        
        # Initialize with random points
        points = torch.randn(100, 3)
        model.initialize_from_pointcloud(points)
        
        # Render
        rendered = model.render()
        
        assert 'color' in rendered
        assert 'depth' in rendered
        assert 'alpha' in rendered
    
    def test_to_pointcloud(self):
        """Test conversion to point cloud"""
        model = GaussianModel(num_gaussians=100)
        
        points = torch.randn(100, 3)
        model.initialize_from_pointcloud(points)
        
        pc = model.to_pointcloud()
        
        assert 'points' in pc
        assert 'colors' in pc
        assert pc['points'].shape == (100, 3)


class TestShapeVAE:
    """Test Shape VAE model"""
    
    def test_forward(self, dummy_pointcloud):
        """Test forward pass"""
        model = ShapeVAE(
            encoder={'backbone': 'pointnet', 'hidden_dims': [256, 512, 512]},
            decoder={'type': 'mlp', 'hidden_dims': [512, 512, 256]},
            latent_dim=128
        )
        
        output = model(dummy_pointcloud)
        
        assert 'reconstructed' in output
        assert 'mu' in output
        assert 'logvar' in output
        assert 'z' in output
    
    def test_encode_decode(self, dummy_pointcloud):
        """Test encode and decode separately"""
        model = ShapeVAE(
            encoder={'backbone': 'pointnet', 'hidden_dims': [256, 512, 512]},
            decoder={'type': 'mlp', 'hidden_dims': [512, 512, 256]},
            latent_dim=128
        )
        
        # Encode
        mu, logvar = model.encode(dummy_pointcloud)
        assert mu.shape == (dummy_pointcloud.shape[0], 128)
        
        # Decode
        z = model.reparameterize(mu, logvar)
        reconstructed = model.decode(z)
        assert reconstructed.shape[1:] == dummy_pointcloud.shape[1:]


class TestLosses:
    """Test loss functions"""
    
    def test_color_loss(self, dummy_image):
        """Test color loss"""
        loss_fn = ColorLoss(loss_type='l1')
        
        pred = dummy_image
        target = dummy_image + torch.randn_like(dummy_image) * 0.1
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
    
    def test_depth_loss(self, batch_size, image_size):
        """Test depth loss"""
        loss_fn = DepthLoss(loss_type='l1')
        
        pred = torch.rand(batch_size, 1, *image_size)
        target = torch.rand(batch_size, 1, *image_size)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
    
    def test_chamfer_distance(self, dummy_pointcloud):
        """Test Chamfer distance"""
        pred = dummy_pointcloud
        target = dummy_pointcloud + torch.randn_like(dummy_pointcloud) * 0.1
        
        cd = chamfer_distance(pred, target)
        
        assert cd.item() >= 0


@pytest.mark.gpu
class TestModelsGPU:
    """Test models on GPU"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gaussian_model_gpu(self):
        """Test Gaussian model on GPU"""
        device = torch.device('cuda')
        
        model = GaussianModel(num_gaussians=100).to(device)
        points = torch.randn(100, 3).to(device)
        
        model.initialize_from_pointcloud(points)
        rendered = model.render()
        
        assert rendered['color'].device.type == 'cuda'