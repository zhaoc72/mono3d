"""Tests for engine module"""

import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf

from mono3d.engine import BaseTrainer, SingleImageInferencer
from mono3d.models import GaussianModel


@pytest.mark.slow
class TestTrainer:
    """Test training functionality"""
    
    def test_trainer_initialization(self, test_config, device):
        """Test trainer initialization"""
        class DummyTrainer(BaseTrainer):
            def build_model(self):
                return GaussianModel(num_gaussians=100)
            
            def train_epoch(self):
                return {'loss': 1.0}
            
            def validate(self):
                return {'loss': 1.0}
        
        trainer = DummyTrainer(test_config, device)

        assert trainer.model is not None
        assert trainer.optimizer is not None
    
    def test_checkpoint_save_load(self, test_config, device, tmp_output_dir):
        """Test checkpoint saving and loading"""
        class DummyTrainer(BaseTrainer):
            def build_model(self):
                return GaussianModel(num_gaussians=100)
            
            def train_epoch(self):
                return {'loss': 1.0}
            
            def validate(self):
                return {'loss': 1.0}
        
        trainer = DummyTrainer(test_config, device)
        trainer.checkpoint_dir = tmp_output_dir
        
        # Save checkpoint
        checkpoint_path = tmp_output_dir / "test.pth"
        trainer.save_checkpoint("test.pth")
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        assert trainer.current_epoch >= 0


class TestInferencer:
    """Tests for the single image inferencer."""

    def test_single_image_inference(self, tmp_path):
        image_path = tmp_path / "test_image.png"
        image = Image.fromarray((np.ones((32, 32, 3)) * 127).astype(np.uint8))
        image.save(image_path)

        cfg = OmegaConf.create(
            {
                "device": "cpu",
                "paths": {"output_dir": str(tmp_path)},
                "model": {
                    "dinov3": {},
                    "detector": {},
                    "sam2": {},
                    "depth": {"max_depth": 5.0},
                    "gaussian": {"num_gaussians": 32},
                    "shape_prior": {"type": "none"},
                },
                "inference": {
                    "export_format": "ply",
                    "optimization_iterations": 0,
                    "focal_length": 50.0,
                },
            }
        )

        inferencer = SingleImageInferencer(cfg, torch.device("cpu"))
        result = inferencer.infer(image_path)

        assert "pointcloud" in result
        assert result["pointcloud"]["points"].shape[1] == 3