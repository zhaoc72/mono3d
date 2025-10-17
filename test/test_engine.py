"""Tests for engine module"""

import pytest
import torch
from pathlib import Path

from mono3d.engine import BaseTrainer
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