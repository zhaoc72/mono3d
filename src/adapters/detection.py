"""Detection adapter working with DINOv3 patch tokens."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pipeline_types import DetectionOutput
from ..utils import LOGGER


@dataclass
class DetectionAdapterConfig:
    """Runtime settings for the lightweight DETR-style adapter."""

    checkpoint_path: str = ""
    feature_dim: int = 1024
    num_classes: int = 91
    class_names: Optional[Sequence[str]] = None
    score_threshold: float = 0.25


def build_detection_adapter(
    config: DetectionAdapterConfig,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
) -> "DetectionAdapter":
    """Instantiate the lightweight detection adapter."""

    checkpoint_path = config.checkpoint_path
    LOGGER.info("Building detection adapter (checkpoint: %s)", checkpoint_path or "<random>")

    adapter = DetectionAdapter(
        feature_dim=config.feature_dim,
        num_classes=config.num_classes,
        class_names=config.class_names,
        score_threshold=config.score_threshold,
        device=device,
        dtype=torch_dtype,
    )

    # Load checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # Try to load compatible weights
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Filter and load compatible keys
            model_dict = adapter.model.state_dict()
            compatible_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            
            if compatible_dict:
                adapter.model.load_state_dict(compatible_dict, strict=False)
                LOGGER.info(f"✓ Loaded {len(compatible_dict)} compatible parameters")
            else:
                LOGGER.warning("No compatible parameters found in checkpoint")
                
        except Exception as e:
            LOGGER.warning(f"Failed to load checkpoint: {e}")
            LOGGER.info("Using randomly initialized weights")
    else:
        LOGGER.info("No checkpoint provided, using randomly initialized weights")

    adapter.model = adapter.model.to(device=device, dtype=torch_dtype)
    adapter.model.eval()

    return adapter


def build_coco_adapter(
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> "DetectionAdapter":
    """Backward-compatible helper mirroring the old API."""

    config = DetectionAdapterConfig(
        checkpoint_path=checkpoint_path,
        feature_dim=kwargs.get("feature_dim", 1024),
        num_classes=kwargs.get("num_classes", 91),
        score_threshold=kwargs.get("score_threshold", 0.25),
    )
    return build_detection_adapter(config, device=device, torch_dtype=torch_dtype)


class SimpleDETR(nn.Module):
    """Simplified DETR-style detection head."""
    
    def __init__(self, feature_dim: int = 1024, num_classes: int = 91, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Feature projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Simple transformer-like layers
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        
        # Detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        
    def forward(
        self,
        features: torch.Tensor,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
    ):
        """
        Args:
            features: [1, N, D] patch features
            image_size: (width, height)
            grid_size: (grid_h, grid_w)
        
        Returns:
            boxes: [M, 4] in (x1, y1, x2, y2) format
            scores: [M]
            labels: [M]
        """
        # Project features
        x = self.input_proj(features)  # [1, N, hidden_dim]
        
        # Encode
        x = self.encoder(x)  # [1, N, hidden_dim]
        
        # Predict classes and boxes
        class_logits = self.class_embed(x)  # [1, N, num_classes]
        bbox_pred = self.bbox_embed(x)  # [1, N, 4]
        
        # Convert to image coordinates
        boxes, scores, labels = self._postprocess(
            class_logits[0],
            bbox_pred[0],
            image_size,
            grid_size,
        )
        
        return boxes, scores, labels
    
    def _postprocess(
        self,
        class_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
    ):
        """Post-process predictions."""
        # Get scores and labels
        probs = F.softmax(class_logits, dim=-1)
        scores, labels = probs.max(dim=-1)
        
        # Filter background (label 0)
        fg_mask = labels > 0
        scores = scores[fg_mask]
        labels = labels[fg_mask]
        bbox_pred = bbox_pred[fg_mask]
        
        if len(scores) == 0:
            return (
                torch.zeros((0, 4), dtype=bbox_pred.dtype, device=bbox_pred.device),
                torch.zeros((0,), dtype=scores.dtype, device=scores.device),
                torch.zeros((0,), dtype=torch.long, device=labels.device),
            )
        
        # Convert bbox predictions to image coordinates
        # bbox_pred is in format [cx, cy, w, h] normalized to [0, 1]
        width, height = image_size
        grid_h, grid_w = grid_size
        
        cx = bbox_pred[:, 0].sigmoid() * width
        cy = bbox_pred[:, 1].sigmoid() * height
        w = bbox_pred[:, 2].sigmoid() * width
        h = bbox_pred[:, 3].sigmoid() * height
        
        # Convert to [x1, y1, x2, y2]
        x1 = (cx - w / 2).clamp(0, width - 1)
        y1 = (cy - h / 2).clamp(0, height - 1)
        x2 = (cx + w / 2).clamp(0, width - 1)
        y2 = (cy + h / 2).clamp(0, height - 1)
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        return boxes, scores, labels


class DetectionAdapter:
    """Wrapper for detection model."""
    
    # COCO class names
    COCO_CLASSES = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(
        self,
        feature_dim: int = 1024,
        num_classes: int = 91,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        class_names: Optional[Sequence[str]] = None,
        score_threshold: float = 0.25,
    ):
        self.model = SimpleDETR(feature_dim, num_classes)
        self.device = torch.device(device)
        self.dtype = dtype
        if class_names is None:
            class_names = self.COCO_CLASSES[:num_classes]
        self.class_names = list(class_names)
        self.score_threshold = float(score_threshold)

    def _ensure_input_dim(self, feature_dim: int) -> None:
        """Resize the input projection layer if the backbone feature dimension changes."""

        current_dim = getattr(self.model, "feature_dim", feature_dim)
        if feature_dim == current_dim:
            return

        LOGGER.info(
            "Adapting detection adapter input dimension from %d to %d",
            current_dim,
            feature_dim,
        )

        new_proj = nn.Linear(feature_dim, self.model.hidden_dim)
        nn.init.xavier_uniform_(new_proj.weight)
        if new_proj.bias is not None:
            nn.init.zeros_(new_proj.bias)
        self.model.input_proj = new_proj.to(device=self.device, dtype=self.dtype)
        self.model.feature_dim = feature_dim

    @torch.inference_mode()
    def predict(
        self,
        patch_tokens: np.ndarray,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> DetectionOutput:
        """
        Run detection on patch tokens.
        
        Args:
            patch_tokens: Patch tokens from DINOv3 backbone [N, D]
            image_size: Original image size (width, height)
            grid_size: Grid size (height, width)
            
        Returns:
            DetectionOutput with boxes, class_ids, scores, and class_names
        """
        # Convert to tensor
        tokens_tensor = torch.from_numpy(patch_tokens).to(
            device=self.device, dtype=self.dtype
        )

        # Ensure the adapter can consume the provided feature dimension
        self._ensure_input_dim(tokens_tensor.shape[-1])

        # Reshape to [1, N, D] for batch processing
        tokens_tensor = tokens_tensor.unsqueeze(0)

        # Run detection
        try:
            boxes, scores, labels = self.model(tokens_tensor, image_size, grid_size)
        except Exception as e:
            LOGGER.warning(f"Detection failed: {e}, returning empty predictions")
            return DetectionOutput(
                boxes=np.zeros((0, 4), dtype=np.float32),
                class_ids=np.zeros((0,), dtype=np.int32),
                scores=np.zeros((0,), dtype=np.float32),
                class_names=self.class_names,
            )
        
        # Convert to numpy
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy().astype(np.int32)
        
        # Filter by score threshold
        valid_mask = scores_np >= self.score_threshold

        boxes_np = boxes_np[valid_mask]
        scores_np = scores_np[valid_mask]
        labels_np = labels_np[valid_mask]

        LOGGER.debug(
            "Detection: %d boxes (score >= %.2f)",
            len(boxes_np),
            self.score_threshold,
        )
        
        return DetectionOutput(
            boxes=boxes_np,
            class_ids=labels_np,
            scores=scores_np,
            class_names=self.class_names,
        )