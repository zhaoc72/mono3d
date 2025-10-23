"""DINOv3 feature extraction and saliency proposal utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms

from .utils import LOGGER


@dataclass
class Dinov3Config:
    repo_or_dir: str
    model_name: str
    use_torch_hub: bool = True
    torchhub_source: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_layer: int = -1


class DINOv3FeatureExtractor:
    """Wrapper around the official DINOv3 models for zero-shot proposals."""

    def __init__(self, config: Dinov3Config, device: torch.device | str, dtype: torch.dtype) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.model: nn.Module = self._load_model()
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.last_attention: Optional[torch.Tensor] = None
        self._supports_direct_attention = False
        self._hook_handles = []
        self._register_attention_hook()

    def _load_model(self) -> nn.Module:
        if self.config.use_torch_hub:
            LOGGER.info(
                "Loading DINOv3 model %s from %s via torch.hub",
                self.config.model_name,
                self.config.repo_or_dir,
            )
            hub_kwargs = {
                "trust_repo": True,
                "pretrained": self.config.checkpoint_path is None,
            }
            if self.config.torchhub_source is not None:
                hub_kwargs["source"] = self.config.torchhub_source
            elif Path(self.config.repo_or_dir).exists():
                hub_kwargs["source"] = "local"
            model = torch.hub.load(
                self.config.repo_or_dir,
                self.config.model_name,
                **hub_kwargs,
            )
        else:
            import importlib

            module = importlib.import_module(self.config.repo_or_dir)
            build = getattr(module, "build_model", None)
            if build is None:
                raise AttributeError("Module must expose a build_model function when use_torch_hub=False")
            model = build(self.config.model_name)

        if self.config.checkpoint_path:
            LOGGER.info("Loading DINOv3 checkpoint from %s", self.config.checkpoint_path)
            state = torch.load(self.config.checkpoint_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                LOGGER.warning("Missing keys when loading DINOv3 checkpoint: %s", missing)
            if unexpected:
                LOGGER.warning("Unexpected keys when loading DINOv3 checkpoint: %s", unexpected)

        return model.to(self.device)

    def _register_attention_hook(self) -> None:
        """Register hook to capture attention weights using monkey patching."""
        
        # Check if model has get_last_selfattention method
        self._supports_direct_attention = hasattr(self.model, "get_last_selfattention")
        
        if self._supports_direct_attention:
            LOGGER.debug("Using model's get_last_selfattention for attention extraction")
            return
        
        # For DINOv3, monkey patch the attention computation
        if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
            last_block = self.model.blocks[-1]
            
            if hasattr(last_block, 'attn'):
                attn_module = last_block.attn
                
                # Save original forward
                original_forward = attn_module.forward.__func__ if hasattr(attn_module.forward, '__func__') else attn_module.forward
                
                # Create wrapper that captures attention
                def wrapped_forward(self, x, rope=None):
                    """Wrapped forward that captures attention weights."""
                    B, N, C = x.shape
                    
                    # Compute qkv (standard ViT attention)
                    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    # Apply RoPE if provided and if the module supports it
                    if rope is not None and hasattr(self, 'rope'):
                        q = self.rope(q, rope)
                        k = self.rope(k, rope)
                    
                    # Compute attention
                    attn = (q @ k.transpose(-2, -1)) * self.scale
                    attn = attn.softmax(dim=-1)
                    
                    # Capture attention weights
                    # Access the extractor instance through closure
                    extractor_instance.last_attention = attn.detach()
                    
                    attn = self.attn_drop(attn)
                    
                    # Apply attention to values
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
                    
                    return x
                
                # Store reference to self for closure
                extractor_instance = self
                
                # Bind the new forward method
                import types
                attn_module.forward = types.MethodType(wrapped_forward, attn_module)
                
                LOGGER.debug("Successfully monkey-patched attention forward method")
            else:
                LOGGER.warning("Last block does not have 'attn' attribute")
        else:
            LOGGER.warning("Model does not have 'blocks' attribute or blocks is empty")

    def _prepare(self, image: np.ndarray) -> torch.Tensor:
        tensor = self.preprocess(image).unsqueeze(0)
        return tensor.to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        inputs = self._prepare(image)
        self.last_attention = None
        
        if self.device.type == "cuda":
            # Use updated autocast API
            autocast_context = torch.amp.autocast('cuda', enabled=self.dtype == torch.float16)
        else:
            from contextlib import nullcontext
            autocast_context = nullcontext()
        
        with autocast_context:
            outputs = self.model(inputs)
            
            # Try to get attention using model's built-in method if available
            attention = None
            if self._supports_direct_attention:
                try:
                    if hasattr(self.model, 'get_last_selfattention'):
                        method = self.model.get_last_selfattention
                        try:
                            attention = method(inputs)
                        except TypeError:
                            try:
                                attention = method()
                            except:
                                LOGGER.debug("get_last_selfattention() failed, using captured attention")
                                attention = None
                except Exception as e:
                    LOGGER.debug(f"Failed to get attention via get_last_selfattention: {e}")
                    attention = None
            
            # Use captured attention from monkey patch
            if attention is None:
                attention = self.last_attention
            
            if attention is None:
                LOGGER.error("Failed to capture attention weights")
        
        if isinstance(outputs, torch.Tensor):
            feats = outputs
        elif isinstance(outputs, dict):
            key = "x_norm_clstoken" if "x_norm_clstoken" in outputs else next(iter(outputs.values()))
            feats = outputs[key]
        else:
            feats = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        
        return {
            "features": feats,
            "attention": attention,
        }

    @staticmethod
    def attention_to_heatmap(attention: torch.Tensor, image_size: Tuple[int, int]) -> np.ndarray:
        if attention is None:
            raise ValueError(
                "Attention maps are not available. "
                "Please check the DINOv3 model initialization."
            )
        
        # Handle different attention tensor formats
        if isinstance(attention, (tuple, list)):
            attn = attention[0]
        else:
            attn = attention
        
        # Expected format: [batch, num_heads, num_tokens, num_tokens]
        if attn.dim() == 4:
            # Average over attention heads
            attn = attn.mean(dim=1)
        
        # Now should be [batch, num_tokens, num_tokens]
        if attn.dim() == 3:
            # Take CLS token attention to all patches (index 0 to 1:)
            # CLS token is at position 0, patches start at position 1
            attn = attn[:, 0, 1:]
        
        # Average over batch if needed
        if attn.dim() == 2:
            attn = attn[0]
        
        # Should now be 1D tensor with num_patches elements
        attn = attn.squeeze() if attn.dim() > 1 else attn
        
        num_patches = attn.shape[0]
        
        # Try to find the best square grid size
        # DINOv3 may have extra tokens (CLS + register tokens)
        side = int(num_patches ** 0.5)
        
        # Check if we have a perfect square
        if side * side == num_patches:
            # Perfect square - use as is
            pass
        elif side * side < num_patches < (side + 1) * (side + 1):
            # Not a perfect square - we likely have extra tokens
            # Try to find how many extra tokens we have
            target_patches = side * side
            extra_tokens = num_patches - target_patches
            
            LOGGER.debug(
                f"Attention has {num_patches} tokens, using {side}x{side}={target_patches} patches "
                f"(ignoring {extra_tokens} extra tokens)"
            )
            
            # Use only the first target_patches tokens (after CLS)
            attn = attn[:target_patches]
            num_patches = target_patches
        else:
            # Fallback: just use the closest square
            LOGGER.warning(
                f"Unexpected token count {num_patches}. "
                f"Using closest square grid: {side}x{side}"
            )
            attn = attn[:side*side]
            num_patches = side * side
        
        # Reshape to 2D spatial grid
        attn_map = attn.reshape(side, side).cpu().numpy()
        
        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
        
        # Resize to image dimensions
        resized = cv2.resize(attn_map, image_size, interpolation=cv2.INTER_CUBIC)
        
        return resized