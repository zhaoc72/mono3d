"""SAM2 segmentation utilities leveraging the official Meta release."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .utils import LOGGER, chunk_iterable


@dataclass
class Sam2Config:
    checkpoint_path: Optional[str] = None
    model_config: Optional[str] = None
    model_id: Optional[str] = None
    revision: Optional[str] = None
    load_in_8bit: bool = False
    max_batch_size: int = 32
    backend: str = "official"


class SAM2Segmenter:
    """Wraps the official SAM2 Hugging Face interface for batch inference."""

    def __init__(self, config: Sam2Config, device: torch.device | str, dtype: torch.dtype) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self._backend = config.backend
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.predictor: Optional[Any] = None
        self._load_model()
        if self.model is not None:
            self.model.eval()

    def _load_model(self) -> None:
        if self.config.backend == "official" or self.config.model_id is None:
            self._load_official_model()
        else:
            self._load_huggingface_model()

    def _load_official_model(self) -> None:
        if not self.config.checkpoint_path or not self.config.model_config:
            raise ValueError("SAM2 official backend requires checkpoint_path and model_config")
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "Failed to import SAM2 official modules. Ensure facebookresearch/sam2 is installed."
            ) from exc

        LOGGER.info(
            "Loading SAM2 official model from %s with config %s",
            self.config.checkpoint_path,
            self.config.model_config,
        )
        
        import os
        from pathlib import Path
        
        # Method 1: Try to use build_sam2 with proper Hydra initialization
        try:
            # Find the sam2 package location
            import sam2
            sam2_package_path = Path(sam2.__file__).parent
            config_dir = sam2_package_path / "configs"
            
            LOGGER.info(f"SAM2 package path: {sam2_package_path}")
            LOGGER.info(f"Config directory: {config_dir}")
            
            if not config_dir.exists():
                raise FileNotFoundError(f"SAM2 configs directory not found at {config_dir}")
            
            # Initialize Hydra with the correct config path
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            
            # Clear any existing Hydra instance
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            # Initialize with SAM2's config directory
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                # Now build_sam2 should work
                sam2_model = build_sam2(
                    self.config.model_config, 
                    self.config.checkpoint_path, 
                    device=self.device
                )
            
            LOGGER.info("✅ Successfully loaded SAM2 model using Hydra initialization")
            
        except Exception as e:
            LOGGER.warning(f"Failed to load with Hydra initialization: {e}")
            LOGGER.info("Attempting alternative loading method...")
            
            # Method 2: Load config manually and build model directly
            try:
                import yaml
                import sam2
                from hydra.utils import instantiate
                from omegaconf import OmegaConf
                
                # Find config file
                sam2_package_path = Path(sam2.__file__).parent
                config_path = sam2_package_path / "configs" / f"{self.config.model_config}.yaml"
                
                if not config_path.exists():
                    raise FileNotFoundError(f"Config file not found: {config_path}")
                
                LOGGER.info(f"Loading config from: {config_path}")
                
                # Load the YAML config
                with open(config_path, 'r') as f:
                    cfg_dict = yaml.safe_load(f)
                
                # Convert to OmegaConf
                cfg = OmegaConf.create(cfg_dict)
                
                # Instantiate the model using the config
                sam2_model = instantiate(cfg, _recursive_=False)
                
                # Load checkpoint
                LOGGER.info(f"Loading checkpoint: {self.config.checkpoint_path}")
                checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
                
                # Load state dict
                missing_keys, unexpected_keys = sam2_model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    LOGGER.warning(f"Missing keys: {missing_keys[:5]}...")
                if unexpected_keys:
                    LOGGER.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
                
                # Move to device
                sam2_model = sam2_model.to(self.device)
                
                LOGGER.info("✅ Successfully loaded SAM2 model using manual config loading")
                
            except Exception as e2:
                LOGGER.error(f"Both loading methods failed. Last error: {e2}")
                raise RuntimeError(
                    f"Failed to load SAM2 model. "
                    f"Config: {self.config.model_config}, "
                    f"Checkpoint: {self.config.checkpoint_path}"
                ) from e2
        
        # Create predictor
        self.predictor = SAM2ImagePredictor(sam2_model, device=self.device)
        self.model = sam2_model
        self.processor = None
        self._backend = "official"
        
        LOGGER.info("SAM2 model loaded successfully")

    def _load_huggingface_model(self) -> None:
        from transformers import Sam2Model, Sam2Processor

        if not self.config.model_id:
            raise ValueError("Hugging Face backend requires model_id")
        LOGGER.info("Loading SAM2 model %s", self.config.model_id)
        kwargs: Dict[str, Any] = {}
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        if self.config.load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
        model = Sam2Model.from_pretrained(self.config.model_id, **kwargs)
        processor = Sam2Processor.from_pretrained(self.config.model_id, **kwargs)
        if not self.config.load_in_8bit:
            model = model.to(self.device, dtype=self.dtype)
        self.processor = processor
        self.model = model
        self._backend = "huggingface"

    def _prepare_inputs(
        self,
        image: np.ndarray,
        boxes: Optional[List[Sequence[int]]] = None,
        points: Optional[List[Sequence[Tuple[int, int]]]] = None,
        labels: Optional[List[Sequence[int]]] = None,
    ) -> Dict[str, Any]:
        image_input = image
        kwargs: Dict[str, Any] = {"images": image_input, "return_tensors": "pt"}
        if boxes is not None:
            box_tensor = [[[float(coord) for coord in box] for box in boxes]]
            kwargs["input_boxes"] = box_tensor
        if points is not None and labels is not None:
            point_tensor = [
                [[float(p[0]), float(p[1])] for p in prompt_points] for prompt_points in points
            ]
            label_tensor = [[int(label) for label in prompt_labels] for prompt_labels in labels]
            kwargs["input_points"] = [point_tensor]
            kwargs["input_labels"] = [label_tensor]
        inputs = self.processor(**kwargs)
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def _segment_huggingface(
        self,
        image: np.ndarray,
        boxes: Optional[List[Sequence[int]]],
        points: Optional[List[Sequence[Tuple[int, int]]]],
        labels: Optional[List[Sequence[int]]],
    ) -> List[np.ndarray]:
        if self.processor is None or self.model is None:
            raise RuntimeError("Hugging Face SAM2 backend is not initialized")
        inputs = self._prepare_inputs(image, boxes=boxes, points=points, labels=labels)
        if self.device.type == "cuda" and not self.config.load_in_8bit:
            autocast_context = torch.amp.autocast('cuda', enabled=self.dtype == torch.float16)
        else:
            from contextlib import nullcontext

            autocast_context = nullcontext()
        with autocast_context:
            outputs = self.model(**inputs)
        processed_masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        iou_scores = getattr(outputs, "iou_scores", None)
        masks: List[np.ndarray] = []
        for idx, mask_tensor in enumerate(processed_masks):
            mask_np = mask_tensor.cpu().numpy()
            if mask_np.ndim == 4:
                mask_np = mask_np[:, 0]
            if mask_np.ndim == 3 and mask_np.shape[0] > 1 and iou_scores is not None:
                scores = iou_scores[idx].cpu().numpy().reshape(-1)
                best_idx = int(scores.argmax())
                masks.append(mask_np[best_idx])
            elif mask_np.ndim == 3:
                masks.append(mask_np[0])
            elif mask_np.ndim == 2:
                masks.append(mask_np)
            else:
                raise ValueError(f"Unexpected mask tensor shape: {mask_np.shape}")
        return masks

    def _prepare_official_prompt(
        self,
        image: np.ndarray,
        box: Optional[Sequence[int]],
        points: Optional[Sequence[Tuple[int, int]]],
        labels: Optional[Sequence[int]],
    ) -> Dict[str, np.ndarray]:
        if points is not None and labels is None:
            raise ValueError("Point prompts require labels")
        prompt: Dict[str, np.ndarray] = {}
        if box is not None:
            prompt["box"] = np.array(box, dtype=np.float32)[None, :]
        if points is not None and labels is not None:
            prompt["point_coords"] = np.asarray(points, dtype=np.float32)
            prompt["point_labels"] = np.asarray(labels, dtype=np.int64)
        prompt["image"] = image
        return prompt

    def _segment_official(
        self,
        image: np.ndarray,
        boxes: Optional[List[Sequence[int]]],
        points: Optional[List[Sequence[Tuple[int, int]]]],
        labels: Optional[List[Sequence[int]]],
    ) -> List[np.ndarray]:
        if self.predictor is None:
            raise RuntimeError("Official SAM2 backend is not initialized")
        
        # Ensure image is in the correct format
        rgb_image = image
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for SAM2
        # IMPORTANT: Use copy() to avoid negative strides
        bgr_image = rgb_image[:, :, ::-1].copy()
        
        # Set the image
        self.predictor.set_image(bgr_image)

        num_prompts = 0
        if boxes is not None:
            num_prompts = len(boxes)
        if points is not None:
            num_prompts = max(num_prompts, len(points))
        results: List[np.ndarray] = []

        for idx in range(num_prompts):
            box = boxes[idx] if boxes is not None and idx < len(boxes) else None
            pts = points[idx] if points is not None and idx < len(points) else None
            lbl = labels[idx] if labels is not None and idx < len(labels) else None
            prompt = self._prepare_official_prompt(bgr_image, box, pts, lbl)
            predict_kwargs = {
                k: v for k, v in prompt.items() if k in {"point_coords", "point_labels", "box"}
            }
            masks, scores, _ = self.predictor.predict(multimask_output=True, **predict_kwargs)
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            if scores is not None and len(scores) > 0:
                best = int(np.argmax(scores))
                mask = masks[best]
            else:
                mask = masks[0]
            results.append(mask.astype(np.uint8))
        return results

    @torch.inference_mode()
    def segment(
        self,
        image: np.ndarray,
        boxes: Optional[List[Sequence[int]]] = None,
        points: Optional[List[Sequence[Tuple[int, int]]]] = None,
        labels: Optional[List[Sequence[int]]] = None,
    ) -> List[np.ndarray]:
        if self._backend == "huggingface":
            return self._segment_huggingface(image, boxes, points, labels)
        return self._segment_official(image, boxes, points, labels)

    @torch.inference_mode()
    def segment_batched(
        self,
        image: np.ndarray,
        boxes: List[Sequence[int]],
        points: Optional[List[Sequence[Tuple[int, int]]]] = None,
        labels: Optional[List[Sequence[int]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        if self._backend == "huggingface":
            if batch_size is None:
                batch_size = self.config.max_batch_size
            batched_masks: List[np.ndarray] = []
            offset = 0
            for chunk_boxes in chunk_iterable(boxes, batch_size):
                chunk_points = None
                chunk_labels = None
                if points is not None and labels is not None:
                    chunk_points = points[offset : offset + len(chunk_boxes)]
                    chunk_labels = labels[offset : offset + len(chunk_boxes)]
                masks = self.segment(image, boxes=list(chunk_boxes), points=chunk_points, labels=chunk_labels)
                batched_masks.extend(masks)
                offset += len(chunk_boxes)
            return batched_masks

        # Official backend processes prompts sequentially
        return self.segment(image, boxes=boxes, points=points, labels=labels)