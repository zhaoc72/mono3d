"""统一推理引擎实现。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..data.utils import CameraParams, depth_to_pointcloud
from ..registry import build
from ..utils.io import save_mesh, save_pointcloud

log = logging.getLogger(__name__)

def _to_container(cfg_section: Any) -> Dict[str, Any]:
    """将OmegaConf节点安全转换为dict。"""

    if isinstance(cfg_section, DictConfig):
        return OmegaConf.to_container(cfg_section, resolve=True)  # type: ignore[arg-type]
    return cfg_section or {}


def infer(cfg: DictConfig) -> None:
    """统一推理入口。"""

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    input_path = cfg.get("input", cfg.get("image", None))
    if input_path is None:
        raise ValueError("Please specify input image or video path via cfg.input")

    
    input_path = Path(input_path)

    inferencer: BaseInferencer

    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        inferencer = SingleImageInferencer(cfg, device)

    elif input_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
        inferencer = VideoInferencer(cfg, device)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    result = inferencer.infer(input_path)

    inference_cfg = _to_container(cfg.get("inference", {}))
    export_format = inference_cfg.get("export_format", "ply")

    output_root = Path(cfg.paths.output_dir) if hasattr(cfg, "paths") else Path("outputs")
    output_dir = output_root / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input_path.stem
    if export_format in {"ply", "both"} and "pointcloud" in result:
        save_pointcloud(result["pointcloud"], output_dir / f"{output_name}.ply")
    if export_format in {"obj", "both"} and "mesh" in result:
        save_mesh(result["mesh"], output_dir / f"{output_name}.obj")

    log.info("Results saved to %s", output_dir)


class BaseInferencer:
    """推理器基类，方便未来扩展。"""

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def infer(self, input_path: Path) -> Dict[str, Any]:  # pragma: no cover - interface

        raise NotImplementedError


class SingleImageInferencer(BaseInferencer):
    """单图像推理器，串联整个重建流程。"""

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__(cfg, device)

        model_cfg = _to_container(getattr(cfg, "model", {}))
        self.inference_cfg = _to_container(cfg.get("inference", {}))
        self.paths_cfg = _to_container(cfg.get("paths", {}))

        log.info("Loading frontend models...")
        self.dino = build("model", "dinov3", **model_cfg.get("dinov3", {})).to(device).eval()
        self.detector = build("model", "grounding_dino", **model_cfg.get("detector", {})).to(device).eval()
        self.sam = build("model", "sam2", **model_cfg.get("sam2", {})).to(device).eval()
        self.depth = build("model", "depth_anything", **model_cfg.get("depth", {})).to(device).eval()

        log.info("Loading shape priors and Gaussian model...")
        shape_prior_cfg = model_cfg.get("shape_prior", {})
        self.explicit_prior = None
        self.implicit_prior = None

        prior_type = str(shape_prior_cfg.get("type", "explicit")).lower()
        if prior_type in {"explicit", "hybrid"}:
            explicit_cfg = dict(shape_prior_cfg.get("explicit", {}))
            template_dir = explicit_cfg.get("template_dir")
            if template_dir is None:
                data_dir = self.paths_cfg.get("data_dir", "data")
                template_dir = str(Path(data_dir) / "templates")
            explicit_cfg.setdefault("template_dir", template_dir)
            self.explicit_prior = build("model", "explicit_prior", **explicit_cfg).to(device).eval()

        if prior_type in {"implicit", "hybrid"}:
            implicit_cfg = dict(shape_prior_cfg.get("implicit", {}))
            if implicit_cfg:
                self.implicit_prior = build("model", "implicit_prior", **implicit_cfg).to(device).eval()

        initializer_cfg = model_cfg.get("initializer", {})
        self.init_net = None
        if initializer_cfg:
            self.init_net = build("model", "shape_init_net", **initializer_cfg).to(device).eval()

        gaussian_cfg = dict(model_cfg.get("gaussian", {}))
        self.gaussian_opt_cfg = gaussian_cfg.pop("optimization", {})
        self.gaussian_cfg = gaussian_cfg
        self._base_gaussian = self._build_gaussian_model()

        log.info("Models loaded successfully")

    def _build_gaussian_model(self):
        model = build("model", "gaussian", **self.gaussian_cfg)
        return model.to(self.device)

    @torch.no_grad()
    def infer(self, image_path: Path) -> Dict[str, Any]:
        log.info("Processing %s", image_path)

        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = self._preprocess_image(pil_image)

        detection_prompt = self.inference_cfg.get("category", None)
        detections = self.detector(image_tensor, text_prompts=detection_prompt)
        detection = self._select_detection(detections, detection_prompt)

        mask = self._segment_object(image_tensor, detection)
        depth = self.depth.predict_metric_depth(image_tensor)
        features = self.dino(image_tensor)

        camera = self._build_default_camera(pil_image)
        pointcloud, colors = self._depth_to_pointcloud(depth, mask, image_tensor, camera)

        category = detection.get("label") or self.inference_cfg.get("category", "object")
        initial_shape = self._prepare_initial_shape(features, depth, mask, pointcloud, category)

        gaussian_model = self._build_gaussian_model()
        gaussian_model.initialize_from_shape(initial_shape)

        iterations = self.inference_cfg.get(
            "optimization_iterations",
            self.gaussian_opt_cfg.get("iterations", 300),
        )

        gaussian_camera = [self._camera_to_gaussian(camera, mask.shape[-2:])]
        if iterations > 0:
            gaussian_model.optimize(
                images=image_tensor,
                depths=depth,
                masks=mask,
                cameras=gaussian_camera,
                iterations=iterations,
            )

        return {
            "pointcloud": gaussian_model.to_pointcloud(),
            "mesh": gaussian_model.to_mesh(),
            "gaussian": gaussian_model,
        }

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _select_detection(
        self,
        detections: Dict[str, Any],
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        boxes = detections.get("boxes")
        scores = detections.get("scores")
        labels = detections.get("labels", [])

        if boxes is None or boxes.numel() == 0:
            H, W = detections.get("image_size", (0, 0))
            box = torch.tensor([0.0, 0.0, float(W), float(H)], device=self.device)
            return {"box": box, "label": prompt, "score": 1.0}

        if boxes.dim() == 3:
            boxes = boxes[0]
        if scores is not None and scores.dim() == 2:
            scores = scores[0]

        if scores is None or scores.numel() == 0:
            idx = 0
        else:
            idx = int(torch.argmax(scores).item())

        box = boxes[idx].to(self.device)
        label_candidates = labels[0] if labels else []
        label = label_candidates[idx] if idx < len(label_candidates) else prompt
        score = float(scores[idx].item()) if scores is not None else 1.0

        return {"box": box, "label": label, "score": score}

    def _segment_object(self, image: torch.Tensor, detection: Dict[str, Any]) -> torch.Tensor:
        box = detection.get("box")
        prompts = {"boxes": box.unsqueeze(0)} if box is not None else None
        mask = self.sam(image, prompts)
        return mask.to(self.device)

    def _build_default_camera(self, image: Image.Image) -> CameraParams:
        width, height = image.size
        focal_length = self.inference_cfg.get("focal_length", max(width, height))

        return CameraParams(
            fx=float(focal_length),
            fy=float(focal_length),
            cx=width / 2.0,
            cy=height / 2.0,
            R=np.eye(3),
            t=np.zeros(3),
            width=width,
            height=height,
        )

    def _depth_to_pointcloud(
        self,
        depth: torch.Tensor,
        mask: torch.Tensor,
        image: torch.Tensor,
        camera: CameraParams,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        depth_np = depth.squeeze().detach().cpu().numpy()
        mask_np = mask.squeeze().detach().cpu().numpy() > 0.5
        color_np = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        color_np = (color_np * 255.0).astype(np.uint8)

        pc = depth_to_pointcloud(depth_np, camera, mask_np, color_np)
        points = torch.from_numpy(pc["points"]).float().to(self.device)

        colors = None
        if "colors" in pc:
            colors = torch.from_numpy(pc["colors"]).float().to(self.device)

        return points, colors

    def _extract_features_for_init(self, features: Any) -> Optional[torch.Tensor]:
        if isinstance(features, dict):
            if "global" in features:
                feat = features["global"]
            else:
                feat = next(iter(features.values()))
        else:
            feat = features

        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        if not isinstance(feat, torch.Tensor):
            return None

        return feat.reshape(feat.shape[0], -1)

    def _prepare_initial_shape(
        self,
        features: Any,
        depth: torch.Tensor,
        mask: torch.Tensor,
        pointcloud: torch.Tensor,
        category: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        points: List[torch.Tensor] = []

        if pointcloud is not None and pointcloud.numel() > 0:
            points.append(pointcloud)

        if self.explicit_prior is not None:
            explicit_shape = self.explicit_prior.initialize(
                category=category or "object",
                pointcloud=pointcloud,
                depth=depth.squeeze(0),
                mask=mask.squeeze(0),
            )
            explicit_points = explicit_shape.get("points")
            if explicit_points is not None:
                points.append(explicit_points.to(self.device))

        if self.implicit_prior is not None:
            latent = torch.zeros(1, self.implicit_prior.latent_dim, device=self.device)
            implicit_points = self.implicit_prior.decode(latent).squeeze(0)
            points.append(implicit_points.to(self.device))

        if self.init_net is not None:
            feat_tensor = self._extract_features_for_init(features)
            if feat_tensor is not None:
                init_output = self.init_net.predict_shape(feat_tensor)
                init_points = init_output.get("points")
                if init_points is not None:
                    points.append(init_points.squeeze(0).to(self.device))

        if not points:
            points.append(torch.zeros((self._base_gaussian.num_gaussians, 3), device=self.device))

        combined_points = torch.cat(points, dim=0)
        return {"points": combined_points}

    def _camera_to_gaussian(self, camera: CameraParams, image_size: Sequence[int]) -> Dict[str, Any]:
        height, width = image_size
        FoVx = 2 * np.arctan(width / (2 * camera.fx)) if camera.fx > 0 else np.pi / 2
        FoVy = 2 * np.arctan(height / (2 * camera.fy)) if camera.fy > 0 else np.pi / 2

        world_view = torch.eye(4, device=self.device, dtype=torch.float32)
        projection = torch.eye(4, device=self.device, dtype=torch.float32)

        return {
            "image_width": width,
            "image_height": height,
            "FoVx": FoVx,
            "FoVy": FoVy,
            "world_view_transform": world_view,
            "projection_matrix": projection,
            "camera_center": torch.zeros(3, device=self.device),
        }


class VideoInferencer(BaseInferencer):  # pragma: no cover - placeholder implementation
    """视频推理器（占位）。"""

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__(cfg, device)

    def infer(self, video_path: Path) -> Dict[str, Any]:
        raise NotImplementedError("Video inference is not implemented yet")


__all__ = [
    "infer",
    "SingleImageInferencer",
    "VideoInferencer",
]

