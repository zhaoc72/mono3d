"""Class-aware prompt generation pipeline integrating detection and segmentation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

try:  # pragma: no cover - optional dependency in CI
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback for headless environments
    cv2 = None  # type: ignore

import numpy as np
import torch

from .dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from .sam2_segmenter import SAM2Segmenter, Sam2Config
from .utils import LOGGER


class DetectionAdapter(Protocol):
    """Lightweight protocol describing the detection adapter interface."""

    def predict(
        self,
        patch_tokens: np.ndarray,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> "DetectionOutput":
        ...


class SegmentationAdapter(Protocol):
    """Protocol describing the segmentation adapter interface."""

    def predict(
        self,
        patch_tokens: np.ndarray,
        image_size: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> "SegmentationOutput":
        ...


@dataclass
class DetectionOutput:
    """Container for detection adapter predictions."""

    boxes: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray
    class_names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if self.boxes.ndim != 2 or self.boxes.shape[-1] != 4:
            raise ValueError("DetectionOutput.boxes must be shaped (N, 4)")
        if self.class_ids.shape[0] != self.boxes.shape[0] or self.scores.shape[0] != self.boxes.shape[0]:
            raise ValueError("DetectionOutput arrays must have identical leading dimensions")


@dataclass
class SegmentationOutput:
    """Container for segmentation adapter outputs."""

    logits: np.ndarray
    class_names: Optional[Sequence[str]] = None
    activation: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.logits.ndim != 3:
            raise ValueError("SegmentationOutput.logits must be shaped (C, H, W)")
        act = self.activation.lower()
        if act not in {"sigmoid", "softmax"}:
            raise ValueError("SegmentationOutput.activation must be 'sigmoid' or 'softmax'")


@dataclass
class FusionWeights:
    det: float = 0.4
    seg: float = 0.4
    obj: float = 0.2

    def normalized(self) -> Tuple[float, float, float]:
        weights = np.array([self.det, self.seg, self.obj], dtype=np.float32)
        total = float(weights.sum())
        if total <= 1e-8:
            return 0.4, 0.4, 0.2
        weights /= total
        return float(weights[0]), float(weights[1]), float(weights[2])


@dataclass
class PromptFusionConfig:
    """Configuration controlling detection/segmentation fusion into SAM2 prompts."""

    objectness_threshold: float = 0.45
    segmentation_threshold: float = 0.4
    class_probability_threshold: float = 0.35
    detection_score_threshold: float = 0.25
    min_component_area: int = 60
    min_prompt_area: int = 60
    max_prompts_per_class: int = 10
    nms_iou_threshold: float = 0.6
    score_weights: FusionWeights = field(default_factory=FusionWeights)
    include_mask_prompts: bool = True


@dataclass
class PromptPostProcessConfig:
    """Morphological refinements applied to SAM2 outputs."""

    enable: bool = True
    closing_kernel: int = 5
    opening_kernel: int = 3
    min_instance_area: int = 60


@dataclass
class ClassAwarePrompt:
    """Prompt description used for SAM2 inference."""

    box: Tuple[int, int, int, int]
    point: Tuple[int, int]
    mask_seed: Optional[np.ndarray]
    class_id: int
    class_name: str
    score: float
    processed_box: Optional[Tuple[int, int, int, int]] = None
    processed_point: Optional[Tuple[int, int]] = None
    mask_seed_processed: Optional[np.ndarray] = None


@dataclass
class ClassAwareInstance:
    """Final instance prediction returned by the pipeline."""

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    score: float
    prompt: ClassAwarePrompt


@dataclass
class ClassAwarePipelineResult:
    """Aggregate outputs produced by the class-aware pipeline."""

    instances: List[ClassAwareInstance]
    prompts: List[ClassAwarePrompt]
    detection: DetectionOutput
    segmentation: SegmentationOutput
    objectness_map: np.ndarray
    attention_map: Optional[np.ndarray]
    patch_map: np.ndarray
    segmentation_probs: np.ndarray
    processed_shape: Tuple[int, int]
    original_shape: Tuple[int, int]
    detection_processed: Optional[DetectionOutput] = None


def _resize_map(data: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    height, width = image_size
    if cv2 is not None:
        return cv2.resize(data, (width, height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(data).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _clip_box(box: Sequence[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = image_size
    x1, y1, x2, y2 = box
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), 0, width - 1))
    y2 = int(np.clip(np.ceil(y2), 0, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _scale_box_to_image(
    box: Tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = box
    x1 = int(round(x1 * scale_x))
    y1 = int(round(y1 * scale_y))
    x2 = int(round((x2 + 1) * scale_x)) - 1
    y2 = int(round((y2 + 1) * scale_y)) - 1
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _scale_point_to_image(
    point: Tuple[int, int], scale_x: float, scale_y: float, image_shape: Tuple[int, int]
) -> Tuple[int, int]:
    height, width = image_shape
    x, y = point
    x_img = int(np.clip(round(x * scale_x), 0, width - 1))
    y_img = int(np.clip(round(y * scale_y), 0, height - 1))
    return x_img, y_img


def _resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if mask.shape == (target_h, target_w):
        return mask.astype(np.uint8)
    if cv2 is not None:
        resized = cv2.resize(
            mask.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(np.uint8)
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).to(dtype=torch.uint8).numpy()


def _connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    binary = mask.astype(np.uint8)
    if cv2 is not None:
        return cv2.connectedComponents(binary, connectivity=8)

    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current = 1
    stack: List[Tuple[int, int]] = []
    neighbours = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for y in range(height):
        for x in range(width):
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue
            stack.append((y, x))
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbours:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if binary[ny, nx] == 0 or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = current
                    stack.append((ny, nx))
            current += 1
    return current, labels


def _classwise_nms(prompts: List[ClassAwarePrompt], iou_thr: float) -> List[ClassAwarePrompt]:
    results: List[ClassAwarePrompt] = []
    by_class: Dict[int, List[ClassAwarePrompt]] = {}
    for prompt in prompts:
        by_class.setdefault(prompt.class_id, []).append(prompt)
    for _, prompt_list in by_class.items():
        prompt_list.sort(key=lambda item: item.score, reverse=True)
        kept: List[ClassAwarePrompt] = []
        for candidate in prompt_list:
            if all(_iou(candidate.box, other.box) < iou_thr for other in kept):
                kept.append(candidate)
        results.extend(kept)
    return results


class ClassAwarePromptPipeline:
    """Pipeline that fuses detection and segmentation adapters into SAM2 prompts."""

    def __init__(
        self,
        dinov3_config: Dinov3Config,
        sam2_config: Sam2Config,
        fusion_config: PromptFusionConfig,
        postprocess_config: PromptPostProcessConfig,
        detection_adapter: DetectionAdapter,
        segmentation_adapter: SegmentationAdapter,
        foreground_class_ids: Optional[Iterable[int]] = None,
        background_class_ids: Optional[Iterable[int]] = None,
        device: str = "cuda",
        dtype: torch.dtype | str = torch.float32,
        extractor: Optional[DINOv3FeatureExtractor] = None,
        segmenter: Optional[SAM2Segmenter] = None,
    ) -> None:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.extractor = extractor or DINOv3FeatureExtractor(dinov3_config, device, torch_dtype)
        self.segmenter = segmenter or SAM2Segmenter(sam2_config, device, torch_dtype)
        self.detection_adapter = detection_adapter
        self.segmentation_adapter = segmentation_adapter
        self.fusion_config = fusion_config
        self.postprocess_config = postprocess_config
        self.foreground_class_ids = (
            set(int(idx) for idx in foreground_class_ids)
            if foreground_class_ids is not None
            else None
        )
        self.background_class_ids = (
            set(int(idx) for idx in background_class_ids)
            if background_class_ids is not None
            else set()
        )

    def _compute_foreground_mask(
        self,
        seg_probs: np.ndarray,
        obj_map: np.ndarray,
        candidate_classes: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.fusion_config
        obj_mask = obj_map > cfg.objectness_threshold
        if len(candidate_classes) == 0:
            return obj_mask, np.zeros_like(obj_map), np.zeros_like(obj_map, dtype=np.int32)
        class_probs = seg_probs[candidate_classes, :, :]
        max_probs = class_probs.max(axis=0)
        class_indices = class_probs.argmax(axis=0)
        fg_mask = (max_probs > cfg.segmentation_threshold) & obj_mask
        mapped_indices = np.zeros_like(class_indices)
        for idx, class_id in enumerate(candidate_classes):
            mapped_indices[class_indices == idx] = class_id
        return fg_mask, max_probs, mapped_indices

    def _generate_prompts(
        self,
        detections: DetectionOutput,
        seg_probs: np.ndarray,
        obj_map: np.ndarray,
        processed_shape: Tuple[int, int],
        original_shape: Tuple[int, int],
        class_names: Sequence[str],
        fg_mask: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> List[ClassAwarePrompt]:
        cfg = self.fusion_config
        weights = cfg.score_weights.normalized()
        prompts: List[ClassAwarePrompt] = []

        for box, cls_id, det_score in zip(detections.boxes, detections.class_ids, detections.scores):
            class_id = int(cls_id)
            if det_score < cfg.detection_score_threshold:
                continue
            if self.foreground_class_ids is not None and class_id not in self.foreground_class_ids:
                continue
            if class_id in self.background_class_ids:
                continue
            if class_id < 0 or class_id >= seg_probs.shape[0]:
                continue

            prob_map = seg_probs[class_id]
            mask_c = (prob_map > cfg.class_probability_threshold) & fg_mask
            if not mask_c.any():
                continue

            x1, y1, x2, y2 = _clip_box(box, processed_shape)
            if x2 <= x1 or y2 <= y1:
                continue

            local_mask = np.zeros_like(mask_c, dtype=bool)
            local_mask[y1 : y2 + 1, x1 : x2 + 1] = True
            component_mask = (mask_c & local_mask).astype(np.uint8)
            if component_mask.sum() < cfg.min_component_area:
                continue

            num_labels, labels = _connected_components(component_mask)
            for label_idx in range(1, num_labels):
                region = (labels == label_idx)
                if int(region.sum()) < cfg.min_prompt_area:
                    continue

                ys, xs = np.where(region)
                region_probs = prob_map[region]
                if region_probs.size == 0:
                    continue
                prob_peak = float(region_probs.max())
                obj_mean = float(obj_map[region].mean()) if region.sum() > 0 else 0.0
                det_w, seg_w, obj_w = weights
                score = det_w * float(det_score) + seg_w * prob_peak + obj_w * obj_mean

                weighted_x = float((xs * region_probs).sum() / (region_probs.sum() + 1e-6))
                weighted_y = float((ys * region_probs).sum() / (region_probs.sum() + 1e-6))
                processed_point = (int(round(weighted_x)), int(round(weighted_y)))

                box_mask = np.zeros_like(region, dtype=np.uint8)
                box_mask[region] = 1
                processed_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                box = _scale_box_to_image(processed_bbox, scale_x, scale_y, original_shape)
                point = _scale_point_to_image(processed_point, scale_x, scale_y, original_shape)
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                mask_seed_processed = box_mask if self.fusion_config.include_mask_prompts else None
                mask_seed = (
                    _resize_mask_to_shape(mask_seed_processed, original_shape)
                    if mask_seed_processed is not None
                    else None
                )

                prompts.append(
                    ClassAwarePrompt(
                        box=box,
                        point=point,
                        mask_seed=mask_seed,
                        class_id=class_id,
                        class_name=class_name,
                        score=score,
                        processed_box=processed_bbox,
                        processed_point=processed_point,
                        mask_seed_processed=mask_seed_processed,
                    )
                )

        if not prompts:
            return []

        prompts = _classwise_nms(prompts, cfg.nms_iou_threshold)

        if cfg.max_prompts_per_class > 0:
            capped: List[ClassAwarePrompt] = []
            prompts_by_class: Dict[int, List[ClassAwarePrompt]] = {}
            for prompt in prompts:
                prompts_by_class.setdefault(prompt.class_id, []).append(prompt)
            for cls_prompts in prompts_by_class.values():
                cls_prompts.sort(key=lambda item: item.score, reverse=True)
                capped.extend(cls_prompts[: cfg.max_prompts_per_class])
            prompts = capped

        prompts.sort(key=lambda item: item.score, reverse=True)
        return prompts

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        cfg = self.postprocess_config
        if not cfg.enable:
            return mask
        if cv2 is None:
            return mask.astype(np.uint8)
        kernel_close = max(1, int(cfg.closing_kernel))
        kernel_open = max(1, int(cfg.opening_kernel))
        mask_proc = mask.astype(np.uint8)
        if kernel_close > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
            mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_CLOSE, kernel)
        if kernel_open > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
            mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_OPEN, kernel)
        return mask_proc

    @torch.inference_mode()
    def run(self, image: np.ndarray) -> ClassAwarePipelineResult:
        """Execute the class-aware pipeline on the provided RGB image."""

        LOGGER.debug(
            "Running zero-shot class-aware pipeline: DINOv3 features → adapters → SAM2 refinement"
        )

        features = self.extractor.extract_features(image)
        patch_tokens = features["patch_tokens"]
        grid_size = tuple(features["grid_size"])
        processed_shape = tuple(features["processed_image_shape"])
        original_shape = image.shape[:2]
        objectness_map = features.get("objectness_map")
        patch_map = features.get("patch_map")
        attention_map = features.get("attention_map")
        if attention_map is not None:
            attention_map = _resize_map(attention_map, processed_shape).astype(np.float32)
        if patch_map is None:
            raise ValueError("Feature extractor must provide patch_map for visualization")
        patch_map_np = np.asarray(patch_map)

        if objectness_map is None:
            LOGGER.warning("Objectness map not provided by extractor; defaulting to uniform foreground")
            objectness_map = np.ones(grid_size, dtype=np.float32)

        det_output = self.detection_adapter.predict(patch_tokens, processed_shape[::-1], grid_size)
        seg_output = self.segmentation_adapter.predict(patch_tokens, processed_shape[::-1], grid_size)

        det_processed = DetectionOutput(
            boxes=np.asarray(det_output.boxes, dtype=np.float32).copy(),
            class_ids=np.asarray(det_output.class_ids, dtype=np.int32).copy(),
            scores=np.asarray(det_output.scores, dtype=np.float32).copy(),
            class_names=det_output.class_names,
        )

        class_names: List[str] = []
        if seg_output.class_names is not None:
            class_names = list(seg_output.class_names)
        elif det_output.class_names is not None:
            class_names = list(det_output.class_names)
        if not class_names:
            class_names = [f"class_{idx}" for idx in range(seg_output.logits.shape[0])]

        obj_map_resized = _resize_map(objectness_map, processed_shape).astype(np.float32)

        logits = seg_output.logits
        if seg_output.activation.lower() == "softmax":
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        else:
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        probs = np.asarray(probs, dtype=np.float32)

        processed_h, processed_w = processed_shape
        orig_h, orig_w = original_shape
        scale_x = float(orig_w) / float(max(processed_w, 1))
        scale_y = float(orig_h) / float(max(processed_h, 1))

        scaled_boxes: List[Tuple[int, int, int, int]] = []
        for box in det_output.boxes:
            scaled_boxes.append(
                _scale_box_to_image(
                    (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    scale_x,
                    scale_y,
                    original_shape,
                )
            )
        det_scaled = DetectionOutput(
            boxes=np.asarray(scaled_boxes, dtype=np.float32) if scaled_boxes else np.zeros((0, 4), dtype=np.float32),
            class_ids=np.asarray(det_output.class_ids, dtype=np.int32).copy(),
            scores=np.asarray(det_output.scores, dtype=np.float32).copy(),
            class_names=det_output.class_names,
        )

        candidate_classes: Sequence[int]
        if self.foreground_class_ids is not None:
            candidate_classes = sorted(self.foreground_class_ids)
        else:
            candidate_classes = [
                idx
                for idx in range(probs.shape[0])
                if idx not in self.background_class_ids
            ]

        fg_mask, _, _ = self._compute_foreground_mask(probs, obj_map_resized, candidate_classes)

        prompts = self._generate_prompts(
            det_output,
            probs,
            obj_map_resized,
            processed_shape,
            original_shape,
            class_names,
            fg_mask,
            scale_x,
            scale_y,
        )

        if not prompts:
            LOGGER.info("No prompts survived fusion; returning empty result")
            return ClassAwarePipelineResult(
                instances=[],
                prompts=[],
                detection=det_scaled,
                segmentation=seg_output,
                objectness_map=obj_map_resized,
                attention_map=attention_map,
                patch_map=patch_map_np,
                segmentation_probs=probs,
                processed_shape=processed_shape,
                original_shape=original_shape,
                detection_processed=det_processed,
            )

        boxes = [prompt.box for prompt in prompts]
        points = [[prompt.point] for prompt in prompts]
        labels = [[1] for _ in prompts]
        mask_inputs: Optional[List[np.ndarray]]
        if self.fusion_config.include_mask_prompts:
            mask_inputs = [prompt.mask_seed for prompt in prompts]
            if any(seed is None for seed in mask_inputs):
                mask_inputs = [
                    prompt.mask_seed
                    if prompt.mask_seed is not None
                    else np.zeros(original_shape, dtype=np.uint8)
                    for prompt in prompts
                ]
        else:
            mask_inputs = None
        if mask_inputs is not None and len(mask_inputs) == 0:
            mask_inputs = None

        masks = self.segmenter.segment(
            image,
            boxes=boxes,
            points=points,
            labels=labels,
            mask_inputs=mask_inputs,
        )

        instances: List[ClassAwareInstance] = []
        min_area = max(0, int(self.postprocess_config.min_instance_area))
        for prompt, mask in zip(prompts, masks):
            processed_mask = self._postprocess_mask(mask)
            area = int(processed_mask.sum())
            if area < min_area:
                continue
            bbox = _bbox_from_mask(processed_mask)
            instances.append(
                ClassAwareInstance(
                    mask=processed_mask.astype(np.uint8),
                    bbox=bbox,
                    class_id=prompt.class_id,
                    class_name=prompt.class_name,
                    score=prompt.score,
                    prompt=prompt,
                )
            )

        return ClassAwarePipelineResult(
            instances=instances,
            prompts=prompts,
            detection=det_scaled,
            segmentation=seg_output,
            objectness_map=obj_map_resized,
            attention_map=attention_map,
            patch_map=patch_map_np,
            segmentation_probs=probs,
            processed_shape=processed_shape,
            original_shape=original_shape,
            detection_processed=det_processed,
        )

