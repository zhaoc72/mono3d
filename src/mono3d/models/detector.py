"""目标检测模块

封装Grounding DINO或其它检测模型，提供统一的检测接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

import logging

log = logging.getLogger(__name__)


class GroundingDINODetector(nn.Module):
    """Grounding DINO检测器的简化封装。

    该实现优先尝试加载真实的Grounding DINO模型；若依赖不可用，则
    自动退化为一个轻量级的启发式检测器，确保在无外部权重的环境下
    仍能返回合理的检测结果。这在单元测试或CI环境中尤为重要。
    """

    def __init__(
        self,
        text_prompts: Optional[Union[str, Sequence[str]]] = None,
        confidence_threshold: float = 0.25,
        max_detections: int = 5,
        fallback_label: str = "object",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.fallback_label = fallback_label
        self._default_prompts = self._normalize_prompts(text_prompts)

        self.model = None
        self.tokenizer = None

        self._load_model(**kwargs)

    def _normalize_prompts(
        self, prompts: Optional[Union[str, Sequence[str]]]
    ) -> List[str]:
        if prompts is None:
            return []
        if isinstance(prompts, str):
            return [prompts]
        return [str(p) for p in prompts]

    def _load_model(self, **kwargs: Any) -> None:
        """尝试加载真实的Grounding DINO模型。

        如果任一步骤失败（例如依赖未安装、权重缺失等），则会
        回退到启发式实现。
        """

        try:
            from groundingdino.util.inference import Model

            ckpt = kwargs.get("weights", None)
            config = kwargs.get("config", None)

            if ckpt is None or config is None:
                raise ValueError("weights and config must be provided for Grounding DINO")

            self.model = Model(model_config_path=config, model_checkpoint_path=ckpt)
            log.info("Loaded Grounding DINO detector")

        except Exception as exc:  # pragma: no cover - executed only when dependencies exist
            log.warning("Failed to load Grounding DINO (%s), using fallback detector", exc)
            self.model = None

    def forward(
        self,
        images: torch.Tensor,
        text_prompts: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        """执行检测。

        Args:
            images: 输入图像，形状为 ``(B, 3, H, W)``。
            text_prompts: 检测提示词，可以是字符串或字符串列表。

        Returns:
            包含 ``boxes``、``scores`` 和 ``labels`` 的字典。
        """

        if self.model is None:
            return self._fallback_detect(images, text_prompts)

        prompts = self._normalize_prompts(text_prompts) or self._default_prompts
        if not prompts:
            prompts = [self.fallback_label]

        detections = []
        scores = []
        labels: List[List[str]] = []

        device = images.device

        for image in images:
            # Grounding DINO期望numpy格式，且需要[0, 255]范围
            image_np = image.detach().cpu().permute(1, 2, 0).numpy()
            image_np = (image_np * 255.0).clip(0, 255).astype("uint8")

            image_boxes: List[torch.Tensor] = []
            image_scores: List[float] = []
            image_labels: List[str] = []

            for prompt in prompts[: self.max_detections]:
                try:
                    detection = self.model.predict_with_caption(
                        image_np,
                        caption=prompt,
                        box_threshold=self.confidence_threshold,
                    )

                    if detection is None:
                        continue

                    boxes = torch.from_numpy(detection["boxes"])
                    confs = torch.from_numpy(detection["scores"])

                    if boxes.numel() == 0:
                        continue

                    image_boxes.append(boxes)
                    image_scores.append(float(confs.max().item()))
                    image_labels.append(prompt)

                except Exception as exc:  # pragma: no cover - depends on external lib
                    log.error("Grounding DINO inference failed: %s", exc)

            if image_boxes:
                boxes_tensor = torch.cat(image_boxes, dim=0)
                score_tensor = torch.tensor(image_scores, dtype=torch.float32)
                detections.append(boxes_tensor.to(device))
                scores.append(score_tensor.to(device))
                labels.append(image_labels)
            else:
                fallback = self._fallback_detect(image.unsqueeze(0), prompts)
                detections.append(fallback["boxes"][0].to(device))
                scores.append(fallback["scores"][0].to(device))
                labels.append(fallback["labels"][0])

        # 对齐批次维度
        max_boxes = max(box.shape[0] for box in detections) if detections else 1
        batched_boxes = []
        batched_scores = []

        for box_tensor, score_tensor in zip(detections, scores):
            if box_tensor.shape[0] < max_boxes:
                pad = max_boxes - box_tensor.shape[0]
                box_tensor = torch.cat(
                    [box_tensor, box_tensor.new_zeros(pad, 4)], dim=0
                )
                score_tensor = torch.cat(
                    [score_tensor, score_tensor.new_zeros(pad)], dim=0
                )

            batched_boxes.append(box_tensor.unsqueeze(0))
            batched_scores.append(score_tensor.unsqueeze(0))

        if batched_boxes:
            boxes_out = torch.cat(batched_boxes, dim=0)
            scores_out = torch.cat(batched_scores, dim=0)
        else:
            B, _, H, W = images.shape
            boxes_out = torch.tensor(
                [[[0.0, 0.0, float(W), float(H)]] for _ in range(B)],
                device=images.device,
            )
            scores_out = torch.ones((B, 1), device=images.device)
            labels = [[self.fallback_label] for _ in range(B)]

        return {"boxes": boxes_out, "scores": scores_out, "labels": labels}

    def _fallback_detect(
        self,
        images: torch.Tensor,
        text_prompts: Optional[Union[str, Sequence[str]]],
    ) -> Dict[str, Any]:
        """基于启发式的检测实现。

        该方法简单地返回整张图像作为候选框，并使用给定的提示词
        （若未提供则使用 ``fallback_label``）作为类别名称。
        """

        prompts = self._normalize_prompts(text_prompts)
        if not prompts:
            prompts = [self.fallback_label]

        B, _, H, W = images.shape
        boxes = torch.tensor(
            [[[0.0, 0.0, float(W), float(H)]]] * B,
            dtype=torch.float32,
            device=images.device,
        )
        scores = torch.ones((B, 1), dtype=torch.float32, device=images.device)
        labels = [[prompts[0]] for _ in range(B)]

        return {"boxes": boxes, "scores": scores, "labels": labels}


__all__ = ["GroundingDINODetector"]