"""DINOv3 detection & segmentation inference with Accelerate auto-sharding.

该脚本提供一个统一入口，利用 Hugging Face Accelerate 的 ``device_map``
能力在多张 GPU 上运行 ViT-7B 级别的 DINOv3 模型。脚本会自动：

* 通过 ``torch.hub`` 加载官方 backbone + detection/segmentation adapter；
* 基于 Accelerate ``infer_auto_device_map`` 推理模型的模块切分方案；
* 将模型参数分发到多卡（默认 `device_map="auto"`）；
* 统一使用 float32 精度执行检测、语义分割，或一次同时完成二者；
* 依据 `--save-visuals` 选项输出检测框/语义伪彩等中间可视化结果。

使用示例：

```bash
python tools/dinov3_accelerate_inference.py \
    --repo ../dinov3 \
    --task both \
    --backbone /path/to/dinov3_vit7b16_lvd1689m.pth \
    --detection-adapter /path/to/dinov3_vit7b16_coco.pth \
    --segmentation-adapter /path/to/dinov3_vit7b16_ade20k.pth \
    --input /path/to/images \
    --output /path/to/save \
    --device-map auto \
    --save-visuals
```

注意：

* 脚本假设存在 4 张 RTX 4090，且 CUDA 驱动可用；
* 检测部分依赖 Detectron2，请确保官方 DINOv3 仓库安装了相关依赖；
* 语义分割依赖 ``dinov3.eval.segmentation.inference.make_inference``。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 Accelerate 推理脚本")
    parser.add_argument("--repo", required=True, help="本地 DINOv3 仓库路径")
    parser.add_argument(
        "--task",
        required=True,
        choices=["detection", "segmentation", "both"],
        help="选择执行检测、语义分割或二者同时",
    )
    parser.add_argument("--backbone", required=True, help="ViT-7B backbone checkpoint")
    parser.add_argument(
        "--detection-adapter",
        help="检测 Adapter checkpoint，task 包含 detection 时必需",
    )
    parser.add_argument(
        "--segmentation-adapter",
        help="分割 Adapter checkpoint，task 包含 segmentation 时必需",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入图像目录或单张图片路径",
    )
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Accelerate device_map 配置，默认 auto",
    )
    parser.add_argument(
        "--visible-devices",
        default=None,
        help="显式设置 CUDA_VISIBLE_DEVICES，例如 0,1,2,3",
    )
    parser.add_argument(
        "--min-gpus",
        type=int,
        default=4,
        help="要求可见的最少 GPU 数，默认 4（四张 4090）",
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help="每张 GPU 的最大可用显存（GiB）。例如 22 或 22GiB，默认自动推断",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=896,
        help="输入图像统一 resize 大小，默认 896",
    )
    parser.add_argument("--save-visuals", action="store_true", help="保存检测/分割的可视化结果")
    parser.add_argument(
        "--color-map",
        default="Spectral",
        help="语义分割叠加图使用的 matplotlib colormap 名称",
    )
    parser.add_argument(
        "--no-split", nargs="*", default=["Block"], help="device_map 自动推理时禁止拆分的模块类名"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")


def resolve_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {path}")
    candidates: List[Path] = []
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidates.extend(path.rglob(f"*{suffix}"))
    return sorted(candidates)


def resolve_tasks(task: str) -> Tuple[bool, bool]:
    if task == "detection":
        return True, False
    if task == "segmentation":
        return False, True
    if task == "both":
        return True, True
    raise ValueError(f"未知 task: {task}")


def _normalize_memory_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("显存上限配置不能为空")
    if value.lower().endswith("gib"):
        value = value[:-3]
    numeric = float(value)
    return f"{numeric}GiB"


def _device_key_to_index(device_key: str) -> int:
    key = device_key.strip()
    if not key:
        raise ValueError("显卡编号不能为空")
    if key.startswith("cuda"):
        parts = key.split(":")
        if len(parts) == 1:
            raise ValueError(f"无法从 {device_key!r} 推断出显卡编号")
        key = parts[-1]
    return int(key)


def apply_visible_devices(spec: Optional[str]) -> None:
    if spec is None:
        return
    value = spec.strip()
    if not value:
        raise ValueError("--visible-devices 不能为空字符串")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    LOGGER.info("设置 CUDA_VISIBLE_DEVICES=%s", value)
    # 触发一次设备查询，确保新的环境变量生效
    if torch.cuda.is_available():
        torch.cuda.device_count()


def ensure_minimum_gpus(min_required: int) -> Sequence[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 设备，请检查驱动与环境配置")
    count = torch.cuda.device_count()
    indices = list(range(count))
    if min_required > 0 and count < min_required:
        raise RuntimeError(
            f"仅检测到 {count} 张 GPU，低于要求的 {min_required} 张，请检查 CUDA_VISIBLE_DEVICES 或驱动设置"
        )
    device_names = [torch.cuda.get_device_name(i) for i in indices]
    LOGGER.info("当前可见 GPU：%s", ", ".join(f"{i}:{name}" for i, name in zip(indices, device_names)))
    return indices


def _normalize_max_memory_keys(mapping: Optional[Mapping[int | str, str]]) -> Optional[Dict[int, str]]:
    if mapping is None:
        return None
    normalized: Dict[int, str] = {}
    for key, value in mapping.items():
        idx = _device_key_to_index(str(key))
        normalized[idx] = value
    return normalized or None


def build_max_memory(arg: Optional[str], available_devices: Sequence[int]) -> Optional[Dict[int, str]]:
    if not torch.cuda.is_available() or not available_devices:
        return None

    if arg is None:
        memory: Dict[int, str] = {}
        for idx in available_devices:
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory // (1024**3)
            usable = max(total_gb - 2, 1)
            memory[idx] = f"{usable}GiB"
        return memory

    memory: Dict[int, str] = {}
    if "," in arg:
        for segment in arg.split(","):
            if not segment.strip():
                continue
            if ":" not in segment:
                raise ValueError("--max-memory 自定义映射需要使用 device:value 形式，例如 cuda:0:21GiB")
            device_key, raw_value = segment.rsplit(":", 1)
            device_index = _device_key_to_index(device_key)
            if device_index not in available_devices:
                raise ValueError(
                    f"设备 {device_index} 不在当前可见 GPU 列表 {list(available_devices)} 中，请检查 --visible-devices"
                )
            memory[device_index] = _normalize_memory_value(raw_value)
        return memory or None

    normalized_value = _normalize_memory_value(arg)
    for idx in available_devices:
        memory[idx] = normalized_value
    return memory


def prepare_device_map(
    model: torch.nn.Module,
    *,
    device_map_option: str,
    max_memory: Optional[Dict[int, str]],
    no_split: Sequence[str],
) -> Tuple[Mapping[str, str] | str, Optional[str]]:
    if device_map_option != "auto":
        LOGGER.info("使用手动 device_map: %s", device_map_option)
        return device_map_option, None

    if not torch.cuda.is_available():
        LOGGER.warning("未检测到 CUDA，device_map=auto 将退化为 CPU 推理")
        return "cpu", None

    from accelerate.utils import get_balanced_memory, infer_auto_device_map

    normalized_max_memory = _normalize_max_memory_keys(max_memory)
    LOGGER.info("根据显存推断 device_map (max_memory=%s)", normalized_max_memory)
    dtype = next((param.dtype for param in model.parameters()), torch.float32)

    balanced: Optional[Mapping[str, str]] = None
    try:
        balanced = get_balanced_memory(
            model,
            max_memory=normalized_max_memory,
            no_split_module_classes=list(no_split),
            dtype=dtype,
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        LOGGER.debug("get_balanced_memory 失败，将直接使用用户提供的 max_memory: %s", exc)

    max_memory_for_infer = _normalize_max_memory_keys(balanced) or normalized_max_memory

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory_for_infer,
        no_split_module_classes=list(no_split),
        dtype=dtype,
    )

    if isinstance(device_map, Mapping):
        module_names = {name for name, _ in model.named_modules()}
        if "" not in module_names:
            module_names.add("")
        normalized: "OrderedDict[str, str]" = OrderedDict()
        remapped = []
        dropped = []
        for name, target in device_map.items():
            if name in module_names:
                normalized[name] = target
                continue
            ancestor = name
            matched = False
            while "." in ancestor:
                ancestor = ancestor.rsplit(".", 1)[0]
                if ancestor in module_names:
                    normalized[ancestor] = target
                    remapped.append((name, ancestor))
                    matched = True
                    break
            if not matched:
                dropped.append(name)

        if not normalized:
            # 若所有条目均未匹配，则回退为整体单设备执行
            fallback_device = next(iter(device_map.values()))
            normalized[""] = fallback_device
            LOGGER.warning(
                "device_map 归一化后为空，回退为整模型设备: %s", fallback_device
            )

        if remapped:
            LOGGER.debug("device_map 条目回退到父模块: %s", remapped)
        if dropped:
            LOGGER.debug("device_map 条目被丢弃（未找到模块）: %s", dropped)

        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("归一化前 device_map 条目数：%d", len(device_map))
        device_map = normalized
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("归一化后 device_map 条目数：%d", len(device_map))

    primary = next((device for device in device_map.values() if device != "cpu"), "cpu")
    LOGGER.info("推断得到 primary device: %s", primary)
    return device_map, primary


def dispatch_model_if_needed(
    hub_model,
    *,
    device_map: Mapping[str, str] | str,
) -> Tuple[torch.nn.Module, Optional[str]]:
    from accelerate import dispatch_model

    if isinstance(device_map, str):
        target_device = device_map
        if hasattr(hub_model, "model") and isinstance(hub_model.model, torch.nn.Module):
            LOGGER.info("将 Detectron2 模型移动到 %s", target_device)
            hub_model.model.to(target_device)
            return hub_model.model, target_device
        LOGGER.info("将模型移动到 %s", target_device)
        return hub_model.to(target_device), target_device  # type: ignore[return-value]

    module = hub_model
    attribute_owner = None
    if hasattr(hub_model, "model") and isinstance(hub_model.model, torch.nn.Module):
        module = hub_model.model
        attribute_owner = hub_model

    LOGGER.info("使用 Accelerate dispatch_model 进行模型分片")
    sharded = dispatch_model(module, device_map=device_map)
    if attribute_owner is not None:
        attribute_owner.model = sharded
    return sharded, next((device for device in set(device_map.values()) if device != "cpu"), "cpu")


def get_primary_device(device_map: Mapping[str, str] | str | None) -> str:
    if isinstance(device_map, str):
        return device_map
    if isinstance(device_map, Mapping):
        for device in device_map.values():
            if device != "cpu":
                return device
        return "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def coco_class_names() -> List[str]:
    # 80 类 COCO 类别，官方 detectron2 默认顺序
    return [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


def annotate_detections(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    classes: Sequence[int],
    scores: Sequence[float],
    class_names: Sequence[str],
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    for bbox, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = bbox
        label = class_names[int(cls)] if int(cls) < len(class_names) else str(cls)
        caption = f"{label}: {score:.2f}"
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
        text_w, text_h = draw.textsize(caption, font=font)
        text_top = max(y1 - text_h - 4, 0)
        bg_coords = [(x1, text_top), (x1 + text_w + 8, text_top + text_h + 4)]
        draw.rectangle(bg_coords, fill=(255, 0, 0))
        draw.text((x1 + 4, text_top + 2), caption, fill=(255, 255, 255), font=font)

    return annotated


def get_module_dtype(module) -> torch.dtype:
    """Try to read the dtype from a hub predictor or plain nn.Module."""

    if isinstance(module, torch.nn.Module):
        for parameter in module.parameters():
            return parameter.dtype
    if hasattr(module, "model") and isinstance(module.model, torch.nn.Module):
        for parameter in module.model.parameters():
            return parameter.dtype
    return torch.float32


def run_detection_inference(
    predictor,
    images: Iterable[Path],
    output_dir: Path,
    *,
    save_visuals: bool,
    device: Optional[str],
) -> None:
    import numpy as np

    class_names = coco_class_names()
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        LOGGER.info("[detection] 处理 %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        # detectron2 期望 BGR，确保数组为连续内存防止负 stride
        bgr_image = np.array(image)[:, :, ::-1].copy()
        try:
            outputs = predictor(bgr_image)
        except ValueError as exc:
            message = str(exc)
            if "not supported" not in message.lower():
                raise

            LOGGER.debug("Detectron2 numpy 输入失败，回退为 tensor batch：%s", message)
            tensor = torch.from_numpy(bgr_image).permute(2, 0, 1).contiguous()
            model_dtype = get_module_dtype(predictor)
            target_dtype = model_dtype if model_dtype.is_floating_point else torch.float32
            tensor = tensor.to(dtype=target_dtype)
            if device is not None:
                tensor = tensor.to(device=device, dtype=target_dtype)
            outputs = predictor([tensor])
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        classes = instances.pred_classes.numpy().tolist()

        records = []
        for bbox, score, cls in zip(boxes, scores, classes):
            record = {
                "bbox_xyxy": bbox,
                "score": float(score),
                "category_id": int(cls),
                "category_name": class_names[int(cls)] if int(cls) < len(class_names) else str(cls),
            }
            records.append(record)

        with (output_dir / f"{image_path.stem}_detections.json").open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)

        if save_visuals:
            annotated = annotate_detections(image, boxes, classes, scores, class_names)
            annotated.save(output_dir / f"{image_path.stem}_detection_overlay.png")


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------


def build_segmentation_transform(image_size: int) -> v2.Compose:
    """匹配官方示例的图像预处理流程。"""

    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def run_segmentation_inference(
    segmentor: torch.nn.Module,
    images: Iterable[Path],
    output_dir: Path,
    *,
    device: str,
    dtype: torch.dtype,
    image_size: int,
    save_visuals: bool,
    color_map: str,
) -> None:
    import matplotlib
    import numpy as np
    from functools import partial
    from dinov3.eval.segmentation.inference import make_inference

    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap(color_map)
    transform = build_segmentation_transform(image_size)

    for image_path in images:
        LOGGER.info("[segmentation] 处理 %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        tensor = tensor.to(device, dtype=dtype)

        with torch.inference_mode():
            autocast_device = "cuda" if device.startswith("cuda") else "cpu"
            if dtype in (torch.float16, torch.bfloat16):
                autocast_context = torch.autocast(autocast_device, dtype=dtype)
            else:
                autocast_context = nullcontext()
            with autocast_context:
                _ = segmentor(tensor)
                segmentation = make_inference(
                    tensor,
                    segmentor,
                    inference_mode="slide",
                    decoder_head_type="m2f",
                    rescale_to=image.size[::-1],
                    n_output_channels=150,
                    crop_size=(image_size, image_size),
                    stride=(image_size, image_size),
                    output_activation=partial(torch.nn.functional.softmax, dim=1),
                ).argmax(dim=1, keepdim=True)

        array = segmentation[0, 0].to("cpu", dtype=torch.int32).numpy().astype("uint16")
        seg_image = Image.fromarray(array, mode="I;16")
        seg_image.save(output_dir / f"{image_path.stem}_segmentation.png")

        if save_visuals:
            normalized = array.astype("float32") / max(array.max(), 1)
            colored = cmap(normalized)[..., :3]
            overlay = Image.fromarray((colored * 255).astype("uint8"))
            overlay.save(output_dir / f"{image_path.stem}_overlay.png")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    apply_visible_devices(args.visible_devices)

    repo_path = resolve_path(args.repo)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output)

    LOGGER.info("加载图像：%s", input_path)
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"未找到任何图像：{input_path}")

    need_detection, need_segmentation = resolve_tasks(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    if need_detection and not args.detection_adapter:
        raise ValueError("task 包含 detection 时必须提供 --detection-adapter")
    if need_segmentation and not args.segmentation_adapter:
        raise ValueError("task 包含 segmentation 时必须提供 --segmentation-adapter")

    dtype = torch.float32
    LOGGER.info("推理统一使用 float32 精度")

    available_devices: Sequence[int] = []
    if args.device_map == "auto":
        available_devices = ensure_minimum_gpus(args.min_gpus)
    max_memory = build_max_memory(args.max_memory, available_devices)

    detection_model = None
    detection_device = None
    segmentation_model = None
    segmentation_device = None

    if need_detection:
        LOGGER.info("通过 torch.hub 加载检测模型")
        detection_model = torch.hub.load(
            str(repo_path),
            "dinov3_vit7b16_de",
            source="local",
            weights=args.detection_adapter,
            backbone_weights=args.backbone,
        )

        if hasattr(detection_model, "model") and isinstance(detection_model.model, torch.nn.Module):
            detection_model.model = detection_model.model.to(dtype=dtype)
        elif isinstance(detection_model, torch.nn.Module):
            detection_model = detection_model.to(dtype=dtype)

        device_map_det, primary_det = prepare_device_map(
            detection_model.model if hasattr(detection_model, "model") else detection_model,
            device_map_option=args.device_map,
            max_memory=max_memory,
            no_split=args.no_split,
        )
        _, dispatch_primary_det = dispatch_model_if_needed(detection_model, device_map=device_map_det)
        detection_device = primary_det or dispatch_primary_det or get_primary_device(device_map_det)
        LOGGER.info("检测模型主设备：%s", detection_device)

    if need_segmentation:
        LOGGER.info("通过 torch.hub 加载分割模型")
        segmentation_model = torch.hub.load(
            str(repo_path),
            "dinov3_vit7b16_ms",
            source="local",
            weights=args.segmentation_adapter,
            backbone_weights=args.backbone,
        )

        if isinstance(segmentation_model, torch.nn.Module):
            segmentation_model = segmentation_model.to(dtype=dtype)

        device_map_seg, primary_seg = prepare_device_map(
            segmentation_model,
            device_map_option=args.device_map,
            max_memory=max_memory,
            no_split=args.no_split,
        )
        segmentation_model, dispatch_primary = dispatch_model_if_needed(
            segmentation_model, device_map=device_map_seg
        )
        segmentation_device = primary_seg or dispatch_primary or get_primary_device(device_map_seg)
        LOGGER.info("分割模型主设备：%s", segmentation_device)

    if detection_model is not None:
        run_detection_inference(
            detection_model,
            images,
            output_dir / "detection" if need_segmentation else output_dir,
            save_visuals=args.save_visuals,
            device=detection_device,
        )

    if segmentation_model is not None and segmentation_device is not None:
        run_segmentation_inference(
            segmentation_model,
            images,
            output_dir / "segmentation" if need_detection else output_dir,
            device=segmentation_device,
            dtype=dtype,
            image_size=args.image_size,
            save_visuals=args.save_visuals,
            color_map=args.color_map,
        )

    LOGGER.info("推理完成，结果保存在 %s", output_dir)
    
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
