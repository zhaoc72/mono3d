"""DINOv3 detection & segmentation inference - 单卡/多卡自适应版本."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 推理脚本 (单卡/多卡自适应)")
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
    
    # 设备配置
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default=None,
        help="单卡模式：指定单个 GPU (如 cuda:0)",
    )
    device_group.add_argument(
        "--device-map",
        default=None,
        help="多卡模式：Accelerate device_map (如 auto)",
    )
    
    parser.add_argument(
        "--visible-devices",
        default=None,
        help="多卡模式：显式设置 CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--min-gpus",
        type=int,
        default=0,
        help="多卡模式：要求可见的最少 GPU 数",
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help="多卡模式：每张 GPU 的最大可用显存（GiB）",
    )
    
    # 检测参数
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="检测置信度阈值，默认 0.5",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="每张图像最多保留的检测数，默认 100",
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=896,
        help="输入图像统一 resize 大小",
    )
    parser.add_argument("--save-visuals", action="store_true", help="保存可视化结果")
    parser.add_argument(
        "--color-map",
        default="Spectral",
        help="语义分割 colormap",
    )
    parser.add_argument(
        "--no-split", nargs="*", default=["Block"], help="多卡模式：禁止拆分的模块"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="启动前清理 torch.hub 缓存",
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
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s"
    )


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


def clear_torch_hub_cache() -> None:
    """清理 torch.hub 缓存目录"""
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    if cache_dir.exists():
        try:
            LOGGER.info("清理 torch.hub 缓存: %s", cache_dir)
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("缓存清理完成")
        except Exception as exc:
            LOGGER.warning("无法清理缓存: %s", exc)


def verify_checkpoint(checkpoint_path: Path) -> bool:
    """验证 checkpoint 文件是否完整"""
    if not checkpoint_path.exists():
        LOGGER.error("Checkpoint 文件不存在: %s", checkpoint_path)
        return False
    
    file_size_gb = checkpoint_path.stat().st_size / (1024**3)
    LOGGER.info("Checkpoint 文件大小: %.2f GB", file_size_gb)
    
    if "lvd1689m" in checkpoint_path.name and file_size_gb < 15:
        LOGGER.error("=" * 70)
        LOGGER.error("Checkpoint 文件可能不完整！")
        LOGGER.error("当前大小: %.2f GB，期望大小: ~16 GB", file_size_gb)
        LOGGER.error("请重新下载: %s", checkpoint_path)
        LOGGER.error("=" * 70)
        return False
    
    try:
        LOGGER.info("验证 checkpoint 文件完整性...")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        LOGGER.info("✓ Checkpoint 文件验证通过")
        del state_dict
        return True
    except Exception as exc:
        LOGGER.error("=" * 70)
        LOGGER.error("Checkpoint 文件损坏: %s", exc)
        LOGGER.error("文件路径: %s", checkpoint_path)
        LOGGER.error("")
        LOGGER.error("解决方案：")
        LOGGER.error("1. 删除损坏的文件:")
        LOGGER.error("   rm '%s'", checkpoint_path)
        LOGGER.error("2. 清理 torch.hub 缓存:")
        LOGGER.error("   rm -rf ~/.cache/torch/hub/checkpoints/*")
        LOGGER.error("3. 重新下载完整的 checkpoint 文件")
        LOGGER.error("=" * 70)
        return False


def apply_visible_devices(spec: Optional[str]) -> None:
    if spec is None:
        return
    value = spec.strip()
    if not value:
        raise ValueError("--visible-devices 不能为空")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    LOGGER.info("设置 CUDA_VISIBLE_DEVICES=%s", value)
    if torch.cuda.is_available():
        torch.cuda.device_count()


def ensure_minimum_gpus(min_required: int) -> Sequence[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 设备")
    count = torch.cuda.device_count()
    indices = list(range(count))
    if min_required > 0 and count < min_required:
        raise RuntimeError(f"仅检测到 {count} 张 GPU，低于要求的 {min_required} 张")
    device_names = [torch.cuda.get_device_name(i) for i in indices]
    LOGGER.info("当前可见 GPU：%s", ", ".join(f"{i}:{name}" for i, name in zip(indices, device_names)))
    return indices


@contextmanager
def temporarily_disable_cuda_for_hub() -> Iterator[bool]:
    """暂时屏蔽 CUDA，让 torch.hub 在 CPU 上初始化"""
    if not torch.cuda.is_available():
        yield False
        return

    original_is_available = torch.cuda.is_available
    original_device_count = torch.cuda.device_count
    visible_backup = os.environ.get("CUDA_VISIBLE_DEVICES")

    def _false() -> bool:
        return False

    def _zero() -> int:
        return 0

    torch.cuda.is_available = _false  # type: ignore[assignment]
    torch.cuda.device_count = _zero  # type: ignore[assignment]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        yield True
    finally:
        torch.cuda.is_available = original_is_available  # type: ignore[assignment]
        torch.cuda.device_count = original_device_count  # type: ignore[assignment]
        if visible_backup is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_backup
        try:
            torch.cuda.device_count()
        except Exception:
            pass


def load_torch_hub_model(repo: Path, entrypoint: str, **kwargs):
    """加载 torch.hub 模型，在 CPU 上初始化"""
    load_kwargs = dict(kwargs)
    load_kwargs.setdefault("source", "local")
    
    with temporarily_disable_cuda_for_hub() as disabled:
        try:
            model = torch.hub.load(str(repo), entrypoint, **load_kwargs)
        except RuntimeError as exc:
            error_msg = str(exc)
            if "PytorchStreamReader failed" in error_msg or "failed finding central directory" in error_msg:
                LOGGER.error("=" * 70)
                LOGGER.error("Checkpoint 文件已损坏！")
                LOGGER.error("=" * 70)
                LOGGER.error("错误: %s", exc)
                LOGGER.error("")
                LOGGER.error("修复步骤:")
                LOGGER.error("1. 清理 torch.hub 缓存:")
                LOGGER.error("   rm -rf ~/.cache/torch/hub/checkpoints/*")
                LOGGER.error("")
                LOGGER.error("2. 验证你的 checkpoint 文件:")
                if kwargs.get("backbone_weights"):
                    LOGGER.error("   ls -lh %s", kwargs["backbone_weights"])
                if kwargs.get("weights"):
                    LOGGER.error("   ls -lh %s", kwargs["weights"])
                LOGGER.error("")
                LOGGER.error("3. 如果文件不完整，请重新下载")
                LOGGER.error("=" * 70)
            raise

    if disabled and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 确保模型在 CPU 上
    if isinstance(model, torch.nn.Module):
        model.to("cpu")
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        model.model.to("cpu")

    return model


def build_max_memory(arg: Optional[str], available_devices: Sequence[int]) -> Optional[Dict[int, str]]:
    """构建 max_memory 配置（多卡模式）"""
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
    value = arg.strip()
    if value.lower().endswith("gib"):
        value = value[:-3]
    for idx in available_devices:
        memory[idx] = f"{float(value)}GiB"
    return memory


def prepare_device_map(
    model: torch.nn.Module,
    *,
    device_map_option: str,
    max_memory: Optional[Dict[int, str]],
    no_split: Sequence[str],
) -> Tuple[Mapping[str, str] | str, Optional[str]]:
    """多卡模式：准备 device_map"""
    if device_map_option != "auto":
        LOGGER.info("使用手动 device_map: %s", device_map_option)
        return device_map_option, None

    if not torch.cuda.is_available():
        LOGGER.warning("未检测到 CUDA，使用 CPU")
        return "cpu", None

    from accelerate.utils import infer_auto_device_map

    dtype = torch.float32  # 统一使用 float32
    
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=list(no_split),
        dtype=dtype,
    )

    primary = next((device for device in device_map.values() if device != "cpu"), "cpu")
    LOGGER.info("推断 device_map，primary device: %s", primary)
    return device_map, primary


def dispatch_model_if_needed(
    hub_model,
    *,
    device_map: Mapping[str, str] | str,
) -> Tuple[torch.nn.Module, Optional[str]]:
    """多卡模式：分片模型"""
    from accelerate import dispatch_model

    if isinstance(device_map, str):
        target_device = device_map
        if hasattr(hub_model, "model") and isinstance(hub_model.model, torch.nn.Module):
            hub_model.model.to(target_device)
            return hub_model.model, target_device
        return hub_model.to(target_device), target_device  # type: ignore[return-value]

    module = hub_model
    if hasattr(hub_model, "model") and isinstance(hub_model.model, torch.nn.Module):
        module = hub_model.model

    if next(module.parameters(), None) is not None:
        current_device = next(module.parameters()).device
        if current_device.type == "cuda":
            module.to("cpu")
            torch.cuda.empty_cache()

    sharded = dispatch_model(module, device_map=device_map)
    if hasattr(hub_model, "model"):
        hub_model.model = sharded
    
    primary = next((device for device in set(device_map.values()) if device != "cpu"), "cpu")
    return sharded, primary


# ---------------------------------------------------------------------------
# Detection & Segmentation
# ---------------------------------------------------------------------------


def coco_class_names() -> List[str]:
    """
    COCO 数据集的 80 个类别名称。
    注意：COCO 类别 ID 是 1-90，但只有 80 个有效类别。
    某些 ID 被跳过了（如 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91）
    """
    # 完整的 COCO 类别映射（索引对应类别ID）
    # ID 0 通常是背景类，但在检测输出中可能不使用
    return [
        "__background__",  # 0 (某些实现会包含背景类)
        "person",          # 1
        "bicycle",         # 2
        "car",             # 3
        "motorcycle",      # 4
        "airplane",        # 5
        "bus",             # 6
        "train",           # 7
        "truck",           # 8
        "boat",            # 9
        "traffic light",   # 10
        "fire hydrant",    # 11
        "street sign",     # 12 (通常跳过)
        "stop sign",       # 13
        "parking meter",   # 14
        "bench",           # 15
        "bird",            # 16
        "cat",             # 17
        "dog",             # 18
        "horse",           # 19
        "sheep",           # 20
        "cow",             # 21
        "elephant",        # 22
        "bear",            # 23
        "zebra",           # 24
        "giraffe",         # 25
        "hat",             # 26 (通常跳过)
        "backpack",        # 27
        "umbrella",        # 28
        "shoe",            # 29 (通常跳过)
        "eye glasses",     # 30 (通常跳过)
        "handbag",         # 31
        "tie",             # 32
        "suitcase",        # 33
        "frisbee",         # 34
        "skis",            # 35
        "snowboard",       # 36
        "sports ball",     # 37
        "kite",            # 38
        "baseball bat",    # 39
        "baseball glove",  # 40
        "skateboard",      # 41
        "surfboard",       # 42
        "tennis racket",   # 43
        "bottle",          # 44
        "plate",           # 45 (通常跳过)
        "wine glass",      # 46
        "cup",             # 47
        "fork",            # 48
        "knife",           # 49
        "spoon",           # 50
        "bowl",            # 51
        "banana",          # 52
        "apple",           # 53
        "sandwich",        # 54
        "orange",          # 55
        "broccoli",        # 56
        "carrot",          # 57
        "hot dog",         # 58
        "pizza",           # 59
        "donut",           # 60
        "cake",            # 61
        "chair",           # 62
        "couch",           # 63
        "potted plant",    # 64
        "bed",             # 65
        "mirror",          # 66 (通常跳过)
        "dining table",    # 67
        "window",          # 68 (通常跳过)
        "desk",            # 69 (通常跳过)
        "toilet",          # 70
        "door",            # 71 (通常跳过)
        "tv",              # 72
        "laptop",          # 73
        "mouse",           # 74
        "remote",          # 75
        "keyboard",        # 76
        "cell phone",      # 77
        "microwave",       # 78
        "oven",            # 79
        "toaster",         # 80
        "sink",            # 81
        "refrigerator",    # 82
        "blender",         # 83 (通常跳过)
        "book",            # 84
        "clock",           # 85
        "vase",            # 86
        "scissors",        # 87
        "teddy bear",      # 88
        "hair drier",      # 89
        "toothbrush",      # 90
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
        text_bbox = draw.textbbox((x1, y1 - 20), caption, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_top = max(y1 - text_h - 4, 0)
        draw.rectangle([(x1, text_top), (x1 + text_w + 8, text_top + text_h + 4)], fill=(255, 0, 0))
        draw.text((x1 + 4, text_top + 2), caption, fill=(255, 255, 255), font=font)

    return annotated


def run_detection_inference(
    predictor,
    images: Iterable[Path],
    output_dir: Path,
    *,
    save_visuals: bool,
    confidence_threshold: float = 0.5,
    max_detections: int = 100,
) -> None:
    import numpy as np

    class_names = coco_class_names()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取模型的 device
    model_device = torch.device("cuda:0")
    try:
        if hasattr(predictor, "model") and hasattr(predictor.model, "parameters"):
            first_param = next(predictor.model.parameters())
            model_device = first_param.device
        elif hasattr(predictor, "parameters"):
            first_param = next(predictor.parameters())
            model_device = first_param.device
        elif hasattr(predictor, "detector") and hasattr(predictor.detector, "parameters"):
            first_param = next(predictor.detector.parameters())
            model_device = first_param.device
    except (StopIteration, AttributeError):
        pass
    
    LOGGER.info("检测推理使用 device: %s, dtype: float32", model_device)
    LOGGER.info("置信度阈值: %.2f, 最大检测数: %d", confidence_threshold, max_detections)

    for image_path in images:
        LOGGER.info("[detection] 处理 %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        
        # 转换为 tensor: [C, H, W]，归一化到 [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # 创建 batch: [1, C, H, W]
        batch = image_tensor.unsqueeze(0)
        
        # 移动到模型设备
        batch = batch.to(device=model_device)
        
        LOGGER.debug("输入 tensor shape: %s, device: %s, dtype: %s", 
                     batch.shape, batch.device, batch.dtype)
        
        # 推理
        with torch.inference_mode():
            outputs = predictor(batch)
        
        # 处理输出 - DINOv3 格式是 [{'scores': ..., 'labels': ..., 'boxes': ...}]
        if isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
            if isinstance(output, dict):
                # DINOv3 检测器的标准输出格式
                if 'boxes' in output and 'scores' in output and 'labels' in output:
                    LOGGER.debug("检测到 DINOv3 标准输出格式")
                    
                    # 提取结果并移到 CPU
                    boxes_tensor = output['boxes']
                    scores_tensor = output['scores']
                    labels_tensor = output['labels']
                    
                    # 转换为 numpy
                    if isinstance(boxes_tensor, torch.Tensor):
                        boxes = boxes_tensor.cpu().numpy()
                        scores = scores_tensor.cpu().numpy()
                        classes = labels_tensor.cpu().numpy()
                    else:
                        boxes = np.array(boxes_tensor)
                        scores = np.array(scores_tensor)
                        classes = np.array(classes_tensor)
                    
                    LOGGER.info("原始检测数: %d", len(boxes))
                    
                    # 应用置信度阈值过滤
                    confidence_mask = scores >= confidence_threshold
                    boxes = boxes[confidence_mask]
                    scores = scores[confidence_mask]
                    classes = classes[confidence_mask]
                    
                    LOGGER.info("置信度过滤后: %d 个目标 (阈值: %.2f)", 
                               len(boxes), confidence_threshold)
                    
                    # 按置信度排序，保留 top-k
                    if len(boxes) > max_detections:
                        top_indices = np.argsort(scores)[::-1][:max_detections]
                        boxes = boxes[top_indices]
                        scores = scores[top_indices]
                        classes = classes[top_indices]
                        LOGGER.info("保留 top-%d 检测结果", max_detections)
                    
                    LOGGER.info("最终检测到 %d 个目标", len(boxes))
                    
                    # 转换为 list 用于 JSON 序列化
                    boxes = boxes.tolist()
                    scores = scores.tolist()
                    classes = classes.tolist()
                    
                    # 构建结果记录
                    records = []
                    for bbox, score, cls in zip(boxes, scores, classes):
                        records.append({
                            "bbox_xyxy": bbox,
                            "score": float(score),
                            "category_id": int(cls),
                            "category_name": class_names[int(cls)] if int(cls) < len(class_names) else str(cls),
                        })
                    
                    # 保存 JSON
                    json_path = output_dir / f"{image_path.stem}_detections.json"
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(records, f, ensure_ascii=False, indent=2)
                    LOGGER.info("保存检测结果: %s", json_path)
                    
                    # 保存可视化
                    if save_visuals and boxes:
                        annotated = annotate_detections(image, boxes, classes, scores, class_names)
                        vis_path = output_dir / f"{image_path.stem}_detection_overlay.png"
                        annotated.save(vis_path)
                        LOGGER.info("保存可视化: %s", vis_path)
                    
                    continue
        
        # 如果上面的处理没有成功，尝试传统的 Detectron2 格式
        instances = None
        
        if isinstance(outputs, dict) and "instances" in outputs:
            instances = outputs["instances"]
            LOGGER.debug("从 dict 提取 instances")
        elif isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
            if isinstance(output, dict) and "instances" in output:
                instances = output["instances"]
                LOGGER.debug("从 list[0] dict 提取 instances")
            elif hasattr(output, "pred_boxes"):
                instances = output
                LOGGER.debug("list[0] 直接是 instances 对象")
        elif hasattr(outputs, "pred_boxes"):
            instances = outputs
            LOGGER.debug("输出直接是 instances 对象")
        
        if instances is None:
            LOGGER.error("=" * 70)
            LOGGER.error("无法从输出中提取检测结果")
            LOGGER.error("输出类型: %s", type(outputs))
            if isinstance(outputs, dict):
                LOGGER.error("输出键: %s", list(outputs.keys()))
            elif isinstance(outputs, list) and len(outputs) > 0:
                LOGGER.error("输出列表长度: %d", len(outputs))
                LOGGER.error("第一个元素类型: %s", type(outputs[0]))
                if isinstance(outputs[0], dict):
                    LOGGER.error("第一个元素键: %s", list(outputs[0].keys()))
            LOGGER.error("=" * 70)
            continue
        
        # Detectron2 格式处理（带过滤）
        instances = instances.to("cpu")
        
        # 应用置信度阈值
        keep = instances.scores >= confidence_threshold
        instances = instances[keep]
        
        # 按置信度排序并保留 top-k
        if len(instances) > max_detections:
            top_indices = instances.scores.argsort(descending=True)[:max_detections]
            instances = instances[top_indices]
        
        boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        classes = instances.pred_classes.numpy().tolist()
        
        LOGGER.info("最终检测到 %d 个目标", len(boxes))
        
        records = []
        for bbox, score, cls in zip(boxes, scores, classes):
            records.append({
                "bbox_xyxy": bbox,
                "score": float(score),
                "category_id": int(cls),
                "category_name": class_names[int(cls)] if int(cls) < len(class_names) else str(cls),
            })
        
        json_path = output_dir / f"{image_path.stem}_detections.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        LOGGER.info("保存检测结果: %s", json_path)
        
        if save_visuals and boxes:
            annotated = annotate_detections(image, boxes, classes, scores, class_names)
            vis_path = output_dir / f"{image_path.stem}_detection_overlay.png"
            annotated.save(vis_path)
            LOGGER.info("保存可视化: %s", vis_path)


def build_segmentation_transform(image_size: int) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def run_segmentation_inference(
    segmentor: torch.nn.Module,
    images: Iterable[Path],
    output_dir: Path,
    *,
    device: str,
    image_size: int,
    save_visuals: bool,
    color_map: str,
) -> None:
    import matplotlib
    from functools import partial
    from dinov3.eval.segmentation.inference import make_inference

    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap(color_map)
    transform = build_segmentation_transform(image_size)

    LOGGER.info("分割推理使用 device: %s, dtype: float32", device)

    for image_path in images:
        LOGGER.info("[segmentation] 处理 %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.inference_mode():
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
        seg_path = output_dir / f"{image_path.stem}_segmentation.png"
        seg_image.save(seg_path)
        LOGGER.info("保存分割结果: %s", seg_path)

        if save_visuals:
            normalized = array.astype("float32") / max(array.max(), 1)
            colored = cmap(normalized)[..., :3]
            overlay = Image.fromarray((colored * 255).astype("uint8"))
            overlay_path = output_dir / f"{image_path.stem}_overlay.png"
            overlay.save(overlay_path)
            LOGGER.info("保存伪彩叠加: %s", overlay_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    # 清理缓存
    if args.clear_cache:
        clear_torch_hub_cache()

    # 验证 checkpoint
    backbone_path = resolve_path(args.backbone)
    if not verify_checkpoint(backbone_path):
        return 1

    # 确定模式
    if args.device is not None:
        use_single_gpu = True
        device = args.device
        LOGGER.info("=" * 70)
        LOGGER.info("单卡模式: %s (float32)", device)
        LOGGER.info("=" * 70)
    elif args.device_map is not None:
        use_single_gpu = False
        apply_visible_devices(args.visible_devices)
        LOGGER.info("=" * 70)
        LOGGER.info("多卡模式 (float32)")
        LOGGER.info("=" * 70)
    else:
        # 自动检测
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory // (1024**3)
            if total_gb >= 40:
                use_single_gpu = True
                device = "cuda:0"
                LOGGER.info("=" * 70)
                LOGGER.info("自动选择单卡模式 (%d GB, float32)", total_gb)
                LOGGER.info("=" * 70)
            else:
                use_single_gpu = False
                args.device_map = "auto"
                LOGGER.info("=" * 70)
                LOGGER.info("自动选择多卡模式 (%d GB, float32)", total_gb)
                LOGGER.info("=" * 70)
        else:
            raise RuntimeError("未检测到 CUDA")

    repo_path = resolve_path(args.repo)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output)

    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"未找到图像: {input_path}")
    LOGGER.info("找到 %d 张图像", len(images))

    need_detection, need_segmentation = resolve_tasks(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if need_detection and not args.detection_adapter:
        raise ValueError("detection 需要 --detection-adapter")
    if need_segmentation and not args.segmentation_adapter:
        raise ValueError("segmentation 需要 --segmentation-adapter")

    # 多卡准备
    available_devices: Sequence[int] = []
    max_memory = None
    if not use_single_gpu:
        available_devices = ensure_minimum_gpus(args.min_gpus)
        max_memory = build_max_memory(args.max_memory, available_devices)

    # 加载检测模型
    detection_model = None
    if need_detection:
        LOGGER.info("-" * 70)
        LOGGER.info("加载检测模型...")
        detection_model = load_torch_hub_model(
            repo_path,
            "dinov3_vit7b16_de",
            weights=args.detection_adapter,
            backbone_weights=args.backbone,
        )

        if use_single_gpu:
            LOGGER.info("移动检测模型到 %s (float32)", device)
            # 直接移动到设备，不指定 dtype (默认 float32)
            detection_model = detection_model.to(device=device)
            detection_model.eval()
            LOGGER.info("✓ 检测模型已加载")
        else:
            device_map_det, _ = prepare_device_map(
                detection_model.model if hasattr(detection_model, "model") else detection_model,
                device_map_option=args.device_map,
                max_memory=max_memory,
                no_split=args.no_split,
            )
            dispatch_model_if_needed(detection_model, device_map=device_map_det)

    # 加载分割模型
    segmentation_model = None
    segmentation_device = None
    if need_segmentation:
        LOGGER.info("-" * 70)
        LOGGER.info("加载分割模型...")
        segmentation_model = load_torch_hub_model(
            repo_path,
            "dinov3_vit7b16_ms",
            weights=args.segmentation_adapter,
            backbone_weights=args.backbone,
        )

        if use_single_gpu:
            LOGGER.info("移动分割模型到 %s (float32)", device)
            segmentation_model = segmentation_model.to(device=device)
            segmentation_model.eval()
            segmentation_device = device
            LOGGER.info("✓ 分割模型已加载")
        else:
            device_map_seg, primary_seg = prepare_device_map(
                segmentation_model,
                device_map_option=args.device_map,
                max_memory=max_memory,
                no_split=args.no_split,
            )
            segmentation_model, _ = dispatch_model_if_needed(
                segmentation_model, device_map=device_map_seg
            )
            segmentation_device = primary_seg or "cuda:0"

    # 推理
    LOGGER.info("=" * 70)
    LOGGER.info("开始推理...")
    LOGGER.info("=" * 70)

    if detection_model is not None:
        run_detection_inference(
            detection_model,
            images,
            output_dir / "detection" if need_segmentation else output_dir,
            save_visuals=args.save_visuals,
            confidence_threshold=args.confidence_threshold,
            max_detections=args.max_detections,
        )

    if segmentation_model is not None and segmentation_device is not None:
        run_segmentation_inference(
            segmentation_model,
            images,
            output_dir / "segmentation" if need_detection else output_dir,
            device=segmentation_device,
            image_size=args.image_size,
            save_visuals=args.save_visuals,
            color_map=args.color_map,
        )

    LOGGER.info("=" * 70)
    LOGGER.info("完成！结果: %s", output_dir)
    LOGGER.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())