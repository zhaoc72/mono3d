"""DINOv3 detection & segmentation inference - 单卡版本."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 推理脚本（单卡版本）")
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
        "--device",
        default="cuda:0",
        help="GPU 设备，默认 cuda:0",
    )
    
    # 检测参数
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="检测置信度阈值，默认 0.5",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="每张图像最多保留的检测数，默认 100",
    )
    
    # 分割参数
    parser.add_argument(
        "--image-size",
        type=int,
        default=896,
        help="分割输入图像大小，默认 896",
    )
    parser.add_argument(
        "--color-map",
        default="Spectral",
        help="语义分割 colormap，默认 Spectral",
    )
    
    parser.add_argument("--save-visuals", action="store_true", help="保存可视化结果")
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
    """列出所有图像文件"""
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {path}")
    candidates: List[Path] = []
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidates.extend(path.rglob(f"*{suffix}"))
    return sorted(candidates)


def resolve_tasks(task: str) -> Tuple[bool, bool]:
    """解析任务类型"""
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
        LOGGER.error("Checkpoint 文件可能不完整！")
        return False
    
    try:
        LOGGER.info("验证 checkpoint 文件完整性...")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        LOGGER.info("✓ Checkpoint 文件验证通过")
        LOGGER.debug("Checkpoint 包含 %d 个参数", len(state_dict))
        del state_dict
        return True
    except Exception as exc:
        LOGGER.error("Checkpoint 文件损坏: %s", exc)
        return False


def load_detection_model(repo: Path, device: str, backbone_weights: str, adapter_weights: str):
    """加载检测模型，显式加载权重"""
    LOGGER.info("加载检测模型...")
    
    # 方式1: 使用 torch.hub.load（让 hubconf 处理权重加载）
    try:
        model = torch.hub.load(
            str(repo),
            "dinov3_vit7b16_de",
            source="local",
            weights=adapter_weights,
            backbone_weights=backbone_weights,
            pretrained=True,
        )
        LOGGER.debug("✓ 通过 torch.hub.load 加载成功")
    except Exception as e:
        LOGGER.warning("torch.hub.load 失败: %s，尝试手动加载", e)
        
        # 方式2: 手动加载（如果 torch.hub 失败）
        import sys
        sys.path.insert(0, str(repo))
        from dinov3.models.vision_transformer import vit_large
        from dinov3.eval.detection.models import Detector
        
        # 加载 backbone
        backbone = vit_large(patch_size=16, num_register_tokens=4)
        backbone_state = torch.load(backbone_weights, map_location="cpu")
        if 'model' in backbone_state:
            backbone_state = backbone_state['model']
        backbone.load_state_dict(backbone_state, strict=False)
        
        # 加载检测头
        model = Detector(backbone, num_classes=80)
        adapter_state = torch.load(adapter_weights, map_location="cpu")
        if 'model' in adapter_state:
            adapter_state = adapter_state['model']
        model.load_state_dict(adapter_state, strict=False)
        
        LOGGER.debug("✓ 通过手动加载成功")
    
    model = model.to(device)
    model.eval()
    LOGGER.info("✓ 检测模型已加载到 %s", device)
    
    return model


def load_segmentation_model(repo: Path, device: str, backbone_weights: str, adapter_weights: str):
    """加载分割模型，显式加载权重"""
    LOGGER.info("加载分割模型...")
    
    # 方式1: 使用 torch.hub.load
    try:
        model = torch.hub.load(
            str(repo),
            "dinov3_vit7b16_ms",
            source="local",
            weights=adapter_weights,
            backbone_weights=backbone_weights,
            pretrained=True,
        )
        LOGGER.debug("✓ 通过 torch.hub.load 加载成功")
    except Exception as e:
        LOGGER.warning("torch.hub.load 失败: %s，尝试手动加载", e)
        
        # 方式2: 手动加载
        import sys
        sys.path.insert(0, str(repo))
        from dinov3.models.vision_transformer import vit_large
        from dinov3.eval.segmentation.models import Segmentor
        
        # 加载 backbone
        backbone = vit_large(patch_size=16, num_register_tokens=4)
        backbone_state = torch.load(backbone_weights, map_location="cpu")
        if 'model' in backbone_state:
            backbone_state = backbone_state['model']
        backbone.load_state_dict(backbone_state, strict=False)
        
        # 加载分割头
        model = Segmentor(backbone, num_classes=150)
        adapter_state = torch.load(adapter_weights, map_location="cpu")
        if 'model' in adapter_state:
            adapter_state = adapter_state['model']
        model.load_state_dict(adapter_state, strict=False)
        
        LOGGER.debug("✓ 通过手动加载成功")
    
    model = model.to(device)
    model.eval()
    LOGGER.info("✓ 分割模型已加载到 %s", device)
    
    return model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def coco_class_names() -> List[str]:
    """COCO 80 类类别名称（索引 0-79）"""
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ]


def get_coco_class_name(label_id: int) -> str:
    """获取 COCO 类别名称，自动处理 0-based 或 1-based 索引"""
    class_names = coco_class_names()
    
    if 1 <= label_id <= 80:
        return class_names[label_id - 1]
    elif 0 <= label_id < 80:
        return class_names[label_id]
    else:
        return f"class_{label_id}"


def annotate_detections(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    classes: Sequence[int],
    scores: Sequence[float],
) -> Image.Image:
    """在图像上绘制检测框"""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    for bbox, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = bbox
        label = get_coco_class_name(int(cls))
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
    device: str,
    *,
    save_visuals: bool,
    confidence_threshold: float = 0.5,
    max_detections: int = 100,
) -> None:
    """执行检测推理"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("检测推理 | 设备: %s, dtype: float32", device)
    LOGGER.info("检测参数 | 置信度阈值: %.2f, 最大检测数: %d", confidence_threshold, max_detections)

    for image_path in images:
        LOGGER.info("[检测] 处理 %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        batch = image_tensor.to(device)
        
        with torch.inference_mode():
            outputs = predictor(batch)
        
        if not isinstance(outputs, list) or len(outputs) == 0:
            LOGGER.error("无法解析检测输出")
            continue
            
        output = outputs[0]
        if not isinstance(output, dict) or not all(k in output for k in ['boxes', 'scores', 'labels']):
            LOGGER.error("无法解析检测输出")
            continue
        
        boxes_tensor = output['boxes']
        scores_tensor = output['scores']
        labels_tensor = output['labels']
        
        if isinstance(boxes_tensor, torch.Tensor):
            boxes = boxes_tensor.cpu().numpy()
            scores = scores_tensor.cpu().numpy()
            classes = labels_tensor.cpu().numpy()
        else:
            boxes = np.array(boxes_tensor)
            scores = np.array(scores_tensor)
            classes = np.array(labels_tensor)
        
        LOGGER.debug("原始检测数: %d, 标签范围: [%d, %d]", 
                    len(boxes), int(classes.min()), int(classes.max()))
        
        confidence_mask = scores >= confidence_threshold
        boxes = boxes[confidence_mask]
        scores = scores[confidence_mask]
        classes = classes[confidence_mask]
        
        LOGGER.debug("置信度过滤后: %d (阈值: %.2f)", len(boxes), confidence_threshold)
        
        if len(boxes) > max_detections:
            top_indices = np.argsort(scores)[::-1][:max_detections]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            classes = classes[top_indices]
        
        LOGGER.info("✓ 检测到 %d 个目标", len(boxes))
        
        boxes_list = boxes.tolist()
        scores_list = scores.tolist()
        classes_list = classes.tolist()
        
        records = []
        for bbox, score, cls in zip(boxes_list, scores_list, classes_list):
            cls_int = int(cls)
            records.append({
                "bbox_xyxy": bbox,
                "score": float(score),
                "category_id": cls_int,
                "category_name": get_coco_class_name(cls_int),
            })
        
        json_path = output_dir / f"{image_path.stem}_detections.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        LOGGER.debug("保存 JSON: %s", json_path)
        
        if save_visuals and boxes_list:
            annotated = annotate_detections(image, boxes_list, classes_list, scores_list)
            vis_path = output_dir / f"{image_path.stem}_detection_overlay.png"
            annotated.save(vis_path)
            LOGGER.debug("保存可视化: %s", vis_path)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def ade20k_class_names() -> List[str]:
    """ADE20K 150 类类别名称（索引 0-149）"""
    return [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
        'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
        'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
        'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
        'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
        'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
        'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
        'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
        'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
        'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer',
        'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel',
        'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth',
        'television', 'airplane', 'dirt track', 'apparel', 'pole', 'land',
        'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage',
        'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag',
        'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
        'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen',
        'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray',
        'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board',
        'shower', 'radiator', 'glass', 'clock', 'flag'
    ]


def build_segmentation_transform(image_size: int) -> v2.Compose:
    """构建分割预处理流程"""
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
    device: str,
    *,
    image_size: int,
    save_visuals: bool,
    color_map: str,
) -> None:
    """执行语义分割推理（Mask2Former 风格模型）- 优化可视化"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random

    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap(color_map)
    transform = build_segmentation_transform(image_size)
    class_names = ade20k_class_names()

    LOGGER.info("分割推理 | 设备: %s, dtype: float32", device)
    LOGGER.info("分割参数 | 图像大小: %d, 配色: %s", image_size, color_map)

    for image_path in images:
        LOGGER.info("[分割] 处理 %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (W, H)
        
        LOGGER.debug("原始图像大小: %s", original_size)
        
        # 预处理
        tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
        LOGGER.debug("输入 tensor shape: %s", tensor.shape)

        # 推理
        with torch.inference_mode():
            output = segmentor(tensor)
            
            if not isinstance(output, dict) or 'pred_masks' not in output or 'pred_logits' not in output:
                LOGGER.error("输出格式错误")
                continue
            
            pred_masks = output['pred_masks']
            pred_logits = output['pred_logits']
            
            LOGGER.debug("pred_logits shape: %s", pred_logits.shape)
            LOGGER.debug("pred_masks shape: %s", pred_masks.shape)
            
            # 转换类型
            if pred_logits.dtype == torch.bfloat16:
                pred_logits = pred_logits.to(torch.float32)
            if pred_masks.dtype == torch.bfloat16:
                pred_masks = pred_masks.to(torch.float32)
            
            pred_logits = pred_logits[0]
            pred_masks = pred_masks[0]
            
            # 获取类别概率
            pred_probs = pred_logits.softmax(dim=-1)
            fg_probs = pred_probs[:, :-1]
            bg_probs = pred_probs[:, -1]
            fg_max_probs, fg_max_classes = fg_probs.max(dim=-1)
            
            # 过滤条件
            fg_confidence_threshold = 0.5
            valid_mask = (fg_max_probs > bg_probs) & (fg_max_probs > fg_confidence_threshold)
            
            LOGGER.info("有效前景查询: %d / 100", valid_mask.sum().item())
            
            # 上采样掩码
            pred_masks_resized = torch.nn.functional.interpolate(
                pred_masks.unsqueeze(0),
                size=(original_size[1], original_size[0]),
                mode='bilinear',
                align_corners=False
            )[0]
            
            pred_masks_sigmoid = pred_masks_resized.sigmoid()
            
            # 初始化分割图
            final_seg = torch.zeros(
                (original_size[1], original_size[0]), 
                dtype=torch.long,
                device=device
            )
            
            max_mask_scores = torch.zeros(
                (original_size[1], original_size[0]), 
                dtype=torch.float32,
                device=device
            )
            
            mask_threshold = 0.5
            min_area = 50
            
            valid_queries_info = []
            
            # 处理每个有效查询
            for query_idx in range(100):
                if not valid_mask[query_idx]:
                    continue
                
                class_id = fg_max_classes[query_idx].item()
                class_prob = fg_max_probs[query_idx].item()
                bg_prob = bg_probs[query_idx].item()
                mask = pred_masks_sigmoid[query_idx]
                
                binary_mask = mask > mask_threshold
                mask_area = binary_mask.sum().item()
                
                if mask_area < min_area:
                    continue
                
                valid_queries_info.append({
                    'query': query_idx,
                    'class': class_id,
                    'fg_prob': class_prob,
                    'bg_prob': bg_prob,
                    'area': mask_area,
                })
                
                weighted_mask = mask * class_prob
                update_mask = weighted_mask > max_mask_scores
                
                final_seg[update_mask] = class_id
                max_mask_scores[update_mask] = weighted_mask[update_mask]
            
            LOGGER.info("最终有效查询: %d", len(valid_queries_info))
            
            # 显示查询信息
            if valid_queries_info:
                for info in sorted(valid_queries_info, key=lambda x: -x['fg_prob']):
                    class_name = class_names[info['class']] if info['class'] < len(class_names) else f"class_{info['class']}"
                    LOGGER.info(
                        "  查询 %3d: %-20s | 概率: %.3f | 面积: %6d px",
                        info['query'], class_name, info['fg_prob'], info['area']
                    )
            
            seg_array = final_seg.cpu().numpy().astype(np.uint8)
        
        # 统计
        unique_labels = np.unique(seg_array)
        label_counts = Counter(seg_array.flatten())
        total_pixels = seg_array.size
        
        fg_pixels = total_pixels - label_counts.get(0, 0)
        fg_coverage = fg_pixels / total_pixels * 100
        
        LOGGER.info("检测到类别: %d | 前景覆盖: %.1f%%", 
                   len(unique_labels) - (1 if 0 in unique_labels else 0), fg_coverage)
        
        # 保存原始分割图
        seg_path = output_dir / f"{image_path.stem}_segmentation.png"
        seg_image = Image.fromarray(seg_array, mode='L')
        seg_image.save(seg_path)
        LOGGER.debug("保存分割图: %s", seg_path)

        # ============================================================
        # 可视化部分 - 多种输出
        # ============================================================
        if save_visuals:
            # 1. 生成固定的类别颜色映射（使用 HSV 色彩空间保证区分度）
            random.seed(42)  # 固定随机种子，保证颜色一致
            
            def generate_distinct_colors(n):
                """生成 n 个区分度高的颜色"""
                colors = []
                for i in range(n):
                    hue = i / n
                    saturation = 0.7 + (random.random() * 0.3)  # 0.7-1.0
                    value = 0.7 + (random.random() * 0.3)       # 0.7-1.0
                    rgb = plt.cm.hsv(hue)[:3]
                    # 调整饱和度和亮度
                    rgb = np.array(rgb)
                    rgb = rgb * value
                    colors.append(rgb)
                return colors
            
            # 为所有前景类别生成颜色
            fg_labels = [l for l in unique_labels if l != 0]
            if len(fg_labels) > 0:
                class_colors = generate_distinct_colors(150)  # 为所有可能的类生成颜色
                
                # --------------------------------------------------------
                # 可视化 1: 彩色分割掩码（无背景）
                # --------------------------------------------------------
                seg_colored = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
                
                for label in fg_labels:
                    mask = seg_array == label
                    color = (np.array(class_colors[label]) * 255).astype(np.uint8)
                    seg_colored[mask] = color
                
                # 保存纯分割图
                mask_only_path = output_dir / f"{image_path.stem}_mask_only.png"
                Image.fromarray(seg_colored).save(mask_only_path)
                LOGGER.debug("保存纯分割掩码: %s", mask_only_path)
                
                # --------------------------------------------------------
                # 可视化 2: 半透明叠加
                # --------------------------------------------------------
                overlay_img = Image.fromarray(seg_colored)
                alpha = 0.5
                blended = Image.blend(image.convert('RGB'), overlay_img, alpha=alpha)
                
                overlay_path = output_dir / f"{image_path.stem}_overlay.png"
                blended.save(overlay_path)
                LOGGER.debug("保存叠加图: %s", overlay_path)
                
                # --------------------------------------------------------
                # 可视化 3: 带图例的完整可视化（使用 matplotlib）
                # --------------------------------------------------------
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Segmentation Results: {image_path.name}', fontsize=16, fontweight='bold')
                
                # 子图 1: 原图
                axes[0, 0].imshow(image)
                axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                # 子图 2: 纯分割掩码
                axes[0, 1].imshow(seg_colored)
                axes[0, 1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                # 子图 3: 半透明叠加
                axes[1, 0].imshow(blended)
                axes[1, 0].set_title('Overlay (α=0.5)', fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                # 子图 4: 边界叠加
                # 计算边界
                from scipy import ndimage
                boundaries = np.zeros_like(seg_array, dtype=bool)
                for label in fg_labels:
                    mask = seg_array == label
                    eroded = ndimage.binary_erosion(mask)
                    boundary = mask & ~eroded
                    boundaries |= boundary
                
                # 在原图上绘制边界
                img_with_boundaries = np.array(image).copy()
                img_with_boundaries[boundaries] = [255, 255, 0]  # 黄色边界
                
                axes[1, 1].imshow(img_with_boundaries)
                axes[1, 1].set_title('Boundaries', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                # 添加图例（只显示检测到的类别）
                legend_elements = []
                sorted_labels = sorted(
                    [(l, label_counts[l]) for l in fg_labels],
                    key=lambda x: -x[1]
                )[:15]  # 最多显示前 15 个类别
                
                for label, count in sorted_labels:
                    class_name = class_names[label] if label < len(class_names) else f"class_{label}"
                    pct = count / total_pixels * 100
                    color = class_colors[label]
                    legend_elements.append(
                        Patch(facecolor=color, label=f'{class_name} ({pct:.1f}%)')
                    )
                
                # 在图外添加图例
                fig.legend(
                    handles=legend_elements,
                    loc='center',
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=5,
                    fontsize=10,
                    frameon=True,
                    title='Detected Classes (Top 15)',
                    title_fontsize=12
                )
                
                plt.tight_layout()
                
                # 保存完整可视化
                full_vis_path = output_dir / f"{image_path.stem}_full_visualization.png"
                plt.savefig(full_vis_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                LOGGER.debug("保存完整可视化: %s", full_vis_path)
                
                # --------------------------------------------------------
                # 可视化 4: 类别标签叠加（在图像上标注类别名）
                # --------------------------------------------------------
                from PIL import ImageDraw, ImageFont
                
                img_with_labels = image.copy()
                draw = ImageDraw.Draw(img_with_labels)
                
                # 尝试加载字体
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", size=14)
                    font_small = ImageFont.truetype("DejaVuSans.ttf", size=10)
                except:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()
                
                # 为每个类别找到质心并标注
                for label in fg_labels:
                    mask = seg_array == label
                    
                    # 计算质心
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) == 0:
                        continue
                    
                    centroid_x = int(x_coords.mean())
                    centroid_y = int(y_coords.mean())
                    
                    # 获取类别信息
                    class_name = class_names[label] if label < len(class_names) else f"class_{label}"
                    pct = label_counts[label] / total_pixels * 100
                    text = f"{class_name}\n{pct:.1f}%"
                    
                    # 获取颜色
                    color = tuple((np.array(class_colors[label]) * 255).astype(int).tolist())
                    
                    # 绘制半透明背景
                    bbox = draw.textbbox((centroid_x, centroid_y), text, font=font_small)
                    padding = 4
                    draw.rectangle(
                        [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding],
                        fill=(0, 0, 0, 180)
                    )
                    
                    # 绘制文字
                    draw.text(
                        (centroid_x, centroid_y),
                        text,
                        fill=(255, 255, 255),
                        font=font_small,
                        anchor="mm"
                    )
                    
                    # 绘制指示点
                    point_size = 3
                    draw.ellipse(
                        [centroid_x-point_size, centroid_y-point_size,
                         centroid_x+point_size, centroid_y+point_size],
                        fill=color,
                        outline=(255, 255, 255),
                        width=1
                    )
                
                labeled_path = output_dir / f"{image_path.stem}_labeled.png"
                img_with_labels.save(labeled_path)
                LOGGER.debug("保存标注图: %s", labeled_path)
                
                LOGGER.info("✓ 生成了 5 种可视化输出")
        
        LOGGER.info("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.clear_cache:
        clear_torch_hub_cache()

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA")
    
    device = args.device
    LOGGER.info("=" * 80)
    LOGGER.info("单卡模式: %s (float32)", device)
    LOGGER.info("=" * 80)

    backbone_path = resolve_path(args.backbone)
    if not verify_checkpoint(backbone_path):
        return 1

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
        raise ValueError("检测任务需要 --detection-adapter")
    if need_segmentation and not args.segmentation_adapter:
        raise ValueError("分割任务需要 --segmentation-adapter")

    # 加载模型
    detection_model = None
    if need_detection:
        LOGGER.info("-" * 80)
        detection_model = load_detection_model(
            repo_path,
            device,
            str(backbone_path),
            args.detection_adapter,
        )

    segmentation_model = None
    if need_segmentation:
        LOGGER.info("-" * 80)
        segmentation_model = load_segmentation_model(
            repo_path,
            device,
            str(backbone_path),
            args.segmentation_adapter,
        )

    # 推理
    LOGGER.info("=" * 80)
    LOGGER.info("开始推理...")
    LOGGER.info("=" * 80)

    if detection_model is not None:
        det_output = output_dir / "detection" if need_segmentation else output_dir
        run_detection_inference(
            detection_model,
            images,
            det_output,
            device,
            save_visuals=args.save_visuals,
            confidence_threshold=args.confidence_threshold,
            max_detections=args.max_detections,
        )

    if segmentation_model is not None:
        seg_output = output_dir / "segmentation" if need_detection else output_dir
        run_segmentation_inference(
            segmentation_model,
            images,
            seg_output,
            device,
            image_size=args.image_size,
            save_visuals=args.save_visuals,
            color_map=args.color_map,
        )

    LOGGER.info("=" * 80)
    LOGGER.info("✓ 完成！结果保存在: %s", output_dir)
    LOGGER.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())