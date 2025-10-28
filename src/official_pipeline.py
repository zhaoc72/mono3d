"""使用官方DINOv3代码的零样本实例分割管道"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as v2
from PIL import Image

from ..utils import LOGGER


class OfficialDINOv3Pipeline:
    """使用官方DINOv3代码的完整管道"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        self.config = config
        self.official_config = config['dinov3_official']
        self.repo_dir = self.official_config['repo_dir']
        
        # 添加官方代码路径到Python路径
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        
        # 多GPU配置
        self.multi_gpu_config = self.official_config.get('multi_gpu', {})
        self.use_multi_gpu = self.multi_gpu_config.get('enabled', True)
        self.num_gpus = torch.cuda.device_count()
        
        if self.use_multi_gpu and self.num_gpus >= 2:
            LOGGER.info(f"🚀 Multi-GPU enabled: {self.num_gpus} GPUs detected")
            self.device = torch.device("cuda:0")  # 主设备
        else:
            self.device = torch.device(device or "cuda")
            LOGGER.info(f"📌 Single GPU mode: {self.device}")
        
        # 推理配置
        self.inference_config = self.official_config['inference']
        self.image_size = self.inference_config['image_size']
        self.dtype = self._resolve_dtype(self.inference_config['dtype'])
        
        # 加载模型
        self._load_models()
        
        # 图像预处理
        self.transform = self._make_transform(self.image_size)
    
    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        """解析dtype字符串"""
        lookup = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return lookup.get(dtype_str.lower(), torch.bfloat16)
    
    def _make_transform(self, resize_size: int) -> v2.Compose:
        """创建图像预处理pipeline"""
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])
    
    def _load_with_model_parallel(
        self,
        model: nn.Module,
        model_name: str,
    ) -> nn.Module:
        """使用模型并行加载大模型"""
        try:
            from accelerate import (
                init_empty_weights,
                load_checkpoint_and_dispatch,
                infer_auto_device_map,
            )
        except ImportError:
            raise ImportError("请安装 accelerate: pip install accelerate")
        
        LOGGER.info(f"🔄 Loading {model_name} with model parallelism...")
        
        # 配置显存分配
        max_memory = self.multi_gpu_config.get('max_memory', {
            0: "18GiB",
            1: "18GiB",
            2: "18GiB",
            3: "18GiB",
            "cpu": "100GiB",
        })
        
        # 推断设备映射
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Block"],  # 不切分Transformer块
            dtype=self.dtype,
        )
        
        # 打印设备分配
        device_stats = {}
        for key, device in device_map.items():
            device_stats[device] = device_stats.get(device, 0) + 1
        
        LOGGER.info(f"📊 {model_name} device allocation:")
        for device in sorted(device_stats.keys()):
            count = device_stats[device]
            LOGGER.info(f"   - {device}: {count} modules")
        
        return model
    
    def _load_models(self):
        """加载所有模型"""
        LOGGER.info("=" * 70)
        LOGGER.info("🧠 Loading DINOv3 Official Models")
        LOGGER.info("=" * 70)
        
        backbone_cfg = self.official_config['backbone']
        detection_cfg = self.official_config['detection']
        segmentation_cfg = self.official_config['segmentation']
        
        # 加载检测模型
        LOGGER.info(f"\n🎯 Loading detector: {detection_cfg['model_name']}")
        LOGGER.info(f"   Weights: {detection_cfg['checkpoint_path']}")
        LOGGER.info(f"   Backbone: {backbone_cfg['checkpoint_path']}")
        
        self.detector = torch.hub.load(
            self.repo_dir,
            detection_cfg['model_name'],
            source="local",
            weights=detection_cfg['checkpoint_path'],
            backbone_weights=backbone_cfg['checkpoint_path'],
            trust_repo=True,
        )
        
        if self.use_multi_gpu and self.num_gpus >= 2:
            self.detector = self._load_with_model_parallel(
                self.detector, "Detector"
            )
        else:
            self.detector = self.detector.to(self.device, dtype=self.dtype)
        
        self.detector.eval()
        LOGGER.info("   ✅ Detector loaded")
        
        # 加载分割模型
        LOGGER.info(f"\n🖼️  Loading segmentor: {segmentation_cfg['model_name']}")
        LOGGER.info(f"   Weights: {segmentation_cfg['checkpoint_path']}")
        LOGGER.info(f"   Backbone: {backbone_cfg['checkpoint_path']}")
        
        self.segmentor = torch.hub.load(
            self.repo_dir,
            segmentation_cfg['model_name'],
            source="local",
            weights=segmentation_cfg['checkpoint_path'],
            backbone_weights=backbone_cfg['checkpoint_path'],
            trust_repo=True,
        )
        
        if self.use_multi_gpu and self.num_gpus >= 2:
            self.segmentor = self._load_with_model_parallel(
                self.segmentor, "Segmentor"
            )
        else:
            self.segmentor = self.segmentor.to(self.device, dtype=self.dtype)
        
        self.segmentor.eval()
        LOGGER.info("   ✅ Segmentor loaded")
        
        # 打印显存使用情况
        if torch.cuda.is_available():
            LOGGER.info("\n🎯 GPU Memory Status:")
            for i in range(min(self.num_gpus, 4)):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                LOGGER.info(
                    f"   GPU {i}: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved"
                )
        
        LOGGER.info("=" * 70)
        LOGGER.info("✅ All models loaded successfully")
        LOGGER.info("=" * 70)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # 应用transform
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0)  # 添加batch维度
    
    @torch.inference_mode()
    def run_detection(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """运行检测"""
        LOGGER.info("🎯 Running detection...")
        
        # 预处理
        batch_img = self._preprocess_image(image).to(self.device)
        
        # 推理
        with torch.autocast('cuda', dtype=self.dtype):
            predictions = self.detector(batch_img)
        
        # 解析结果
        # DINOv3 detector返回格式：[{'boxes': ..., 'scores': ..., 'labels': ...}]
        pred = predictions[0] if isinstance(predictions, list) else predictions
        
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # 过滤低分检测
        threshold = self.official_config['detection']['score_threshold']
        valid_mask = scores >= threshold
        
        result = {
            'boxes': boxes[valid_mask],
            'scores': scores[valid_mask],
            'labels': labels[valid_mask],
            'num_detections': int(valid_mask.sum()),
        }
        
        LOGGER.info(f"   ✅ Detected {result['num_detections']} objects")
        return result
    
    @torch.inference_mode()
    def run_segmentation(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """运行分割"""
        LOGGER.info("🖼️  Running segmentation...")
        
        # 导入分割推理工具
        from dinov3.eval.segmentation.inference import make_inference
        from functools import partial
        
        seg_cfg = self.official_config['segmentation']
        inf_cfg = self.inference_config
        
        # 预处理
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        original_size = pil_image.size  # (W, H)
        
        batch_img = self._preprocess_image(image).to(self.device)
        
        # 滑窗推理（处理大分辨率图像）
        with torch.autocast('cuda', dtype=self.dtype):
            segmentation_map = make_inference(
                batch_img,
                self.segmentor,
                inference_mode=seg_cfg.get('inference_mode', 'slide'),
                decoder_head_type=seg_cfg.get('decoder_head_type', 'm2f'),
                rescale_to=original_size,
                n_output_channels=seg_cfg['num_classes'],
                crop_size=tuple(inf_cfg['crop_size']),
                stride=tuple(inf_cfg['stride']),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )
        
        # 转换为numpy
        # segmentation_map: [B, C, H, W]
        probs = segmentation_map[0].cpu().numpy()  # [C, H, W]
        class_map = segmentation_map.argmax(dim=1, keepdim=False)[0].cpu().numpy()  # [H, W]
        
        result = {
            'probs': probs,  # [C, H, W] 概率图
            'class_map': class_map,  # [H, W] 类别索引
            'num_classes': seg_cfg['num_classes'],
        }
        
        LOGGER.info(f"   ✅ Segmentation complete: {result['num_classes']} classes")
        return result
    
    def run(self, image: np.ndarray) -> Dict[str, Any]:
        """完整推理管道"""
        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("🚀 Running Official DINOv3 Pipeline")
        LOGGER.info("=" * 70)
        
        # 运行检测
        detection_result = self.run_detection(image)
        
        # 运行分割
        segmentation_result = self.run_segmentation(image)
        
        # 合并结果
        result = {
            'detection': detection_result,
            'segmentation': segmentation_result,
            'image_shape': image.shape[:2],
            'processed_size': (self.image_size, self.image_size),
        }
        
        LOGGER.info("=" * 70)
        LOGGER.info("✅ Pipeline complete")
        LOGGER.info("=" * 70)
        
        return result