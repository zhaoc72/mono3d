"""使用官方DINOv3代码的零样本实例分割管道"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from .utils import LOGGER


class OfficialDINOv3Pipeline:
    """使用官方DINOv3代码的完整管道"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        self.config = config
        self.official_config = config.get('dinov3_official') or config.get('model', {}).get('dinov3_official', {})
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
    
    def _make_transform(self, resize_size: int) -> transforms.Compose:
        """创建图像预处理pipeline"""
        resize = transforms.Resize((resize_size, resize_size))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([resize, to_tensor, normalize])
    
    def _load_models(self):
        """加载所有模型"""
        LOGGER.info("=" * 70)
        LOGGER.info("🧠 Loading DINOv3 Official Models (Simplified Loading)")
        LOGGER.info("=" * 70)
        
        backbone_cfg = self.official_config['backbone']
        detection_cfg = self.official_config['detection']
        segmentation_cfg = self.official_config['segmentation']
        
        # ========== 策略：只加载一个模型，分时复用 ==========
        # 由于ViT-7B太大，我们采用：先加载检测器跑推理，然后清空显存，再加载分割器
        
        LOGGER.info("\n📝 Note: Using sequential loading strategy due to model size")
        LOGGER.info("   - Detector and Segmentor will be loaded on-demand")
        LOGGER.info("   - This saves memory but requires loading twice per image")
        
        # 不在初始化时加载模型，而是在推理时按需加载
        self.detector = None
        self.segmentor = None
        
        # 保存配置供后续加载使用
        self.backbone_cfg = backbone_cfg
        self.detection_cfg = detection_cfg
        self.segmentation_cfg = segmentation_cfg
        
        LOGGER.info("=" * 70)
        LOGGER.info("✅ Model loading strategy initialized")
        LOGGER.info("=" * 70)

    def _load_detector_if_needed(self):
        """按需加载检测器"""
        if self.detector is not None:
            return
        
        LOGGER.info("🔄 Loading detector on-demand...")
        
        # 如果分割器在显存中，先清理
        if self.segmentor is not None:
            LOGGER.info("   Clearing segmentor from memory...")
            del self.segmentor
            self.segmentor = None
            torch.cuda.empty_cache()
        
        self.detector = torch.hub.load(
            self.repo_dir,
            self.detection_cfg['model_name'],
            source="local",
            weights=self.detection_cfg['checkpoint_path'],
            backbone_weights=self.backbone_cfg['checkpoint_path'],
            trust_repo=True,
        )
        
        # 移到GPU 0
        self.detector = self.detector.to("cuda:3")
        self.detector.eval()
        
        LOGGER.info("   ✅ Detector loaded on GPU 3")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            LOGGER.info(f"   💾 GPU 0: {allocated:.2f}GB allocated")

    def _load_segmentor_if_needed(self):
        """按需加载分割器"""
        if self.segmentor is not None:
            return
        
        LOGGER.info("🔄 Loading segmentor on-demand...")
        
        # 如果检测器在显存中，先清理
        if self.detector is not None:
            LOGGER.info("   Clearing detector from memory...")
            del self.detector
            self.detector = None
            torch.cuda.empty_cache()
        
        self.segmentor = torch.hub.load(
            self.repo_dir,
            self.segmentation_cfg['model_name'],
            source="local",
            weights=self.segmentation_cfg['checkpoint_path'],
            backbone_weights=self.backbone_cfg['checkpoint_path'],
            trust_repo=True,
        )
        
        # 移到GPU 3
        self.segmentor = self.segmentor.to("cuda:0")
        self.segmentor.eval()
        
        LOGGER.info("   ✅ Segmentor loaded on GPU 0")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            LOGGER.info(f"   💾 GPU 3: {allocated:.2f}GB allocated")
    
    
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
        
        # 按需加载检测器
        self._load_detector_if_needed()
        
        # 预处理并移到GPU 0
        batch_img = self._preprocess_image(image).to("cuda:0")
        
        # 推理
        predictions = self.detector(batch_img)
        
        # 解析结果
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
        
        # 按需加载分割器
        self._load_segmentor_if_needed()
        
        # 导入分割推理工具
        from dinov3.eval.segmentation.inference import make_inference
        from functools import partial
        
        seg_cfg = self.official_config['segmentation']
        inf_cfg = self.inference_config
        
        # 预处理
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        original_size = pil_image.size  # (W, H)
        
        # 移到GPU 0
        batch_img = self._preprocess_image(image).to("cuda:0")
        
        # 滑窗推理
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
        probs = segmentation_map[0].cpu().numpy()
        class_map = segmentation_map.argmax(dim=1, keepdim=False)[0].cpu().numpy()
        
        result = {
            'probs': probs,
            'class_map': class_map,
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