"""数据变换

提供图像、点云等数据的预处理和增强功能。
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor:
    """转换为Tensor"""
    
    def __init__(self, keys: Optional[List[str]] = None):
        """
        Args:
            keys: 要转换的键列表，None表示转换所有图像数据
        """
        self.keys = keys or ['image', 'depth', 'mask']
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            if key not in sample:
                continue
            
            value = sample[key]
            
            if isinstance(value, Image.Image):
                # PIL Image -> Tensor
                sample[key] = TF.to_tensor(value)
            
            elif isinstance(value, np.ndarray):
                # NumPy -> Tensor
                if value.ndim == 2:  # (H, W)
                    sample[key] = torch.from_numpy(value).unsqueeze(0)
                elif value.ndim == 3:  # (H, W, C)
                    sample[key] = torch.from_numpy(value).permute(2, 0, 1)
                else:
                    sample[key] = torch.from_numpy(value)
        
        return sample


class Normalize:
    """归一化"""
    
    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        keys: Optional[List[str]] = None,
    ):
        self.mean = mean
        self.std = std
        self.keys = keys or ['image']
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            if key in sample and isinstance(sample[key], torch.Tensor):
                sample[key] = TF.normalize(sample[key], self.mean, self.std)
        
        return sample


class Resize:
    """调整大小"""
    
    def __init__(
        self,
        size: Tuple[int, int],
        interpolation: str = 'bilinear',
        keys: Optional[List[str]] = None,
    ):
        """
        Args:
            size: (H, W)
            interpolation: 插值方法
            keys: 要调整的键
        """
        self.size = size
        self.interpolation = interpolation
        self.keys = keys or ['image', 'depth', 'mask']
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            if key not in sample:
                continue
            
            value = sample[key]
            
            if isinstance(value, Image.Image):
                mode = Image.BILINEAR if key == 'image' else Image.NEAREST
                sample[key] = value.resize(self.size[::-1], mode)
            
            elif isinstance(value, torch.Tensor):
                # (C, H, W) or (H, W)
                if value.ndim == 2:
                    value = value.unsqueeze(0)
                
                mode = self.interpolation if key == 'image' else 'nearest'
                sample[key] = TF.resize(
                    value,
                    self.size,
                    interpolation=T.InterpolationMode.BILINEAR if mode == 'bilinear' else T.InterpolationMode.NEAREST
                )
                
                if key in ['depth', 'mask']:
                    sample[key] = sample[key].squeeze(0)
        
        return sample


class RandomFlip:
    """随机水平翻转"""
    
    def __init__(self, p: float = 0.5, keys: Optional[List[str]] = None):
        self.p = p
        self.keys = keys or ['image', 'depth', 'mask']
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            for key in self.keys:
                if key not in sample:
                    continue
                
                value = sample[key]
                
                if isinstance(value, Image.Image):
                    sample[key] = TF.hflip(value)
                elif isinstance(value, torch.Tensor):
                    sample[key] = TF.hflip(value)
        
        return sample


class ColorJitter:
    """颜色抖动"""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'image' in sample:
            value = sample['image']
            
            if isinstance(value, Image.Image):
                sample['image'] = self.jitter(value)
            elif isinstance(value, torch.Tensor):
                sample['image'] = self.jitter(TF.to_pil_image(value))
                sample['image'] = TF.to_tensor(sample['image'])
        
        return sample


class RandomCrop:
    """随机裁剪"""
    
    def __init__(
        self,
        size: Tuple[int, int],
        keys: Optional[List[str]] = None,
    ):
        self.size = size
        self.keys = keys or ['image', 'depth', 'mask']
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 获取裁剪参数
        if 'image' in sample:
            img = sample['image']
            if isinstance(img, Image.Image):
                w, h = img.size
            else:  # Tensor
                _, h, w = img.shape
            
            th, tw = self.size
            
            if w == tw and h == th:
                return sample
            
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)
            
            # 应用到所有键
            for key in self.keys:
                if key not in sample:
                    continue
                
                value = sample[key]
                
                if isinstance(value, Image.Image):
                    sample[key] = TF.crop(value, top, left, th, tw)
                elif isinstance(value, torch.Tensor):
                    sample[key] = TF.crop(value, top, left, th, tw)
        
        return sample


class DepthNormalize:
    """深度归一化"""
    
    def __init__(self, max_depth: float = 10.0):
        self.max_depth = max_depth
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'depth' in sample:
            depth = sample['depth']
            
            if isinstance(depth, torch.Tensor):
                # 归一化到 [0, 1]
                sample['depth'] = torch.clamp(depth / self.max_depth, 0, 1)
            elif isinstance(depth, np.ndarray):
                sample['depth'] = np.clip(depth / self.max_depth, 0, 1)
        
        return sample


class MaskBinarize:
    """掩码二值化"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'mask' in sample:
            mask = sample['mask']
            
            if isinstance(mask, torch.Tensor):
                sample['mask'] = (mask > self.threshold).float()
            elif isinstance(mask, np.ndarray):
                sample['mask'] = (mask > self.threshold).astype(np.float32)
        
        return sample


def get_default_transforms(cfg, split: str = 'train'):
    """获取默认变换
    
    Args:
        cfg: 配置对象
        split: 数据划分
        
    Returns:
        Compose变换
    """
    transforms_list = []
    
    # 调整大小
    if 'size' in cfg.image:
        transforms_list.append(Resize(cfg.image.size))
    
    # 数据增强（仅训练集）
    if split == 'train' and cfg.augmentation.enabled:
        if cfg.augmentation.random_flip > 0:
            transforms_list.append(RandomFlip(p=cfg.augmentation.random_flip))
        
        if cfg.augmentation.get('color_jitter'):
            jitter = cfg.augmentation.color_jitter
            transforms_list.append(
                ColorJitter(
                    brightness=jitter.brightness,
                    contrast=jitter.contrast,
                    saturation=jitter.saturation,
                    hue=jitter.hue,
                )
            )
        
        if cfg.augmentation.random_crop:
            transforms_list.append(RandomCrop(cfg.image.size))
    
    # 转换为Tensor
    transforms_list.append(ToTensor())
    
    # 深度和掩码处理
    transforms_list.append(DepthNormalize(max_depth=cfg.model.depth.max_depth))
    transforms_list.append(MaskBinarize())
    
    # 归一化
    transforms_list.append(
        Normalize(
            mean=cfg.image.normalize.mean,
            std=cfg.image.normalize.std,
        )
    )
    
    return Compose(transforms_list)