"""数据集基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging

log = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """数据集基类
    
    所有数据集应继承此类并实现必要的抽象方法。
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
    ):
        """初始化数据集
        
        Args:
            root: 数据集根目录
            split: 数据划分 (train/val/test)
            transform: 数据变换
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        # 验证目录
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        
        # 初始化缓存
        self.cache = None
        if self.use_cache and self.cache_dir:
            self._init_cache()
        
        # 加载数据索引
        self.samples = self._load_samples()
        
        log.info(
            f"Initialized {self.__class__.__name__} "
            f"with {len(self.samples)} samples ({split} split)"
        )
    
    def _init_cache(self):
        """初始化缓存（子类可重写）"""
        from .cache import LMDBCache
        
        cache_path = self.cache_dir / "features.lmdb"
        if cache_path.exists():
            try:
                self.cache = LMDBCache(cache_path, split=self.split)
                log.info(f"Loaded cache from {cache_path}")
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")
                self.cache = None
    
    @abstractmethod
    def _load_samples(self) -> list:
        """加载样本索引
        
        Returns:
            样本列表，每个样本是一个字典，包含必要的元数据
        """
        pass
    
    @abstractmethod
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """加载单个样本的原始数据
        
        Args:
            idx: 样本索引
            
        Returns:
            样本字典，至少包含:
                - image: PIL.Image 或 numpy array
                - image_id: 图像ID
                - category: 类别标签
        """
        pass
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            处理后的样本字典
        """
        # 从缓存加载
        if self.cache is not None:
            try:
                cached = self.cache[idx]
                # 缓存中已包含预处理特征
                return cached
            except (KeyError, IndexError):
                log.warning(f"Cache miss for index {idx}, loading from disk")
        
        # 从磁盘加载
        sample = self._load_sample(idx)
        
        # 应用变换
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def get_sample_meta(self, idx: int) -> Dict[str, Any]:
        """获取样本元数据（不加载图像）
        
        Args:
            idx: 样本索引
            
        Returns:
            元数据字典
        """
        return self.samples[idx].copy()
    
    def get_categories(self) -> list:
        """获取所有类别
        
        Returns:
            类别列表
        """
        if hasattr(self, 'categories'):
            return self.categories
        
        # 从样本中提取唯一类别
        categories = set()
        for sample in self.samples:
            if 'category' in sample:
                categories.add(sample['category'])
        
        return sorted(list(categories))
    
    def filter_by_category(self, categories: list):
        """按类别过滤样本
        
        Args:
            categories: 要保留的类别列表
        """
        filtered = []
        for sample in self.samples:
            if sample.get('category') in categories:
                filtered.append(sample)
        
        self.samples = filtered
        log.info(f"Filtered to {len(self.samples)} samples")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root={self.root}, "
            f"split={self.split}, "
            f"num_samples={len(self.samples)})"
        )


class MultiViewDataset(BaseDataset):
    """多视角数据集基类
    
    用于CO3D、ScanNet等包含多视角的数据集。
    """
    
    @abstractmethod
    def get_views(self, idx: int) -> Tuple[list, list]:
        """获取某个物体的所有视角
        
        Args:
            idx: 物体索引
            
        Returns:
            (images, cameras): 图像列表和相机参数列表
        """
        pass
    
    @abstractmethod
    def get_camera_params(self, idx: int, view_idx: int) -> Dict[str, Any]:
        """获取相机参数
        
        Args:
            idx: 样本索引
            view_idx: 视角索引
            
        Returns:
            相机参数字典，包含内参、外参等
        """
        pass


class VideoDataset(BaseDataset):
    """视频数据集基类
    
    用于处理视频序列的数据集。
    """
    
    @abstractmethod
    def get_frames(self, idx: int, num_frames: Optional[int] = None) -> list:
        """获取视频帧
        
        Args:
            idx: 视频索引
            num_frames: 要提取的帧数（None表示全部）
            
        Returns:
            帧列表
        """
        pass
    
    @abstractmethod
    def get_frame_timestamps(self, idx: int) -> list:
        """获取帧时间戳
        
        Args:
            idx: 视频索引
            
        Returns:
            时间戳列表
        """
        pass