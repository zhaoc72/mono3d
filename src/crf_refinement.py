"""Conditional Random Field (CRF) for mask boundary refinement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .utils import LOGGER


@dataclass
class CRFConfig:
    """Configuration for CRF refinement."""
    
    enable: bool = False  # 是否启用 CRF
    max_iterations: int = 5
    pos_w: float = 3.0  # 位置权重
    pos_xy_std: float = 3.0  # 位置标准差
    bi_w: float = 5.0  # 双边权重
    bi_xy_std: float = 50.0  # 双边位置标准差
    bi_rgb_std: float = 5.0  # 双边颜色标准差


class CRFRefiner:
    """CRF-based mask boundary refinement using dense CRF."""
    
    def __init__(self, config: CRFConfig):
        self.config = config
        
        # 检查依赖
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
            self.dcrf = dcrf
            self.unary_from_softmax = unary_from_softmax
            self.available = True
        except ImportError:
            LOGGER.warning(
                "pydensecrf not available. CRF refinement disabled. "
                "Install with: pip install git+https://github.com/lucasb-eyer/pydensecrf.git"
            )
            self.available = False
    
    def refine_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        使用 DenseCRF 细化单个掩码
        
        Args:
            image: RGB 图像 [H, W, 3], uint8
            mask: 二值掩码 [H, W], bool or uint8
            
        Returns:
            refined_mask: 细化后的掩码 [H, W], bool
        """
        if not self.config.enable or not self.available:
            return mask.astype(bool)
        
        height, width = image.shape[:2]
        
        # 转换掩码为概率
        mask_prob = mask.astype(np.float32)
        prob_fg = mask_prob
        prob_bg = 1 - mask_prob
        
        # 创建 unary potential
        probs = np.stack([prob_bg, prob_fg], axis=0)  # [2, H, W]
        unary = self.unary_from_softmax(probs)
        unary = np.ascontiguousarray(unary)
        
        # 创建 DenseCRF
        d = self.dcrf.DenseCRF2D(width, height, 2)
        d.setUnaryEnergy(unary)
        
        # 添加 pairwise potential
        # 1. 位置 pairwise (平滑约束)
        d.addPairwiseGaussian(
            sxy=(self.config.pos_xy_std, self.config.pos_xy_std),
            compat=self.config.pos_w,
            kernel=self.dcrf.DIAG_KERNEL,
            normalization=self.dcrf.NORMALIZE_SYMMETRIC
        )
        
        # 2. 双边 pairwise (颜色+位置)
        d.addPairwiseBilateral(
            sxy=(self.config.bi_xy_std, self.config.bi_xy_std),
            srgb=(self.config.bi_rgb_std, self.config.bi_rgb_std, self.config.bi_rgb_std),
            rgbim=image.astype(np.uint8),
            compat=self.config.bi_w,
            kernel=self.dcrf.DIAG_KERNEL,
            normalization=self.dcrf.NORMALIZE_SYMMETRIC
        )
        
        # 推理
        Q = d.inference(self.config.max_iterations)
        Q = np.array(Q).reshape((2, height, width))
        
        # 取前景概率最大的作为refined mask
        refined_mask = np.argmax(Q, axis=0).astype(bool)
        
        return refined_mask
    
    def refine_masks_batch(
        self,
        image: np.ndarray,
        masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        批量细化掩码
        
        Args:
            image: RGB 图像
            masks: 掩码列表
            
        Returns:
            细化后的掩码列表
        """
        if not self.config.enable or not self.available:
            return masks
        
        refined = []
        for i, mask in enumerate(masks):
            refined_mask = self.refine_mask(image, mask)
            refined.append(refined_mask)
            
            if (i + 1) % 10 == 0:
                LOGGER.info(f"CRF refined {i + 1}/{len(masks)} masks")
        
        return refined


def bilateral_filter_mask(
    mask: np.ndarray,
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    使用双边滤波细化掩码边界
    
    这是 CRF 的轻量级替代方案
    
    Args:
        mask: 二值掩码
        image: RGB 图像
        d: 滤波器直径
        sigma_color: 颜色空间标准差
        sigma_space: 坐标空间标准差
        
    Returns:
        细化后的掩码
    """
    import cv2
    
    # 转换为浮点
    mask_float = mask.astype(np.float32)
    
    # 应用双边滤波
    filtered = cv2.bilateralFilter(
        mask_float,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    
    # 二值化
    refined_mask = (filtered > 0.5).astype(bool)
    
    return refined_mask