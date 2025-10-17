"""评估指标

提供各种3D重建的评估指标。
"""

from typing import Dict, Any, Optional
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging

log = logging.getLogger(__name__)


def compute_chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    bidirectional: bool = True
) -> float:
    """计算Chamfer距离
    
    Args:
        pred: (N, 3) 预测点云
        target: (M, 3) 目标点云
        bidirectional: 是否双向
        
    Returns:
        Chamfer距离
    """
    # pred -> target
    dist_matrix = torch.cdist(pred, target)  # (N, M)
    min_dist_pred_to_target = dist_matrix.min(dim=1)[0]  # (N,)
    chamfer_pred_to_target = min_dist_pred_to_target.mean().item()
    
    if not bidirectional:
        return chamfer_pred_to_target
    
    # target -> pred
    min_dist_target_to_pred = dist_matrix.min(dim=0)[0]  # (M,)
    chamfer_target_to_pred = min_dist_target_to_pred.mean().item()
    
    # 双向平均
    chamfer = (chamfer_pred_to_target + chamfer_target_to_pred) / 2
    
    return chamfer


def compute_iou(
    pred_voxels: torch.Tensor,
    target_voxels: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """计算3D IoU
    
    Args:
        pred_voxels: (D, H, W) 预测体素
        target_voxels: (D, H, W) 目标体素
        threshold: 二值化阈值
        
    Returns:
        IoU值
    """
    pred_binary = (pred_voxels > threshold).float()
    target_binary = (target_voxels > threshold).float()
    
    intersection = (pred_binary * target_binary).sum().item()
    union = pred_binary.sum().item() + target_binary.sum().item() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    
    return iou


def compute_fscore(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.01
) -> Dict[str, float]:
    """计算F-score（Precision和Recall）
    
    Args:
        pred: (N, 3) 预测点云
        target: (M, 3) 目标点云
        threshold: 距离阈值
        
    Returns:
        包含precision, recall, fscore的字典
    """
    # 计算距离
    dist_matrix = torch.cdist(pred, target)  # (N, M)
    
    # Precision: pred中有多少点在target附近
    min_dist_pred_to_target = dist_matrix.min(dim=1)[0]
    precision = (min_dist_pred_to_target < threshold).float().mean().item()
    
    # Recall: target中有多少点在pred附近
    min_dist_target_to_pred = dist_matrix.min(dim=0)[0]
    recall = (min_dist_target_to_pred < threshold).float().mean().item()
    
    # F-score
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """计算PSNR
    
    Args:
        pred: 预测图像
        target: 目标图像
        data_range: 数据范围
        
    Returns:
        PSNR值
    """
    if data_range is None:
        data_range = target.max() - target.min()
    
    return psnr(target, pred, data_range=data_range)


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: Optional[float] = None,
    multichannel: bool = True
) -> float:
    """计算SSIM
    
    Args:
        pred: 预测图像
        target: 目标图像
        data_range: 数据范围
        multichannel: 是否多通道
        
    Returns:
        SSIM值
    """
    if data_range is None:
        data_range = target.max() - target.min()
    
    return ssim(target, pred, data_range=data_range, multichannel=multichannel, channel_axis=-1 if multichannel else None)


def evaluate_reconstruction(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    pred_images: Optional[np.ndarray] = None,
    target_images: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """综合评估重建质量
    
    Args:
        pred_points: (N, 3) 预测点云
        target_points: (M, 3) 目标点云
        pred_images: 可选的渲染图像
        target_images: 可选的目标图像
        
    Returns:
        评估指标字典
    """
    metrics = {}
    
    # Chamfer距离
    metrics['chamfer'] = compute_chamfer_distance(pred_points, target_points)
    
    # F-score
    fscore_results = compute_fscore(pred_points, target_points)
    metrics.update(fscore_results)
    
    # 图像指标
    if pred_images is not None and target_images is not None:
        metrics['psnr'] = compute_psnr(pred_images, target_images)
        metrics['ssim'] = compute_ssim(pred_images, target_images)
    
    return metrics


def compute_normal_consistency(
    pred_normals: torch.Tensor,
    target_normals: torch.Tensor
) -> float:
    """计算法线一致性
    
    Args:
        pred_normals: (N, 3) 预测法线
        target_normals: (N, 3) 目标法线
        
    Returns:
        平均余弦相似度
    """
    # 归一化
    pred_normals = pred_normals / (torch.norm(pred_normals, dim=1, keepdim=True) + 1e-8)
    target_normals = target_normals / (torch.norm(target_normals, dim=1, keepdim=True) + 1e-8)
    
    # 余弦相似度
    cos_sim = (pred_normals * target_normals).sum(dim=1).abs()
    
    return cos_sim.mean().item()


def compute_depth_error(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """计算深度误差指标
    
    Args:
        pred_depth: (H, W) 预测深度
        target_depth: (H, W) 目标深度
        mask: (H, W) 可选掩码
        
    Returns:
        误差指标字典
    """
    if mask is not None:
        pred_depth = pred_depth[mask]
        target_depth = target_depth[mask]
    
    # 去除无效值
    valid = (target_depth > 0) & (pred_depth > 0)
    pred_depth = pred_depth[valid]
    target_depth = target_depth[valid]
    
    if len(pred_depth) == 0:
        return {
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'rmse': float('inf'),
            'rmse_log': float('inf'),
        }
    
    # 绝对相对误差
    abs_rel = (torch.abs(pred_depth - target_depth) / target_depth).mean().item()
    
    # 平方相对误差
    sq_rel = (((pred_depth - target_depth) ** 2) / target_depth).mean().item()
    
    # RMSE
    rmse = torch.sqrt(((pred_depth - target_depth) ** 2).mean()).item()
    
    # RMSE log
    rmse_log = torch.sqrt(
        ((torch.log(pred_depth) - torch.log(target_depth)) ** 2).mean()
    ).item()
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
    }