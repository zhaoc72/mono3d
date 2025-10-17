"""损失函数

实现各种用于3D重建的损失函数。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


class ColorLoss(nn.Module):
    """颜色重建损失"""
    
    def __init__(self, loss_type: str = 'l1', use_perceptual: bool = False):
        """初始化颜色损失
        
        Args:
            loss_type: 损失类型 ('l1', 'l2', 'smooth_l1')
            use_perceptual: 是否使用感知损失
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.use_perceptual = use_perceptual
        
        if use_perceptual:
            # VGG感知损失
            try:
                import torchvision.models as models
                vgg = models.vgg16(pretrained=True).features[:16]
                vgg.eval()
                for param in vgg.parameters():
                    param.requires_grad = False
                self.vgg = vgg
            except:
                log.warning("Failed to load VGG for perceptual loss")
                self.use_perceptual = False
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            pred: (B, 3, H, W) 预测颜色
            target: (B, 3, H, W) 目标颜色
            mask: (B, 1, H, W) 可选掩码
            
        Returns:
            标量损失
        """
        # 应用掩码
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        # 像素损失
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred, target)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 感知损失
        if self.use_perceptual:
            pred_features = self.vgg(pred)
            target_features = self.vgg(target)
            perceptual_loss = F.mse_loss(pred_features, target_features)
            loss = loss + 0.1 * perceptual_loss
        
        return loss


class DepthLoss(nn.Module):
    """深度损失"""
    
    def __init__(
        self,
        loss_type: str = 'l1',
        use_gradient: bool = True,
        gradient_weight: float = 0.5
    ):
        """初始化深度损失
        
        Args:
            loss_type: 损失类型
            use_gradient: 是否使用梯度损失
            gradient_weight: 梯度损失权重
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.use_gradient = use_gradient
        self.gradient_weight = gradient_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            pred: (B, 1, H, W) 预测深度
            target: (B, 1, H, W) 目标深度
            mask: (B, 1, H, W) 可选掩码
            
        Returns:
            标量损失
        """
        # 应用掩码
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        # 深度值损失
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred, target)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, target)
        elif self.loss_type == 'scale_invariant':
            loss = self._scale_invariant_loss(pred, target, mask)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 梯度损失
        if self.use_gradient:
            grad_loss = self._gradient_loss(pred, target, mask)
            loss = loss + self.gradient_weight * grad_loss
        
        return loss
    
    def _scale_invariant_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """尺度不变损失"""
        # Log space
        log_pred = torch.log(pred + 1e-8)
        log_target = torch.log(target + 1e-8)
        
        diff = log_pred - log_target
        
        if mask is not None:
            diff = diff * mask
            n = mask.sum() + 1e-8
        else:
            n = diff.numel()
        
        # Scale-invariant loss
        loss = (diff ** 2).sum() / n - (diff.sum() ** 2) / (n ** 2)
        
        return loss
    
    def _gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """梯度损失（平滑性）"""
        # 计算梯度
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 损失
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        
        return loss_dx + loss_dy


class MaskLoss(nn.Module):
    """掩码损失"""
    
    def __init__(self, use_dice: bool = True):
        """初始化掩码损失
        
        Args:
            use_dice: 是否使用Dice损失
        """
        super().__init__()
        self.use_dice = use_dice
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            pred: (B, 1, H, W) 预测掩码（sigmoid激活后）
            target: (B, 1, H, W) 目标掩码
            
        Returns:
            标量损失
        """
        # BCE损失
        bce_loss = F.binary_cross_entropy(pred, target)
        
        if not self.use_dice:
            return bce_loss
        
        # Dice损失
        dice_loss = self._dice_loss(pred, target)
        
        return bce_loss + dice_loss
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice损失"""
        smooth = 1.0
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        dice = (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        
        return 1.0 - dice


class ShapePriorLoss(nn.Module):
    """形状先验损失"""
    
    def __init__(
        self,
        prior_type: str = 'vae',  # 'vae', 'gan', 'chamfer'
        weight: float = 0.01
    ):
        """初始化形状先验损失
        
        Args:
            prior_type: 先验类型
            weight: 损失权重
        """
        super().__init__()
        
        self.prior_type = prior_type
        self.weight = weight
    
    def forward(
        self,
        shape: torch.Tensor,
        prior_model: Optional[nn.Module] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            shape: (B, N, 3) 重建形状
            prior_model: 先验模型
            mu: VAE均值
            logvar: VAE对数方差
            
        Returns:
            标量损失
        """
        if self.prior_type == 'vae':
            # KL散度
            if mu is not None and logvar is not None:
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld = kld / mu.shape[0]
                return self.weight * kld
            else:
                return torch.tensor(0.0, device=shape.device)
        
        elif self.prior_type == 'gan':
            # GAN判别器损失
            if prior_model is not None:
                # 判别器评分
                score = prior_model(shape)
                # 鼓励真实性
                loss = F.binary_cross_entropy_with_logits(
                    score,
                    torch.ones_like(score)
                )
                return self.weight * loss
            else:
                return torch.tensor(0.0, device=shape.device)
        
        elif self.prior_type == 'chamfer':
            # 与模板的Chamfer距离
            if prior_model is not None:
                template = prior_model.get_template()
                if template is not None:
                    loss = chamfer_distance(shape, template)
                    return self.weight * loss
            return torch.tensor(0.0, device=shape.device)
        
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")


class NormalSmoothLoss(nn.Module):
    """法线平滑损失"""
    
    def __init__(self, weight: float = 0.05):
        """初始化法线平滑损失
        
        Args:
            weight: 损失权重
        """
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        k_neighbors: int = 8
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            points: (B, N, 3) 点坐标
            normals: (B, N, 3) 点法线
            k_neighbors: 邻域大小
            
        Returns:
            标量损失
        """
        # 找k近邻
        dist_matrix = torch.cdist(points, points)  # (B, N, N)
        _, indices = torch.topk(dist_matrix, k=k_neighbors + 1, dim=2, largest=False)
        indices = indices[:, :, 1:]  # 排除自己
        
        # 获取邻域法线
        B, N = points.shape[:2]
        neighbor_normals = torch.gather(
            normals.unsqueeze(2).expand(B, N, k_neighbors, 3),
            1,
            indices.unsqueeze(-1).expand(B, N, k_neighbors, 3)
        )
        
        # 计算法线差异
        normal_diff = normals.unsqueeze(2) - neighbor_normals
        loss = (normal_diff ** 2).mean()
        
        return self.weight * loss


class TotalLoss(nn.Module):
    """总损失：组合多个损失
    """
    
    def __init__(
        self,
        color_weight: float = 1.0,
        depth_weight: float = 0.1,
        mask_weight: float = 0.5,
        shape_prior_weight: float = 0.01,
        normal_smooth_weight: float = 0.05,
        use_perceptual: bool = False,
        **kwargs
    ):
        """初始化总损失
        
        Args:
            color_weight: 颜色损失权重
            depth_weight: 深度损失权重
            mask_weight: 掩码损失权重
            shape_prior_weight: 形状先验损失权重
            normal_smooth_weight: 法线平滑损失权重
            use_perceptual: 是否使用感知损失
        """
        super().__init__()
        
        self.color_weight = color_weight
        self.depth_weight = depth_weight
        self.mask_weight = mask_weight
        self.shape_prior_weight = shape_prior_weight
        self.normal_smooth_weight = normal_smooth_weight
        
        # 实例化各个损失
        self.color_loss = ColorLoss(use_perceptual=use_perceptual)
        self.depth_loss = DepthLoss()
        self.mask_loss = MaskLoss()
        self.shape_prior_loss = ShapePriorLoss(weight=shape_prior_weight)
        self.normal_smooth_loss = NormalSmoothLoss(weight=normal_smooth_weight)
    
    def forward(
        self,
        rendered: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        shape: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        prior_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """计算总损失
        
        Args:
            rendered: 渲染结果字典，包含:
                - color: (B, 3, H, W)
                - depth: (B, 1, H, W)
                - alpha: (B, 1, H, W)
            targets: 目标字典，包含:
                - image: (B, 3, H, W)
                - depth: (B, 1, H, W)
                - mask: (B, 1, H, W)
            shape: (B, N, 3) 可选的形状（用于先验损失）
            normals: (B, N, 3) 可选的法线（用于平滑损失）
            prior_params: 形状先验参数（如mu, logvar）
            
        Returns:
            损失字典，包含各个损失项和总损失
        """
        losses = {}
        
        # 颜色损失
        if 'color' in rendered and 'image' in targets:
            mask = targets.get('mask', None)
            losses['color'] = self.color_loss(
                rendered['color'],
                targets['image'],
                mask
            ) * self.color_weight
        
        # 深度损失
        if 'depth' in rendered and 'depth' in targets:
            mask = targets.get('mask', None)
            losses['depth'] = self.depth_loss(
                rendered['depth'],
                targets['depth'],
                mask
            ) * self.depth_weight
        
        # 掩码损失
        if 'alpha' in rendered and 'mask' in targets:
            losses['mask'] = self.mask_loss(
                rendered['alpha'],
                targets['mask']
            ) * self.mask_weight
        
        # 形状先验损失
        if shape is not None and self.shape_prior_weight > 0:
            mu = prior_params.get('mu', None) if prior_params else None
            logvar = prior_params.get('logvar', None) if prior_params else None
            
            losses['shape_prior'] = self.shape_prior_loss(
                shape,
                mu=mu,
                logvar=logvar
            )
        
        # 法线平滑损失
        if shape is not None and normals is not None and self.normal_smooth_weight > 0:
            losses['normal_smooth'] = self.normal_smooth_loss(shape, normals)
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses


class EikonalLoss(nn.Module):
    """Eikonal损失（用于SDF网络）
    
    约束SDF梯度的模长为1。
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, gradients: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            gradients: (B, N, 3) SDF梯度
            
        Returns:
            标量损失
        """
        grad_norm = torch.norm(gradients, dim=-1)
        loss = ((grad_norm - 1.0) ** 2).mean()
        return self.weight * loss


class SilhouetteLoss(nn.Module):
    """轮廓损失
    
    约束渲染的轮廓与目标轮廓一致。
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        rendered_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            rendered_mask: (B, 1, H, W) 渲染掩码
            target_mask: (B, 1, H, W) 目标掩码
            
        Returns:
            标量损失
        """
        # IoU损失
        intersection = (rendered_mask * target_mask).sum(dim=(1, 2, 3))
        union = rendered_mask.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3)) - intersection
        
        iou = (intersection + 1e-8) / (union + 1e-8)
        loss = (1.0 - iou).mean()
        
        return self.weight * loss


class LaplacianSmoothLoss(nn.Module):
    """拉普拉斯平滑损失（用于网格）
    
    约束网格表面平滑。
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            vertices: (B, V, 3) 顶点坐标
            faces: (F, 3) 面索引
            
        Returns:
            标量损失
        """
        # 构建邻接矩阵
        V = vertices.shape[1]
        adjacency = torch.zeros(V, V, device=vertices.device)
        
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adjacency[face[i], face[j]] = 1
        
        # 计算度数
        degree = adjacency.sum(dim=1)
        
        # 拉普拉斯矩阵
        laplacian = torch.diag(degree) - adjacency
        
        # 拉普拉斯坐标
        laplacian_coords = torch.matmul(laplacian, vertices.squeeze(0))
        
        # 损失：最小化拉普拉斯坐标的L2范数
        loss = (laplacian_coords ** 2).sum(dim=1).mean()
        
        return self.weight * loss


class ConsistencyLoss(nn.Module):
    """多视角一致性损失
    
    约束多个视角的重建一致。
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        rendered_views: List[torch.Tensor],
        target_views: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            rendered_views: 渲染视图列表，每个 (3, H, W)
            target_views: 目标视图列表，每个 (3, H, W)
            
        Returns:
            标量损失
        """
        total_loss = 0
        
        for rendered, target in zip(rendered_views, target_views):
            loss = F.mse_loss(rendered, target)
            total_loss += loss
        
        return self.weight * total_loss / len(rendered_views)


class PhotometricLoss(nn.Module):
    """光度一致性损失（用于多视角重建）"""
    
    def __init__(self, weight: float = 1.0, ssim_weight: float = 0.85):
        super().__init__()
        self.weight = weight
        self.ssim_weight = ssim_weight
        self.l1_weight = 1.0 - ssim_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            pred: (B, 3, H, W) 预测图像
            target: (B, 3, H, W) 目标图像
            
        Returns:
            标量损失
        """
        # L1损失
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM损失
        ssim_loss = self._ssim_loss(pred, target)
        
        # 组合
        loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        return self.weight * loss
    
    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算SSIM损失"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, 3, 1, 1)
        mu_target = F.avg_pool2d(target, 3, 1, 1)
        
        sigma_pred = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_target ** 2
        sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target
        
        ssim = (
            (2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)
        ) / (
            (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)
        )
        
        return 1.0 - ssim.mean()


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Chamfer距离（双向最近邻距离）
    
    Args:
        pred: (B, M, 3) 预测点云
        target: (B, N, 3) 目标点云
        reduction: 'mean', 'sum', 'none'
        
    Returns:
        Chamfer距离
    """
    # pred -> target
    dist_matrix = torch.cdist(pred, target)  # (B, M, N)
    min_dist_pred_to_target, _ = dist_matrix.min(dim=2)  # (B, M)
    
    # target -> pred
    min_dist_target_to_pred, _ = dist_matrix.min(dim=1)  # (B, N)
    
    # 双向距离
    chamfer = min_dist_pred_to_target.sum(dim=1) + min_dist_target_to_pred.sum(dim=1)
    
    if reduction == 'mean':
        return chamfer.mean()
    elif reduction == 'sum':
        return chamfer.sum()
    else:
        return chamfer


def earth_mover_distance(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """Earth Mover's Distance (需要额外的求解器)
    
    这里提供一个简化的近似实现。
    
    Args:
        pred: (B, M, 3)
        target: (B, N, 3)
        
    Returns:
        标量距离
    """
    # 简化：使用匈牙利算法的近似
    # 实际应用中应使用专门的EMD求解器
    
    # 这里仅作为占位符
    return chamfer_distance(pred, target)


def compute_loss(
    rendered: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_config: Dict[str, float],
    shape: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """便捷的损失计算函数
    
    Args:
        rendered: 渲染结果
        targets: 目标
        loss_config: 损失权重配置
        shape: 形状（可选）
        normals: 法线（可选）
        
    Returns:
        损失字典
    """
    total_loss_fn = TotalLoss(
        color_weight=loss_config.get('color', 1.0),
        depth_weight=loss_config.get('depth', 0.1),
        mask_weight=loss_config.get('mask', 0.5),
        shape_prior_weight=loss_config.get('shape_prior', 0.01),
        normal_smooth_weight=loss_config.get('normal_smooth', 0.05),
    )
    
    losses = total_loss_fn(rendered, targets, shape, normals)
    
    return losses