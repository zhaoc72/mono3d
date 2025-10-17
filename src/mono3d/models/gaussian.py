"""3D Gaussian Splatting模型

实现3DGS的核心表示、渲染和优化功能。
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

log = logging.getLogger(__name__)


class GaussianModel(nn.Module):
    """3D Gaussian Splatting模型
    
    使用可微分的3D高斯表示场景。
    论文: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
    """
    
    def __init__(
        self,
        num_gaussians: int = 10000,
        sh_degree: int = 3,
        learn_opacity: bool = True,
        learn_scaling: bool = True,
        learn_rotation: bool = True,
        **kwargs
    ):
        """初始化Gaussian模型
        
        Args:
            num_gaussians: 初始高斯数量
            sh_degree: 球谐函数阶数（用于颜色表示）
            learn_opacity: 是否学习不透明度
            learn_scaling: 是否学习尺度
            learn_rotation: 是否学习旋转
        """
        super().__init__()
        
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        self.learn_opacity = learn_opacity
        self.learn_scaling = learn_scaling
        self.learn_rotation = learn_rotation
        
        # 高斯参数（作为可学习参数）
        self._xyz = nn.Parameter(torch.zeros(num_gaussians, 3))
        self._features_dc = nn.Parameter(torch.zeros(num_gaussians, 1, 3))  # SH 0阶
        self._features_rest = nn.Parameter(
            torch.zeros(num_gaussians, (sh_degree + 1) ** 2 - 1, 3)
        )  # SH高阶
        
        self._opacity = nn.Parameter(torch.ones(num_gaussians, 1))
        self._scaling = nn.Parameter(torch.zeros(num_gaussians, 3))
        self._rotation = nn.Parameter(torch.zeros(num_gaussians, 4))
        
        # 初始化旋转为单位四元数
        self._rotation.data[:, 0] = 1.0
        
        # 优化器状态（用于自适应密度控制）
        self.xyz_gradient_accum = torch.zeros(num_gaussians, 1)
        self.denom = torch.zeros(num_gaussians, 1)
        
        # 渲染器
        self.renderer = GaussianRenderer()
        
        log.info(
            f"Initialized GaussianModel with {num_gaussians} gaussians, "
            f"SH degree={sh_degree}"
        )
    
    @property
    def xyz(self) -> torch.Tensor:
        """获取位置"""
        return self._xyz
    
    @property
    def opacity(self) -> torch.Tensor:
        """获取不透明度（sigmoid激活）"""
        return torch.sigmoid(self._opacity)
    
    @property
    def scaling(self) -> torch.Tensor:
        """获取尺度（exp激活保证正值）"""
        return torch.exp(self._scaling)
    
    @property
    def rotation(self) -> torch.Tensor:
        """获取旋转（归一化四元数）"""
        return F.normalize(self._rotation, dim=-1)
    
    @property
    def features(self) -> torch.Tensor:
        """获取完整的球谐特征"""
        return torch.cat([self._features_dc, self._features_rest], dim=1)
    
    def get_covariance(self) -> torch.Tensor:
        """计算协方差矩阵
        
        协方差 = R * S * S^T * R^T
        其中 R 是旋转矩阵，S 是尺度矩阵
        
        Returns:
            (N, 3, 3) 协方差矩阵
        """
        # 四元数转旋转矩阵
        R = self._quat_to_rotation_matrix(self.rotation)
        
        # 尺度矩阵（对角）
        S = torch.diag_embed(self.scaling)
        
        # 协方差
        RS = torch.bmm(R, S)
        covariance = torch.bmm(RS, RS.transpose(1, 2))
        
        return covariance
    
    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """四元数转旋转矩阵
        
        Args:
            quat: (N, 4) 四元数 [w, x, y, z]
            
        Returns:
            (N, 3, 3) 旋转矩阵
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=1),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=1),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=1),
        ], dim=1)
        
        return R
    
    def initialize_from_pointcloud(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
    ):
        """从点云初始化高斯
        
        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 可选的颜色
            normals: (N, 3) 可选的法线
        """
        num_points = points.shape[0]
        
        if num_points > self.num_gaussians:
            # 下采样
            indices = torch.randperm(num_points)[:self.num_gaussians]
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
            if normals is not None:
                normals = normals[indices]
        elif num_points < self.num_gaussians:
            # 上采样（复制+扰动）
            repeat_times = self.num_gaussians // num_points + 1
            points = points.repeat(repeat_times, 1)[:self.num_gaussians]
            
            # 添加小扰动
            noise = torch.randn_like(points) * 0.01
            points = points + noise
            
            if colors is not None:
                colors = colors.repeat(repeat_times, 1)[:self.num_gaussians]
            if normals is not None:
                normals = normals.repeat(repeat_times, 1)[:self.num_gaussians]
        
        # 设置位置
        self._xyz.data = points.to(self._xyz.device)
        
        # 设置颜色
        if colors is not None:
            self._features_dc.data = colors.unsqueeze(1).to(self._features_dc.device)
        
        # 设置初始尺度（基于最近邻距离）
        dist_sq = torch.cdist(points, points) ** 2
        dist_sq[torch.arange(num_points), torch.arange(num_points)] = float('inf')
        nearest_dist = dist_sq.min(dim=1)[0]
        scales = torch.log(torch.sqrt(nearest_dist) + 1e-7)
        self._scaling.data = scales.unsqueeze(1).repeat(1, 3).to(self._scaling.device)
        
        log.info(f"Initialized {self.num_gaussians} gaussians from pointcloud")
    
    def initialize_from_shape(self, shape: Dict[str, torch.Tensor]):
        """从形状字典初始化
        
        Args:
            shape: 形状字典，包含 'points' 等
        """
        points = shape['points']
        colors = shape.get('colors', None)
        normals = shape.get('normals', None)
        
        self.initialize_from_pointcloud(points, colors, normals)
    
    def render(
        self,
        viewpoint_camera: Optional[Dict[str, Any]] = None,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """渲染高斯到图像
        
        Args:
            viewpoint_camera: 相机参数字典
            bg_color: 背景颜色 (3,)
            
        Returns:
            渲染结果字典，包含:
                - color: (3, H, W) 颜色
                - depth: (1, H, W) 深度
                - alpha: (1, H, W) 不透明度
        """
        if viewpoint_camera is None:
            # 使用默认相机
            viewpoint_camera = self._get_default_camera()
        
        if bg_color is None:
            bg_color = torch.zeros(3, device=self._xyz.device)
        
        # 调用渲染器
        rendered = self.renderer.render(
            gaussians={
                'xyz': self.xyz,
                'features': self.features,
                'opacity': self.opacity,
                'scaling': self.scaling,
                'rotation': self.rotation,
            },
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
        )
        
        return rendered
    
    def _get_default_camera(self) -> Dict[str, Any]:
        """获取默认相机（用于可视化）"""
        return {
            'image_width': 512,
            'image_height': 512,
            'FoVx': np.pi / 2,
            'FoVy': np.pi / 2,
            'world_view_transform': torch.eye(4, device=self._xyz.device),
            'projection_matrix': torch.eye(4, device=self._xyz.device),
            'camera_center': torch.zeros(3, device=self._xyz.device),
        }
    
    def densify(self, grad_threshold: float = 0.0002, size_threshold: float = 20.0):
        """自适应密度控制：增加高斯
        
        在梯度大的区域分裂/克隆高斯。
        
        Args:
            grad_threshold: 梯度阈值
            size_threshold: 尺度阈值（用于判断分裂还是克隆）
        """
        # 计算平均梯度
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        # 选择需要增加的高斯
        selected_mask = torch.norm(grads, dim=-1) >= grad_threshold
        
        # 大高斯：分裂
        large_mask = selected_mask & (self.scaling.max(dim=-1)[0] > size_threshold)
        
        # 小高斯：克隆
        small_mask = selected_mask & ~large_mask
        
        # 分裂
        if large_mask.sum() > 0:
            self._split_gaussians(large_mask)
        
        # 克隆
        if small_mask.sum() > 0:
            self._clone_gaussians(small_mask)
    
    def _split_gaussians(self, mask: torch.Tensor):
        """分裂高斯"""
        num_split = mask.sum().item()
        
        if num_split == 0:
            return
        
        # 获取要分裂的高斯
        split_xyz = self._xyz[mask]
        split_scaling = self._scaling[mask]
        split_rotation = self._rotation[mask]
        
        # 沿着最大尺度方向分裂成两个
        stds = torch.exp(split_scaling)
        samples = torch.randn(num_split, 2, 3, device=self._xyz.device)
        samples = samples * stds.unsqueeze(1)
        
        # 旋转采样点
        R = self._quat_to_rotation_matrix(F.normalize(split_rotation, dim=-1))
        samples = torch.bmm(samples, R.transpose(1, 2))
        
        # 新位置
        new_xyz1 = split_xyz + samples[:, 0]
        new_xyz2 = split_xyz + samples[:, 1]
        
        # 缩小尺度
        new_scaling = split_scaling - np.log(1.6)
        
        # 添加新高斯
        self._extend_parameters(
            torch.cat([new_xyz1, new_xyz2], dim=0),
            new_scaling.repeat(2, 1),
            split_rotation.repeat(2, 1),
            self._features_dc[mask].repeat(2, 1, 1),
            self._features_rest[mask].repeat(2, 1, 1),
            self._opacity[mask].repeat(2, 1),
        )
        
        # 移除原高斯
        self._prune_gaussians(mask)
        
        log.debug(f"Split {num_split} gaussians")
    
    def _clone_gaussians(self, mask: torch.Tensor):
        """克隆高斯"""
        num_clone = mask.sum().item()
        
        if num_clone == 0:
            return
        
        # 直接复制
        self._extend_parameters(
            self._xyz[mask],
            self._scaling[mask],
            self._rotation[mask],
            self._features_dc[mask],
            self._features_rest[mask],
            self._opacity[mask],
        )
        
        log.debug(f"Cloned {num_clone} gaussians")
    
    def prune(self, min_opacity: float = 0.005, max_scale: float = 100.0):
        """修剪不必要的高斯
        
        Args:
            min_opacity: 最小不透明度阈值
            max_scale: 最大尺度阈值
        """
        # 选择要保留的高斯
        opacity_mask = self.opacity.squeeze() > min_opacity
        scale_mask = self.scaling.max(dim=-1)[0] < max_scale
        
        keep_mask = opacity_mask & scale_mask
        prune_mask = ~keep_mask
        
        if prune_mask.sum() > 0:
            self._prune_gaussians(prune_mask)
            log.debug(f"Pruned {prune_mask.sum().item()} gaussians")
    
    def _extend_parameters(
        self,
        new_xyz: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacity: torch.Tensor,
    ):
        """扩展参数（添加新高斯）"""
        # 拼接
        self._xyz = nn.Parameter(
            torch.cat([self._xyz.data, new_xyz], dim=0)
        )
        self._scaling = nn.Parameter(
            torch.cat([self._scaling.data, new_scaling], dim=0)
        )
        self._rotation = nn.Parameter(
            torch.cat([self._rotation.data, new_rotation], dim=0)
        )
        self._features_dc = nn.Parameter(
            torch.cat([self._features_dc.data, new_features_dc], dim=0)
        )
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest.data, new_features_rest], dim=0)
        )
        self._opacity = nn.Parameter(
            torch.cat([self._opacity.data, new_opacity], dim=0)
        )
        
        # 扩展累积梯度
        num_new = new_xyz.shape[0]
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros(num_new, 1, device=self._xyz.device)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros(num_new, 1, device=self._xyz.device)
        ], dim=0)
        
        self.num_gaussians = self._xyz.shape[0]
    
    def _prune_gaussians(self, mask: torch.Tensor):
        """修剪高斯（移除mask为True的）"""
        keep_mask = ~mask
        
        self._xyz = nn.Parameter(self._xyz.data[keep_mask])
        self._scaling = nn.Parameter(self._scaling.data[keep_mask])
        self._rotation = nn.Parameter(self._rotation.data[keep_mask])
        self._features_dc = nn.Parameter(self._features_dc.data[keep_mask])
        self._features_rest = nn.Parameter(self._features_rest.data[keep_mask])
        self._opacity = nn.Parameter(self._opacity.data[keep_mask])
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]
        
        self.num_gaussians = self._xyz.shape[0]
    
    def to_pointcloud(self) -> Dict[str, np.ndarray]:
        """转换为点云
        
        Returns:
            点云字典，包含 'points' 和 'colors'
        """
        points = self.xyz.detach().cpu().numpy()
        
        # 使用SH 0阶作为颜色
        colors = self._features_dc.detach().squeeze(1).cpu().numpy()
        colors = (colors + 0.5).clip(0, 1)  # 从[-0.5, 0.5]映射到[0, 1]
        
        return {
            'points': points,
            'colors': colors,
        }
    
    def to_mesh(self) -> Dict[str, np.ndarray]:
        """转换为网格（使用Marching Cubes）
        
        Returns:
            网格字典，包含 'vertices' 和 'faces'
        """
        # 简化实现：创建体素网格并提取等值面
        # 实际应用中可以使用更复杂的表面重建算法
        
        try:
            from skimage import measure
        except ImportError:
            log.error("scikit-image not installed, cannot extract mesh")
            return {'vertices': np.array([]), 'faces': np.array([])}
        
        # 创建体素网格
        resolution = 128
        bounds = self.xyz.detach().cpu().numpy()
        
        x_min, y_min, z_min = bounds.min(axis=0)
        x_max, y_max, z_max = bounds.max(axis=0)
        
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        z = np.linspace(z_min, z_max, resolution)
        
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        
        # 计算密度（简化：距离最近高斯的加权和）
        grid_points_torch = torch.from_numpy(grid_points).float().to(self._xyz.device)
        densities = self._compute_density_at_points(grid_points_torch)
        densities = densities.cpu().numpy().reshape(resolution, resolution, resolution)
        
        # Marching Cubes
        try:
            vertices, faces, _, _ = measure.marching_cubes(densities, level=0.5)
            
            # 映射回世界坐标
            vertices[:, 0] = vertices[:, 0] / resolution * (x_max - x_min) + x_min
            vertices[:, 1] = vertices[:, 1] / resolution * (y_max - y_min) + y_min
            vertices[:, 2] = vertices[:, 2] / resolution * (z_max - z_min) + z_min
            
            return {
                'vertices': vertices,
                'faces': faces,
            }
        except Exception as e:
            log.error(f"Marching cubes failed: {e}")
            return {'vertices': np.array([]), 'faces': np.array([])}
    
    def _compute_density_at_points(self, points: torch.Tensor) -> torch.Tensor:
        """计算点处的密度
        
        Args:
            points: (M, 3)
            
        Returns:
            (M,) 密度值
        """
        # 简化：使用不透明度加权的高斯核
        distances = torch.cdist(points, self.xyz)  # (M, N)
        
        # 高斯权重
        scales = self.scaling.mean(dim=1)  # (N,)
        weights = torch.exp(-distances ** 2 / (2 * scales ** 2))  # (M, N)
        
        # 加权不透明度
        density = (weights * self.opacity.squeeze()).sum(dim=1)
        
        return density


class GaussianRenderer(nn.Module):
    """高斯渲染器
    
    实现可微分的高斯投影和混合。
    """
    
    def __init__(self):
        super().__init__()
    
    def render(
        self,
        gaussians: Dict[str, torch.Tensor],
        viewpoint_camera: Dict[str, Any],
        bg_color: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """渲染高斯
        
        Args:
            gaussians: 高斯参数字典
            viewpoint_camera: 相机参数
            bg_color: 背景颜色
            
        Returns:
            渲染结果
        """
        # 简化实现：使用点投影+alpha混合
        # 实际应用中应使用高效的CUDA实现
        
        xyz = gaussians['xyz']
        features = gaussians['features']
        opacity = gaussians['opacity']
        
        # 获取图像尺寸
        H = viewpoint_camera['image_height']
        W = viewpoint_camera['image_width']
        
        # 投影到图像平面（简化：假设正交投影）
        # 实际应使用相机的投影矩阵
        xy = xyz[:, :2]  # 简化：只取x, y
        
        # 归一化到[0, 1]
        xy_min = xy.min(dim=0)[0]
        xy_max = xy.max(dim=0)[0]
        xy_norm = (xy - xy_min) / (xy_max - xy_min + 1e-8)
        
        # 映射到像素坐标
        pixel_coords = xy_norm * torch.tensor([W, H], device=xy.device)
        
        # 创建图像
        color_img = torch.zeros(3, H, W, device=xyz.device) + bg_color.view(3, 1, 1)
        alpha_img = torch.zeros(1, H, W, device=xyz.device)
        depth_img = torch.zeros(1, H, W, device=xyz.device)
        
        # 简化渲染：splat每个高斯
        # 实际应按深度排序并进行alpha混合
        for i in range(xyz.shape[0]):
            x, y = pixel_coords[i]
            x_int, y_int = int(x), int(y)
            
            if 0 <= x_int < W and 0 <= y_int < H:
                # SH 0阶颜色
                color = features[i, 0, :] + 0.5
                color = color.clamp(0, 1)
                
                alpha = opacity[i, 0]
                
                # Alpha混合
                color_img[:, y_int, x_int] = (
                    alpha * color + (1 - alpha) * color_img[:, y_int, x_int]
                )
                alpha_img[0, y_int, x_int] = (
                    alpha + (1 - alpha) * alpha_img[0, y_int, x_int]
                )
                depth_img[0, y_int, x_int] = xyz[i, 2]
        
        return {
            'color': color_img,
            'depth': depth_img,
            'alpha': alpha_img,
        }


def optimize_gaussians(
    gaussian_model: GaussianModel,
    images: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    cameras: List[Dict[str, Any]],
    iterations: int = 3000,
    position_lr: float = 1.6e-4,
    scale_lr: float = 5e-3,
    rotation_lr: float = 1e-3,
    opacity_lr: float = 5e-2,
    color_lr: float = 2.5e-3,
    densify_grad_threshold: float = 0.0002,
    densify_interval: int = 100,
    prune_interval: int = 100,
) -> GaussianModel:
    """优化高斯模型
    
    Args:
        gaussian_model: 高斯模型
        images: 目标图像 (B, 3, H, W)
        depths: 目标深度 (B, 1, H, W)
        masks: 掩码 (B, 1, H, W)
        cameras: 相机参数列表
        iterations: 迭代次数
        position_lr, scale_lr, ...: 各参数的学习率
        densify_grad_threshold: 密度控制梯度阈值
        densify_interval: 密度控制间隔
        prune_interval: 修剪间隔
        
    Returns:
        优化后的模型
    """
    # 创建优化器
    params = [
        {'params': [gaussian_model._xyz], 'lr': position_lr, 'name': 'xyz'},
        {'params': [gaussian_model._scaling], 'lr': scale_lr, 'name': 'scaling'},
        {'params': [gaussian_model._rotation], 'lr': rotation_lr, 'name': 'rotation'},
        {'params': [gaussian_model._opacity], 'lr': opacity_lr, 'name': 'opacity'},
        {'params': [gaussian_model._features_dc], 'lr': color_lr, 'name': 'f_dc'},
        {'params': [gaussian_model._features_rest], 'lr': color_lr / 20.0, 'name': 'f_rest'},
    ]
    
    optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    
    # 优化循环
    for iteration in range(iterations):
        # 随机选择一个视角
        idx = torch.randint(0, len(cameras), (1,)).item()
        
        # 渲染
        rendered = gaussian_model.render(cameras[idx])
        
        # 计算损失
        loss = F.mse_loss(rendered['color'], images[idx])
        
        if depths is not None:
            depth_loss = F.l1_loss(
                rendered['depth'] * masks[idx],
                depths[idx] * masks[idx]
            )
            loss = loss + 0.1 * depth_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 记录梯度（用于密度控制）
        if gaussian_model._xyz.grad is not None:
            gaussian_model.xyz_gradient_accum += torch.norm(
                gaussian_model._xyz.grad,
                dim=-1,
                keepdim=True
            )
            gaussian_model.denom += 1
        
        optimizer.step()
        
        # 密度控制
        if iteration > 0 and iteration % densify_interval == 0:
            gaussian_model.densify(grad_threshold=densify_grad_threshold)
            gaussian_model.xyz_gradient_accum.zero_()
            gaussian_model.denom.zero_()
        
        # 修剪
        if iteration > 0 and iteration % prune_interval == 0:
            gaussian_model.prune()
        
        if iteration % 100 == 0:
            log.info(f"Iteration {iteration}/{iterations}, Loss: {loss.item():.4f}")
    
    return gaussian_model