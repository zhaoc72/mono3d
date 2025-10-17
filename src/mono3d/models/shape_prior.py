"""形状先验模型

实现显式和隐式形状先验，用于约束3D重建。
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from .networks import MLP, PointNetEncoder, SDFNetwork, OccupancyNetwork

log = logging.getLogger(__name__)


class ExplicitShapePrior(nn.Module):
    """显式形状先验
    
    使用预定义的模板形状（网格、点云）作为先验。
    """
    
    def __init__(
        self,
        template_dir: str,
        representation: str = 'mesh',
        alignment_method: str = 'icp',
        alignment_iterations: int = 50,
        **kwargs
    ):
        """初始化显式形状先验
        
        Args:
            template_dir: 模板目录
            representation: 表示类型 ('mesh', 'pointcloud', 'voxel')
            alignment_method: 对齐方法 ('icp', 'pca', 'learned')
            alignment_iterations: 对齐迭代次数
        """
        super().__init__()
        
        self.template_dir = Path(template_dir)
        self.representation = representation
        self.alignment_method = alignment_method
        self.alignment_iterations = alignment_iterations
        
        # 加载模板形状
        self.templates = {}
        self._load_templates()
        
        log.info(
            f"Initialized ExplicitShapePrior with {len(self.templates)} templates"
        )
    
    def _load_templates(self):
        """加载所有类别的模板"""
        if not self.template_dir.exists():
            log.warning(f"Template directory not found: {self.template_dir}")
            return
        
        # 遍历类别目录
        for category_dir in self.template_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            
            # 根据表示类型加载
            if self.representation == 'mesh':
                template_file = category_dir / 'template.obj'
            elif self.representation == 'pointcloud':
                template_file = category_dir / 'template.ply'
            elif self.representation == 'voxel':
                template_file = category_dir / 'template.npy'
            else:
                continue
            
            if template_file.exists():
                try:
                    template = self._load_template_file(template_file)
                    self.templates[category] = template
                    log.debug(f"Loaded template for category: {category}")
                except Exception as e:
                    log.error(f"Failed to load template {template_file}: {e}")
    
    def _load_template_file(self, filepath: Path) -> Dict[str, Any]:
        """加载模板文件"""
        if filepath.suffix == '.obj':
            return self._load_obj(filepath)
        elif filepath.suffix == '.ply':
            return self._load_ply(filepath)
        elif filepath.suffix == '.npy':
            return {'voxels': np.load(filepath)}
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _load_obj(self, filepath: Path) -> Dict[str, np.ndarray]:
        """加载OBJ文件"""
        vertices = []
        faces = []
        
        with open(filepath) as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()
                    face = []
                    for part in parts[1:]:
                        v_idx = int(part.split('/')[0]) - 1
                        face.append(v_idx)
                    if len(face) == 3:
                        faces.append(face)
        
        return {
            'vertices': np.array(vertices, dtype=np.float32),
            'faces': np.array(faces, dtype=np.int32),
        }
    
    def _load_ply(self, filepath: Path) -> Dict[str, np.ndarray]:
        """加载PLY文件（简化版）"""
        import struct
        
        vertices = []
        colors = []
        
        with open(filepath, 'rb') as f:
            # 跳过header
            line = f.readline()
            while not line.startswith(b'end_header'):
                if line.startswith(b'element vertex'):
                    num_vertices = int(line.split()[-1])
                line = f.readline()
            
            # 读取顶点（假设格式：x y z r g b）
            for _ in range(num_vertices):
                data = struct.unpack('ffffff', f.read(24))
                vertices.append(data[:3])
                colors.append(data[3:])
        
        result = {'points': np.array(vertices, dtype=np.float32)}
        if len(colors) > 0:
            result['colors'] = np.array(colors, dtype=np.float32)
        
        return result
    
    def get_template(self, category: str) -> Optional[Dict[str, Any]]:
        """获取类别的模板
        
        Args:
            category: 类别名称
            
        Returns:
            模板字典或None
        """
        return self.templates.get(category, None)
    
    def initialize(
        self,
        category: str,
        pointcloud: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """初始化形状
        
        Args:
            category: 物体类别
            pointcloud: 观测点云 (N, 3)
            depth: 深度图
            mask: 掩码
            
        Returns:
            初始化的形状字典
        """
        # 获取模板
        template = self.get_template(category)
        
        if template is None:
            log.warning(f"No template found for category: {category}")
            # 返回空点云或从深度生成
            if pointcloud is not None:
                return {'points': pointcloud}
            else:
                return {'points': torch.zeros(1000, 3)}
        
        # 转换为tensor
        if self.representation == 'mesh':
            template_points = torch.from_numpy(template['vertices']).float()
        elif self.representation == 'pointcloud':
            template_points = torch.from_numpy(template['points']).float()
        else:
            # Voxel转点云
            template_points = self._voxel_to_pointcloud(template['voxels'])
        
        # 对齐到观测
        if pointcloud is not None:
            aligned_points = self._align_template(template_points, pointcloud)
        else:
            aligned_points = template_points
        
        result = {'points': aligned_points}
        
        # 如果是网格，也返回面
        if self.representation == 'mesh' and 'faces' in template:
            result['faces'] = torch.from_numpy(template['faces']).long()
        
        return result
    
    def _align_template(
        self,
        template_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """对齐模板到目标点云
        
        Args:
            template_points: (M, 3) 模板点
            target_points: (N, 3) 目标点
            
        Returns:
            (M, 3) 对齐后的模板点
        """
        if self.alignment_method == 'icp':
            return self._icp_alignment(template_points, target_points)
        elif self.alignment_method == 'pca':
            return self._pca_alignment(template_points, target_points)
        else:
            # 简单的中心+尺度对齐
            return self._simple_alignment(template_points, target_points)
    
    def _simple_alignment(
        self,
        template_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """简单对齐（中心+尺度）"""
        # 计算中心
        template_center = template_points.mean(dim=0)
        target_center = target_points.mean(dim=0)
        
        # 计算尺度
        template_scale = (template_points - template_center).norm(dim=1).mean()
        target_scale = (target_points - target_center).norm(dim=1).mean()
        
        scale_factor = target_scale / (template_scale + 1e-8)
        
        # 对齐
        aligned = (template_points - template_center) * scale_factor + target_center
        
        return aligned
    
    def _icp_alignment(
        self,
        template_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """ICP对齐（简化实现）"""
        source = template_points.clone()
        
        for iteration in range(self.alignment_iterations):
            # 找最近邻
            distances = torch.cdist(source, target_points)
            nearest_indices = distances.argmin(dim=1)
            nearest_points = target_points[nearest_indices]
            
            # 计算变换
            source_center = source.mean(dim=0)
            target_center = nearest_points.mean(dim=0)
            
            source_centered = source - source_center
            target_centered = nearest_points - target_center
            
            # SVD求旋转
            H = source_centered.T @ target_centered
            U, S, Vt = torch.svd(H)
            R = Vt.T @ U.T
            
            # 应用变换
            source = source_centered @ R.T + target_center
        
        return source
    
    def _pca_alignment(
        self,
        template_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """PCA对齐"""
        # 中心化
        template_center = template_points.mean(dim=0)
        target_center = target_points.mean(dim=0)
        
        template_centered = template_points - template_center
        target_centered = target_points - target_center
        
        # PCA
        _, _, template_V = torch.svd(template_centered)
        _, _, target_V = torch.svd(target_centered)
        
        # 旋转矩阵
        R = target_V @ template_V.T
        
        # 尺度
        template_scale = template_centered.norm(dim=1).mean()
        target_scale = target_centered.norm(dim=1).mean()
        scale = target_scale / (template_scale + 1e-8)
        
        # 应用变换
        aligned = template_centered @ R.T * scale + target_center
        
        return aligned
    
    def _voxel_to_pointcloud(self, voxels: np.ndarray) -> torch.Tensor:
        """体素转点云"""
        occupied = np.argwhere(voxels > 0.5)
        points = torch.from_numpy(occupied).float()
        
        # 归一化到[-1, 1]
        points = (points / voxels.shape[0]) * 2 - 1
        
        return points
    
    def forward(self, features: torch.Tensor, category: str) -> Dict[str, torch.Tensor]:
        """前向传播（用于与其他模块兼容）"""
        return self.initialize(category)


class ImplicitShapePrior(nn.Module):
    """隐式形状先验
    
    使用神经网络学习形状分布的隐式表示。
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        encoder_type: str = 'pointnet',
        encoder_hidden_dims: List[int] = [256, 512, 512],
        decoder_type: str = 'mlp',
        decoder_hidden_dims: List[int] = [512, 512, 256],
        output_type: str = 'pointcloud',
        num_output_points: int = 2048,
        **kwargs
    ):
        """初始化隐式形状先验
        
        Args:
            latent_dim: 潜在向量维度
            encoder_type: 编码器类型 ('pointnet', 'mlp')
            encoder_hidden_dims: 编码器隐藏层维度
            decoder_type: 解码器类型 ('mlp', 'sdf', 'occupancy')
            decoder_hidden_dims: 解码器隐藏层维度
            output_type: 输出类型 ('pointcloud', 'sdf', 'occupancy')
            num_output_points: 输出点数
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_type = output_type
        self.num_output_points = num_output_points
        
        # 编码器
        if encoder_type == 'pointnet':
            self.encoder = PointNetEncoder(
                input_dim=3,
                hidden_dims=encoder_hidden_dims,
                output_dim=latent_dim,
            )
        else:
            self.encoder = MLP(
                input_dim=3 * num_output_points,  # 展平的点云
                hidden_dims=encoder_hidden_dims,
                output_dim=latent_dim,
            )
        
        # 解码器
        if decoder_type == 'sdf':
            self.decoder = SDFNetwork(
                input_dim=3 + latent_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=1,
            )
        elif decoder_type == 'occupancy':
            self.decoder = OccupancyNetwork(
                input_dim=3 + latent_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=1,
            )
        else:  # 'mlp'
            self.decoder = MLP(
                input_dim=latent_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=3 * num_output_points,
            )
        
        log.info(
            f"Initialized ImplicitShapePrior "
            f"(latent_dim={latent_dim}, output_type={output_type})"
        )
    
    def encode(self, shape: torch.Tensor) -> torch.Tensor:
        """编码形状到潜在空间
        
        Args:
            shape: (B, N, 3) 点云
            
        Returns:
            (B, latent_dim) 潜在向量
        """
        return self.encoder(shape)
    
    def decode(
        self,
        latent: torch.Tensor,
        query_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """从潜在向量解码形状
        
        Args:
            latent: (B, latent_dim)
            query_points: (B, M, 3) 查询点（用于SDF/occupancy）
            
        Returns:
            解码的形状
        """
        if self.output_type in ['sdf', 'occupancy']:
            # 需要查询点
            if query_points is None:
                # 生成均匀采样的查询点
                query_points = self._generate_query_points(
                    latent.shape[0],
                    self.num_output_points
                ).to(latent.device)
            
            # 扩展潜在向量
            B, M = query_points.shape[:2]
            latent_expanded = latent.unsqueeze(1).expand(B, M, -1)
            
            # 拼接查询点和潜在向量
            decoder_input = torch.cat([query_points, latent_expanded], dim=-1)
            
            # 解码
            output = self.decoder(decoder_input)
            
            return output
        
        else:  # 'pointcloud'
            # 直接解码为点云
            output = self.decoder(latent)
            output = output.reshape(-1, self.num_output_points, 3)
            return output
    
    def forward(
        self,
        shape: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            shape: (B, N, 3) 输入形状（编码）
            latent: (B, latent_dim) 潜在向量（解码）
            
        Returns:
            包含编码和解码结果的字典
        """
        result = {}
        
        if shape is not None:
            # 编码
            latent_encoded = self.encode(shape)
            result['latent'] = latent_encoded
            
            # 重建
            reconstructed = self.decode(latent_encoded)
            result['reconstructed'] = reconstructed
        
        if latent is not None:
            # 从给定潜在向量解码
            decoded = self.decode(latent)
            result['decoded'] = decoded
        
        return result
    
    def _generate_query_points(
        self,
        batch_size: int,
        num_points: int
    ) -> torch.Tensor:
        """生成均匀采样的查询点
        
        Args:
            batch_size: 批大小
            num_points: 点数
            
        Returns:
            (batch_size, num_points, 3) 查询点
        """
        # 在[-1, 1]^3中均匀采样
        points = torch.rand(batch_size, num_points, 3) * 2 - 1
        return points
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """从先验分布采样
        
        Args:
            num_samples: 采样数量
            
        Returns:
            (num_samples, latent_dim) 潜在向量
        """
        # 从标准正态分布采样
        latent = torch.randn(num_samples, self.latent_dim)
        return latent


class ShapeVAE(nn.Module):
    """形状变分自编码器
    
    学习形状的概率分布。
    """
    
    def __init__(
        self,
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
        latent_dim: int = 128,
        **kwargs
    ):
        """初始化ShapeVAE
        
        Args:
            encoder: 编码器配置
            decoder: 解码器配置
            latent_dim: 潜在维度
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器
        encoder_type = encoder.get('backbone', 'pointnet')
        hidden_dims = encoder.get('hidden_dims', [256, 512, 512])
        
        if encoder_type == 'pointnet':
            self.encoder_backbone = PointNetEncoder(
                input_dim=3,
                hidden_dims=hidden_dims,
                output_dim=hidden_dims[-1],
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # 均值和方差层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_type = decoder.get('type', 'mlp')
        decoder_hidden_dims = decoder.get('hidden_dims', [512, 512, 256])
        output_dim = decoder.get('output_dim', 3)
        
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=output_dim * 2048,  # 假设输出2048个点
            activation='relu',
        )
        
        self.num_output_points = 2048
        
        log.info(f"Initialized ShapeVAE (latent_dim={latent_dim})")
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码到潜在空间
        
        Args:
            x: (B, N, 3) 点云
            
        Returns:
            (mu, logvar): 均值和对数方差
        """
        features = self.encoder_backbone(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧
        
        Args:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            
        Returns:
            (B, latent_dim) 采样的潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码
        
        Args:
            z: (B, latent_dim)
            
        Returns:
            (B, N, 3) 重建的点云
        """
        output = self.decoder(z)
        output = output.reshape(-1, self.num_output_points, 3)
        return output
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: (B, N, 3) 输入点云
            
        Returns:
            包含重建、均值、方差的字典
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }
    
    def loss(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_weight: float = 0.001,
    ) -> torch.Tensor:
        """计算VAE损失
        
        Args:
            reconstructed: 重建结果
            target: 目标
            mu: 均值
            logvar: 对数方差
            kld_weight: KL散度权重
            
        Returns:
            总损失
        """
        # 重建损失（Chamfer距离）
        recon_loss = self._chamfer_distance(reconstructed, target)
        
        # KL散度
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / mu.shape[0]  # 平均
        
        # 总损失
        total_loss = recon_loss + kld_weight * kld_loss
        
        return total_loss
    
    def _chamfer_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算Chamfer距离
        
        Args:
            pred: (B, M, 3)
            target: (B, N, 3)
            
        Returns:
            标量损失
        """
        # pred -> target
        dist_matrix = torch.cdist(pred, target)  # (B, M, N)
        min_dist_pred_to_target = dist_matrix.min(dim=2)[0]  # (B, M)
        
        # target -> pred
        min_dist_target_to_pred = dist_matrix.min(dim=1)[0]  # (B, N)
        
        # Chamfer距离
        chamfer = (
            min_dist_pred_to_target.mean(dim=1) +
            min_dist_target_to_pred.mean(dim=1)
        )
        
        return chamfer.mean()
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """从先验采样
        
        Args:
            num_samples: 采样数量
            device: 设备
            
        Returns:
            (num_samples, N, 3) 采样的形状
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        shapes = self.decode(z)
        return shapes