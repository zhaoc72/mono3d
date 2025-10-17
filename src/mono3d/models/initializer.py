"""形状初始化网络

从单张图像预测初始形状参数，用于加速重建收敛。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .networks import MLP, PointNetEncoder, TransformerBlock

log = logging.getLogger(__name__)


class ShapeInitNet(nn.Module):
    """形状初始化网络
    
    从图像特征预测初始形状参数（如潜在向量、点云等）。
    """
    
    def __init__(
        self,
        feature_dim: int = 768,  # DINOv3特征维度
        latent_dim: int = 128,
        hidden_dims: List[int] = [512, 256, 256],
        output_type: str = 'latent',  # 'latent', 'pointcloud', 'sdf'
        num_output_points: int = 2048,
        use_transformer: bool = False,
        num_transformer_layers: int = 4,
        **kwargs
    ):
        """初始化ShapeInitNet
        
        Args:
            feature_dim: 输入特征维度
            latent_dim: 潜在向量维度
            hidden_dims: 隐藏层维度
            output_type: 输出类型
            num_output_points: 输出点数（用于pointcloud输出）
            use_transformer: 是否使用Transformer
            num_transformer_layers: Transformer层数
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.output_type = output_type
        self.num_output_points = num_output_points
        
        # 特征编码器
        if use_transformer:
            # Transformer编码器
            self.encoder = nn.ModuleList([
                TransformerBlock(feature_dim)
                for _ in range(num_transformer_layers)
            ])
            self.use_transformer = True
        else:
            # MLP编码器
            self.encoder = MLP(
                input_dim=feature_dim,
                hidden_dims=hidden_dims,
                output_dim=latent_dim,
                activation='relu',
                dropout=0.1,
            )
            self.use_transformer = False
        
        # 输出头
        if output_type == 'latent':
            # 直接输出潜在向量
            if use_transformer:
                self.output_head = nn.Linear(feature_dim, latent_dim)
            else:
                self.output_head = nn.Identity()
        
        elif output_type == 'pointcloud':
            # 输出点云坐标
            output_dim = num_output_points * 3
            if use_transformer:
                self.output_head = MLP(
                    input_dim=feature_dim,
                    hidden_dims=[512, 512],
                    output_dim=output_dim,
                )
            else:
                self.output_head = MLP(
                    input_dim=latent_dim,
                    hidden_dims=[256, 512],
                    output_dim=output_dim,
                )
        
        elif output_type == 'sdf':
            # 输出SDF参数
            if use_transformer:
                self.output_head = nn.Linear(feature_dim, latent_dim)
            else:
                self.output_head = nn.Identity()
        
        else:
            raise ValueError(f"Unknown output type: {output_type}")
        
        log.info(
            f"Initialized ShapeInitNet "
            f"(output_type={output_type}, latent_dim={latent_dim})"
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: (B, C) 全局特征 或 (B, N, C) patch特征
            mask: (B, N) 可选的掩码（用于Transformer）
            
        Returns:
            预测结果字典
        """
        # 编码
        if self.use_transformer:
            # Transformer编码
            x = features
            for layer in self.encoder:
                x = layer(x)
            
            # 全局池化
            if mask is not None:
                # 掩码池化
                mask_expanded = mask.unsqueeze(-1)
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                x = x.mean(dim=1)
        else:
            # MLP编码
            if features.ndim == 3:
                # Flatten patch features
                features = features.mean(dim=1)
            x = self.encoder(features)
        
        # 输出
        output = self.output_head(x)
        
        # 格式化输出
        result = {}
        
        if self.output_type == 'latent':
            result['latent'] = output
        
        elif self.output_type == 'pointcloud':
            # 重塑为点云
            points = output.reshape(-1, self.num_output_points, 3)
            result['points'] = points
        
        elif self.output_type == 'sdf':
            result['sdf_latent'] = output
        
        return result
    
    def predict_shape(
        self,
        features: torch.Tensor,
        shape_prior_decoder: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """预测完整形状
        
        Args:
            features: 图像特征
            shape_prior_decoder: 可选的形状先验解码器
            
        Returns:
            形状字典
        """
        # 预测潜在向量或点云
        output = self.forward(features)
        
        # 如果输出是潜在向量且提供了解码器，则解码
        if self.output_type == 'latent' and shape_prior_decoder is not None:
            latent = output['latent']
            decoded = shape_prior_decoder.decode(latent)
            output['points'] = decoded
        
        return output


class MultiTaskInitNet(nn.Module):
    """多任务初始化网络
    
    同时预测形状、姿态、尺度等多个任务。
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        latent_dim: int = 128,
        predict_pose: bool = True,
        predict_scale: bool = True,
        predict_category: bool = False,
        num_categories: int = 10,
        **kwargs
    ):
        """初始化多任务网络
        
        Args:
            feature_dim: 特征维度
            latent_dim: 形状潜在维度
            predict_pose: 是否预测姿态
            predict_scale: 是否预测尺度
            predict_category: 是否预测类别
            num_categories: 类别数
        """
        super().__init__()
        
        self.predict_pose = predict_pose
        self.predict_scale = predict_scale
        self.predict_category = predict_category
        
        # 共享编码器
        self.shared_encoder = MLP(
            input_dim=feature_dim,
            hidden_dims=[512, 512, 256],
            output_dim=256,
            activation='relu',
            dropout=0.1,
        )
        
        # 形状预测头
        self.shape_head = MLP(
            input_dim=256,
            hidden_dims=[256],
            output_dim=latent_dim,
        )
        
        # 姿态预测头（旋转 + 平移）
        if predict_pose:
            self.pose_head = MLP(
                input_dim=256,
                hidden_dims=[128],
                output_dim=7,  # 4 (quaternion) + 3 (translation)
            )
        
        # 尺度预测头
        if predict_scale:
            self.scale_head = MLP(
                input_dim=256,
                hidden_dims=[128],
                output_dim=3,  # x, y, z scales
            )
        
        # 类别预测头
        if predict_category:
            self.category_head = MLP(
                input_dim=256,
                hidden_dims=[128],
                output_dim=num_categories,
            )
        
        log.info("Initialized MultiTaskInitNet")
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: (B, C) 图像特征
            
        Returns:
            预测结果字典
        """
        # 共享编码
        x = self.shared_encoder(features)
        
        result = {}
        
        # 形状
        result['shape_latent'] = self.shape_head(x)
        
        # 姿态
        if self.predict_pose:
            pose = self.pose_head(x)
            # 分离旋转和平移
            rotation_quat = F.normalize(pose[:, :4], dim=-1)  # 归一化四元数
            translation = pose[:, 4:]
            
            result['rotation'] = rotation_quat
            result['translation'] = translation
        
        # 尺度
        if self.predict_scale:
            scale = self.scale_head(x)
            result['scale'] = torch.exp(scale)  # 保证正值
        
        # 类别
        if self.predict_category:
            logits = self.category_head(x)
            result['category_logits'] = logits
            result['category'] = torch.argmax(logits, dim=-1)
        
        return result


class HierarchicalInitNet(nn.Module):
    """层次化初始化网络
    
    从粗到细预测形状，先预测粗略形状再细化。
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        coarse_points: int = 512,
        fine_points: int = 2048,
        latent_dim: int = 128,
        **kwargs
    ):
        """初始化层次化网络
        
        Args:
            feature_dim: 特征维度
            coarse_points: 粗略点数
            fine_points: 精细点数
            latent_dim: 潜在维度
        """
        super().__init__()
        
        self.coarse_points = coarse_points
        self.fine_points = fine_points
        
        # 粗略形状预测
        self.coarse_predictor = MLP(
            input_dim=feature_dim,
            hidden_dims=[512, 512],
            output_dim=coarse_points * 3,
        )
        
        # 细化网络：结合粗略形状和图像特征
        self.refine_encoder = PointNetEncoder(
            input_dim=3,
            hidden_dims=[64, 128, 256],
            output_dim=256,
        )
        
        self.refine_predictor = MLP(
            input_dim=256 + feature_dim,
            hidden_dims=[512, 512, 512],
            output_dim=fine_points * 3,
        )
        
        log.info(
            f"Initialized HierarchicalInitNet "
            f"(coarse={coarse_points}, fine={fine_points})"
        )
    
    def forward(
        self,
        features: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: (B, C) 图像特征
            return_intermediate: 是否返回中间结果
            
        Returns:
            预测结果字典
        """
        # 粗略预测
        coarse_output = self.coarse_predictor(features)
        coarse_points = coarse_output.reshape(-1, self.coarse_points, 3)
        
        # 编码粗略形状
        coarse_features = self.refine_encoder(coarse_points)
        
        # 结合图像特征
        combined = torch.cat([coarse_features, features], dim=-1)
        
        # 精细预测
        fine_output = self.refine_predictor(combined)
        fine_points = fine_output.reshape(-1, self.fine_points, 3)
        
        result = {'points': fine_points}
        
        if return_intermediate:
            result['coarse_points'] = coarse_points
        
        return result


class CategorySpecificInitNet(nn.Module):
    """类别特定初始化网络
    
    为每个类别训练独立的初始化网络。
    """
    
    def __init__(
        self,
        categories: List[str],
        feature_dim: int = 768,
        latent_dim: int = 128,
        shared_encoder: bool = True,
        **kwargs
    ):
        """初始化类别特定网络
        
        Args:
            categories: 类别列表
            feature_dim: 特征维度
            latent_dim: 潜在维度
            shared_encoder: 是否使用共享编码器
        """
        super().__init__()
        
        self.categories = categories
        self.num_categories = len(categories)
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        # 共享编码器（可选）
        if shared_encoder:
            self.shared_encoder = MLP(
                input_dim=feature_dim,
                hidden_dims=[512, 256],
                output_dim=256,
            )
            decoder_input_dim = 256
        else:
            self.shared_encoder = None
            decoder_input_dim = feature_dim
        
        # 每个类别一个解码器
        self.decoders = nn.ModuleDict({
            cat: MLP(
                input_dim=decoder_input_dim,
                hidden_dims=[256, 256],
                output_dim=latent_dim,
            )
            for cat in categories
        })
        
        log.info(f"Initialized CategorySpecificInitNet for {len(categories)} categories")
    
    def forward(
        self,
        features: torch.Tensor,
        categories: List[str]
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: (B, C) 图像特征
            categories: 长度为B的类别列表
            
        Returns:
            预测结果字典
        """
        batch_size = features.shape[0]
        
        # 共享编码（可选）
        if self.shared_encoder is not None:
            features = self.shared_encoder(features)
        
        # 为每个样本使用对应类别的解码器
        outputs = []
        for i in range(batch_size):
            category = categories[i]
            
            if category not in self.decoders:
                log.warning(f"Unknown category: {category}, using first decoder")
                category = self.categories[0]
            
            decoder = self.decoders[category]
            output = decoder(features[i:i+1])
            outputs.append(output)
        
        # 堆叠
        latents = torch.cat(outputs, dim=0)
        
        return {'latent': latents}


def train_init_net(
    model: nn.Module,
    dataloader,
    shape_prior_decoder: Optional[nn.Module],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
) -> nn.Module:
    """训练初始化网络
    
    Args:
        model: 初始化网络
        dataloader: 数据加载器
        shape_prior_decoder: 形状先验解码器（用于监督）
        device: 设备
        epochs: 训练轮数
        lr: 学习率
        
    Returns:
        训练后的模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            features = batch['features'].to(device)
            gt_shapes = batch['shape'].to(device)  # 真值形状
            
            # 前向传播
            pred = model(features)
            
            # 如果预测的是潜在向量，需要解码
            if 'latent' in pred and shape_prior_decoder is not None:
                pred_shapes = shape_prior_decoder.decode(pred['latent'])
            elif 'points' in pred:
                pred_shapes = pred['points']
            else:
                raise ValueError("Cannot extract predicted shapes")
            
            # 计算损失（Chamfer距离）
            loss = chamfer_distance(pred_shapes, gt_shapes)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        log.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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