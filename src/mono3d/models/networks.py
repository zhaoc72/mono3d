"""通用网络组件

提供MLP、PointNet编码器、注意力模块等通用网络组件。
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

log = logging.getLogger(__name__)


class MLP(nn.Module):
    """多层感知机
    
    可配置的全连接网络。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False,
        last_activation: bool = False,
    ):
        """初始化MLP
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数 ('relu', 'leaky_relu', 'gelu', 'silu')
            dropout: Dropout概率
            batch_norm: 是否使用BatchNorm
            last_activation: 最后一层是否使用激活函数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建层
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # 线性层
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # 中间层添加激活、BN、Dropout
            if i < len(dims) - 2 or last_activation:
                # BatchNorm
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                
                # 激活函数
                layers.append(self._get_activation(activation))
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, input_dim) 或 (B, N, input_dim)
            
        Returns:
            (B, output_dim) 或 (B, N, output_dim)
        """
        return self.layers(x)


class PointNetEncoder(nn.Module):
    """PointNet编码器
    
    用于点云特征提取。
    论文: https://arxiv.org/abs/1612.00593
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 512,
        use_batch_norm: bool = True,
        global_pooling: str = 'max',
    ):
        """初始化PointNet编码器
        
        Args:
            input_dim: 输入特征维度（通常为3: xyz）
            hidden_dims: 隐藏层维度
            output_dim: 全局特征维度
            use_batch_norm: 是否使用BatchNorm
            global_pooling: 全局池化方式 ('max', 'avg', 'both')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.global_pooling = global_pooling
        
        # 点级别的MLP
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i + 1], 1))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        
        self.point_mlp = nn.Sequential(*layers)
        
        # 全局特征MLP
        final_input_dim = hidden_dims[-1]
        if global_pooling == 'both':
            final_input_dim *= 2
        
        self.global_mlp = nn.Sequential(
            nn.Linear(final_input_dim, output_dim),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, N, input_dim) 或 (B, input_dim, N)
            
        Returns:
            (B, output_dim) 全局特征
        """
        # 确保格式为 (B, C, N)
        if x.shape[1] != self.input_dim:
            x = x.transpose(1, 2)
        
        # 点级别特征
        point_features = self.point_mlp(x)  # (B, C, N)
        
        # 全局池化
        if self.global_pooling == 'max':
            global_feature = torch.max(point_features, dim=2)[0]
        elif self.global_pooling == 'avg':
            global_feature = torch.mean(point_features, dim=2)
        elif self.global_pooling == 'both':
            max_feat = torch.max(point_features, dim=2)[0]
            avg_feat = torch.mean(point_features, dim=2)
            global_feature = torch.cat([max_feat, avg_feat], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.global_pooling}")
        
        # 全局MLP
        output = self.global_mlp(global_feature)
        
        return output
    
    def forward_with_point_features(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回全局特征和点特征
        
        Args:
            x: (B, N, input_dim)
            
        Returns:
            (global_features, point_features)
        """
        if x.shape[1] != self.input_dim:
            x = x.transpose(1, 2)
        
        point_features = self.point_mlp(x)
        
        if self.global_pooling == 'max':
            global_feature = torch.max(point_features, dim=2)[0]
        elif self.global_pooling == 'avg':
            global_feature = torch.mean(point_features, dim=2)
        else:
            max_feat = torch.max(point_features, dim=2)[0]
            avg_feat = torch.mean(point_features, dim=2)
            global_feature = torch.cat([max_feat, avg_feat], dim=1)
        
        global_feature = self.global_mlp(global_feature)
        
        return global_feature, point_features


class AttentionBlock(nn.Module):
    """多头自注意力模块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        """初始化注意力模块
        
        Args:
            dim: 特征维度
            num_heads: 注意力头数
            qkv_bias: QKV线性层是否使用偏置
            dropout: Dropout概率
        """
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, N, C)
            
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer块（注意力 + FFN）"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """初始化Transformer块
        
        Args:
            dim: 特征维度
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度扩展比例
            dropout: Dropout概率
        """
        super().__init__()
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 注意力
        self.attn = AttentionBlock(dim, num_heads, dropout=dropout)
        
        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            input_dim=dim,
            hidden_dims=[mlp_hidden_dim],
            output_dim=dim,
            activation='gelu',
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, N, C)
            
        Returns:
            (B, N, C)
        """
        # 注意力 + 残差
        x = x + self.attn(self.norm1(x))
        
        # FFN + 残差
        x = x + self.mlp(self.norm2(x))
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码（正弦）"""
    
    def __init__(
        self,
        dim: int,
        max_len: int = 5000,
        dropout: float = 0.0,
    ):
        """初始化位置编码
        
        Args:
            dim: 特征维度
            max_len: 最大序列长度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        
        # 注册为buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码
        
        Args:
            x: (B, N, C)
            
        Returns:
            (B, N, C)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SDFNetwork(nn.Module):
    """SDF网络（用于隐式表面表示）"""
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [256, 256, 256, 256],
        output_dim: int = 1,
        skip_connections: Optional[List[int]] = None,
        geometric_init: bool = True,
    ):
        """初始化SDF网络
        
        Args:
            input_dim: 输入维度（通常为3: xyz）
            hidden_dims: 隐藏层维度
            output_dim: 输出维度（通常为1: sdf值）
            skip_connections: 跳跃连接的层索引
            geometric_init: 是否使用几何初始化
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.skip_connections = skip_connections or [len(hidden_dims) // 2]
        
        # 构建网络
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # 跳跃连接
            if i in self.skip_connections:
                in_dim = dims[i] + input_dim
            else:
                in_dim = dims[i]
            
            layer = nn.Linear(in_dim, dims[i + 1])
            
            # 几何初始化
            if geometric_init and i == 0:
                # 第一层：球面初始化
                torch.nn.init.normal_(layer.weight, mean=0, std=np.sqrt(2 / in_dim))
                torch.nn.init.constant_(layer.bias, 0)
            
            layers.append(layer)
            
            if i < len(dims) - 2:
                layers.append(nn.Softplus(beta=100))
        
        self.layers = nn.ModuleList(layers)
        
        # 输出层
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
        if geometric_init:
            # 输出层：偏置为-radius
            torch.nn.init.normal_(self.output_layer.weight, mean=0, std=1e-5)
            torch.nn.init.constant_(self.output_layer.bias, -0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, 3) 或 (B, N, 3) 三维坐标
            
        Returns:
            (B, 1) 或 (B, N, 1) SDF值
        """
        input_x = x
        
        # 通过网络
        layer_idx = 0
        for i, layer in enumerate(self.layers):
            # 跳跃连接
            if i // 2 in self.skip_connections and i % 2 == 0:
                x = torch.cat([x, input_x], dim=-1)
            
            x = layer(x)
        
        # 输出
        sdf = self.output_layer(x)
        
        return sdf
    
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """计算SDF梯度（法线）
        
        Args:
            x: (B, 3)
            
        Returns:
            (B, 3) 归一化梯度（法线）
        """
        x.requires_grad_(True)
        
        sdf = self.forward(x)
        
        # 计算梯度
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # 归一化
        grad = F.normalize(grad, dim=-1)
        
        return grad


class OccupancyNetwork(nn.Module):
    """占据网络（用于体素占据预测）"""
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [256, 256, 256],
        output_dim: int = 1,
    ):
        """初始化占据网络
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()
        
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation='relu',
            last_activation=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, 3) 或 (B, N, 3)
            
        Returns:
            (B, 1) 或 (B, N, 1) 占据概率 [0, 1]
        """
        logits = self.mlp(x)
        occupancy = torch.sigmoid(logits)
        return occupancy


def init_weights(module: nn.Module, init_type: str = 'normal', gain: float = 0.02):
    """初始化网络权重
    
    Args:
        module: 网络模块
        init_type: 初始化类型 ('normal', 'xavier', 'kaiming', 'orthogonal')
        gain: 增益系数
    """
    classname = module.__class__.__name__
    
    if hasattr(module, 'weight') and (
        classname.find('Conv') != -1 or classname.find('Linear') != -1
    ):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(module.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight.data, gain=gain)
        else:
            raise NotImplementedError(f'Unknown init type: {init_type}')
        
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)