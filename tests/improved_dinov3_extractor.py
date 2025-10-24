"""
改进的DINOv3特征提取器 - 集成多种热图生成方法

这个版本添加了物体性评分和组合方法，可以替代当前的attention方法
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, Optional


class ImprovedDINOv3Extractor:
    """
    改进的DINOv3特征提取器
    
    支持多种热图生成方法:
    1. objectness - 物体性评分 (推荐)
    2. combined - 组合方法 (物体性 + 异常度 + 对比度)
    3. attention - 原始attention方法 (baseline)
    """
    
    def __init__(self, model, device, heatmap_method='objectness'):
        """
        Args:
            model: DINOv3模型
            device: 计算设备
            heatmap_method: 热图生成方法 ('objectness', 'combined', 'attention')
        """
        self.model = model
        self.device = device
        self.heatmap_method = heatmap_method
        
        # 用于attention方法
        self.last_attention = None
        
        if heatmap_method == 'attention':
            self._setup_attention_hook()
    
    def _setup_attention_hook(self):
        """设置attention捕获hook"""
        if not hasattr(self.model, 'blocks') or len(self.model.blocks) == 0:
            return
        
        last_block = self.model.blocks[-1]
        if not hasattr(last_block, 'attn'):
            return
        
        attn_module = last_block.attn
        original_forward = attn_module.forward
        
        def wrapped_forward(self, x, rope=None):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            if rope is not None and hasattr(self, 'rope'):
                q = self.rope(q, rope)
                k = self.rope(k, rope)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            
            # 捕获attention
            extractor_instance.last_attention = attn.detach()
            
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        extractor_instance = self
        import types
        attn_module.forward = types.MethodType(wrapped_forward, attn_module)
    
    def extract_patch_features(self, image_tensor):
        """
        提取patch特征
        
        Returns:
            patch_features: [N, D] tensor
            H, W: patch网格大小
        """
        with torch.no_grad():
            features_dict = self.model.forward_features(image_tensor)
            patch_features = features_dict['x_norm_patchtokens'][0]  # [N, D]
        
        H = W = int(np.sqrt(len(patch_features)))
        return patch_features, H, W
    
    def compute_objectness_heatmap(self, patch_features, H, W):
        """
        计算物体性热图
        
        物体性定义: 与邻近patches的相似度低 = 独特 = 可能是物体
        """
        # 归一化特征
        patch_features_norm = F.normalize(patch_features, dim=-1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())
        
        # 对每个patch，计算其与最相似的K个邻居的平均相似度
        K = 20
        topk_sim, _ = torch.topk(similarity_matrix, k=K+1, dim=1, largest=True)
        avg_similarity = topk_sim[:, 1:].mean(dim=1)  # 排除自己
        
        # 物体性 = 1 - 相似度
        objectness = 1 - avg_similarity
        
        # 重塑为2D
        objectness_2d = objectness[:H*W].reshape(H, W).cpu().numpy()
        
        return objectness_2d
    
    def compute_anomaly_heatmap(self, patch_features, H, W):
        """计算异常度热图 (与全局均值的距离)"""
        mean_feature = patch_features.mean(dim=0, keepdim=True)
        distances = torch.norm(patch_features - mean_feature, dim=-1)
        
        anomaly_2d = distances[:H*W].reshape(H, W).cpu().numpy()
        return anomaly_2d
    
    def compute_contrast_heatmap(self, patch_features, H, W):
        """计算局部对比度热图"""
        feature_map = patch_features.reshape(H, W, -1)
        contrast = torch.zeros(H, W, device=self.device)
        
        # 计算与8邻域的差异
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                i_start = max(0, -di)
                i_end = H + min(0, -di)
                j_start = max(0, -dj)
                j_end = W + min(0, -dj)
                
                center = feature_map[i_start:i_end, j_start:j_end, :]
                neighbor = feature_map[i_start+di:i_end+di, j_start+dj:j_end+dj, :]
                
                diff = torch.norm(center - neighbor, dim=-1)
                contrast[i_start:i_end, j_start:j_end] += diff
        
        return contrast.cpu().numpy()
    
    def compute_combined_heatmap(self, patch_features, H, W):
        """
        计算组合热图
        
        组合: 40% 物体性 + 30% 异常度 + 30% 对比度
        """
        # 计算各个分量
        objectness = self.compute_objectness_heatmap(patch_features, H, W)
        anomaly = self.compute_anomaly_heatmap(patch_features, H, W)
        contrast = self.compute_contrast_heatmap(patch_features, H, W)
        
        # 归一化
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        objectness_norm = normalize(objectness)
        anomaly_norm = normalize(anomaly)
        contrast_norm = normalize(contrast)
        
        # 组合
        combined = (
            objectness_norm * 0.4 +
            anomaly_norm * 0.3 +
            contrast_norm * 0.3
        )
        
        return combined
    
    def compute_attention_heatmap(self, image_tensor, H, W):
        """计算attention热图 (原始方法)"""
        # 运行前向传播以捕获attention
        with torch.no_grad():
            _ = self.model(image_tensor)
        
        attention = self.last_attention
        
        if attention is None:
            raise ValueError("Failed to capture attention")
        
        # 处理attention tensor
        if isinstance(attention, (tuple, list)):
            attn = attention[0]
        else:
            attn = attention
        
        # [batch, num_heads, num_tokens, num_tokens]
        if attn.dim() == 4:
            attn = attn.mean(dim=1)  # 平均所有heads
        
        # [batch, num_tokens, num_tokens]
        if attn.dim() == 3:
            attn = attn[:, 0, 1:]  # CLS token对patches的attention
        
        attn = attn.squeeze() if attn.dim() > 1 else attn
        
        # 处理可能的额外tokens
        num_patches = attn.shape[0]
        side = int(num_patches ** 0.5)
        
        if side * side < num_patches:
            attn = attn[:side*side]
        
        attn_map = attn.reshape(side, side).cpu().numpy()
        
        # 归一化
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
        
        return attn_map
    
    def generate_heatmap(self, image, image_size: Tuple[int, int]) -> np.ndarray:
        """
        生成热图
        
        Args:
            image: 输入图像 (numpy array, RGB)
            image_size: 目标尺寸 (width, height)
        
        Returns:
            heatmap: 归一化的热图 [0, 1]
        """
        from torchvision import transforms
        
        # 准备输入
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 根据方法生成热图
        if self.heatmap_method == 'objectness':
            patch_features, H, W = self.extract_patch_features(image_tensor)
            heatmap_2d = self.compute_objectness_heatmap(patch_features, H, W)
        
        elif self.heatmap_method == 'combined':
            patch_features, H, W = self.extract_patch_features(image_tensor)
            heatmap_2d = self.compute_combined_heatmap(patch_features, H, W)
        
        elif self.heatmap_method == 'attention':
            patch_features, H, W = self.extract_patch_features(image_tensor)
            heatmap_2d = self.compute_attention_heatmap(image_tensor, H, W)
        
        else:
            raise ValueError(f"Unknown heatmap method: {self.heatmap_method}")
        
        # 调整到目标尺寸
        heatmap = cv2.resize(heatmap_2d, image_size, interpolation=cv2.INTER_CUBIC)
        
        # 确保归一化
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap


# 使用示例
def example_usage():
    """展示如何使用改进的提取器"""
    import torch
    
    # 加载DINOv3模型
    model = torch.hub.load(
        "/media/pc/D/zhaochen/mono3d/dinov3",
        "dinov3_vitl16",
        source="local",
        trust_repo=True,
        pretrained=False
    )
    
    checkpoint_path = "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth"
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    
    device = torch.device("cuda")
    model = model.to(device).eval()
    
    # 创建提取器 (使用物体性方法)
    extractor = ImprovedDINOv3Extractor(model, device, heatmap_method='objectness')
    
    # 加载图像
    import cv2
    image = cv2.imread("image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 生成热图
    heatmap = extractor.generate_heatmap(image, (image.shape[1], image.shape[0]))
    
    # 现在可以用这个heatmap生成prompts并运行SAM2
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")


if __name__ == "__main__":
    example_usage()