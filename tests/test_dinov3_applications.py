#!/usr/bin/env python3
"""测试 DINOv3 的官方示例应用"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

print("=" * 70)
print("DINOv3 官方应用示例")
print("=" * 70)

# 加载图像
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
image = Image.open(image_path).convert('RGB')
image_np = np.array(image)

print(f"\n图像: {image_np.shape}")

# 加载模型
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

print(f"✅ 模型加载成功")

# 准备输入
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_tensor = transform(image).unsqueeze(0).to(device)

output_dir = Path("outputs/dinov3_applications")
output_dir.mkdir(exist_ok=True, parents=True)

# 应用 1: 语义分割 (Emerging Properties)
print("\n" + "=" * 70)
print("应用 1: 基于特征聚类的伪分割")
print("=" * 70)

with torch.no_grad():
    features_dict = model.forward_features(image_tensor)
    
    if isinstance(features_dict, dict) and 'x_norm_patchtokens' in features_dict:
        patch_features = features_dict['x_norm_patchtokens'][0]  # [N, D]
        
        # K-means 聚类
        from sklearn.cluster import KMeans
        
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(patch_features.cpu().numpy())
        
        H = W = int(np.sqrt(len(labels)))
        segmentation = labels.reshape(H, W)
        
        # 着色
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
        ])
        
        seg_colored = colors[segmentation]
        seg_resized = cv2.resize(
            seg_colored.astype(np.uint8),
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        cv2.imwrite(str(output_dir / "kmeans_segmentation.png"), 
                   cv2.cvtColor(seg_resized, cv2.COLOR_RGB2BGR))
        
        # 叠加
        overlay = cv2.addWeighted(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            0.6,
            cv2.cvtColor(seg_resized, cv2.COLOR_RGB2BGR),
            0.4,
            0
        )
        cv2.imwrite(str(output_dir / "kmeans_overlay.png"), overlay)
        
        print(f"✅ K-means 伪分割 ({n_clusters} 类)")

# 应用 2: 边缘检测
print("\n" + "=" * 70)
print("应用 2: 基于特征差异的边缘检测")
print("=" * 70)

with torch.no_grad():
    features_dict = model.forward_features(image_tensor)
    
    if isinstance(features_dict, dict) and 'x_norm_patchtokens' in features_dict:
        patch_features = features_dict['x_norm_patchtokens'][0]  # [N, D]
        
        H = W = int(np.sqrt(len(patch_features)))
        feature_map = patch_features.reshape(H, W, -1)
        
        # 计算相邻 patch 的特征差异
        # 水平差异
        diff_h = torch.norm(feature_map[:-1, :, :] - feature_map[1:, :, :], dim=-1)
        # 垂直差异
        diff_v = torch.norm(feature_map[:, :-1, :] - feature_map[:, 1:, :], dim=-1)
        
        # 组合 (确保在同一设备上)
        edge_map = torch.zeros(H, W, device=feature_map.device)
        edge_map[:-1, :] += diff_h
        edge_map[1:, :] += diff_h
        edge_map[:, :-1] += diff_v
        edge_map[:, 1:] += diff_v
        
        edge_map = edge_map.cpu().numpy()
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
        
        # 调整大小
        edge_resized = cv2.resize(
            edge_map,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # 保存
        edge_vis = (edge_resized * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "feature_edges.png"), edge_vis)
        
        # 叠加
        edge_colored = cv2.applyColorMap(edge_vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            0.6,
            edge_colored,
            0.4,
            0
        )
        cv2.imwrite(str(output_dir / "feature_edges_overlay.png"), overlay)
        
        print(f"✅ 特征边缘检测")

# 应用 3: 显著性（通过与均值的距离）
print("\n" + "=" * 70)
print("应用 3: 显著性检测（特征异常度）")
print("=" * 70)

with torch.no_grad():
    features_dict = model.forward_features(image_tensor)
    
    if isinstance(features_dict, dict) and 'x_norm_patchtokens' in features_dict:
        patch_features = features_dict['x_norm_patchtokens'][0]  # [N, D]
        
        # 计算每个 patch 与全局均值的距离
        mean_feature = patch_features.mean(dim=0, keepdim=True)
        distances = torch.norm(patch_features - mean_feature, dim=-1)
        
        H = W = int(np.sqrt(len(distances)))
        saliency_map = distances.reshape(H, W).cpu().numpy()
        
        # 归一化
        saliency_map = (saliency_map - saliency_map.min()) / \
                      (saliency_map.max() - saliency_map.min())
        
        # 调整大小
        saliency_resized = cv2.resize(
            saliency_map,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # 保存
        saliency_vis = (saliency_resized * 255).astype(np.uint8)
        saliency_colored = cv2.applyColorMap(saliency_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / "saliency_anomaly.png"), saliency_colored)
        
        overlay = cv2.addWeighted(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            0.6,
            saliency_colored,
            0.4,
            0
        )
        cv2.imwrite(str(output_dir / "saliency_anomaly_overlay.png"), overlay)
        
        print(f"✅ 异常度显著性")

# 应用 4: 物体性（Objectness）
print("\n" + "=" * 70)
print("应用 4: 物体性评分")
print("=" * 70)

with torch.no_grad():
    features_dict = model.forward_features(image_tensor)
    
    if isinstance(features_dict, dict) and 'x_norm_patchtokens' in features_dict:
        patch_features = features_dict['x_norm_patchtokens'][0]  # [N, D]
        
        # 归一化特征
        patch_features_norm = F.normalize(patch_features, dim=-1)
        
        # 计算每个 patch 的"独特性"
        # 独特性高 = 与其他 patches 不相似 = 可能是物体
        similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())
        
        # 对每个 patch，计算其最相似的 K 个邻居的平均相似度
        # 相似度低 = 独特 = 物体性高
        K = 10
        topk_sim, _ = torch.topk(similarity_matrix, k=K+1, dim=1, largest=True)
        avg_similarity = topk_sim[:, 1:].mean(dim=1)  # 排除自己
        
        # 物体性 = 1 - 相似度
        objectness = 1 - avg_similarity
        
        H = W = int(np.sqrt(len(objectness)))
        objectness_map = objectness.reshape(H, W).cpu().numpy()
        
        # 归一化
        objectness_map = (objectness_map - objectness_map.min()) / \
                        (objectness_map.max() - objectness_map.min())
        
        # 调整大小
        objectness_resized = cv2.resize(
            objectness_map,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # 保存
        objectness_vis = (objectness_resized * 255).astype(np.uint8)
        objectness_colored = cv2.applyColorMap(objectness_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / "objectness.png"), objectness_colored)
        
        overlay = cv2.addWeighted(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            0.6,
            objectness_colored,
            0.4,
            0
        )
        cv2.imwrite(str(output_dir / "objectness_overlay.png"), overlay)
        
        print(f"✅ 物体性评分")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
print(f"\n所有结果保存在: {output_dir}/")
print("\n应用示例:")
print("  1. kmeans_segmentation.png - K-means 伪分割")
print("  2. feature_edges.png - 基于特征的边缘检测")
print("  3. saliency_anomaly.png - 异常度显著性")
print("  4. objectness.png - 物体性评分")