#!/usr/bin/env python3
"""DINOv3 特征可视化 - 简化版"""

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

print("=" * 70)
print("DINOv3 特征可视化 - 聚焦最有用的方法")
print("=" * 70)

# 加载图像
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00061.jpg"
image = Image.open(image_path).convert('RGB')
image_np = np.array(image)
print(f"图像: {image_np.shape}")

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
print(f"✅ 模型加载成功 (device: {device})\n")

# 准备输入
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_tensor = transform(image).unsqueeze(0).to(device)

output_dir = Path("outputs/dinov3_simple_viz")
output_dir.mkdir(exist_ok=True, parents=True)

# 保存原图
cv2.imwrite(str(output_dir / "original.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

# 提取特征
with torch.no_grad():
    features_dict = model.forward_features(image_tensor)
    patch_features = features_dict['x_norm_patchtokens'][0]  # [N, D]
    
    H = W = int(np.sqrt(len(patch_features)))
    print(f"Patch 网格: {H}x{W} ({len(patch_features)} patches)")
    print(f"特征维度: {patch_features.shape[1]}\n")

def save_heatmap(data, name, title):
    """保存热图"""
    # 归一化
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # 调整大小
    data_resized = cv2.resize(
        data,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    
    # 保存热图
    heatmap_vis = (data_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{name}.png"), heatmap_colored)
    
    # 叠加
    overlay = cv2.addWeighted(
        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), 0.6,
        heatmap_colored, 0.4, 0
    )
    cv2.imwrite(str(output_dir / f"{name}_overlay.png"), overlay)
    
    print(f"✅ {title}")
    print(f"   范围: [{data.min():.3f}, {data.max():.3f}]")
    return data_resized

# 方法 1: 特征范数（强度）
print("=" * 70)
print("方法 1: 特征范数")
print("=" * 70)
feature_norms = torch.norm(patch_features, dim=-1).cpu().numpy()
feature_norms_2d = feature_norms.reshape(H, W)
save_heatmap(feature_norms_2d, "01_feature_norm", "特征范数")

# 方法 2: 物体性（Objectness）
print("\n" + "=" * 70)
print("方法 2: 物体性评分")
print("=" * 70)
patch_features_norm = F.normalize(patch_features, dim=-1)
similarity_matrix = torch.mm(patch_features_norm, patch_features_norm.t())

K = 20
topk_sim, _ = torch.topk(similarity_matrix, k=K+1, dim=1, largest=True)
avg_similarity = topk_sim[:, 1:].mean(dim=1)
objectness = (1 - avg_similarity).cpu().numpy()
objectness_2d = objectness.reshape(H, W)
save_heatmap(objectness_2d, "02_objectness", "物体性评分")

# 方法 3: 异常度（与全局均值的距离）
print("\n" + "=" * 70)
print("方法 3: 异常度")
print("=" * 70)
mean_feature = patch_features.mean(dim=0, keepdim=True)
distances = torch.norm(patch_features - mean_feature, dim=-1).cpu().numpy()
anomaly_2d = distances.reshape(H, W)
save_heatmap(anomaly_2d, "03_anomaly", "异常度")

# 方法 4: 局部对比度
print("\n" + "=" * 70)
print("方法 4: 局部对比度")
print("=" * 70)
feature_map = patch_features.reshape(H, W, -1)

# 计算每个 patch 与其 8 邻域的平均差异
contrast = torch.zeros(H, W, device=device)
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        if di == 0 and dj == 0:
            continue
        
        # 提取邻域
        i_start = max(0, -di)
        i_end = H + min(0, -di)
        j_start = max(0, -dj)
        j_end = W + min(0, -dj)
        
        center = feature_map[i_start:i_end, j_start:j_end, :]
        neighbor = feature_map[i_start+di:i_end+di, j_start+dj:j_end+dj, :]
        
        diff = torch.norm(center - neighbor, dim=-1)
        contrast[i_start:i_end, j_start:j_end] += diff

contrast = contrast.cpu().numpy()
save_heatmap(contrast, "04_local_contrast", "局部对比度")

# 方法 5: 谱聚类显著性
print("\n" + "=" * 70)
print("方法 5: 谱方法")
print("=" * 70)
# 使用相似度矩阵的特征值分解
sim_np = similarity_matrix.cpu().numpy()

# Laplacian
degree = sim_np.sum(axis=1)
L = np.diag(degree) - sim_np

# 特征分解
eigenvalues, eigenvectors = np.linalg.eigh(L)

# 使用第二小的特征向量（Fiedler vector）
fiedler = eigenvectors[:, 1]
fiedler_2d = fiedler.reshape(H, W)
save_heatmap(fiedler_2d, "05_spectral", "谱方法（Fiedler vector）")

# 方法 6: 组合最佳方法
print("\n" + "=" * 70)
print("方法 6: 组合方法")
print("=" * 70)
# 确保所有数组都是 2D 的
objectness_norm = (objectness_2d - objectness_2d.min()) / (objectness_2d.max() - objectness_2d.min() + 1e-8)
anomaly_norm = (anomaly_2d - anomaly_2d.min()) / (anomaly_2d.max() - anomaly_2d.min() + 1e-8)
contrast_norm = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)

# 组合
combined = (
    objectness_norm * 0.4 +
    anomaly_norm * 0.3 +
    contrast_norm * 0.3
)
save_heatmap(combined, "06_combined", "组合方法")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
print(f"\n结果保存在: {output_dir}/")
print("\n生成的热图（按推荐顺序）:")
print("  1. 02_objectness_overlay.png - 物体性评分（推荐）")
print("  2. 06_combined_overlay.png - 组合方法（推荐）")
print("  3. 04_local_contrast_overlay.png - 局部对比度")
print("  4. 03_anomaly_overlay.png - 异常度")
print("  5. 01_feature_norm_overlay.png - 特征范数")
print("  6. 05_spectral_overlay.png - 谱方法")
print("\n请查看这些图像，特别是前两个！")