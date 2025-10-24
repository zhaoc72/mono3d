#!/usr/bin/env python3
"""测试官方 DINOv3 的原始输出"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 70)
print("测试官方 DINOv3 输出")
print("=" * 70)

# 加载图像
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"
image = Image.open(image_path).convert('RGB')
image_np = np.array(image)

print(f"\n图像信息:")
print(f"  路径: {image_path}")
print(f"  形状: {image_np.shape}")

# 加载 DINOv3 模型
print("\n" + "=" * 70)
print("加载 DINOv3 模型")
print("=" * 70)

model = torch.hub.load(
    "/media/pc/D/zhaochen/mono3d/dinov3",
    "dinov3_vitl16",
    source="local",
    trust_repo=True,
    pretrained=False
)

# 加载 checkpoint
checkpoint_path = "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth"
state = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"✅ 模型加载成功 (device: {device})")

# 准备输入
print("\n" + "=" * 70)
print("准备输入")
print("=" * 70)

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((518, 518)),  # DINOv3 推荐的分辨率
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_tensor = transform(image).unsqueeze(0).to(device)
print(f"输入张量形状: {image_tensor.shape}")

# 提取特征
print("\n" + "=" * 70)
print("提取 DINOv3 特征")
print("=" * 70)

with torch.no_grad():
    # 方法 1: 直接 forward
    output = model(image_tensor)
    print(f"直接输出形状: {output.shape}")
    print(f"输出类型: {type(output)}")
    
    # 方法 2: 使用 forward_features
    if hasattr(model, 'forward_features'):
        features = model.forward_features(image_tensor)
        print(f"\nforward_features 输出: {type(features)}")
        if isinstance(features, torch.Tensor):
            print(f"  形状: {features.shape}")
        elif isinstance(features, dict):
            print(f"  字典键: {features.keys()}")
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
    
    # 方法 3: 使用 get_intermediate_layers
    if hasattr(model, 'get_intermediate_layers'):
        print(f"\n使用 get_intermediate_layers:")
        intermediate = model.get_intermediate_layers(image_tensor, n=1)
        print(f"  输出类型: {type(intermediate)}")
        if isinstance(intermediate, list):
            print(f"  列表长度: {len(intermediate)}")
            for i, layer_out in enumerate(intermediate):
                print(f"    Layer {i}: {layer_out.shape}")

# 可视化特征
print("\n" + "=" * 70)
print("可视化 DINOv3 特征")
print("=" * 70)

output_dir = Path("outputs/dinov3_official_test")
output_dir.mkdir(exist_ok=True, parents=True)

# 保存原始图像
cv2.imwrite(
    str(output_dir / "original.png"), 
    cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
)

# 1. CLS token 特征
if output.dim() == 2:  # [batch, features]
    cls_features = output[0].cpu().numpy()
    print(f"\nCLS token 特征:")
    print(f"  维度: {cls_features.shape}")
    print(f"  范围: [{cls_features.min():.3f}, {cls_features.max():.3f}]")
    print(f"  均值: {cls_features.mean():.3f}")
    print(f"  标准差: {cls_features.std():.3f}")

# 2. Patch tokens 特征
with torch.no_grad():
    # 获取所有 token 的特征
    if hasattr(model, 'forward_features'):
        all_features = model.forward_features(image_tensor)
        
        if isinstance(all_features, dict) and 'x_norm_patchtokens' in all_features:
            patch_tokens = all_features['x_norm_patchtokens']
            print(f"\nPatch tokens 特征:")
            print(f"  形状: {patch_tokens.shape}")
            
            # [batch, num_patches, feature_dim]
            B, N, D = patch_tokens.shape
            H = W = int(np.sqrt(N))
            
            print(f"  网格大小: {H}x{W}")
            
            # 计算 patch 特征的范数作为显著性
            patch_norms = torch.norm(patch_tokens[0], dim=-1).cpu().numpy()
            print(f"  范数范围: [{patch_norms.min():.3f}, {patch_norms.max():.3f}]")
            
            # 可视化 patch 范数
            patch_map = patch_norms.reshape(H, W)
            
            # 归一化
            patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min())
            
            # 调整到原始图像大小
            patch_map_resized = cv2.resize(
                patch_map, 
                (image_np.shape[1], image_np.shape[0]), 
                interpolation=cv2.INTER_CUBIC
            )
            
            # 保存热图
            heatmap_vis = (patch_map_resized * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(output_dir / "patch_norms.png"), heatmap_colored)
            
            # 叠加
            overlay = cv2.addWeighted(
                cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
                0.6,
                heatmap_colored,
                0.4,
                0
            )
            cv2.imwrite(str(output_dir / "patch_norms_overlay.png"), overlay)
            print(f"  ✅ 保存 patch 范数热图")

# 3. 尝试使用 PCA 可视化高维特征
print("\n" + "=" * 70)
print("PCA 可视化")
print("=" * 70)

with torch.no_grad():
    if hasattr(model, 'forward_features'):
        all_features = model.forward_features(image_tensor)
        
        if isinstance(all_features, dict) and 'x_norm_patchtokens' in all_features:
            patch_tokens = all_features['x_norm_patchtokens'][0].cpu().numpy()  # [N, D]
            
            from sklearn.decomposition import PCA
            
            # PCA 降到 3 维用于 RGB 可视化
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(patch_tokens)
            
            print(f"PCA 解释方差比: {pca.explained_variance_ratio_}")
            
            # 归一化到 [0, 255]
            for i in range(3):
                pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                                     (pca_features[:, i].max() - pca_features[:, i].min())
            
            pca_features = (pca_features * 255).astype(np.uint8)
            
            # 重塑为图像
            H = W = int(np.sqrt(len(patch_tokens)))
            pca_image = pca_features.reshape(H, W, 3)
            
            # 调整大小
            pca_resized = cv2.resize(
                pca_image,
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            cv2.imwrite(str(output_dir / "pca_features.png"), 
                       cv2.cvtColor(pca_resized, cv2.COLOR_RGB2BGR))
            print(f"  ✅ 保存 PCA 可视化")

# 4. 尝试自相似度
print("\n" + "=" * 70)
print("特征自相似度")
print("=" * 70)

with torch.no_grad():
    if hasattr(model, 'forward_features'):
        all_features = model.forward_features(image_tensor)
        
        if isinstance(all_features, dict) and 'x_norm_patchtokens' in all_features:
            patch_tokens = all_features['x_norm_patchtokens'][0]  # [N, D]
            
            # 计算相似度矩阵
            # 归一化特征
            patch_tokens_norm = torch.nn.functional.normalize(patch_tokens, dim=-1)
            
            # 余弦相似度
            similarity = torch.mm(patch_tokens_norm, patch_tokens_norm.t())  # [N, N]
            
            # 对每个 patch，计算其与所有其他 patch 的平均相似度
            avg_similarity = similarity.mean(dim=1).cpu().numpy()
            
            print(f"平均相似度范围: [{avg_similarity.min():.3f}, {avg_similarity.max():.3f}]")
            
            H = W = int(np.sqrt(len(avg_similarity)))
            similarity_map = avg_similarity.reshape(H, W)
            
            # 归一化
            similarity_map = (similarity_map - similarity_map.min()) / \
                           (similarity_map.max() - similarity_map.min())
            
            # 调整大小
            similarity_resized = cv2.resize(
                similarity_map,
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
            
            # 保存
            heatmap_vis = (similarity_resized * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(output_dir / "self_similarity.png"), heatmap_colored)
            
            overlay = cv2.addWeighted(
                cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
                0.6,
                heatmap_colored,
                0.4,
                0
            )
            cv2.imwrite(str(output_dir / "self_similarity_overlay.png"), overlay)
            print(f"  ✅ 保存自相似度热图")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
print(f"\n所有结果保存在: {output_dir}/")
print("\n生成的文件:")
print("  - original.png: 原始图像")
print("  - patch_norms.png: Patch 特征范数热图")
print("  - patch_norms_overlay.png: 叠加到原图")
print("  - pca_features.png: PCA 降维可视化")
print("  - self_similarity.png: 特征自相似度热图")
print("  - self_similarity_overlay.png: 自相似度叠加")