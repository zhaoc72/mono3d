#!/usr/bin/env python3
"""测试不同的 attention 提取策略"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/mono3d')

from src.dinov3_feature import DINOv3FeatureExtractor, Dinov3Config
from src.data_loader import load_image
from src.utils import to_torch_dtype

print("=" * 70)
print("测试不同的 Attention 提取策略")
print("=" * 70)

# 加载图像
image_path = "/media/pc/D/datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"
sample = load_image(image_path)

print(f"\n图像: {sample.image.shape}")

# 初始化模型
dinov3_cfg = Dinov3Config(
    repo_or_dir="/media/pc/D/zhaochen/mono3d/dinov3",
    model_name="dinov3_vitl16",
    use_torch_hub=True,
    torchhub_source="local",
    checkpoint_path="/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth",
)

device = torch.device("cuda")
dtype = torch.float16

extractor = DINOv3FeatureExtractor(dinov3_cfg, device, dtype)
print("✅ DINOv3 加载成功\n")

# 提取 attention
feats = extractor.extract(sample.image)
attention = feats["attention"]

print(f"Attention 形状: {attention.shape}")
# [batch, num_heads, num_tokens, num_tokens]

B, num_heads, N, _ = attention.shape
print(f"  Batch: {B}, Heads: {num_heads}, Tokens: {N}")

# 移除 CLS token
patch_attn = attention[:, :, 1:, 1:]
num_patches = patch_attn.shape[2]
side = int(num_patches ** 0.5)

print(f"  Patches: {num_patches}, Grid: {side}x{side}")

output_dir = Path("outputs/attention_comparison")
output_dir.mkdir(exist_ok=True, parents=True)

image_size = (sample.image.shape[1], sample.image.shape[0])

def process_and_save(attn_map, name, description):
    """处理并保存 attention map"""
    # 归一化
    attn_map = attn_map.cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 调整大小
    attn_2d = attn_map.reshape(side, side)
    resized = cv2.resize(attn_2d, image_size, interpolation=cv2.INTER_CUBIC)
    
    # 保存原始
    heatmap_vis = (resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{name}_raw.png"), heatmap_colored)
    
    # 叠加到原图
    overlay = sample.image.copy()
    overlay = cv2.addWeighted(
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        0.6,
        heatmap_colored,
        0.4,
        0
    )
    cv2.imwrite(str(output_dir / f"{name}_overlay.png"), overlay)
    
    print(f"  ✅ {name}: {description}")
    print(f"     Range: [{attn_map.min():.3f}, {attn_map.max():.3f}]")
    return resized

print("\n" + "=" * 70)
print("策略 1: CLS token attention (原始方法)")
print("=" * 70)
# CLS token 对所有 patches 的 attention
cls_attn = attention[:, :, 0, 1:].mean(dim=(0, 1))  # [num_patches]
cls_attn = cls_attn[:side*side]
process_and_save(cls_attn, "strategy_1_cls", "CLS token attention to patches")

print("\n" + "=" * 70)
print("策略 2: Mean attention received")
print("=" * 70)
# 每个 patch 接收到的平均 attention
patch_attn_2d = patch_attn[:, :, :side*side, :side*side]
received = patch_attn_2d.mean(dim=(0, 1, 3))  # [num_patches]
process_and_save(received, "strategy_2_received", "Mean attention received by each patch")

print("\n" + "=" * 70)
print("策略 3: Mean attention given")
print("=" * 70)
# 每个 patch 给出的平均 attention
given = patch_attn_2d.mean(dim=(0, 1, 2))  # [num_patches]
process_and_save(given, "strategy_3_given", "Mean attention given by each patch")

print("\n" + "=" * 70)
print("策略 4: 组合 (received + given)")
print("=" * 70)
combined = (received + given) / 2.0
process_and_save(combined, "strategy_4_combined", "Combined attention")

print("\n" + "=" * 70)
print("策略 5: 最大 attention head")
print("=" * 70)
# 找到最显著的 attention head
head_importance = []
for h in range(num_heads):
    head_attn = patch_attn_2d[:, h, :, :].mean(dim=(0, 1, 2))
    variance = head_attn.var().item()
    head_importance.append((h, variance))

head_importance.sort(key=lambda x: x[1], reverse=True)
best_head = head_importance[0][0]

print(f"  最显著的 head: {best_head} (variance={head_importance[0][1]:.4f})")
print(f"  前 5 个 heads: {[h for h, v in head_importance[:5]]}")

best_head_attn = patch_attn_2d[:, best_head, :, :].mean(dim=(0, 1))
process_and_save(best_head_attn, "strategy_5_best_head", f"Best attention head #{best_head}")

print("\n" + "=" * 70)
print("策略 6: Top-K heads 平均")
print("=" * 70)
top_k = 4
top_heads = [h for h, v in head_importance[:top_k]]
print(f"  使用 heads: {top_heads}")

topk_attn = patch_attn_2d[:, top_heads, :, :].mean(dim=(0, 1, 2))
process_and_save(topk_attn, "strategy_6_topk", f"Top-{top_k} attention heads")

print("\n" + "=" * 70)
print("策略 7: Patch self-attention")
print("=" * 70)
# 每个 patch 对自己的 attention (对角线)
self_attn = torch.diagonal(patch_attn_2d, dim1=-2, dim2=-1).mean(dim=(0, 1))
self_attn = self_attn[:side*side]
process_and_save(self_attn, "strategy_7_self", "Patch self-attention")

print("\n" + "=" * 70)
print("策略 8: 增强对比度 (gamma correction)")
print("=" * 70)
# 对组合策略应用 gamma correction
combined_np = combined.cpu().numpy()
combined_np = (combined_np - combined_np.min()) / (combined_np.max() - combined_np.min() + 1e-8)
enhanced = np.power(combined_np, 2.0)  # gamma = 2.0
enhanced = torch.from_numpy(enhanced)
process_and_save(enhanced, "strategy_8_enhanced", "Enhanced with gamma=2.0")

print("\n" + "=" * 70)
print("完成！所有结果保存在: outputs/attention_comparison/")
print("=" * 70)
print("\n建议:")
print("1. 查看每种策略的可视化结果")
print("2. 选择最能突出显著区域的策略")
print("3. 根据结果更新 dinov3_feature.py 中的 attention_to_heatmap 方法")