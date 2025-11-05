# diagnose_mask2former.py
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2
import sys

sys.path.insert(0, '/media/pc/D/zhaochen/mono3d/dinov3')

# 加载模型
model = torch.hub.load(
    '/media/pc/D/zhaochen/mono3d/dinov3',
    'dinov3_vit7b16_ms',
    source='local',
    weights='/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_ade20k.pth',
    backbone_weights='/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vit7b16_lvd1689m.pth',
)
model = model.to('cuda:0').eval()

# 加载图像
image = Image.open('/media/pc/Elements/datasets/coco2017/train2017/000000000532.jpg').convert('RGB')

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((896, 896), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

tensor = transform(image).unsqueeze(0).to('cuda:0')

# 推理
with torch.inference_mode():
    output = model(tensor)

print("=" * 80)
print("输出分析")
print("=" * 80)

pred_logits = output['pred_logits'][0].cpu().float()  # [100, 151]
pred_masks = output['pred_masks'][0].cpu().float()     # [100, 224, 224]

print(f"\npred_logits shape: {pred_logits.shape}")
print(f"pred_masks shape: {pred_masks.shape}")

# 分析 logits
print("\n" + "=" * 80)
print("Logits 分析（前 10 个查询）")
print("=" * 80)

for i in range(10):
    logits = pred_logits[i]
    probs = torch.softmax(logits, dim=0)
    max_prob, max_class = probs.max(dim=0)
    
    # 检查是否是背景类（通常是最后一个类）
    bg_prob = probs[-1].item()
    
    print(f"\n查询 {i}:")
    print(f"  最大概率类别: {max_class.item()} (概率: {max_prob.item():.4f})")
    print(f"  背景类概率: {bg_prob:.4f}")
    print(f"  前5个类别:")
    top5_probs, top5_classes = probs.topk(5)
    for cls, prob in zip(top5_classes, top5_probs):
        print(f"    类别 {cls.item():3d}: {prob.item():.4f}")

# 分析掩码
print("\n" + "=" * 80)
print("Mask 分析（前 10 个查询）")
print("=" * 80)

for i in range(10):
    mask = pred_masks[i]
    print(f"\n查询 {i}:")
    print(f"  mask 范围: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    print(f"  mask 均值: {mask.mean().item():.4f}")
    
    # 应用 sigmoid
    mask_sigmoid = mask.sigmoid()
    print(f"  sigmoid 后范围: [{mask_sigmoid.min().item():.4f}, {mask_sigmoid.max().item():.4f}]")
    print(f"  sigmoid 后均值: {mask_sigmoid.mean().item():.4f}")
    print(f"  sigmoid > 0.5 的像素: {(mask_sigmoid > 0.5).sum().item()} / {mask.numel()}")

print("\n" + "=" * 80)
print("类别统计")
print("=" * 80)

# 对每个查询，找出最可能的类别（排除背景）
valid_queries = []
for i in range(100):
    probs = torch.softmax(pred_logits[i], dim=0)
    
    # 假设最后一个类是背景
    fg_probs = probs[:-1]  # 前 150 个类
    bg_prob = probs[-1]    # 最后一个类
    
    max_fg_prob, max_fg_class = fg_probs.max(dim=0)
    
    if max_fg_prob > bg_prob and max_fg_prob > 0.5:
        valid_queries.append({
            'query_idx': i,
            'class': max_fg_class.item(),
            'fg_prob': max_fg_prob.item(),
            'bg_prob': bg_prob.item()
        })

print(f"\n有效前景查询: {len(valid_queries)} / 100")
print("\n前 20 个有效查询:")
for q in sorted(valid_queries, key=lambda x: -x['fg_prob'])[:20]:
    print(f"  查询 {q['query_idx']:3d}: 类别 {q['class']:3d}, "
          f"前景概率: {q['fg_prob']:.4f}, 背景概率: {q['bg_prob']:.4f}")