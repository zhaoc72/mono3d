#!/usr/bin/env python3
"""诊断 DINOv3 模型结构和 attention 提取"""

import torch
from pathlib import Path

def diagnose_dinov3():
    """诊断 DINOv3 模型"""
    
    print("=" * 70)
    print("DINOv3 模型诊断")
    print("=" * 70)
    
    # 配置
    repo_or_dir = "/media/pc/D/zhaochen/mono3d/dinov3"
    model_name = "dinov3_vitl16"
    checkpoint_path = "/media/pc/D/zhaochen/mono3d/dinov3/checkpoints/dinov3_vitl16_lvd1689m.pth"
    
    print(f"\n模型配置:")
    print(f"  仓库目录: {repo_or_dir}")
    print(f"  模型名称: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # 加载模型
    print("\n" + "=" * 70)
    print("步骤 1: 加载模型")
    print("=" * 70)
    
    try:
        model = torch.hub.load(
            repo_or_dir,
            model_name,
            source="local",
            trust_repo=True,
            pretrained=False
        )
        print("✅ 模型加载成功")
        print(f"模型类型: {type(model)}")
        print(f"模型类名: {model.__class__.__name__}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载 checkpoint
    if checkpoint_path:
        print(f"\n加载 checkpoint...")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("✅ Checkpoint 加载成功")
    
    # 检查模型属性
    print("\n" + "=" * 70)
    print("步骤 2: 检查模型属性")
    print("=" * 70)
    
    print("\n模型的主要属性:")
    for attr in dir(model):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # 检查是否有 get_last_selfattention
    print("\n关键方法检查:")
    has_get_last_attn = hasattr(model, 'get_last_selfattention')
    print(f"  get_last_selfattention: {'✅ 存在' if has_get_last_attn else '❌ 不存在'}")
    
    if has_get_last_attn:
        import inspect
        sig = inspect.signature(model.get_last_selfattention)
        print(f"    签名: {sig}")
    
    # 检查模型结构
    print("\n" + "=" * 70)
    print("步骤 3: 检查模型结构")
    print("=" * 70)
    
    print("\n顶层模块:")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
    
    # 检查是否有 blocks
    if hasattr(model, 'blocks'):
        blocks = model.blocks
        print(f"\n✅ 找到 blocks: {len(blocks)} 个")
        
        # 检查最后一个 block
        if len(blocks) > 0:
            last_block = blocks[-1]
            print(f"\n最后一个 block 的子模块:")
            for name, module in last_block.named_children():
                print(f"  {name}: {type(module).__name__}")
            
            # 检查 attention
            if hasattr(last_block, 'attn'):
                print(f"\n✅ 找到 attn 模块")
                attn = last_block.attn
                print(f"  类型: {type(attn).__name__}")
                print(f"  attn 的子模块:")
                for name, module in attn.named_children():
                    print(f"    {name}: {type(module).__name__}")
    
    # 测试前向传播
    print("\n" + "=" * 70)
    print("步骤 4: 测试前向传播和 attention 提取")
    print("=" * 70)
    
    model.eval()
    
    # 创建测试输入
    test_input = torch.randn(1, 3, 224, 224)
    
    print("\n测试输入形状:", test_input.shape)
    
    # 注册 hook
    attention_captured = {"value": None}
    
    def capture_attention(module, input, output):
        attention_captured["value"] = output.detach()
        print(f"  Hook 捕获到输出形状: {output.shape}")
    
    # 尝试在不同位置注册 hook
    hooks_registered = []
    
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        last_block = model.blocks[-1]
        
        # 尝试所有可能的 attention 相关模块
        for name, module in last_block.named_modules():
            if 'attn' in name.lower():
                handle = module.register_forward_hook(capture_attention)
                hooks_registered.append((name, handle))
                print(f"  注册 hook: {name} ({type(module).__name__})")
    
    # 执行前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ 前向传播完成")
    print(f"  输出类型: {type(output)}")
    if isinstance(output, torch.Tensor):
        print(f"  输出形状: {output.shape}")
    elif isinstance(output, dict):
        print(f"  输出字典键: {output.keys()}")
    
    # 检查是否捕获到 attention
    print(f"\nHook 捕获的 attention: ", end="")
    if attention_captured["value"] is not None:
        print(f"✅ 成功")
        print(f"  形状: {attention_captured['value'].shape}")
    else:
        print("❌ 失败")
    
    # 测试 get_last_selfattention
    if has_get_last_attn:
        print(f"\n测试 get_last_selfattention:")
        try:
            # 尝试不同的调用方式
            try:
                attn = model.get_last_selfattention(test_input)
                print(f"  ✅ 成功 (带参数)")
                print(f"  形状: {attn.shape if attn is not None else None}")
            except TypeError:
                attn = model.get_last_selfattention()
                print(f"  ✅ 成功 (无参数)")
                print(f"  形状: {attn.shape if attn is not None else None}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    # 清理 hooks
    for name, handle in hooks_registered:
        handle.remove()
    
    # 总结
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    
    print("\n建议:")
    if has_get_last_attn:
        print("  1. 模型有 get_last_selfattention 方法，应该使用它")
        print("  2. 需要在前向传播后调用 get_last_selfattention()")
    elif attention_captured["value"] is not None:
        print("  1. Hook 捕获成功，使用 hook 方法")
    else:
        print("  1. ⚠️ 需要找到正确的 attention 提取方法")
        print("  2. 检查模型文档或源代码")
    
    return model


if __name__ == "__main__":
    diagnose_dinov3()