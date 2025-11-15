#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断可学习融合权重问题
检查checkpoint中的权重情况
"""

import torch
import torch.nn.functional as F
import os

def diagnose_checkpoint(checkpoint_path):
    """诊断checkpoint中的融合权重"""
    print("=" * 80)
    print("诊断 Checkpoint 中的融合权重")
    print("=" * 80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return
    
    print(f"\n📦 加载checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查checkpoint结构
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                print("✓ Checkpoint包含'model'键")
            else:
                model_state = checkpoint
                print("✓ Checkpoint直接是模型状态字典")
        else:
            model_state = checkpoint
            print("✓ Checkpoint是模型状态字典")
        
        # 检查fusion_weights_logits
        print("\n" + "-" * 80)
        print("检查 fusion_weights_logits:")
        print("-" * 80)
        
        if 'fusion_weights_logits' in model_state:
            weights = model_state['fusion_weights_logits']
            print(f"✓ 找到 fusion_weights_logits")
            print(f"  - Shape: {weights.shape}")
            print(f"  - Dtype: {weights.dtype}")
            print(f"  - Values (logits): {weights}")
            
            # 计算softmax后的权重
            normalized = F.softmax(weights, dim=0)
            print(f"\n  - 归一化后的权重 (softmax):")
            
            if len(weights) == 2:
                print(f"    * similarity_enhancer: {normalized[0]:.4f}")
                print(f"    * prompt_enhancer: {normalized[1]:.4f}")
                print(f"    * 融合公式: final = r + {normalized[0]:.4f}*r1_delta + {normalized[1]:.4f}*r2_delta")
            elif len(weights) == 3:
                print(f"    * 原始r: {normalized[0]:.4f}")
                print(f"    * similarity_enhancer: {normalized[1]:.4f}")
                print(f"    * prompt_enhancer: {normalized[2]:.4f}")
                print(f"    ⚠️ 这是旧版本的3权重格式！")
                print(f"    * 融合公式: final = {normalized[0]:.4f}*r + {normalized[1]:.4f}*r1 + {normalized[2]:.4f}*r2")
            else:
                print(f"    ⚠️ 未知的权重数量: {len(weights)}")
        else:
            print("❌ Checkpoint中没有 fusion_weights_logits 参数！")
            print("   这意味着checkpoint是在固定权重模式下训练的")
        
        # 检查其他相关参数
        print("\n" + "-" * 80)
        print("检查其他增强器相关参数:")
        print("-" * 80)
        
        enhancer_keys = [k for k in model_state.keys() if 'enhancer' in k.lower() or 'fusion' in k.lower()]
        if enhancer_keys:
            print(f"✓ 找到 {len(enhancer_keys)} 个相关参数:")
            for key in enhancer_keys[:10]:  # 只显示前10个
                param = model_state[key]
                if isinstance(param, torch.Tensor):
                    print(f"  - {key}: shape={param.shape}, dtype={param.dtype}")
                else:
                    print(f"  - {key}: {type(param)}")
        else:
            print("⚠️ 没有找到增强器相关参数")
        
        # 检查模型结构
        print("\n" + "-" * 80)
        print("检查模型结构信息:")
        print("-" * 80)
        
        # 尝试推断模型类型
        has_similarity_enhancer = any('similarity' in k for k in model_state.keys())
        has_prompt_enhancer = any('prompt' in k for k in model_state.keys())
        
        print(f"  - 包含similarity_enhancer: {has_similarity_enhancer}")
        print(f"  - 包含prompt_enhancer: {has_prompt_enhancer}")
        
        # 统计参数数量
        total_params = len(model_state)
        print(f"  - 总参数数量: {total_params}")
        
    except Exception as e:
        print(f"❌ 加载checkpoint失败: {str(e)}")
        import traceback
        traceback.print_exc()

def check_inference_logs(log_file):
    """检查推理日志中的警告"""
    print("\n" + "=" * 80)
    print("检查推理日志中的警告")
    print("=" * 80)
    
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找警告信息
    warnings = []
    for i, line in enumerate(lines):
        if 'warning' in line.lower() or 'Warning' in line or '缺失' in line or 'Missing' in line:
            warnings.append((i+1, line.strip()))
    
    if warnings:
        print(f"⚠️ 找到 {len(warnings)} 个警告:")
        for line_num, warning in warnings[:10]:  # 只显示前10个
            print(f"  Line {line_num}: {warning}")
    else:
        print("✓ 没有找到警告信息")

if __name__ == '__main__':
    # 检查checkpoint
    checkpoint_path = '/T20030104/ynj/semma/ckpts/fusion.pth'
    diagnose_checkpoint(checkpoint_path)
    
    # 检查推理日志
    log_file = '/T20030104/ynj/semma/fusion_output/Ultra/FBNELL/2025-11-15-10-04-19/log.txt'
    check_inference_logs(log_file)
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

