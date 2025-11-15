#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比可学习融合和固定权重融合的效果
分析为什么可学习融合效果差
"""

import torch
import torch.nn.functional as F

def analyze_fusion_weights():
    """分析融合权重"""
    print("=" * 80)
    print("分析融合权重问题")
    print("=" * 80)
    
    # 加载checkpoint中的权重
    checkpoint_path = '/T20030104/ynj/semma/ckpts/fusion.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model']
    
    # 获取训练后的权重
    trained_weights_logits = model_state['fusion_weights_logits']
    trained_weights = F.softmax(trained_weights_logits, dim=0)
    
    print(f"\n训练后的权重（从checkpoint加载）:")
    print(f"  - Logits: {trained_weights_logits}")
    print(f"  - 归一化后: similarity={trained_weights[0]:.4f}, prompt={trained_weights[1]:.4f}")
    
    # 计算固定权重（从flags.yaml）
    fixed_sim_weight = 0.2
    fixed_prompt_weight = 0.8
    fixed_total = fixed_sim_weight + fixed_prompt_weight
    fixed_weights = torch.tensor([
        fixed_sim_weight / fixed_total,  # 0.2 / 1.0 = 0.2
        fixed_prompt_weight / fixed_total  # 0.8 / 1.0 = 0.8
    ])
    
    print(f"\n固定权重（flags.yaml中的配置）:")
    print(f"  - similarity={fixed_weights[0]:.4f}, prompt={fixed_weights[1]:.4f}")
    
    # 对比
    print(f"\n对比分析:")
    print(f"  - 训练后权重: similarity={trained_weights[0]:.4f}, prompt={trained_weights[1]:.4f}")
    print(f"  - 固定权重: similarity={fixed_weights[0]:.4f}, prompt={fixed_weights[1]:.4f}")
    print(f"  - 差异: similarity={abs(trained_weights[0] - fixed_weights[0]):.4f}, prompt={abs(trained_weights[1] - fixed_weights[1]):.4f}")
    
    # 分析可能的问题
    print(f"\n" + "-" * 80)
    print("可能的问题分析:")
    print("-" * 80)
    
    # 问题1：权重学习到了次优值
    if abs(trained_weights[0] - fixed_weights[0]) > 0.1:
        print(f"⚠️ 问题1: 训练后的权重与固定权重差异较大")
        print(f"   训练后的权重可能学习到了次优值")
        print(f"   建议: 使用固定权重或调整训练策略")
    
    # 问题2：权重初始化不一致
    print(f"\n问题2: 权重初始化分析")
    print(f"   训练时初始权重: similarity=0.2, prompt=0.8 (归一化后)")
    print(f"   训练后权重: similarity={trained_weights[0]:.4f}, prompt={trained_weights[1]:.4f}")
    print(f"   权重变化: similarity从0.2变为{trained_weights[0]:.4f} (变化{((trained_weights[0]-0.2)/0.2*100):.1f}%)")
    print(f"            prompt从0.8变为{trained_weights[1]:.4f} (变化{((trained_weights[1]-0.8)/0.8*100):.1f}%)")
    
    # 问题3：检查是否有其他配置不一致
    print(f"\n问题3: 检查配置一致性")
    print(f"   需要确认训练时和推理时的flags.yaml配置是否一致")
    print(f"   特别是:")
    print(f"     - use_learnable_fusion")
    print(f"     - similarity_enhancer_weight")
    print(f"     - prompt_enhancer_weight")
    print(f"     - similarity_threshold_init")
    print(f"     - enhancement_strength_init")
    
    # 建议
    print(f"\n" + "-" * 80)
    print("建议:")
    print("-" * 80)
    print(f"1. 如果训练后的权重效果不好，可以尝试:")
    print(f"   - 使用固定权重模式 (use_learnable_fusion: False)")
    print(f"   - 或者使用训练后的权重值作为固定权重")
    print(f"   - 或者重新训练，调整学习率或初始权重")
    print(f"\n2. 如果训练后的权重与固定权重差异大，说明:")
    print(f"   - 模型学习到了不同的融合策略")
    print(f"   - 需要评估哪种策略更好")
    print(f"\n3. 检查训练日志，确认:")
    print(f"   - 训练时的配置")
    print(f"   - 训练过程中的权重变化")
    print(f"   - 训练时的性能")

if __name__ == '__main__':
    analyze_fusion_weights()

