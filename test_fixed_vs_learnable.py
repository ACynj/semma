#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试固定权重 vs 可学习权重的实际效果
分析为什么可学习融合效果差
"""

import torch
import torch.nn.functional as F

def analyze_weight_impact():
    """分析权重对融合结果的影响"""
    print("=" * 80)
    print("分析固定权重 vs 可学习权重的差异")
    print("=" * 80)
    
    # 加载checkpoint中的权重
    checkpoint_path = '/T20030104/ynj/semma/ckpts/fusion.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model']
    
    trained_weights_logits = model_state['fusion_weights_logits']
    trained_weights = F.softmax(trained_weights_logits, dim=0)
    
    # 固定权重
    fixed_sim_weight = 0.2
    fixed_prompt_weight = 0.8
    fixed_total = fixed_sim_weight + fixed_prompt_weight
    fixed_weights = torch.tensor([
        fixed_sim_weight / fixed_total,
        fixed_prompt_weight / fixed_total
    ])
    
    print(f"\n权重对比:")
    print(f"  训练后权重: similarity={trained_weights[0]:.4f}, prompt={trained_weights[1]:.4f}")
    print(f"  固定权重:   similarity={fixed_weights[0]:.4f}, prompt={fixed_weights[1]:.4f}")
    print(f"  差异:       similarity={abs(trained_weights[0] - fixed_weights[0]):.4f}, prompt={abs(trained_weights[1] - fixed_weights[1]):.4f}")
    
    # 模拟融合过程
    print(f"\n" + "-" * 80)
    print("模拟融合过程（假设r1_delta和r2_delta的典型值）:")
    print("-" * 80)
    
    # 假设的增量值（基于embedding维度64，典型值在-0.1到0.1之间）
    r1_delta_example = torch.randn(64) * 0.05  # similarity增强增量
    r2_delta_example = torch.randn(64) * 0.05  # prompt增强增量
    
    # 使用训练后的权重
    result_trained = trained_weights[0] * r1_delta_example + trained_weights[1] * r2_delta_example
    
    # 使用固定权重
    result_fixed = fixed_weights[0] * r1_delta_example + fixed_weights[1] * r2_delta_example
    
    # 计算差异
    diff = torch.norm(result_trained - result_fixed).item()
    
    print(f"  使用训练后权重的融合结果范数: {torch.norm(result_trained).item():.6f}")
    print(f"  使用固定权重的融合结果范数:   {torch.norm(result_fixed).item():.6f}")
    print(f"  差异范数: {diff:.6f}")
    print(f"  相对差异: {(diff / torch.norm(result_fixed).item() * 100):.2f}%")
    
    # 分析可能的问题
    print(f"\n" + "-" * 80)
    print("可能的问题:")
    print("-" * 80)
    
    # 问题1：权重学习到了次优值
    print(f"\n1. 权重学习到了次优值")
    print(f"   - 训练后权重: similarity={trained_weights[0]:.4f} (比固定权重高12.5%)")
    print(f"   - 这可能意味着模型认为similarity增强更重要")
    print(f"   - 但如果similarity增强效果不好，这会导致性能下降")
    
    # 问题2：训练时的目标函数可能不适合
    print(f"\n2. 训练时的目标函数可能不适合")
    print(f"   - 预训练时使用的是MultiGraphPretraining任务")
    print(f"   - 这个任务可能不适合学习融合权重")
    print(f"   - 融合权重可能在预训练时学习到了次优值")
    
    # 问题3：权重初始化可能影响最终结果
    print(f"\n3. 权重初始化可能影响最终结果")
    print(f"   - 初始权重: similarity=0.2, prompt=0.8")
    print(f"   - 训练后权重: similarity=0.225, prompt=0.775")
    print(f"   - 权重变化不大，说明可能没有充分学习")
    
    # 建议
    print(f"\n" + "-" * 80)
    print("建议:")
    print("-" * 80)
    print(f"1. 立即测试：使用固定权重模式验证")
    print(f"   - 设置 use_learnable_fusion: False")
    print(f"   - 如果效果恢复，说明问题在于训练后的权重")
    print(f"\n2. 如果固定权重效果好，可以:")
    print(f"   - 使用固定权重模式")
    print(f"   - 或者使用训练后的权重值作为固定权重")
    print(f"   - 或者重新训练，调整学习率或训练策略")
    print(f"\n3. 检查训练过程:")
    print(f"   - 查看训练日志，确认权重是否正常学习")
    print(f"   - 检查训练时的性能，确认是否比固定权重好")
    print(f"   - 如果训练时性能就不好，说明训练策略有问题")

if __name__ == '__main__':
    analyze_weight_impact()

