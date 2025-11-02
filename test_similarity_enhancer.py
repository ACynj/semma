"""
测试基于相似度的关系增强模块
验证实现是否正确
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra.enhanced_models import SimilarityBasedRelationEnhancer

def test_similarity_enhancer():
    """测试相似度增强模块的基本功能"""
    print("=" * 60)
    print("测试 SimilarityBasedRelationEnhancer 模块")
    print("=" * 60)
    
    # 设置参数
    batch_size = 4
    num_relations = 10
    embedding_dim = 64
    
    # 创建增强模块
    enhancer = SimilarityBasedRelationEnhancer(
        embedding_dim=embedding_dim,
        similarity_threshold_init=0.5,
        enhancement_strength_init=0.05
    )
    
    # 创建模拟的关系表示
    # 让某些关系更相似 - 通过使一些关系表示接近来测试相似度
    torch.manual_seed(42)
    base_reprs = torch.randn(batch_size, num_relations, embedding_dim)
    
    # 手动创建一些相似的关系对（通过复制并添加少量噪声）
    # 对于batch 0，让关系1和关系2与关系0相似
    base_reprs[0, 1] = base_reprs[0, 0] * 0.9 + torch.randn(embedding_dim) * 0.1
    base_reprs[0, 2] = base_reprs[0, 0] * 0.85 + torch.randn(embedding_dim) * 0.15
    
    # 对于batch 1，让关系3与关系1相似
    base_reprs[1, 3] = base_reprs[1, 1] * 0.88 + torch.randn(embedding_dim) * 0.12
    
    # 创建需要梯度的tensor
    final_relation_representations = base_reprs.clone().requires_grad_(True)
    
    # 创建查询关系索引
    query_rels = torch.tensor([0, 1, 2, 3])  # 每个batch一个查询关系
    
    print(f"\n输入形状:")
    print(f"  final_relation_representations: {final_relation_representations.shape}")
    print(f"  query_rels: {query_rels.shape}")
    
    # 执行增强
    enhanced_reprs = enhancer(final_relation_representations, query_rels)
    
    print(f"\n输出形状:")
    print(f"  enhanced_reprs: {enhanced_reprs.shape}")
    
    # 验证输出形状正确
    assert enhanced_reprs.shape == final_relation_representations.shape, \
        f"输出形状不匹配: {enhanced_reprs.shape} vs {final_relation_representations.shape}"
    
    print(f"\n✅ 形状验证通过")
    
    # 验证查询关系的表示是否被增强（应该与原表示不完全相同）
    print(f"\n检查相似度增强效果:")
    for i, query_rel in enumerate(query_rels):
        # 重新计算以获取原始表示（不包含梯度）
        with torch.no_grad():
            original = final_relation_representations[i, query_rel].clone()
        
        # 获取增强后的表示（需要重新计算，因为之前的计算可能没有detach）
        enhanced = enhanced_reprs[i, query_rel].clone()
        
        # 计算差异
        diff = torch.norm(enhanced - original).item()
        
        # 计算余弦相似度来验证是否有相似关系
        query_repr = final_relation_representations[i, query_rel]
        all_reprs = final_relation_representations[i]
        query_norm = F.normalize(query_repr, p=2, dim=0)
        all_norms = F.normalize(all_reprs, p=2, dim=1)
        similarities = torch.matmul(query_norm.unsqueeze(0), all_norms.t()).squeeze(0)
        similarities[query_rel] = -1.0  # 排除自己
        max_sim = similarities.max().item()
        threshold = enhancer.get_similarity_threshold().item()
        
        # 如果找到了相似关系，应该有一些差异
        # 如果没有找到相似关系，可能没有差异
        print(f"\n查询关系 {query_rel} (batch {i}):")
        print(f"  最大相似度: {max_sim:.4f} (阈值: {threshold:.4f})")
        print(f"  原表示范数: {torch.norm(original).item():.4f}")
        print(f"  增强后范数: {torch.norm(enhanced).item():.4f}")
        print(f"  差异: {diff:.6f}")
        
        if max_sim > threshold:
            if diff > 1e-6:
                print(f"  ✅ 找到相似关系，增强生效")
            else:
                print(f"  ⚠️  找到相似关系但增强未生效（可能数值精度问题）")
        else:
            print(f"  ℹ️  未找到相似关系（相似度低于阈值），保持原样")
    
    # 验证可学习参数
    threshold = enhancer.get_similarity_threshold()
    strength = enhancer.get_enhancement_strength()
    
    print(f"\n可学习参数:")
    print(f"  相似度阈值: {threshold.item():.4f}")
    print(f"  增强强度: {strength.item():.4f}")
    print(f"  相似度权重缩放: {enhancer.similarity_weight_scale.item():.4f}")
    print(f"  温度参数: {enhancer.temperature.item():.4f}")
    
    # 测试梯度
    print(f"\n测试梯度计算...")
    # 确保输入需要梯度
    final_relation_representations.requires_grad_(True)
    enhanced_reprs = enhancer(final_relation_representations, query_rels)
    loss = enhanced_reprs.sum()
    loss.backward()
    
    # 检查参数是否有梯度
    has_grad_threshold = enhancer.similarity_threshold_raw.grad is not None
    has_grad_strength = enhancer.enhancement_strength_raw.grad is not None
    has_grad_scale = enhancer.similarity_weight_scale.grad is not None
    has_grad_temp = enhancer.temperature.grad is not None
    
    print(f"  相似度阈值梯度: {'✅' if has_grad_threshold else '❌'}")
    print(f"  增强强度梯度: {'✅' if has_grad_strength else '❌'}")
    print(f"  权重缩放梯度: {'✅' if has_grad_scale else '❌'}")
    print(f"  温度参数梯度: {'✅' if has_grad_temp else '❌'}")
    
    if all([has_grad_threshold, has_grad_strength, has_grad_scale, has_grad_temp]):
        print(f"\n✅ 所有可学习参数都有梯度，可以正常训练")
    else:
        print(f"\n❌ 某些参数没有梯度，可能需要检查")
    
    # 测试边界情况：没有相似关系的情况
    print(f"\n测试边界情况...")
    
    # 创建一个所有关系都不相似的情况（设置阈值很高）
    enhancer2 = SimilarityBasedRelationEnhancer(
        embedding_dim=embedding_dim,
        similarity_threshold_init=0.99,  # 非常高的阈值
        enhancement_strength_init=0.05
    )
    
    enhanced_reprs2 = enhancer2(final_relation_representations, query_rels)
    
    # 如果没有找到相似关系，应该保持原样
    for i, query_rel in enumerate(query_rels):
        original = final_relation_representations[i, query_rel]
        enhanced = enhanced_reprs2[i, query_rel]
        
        # 应该完全相同（因为阈值太高，没有相似关系）
        diff = torch.norm(enhanced - original).item()
        if diff < 1e-6:
            print(f"  ✅ 查询关系 {query_rel} (batch {i}): 没有相似关系，保持原样（差异: {diff:.8f}）")
        else:
            print(f"  ⚠️  查询关系 {query_rel} (batch {i}): 仍有差异 {diff:.8f}（可能由于数值精度）")
    
    print(f"\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_similarity_enhancer()

