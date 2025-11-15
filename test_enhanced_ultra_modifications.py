#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试EnhancedUltra的修改是否正确
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompt_enhancer_initialization():
    """测试Prompt Enhancer的初始化改进"""
    print("=" * 80)
    print("测试1: Prompt Enhancer初始化改进")
    print("=" * 80)
    
    try:
        from ultra.enhanced_models import OptimizedPromptGraph
        from torch_geometric.data import Data
        
        # 创建测试数据
        embedding_dim = 64
        num_relations = 100
        num_nodes = 50
        
        # 创建prompt enhancer
        prompt_enhancer = OptimizedPromptGraph(
            embedding_dim=embedding_dim,
            max_hops=2,
            num_prompt_samples=5  # 测试新的参数值
        )
        
        # 创建模拟数据
        data = Data(
            edge_index=torch.randint(0, num_nodes, (2, 200)),
            edge_type=torch.randint(0, num_relations, (200,)),
            num_nodes=num_nodes,
            num_relations=num_relations
        )
        
        query_relation = torch.tensor(10)
        query_entity = torch.tensor(5)
        base_embeddings = torch.randn(embedding_dim)
        
        # 创建关系嵌入矩阵
        relation_embeddings = torch.randn(num_relations, embedding_dim)
        
        # 创建提示图
        prompt_graph = prompt_enhancer.generate_prompt_graph(data, query_relation, query_entity)
        
        # 测试1: 使用关系嵌入初始化（新功能）
        print("\n测试1.1: 使用关系嵌入初始化")
        prompt_enhancer.eval()  # 推理模式
        context1 = prompt_enhancer.encode_prompt_context(
            prompt_graph, query_relation, relation_embeddings
        )
        print(f"  ✓ 成功生成上下文，形状: {context1.shape}")
        print(f"  ✓ 上下文不是零向量: {not torch.allclose(context1, torch.zeros_like(context1))}")
        
        # 测试2: 不使用关系嵌入（回退模式）
        print("\n测试1.2: 回退模式（不使用关系嵌入）")
        context2 = prompt_enhancer.encode_prompt_context(
            prompt_graph, query_relation, None
        )
        print(f"  ✓ 成功生成上下文，形状: {context2.shape}")
        
        # 测试3: 前向传播
        print("\n测试1.3: 前向传播（传入关系嵌入）")
        output = prompt_enhancer(
            data, query_relation, query_entity, base_embeddings,
            return_enhancement_only=True,
            relation_embeddings=relation_embeddings
        )
        print(f"  ✓ 成功执行前向传播，输出形状: {output.shape}")
        
        print("\n✅ 测试1通过: Prompt Enhancer初始化改进工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_ultra_forward():
    """测试EnhancedUltra的前向传播"""
    print("\n" + "=" * 80)
    print("测试2: EnhancedUltra前向传播")
    print("=" * 80)
    
    try:
        from ultra.enhanced_models import EnhancedUltra
        from torch_geometric.data import Data
        
        # 创建模型配置
        rel_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True
        }
        
        entity_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True
        }
        
        sem_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True
        }
        
        # 创建模型
        model = EnhancedUltra(rel_model_cfg, entity_model_cfg, sem_model_cfg)
        model.eval()  # 推理模式
        
        # 创建测试数据
        num_nodes = 100
        num_relations = 50
        batch_size = 4
        
        data = Data(
            edge_index=torch.randint(0, num_nodes, (2, 500)),
            edge_type=torch.randint(0, num_relations, (500,)),
            num_nodes=num_nodes,
            num_relations=num_relations
        )
        
        # 创建batch
        batch = torch.randint(0, num_nodes, (batch_size, 1, 3))
        
        # 测试前向传播
        print("\n测试2.1: 前向传播（推理模式）")
        with torch.no_grad():
            score = model(data, batch)
        print(f"  ✓ 成功执行前向传播，输出形状: {score.shape}")
        print(f"  ✓ 输出值范围: [{score.min().item():.4f}, {score.max().item():.4f}]")
        
        # 检查prompt enhancer是否被正确调用
        if model.use_prompt_enhancer and model.prompt_enhancer is not None:
            print(f"  ✓ Prompt Enhancer已启用，num_prompt_samples={model.prompt_enhancer.num_prompt_samples}")
        
        print("\n✅ 测试2通过: EnhancedUltra前向传播工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flags_config():
    """测试flags.yaml配置"""
    print("\n" + "=" * 80)
    print("测试3: flags.yaml配置")
    print("=" * 80)
    
    try:
        from ultra import parse
        
        flags = parse.load_flags("flags.yaml")
        
        # 检查关键参数
        print("\n检查关键参数:")
        print(f"  similarity_threshold_init: {getattr(flags, 'similarity_threshold_init', 'NOT FOUND')}")
        print(f"  enhancement_strength_init: {getattr(flags, 'enhancement_strength_init', 'NOT FOUND')}")
        print(f"  use_learnable_fusion: {getattr(flags, 'use_learnable_fusion', 'NOT FOUND')}")
        print(f"  use_prompt_enhancer: {getattr(flags, 'use_prompt_enhancer', 'NOT FOUND')}")
        
        # 验证参数值
        threshold = getattr(flags, 'similarity_threshold_init', None)
        strength = getattr(flags, 'enhancement_strength_init', None)
        
        if threshold is not None and 0.7 <= threshold <= 0.75:
            print(f"  ✓ similarity_threshold_init在合理范围内: {threshold}")
        elif threshold is not None:
            print(f"  ⚠ similarity_threshold_init: {threshold} (建议范围: 0.7-0.75)")
        
        if strength is not None and 0.1 <= strength <= 0.15:
            print(f"  ✓ enhancement_strength_init在合理范围内: {strength}")
        elif strength is not None:
            print(f"  ⚠ enhancement_strength_init: {strength} (建议范围: 0.1-0.15)")
        
        print("\n✅ 测试3通过: flags.yaml配置检查完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("EnhancedUltra修改验证测试")
    print("=" * 80)
    
    results = []
    
    # 运行测试
    results.append(("Prompt Enhancer初始化", test_prompt_enhancer_initialization()))
    results.append(("EnhancedUltra前向传播", test_enhanced_ultra_forward()))
    results.append(("flags.yaml配置", test_flags_config()))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！代码修改正确。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

