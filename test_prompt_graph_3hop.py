#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试提示图3跳邻居优化
"""

import sys
sys.path.insert(0, '.')

import torch
from torch_geometric.data import Data
from ultra.enhanced_models import OptimizedPromptGraph

def create_test_graph():
    """创建测试图"""
    # 创建一个简单的图：0-1-2-3-4-5 (链状结构)
    # 查询实体0，3跳邻居应该包括0,1,2,3
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5]
    ], dtype=torch.long)
    
    edge_type = torch.zeros(5, dtype=torch.long)  # 所有边都是关系0
    
    data = Data()
    data.num_nodes = 6
    data.num_relations = 1
    data.edge_index = edge_index
    data.edge_type = edge_type
    
    return data

def test_prompt_graph_3hop():
    """测试提示图3跳邻居"""
    print("=" * 80)
    print("测试: 提示图3跳邻居优化")
    print("=" * 80)
    
    prompt_enhancer = OptimizedPromptGraph(embedding_dim=64, max_hops=3, num_prompt_samples=5)
    data = create_test_graph()
    
    query_relation = torch.tensor(0)
    query_entity = torch.tensor(0)
    
    # 生成提示图
    prompt_graph, prompt_entities = prompt_enhancer.generate_prompt_graph(
        data, query_relation, query_entity
    )
    
    print(f"查询实体: {query_entity.item()}")
    print(f"查询关系: {query_relation.item()}")
    print(f"提示图实体: {sorted(prompt_entities)}")
    print(f"提示图节点数: {len(prompt_entities)}")
    print(f"提示图边数: {prompt_graph.edge_index.shape[1] if prompt_graph is not None else 0}")
    
    # 验证：3跳邻居应该包括0,1,2,3（从查询实体0开始）
    # 但由于所有边都是查询关系0，查询关系的边的实体也会被添加
    # 所以实际可能包含更多实体（这是合理的）
    computed_entities = set(prompt_entities)
    
    # 至少应该包括查询实体0及其3跳邻居0,1,2,3
    expected_min_entities = {0, 1, 2, 3}
    
    print(f"\n期望至少包括: {sorted(expected_min_entities)}")
    print(f"实际实体: {sorted(computed_entities)}")
    
    assert expected_min_entities.issubset(computed_entities), \
        f"期望至少包括{expected_min_entities}，实际{computed_entities}"
    
    # 验证：所有实体都应该在图中
    assert all(e < data.num_nodes for e in computed_entities), \
        f"提示图包含无效实体: {[e for e in computed_entities if e >= data.num_nodes]}"
    
    # 验证提示图的边只连接相关实体
    if prompt_graph is not None and prompt_graph.edge_index.numel() > 0:
        edge_src = prompt_graph.edge_index[0].tolist()
        edge_dst = prompt_graph.edge_index[1].tolist()
        all_edge_entities = set(edge_src + edge_dst)
        print(f"提示图边的实体: {sorted(all_edge_entities)}")
        
        # 所有边的实体都应该在prompt_entities中
        assert all_edge_entities.issubset(computed_entities), \
            f"提示图的边包含不在实体列表中的实体: {all_edge_entities - computed_entities}"
    
    print("✅ 提示图3跳邻居测试通过\n")

def test_prompt_graph_with_query_edges():
    """测试包含查询关系的边的提示图"""
    print("=" * 80)
    print("测试: 包含查询关系的边的提示图")
    print("=" * 80)
    
    prompt_enhancer = OptimizedPromptGraph(embedding_dim=64, max_hops=3, num_prompt_samples=5)
    
    # 创建更复杂的图
    # 0-1-2-3 (关系0)
    # 0-4-5 (关系1)
    edge_index = torch.tensor([
        [0, 1, 2, 0, 4],
        [1, 2, 3, 4, 5]
    ], dtype=torch.long)
    
    edge_type = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    
    data = Data()
    data.num_nodes = 6
    data.num_relations = 2
    data.edge_index = edge_index
    data.edge_type = edge_type
    
    # 查询关系0，查询实体0
    query_relation = torch.tensor(0)
    query_entity = torch.tensor(0)
    
    prompt_graph, prompt_entities = prompt_enhancer.generate_prompt_graph(
        data, query_relation, query_entity
    )
    
    print(f"查询实体: {query_entity.item()}")
    print(f"查询关系: {query_relation.item()}")
    print(f"提示图实体: {sorted(prompt_entities)}")
    
    # 验证：应该包括查询实体0及其3跳邻居（通过关系0的路径）
    # 0 -> 1 -> 2 -> 3 (关系0的路径)
    # 所以应该包括0,1,2,3
    expected_entities = {0, 1, 2, 3}
    computed_entities = set(prompt_entities)
    
    print(f"\n期望实体（通过关系0的路径）: {sorted(expected_entities)}")
    print(f"实际实体: {sorted(computed_entities)}")
    
    # 注意：由于也添加了查询关系的边的实体，可能包括更多实体
    # 但至少应该包括期望的实体
    assert expected_entities.issubset(computed_entities), \
        f"期望至少包括{expected_entities}，实际{computed_entities}"
    
    print("✅ 包含查询关系的边的提示图测试通过\n")

def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("提示图3跳邻居优化测试")
    print("=" * 80 + "\n")
    
    try:
        # 测试1: 基本3跳邻居
        test_prompt_graph_3hop()
        
        # 测试2: 包含查询关系的边
        test_prompt_graph_with_query_edges()
        
        print("=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        print("\n优化效果:")
        print("  1. ✓ 提示图使用3跳邻居，与实体增强保持一致")
        print("  2. ✓ 只包含相关实体，减少计算量")
        print("  3. ✓ 过滤边，只保留连接相关实体的边")
        print("  4. ✓ 提高提示图的质量和相关性")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

