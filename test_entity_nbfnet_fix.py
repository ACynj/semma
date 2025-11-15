#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试EntityNBFNet计算修复
"""

import sys
import os
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_entity_nbfnet_bellmanford():
    """测试EntityNBFNet.bellmanford的正确调用方式"""
    print("=" * 80)
    print("测试EntityNBFNet.bellmanford")
    print("=" * 80)
    
    from ultra.models import EntityNBFNet
    
    # 创建EntityNBFNet
    embedding_dim = 64
    entity_model = EntityNBFNet(
        input_dim=embedding_dim,
        hidden_dims=[64, 64, 64, 64, 64, 64],
        num_relation=1
    )
    entity_model.eval()
    
    # 创建测试数据
    num_nodes = 100
    num_relations = 50
    num_edges = 200
    
    data = Data()
    data.num_nodes = num_nodes
    data.num_relations = num_relations
    data.edge_index = torch.randint(0, num_nodes, (2, num_edges))
    # 注意：edge_type的最大值应该 <= num_relations - 1
    # 但如果有逆关系，edge_type可能包含 [0, num_relations-1] 和 [num_relations, 2*num_relations-1]
    # 所以需要确保relation_representations包含足够的关系数
    data.edge_type = torch.randint(0, num_relations, (num_edges,))
    
    # 设置query
    batch_size = 5
    num_rels = num_relations
    entity_model.query = torch.randn(batch_size, num_rels, embedding_dim)
    
    # 设置layers的relation
    relation_representations = torch.randn(num_rels, embedding_dim)
    for layer in entity_model.layers:
        if hasattr(layer, 'relation'):
            layer.relation = relation_representations
    
    # 测试bellmanford
    h_indices = torch.randint(0, num_nodes, (batch_size,))
    r_indices = torch.randint(0, num_rels, (batch_size,))
    
    print(f"数据形状:")
    print(f"  - edge_index: {data.edge_index.shape}")
    print(f"  - edge_type: {data.edge_type.shape}")
    print(f"  - num_nodes: {data.num_nodes}")
    print(f"  - num_relations: {data.num_relations}")
    print(f"  - query: {entity_model.query.shape}")
    print(f"  - h_indices: {h_indices.shape}")
    print(f"  - r_indices: {r_indices.shape}")
    
    try:
        with torch.no_grad():
            result = entity_model.bellmanford(data, h_indices, r_indices)
        print(f"\n✓ bellmanford成功")
        print(f"  - node_feature形状: {result['node_feature'].shape}")
        return True
    except Exception as e:
        print(f"\n✗ bellmanford失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_entity_nbfnet_bellmanford()
    sys.exit(0 if success else 1)

