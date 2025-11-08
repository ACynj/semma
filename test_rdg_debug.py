#!/usr/bin/env python
"""调试RDG集成问题"""

import os
import sys
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(2)

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from torch_geometric.data import Data
from ultra.rdg import build_rdg_edges, RDGConfig
from ultra import parse, tasks

mydir = os.getcwd()
flags = parse.load_flags(os.path.join(mydir, "flags.yaml"))

def create_test_graph():
    edge_index = torch.tensor([
        [0, 1, 0, 3, 0, 4, 4, 1, 5],
        [1, 2, 3, 2, 4, 1, 1, 2, 3]
    ], dtype=torch.long)
    
    edge_type = torch.tensor([0, 1, 2, 1, 3, 1, 1, 0, 1], dtype=torch.long)
    
    graph = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=6,
        num_relations=4
    )
    return graph

print("="*60)
print("调试RDG集成")
print("="*60)

# 测试1: 直接构建RDG
print("\n[测试1] 直接构建RDG")
graph1 = create_test_graph()
config = RDGConfig(enabled=True, min_dependency_weight=0.001)
rdg_edge_index, rdg_edge_weights, tau, deps = build_rdg_edges(graph1, config)
print(f"✓ RDG边数: {rdg_edge_index.size(1)}")
print(f"✓ 依赖边数: {len(deps)}")

# 测试2: 通过build_relation_graph集成
print("\n[测试2] 通过build_relation_graph集成")
print(f"原始flags.use_rdg = {getattr(flags, 'use_rdg', None)}")

# 启用RDG
flags.use_rdg = True
if not hasattr(flags, 'rdg_min_weight'):
    flags.rdg_min_weight = 0.001
if not hasattr(flags, 'rdg_precedence_method'):
    flags.rdg_precedence_method = 'indegree'
if not hasattr(flags, 'rdg_normalize_weights'):
    flags.rdg_normalize_weights = True

print(f"设置后flags.use_rdg = {flags.use_rdg}")
print(f"hasattr(flags, 'use_rdg') = {hasattr(flags, 'use_rdg')}")
print(f"getattr(flags, 'use_rdg', False) = {getattr(flags, 'use_rdg', False)}")

# 检查build_relation_graph中的检查逻辑
rdg_enabled_check = hasattr(flags, 'use_rdg') and getattr(flags, 'use_rdg', False)
print(f"\nbuild_relation_graph中的检查:")
print(f"  hasattr(flags, 'use_rdg') = {hasattr(flags, 'use_rdg')}")
print(f"  getattr(flags, 'use_rdg', False) = {getattr(flags, 'use_rdg', False)}")
print(f"  rdg_enabled = {rdg_enabled_check}")

graph2 = create_test_graph()
print("\n调用build_relation_graph...")
graph2 = tasks.build_relation_graph(graph2)

if hasattr(graph2, 'relation_graph'):
    rel_graph = graph2.relation_graph
    print(f"\n结果:")
    print(f"  关系图边类型数: {rel_graph.num_relations}")
    print(f"  关系图边数: {rel_graph.edge_index.size(1)}")
    print(f"  rdg_precedence存在: {hasattr(graph2, 'rdg_precedence')}")
    print(f"  rdg_dependency_edges存在: {hasattr(graph2, 'rdg_dependency_edges')}")
    
    if hasattr(graph2, 'rdg_dependency_edges'):
        print(f"  rdg_dependency_edges数量: {len(graph2.rdg_dependency_edges)}")
    
    # 检查边类型
    if rel_graph.edge_index.size(1) > 0:
        edge_type_counts = torch.bincount(rel_graph.edge_type, minlength=5)
        print(f"  边类型分布: {edge_type_counts.tolist()}")
        if edge_type_counts[4].item() > 0:
            print(f"  ✓ 找到{edge_type_counts[4].item()}条RDG边（类型4）")
        else:
            print(f"  ⚠ 没有找到RDG边（类型4）")
else:
    print("❌ graph2没有relation_graph属性")

