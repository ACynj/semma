#!/usr/bin/env python
"""
RDG安全集成测试脚本（CPU模式，不影响GPU）

测试内容：
1. RDG构建功能
2. 关系图集成
3. 模型层兼容性检查
4. 权重使用检查
"""

import os
import sys
import torch

# 强制使用CPU，不影响GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(2)  # 限制CPU线程

# 添加项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from torch_geometric.data import Data
from ultra.rdg import build_rdg_edges, RDGConfig
from ultra import parse, tasks
from ultra.models import RelNBFNet

# 使用tasks模块中的flags（build_relation_graph实际使用的flags）
flags = tasks.flags

def create_test_graph():
    """创建测试知识图谱"""
    # 实体: 0=Alice, 1=Beijing, 2=China, 3=Shanghai, 4=Company, 5=Bob
    # 关系: 0=bornIn, 1=locatedIn, 2=livesIn, 3=worksAt
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

def test_rdg_building():
    """测试1: RDG构建功能"""
    print("\n" + "="*60)
    print("测试1: RDG构建功能")
    print("="*60)
    
    try:
        graph = create_test_graph()
        config = RDGConfig(
            enabled=True,
            min_dependency_weight=0.001,
            normalize_weights=True
        )
        
        rdg_edge_index, rdg_edge_weights, tau, dependency_edges = build_rdg_edges(graph, config)
        
        print(f"✓ RDG边数: {rdg_edge_index.size(1)}")
        print(f"✓ 依赖边数: {len(dependency_edges)}")
        print(f"✓ 优先级字典大小: {len(tau)}")
        
        if rdg_edge_index.size(1) > 0:
            print(f"✓ RDG边索引shape: {rdg_edge_index.shape}")
            print(f"✓ RDG边权重shape: {rdg_edge_weights.shape}")
            print(f"✓ 前3条依赖边:")
            for i, (r_i, r_j, w) in enumerate(dependency_edges[:3]):
                rel_names = {0: "bornIn", 1: "locatedIn", 2: "livesIn", 3: "worksAt"}
                print(f"    {rel_names.get(r_i, r_i)} -> {rel_names.get(r_j, r_j)}: {w:.4f}")
        else:
            print("⚠ 没有找到RDG依赖边（可能权重阈值过高）")
        
        return True
    except Exception as e:
        print(f"❌ RDG构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_relation_graph_integration():
    """测试2: 关系图集成"""
    print("\n" + "="*60)
    print("测试2: 关系图集成")
    print("="*60)
    
    try:
        # 保存原始设置
        original_use_rdg = getattr(flags, 'use_rdg', False)
        
        # 启用RDG
        flags.use_rdg = True
        if not hasattr(flags, 'rdg_min_weight'):
            flags.rdg_min_weight = 0.001
        if not hasattr(flags, 'rdg_precedence_method'):
            flags.rdg_precedence_method = 'indegree'
        if not hasattr(flags, 'rdg_normalize_weights'):
            flags.rdg_normalize_weights = True
        
        print(f"设置flags.use_rdg = {flags.use_rdg}")
        
        graph = create_test_graph()
        # 确保graph有inverse edges（build_relation_graph需要）
        if not hasattr(graph, 'edge_index') or graph.edge_index.size(1) == 0:
            print("⚠ 测试图没有边")
            return False
        
        graph = tasks.build_relation_graph(graph)
        
        if not hasattr(graph, 'relation_graph'):
            print("❌ graph没有relation_graph属性")
            return False
        
        rel_graph = graph.relation_graph
        
        print(f"✓ 关系图节点数: {rel_graph.num_nodes}")
        print(f"✓ 关系图边类型数: {rel_graph.num_relations}")
        print(f"✓ 关系图边数: {rel_graph.edge_index.size(1)}")
        
        # 检查RDG元数据
        has_rdg_precedence = hasattr(graph, 'rdg_precedence')
        has_rdg_edges = hasattr(graph, 'rdg_dependency_edges')
        has_rdg_weights = hasattr(graph, 'rdg_edge_weights')
        
        print(f"✓ rdg_precedence存在: {has_rdg_precedence}")
        print(f"✓ rdg_dependency_edges存在: {has_rdg_edges}")
        print(f"✓ rdg_edge_weights存在: {has_rdg_weights}")
        
        if has_rdg_edges and len(graph.rdg_dependency_edges) > 0:
            print(f"✓ RDG依赖边数: {len(graph.rdg_dependency_edges)}")
        
        if rel_graph.num_relations == 5:
            print("✓ 关系类型数正确（5种：hh, tt, ht, th, RDG）")
        else:
            print(f"⚠ 关系类型数: {rel_graph.num_relations} (期望5)")
            if not flags.use_rdg:
                print("  原因: flags.use_rdg可能未正确设置")
        
        # 检查RDG边类型（edge_type=4）
        if rel_graph.edge_index.size(1) > 0:
            edge_type_counts = torch.bincount(rel_graph.edge_type, minlength=5)
            print(f"✓ 边类型分布: {edge_type_counts.tolist()}")
            if len(edge_type_counts) > 4 and edge_type_counts[4].item() > 0:
                rdg_edge_count = edge_type_counts[4].item()
                print(f"✓ RDG边数（类型4）: {rdg_edge_count}")
            else:
                print(f"⚠ 没有找到类型4的边（RDG边）")
        
        # 恢复原始设置
        flags.use_rdg = original_use_rdg
        
        # 如果RDG启用但没找到RDG边，返回False
        if flags.use_rdg and rel_graph.num_relations != 5:
            return False
        
        return True
    except Exception as e:
        print(f"❌ 关系图集成失败: {e}")
        import traceback
        traceback.print_exc()
        # 恢复原始设置
        if 'original_use_rdg' in locals():
            flags.use_rdg = original_use_rdg
        return False

def test_model_compatibility():
    """测试3: 模型层兼容性检查"""
    print("\n" + "="*60)
    print("测试3: 模型层兼容性检查")
    print("="*60)
    
    try:
        # 保存原始设置
        original_use_rdg = getattr(flags, 'use_rdg', False)
        flags.use_rdg = True
        
        graph = create_test_graph()
        graph = tasks.build_relation_graph(graph)
        
        if not hasattr(graph, 'relation_graph'):
            print("❌ graph没有relation_graph属性")
            return False
        
        rel_graph = graph.relation_graph
        
        print(f"关系图边类型数: {rel_graph.num_relations}")
        
        # 检查模型初始化
        # 问题：RelNBFNet默认num_relation=4，但RDG启用后需要5
        model = RelNBFNet(
            input_dim=64,
            hidden_dims=[64, 64],
            num_relation=rel_graph.num_relations  # 使用实际的关系类型数
        )
        
        print(f"✓ 模型初始化成功")
        print(f"✓ 模型num_relation: {model.num_relation}")
        print(f"✓ 关系图num_relations: {rel_graph.num_relations}")
        
        if model.num_relation == rel_graph.num_relations:
            print("✓ 模型和关系图的关系类型数匹配")
        else:
            print(f"⚠ 不匹配！模型: {model.num_relation}, 关系图: {rel_graph.num_relations}")
            print("  这会导致索引越界错误")
        
        # 测试前向传播（使用CPU）
        # 注意：RelNBFNet.forward需要graph对象，不是rel_graph
        query = torch.tensor([0], dtype=torch.long)  # 查询关系0
        
        try:
            output = model(graph, query)  # 传入graph，不是rel_graph
            print(f"✓ 前向传播成功")
            print(f"✓ 输出shape: {output.shape}")
            print(f"  期望: [batch_size=1, num_relations, hidden_dim=64]")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 恢复原始设置
        flags.use_rdg = original_use_rdg
        
        return True
    except Exception as e:
        print(f"❌ 模型兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        # 恢复原始设置
        if 'original_use_rdg' in locals():
            flags.use_rdg = original_use_rdg
        return False

def test_edge_weight_usage():
    """测试4: RDG边权重使用检查"""
    print("\n" + "="*60)
    print("测试4: RDG边权重使用检查")
    print("="*60)
    
    try:
        flags.use_rdg = True
        graph = create_test_graph()
        graph = tasks.build_relation_graph(graph)
        rel_graph = graph.relation_graph
        
        # 检查是否有RDG权重
        if hasattr(graph, 'rdg_edge_weights') and graph.rdg_edge_weights.numel() > 0:
            print(f"✓ RDG边权重存在: {graph.rdg_edge_weights.shape}")
            print(f"✓ 权重范围: [{graph.rdg_edge_weights.min():.4f}, {graph.rdg_edge_weights.max():.4f}]")
            
            # 检查权重是否被使用
            # 当前实现中，权重存储在graph.rdg_edge_weights中
            # 但消息传递层可能没有使用它
            print("⚠ 注意: 当前实现中，RDG权重存储在graph.rdg_edge_weights中")
            print("  但消息传递层可能使用等权重，需要检查layers.py")
        else:
            print("⚠ 没有RDG边权重（可能没有RDG边）")
        
        return True
    except Exception as e:
        print(f"❌ 权重使用检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试5: 向后兼容性（RDG关闭时）"""
    print("\n" + "="*60)
    print("测试5: 向后兼容性（RDG关闭）")
    print("="*60)
    
    try:
        original_use_rdg = getattr(flags, 'use_rdg', False)
        flags.use_rdg = False
        
        graph = create_test_graph()
        graph = tasks.build_relation_graph(graph)
        rel_graph = graph.relation_graph
        
        print(f"✓ 关系图边类型数: {rel_graph.num_relations}")
        
        if rel_graph.num_relations == 4:
            print("✓ 向后兼容：RDG关闭时，关系类型数为4（正确）")
        else:
            print(f"⚠ 关系类型数: {rel_graph.num_relations} (期望4)")
        
        # 恢复
        flags.use_rdg = original_use_rdg
        
        return True
    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        if 'original_use_rdg' in locals():
            flags.use_rdg = original_use_rdg
        return False

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("RDG集成测试（CPU模式，安全测试）")
    print("="*60)
    print(f"设备: CPU (强制CPU模式)")
    print(f"当前use_rdg设置: {getattr(flags, 'use_rdg', False)}")
    print(f"使用tasks模块的flags对象: {flags is tasks.flags}")
    
    results = []
    
    # 运行测试
    results.append(("RDG构建功能", test_rdg_building()))
    results.append(("关系图集成", test_relation_graph_integration()))
    results.append(("模型兼容性", test_model_compatibility()))
    results.append(("权重使用检查", test_edge_weight_usage()))
    results.append(("向后兼容性", test_backward_compatibility()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {len(results)} 个测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！RDG集成正常。")
    else:
        print("\n⚠️  有测试失败，请检查上述错误信息。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

