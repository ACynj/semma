"""
测试语义增强的 RDG 构建功能

这个脚本测试语义增强的 RDG 模块，包括：
1. 语义增强函数的基本功能
2. 不同过滤模式（filter, weight, both）
3. 与 SEMMA 语义信息的集成
4. 完整流程测试（CPU 模式）
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultra.rdg import (
    extract_relation_dependencies,
    enhance_rdg_with_semantics,
    build_semantic_enhanced_rdg_edges,
    build_rdg_edges,
    RDGConfig
)


def create_test_graph():
    """创建一个测试用的知识图谱"""
    # 创建一个简单的知识图谱
    # 实体: 0, 1, 2, 3, 4
    # 关系: 0 (bornIn), 1 (locatedIn), 2 (livesIn), 3 (worksAt)
    
    # 边: (Alice, bornIn, Beijing), (Beijing, locatedIn, China), etc.
    edge_index = torch.tensor([
        [0, 1, 0, 3, 0, 4, 4, 1, 5],  # head entities
        [1, 2, 3, 2, 4, 1, 1, 2, 3]   # tail entities
    ], dtype=torch.long)
    
    edge_type = torch.tensor([0, 1, 2, 1, 3, 1, 1, 0, 1], dtype=torch.long)
    
    graph = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=6,
        num_relations=4
    )
    
    return graph


def create_semantic_similarity_matrix(num_relations, device='cpu'):
    """创建测试用的语义相似性矩阵"""
    # 创建一个模拟的语义相似性矩阵
    # 关系 0 和 1 相似（bornIn 和 locatedIn 都涉及位置）
    # 关系 2 和 3 相似（livesIn 和 worksAt 都涉及人的活动）
    
    similarity_matrix = torch.zeros((num_relations, num_relations), device=device)
    
    # 对角线设为 1.0（自己和自己完全相似）
    for i in range(num_relations):
        similarity_matrix[i, i] = 1.0
    
    # 设置一些相似关系
    similarity_matrix[0, 1] = 0.8  # bornIn 和 locatedIn 相似
    similarity_matrix[1, 0] = 0.8
    similarity_matrix[2, 3] = 0.7  # livesIn 和 worksAt 相似
    similarity_matrix[3, 2] = 0.7
    similarity_matrix[0, 2] = 0.6  # bornIn 和 livesIn 有一定相似性
    similarity_matrix[2, 0] = 0.6
    
    # 其他关系相似性较低
    similarity_matrix[1, 3] = 0.3
    similarity_matrix[3, 1] = 0.3
    similarity_matrix[0, 3] = 0.4
    similarity_matrix[3, 0] = 0.4
    similarity_matrix[1, 2] = 0.5
    similarity_matrix[2, 1] = 0.5
    
    return similarity_matrix


def test_enhance_rdg_with_semantics():
    """测试语义增强函数"""
    print("\n" + "="*60)
    print("测试 1: 语义增强函数 (enhance_rdg_with_semantics)")
    print("="*60)
    
    # 创建测试数据
    dependency_edges = [
        (0, 1, 0.3),  # bornIn -> locatedIn
        (2, 1, 0.2),  # livesIn -> locatedIn
        (3, 1, 0.4),  # worksAt -> locatedIn
        (1, 0, 0.1),  # locatedIn -> bornIn
    ]
    
    num_relations = 4
    semantic_sim_matrix = create_semantic_similarity_matrix(num_relations)
    
    # 测试 filter 模式
    print("\n[测试 1.1] Filter 模式（只保留相似度 >= 阈值的边）")
    config_filter = RDGConfig(
        use_semantic_enhancement=True,
        semantic_similarity_threshold=0.6,  # 归一化后阈值
        semantic_filter_mode='filter'
    )
    
    enhanced_filter = enhance_rdg_with_semantics(
        dependency_edges, semantic_sim_matrix, config_filter
    )
    
    print(f"  原始边数: {len(dependency_edges)}")
    print(f"  增强后边数: {len(enhanced_filter)}")
    print(f"  增强后的边: {enhanced_filter}")
    
    # 验证：相似度高的边应该被保留
    assert len(enhanced_filter) <= len(dependency_edges), "Filter 模式不应该增加边数"
    print("  ✓ Filter 模式测试通过")
    
    # 测试 weight 模式
    print("\n[测试 1.2] Weight 模式（根据相似度调整权重）")
    config_weight = RDGConfig(
        use_semantic_enhancement=True,
        semantic_weight_alpha=0.5,
        semantic_filter_mode='weight'
    )
    
    enhanced_weight = enhance_rdg_with_semantics(
        dependency_edges, semantic_sim_matrix, config_weight
    )
    
    print(f"  原始边数: {len(dependency_edges)}")
    print(f"  增强后边数: {len(enhanced_weight)}")
    print(f"  增强后的边: {enhanced_weight}")
    
    # 验证：weight 模式应该保留所有边，但权重可能改变
    assert len(enhanced_weight) == len(dependency_edges), "Weight 模式应该保留所有边"
    print("  ✓ Weight 模式测试通过")
    
    # 测试 both 模式
    print("\n[测试 1.3] Both 模式（过滤 + 加权）")
    config_both = RDGConfig(
        use_semantic_enhancement=True,
        semantic_similarity_threshold=0.6,
        semantic_weight_alpha=0.5,
        semantic_filter_mode='both'
    )
    
    enhanced_both = enhance_rdg_with_semantics(
        dependency_edges, semantic_sim_matrix, config_both
    )
    
    print(f"  原始边数: {len(dependency_edges)}")
    print(f"  增强后边数: {len(enhanced_both)}")
    print(f"  增强后的边: {enhanced_both}")
    
    assert len(enhanced_both) <= len(dependency_edges), "Both 模式不应该增加边数"
    print("  ✓ Both 模式测试通过")
    
    # 测试禁用语义增强
    print("\n[测试 1.4] 禁用语义增强")
    config_disabled = RDGConfig(use_semantic_enhancement=False)
    enhanced_disabled = enhance_rdg_with_semantics(
        dependency_edges, semantic_sim_matrix, config_disabled
    )
    
    assert enhanced_disabled == dependency_edges, "禁用时应该返回原始边"
    print("  ✓ 禁用语义增强测试通过")
    
    print("\n✓ 所有语义增强函数测试通过！")


def test_build_semantic_enhanced_rdg():
    """测试语义增强的 RDG 构建"""
    print("\n" + "="*60)
    print("测试 2: 语义增强的 RDG 构建 (build_semantic_enhanced_rdg_edges)")
    print("="*60)
    
    # 创建测试图
    graph = create_test_graph()
    num_relations = graph.num_relations
    
    # 创建语义相似性矩阵
    semantic_sim_matrix = create_semantic_similarity_matrix(num_relations)
    
    # 测试不使用语义增强
    print("\n[测试 2.1] 不使用语义增强（结构 RDG）")
    config_no_sem = RDGConfig(
        enabled=True,
        use_semantic_enhancement=False,
        min_dependency_weight=0.001
    )
    
    edge_index_no_sem, edge_weights_no_sem, tau_no_sem, deps_no_sem = build_semantic_enhanced_rdg_edges(
        graph, semantic_similarity_matrix=None, config=config_no_sem
    )
    
    print(f"  RDG 边数: {edge_index_no_sem.size(1)}")
    print(f"  优先级值: {tau_no_sem}")
    print("  ✓ 不使用语义增强测试通过")
    
    # 测试使用语义增强
    print("\n[测试 2.2] 使用语义增强（filter 模式）")
    config_sem_filter = RDGConfig(
        enabled=True,
        use_semantic_enhancement=True,
        semantic_similarity_threshold=0.6,
        semantic_filter_mode='filter',
        min_dependency_weight=0.001
    )
    
    edge_index_sem, edge_weights_sem, tau_sem, deps_sem = build_semantic_enhanced_rdg_edges(
        graph, semantic_similarity_matrix=semantic_sim_matrix, config=config_sem_filter
    )
    
    print(f"  结构 RDG 边数: {edge_index_no_sem.size(1)}")
    print(f"  语义增强 RDG 边数: {edge_index_sem.size(1)}")
    print(f"  优先级值: {tau_sem}")
    
    # 验证：语义增强后边数应该 <= 结构 RDG 边数（filter 模式）
    assert edge_index_sem.size(1) <= edge_index_no_sem.size(1), \
        "Filter 模式下语义增强后边数不应该增加"
    print("  ✓ 语义增强（filter 模式）测试通过")
    
    # 测试使用语义增强（weight 模式）
    print("\n[测试 2.3] 使用语义增强（weight 模式）")
    config_sem_weight = RDGConfig(
        enabled=True,
        use_semantic_enhancement=True,
        semantic_weight_alpha=0.5,
        semantic_filter_mode='weight',
        min_dependency_weight=0.001
    )
    
    edge_index_sem_weight, edge_weights_sem_weight, tau_sem_weight, deps_sem_weight = \
        build_semantic_enhanced_rdg_edges(
            graph, semantic_similarity_matrix=semantic_sim_matrix, config=config_sem_weight
        )
    
    print(f"  结构 RDG 边数: {edge_index_no_sem.size(1)}")
    print(f"  语义增强 RDG 边数: {edge_index_sem_weight.size(1)}")
    
    # 验证：weight 模式下边数应该相同
    assert edge_index_sem_weight.size(1) == edge_index_no_sem.size(1), \
        "Weight 模式下边数应该相同"
    print("  ✓ 语义增强（weight 模式）测试通过")
    
    print("\n✓ 所有语义增强 RDG 构建测试通过！")


def test_integration_with_relation_graph2():
    """测试与 relation_graph2 的集成"""
    print("\n" + "="*60)
    print("测试 3: 与 relation_graph2 的集成")
    print("="*60)
    
    # 创建测试图
    graph = create_test_graph()
    num_relations = graph.num_relations
    
    # 创建模拟的 relation_graph2（包含语义信息）
    embeddings = torch.randn(num_relations, 64)  # 64 维嵌入
    relation_similarity_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
    )
    
    from torch_geometric.data import Data as PyGData
    graph.relation_graph2 = PyGData(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        num_nodes=num_relations,
        num_relations=1,
        relation_embeddings=embeddings,
        relation_similarity_matrix=relation_similarity_matrix
    )
    
    # 测试从 relation_graph2 自动获取语义信息
    print("\n[测试 3.1] 从 relation_graph2 自动获取语义相似性矩阵")
    config_auto = RDGConfig(
        enabled=True,
        use_semantic_enhancement=True,
        semantic_filter_mode='filter',
        semantic_similarity_threshold=0.5,
        min_dependency_weight=0.001
    )
    
    edge_index_auto, edge_weights_auto, tau_auto, deps_auto = \
        build_semantic_enhanced_rdg_edges(
            graph, semantic_similarity_matrix=None, config=config_auto
        )
    
    print(f"  自动获取语义信息后 RDG 边数: {edge_index_auto.size(1)}")
    print("  ✓ 自动获取语义信息测试通过")
    
    # 测试从 relation_embeddings 计算相似性矩阵
    print("\n[测试 3.2] 从 relation_embeddings 计算相似性矩阵")
    graph.relation_graph2.relation_similarity_matrix = None  # 移除相似性矩阵
    graph.relation_graph2.relation_embeddings = embeddings  # 保留嵌入
    
    edge_index_from_emb, edge_weights_from_emb, tau_from_emb, deps_from_emb = \
        build_semantic_enhanced_rdg_edges(
            graph, semantic_similarity_matrix=None, config=config_auto
        )
    
    print(f"  从嵌入计算相似性后 RDG 边数: {edge_index_from_emb.size(1)}")
    print("  ✓ 从嵌入计算相似性测试通过")
    
    print("\n✓ 所有集成测试通过！")


def test_comparison_with_original_rdg():
    """对比原始 RDG 和语义增强 RDG"""
    print("\n" + "="*60)
    print("测试 4: 原始 RDG vs 语义增强 RDG 对比")
    print("="*60)
    
    graph = create_test_graph()
    num_relations = graph.num_relations
    semantic_sim_matrix = create_semantic_similarity_matrix(num_relations)
    
    # 构建原始 RDG
    config_original = RDGConfig(enabled=True, min_dependency_weight=0.001)
    edge_index_orig, edge_weights_orig, tau_orig, deps_orig = build_rdg_edges(graph, config_original)
    
    # 构建语义增强 RDG（filter 模式）
    config_enhanced = RDGConfig(
        enabled=True,
        use_semantic_enhancement=True,
        semantic_similarity_threshold=0.6,
        semantic_filter_mode='filter',
        min_dependency_weight=0.001
    )
    edge_index_enh, edge_weights_enh, tau_enh, deps_enh = build_semantic_enhanced_rdg_edges(
        graph, semantic_similarity_matrix=semantic_sim_matrix, config=config_enhanced
    )
    
    print(f"\n原始 RDG:")
    print(f"  边数: {edge_index_orig.size(1)}")
    print(f"  依赖边: {deps_orig}")
    
    print(f"\n语义增强 RDG (filter 模式):")
    print(f"  边数: {edge_index_enh.size(1)}")
    print(f"  依赖边: {deps_enh}")
    
    # 验证：filter 模式下边数应该减少或相等
    assert edge_index_enh.size(1) <= edge_index_orig.size(1), \
        "Filter 模式下语义增强后边数不应该增加"
    
    print("\n✓ 对比测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("语义增强 RDG 模块测试套件")
    print("="*80)
    print("\n注意: 所有测试在 CPU 模式下运行")
    
    try:
        test_enhance_rdg_with_semantics()
        test_build_semantic_enhanced_rdg()
        test_integration_with_relation_graph2()
        test_comparison_with_original_rdg()
        
        print("\n" + "="*80)
        print("✓ 所有测试通过！语义增强 RDG 模块工作正常。")
        print("="*80)
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ 测试失败: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 确保使用 CPU
    torch.set_default_tensor_type('torch.FloatTensor')
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

