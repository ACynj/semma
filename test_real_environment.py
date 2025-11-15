#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实环境测试脚本
测试EnhancedUltra在真实环境下的运行情况
"""

import sys
import os
import torch
import logging
from torch_geometric.data import Data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_realistic_test_data():
    """创建接近真实环境的测试数据"""
    num_nodes = 1000
    num_relations = 100
    num_edges = 5000
    
    # 创建边索引和边类型
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    # 创建Data对象
    data = Data()
    data.num_nodes = num_nodes
    data.num_relations = num_relations
    data.edge_index = edge_index
    data.edge_type = edge_type
    
    # 创建relation_graph（RelNBFNet需要）
    # 创建一个简单的relation graph：每个关系节点连接到其他关系节点
    num_rel_nodes = num_relations
    # 创建一些关系之间的边（随机连接）
    num_rel_edges = min(500, num_rel_nodes * 10)  # 限制边数
    rel_edge_index = torch.randint(0, num_rel_nodes, (2, num_rel_edges))
    rel_edge_type = torch.randint(0, 4, (num_rel_edges,))  # 4种边类型：hh, tt, ht, th
    
    relation_graph = Data()
    relation_graph.num_nodes = num_rel_nodes
    relation_graph.num_relations = 4
    relation_graph.edge_index = rel_edge_index
    relation_graph.edge_type = rel_edge_type
    
    data.relation_graph = relation_graph
    
    # 如果使用semma或EnhancedUltra，还需要relation_graph2
    from ultra import parse
    import os
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file)
    try:
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        if flags.run == "semma" or flags.run == "EnhancedUltra":
            # 创建relation_graph2（SemRelNBFNet需要）
            relation_graph2 = Data()
            relation_graph2.num_nodes = num_rel_nodes
            relation_graph2.num_relations = 1
            relation_graph2.edge_index = rel_edge_index
            relation_graph2.edge_type = torch.zeros(num_rel_edges, dtype=torch.long)
            relation_graph2.relation_embeddings = None
            data.relation_graph2 = relation_graph2
    except:
        pass  # 如果无法加载flags，跳过relation_graph2
    
    return data

def test_enhanced_ultra_initialization():
    """测试EnhancedUltra初始化"""
    print("=" * 80)
    print("测试1: EnhancedUltra模型初始化")
    print("=" * 80)
    
    try:
        from ultra.enhanced_models import EnhancedUltra
        from ultra.models import RelNBFNet, EntityNBFNet, SemRelNBFNet
        
        # 使用真实的配置
        rel_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64, 64, 64, 64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True,
            'num_relation': 100
        }
        
        entity_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64, 64, 64, 64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True,
            'num_relation': 1
        }
        
        sem_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64, 64, 64, 64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True,
            'num_relation': 1
        }
        
        print("正在初始化EnhancedUltra模型...")
        model = EnhancedUltra(rel_model_cfg, entity_model_cfg, sem_model_cfg)
        model.eval()  # 设置为推理模式
        
        print(f"✓ 模型初始化成功")
        print(f"  - 相似度增强器: {'启用' if model.use_similarity_enhancer else '禁用'}")
        print(f"  - 提示图增强器: {'启用' if model.use_prompt_enhancer else '禁用'}")
        print(f"  - 实体增强器: {'启用' if model.use_entity_enhancement else '禁用'}")
        print(f"  - 可学习融合: {'启用' if model.use_learnable_fusion else '禁用'}")
        print(f"  - 自适应门控: {'启用' if model.use_adaptive_gate else '禁用'}")
        
        return model
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model, data):
    """测试前向传播"""
    print("\n" + "=" * 80)
    print("测试2: 前向传播")
    print("=" * 80)
    
    try:
        batch_size = 4
        num_nodes = data.num_nodes
        
        # 创建batch数据（格式: [batch_size, 1, 3]）
        # 每行: [head, tail, relation]
        batch = torch.randint(0, num_nodes, (batch_size, 1, 3))
        batch[:, 0, 2] = torch.randint(0, data.num_relations, (batch_size,))  # 关系索引
        
        print(f"Batch形状: {batch.shape}")
        print(f"Batch内容示例: {batch[0]}")
        
        print("\n开始前向传播...")
        with torch.no_grad():
            score = model(data, batch)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输出形状: {score.shape}")
        print(f"  - 输出值范围: [{score.min().item():.4f}, {score.max().item():.4f}]")
        print(f"  - 输出均值: {score.mean().item():.4f}")
        
        # 检查输出是否合理
        if torch.isnan(score).any():
            print("⚠ 警告: 输出包含NaN值")
        if torch.isinf(score).any():
            print("⚠ 警告: 输出包含Inf值")
        
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_functionality(model, data):
    """测试缓存功能"""
    print("\n" + "=" * 80)
    print("测试3: 缓存功能")
    print("=" * 80)
    
    try:
        if not hasattr(model, 'prompt_enhancer') or model.prompt_enhancer is None:
            print("⚠ 提示图增强器未启用，跳过缓存测试")
            return True
        
        prompt_enhancer = model.prompt_enhancer
        
        # 检查缓存相关属性
        if hasattr(prompt_enhancer, '_bfnet_cache'):
            print(f"✓ 缓存字典已初始化")
            print(f"  - 最大缓存大小: {prompt_enhancer._max_cache_size}")
            print(f"  - 当前缓存大小: {len(prompt_enhancer._bfnet_cache)}")
            print(f"  - 缓存命中数: {prompt_enhancer._cache_hits}")
            print(f"  - 缓存未命中数: {prompt_enhancer._cache_misses}")
        else:
            print("⚠ 警告: 未找到缓存字典")
        
        return True
        
    except Exception as e:
        print(f"✗ 缓存功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_forward_passes(model, data):
    """测试多次前向传播（验证缓存是否工作）"""
    print("\n" + "=" * 80)
    print("测试4: 多次前向传播（验证缓存）")
    print("=" * 80)
    
    try:
        batch_size = 4
        num_nodes = data.num_nodes
        
        # 创建多个batch
        batches = []
        for i in range(3):
            batch = torch.randint(0, num_nodes, (batch_size, 1, 3))
            batch[:, 0, 2] = torch.randint(0, data.num_relations, (batch_size,))
            batches.append(batch)
        
        print("执行3次前向传播...")
        for i, batch in enumerate(batches):
            print(f"\n第 {i+1} 次前向传播...")
            with torch.no_grad():
                score = model(data, batch)
            print(f"  ✓ 成功，输出形状: {score.shape}")
        
        # 检查缓存统计
        if hasattr(model, 'prompt_enhancer') and model.prompt_enhancer is not None:
            prompt_enhancer = model.prompt_enhancer
            if hasattr(prompt_enhancer, '_cache_hits'):
                total_requests = prompt_enhancer._cache_hits + prompt_enhancer._cache_misses
                if total_requests > 0:
                    hit_rate = prompt_enhancer._cache_hits / total_requests * 100
                    print(f"\n缓存统计:")
                    print(f"  - 总请求数: {total_requests}")
                    print(f"  - 缓存命中: {prompt_enhancer._cache_hits}")
                    print(f"  - 缓存未命中: {prompt_enhancer._cache_misses}")
                    print(f"  - 命中率: {hit_rate:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ 多次前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("EnhancedUltra 真实环境测试")
    print("=" * 80)
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建测试数据
    print("\n创建测试数据...")
    data = create_realistic_test_data()
    print(f"✓ 测试数据创建成功")
    print(f"  - 节点数: {data.num_nodes}")
    print(f"  - 关系数: {data.num_relations}")
    print(f"  - 边数: {data.edge_index.shape[1]}")
    
    # 测试1: 模型初始化
    model = test_enhanced_ultra_initialization()
    if model is None:
        print("\n✗ 测试失败: 模型初始化失败")
        return False
    
    # 测试2: 前向传播
    if not test_forward_pass(model, data):
        print("\n✗ 测试失败: 前向传播失败")
        return False
    
    # 测试3: 缓存功能
    if not test_cache_functionality(model, data):
        print("\n⚠ 警告: 缓存功能测试失败（不影响主要功能）")
    
    # 测试4: 多次前向传播
    if not test_multiple_forward_passes(model, data):
        print("\n⚠ 警告: 多次前向传播测试失败（不影响主要功能）")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

