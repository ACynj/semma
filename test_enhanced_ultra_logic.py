#!/usr/bin/env python3
"""
EnhancedUltra逻辑正确性测试脚本
测试模型在真实场景下的逻辑正确性，包括：
1. 模型初始化
2. 前向传播维度匹配
3. 各模块接口正确性
4. 数据流验证
5. 边界情况处理
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_mock_data(num_nodes=100, num_relations=10, num_edges=200, batch_size=4):
    """创建模拟的图数据"""
    # 创建边索引
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_type = torch.randint(0, num_relations, (num_edges,), dtype=torch.long)
    
    # 创建batch数据 (batch_size, 1+num_negs, 3)
    # 格式: [head, tail, relation]
    # 注意：EntityNBFNet要求batch中所有样本的head索引相同
    batch = torch.zeros(batch_size, 2, 3, dtype=torch.long)  # 1 positive + 1 negative
    head_entity = torch.randint(0, num_nodes, (1,)).item()  # 所有样本使用相同的head
    for i in range(batch_size):
        batch[i, 0, 0] = head_entity  # head (所有样本相同)
        batch[i, 0, 1] = torch.randint(0, num_nodes, (1,))  # tail
        batch[i, 0, 2] = torch.randint(0, num_relations, (1,))  # relation
        # negative sample
        batch[i, 1, 0] = head_entity  # head (所有样本相同)
        batch[i, 1, 1] = torch.randint(0, num_nodes, (1,))
        batch[i, 1, 2] = batch[i, 0, 2]
    
    # 创建Data对象
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=num_relations
    )
    
    # 添加relation_graph（RelNBFNet需要）
    num_rel_nodes = num_relations
    num_rel_edges = min(200, num_rel_nodes * 10)
    rel_edge_index = torch.randint(0, num_rel_nodes, (2, num_rel_edges), dtype=torch.long)
    rel_edge_type = torch.randint(0, 4, (num_rel_edges,), dtype=torch.long)
    
    relation_graph = Data(
        edge_index=rel_edge_index,
        edge_type=rel_edge_type,
        num_nodes=num_rel_nodes,
    )
    relation_graph.num_relations = 4  # 关系图的关系数
    data.relation_graph = relation_graph
    
    # 添加relation_graph2（SemRelNBFNet需要，如果使用SEMMA）
    data.relation_graph2 = relation_graph  # 简化：使用相同的图
    
    # 添加harder_head_rg2（SemRelNBFNet在某些情况下需要）
    data.harder_head_rg2 = {}
    
    return data, batch

def test_model_initialization():
    """测试模型初始化"""
    print("=" * 60)
    print("测试1: 模型初始化")
    print("=" * 60)
    
    try:
        from ultra.enhanced_models import EnhancedUltra
        
        # 创建模型配置
        rel_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'num_relation': 10,
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'num_mlp_layer': 2
        }
        
        entity_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'num_relation': 10,
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'num_mlp_layer': 2
        }
        
        sem_model_cfg = {
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'num_relation': 10,
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'num_mlp_layer': 2
        }
        
        model = EnhancedUltra(
            rel_model_cfg=rel_model_cfg,
            entity_model_cfg=entity_model_cfg,
            sem_model_cfg=sem_model_cfg
        )
        
        print("✅ 模型初始化成功")
        print(f"   - 使用相似度增强器: {model.use_similarity_enhancer}")
        print(f"   - 使用提示图增强器: {model.use_prompt_enhancer}")
        print(f"   - 使用可学习融合: {model.use_learnable_fusion}")
        print(f"   - 使用增强置信度: {model.use_enhancement_confidence}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass_dimensions(model, data, batch):
    """测试前向传播的维度匹配"""
    print("\n" + "=" * 60)
    print("测试2: 前向传播维度匹配")
    print("=" * 60)
    
    try:
        model.eval()
        with torch.no_grad():
            # 前向传播
            output = model(data, batch)
            
            # 检查输出维度
            expected_shape = (batch.shape[0], batch.shape[1])  # (batch_size, 1+num_negs)
            actual_shape = output.shape
            
            print(f"✅ 前向传播成功")
            print(f"   - 期望输出形状: {expected_shape}")
            print(f"   - 实际输出形状: {actual_shape}")
            
            if actual_shape == expected_shape:
                print("   ✅ 输出维度匹配")
            else:
                print(f"   ❌ 输出维度不匹配!")
                return False
            
            # 检查中间表示的维度
            if model.final_relation_representations is not None:
                print(f"   - final_relation_representations形状: {model.final_relation_representations.shape}")
                expected_r_shape = (batch.shape[0], data.num_relations, 64)
                if model.final_relation_representations.shape == expected_r_shape:
                    print("   ✅ 关系表示维度正确")
                else:
                    print(f"   ❌ 关系表示维度错误! 期望: {expected_r_shape}, 实际: {model.final_relation_representations.shape}")
                    return False
            
            if model.enhanced_relation_representations is not None:
                print(f"   - enhanced_relation_representations形状: {model.enhanced_relation_representations.shape}")
                expected_enhanced_shape = (batch.shape[0], data.num_relations, 64)
                if model.enhanced_relation_representations.shape == expected_enhanced_shape:
                    print("   ✅ 增强关系表示维度正确")
                else:
                    print(f"   ❌ 增强关系表示维度错误!")
                    return False
            
            return True
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhancement_modules(model, data, batch):
    """测试增强模块的逻辑正确性"""
    print("\n" + "=" * 60)
    print("测试3: 增强模块逻辑正确性")
    print("=" * 60)
    
    try:
        model.eval()
        with torch.no_grad():
            # 获取基础表示
            query_rels = batch[:, 0, 2]
            r = model.final_relation_representations  # [batch_size, num_relations, embedding_dim]
            
            # 测试Similarity Enhancer
            if model.use_similarity_enhancer and model.similarity_enhancer is not None:
                print("测试 Similarity Enhancer...")
                r1_delta = model.similarity_enhancer(
                    r, 
                    query_rels,
                    return_enhancement_only=True
                )
                
                if r1_delta.shape == r.shape:
                    print("   ✅ Similarity Enhancer输出维度正确")
                    # 检查增量是否合理（不应该太大）
                    max_delta = torch.abs(r1_delta).max().item()
                    print(f"   - 最大增量值: {max_delta:.4f}")
                    if max_delta < 10.0:  # 合理的阈值
                        print("   ✅ 增量值在合理范围内")
                    else:
                        print("   ⚠️  增量值可能过大")
                else:
                    print(f"   ❌ Similarity Enhancer输出维度错误: {r1_delta.shape} vs {r.shape}")
                    return False
            
            # 测试Prompt Enhancer
            if model.use_prompt_enhancer and model.prompt_enhancer is not None:
                print("测试 Prompt Enhancer...")
                query_entities = batch[:, 0, 0]
                
                # 测试单个batch
                i = 0
                query_rel = query_rels[i]
                query_entity = query_entities[i]
                
                prompt_graph, prompt_entities_list = model.prompt_enhancer.generate_prompt_graph(
                    data, query_rel, query_entity
                )
                
                if prompt_graph is not None:
                    print(f"   ✅ 提示图生成成功 (节点数: {prompt_graph.num_nodes}, 边数: {prompt_graph.edge_index.shape[1]})")
                    
                    # 测试编码
                    prompt_context = model.prompt_enhancer.encode_prompt_context(
                        prompt_graph, query_rel, r[i], query_entity
                    )
                    
                    if prompt_context.shape == (64,):
                        print("   ✅ Prompt Enhancer编码输出维度正确")
                    else:
                        print(f"   ❌ Prompt Enhancer编码输出维度错误: {prompt_context.shape}")
                        return False
                else:
                    print("   ⚠️  提示图为空（可能是正常的，取决于图结构）")
            
            return True
            
    except Exception as e:
        print(f"❌ 增强模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_logic(model, data, batch):
    """测试融合逻辑"""
    print("\n" + "=" * 60)
    print("测试4: 融合逻辑")
    print("=" * 60)
    
    try:
        model.eval()
        with torch.no_grad():
            # 运行一次前向传播
            _ = model(data, batch)
            
            r = model.final_relation_representations
            enhanced = model.enhanced_relation_representations
            
            if r is None or enhanced is None:
                print("   ❌ 关系表示为空")
                return False
            
            # 检查融合权重
            if model.use_learnable_fusion and model.fusion_main_weights_logits is not None:
                main_weights = torch.softmax(model.fusion_main_weights_logits, dim=0)
                enhance_weights = torch.softmax(model.fusion_enhance_weights_logits, dim=0)
                
                print(f"   - 主权重: base={main_weights[0]:.3f}, enhance={main_weights[1]:.3f}")
                print(f"   - 增强器权重: sim={enhance_weights[0]:.3f}, prompt={enhance_weights[1]:.3f}")
                
                # 检查权重归一化
                if abs(main_weights.sum().item() - 1.0) < 1e-5:
                    print("   ✅ 主权重归一化正确")
                else:
                    print(f"   ❌ 主权重归一化错误: {main_weights.sum().item()}")
                    return False
                
                if abs(enhance_weights.sum().item() - 1.0) < 1e-5:
                    print("   ✅ 增强器权重归一化正确")
                else:
                    print(f"   ❌ 增强器权重归一化错误: {enhance_weights.sum().item()}")
                    return False
            
            # 检查增强后的表示是否合理
            diff = torch.abs(enhanced - r).mean().item()
            print(f"   - 增强前后平均差异: {diff:.4f}")
            
            if diff < 1.0:  # 合理的阈值
                print("   ✅ 增强幅度合理")
            else:
                print("   ⚠️  增强幅度可能过大")
            
            return True
            
    except Exception as e:
        print(f"❌ 融合逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases(model):
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试5: 边界情况")
    print("=" * 60)
    
    try:
        model.eval()
        
        # 测试1: 空图
        print("测试空图...")
        empty_data = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            num_nodes=10,
            num_relations=5
        )
        # 添加必要的relation_graph
        empty_rel_graph = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            num_nodes=5,
        )
        empty_rel_graph.num_relations = 4
        empty_data.relation_graph = empty_rel_graph
        empty_data.relation_graph2 = empty_rel_graph
        empty_data.harder_head_rg2 = {}
        
        # EntityNBFNet要求batch中所有样本的head索引相同
        empty_batch = torch.zeros(2, 2, 3, dtype=torch.long)
        head_entity = 0  # 所有样本使用相同的head
        empty_batch[0, 0, :] = torch.tensor([head_entity, 1, 0])
        empty_batch[0, 1, :] = torch.tensor([head_entity, 2, 0])  # negative sample
        empty_batch[1, 0, :] = torch.tensor([head_entity, 3, 1])
        empty_batch[1, 1, :] = torch.tensor([head_entity, 4, 1])  # negative sample
        
        with torch.no_grad():
            output = model(empty_data, empty_batch)
            if output.shape == (2, 2):
                print("   ✅ 空图处理正确")
            else:
                print(f"   ❌ 空图处理错误: {output.shape}")
                return False
        
        # 测试2: 单节点图
        print("测试单节点图...")
        single_node_data = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            num_nodes=1,
            num_relations=5
        )
        # 添加必要的relation_graph
        single_rel_graph = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros((0,), dtype=torch.long),
            num_nodes=5,
        )
        single_rel_graph.num_relations = 4
        single_node_data.relation_graph = single_rel_graph
        single_node_data.relation_graph2 = single_rel_graph
        single_node_data.harder_head_rg2 = {}
        
        # EntityNBFNet要求batch中所有样本的head索引相同
        single_batch = torch.zeros(1, 2, 3, dtype=torch.long)
        head_entity = 0
        single_batch[0, 0, :] = torch.tensor([head_entity, 0, 0])
        single_batch[0, 1, :] = torch.tensor([head_entity, 0, 0])  # negative sample (same as positive for simplicity)
        
        with torch.no_grad():
            output = model(single_node_data, single_batch)
            if output.shape == (1, 2):
                print("   ✅ 单节点图处理正确")
            else:
                print(f"   ❌ 单节点图处理错误: {output.shape}")
                return False
        
        # 测试3: 大批量
        print("测试大批量...")
        large_data, large_batch = create_mock_data(num_nodes=50, num_relations=10, num_edges=100, batch_size=16)
        with torch.no_grad():
            output = model(large_data, large_batch)
            if output.shape == (16, 2):
                print("   ✅ 大批量处理正确")
            else:
                print(f"   ❌ 大批量处理错误: {output.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow(model, data, batch):
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("测试6: 梯度流")
    print("=" * 60)
    
    try:
        model.train()
        
        # 前向传播
        output = model(data, batch)
        
        # 计算损失（简单的MSE）
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查关键参数的梯度
        has_gradient = False
        no_gradient = []
        
        if model.use_learnable_fusion and model.fusion_main_weights_logits is not None:
            if model.fusion_main_weights_logits.grad is not None:
                has_gradient = True
                print(f"   ✅ 融合权重有梯度: {model.fusion_main_weights_logits.grad.norm().item():.4f}")
            else:
                no_gradient.append("fusion_main_weights_logits")
        
        if model.use_similarity_enhancer and model.similarity_enhancer is not None:
            if model.similarity_enhancer.similarity_threshold_raw.grad is not None:
                has_gradient = True
                print(f"   ✅ 相似度阈值有梯度: {model.similarity_enhancer.similarity_threshold_raw.grad.norm().item():.4f}")
            else:
                no_gradient.append("similarity_threshold")
        
        if model.use_enhancement_confidence and model.enhancement_confidence is not None:
            for param in model.enhancement_confidence.parameters():
                if param.grad is not None:
                    has_gradient = True
                    print(f"   ✅ 置信度网络有梯度")
                    break
        
        if has_gradient:
            print("   ✅ 梯度流正常")
        else:
            print("   ⚠️  未检测到关键参数的梯度")
        
        if no_gradient:
            print(f"   ⚠️  以下参数无梯度: {no_gradient}")
        
        return True
        
    except Exception as e:
        print(f"❌ 梯度流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("EnhancedUltra 逻辑正确性测试")
    print("=" * 60 + "\n")
    
    # 创建测试数据
    data, batch = create_mock_data(num_nodes=50, num_relations=10, num_edges=100, batch_size=4)
    print(f"测试数据: {data.num_nodes}个节点, {data.num_relations}个关系, {data.edge_index.shape[1]}条边")
    print(f"Batch大小: {batch.shape[0]}\n")
    
    # 测试1: 模型初始化
    model = test_model_initialization()
    if model is None:
        print("\n❌ 模型初始化失败，终止测试")
        return
    
    # 测试2: 前向传播维度
    if not test_forward_pass_dimensions(model, data, batch):
        print("\n❌ 前向传播维度测试失败")
        return
    
    # 测试3: 增强模块
    if not test_enhancement_modules(model, data, batch):
        print("\n❌ 增强模块测试失败")
        return
    
    # 测试4: 融合逻辑
    if not test_fusion_logic(model, data, batch):
        print("\n❌ 融合逻辑测试失败")
        return
    
    # 测试5: 边界情况
    if not test_edge_cases(model):
        print("\n❌ 边界情况测试失败")
        return
    
    # 测试6: 梯度流
    if not test_gradient_flow(model, data, batch):
        print("\n❌ 梯度流测试失败")
        return
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！EnhancedUltra逻辑正确性验证成功")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

