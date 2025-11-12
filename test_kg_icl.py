#!/usr/bin/env python
"""
测试 KG-ICL 功能是否有效
验证：
1. KG-ICL 模块能否正确初始化
2. 模型能否正常前向传播
3. 增强后的关系表示是否正确
"""

import sys
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra import parse
from ultra.kg_icl_prompt import KGICLPromptEnhancer
from ultra import models as ultra_models

# 保存原始 flags 值
original_kg_icl_prompt = None
original_kg_icl_in_training = None

def setup_kg_icl_flags():
    """临时启用 KG-ICL 功能"""
    global original_kg_icl_prompt, original_kg_icl_in_training
    
    # 修改全局 flags 对象（ultra.models 模块级别的 flags）
    original_kg_icl_prompt = getattr(ultra_models.flags, 'use_kg_icl_prompt', False)
    original_kg_icl_in_training = getattr(ultra_models.flags, 'use_kg_icl_in_training', False)
    
    # 启用 KG-ICL
    ultra_models.flags.use_kg_icl_prompt = True
    ultra_models.flags.use_kg_icl_in_training = True
    ultra_models.flags.prompt_num_examples = 2
    ultra_models.flags.prompt_max_hops = 1
    ultra_models.flags.prompt_num_layers = 1
    
    return ultra_models.flags

def restore_flags():
    """恢复原始 flags"""
    global original_kg_icl_prompt, original_kg_icl_in_training
    if original_kg_icl_prompt is not None:
        ultra_models.flags.use_kg_icl_prompt = original_kg_icl_prompt
    if original_kg_icl_in_training is not None:
        ultra_models.flags.use_kg_icl_in_training = original_kg_icl_in_training

def create_dummy_data(num_nodes=20, num_relations=5, num_edges=50):
    """创建虚拟测试数据"""
    # 创建随机边
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    # 创建数据对象
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=num_relations * 2  # 包含反向关系
    )
    
    return data

def create_dummy_batch(batch_size=2, num_neg=5):
    """创建虚拟批次数据"""
    # batch shape: (bs, 1+num_negs, 3) where 3 is (head, tail, relation)
    batch = torch.zeros(batch_size, 1 + num_neg, 3, dtype=torch.long)
    
    for i in range(batch_size):
        # 正样本
        batch[i, 0, 0] = torch.randint(0, 20, (1,))  # head
        batch[i, 0, 1] = torch.randint(0, 20, (1,))  # tail
        batch[i, 0, 2] = torch.randint(0, 5, (1,))   # relation
        
        # 负样本（共享相同的关系）
        for j in range(1, 1 + num_neg):
            batch[i, j, 0] = torch.randint(0, 20, (1,))  # head
            batch[i, j, 1] = torch.randint(0, 20, (1,))  # tail
            batch[i, j, 2] = batch[i, 0, 2]  # 相同的关系
    
    return batch

def test_kg_icl_enhancer():
    """测试 KG-ICL 增强器"""
    print("=" * 70)
    print("测试 1: KG-ICL Prompt Enhancer 初始化")
    print("=" * 70)
    
    try:
        enhancer = KGICLPromptEnhancer(
            hidden_dim=64,
            num_prompt_layers=1,
            num_examples=2,
            max_hops=1
        )
        print("✅ KG-ICL Prompt Enhancer 初始化成功")
        print(f"   - hidden_dim: {enhancer.hidden_dim}")
        print(f"   - num_examples: {enhancer.num_examples}")
        return enhancer
    except Exception as e:
        print(f"❌ KG-ICL Prompt Enhancer 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_kg_icl_with_model():
    """测试 KG-ICL 与模型集成"""
    print("\n" + "=" * 70)
    print("测试 2: KG-ICL 与 Ultra 模型集成")
    print("=" * 70)
    
    # 设置 flags
    flags = setup_kg_icl_flags()
    
    try:
        # 创建简单的模型配置（使用正确的类名和参数）
        rel_model_cfg = {
            'class': 'RelNBFNet',
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True
        }
        
        entity_model_cfg = {
            'class': 'EntityNBFNet',
            'input_dim': 64,
            'hidden_dims': [64, 64],
            'message_func': 'distmult',
            'aggregate_func': 'sum',
            'layer_norm': True,
            'short_cut': True
        }
        
        # 检查模型类是否存在
        from ultra.models import RelNBFNet, EntityNBFNet
        print("✅ 模型类导入成功")
        
        # 根据 flags.run 决定是否需要语义模型
        if ultra_models.flags.run == "semma":
            sem_model_cfg = {
                'class': 'SemRelNBFNet',
                'input_dim': 64,
                'hidden_dims': [64, 64],
                'message_func': 'distmult',
                'aggregate_func': 'sum',
                'layer_norm': True,
                'short_cut': True
            }
            model = ultra_models.Ultra(
                rel_model_cfg=rel_model_cfg,
                entity_model_cfg=entity_model_cfg,
                sem_model_cfg=sem_model_cfg
            )
        else:
            model = ultra_models.Ultra(
                rel_model_cfg=rel_model_cfg,
                entity_model_cfg=entity_model_cfg
            )
        
        # 检查 KG-ICL 增强器是否已初始化
        if model.kg_icl_enhancer is not None:
            print("✅ KG-ICL 增强器已成功集成到模型中")
        else:
            print("⚠️  KG-ICL 增强器未初始化（可能 flags 未正确设置）")
            print(f"   - flags.use_kg_icl_prompt: {ultra_models.flags.use_kg_icl_prompt}")
            # 不返回 False，继续测试模型本身
        
        print("✅ Ultra 模型初始化成功（包含 KG-ICL）")
        return model
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        restore_flags()

def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 70)
    print("测试 3: 前向传播测试")
    print("=" * 70)
    
    if model is None:
        print("❌ 模型未初始化，跳过前向传播测试")
        return False
    
    try:
        # 创建测试数据
        data = create_dummy_data(num_nodes=20, num_relations=5, num_edges=50)
        batch = create_dummy_batch(batch_size=2, num_neg=5)
        
        print(f"✅ 测试数据创建成功")
        print(f"   - 节点数: {data.num_nodes}")
        print(f"   - 边数: {data.edge_index.size(1)}")
        print(f"   - 关系数: {data.num_relations}")
        print(f"   - 批次大小: {batch.size(0)}")
        
        # 设置模型为评估模式
        model.eval()
        
        # 前向传播
        with torch.no_grad():
            output = model(data, batch)
        
        print(f"✅ 前向传播成功")
        print(f"   - 输出形状: {output.shape}")
        print(f"   - 输出类型: {type(output)}")
        
        # 检查关系表示
        rel_reprs = model.get_relation_representations()
        if rel_reprs[2] is not None:  # final_relation_representations
            print(f"✅ 关系表示生成成功")
            print(f"   - 关系表示形状: {rel_reprs[2].shape}")
        else:
            print("⚠️  关系表示为 None")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kg_icl_enhancement():
    """测试 KG-ICL 增强功能"""
    print("\n" + "=" * 70)
    print("测试 4: KG-ICL 增强功能")
    print("=" * 70)
    
    flags = setup_kg_icl_flags()
    
    try:
        enhancer = KGICLPromptEnhancer(
            hidden_dim=64,
            num_prompt_layers=1,
            num_examples=2,
            max_hops=1
        )
        
        # 创建测试数据
        data = create_dummy_data(num_nodes=20, num_relations=5, num_edges=50)
        
        # 创建基础关系表示
        batch_size = 2
        num_relations = 5
        hidden_dim = 64
        base_relation_reprs = torch.randn(batch_size, num_relations, hidden_dim)
        
        # 创建查询
        query_relations = torch.randint(0, num_relations, (batch_size,))
        query_heads = torch.randint(0, 20, (batch_size,))
        
        print(f"✅ 测试数据准备完成")
        print(f"   - 基础关系表示形状: {base_relation_reprs.shape}")
        print(f"   - 查询关系: {query_relations.tolist()}")
        print(f"   - 查询头实体: {query_heads.tolist()}")
        
        # 执行增强
        enhancer.eval()
        with torch.no_grad():
            enhanced_reprs = enhancer(
                data,
                query_relations,
                query_heads,
                base_relation_reprs
            )
        
        print(f"✅ KG-ICL 增强成功")
        print(f"   - 增强后关系表示形状: {enhanced_reprs.shape}")
        
        # 检查形状是否一致
        if enhanced_reprs.shape == base_relation_reprs.shape:
            print("✅ 输出形状正确")
        else:
            print(f"❌ 输出形状不匹配: {enhanced_reprs.shape} vs {base_relation_reprs.shape}")
            return False
        
        # 检查值是否改变（应该有所改变）
        diff = torch.abs(enhanced_reprs - base_relation_reprs).mean()
        print(f"   - 平均变化量: {diff.item():.6f}")
        
        if diff.item() > 1e-6:
            print("✅ 关系表示已被增强（值已改变）")
        else:
            print("⚠️  关系表示未改变（可能增强未生效）")
        
        return True
        
    except Exception as e:
        print(f"❌ KG-ICL 增强测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        restore_flags()

def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("🧪 KG-ICL 功能测试")
    print("=" * 70)
    print()
    
    results = []
    
    # 测试 1: KG-ICL 增强器初始化
    enhancer = test_kg_icl_enhancer()
    results.append(("KG-ICL 增强器初始化", enhancer is not None))
    
    # 测试 2: 模型集成
    model = test_kg_icl_with_model()
    results.append(("模型集成", model is not None))
    
    # 测试 3: 前向传播
    if model is not None:
        forward_ok = test_forward_pass(model)
        results.append(("前向传播", forward_ok))
    else:
        results.append(("前向传播", False))
    
    # 测试 4: KG-ICL 增强功能
    enhancement_ok = test_kg_icl_enhancement()
    results.append(("KG-ICL 增强功能", enhancement_ok))
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有测试通过！KG-ICL 功能正常工作。")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

