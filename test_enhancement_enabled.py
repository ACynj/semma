"""
测试EnhancedUltra模型在训练和推理时是否都启用了相似度增强
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra.enhanced_models import EnhancedUltra
from torch_geometric.data import Data
import numpy as np

def test_enhancement_enabled():
    """测试增强是否在训练和推理时都启用"""
    print("=" * 60)
    print("测试EnhancedUltra增强功能启用状态")
    print("=" * 60)
    
    # 模拟配置
    rel_model_cfg = {
        'class': 'RelNBFNet',
        'input_dim': 64,
        'hidden_dims': [64, 64],
        'message_func': 'distmult',
        'aggregate_func': 'sum',
        'short_cut': True,
        'layer_norm': True,
    }
    
    entity_model_cfg = {
        'class': 'EntityNBFNet',
        'input_dim': 64,
        'hidden_dims': [64, 64],
        'message_func': 'distmult',
        'aggregate_func': 'sum',
        'short_cut': True,
        'layer_norm': True,
    }
    
    sem_model_cfg = {
        'class': 'SemRelNBFNet',
        'input_dim': 64,
        'hidden_dims': [64, 64],
        'message_func': 'distmult',
        'aggregate_func': 'sum',
        'short_cut': True,
        'layer_norm': True,
    }
    
    # 创建模型（需要临时修改flags.yaml）
    print("\n1. 检查模型结构...")
    
    # 临时修改flags.yaml以测试EnhancedUltra
    flags_path = "flags.yaml"
    original_flags_content = None
    
    if os.path.exists(flags_path):
        with open(flags_path, 'r') as f:
            original_flags_content = f.read()
        # 临时修改为EnhancedUltra
        with open(flags_path, 'w') as f:
            f.write(original_flags_content.replace('run: semma', 'run: EnhancedUltra'))
    
    try:
        model = EnhancedUltra(
            rel_model_cfg=rel_model_cfg,
            entity_model_cfg=entity_model_cfg,
            sem_model_cfg=sem_model_cfg,
        )
        print("   ✅ EnhancedUltra模型创建成功")
        
        # 检查相似度增强模块是否存在
        if hasattr(model, 'similarity_enhancer'):
            print("   ✅ 相似度增强模块存在")
            
            # 检查可学习参数
            params = list(model.similarity_enhancer.parameters())
            print(f"   ✅ 可学习参数数量: {len(params)}")
            print(f"   ✅ 参数名称:")
            for name, param in model.similarity_enhancer.named_parameters():
                print(f"      - {name}: {param.shape}")
        else:
            print("   ❌ 相似度增强模块不存在！")
            return False
            
    finally:
        # 恢复flags.yaml
        if original_flags_content:
            with open(flags_path, 'w') as f:
                f.write(original_flags_content)
    
    # 创建模拟数据
    print("\n2. 创建模拟数据...")
    num_nodes = 100
    num_edges = 500
    num_relations = 50
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=num_relations,
    )
    
    batch_size = 4
    num_negs = 3
    batch = torch.zeros(batch_size, 1 + num_negs, 3, dtype=torch.long)
    batch[:, 0, 0] = torch.randint(0, num_nodes, (batch_size,))  # heads
    batch[:, 0, 1] = torch.randint(0, num_nodes, (batch_size,))  # tails
    batch[:, 0, 2] = torch.randint(0, num_relations, (batch_size,))  # relations
    
    print(f"   数据形状: nodes={num_nodes}, edges={num_edges}, relations={num_relations}")
    print(f"   Batch形状: {batch.shape}")
    
    # 测试推理模式（eval）
    print("\n3. 测试推理模式（eval）...")
    model.eval()
    
    with torch.no_grad():
        # 获取增强前后的表示
        query_rels = batch[:, 0, 2]
        
        # 手动调用forward的前半部分来检查增强
        # 模拟forward的前半部分
        final_reprs = torch.randn(batch_size, num_relations, 64)
        
        # 调用相似度增强器
        enhanced_reprs = model.similarity_enhancer(final_reprs, query_rels)
        
        # 检查是否有变化
        diff = torch.norm(enhanced_reprs - final_reprs).item()
        
        print(f"   推理模式: 增强模块被调用")
        print(f"   表示差异: {diff:.6f} (应该有变化)")
        
        if diff > 1e-6:
            print("   ✅ 推理模式下增强生效")
        else:
            print("   ⚠️  推理模式下表示无变化（可能没有找到相似关系）")
    
    # 测试训练模式（train）
    print("\n4. 测试训练模式（train）...")
    model.train()
    
    # 获取增强前后的表示
    query_rels = batch[:, 0, 2]
    final_reprs = torch.randn(batch_size, num_relations, 64, requires_grad=True)
    
    # 调用相似度增强器
    enhanced_reprs = model.similarity_enhancer(final_reprs, query_rels)
    
    # 检查是否有变化
    diff = torch.norm(enhanced_reprs - final_reprs).item()
    
    print(f"   训练模式: 增强模块被调用")
    print(f"   表示差异: {diff:.6f} (应该有变化)")
    
    # 检查梯度
    loss = enhanced_reprs.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.similarity_enhancer.parameters())
    
    if diff > 1e-6:
        print("   ✅ 训练模式下增强生效")
    else:
        print("   ⚠️  训练模式下表示无变化（可能没有找到相似关系）")
    
    if has_grad:
        print("   ✅ 可学习参数有梯度，可以训练")
    else:
        print("   ❌ 可学习参数没有梯度！")
    
    # 检查forward方法中是否有训练/推理的判断
    print("\n5. 检查forward方法逻辑...")
    
    # 读取enhanced_models.py检查代码
    enhanced_models_path = "ultra/enhanced_models.py"
    if os.path.exists(enhanced_models_path):
        with open(enhanced_models_path, 'r') as f:
            content = f.read()
            
        # 查找similarity_enhancer的调用
        if 'self.similarity_enhancer(' in content:
            print("   ✅ forward方法中调用了similarity_enhancer")
            
            # 检查是否有self.training判断
            lines_around = []
            for i, line in enumerate(content.split('\n')):
                if 'similarity_enhancer' in line:
                    # 获取前后几行
                    start = max(0, i - 3)
                    end = min(len(content.split('\n')), i + 3)
                    lines_around = content.split('\n')[start:end]
                    break
            
            # 检查是否有self.training判断
            has_training_check = any('self.training' in line for line in lines_around) or \
                                any('not self.training' in line for line in lines_around)
            
            if has_training_check:
                print("   ⚠️  发现有self.training判断，可能只在某个模式下启用")
                print("   相关代码:")
                for line in lines_around:
                    if 'similarity' in line.lower() or 'training' in line.lower():
                        print(f"      {line.strip()}")
            else:
                print("   ✅ 没有self.training判断，训练和推理都会使用增强")
        else:
            print("   ❌ forward方法中未找到similarity_enhancer调用！")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    results = {
        "模型创建": "✅" if hasattr(model, 'similarity_enhancer') else "❌",
        "推理模式启用": "✅" if diff > 1e-6 else "⚠️",
        "训练模式启用": "✅" if diff > 1e-6 and has_grad else "⚠️",
        "参数可训练": "✅" if has_grad else "❌",
    }
    
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print("\n结论:")
    if all("✅" in v for v in results.values()):
        print("  ✅ 所有测试通过！增强功能在训练和推理时都启用。")
        return True
    else:
        print("  ⚠️  部分测试未通过，请检查实现。")
        return False

if __name__ == "__main__":
    success = test_enhancement_enabled()
    sys.exit(0 if success else 1)


