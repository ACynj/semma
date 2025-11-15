#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试：只测试我们修改的部分
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompt_enhancer_modifications():
    """测试Prompt Enhancer的修改"""
    print("=" * 80)
    print("测试: Prompt Enhancer修改验证")
    print("=" * 80)
    
    try:
        from ultra.enhanced_models import OptimizedPromptGraph
        from torch_geometric.data import Data
        
        embedding_dim = 64
        num_relations = 100
        num_nodes = 50
        
        # 创建prompt enhancer
        prompt_enhancer = OptimizedPromptGraph(
            embedding_dim=embedding_dim,
            max_hops=2,
            num_prompt_samples=5  # 测试新的参数值
        )
        
        # 验证参数已更新
        assert prompt_enhancer.num_prompt_samples == 5, f"num_prompt_samples应该是5，实际是{prompt_enhancer.num_prompt_samples}"
        print(f"✓ num_prompt_samples已更新为: {prompt_enhancer.num_prompt_samples}")
        
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
        relation_embeddings = torch.randn(num_relations, embedding_dim)
        
        # 创建提示图
        prompt_graph = prompt_enhancer.generate_prompt_graph(data, query_relation, query_entity)
        
        # 测试1: 验证新参数relation_embeddings可以传入
        print("\n测试1: encode_prompt_context接受relation_embeddings参数")
        prompt_enhancer.eval()
        context1 = prompt_enhancer.encode_prompt_context(
            prompt_graph, query_relation, relation_embeddings
        )
        assert context1.shape == (embedding_dim,), f"上下文形状错误: {context1.shape}"
        assert not torch.allclose(context1, torch.zeros_like(context1)), "上下文不应该是零向量"
        print(f"  ✓ 成功使用关系嵌入初始化，上下文形状: {context1.shape}")
        print(f"  ✓ 上下文不是零向量（说明初始化改进生效）")
        
        # 测试2: 验证不使用relation_embeddings时的回退逻辑
        print("\n测试2: 回退模式（不使用relation_embeddings）")
        context2 = prompt_enhancer.encode_prompt_context(
            prompt_graph, query_relation, None
        )
        assert context2.shape == (embedding_dim,), f"上下文形状错误: {context2.shape}"
        print(f"  ✓ 回退模式工作正常，上下文形状: {context2.shape}")
        
        # 测试3: 验证forward方法接受relation_embeddings参数
        print("\n测试3: forward方法接受relation_embeddings参数")
        output = prompt_enhancer(
            data, query_relation, query_entity, base_embeddings,
            return_enhancement_only=True,
            relation_embeddings=relation_embeddings
        )
        assert output.shape == (embedding_dim,), f"输出形状错误: {output.shape}"
        print(f"  ✓ forward方法成功接受relation_embeddings参数，输出形状: {output.shape}")
        
        # 测试4: 验证推理时使用关系嵌入而不是零向量
        print("\n测试4: 推理时使用关系嵌入初始化（关键改进）")
        # 使用相同的关系嵌入，应该得到相同的结果（确定性）
        context3 = prompt_enhancer.encode_prompt_context(
            prompt_graph, query_relation, relation_embeddings
        )
        assert torch.allclose(context1, context3), "推理时应该具有确定性"
        print(f"  ✓ 推理时使用关系嵌入，结果具有确定性")
        
        # 对比：如果不使用关系嵌入，结果应该不同
        context4 = prompt_enhancer.encode_prompt_context(
            prompt_graph, query_relation, None
        )
        assert not torch.allclose(context1, context4), "使用关系嵌入和不使用应该产生不同结果"
        print(f"  ✓ 使用关系嵌入和不使用产生不同结果（说明改进生效）")
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！Prompt Enhancer修改正确。")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flags_config():
    """测试flags.yaml配置"""
    print("\n" + "=" * 80)
    print("测试: flags.yaml配置验证")
    print("=" * 80)
    
    try:
        from ultra import parse
        
        flags = parse.load_flags("flags.yaml")
        
        # 检查关键参数
        threshold = getattr(flags, 'similarity_threshold_init', None)
        strength = getattr(flags, 'enhancement_strength_init', None)
        use_learnable = getattr(flags, 'use_learnable_fusion', None)
        
        print(f"\n检查参数值:")
        print(f"  similarity_threshold_init: {threshold}")
        print(f"  enhancement_strength_init: {strength}")
        print(f"  use_learnable_fusion: {use_learnable}")
        
        # 验证参数值
        checks = []
        
        if threshold == 0.72:
            print(f"  ✓ similarity_threshold_init已更新为0.72（从0.85降低）")
            checks.append(True)
        else:
            print(f"  ⚠ similarity_threshold_init: {threshold} (期望: 0.72)")
            checks.append(False)
        
        if strength == 0.12:
            print(f"  ✓ enhancement_strength_init已更新为0.12（从0.09提升）")
            checks.append(True)
        else:
            print(f"  ⚠ enhancement_strength_init: {strength} (期望: 0.12)")
            checks.append(False)
        
        if use_learnable == False:
            print(f"  ✓ use_learnable_fusion已设置为False（使用固定权重）")
            checks.append(True)
        else:
            print(f"  ⚠ use_learnable_fusion: {use_learnable} (期望: False)")
            checks.append(False)
        
        all_ok = all(checks)
        
        if all_ok:
            print("\n✅ flags.yaml配置正确！")
        else:
            print("\n⚠️  部分配置需要检查")
        
        return all_ok
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_syntax():
    """测试代码语法"""
    print("\n" + "=" * 80)
    print("测试: 代码语法检查")
    print("=" * 80)
    
    import py_compile
    
    try:
        # 编译检查
        py_compile.compile('ultra/enhanced_models.py', doraise=True)
        print("✓ ultra/enhanced_models.py 语法正确")
        
        # 检查关键修改点
        with open('ultra/enhanced_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = []
        
        # 检查1: encode_prompt_context是否接受relation_embeddings参数
        if 'def encode_prompt_context(self, prompt_graph, query_relation, relation_embeddings=None):' in content:
            print("✓ encode_prompt_context方法签名正确（包含relation_embeddings参数）")
            checks.append(True)
        else:
            print("✗ encode_prompt_context方法签名可能有问题")
            checks.append(False)
        
        # 检查2: forward方法是否接受relation_embeddings参数
        if 'def forward(self, data, query_relation, query_entity, base_embeddings, return_enhancement_only=False, relation_embeddings=None):' in content:
            print("✓ forward方法签名正确（包含relation_embeddings参数）")
            checks.append(True)
        else:
            print("✗ forward方法签名可能有问题")
            checks.append(False)
        
        # 检查3: 是否使用关系嵌入初始化
        if 'relation_embeddings is not None' in content and 'base_embedding = relation_embeddings' in content:
            print("✓ 代码中包含使用关系嵌入初始化的逻辑")
            checks.append(True)
        else:
            print("✗ 可能缺少使用关系嵌入初始化的逻辑")
            checks.append(False)
        
        # 检查4: num_prompt_samples是否更新为5
        if 'num_prompt_samples=5' in content:
            print("✓ num_prompt_samples已更新为5")
            checks.append(True)
        else:
            print("✗ num_prompt_samples可能未更新")
            checks.append(False)
        
        # 检查5: 调用时是否传入relation_embeddings
        if 'relation_embeddings=r[i]' in content:
            print("✓ EnhancedUltra.forward中正确传入了relation_embeddings")
            checks.append(True)
        else:
            print("✗ EnhancedUltra.forward中可能未传入relation_embeddings")
            checks.append(False)
        
        all_ok = all(checks)
        
        if all_ok:
            print("\n✅ 代码语法和关键修改点检查通过！")
        else:
            print("\n⚠️  部分检查未通过，请确认修改")
        
        return all_ok
        
    except py_compile.PyCompileError as e:
        print(f"\n❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("EnhancedUltra修改验证测试（简化版）")
    print("=" * 80)
    
    results = []
    
    # 运行测试
    results.append(("代码语法检查", test_code_syntax()))
    results.append(("Prompt Enhancer修改", test_prompt_enhancer_modifications()))
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
        print("\n🎉 所有测试通过！代码修改正确，可以安全使用。")
        print("\n主要改进:")
        print("  1. ✓ Prompt Enhancer现在使用关系嵌入初始化（而不是零向量）")
        print("  2. ✓ 相似度阈值从0.85降低到0.72")
        print("  3. ✓ 增强强度从0.09提升到0.12")
        print("  4. ✓ 提示样本数从3增加到5")
        print("  5. ✓ 使用固定权重融合（use_learnable_fusion=False）")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

