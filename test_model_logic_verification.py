#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型逻辑验证测试脚本
验证EnhancedUltra的关键逻辑是否符合设计
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_loading():
    """测试配置加载"""
    logger.info("=" * 60)
    logger.info("测试1: 配置加载")
    logger.info("=" * 60)
    
    try:
        from ultra import parse
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        
        # 检查关键配置
        checks = {
            "max_hops": getattr(flags, 'max_hops', None),
            "num_prompt_samples": getattr(flags, 'num_prompt_samples', None),
            "max_similar_relations": getattr(flags, 'max_similar_relations', None),
            "use_similarity_enhancer": getattr(flags, 'use_similarity_enhancer', None),
            "use_prompt_enhancer": getattr(flags, 'use_prompt_enhancer', None),
            "use_learnable_fusion": getattr(flags, 'use_learnable_fusion', None),
        }
        
        logger.info("配置检查结果:")
        for key, value in checks.items():
            status = "✓" if value is not None else "✗"
            logger.info(f"  {status} {key}: {value}")
        
        # 验证关键值
        assert flags.max_hops == 2, f"max_hops应该是2，实际是{flags.max_hops}"
        assert flags.num_prompt_samples == 15, f"num_prompt_samples应该是15，实际是{flags.num_prompt_samples}"
        assert flags.max_similar_relations == 3, f"max_similar_relations应该是3，实际是{flags.max_similar_relations}"
        
        logger.info("✓ 配置加载测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 配置加载测试失败: {e}")
        return False

def test_model_initialization():
    """测试模型初始化"""
    logger.info("=" * 60)
    logger.info("测试2: 模型初始化")
    logger.info("=" * 60)
    
    try:
        from ultra.enhanced_models import EnhancedUltra
        from ultra import parse
        
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        
        # 创建模拟配置
        rel_model_cfg = {
            'num_relations': 51,
            'hidden_dim': 64,
            'num_layers': 6,
        }
        entity_model_cfg = {
            'num_relations': 51,
            'hidden_dim': 64,
            'num_layers': 6,
        }
        sem_model_cfg = {
            'num_relations': 51,
            'hidden_dim': 64,
        }
        
        model = EnhancedUltra(rel_model_cfg, entity_model_cfg, sem_model_cfg)
        
        # 检查关键组件
        checks = {
            "similarity_enhancer": model.similarity_enhancer is not None if flags.use_similarity_enhancer else model.similarity_enhancer is None,
            "prompt_enhancer": model.prompt_enhancer is not None if flags.use_prompt_enhancer else model.prompt_enhancer is None,
            "entity_model": model.entity_model is not None,
        }
        
        logger.info("模型组件检查结果:")
        for key, value in checks.items():
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}: {value}")
        
        # 检查prompt_enhancer的配置
        if model.prompt_enhancer is not None:
            assert model.prompt_enhancer.max_hops == flags.max_hops, \
                f"prompt_enhancer.max_hops应该是{flags.max_hops}，实际是{model.prompt_enhancer.max_hops}"
            assert model.prompt_enhancer.num_prompt_samples == flags.num_prompt_samples, \
                f"prompt_enhancer.num_prompt_samples应该是{flags.num_prompt_samples}，实际是{model.prompt_enhancer.num_prompt_samples}"
            logger.info(f"  ✓ prompt_enhancer.max_hops: {model.prompt_enhancer.max_hops}")
            logger.info(f"  ✓ prompt_enhancer.num_prompt_samples: {model.prompt_enhancer.num_prompt_samples}")
        
        # 检查similarity_enhancer的配置
        if model.similarity_enhancer is not None:
            assert model.similarity_enhancer.max_similar_relations == flags.max_similar_relations, \
                f"similarity_enhancer.max_similar_relations应该是{flags.max_similar_relations}，实际是{model.similarity_enhancer.max_similar_relations}"
            logger.info(f"  ✓ similarity_enhancer.max_similar_relations: {model.similarity_enhancer.max_similar_relations}")
        
        logger.info("✓ 模型初始化测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 模型初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_key_constants():
    """测试关键常量"""
    logger.info("=" * 60)
    logger.info("测试3: 关键常量检查")
    logger.info("=" * 60)
    
    try:
        from ultra.enhanced_models import OptimizedPromptGraph, EntityRelationJointEnhancer
        
        # 检查OptimizedPromptGraph的常量
        # 注意：这些是类内部的常量，需要通过实例或直接检查代码
        logger.info("检查关键常量（通过代码检查）:")
        
        # 读取文件检查常量
        with open(os.path.join(project_root, "ultra", "enhanced_models.py"), "r", encoding="utf-8") as f:
            content = f.read()
            
        constants_to_check = {
            "MAX_ENTITIES_FOR_NBFNET": 30,
            "MAX_PROMPT_ENTITIES": 6,
            "MAX_ENTITIES_TO_COMPUTE": 100,
        }
        
        for const_name, expected_value in constants_to_check.items():
            # 查找常量定义
            import re
            pattern = rf"{const_name}\s*=\s*(\d+)"
            match = re.search(pattern, content)
            if match:
                actual_value = int(match.group(1))
                status = "✓" if actual_value == expected_value else "✗"
                logger.info(f"  {status} {const_name}: {actual_value} (期望: {expected_value})")
                if actual_value != expected_value:
                    logger.warning(f"    警告: {const_name}的值不符合预期")
            else:
                logger.warning(f"  ✗ 未找到常量 {const_name}")
        
        logger.info("✓ 关键常量检查完成")
        return True
    except Exception as e:
        logger.error(f"✗ 关键常量检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_logic():
    """测试前向传播逻辑"""
    logger.info("=" * 60)
    logger.info("测试4: 前向传播逻辑（简化）")
    logger.info("=" * 60)
    
    try:
        from ultra.enhanced_models import EnhancedUltra
        from torch_geometric.data import Data
        import parse
        
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        
        # 创建模拟数据
        num_nodes = 100
        num_relations = 51
        batch_size = 2
        
        # 创建模拟图数据
        edge_index = torch.randint(0, num_nodes, (2, 200), dtype=torch.long)
        edge_type = torch.randint(0, num_relations, (200,), dtype=torch.long)
        
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
        )
        
        # 创建batch
        batch = {
            'h_index': torch.randint(0, num_nodes, (batch_size,), dtype=torch.long),
            't_index': torch.randint(0, num_nodes, (batch_size,), dtype=torch.long),
            'r_index': torch.randint(0, num_relations, (batch_size,), dtype=torch.long),
        }
        
        # 创建模型
        rel_model_cfg = {
            'num_relations': num_relations,
            'hidden_dim': 64,
            'num_layers': 2,  # 减少层数以加快测试
        }
        entity_model_cfg = {
            'num_relations': num_relations,
            'hidden_dim': 64,
            'num_layers': 2,
        }
        sem_model_cfg = {
            'num_relations': num_relations,
            'hidden_dim': 64,
        }
        
        model = EnhancedUltra(rel_model_cfg, entity_model_cfg, sem_model_cfg)
        model.eval()
        
        # 测试前向传播
        with torch.no_grad():
            try:
                score = model(data, batch)
                logger.info(f"  ✓ 前向传播成功，输出形状: {score.shape}")
                
                # 检查输出形状
                assert score.dim() == 1 or score.dim() == 2, f"输出维度应该是1或2，实际是{score.dim()}"
                assert score.shape[0] == batch_size, f"输出第一维应该是batch_size({batch_size})，实际是{score.shape[0]}"
                
                logger.info("✓ 前向传播逻辑测试通过")
                return True
            except Exception as e:
                logger.error(f"  ✗ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False
    except Exception as e:
        logger.error(f"✗ 前向传播逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("开始模型逻辑验证测试")
    logger.info("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("配置加载", test_config_loading()))
    results.append(("模型初始化", test_model_initialization()))
    results.append(("关键常量", test_key_constants()))
    results.append(("前向传播逻辑", test_forward_logic()))
    
    # 汇总结果
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！模型逻辑验证成功！")
        return 0
    else:
        logger.error("❌ 部分测试失败，请检查代码")
        return 1

if __name__ == "__main__":
    sys.exit(main())

