"""
分析每个测试样本的相似关系参考情况
统计有多少可以参考的、有多少是有效的、多少引入了噪音
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra import datasets, util, parse
from ultra.models import Ultra
from ultra.enhanced_models import EnhancedUltra
from torch_geometric.data import Data
import torch.nn.functional as F

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_flags():
    """加载flags.yaml配置"""
    flags_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "flags.yaml")
    with open(flags_path, 'r', encoding='utf-8') as f:
        flags = yaml.safe_load(f)
    return flags

def load_dataset(dataset_name, dataset_type="transductive"):
    """加载数据集"""
    flags = load_flags()
    
    # 处理数据集名称映射
    dataset_name_mapping = {
        'YAGO310-ht': 'YAGO310',
        'ConceptNet 100k-ht': 'ConceptNet100k',
        'WDsinger-ht': 'WDsinger',
        'AristoV4-ht': 'AristoV4',
        'FB15K237Inductive:v1': 'FB15k237Inductive',  # 注意：小写k
        'FB15K237Inductive:v2': 'FB15k237Inductive',
        'FB15K237Inductive:v3': 'FB15k237Inductive',
        'FB15K237Inductive:v4': 'FB15k237Inductive',
        'WN18RRInductive:v3': 'WN18RRInductive',
        'NELLInductive:v1': 'NELLInductive',
        'NELLInductive:v3': 'NELLInductive',
        'NELLInductive:v4': 'NELLInductive',
        'WKIngram:25': 'WKIngram',
        'NLIngram:25': 'NLIngram',
        'NLIngram:75': 'NLIngram',
        'Metafam': 'Metafam',
        'WikiTopicsMT1:health': 'WikiTopicsMT1',
        'WikiTopicsMT3:infra': 'WikiTopicsMT3',
    }
    
    # 获取实际的数据集类名
    actual_dataset_name = dataset_name_mapping.get(dataset_name, dataset_name)
    
    # 处理版本号
    version = None
    if ':' in dataset_name:
        parts = dataset_name.split(':')
        if len(parts) == 2:
            # 如果已经在mapping中，不要覆盖
            if dataset_name not in dataset_name_mapping:
                # 对于FB15K237Inductive系列，统一使用小写k的类名
                base_name = parts[0]
                if base_name == 'FB15K237Inductive':
                    actual_dataset_name = 'FB15k237Inductive'
                elif base_name in ['WKIngram', 'NLIngram']:
                    # WKIngram和NLIngram需要版本号
                    actual_dataset_name = base_name
                elif base_name.startswith('WikiTopicsMT'):
                    # WikiTopicsMT系列需要版本号
                    actual_dataset_name = base_name
                else:
                    actual_dataset_name = base_name
            version = parts[1]
    
    # 获取数据集路径
    kg_datasets_path = flags.get('kg_datasets_path', '/T20030104/ynj/semma/kg-datasets')
    
    # 构建配置
    # root参数应该是kg_datasets_path（数据集会自动根据name在kg_datasets_path下查找）
    # 根据配置文件，root直接设置为kg_datasets_path
    cfg = {
        'dataset': {
            'class': actual_dataset_name,
            'root': kg_datasets_path,  # 使用kg_datasets_path作为root
        },
        'task': {
            'name': 'InductiveInference' if dataset_type != 'transductive' else 'LinkPrediction'
        }
    }
    
    # 如果有版本号，添加到配置中
    if version:
        cfg['dataset']['version'] = version
    
    try:
        # 将字典转换为对象，以便 build_dataset 可以访问 cfg.dataset
        cfg_obj = SimpleNamespace()
        cfg_obj.dataset = cfg['dataset']
        cfg_obj.task = cfg.get('task', {})
        
        dataset = util.build_dataset(cfg_obj)
        # 数据集通过索引访问：dataset[0] = train, dataset[1] = valid, dataset[2] = test
        train_data = dataset[0]
        valid_data = dataset[1]
        test_data = dataset[2]
        return dataset, train_data, valid_data, test_data
    except Exception as e:
        print(f"❌ 加载数据集失败 {dataset_name} (实际类名: {actual_dataset_name}): {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())
        return None, None, None, None

def load_model(checkpoint_path, dataset, device):
    """加载训练好的模型"""
    flags = load_flags()
    
    # ARE就是EnhanceUltra模型，所以使用EnhancedUltra
    # 但也要检查flags中的设置
    model_type = flags.get('run', 'semma')
    
    # 如果flags中不是EnhancedUltra，但我们要分析ARE，则使用EnhancedUltra
    # 因为用户明确说ARE指的是EnhanceUltra模型
    use_enhanced = True  # 分析ARE，所以总是使用EnhancedUltra
    
    # 使用配置构建模型（简化版，实际应该从配置文件读取）
    # 这里我们尝试直接加载checkpoint，如果失败则使用默认配置
    model = None
    
    # 尝试加载checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location='cpu')
            
            # 尝试从state中推断模型结构
            if 'model' in state:
                state_dict = state['model']
            else:
                state_dict = state
            
            # 检查是否是EnhancedUltra（ARE就是EnhanceUltra）
            if 'similarity_enhancer' in state_dict or use_enhanced or model_type == 'EnhancedUltra':
                from ultra.enhanced_models import EnhancedUltra
                # 需要从state_dict推断配置，这里使用默认配置
                model = EnhancedUltra(
                    rel_model_cfg={'class': 'RelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations},
                    entity_model_cfg={'class': 'EntityNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64]},
                    sem_model_cfg={'class': 'SemRelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations}
                )
            else:
                from ultra.models import Ultra
                model = Ultra(
                    rel_model_cfg={'class': 'RelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations},
                    entity_model_cfg={'class': 'EntityNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64]},
                    sem_model_cfg={'class': 'SemRelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations}
                )
            
            # 加载权重
            if 'model' in state:
                model.load_state_dict(state['model'], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print(f"✅ 加载模型checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  加载checkpoint失败: {e}，使用随机初始化的模型")
            # 如果加载失败，创建默认模型（使用EnhancedUltra，因为ARE就是EnhanceUltra）
            if use_enhanced or model_type == 'EnhancedUltra':
                from ultra.enhanced_models import EnhancedUltra
                model = EnhancedUltra(
                    rel_model_cfg={'class': 'RelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations},
                    entity_model_cfg={'class': 'EntityNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64]},
                    sem_model_cfg={'class': 'SemRelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations}
                )
            else:
                from ultra.models import Ultra
                model = Ultra(
                    rel_model_cfg={'class': 'RelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations},
                    entity_model_cfg={'class': 'EntityNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64]},
                    sem_model_cfg={'class': 'SemRelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations}
                )
    else:
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}，使用随机初始化的模型")
        # 创建默认模型（使用EnhancedUltra，因为ARE就是EnhanceUltra）
        if use_enhanced or model_type == 'EnhancedUltra':
            from ultra.enhanced_models import EnhancedUltra
            model = EnhancedUltra(
                rel_model_cfg={'class': 'RelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations},
                entity_model_cfg={'class': 'EntityNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64]},
                sem_model_cfg={'class': 'SemRelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations}
            )
        else:
            from ultra.models import Ultra
            model = Ultra(
                rel_model_cfg={'class': 'RelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations},
                entity_model_cfg={'class': 'EntityNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64]},
                sem_model_cfg={'class': 'SemRelNBFNet', 'input_dim': 64, 'hidden_dims': [64, 64], 'num_relations': dataset[0].num_relations}
            )
    
    model = model.to(device)
    model.eval()
    return model

def find_similar_relations(model, data, query_rel_idx, threshold=0.8, device='cuda'):
    """
    找到与查询关系相似的关系
    
    Returns:
        similar_rels: list of (rel_idx, similarity) tuples
    """
    model.eval()
    with torch.no_grad():
        try:
            # 确保data在正确的设备上
            if hasattr(data, 'to'):
                data = data.to(device)
            elif hasattr(data, 'keys'):
                # 如果data是Data对象，需要移动所有tensor属性
                for key in data.keys:
                    if isinstance(getattr(data, key), torch.Tensor):
                        setattr(data, key, getattr(data, key).to(device))
            # 如果data已经是torch_geometric.data.Data对象，它应该已经有to方法
            
            # 获取关系表示
            query_rels = torch.tensor([query_rel_idx], device=device)
            
            # 获取关系表示 - 对于Ultra和EnhancedUltra模型
            if hasattr(model, 'relation_model'):
                relation_reprs = model.relation_model(data, query=query_rels)
            elif hasattr(model, 'model') and hasattr(model.model, 'relation_model'):
                # 对于UltraQuery等包装模型
                relation_reprs = model.model.relation_model(data, query=query_rels)
            else:
                return []
            
            if relation_reprs is None or relation_reprs.shape[0] == 0:
                return []
            
            # relation_reprs shape: [batch_size, num_relations, embedding_dim]
            # 获取查询关系的表示
            query_repr = relation_reprs[0, query_rel_idx, :]  # [embedding_dim]
            all_reprs = relation_reprs[0, :, :]  # [num_relations, embedding_dim]
            
            # 计算余弦相似度
            query_norm = F.normalize(query_repr, p=2, dim=0)
            all_norms = F.normalize(all_reprs, p=2, dim=1)
            similarities = torch.matmul(query_norm.unsqueeze(0), all_norms.t()).squeeze(0)
            
            # 排除查询关系本身
            similarities[query_rel_idx] = -1.0
            
            # 找到相似度大于阈值的关系
            above_threshold = similarities > threshold
            valid_indices = torch.where(above_threshold)[0]
            
            similar_rels = [(idx.item(), similarities[idx].item()) for idx in valid_indices]
            similar_rels.sort(key=lambda x: x[1], reverse=True)
            
            return similar_rels
        except Exception as e:
            print(f"⚠️  查找相似关系时出错: {e}")
            return []

def check_reference_effectiveness(similar_rels, train_triples, test_triple, entity_vocab, relation_vocab):
    """
    检查相似关系的有效性（改进版：使用有效性分数进行量化）
    
    Args:
        similar_rels: list of (rel_idx, similarity) tuples
        train_triples: list of (h, r, t) tuples from training data
        test_triple: (h, r, t) tuple from test data
        entity_vocab: entity vocabulary
        relation_vocab: relation vocabulary
    
    Returns:
        effective_refs: list of (rel_idx, similarity, effectiveness_score, reason) tuples
        noise_refs: list of (rel_idx, similarity) tuples
    """
    test_h, test_r, test_t = test_triple
    
    # 构建训练数据的关系-实体对集合（用于快速查找）
    # 对于tail预测：(h, r) -> {t1, t2, ...}
    # 对于head预测：(r, t) -> {h1, h2, ...}
    train_tail_contexts = defaultdict(set)  # (h, r) -> {tails}
    train_head_contexts = defaultdict(set)  # (r, t) -> {heads}
    
    for h, r, t in train_triples:
        train_tail_contexts[(h, r)].add(t)
        train_head_contexts[(r, t)].add(h)
    
    effective_refs = []
    noise_refs = []
    
    # 有效性阈值：只有分数 >= 0.3 才算有效
    effectiveness_threshold = 0.3
    
    # 检查每个相似关系
    for rel_idx, similarity in similar_rels:
        tail_score = 0.0
        head_score = 0.0
        tail_reason = ""
        head_reason = ""
        
        # ========== Tail预测有效性评分 ==========
        if (test_h, rel_idx) in train_tail_contexts:
            candidate_tails = train_tail_contexts[(test_h, rel_idx)]
            candidate_size = len(candidate_tails)
            
            # 情况1：直接匹配（最高分）
            if test_t in candidate_tails:
                # 直接匹配：分数 = 1.0 * 相似度权重
                # 候选集合越小，匹配价值越高（信息更精确）
                size_factor = 1.0 / (1.0 + np.log10(max(candidate_size, 1)))
                tail_score = 1.0 * similarity * size_factor
                tail_reason = f"direct_match(size={candidate_size})"
            else:
                # 情况2：提供候选集合但test_t不在其中
                # 分数 = 基础分 * 相似度权重 * 候选集合质量
                if candidate_size > 0:
                    # 候选集合越小，价值越高（更精确的约束）
                    # 但因为没有直接匹配，分数较低
                    size_factor = 1.0 / (1.0 + np.log10(candidate_size))
                    # 基础分：0.4（提供上下文但未直接匹配）
                    base_score = 0.4
                    tail_score = base_score * similarity * size_factor
                    tail_reason = f"context_only(size={candidate_size})"
        
        # ========== Head预测有效性评分 ==========
        if (rel_idx, test_t) in train_head_contexts:
            candidate_heads = train_head_contexts[(rel_idx, test_t)]
            candidate_size = len(candidate_heads)
            
            # 情况1：直接匹配（最高分）
            if test_h in candidate_heads:
                size_factor = 1.0 / (1.0 + np.log10(max(candidate_size, 1)))
                head_score = 1.0 * similarity * size_factor
                head_reason = f"direct_match(size={candidate_size})"
            else:
                # 情况2：提供候选集合但test_h不在其中
                if candidate_size > 0:
                    size_factor = 1.0 / (1.0 + np.log10(candidate_size))
                    base_score = 0.4
                    head_score = base_score * similarity * size_factor
                    head_reason = f"context_only(size={candidate_size})"
        
        # 选择tail和head中分数更高的作为最终分数
        max_score = max(tail_score, head_score)
        best_reason = tail_reason if tail_score >= head_score else head_reason
        direction = 'tail' if tail_score >= head_score else 'head'
        
        # 判断是否有效
        if max_score >= effectiveness_threshold:
            effective_refs.append((rel_idx, similarity, max_score, direction, best_reason))
        else:
            noise_refs.append((rel_idx, similarity, max_score))  # 也记录分数，便于分析
    
    return effective_refs, noise_refs

def find_checkpoint_path(dataset_name, base_checkpoint='ckpts/optuna_1.pth'):
    """
    查找数据集的checkpoint路径
    
    Args:
        dataset_name: 数据集名称
        base_checkpoint: 基础checkpoint路径
    
    Returns:
        checkpoint_path: checkpoint路径
    """
    flags = load_flags()
    base_path = flags.get('base_path', '/T20030104/ynj/semma')
    
    # 首先尝试使用基础checkpoint
    base_ckpt_path = os.path.join(base_path, base_checkpoint)
    if os.path.exists(base_ckpt_path):
        return base_ckpt_path
    
    # 尝试在optuna_1_output中查找数据集特定的checkpoint
    output_dir = os.path.join(base_path, 'optuna_1_output', 'Ultra')
    
    # 数据集名称映射（与load_dataset中的映射保持一致）
    dataset_name_mapping = {
        'YAGO310-ht': 'YAGO310',
        'ConceptNet 100k-ht': 'ConceptNet100k',
        'WDsinger-ht': 'WDsinger',
        'AristoV4-ht': 'AristoV4',
        'FB15K237Inductive:v1': 'FB15k237Inductive',
        'FB15K237Inductive:v2': 'FB15k237Inductive',
        'FB15K237Inductive:v3': 'FB15k237Inductive',
        'FB15K237Inductive:v4': 'FB15k237Inductive',
        'WN18RRInductive:v3': 'WN18RRInductive',
        'NELLInductive:v1': 'NELLInductive',
        'NELLInductive:v3': 'NELLInductive',
        'NELLInductive:v4': 'NELLInductive',
        'WKIngram:25': 'WKIngram',
        'NLIngram:25': 'NLIngram',
        'NLIngram:75': 'NLIngram',
        'Metafam': 'Metafam',
        'WikiTopicsMT1:health': 'WikiTopicsMT1',
        'WikiTopicsMT3:infra': 'WikiTopicsMT3',
    }
    
    # 如果不在映射中，尝试处理版本号
    if dataset_name not in dataset_name_mapping and ':' in dataset_name:
        parts = dataset_name.split(':')
        if len(parts) == 2:
            base_name = parts[0]
            if base_name == 'FB15K237Inductive':
                mapped_name = 'FB15k237Inductive'
            elif base_name in ['WKIngram', 'NLIngram']:
                mapped_name = base_name
            elif base_name.startswith('WikiTopicsMT'):
                mapped_name = base_name
            else:
                mapped_name = dataset_name_mapping.get(dataset_name, dataset_name)
        else:
            mapped_name = dataset_name_mapping.get(dataset_name, dataset_name)
    else:
        mapped_name = dataset_name_mapping.get(dataset_name, dataset_name)
    
    # 查找数据集文件夹
    dataset_output_dir = os.path.join(output_dir, mapped_name)
    if os.path.exists(dataset_output_dir):
        # 查找最新的checkpoint文件
        import glob
        ckpt_files = glob.glob(os.path.join(dataset_output_dir, '**', '*.pth'), recursive=True)
        if ckpt_files:
            # 返回最新的checkpoint
            ckpt_files.sort(key=os.path.getmtime, reverse=True)
            return ckpt_files[0]
    
    # 如果都找不到，返回基础checkpoint路径（即使不存在）
    return base_ckpt_path

def analyze_dataset_samples(dataset_name, dataset_type, checkpoint_path=None, num_samples=None, device='cuda'):
    """
    分析数据集样本级别的相似关系参考情况
    
    Args:
        dataset_name: 数据集名称
        dataset_type: 数据集类型
        checkpoint_path: 模型checkpoint路径（如果为None，会自动查找）
        num_samples: 分析的样本数量（如果为None，分析所有测试样本）
        device: 设备
    """
    print(f"\n{'='*70}")
    print(f"📊 分析数据集: {dataset_name} ({dataset_type})")
    print(f"{'='*70}")
    
    # 加载数据集
    dataset, train_data, valid_data, test_data = load_dataset(dataset_name, dataset_type)
    if dataset is None:
        return None
    
    # 查找checkpoint路径
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint_path(dataset_name)
        print(f"📁 使用checkpoint: {checkpoint_path}")
    
    # 加载模型（使用EnhancedUltra，因为ARE就是EnhanceUltra）
    model = load_model(checkpoint_path, dataset, device)
    
    # 确保数据在正确的设备上
    test_data = test_data.to(device)
    train_data = train_data.to(device)
    
    # 获取测试三元组
    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    total_test_samples = len(test_triplets)
    
    # 限制样本数量（如果指定了num_samples）
    if num_samples is not None and total_test_samples > num_samples:
        indices = torch.randperm(total_test_samples)[:num_samples]
        test_triplets = test_triplets[indices]
        print(f"⚠️  限制分析样本数量为: {num_samples} (总测试样本数: {total_test_samples})")
    else:
        print(f"📊 分析所有测试样本: {total_test_samples} 个")
    
    # 获取训练三元组（用于有效性检查）
    train_triplets = torch.cat([train_data.edge_index, train_data.edge_type.unsqueeze(0)]).t()
    train_triples_list = [(int(h.item()), int(r.item()), int(t.item())) for h, r, t in train_triplets]
    
    # 构建实体和关系词汇表（简化版，使用索引）
    num_entities = dataset[0].num_nodes
    num_relations = dataset[0].num_relations
    entity_vocab = {i: i for i in range(num_entities)}
    relation_vocab = {i: i for i in range(num_relations)}
    
    # 统计数据
    stats = {
        'total_samples': 0,
        'samples_with_references': 0,
        'total_references': 0,
        'effective_references': 0,
        'noise_references': 0,
        'samples_with_effective_refs': 0,
        'samples_with_only_noise': 0,
        'total_effectiveness_score': 0.0,  # 总有效性分数
        'direct_match_count': 0,  # 直接匹配的数量
        'context_only_count': 0,  # 仅提供上下文的数量
    }
    
    sample_results = []
    
    # 获取相似度阈值
    flags = load_flags()
    threshold = flags.get('similarity_threshold_init', 0.8)
    
    print(f"\n🔍 分析 {len(test_triplets)} 个测试样本...")
    
    for i, triplet in enumerate(tqdm(test_triplets, desc="分析样本")):
        h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
        
        stats['total_samples'] += 1
        
        # 找到相似关系（test_data已经在device上了）
        similar_rels = find_similar_relations(model, test_data, r, threshold=threshold, device=device)
        
        if len(similar_rels) == 0:
            sample_results.append({
                'sample_idx': i,
                'test_triple': (h, r, t),
                'num_references': 0,
                'num_effective': 0,
                'num_noise': 0,
                'has_references': False,
                'has_effective_refs': False,
                'has_only_noise': False,
            })
            continue
        
        stats['samples_with_references'] += 1
        stats['total_references'] += len(similar_rels)
        
        # 检查有效性
        effective_refs, noise_refs = check_reference_effectiveness(
            similar_rels, train_triples_list, (h, r, t), entity_vocab, relation_vocab
        )
        
        stats['effective_references'] += len(effective_refs)
        stats['noise_references'] += len(noise_refs)
        
        # 统计有效性分数和匹配类型
        for ref in effective_refs:
            # ref格式: (rel_idx, similarity, max_score, direction, reason)
            if len(ref) >= 5:
                score = ref[2]
                reason = ref[4]
                stats['total_effectiveness_score'] += score
                if 'direct_match' in reason:
                    stats['direct_match_count'] += 1
                elif 'context_only' in reason:
                    stats['context_only_count'] += 1
        
        # 也统计噪音的分数（虽然它们低于阈值）
        for ref in noise_refs:
            # ref格式: (rel_idx, similarity, max_score)
            if len(ref) >= 3:
                score = ref[2]
                stats['total_effectiveness_score'] += score  # 也计入总分，用于计算平均值
        
        if len(effective_refs) > 0:
            stats['samples_with_effective_refs'] += 1
        
        if len(effective_refs) == 0 and len(noise_refs) > 0:
            stats['samples_with_only_noise'] += 1
        
        sample_results.append({
            'sample_idx': i,
            'test_triple': (h, r, t),
            'num_references': len(similar_rels),
            'num_effective': len(effective_refs),
            'num_noise': len(noise_refs),
            'has_references': True,
            'has_effective_refs': len(effective_refs) > 0,
            'has_only_noise': len(effective_refs) == 0 and len(noise_refs) > 0,
            'effective_refs': effective_refs,
            'noise_refs': noise_refs,
        })
    
    # 计算统计指标
    if stats['total_samples'] > 0:
        stats['reference_rate'] = stats['samples_with_references'] / stats['total_samples']
        stats['effective_rate'] = stats['samples_with_effective_refs'] / stats['total_samples']
        stats['noise_rate'] = stats['samples_with_only_noise'] / stats['total_samples']
    
    if stats['total_references'] > 0:
        stats['reference_effectiveness'] = stats['effective_references'] / stats['total_references']
        stats['reference_noise_ratio'] = stats['noise_references'] / stats['total_references']
        # 平均有效性分数
        stats['avg_effectiveness_score'] = stats['total_effectiveness_score'] / stats['total_references']
    else:
        stats['avg_effectiveness_score'] = 0.0
    
    if stats['effective_references'] > 0:
        # 从sample_results中提取所有有效参考的分数
        all_effective_scores = []
        for sample_result in sample_results:
            if 'effective_refs' in sample_result:
                for ref in sample_result['effective_refs']:
                    if len(ref) >= 3:
                        all_effective_scores.append(ref[2])
        
        if len(all_effective_scores) > 0:
            stats['avg_effective_score'] = sum(all_effective_scores) / len(all_effective_scores)
        else:
            stats['avg_effective_score'] = 0.0
        
        # 直接匹配比例
        stats['direct_match_ratio'] = stats['direct_match_count'] / stats['effective_references']
        stats['context_only_ratio'] = stats['context_only_count'] / stats['effective_references']
    else:
        stats['avg_effective_score'] = 0.0
        stats['direct_match_ratio'] = 0.0
        stats['context_only_ratio'] = 0.0
    
    # 打印统计结果
    print(f"\n📈 统计结果:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  有参考的样本数: {stats['samples_with_references']} ({stats.get('reference_rate', 0)*100:.2f}%)")
    print(f"  总参考数: {stats['total_references']}")
    print(f"  有效参考数: {stats['effective_references']} ({stats.get('reference_effectiveness', 0)*100:.2f}%)")
    print(f"  噪音参考数: {stats['noise_references']} ({stats.get('reference_noise_ratio', 0)*100:.2f}%)")
    print(f"  有有效参考的样本数: {stats['samples_with_effective_refs']} ({stats.get('effective_rate', 0)*100:.2f}%)")
    print(f"  只有噪音的样本数: {stats['samples_with_only_noise']} ({stats.get('noise_rate', 0)*100:.2f}%)")
    print(f"\n📊 有效性量化指标:")
    print(f"  平均有效性分数: {stats.get('avg_effectiveness_score', 0):.4f} (所有参考)")
    print(f"  有效参考平均分数: {stats.get('avg_effective_score', 0):.4f} (仅有效参考)")
    print(f"  直接匹配数: {stats['direct_match_count']} ({stats.get('direct_match_ratio', 0)*100:.2f}% of effective)")
    print(f"  仅上下文数: {stats['context_only_count']} ({stats.get('context_only_ratio', 0)*100:.2f}% of effective)")
    
    return {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'stats': stats,
        'sample_results': sample_results,
    }

def visualize_results(all_results, output_dir='analyze/figures'):
    """可视化分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    datasets = []
    reference_rates = []
    effectiveness_rates = []
    noise_rates = []
    
    for result in all_results:
        if result is None:
            continue
        datasets.append(result['dataset_name'])
        stats = result['stats']
        reference_rates.append(stats.get('reference_rate', 0) * 100)
        effectiveness_rates.append(stats.get('reference_effectiveness', 0) * 100)
        noise_rates.append(stats.get('reference_noise_ratio', 0) * 100)
    
    if len(datasets) == 0:
        print("⚠️  没有数据可可视化")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 参考率对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(datasets)), reference_rates, color='skyblue')
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Reference Rate (%)', fontsize=12)
    ax1.set_title('Percentage of Samples with References', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(reference_rates):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. 有效性率对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(datasets)), effectiveness_rates, color='lightgreen')
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('Effectiveness Rate (%)', fontsize=12)
    ax2.set_title('Percentage of Effective References', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(effectiveness_rates):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 噪音率对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(datasets)), noise_rates, color='lightcoral')
    ax3.set_xlabel('Dataset', fontsize=12)
    ax3.set_ylabel('Noise Rate (%)', fontsize=12)
    ax3.set_title('Percentage of Noise References', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(datasets)))
    ax3.set_xticklabels(datasets, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(noise_rates):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. 综合对比（堆叠柱状图）
    ax4 = axes[1, 1]
    x = np.arange(len(datasets))
    width = 0.6
    effective_bars = ax4.bar(x, effectiveness_rates, width, label='Effective', color='lightgreen')
    noise_bars = ax4.bar(x, noise_rates, width, bottom=effectiveness_rates, label='Noise', color='lightcoral')
    ax4.set_xlabel('Dataset', fontsize=12)
    ax4.set_ylabel('Rate (%)', fontsize=12)
    ax4.set_title('Effective vs Noise References', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '28_sample_level_reference_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {output_path}")
    plt.close()

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analyze/output_log.txt'),
            logging.StreamHandler()
        ]
    )

    # 读取显著提升和下降的数据集
    csv_path = 'analyze/common_features_analysis.csv'
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # 筛选显著提升和下降的数据集，且只选择Inductive(e,r)类型
    improved = df[(df['performance_category'] == 'significantly_improved') & (df['dataset_type'] == 'Inductive(e,r)')]
    degraded = df[(df['performance_category'] == 'significantly_degraded') & (df['dataset_type'] == 'Inductive(e,r)')]
    
    print(f"📊 找到 {len(improved)} 个显著提升的Inductive(e,r)数据集")
    print(f"📊 找到 {len(degraded)} 个显著下降的Inductive(e,r)数据集")
    
    # 分析所有显著提升和下降的Inductive(e,r)数据集
    key_datasets = []
    
    # 显著提升的数据集（只包含Inductive(e,r)）
    for _, row in improved.iterrows():
        dataset_name = row['dataset']
        dataset_type = row['dataset_type']
        key_datasets.append((dataset_name, dataset_type, 'improved'))
    
    # 显著下降的数据集（只包含Inductive(e,r)）
    for _, row in degraded.iterrows():
        dataset_name = row['dataset']
        dataset_type = row['dataset_type']
        key_datasets.append((dataset_name, dataset_type, 'degraded'))
    
    # 额外添加用户指定的数据集（即使它们被标记为stable，但变化较大）
    additional_datasets = {
        'WikiTopicsMT2:sci': 'degraded',  # 下降较多
        'NLIngram:0': 'degraded',  # 下降较多
        'WikiTopicsMT1:tax': 'improved',  # 提升较多
        'WikiTopicsMT3:art': 'improved',  # 提升较多
    }
    
    for dataset_name, category in additional_datasets.items():
        # 检查数据集是否存在且是Inductive(e,r)类型
        dataset_row = df[(df['dataset'] == dataset_name) & (df['dataset_type'] == 'Inductive(e,r)')]
        if len(dataset_row) > 0:
            dataset_type = dataset_row.iloc[0]['dataset_type']
            # 检查是否已经添加过（避免重复）
            if (dataset_name, dataset_type, category) not in key_datasets:
                key_datasets.append((dataset_name, dataset_type, category))
                print(f"➕ 额外添加数据集: {dataset_name} ({category})")
    
    print(f"\n🔍 将分析所有 {len(key_datasets)} 个数据集")
    
    # 分析每个数据集
    all_results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for dataset_name, dataset_type, category in key_datasets:
        try:
            result = analyze_dataset_samples(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                checkpoint_path=None,  # 使用默认路径
                num_samples=None,  # 分析所有测试样本
                device=device
            )
            if result:
                result['category'] = category
                all_results.append(result)
        except Exception as e:
            print(f"❌ 分析数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exception(*sys.exc_info())
            continue
    
    # 可视化结果
    if len(all_results) > 0:
        visualize_results(all_results)
        
        # 保存详细结果
        output_csv = 'analyze/sample_level_reference_analysis.csv'
        rows = []
        for result in all_results:
            if result is None:
                continue
            stats = result['stats']
            rows.append({
                'dataset': result['dataset_name'],
                'category': result.get('category', 'unknown'),
                'total_samples': stats['total_samples'],
                'samples_with_references': stats['samples_with_references'],
                'reference_rate': stats.get('reference_rate', 0),
                'total_references': stats['total_references'],
                'effective_references': stats['effective_references'],
                'reference_effectiveness': stats.get('reference_effectiveness', 0),
                'noise_references': stats['noise_references'],
                'reference_noise_ratio': stats.get('reference_noise_ratio', 0),
                'samples_with_effective_refs': stats['samples_with_effective_refs'],
                'effective_rate': stats.get('effective_rate', 0),
                'samples_with_only_noise': stats['samples_with_only_noise'],
                'noise_rate': stats.get('noise_rate', 0),
                # 新增量化指标
                'avg_effectiveness_score': stats.get('avg_effectiveness_score', 0),
                'avg_effective_score': stats.get('avg_effective_score', 0),
                'direct_match_count': stats.get('direct_match_count', 0),
                'direct_match_ratio': stats.get('direct_match_ratio', 0),
                'context_only_count': stats.get('context_only_count', 0),
                'context_only_ratio': stats.get('context_only_ratio', 0),
            })
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv(output_csv, index=False)
        print(f"\n✅ 详细结果已保存: {output_csv}")
    else:
        print("\n❌ 没有获得任何分析结果")

if __name__ == '__main__':
    main()

