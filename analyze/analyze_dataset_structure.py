#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据集的关系结构化程度
从实际数据文件中提取统计特征，量化判断关系是否高度结构化
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from scipy.spatial.distance import cosine
import yaml

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")

def load_flags():
    """加载配置文件"""
    flags_path = Path(__file__).parent.parent / "flags.yaml"
    with open(flags_path, 'r') as f:
        flags = yaml.safe_load(f)
    return flags

def get_dataset_path_mapping():
    """获取数据集名称到路径的映射"""
    mapping = {
        'Metafam': 'metafam',
        'YAGO310-ht': 'yago310',
        'YAGO310': 'yago310',
        'ConceptNet 100k-ht': 'cnet100k',
        'ConceptNet100k': 'cnet100k',
        'FB15K237': 'FB15k-237',
        'FB15K-237': 'FB15k-237',
        'WN18RR': 'wn18rr',
        'NELL995-ht': 'nell995',
        'NELL995': 'nell995',
        'CoDExSmall-ht': 'codex-s',
        'CoDExLarge-ht': 'codex-l',
        'CoDExMedium': 'codex-m',
        'DBpedia 100k-ht': 'dbp100k',
        'NELL23k-ht': 'NELL23K',
        'WDsinger-ht': 'WD-singer',
        'WD-singer': 'WD-singer',
        'AristoV4-ht': 'aristov4',
        'Hetionet-ht': 'hetionet',
    }
    return mapping

def find_dataset_raw_dir(kg_datasets_path, dataset_name):
    """查找数据集的raw目录"""
    mapping = get_dataset_path_mapping()
    
    # 首先尝试映射
    mapped_name = mapping.get(dataset_name, dataset_name)
    
    # 尝试不同的路径格式
    possible_names = [
        mapped_name,
        dataset_name,
        dataset_name.lower(),
        dataset_name.lower().replace('-', ''),
        dataset_name.lower().replace(' ', ''),
        mapped_name.lower(),
    ]
    
    possible_paths = []
    for name in possible_names:
        # 直接路径
        possible_paths.append(os.path.join(kg_datasets_path, name, "raw"))
        # 在grail子目录中（Inductive数据集）
        possible_paths.append(os.path.join(kg_datasets_path, "grail", f"Ind{name}", "v1", "raw"))
        possible_paths.append(os.path.join(kg_datasets_path, "grail", name, "v1", "raw"))
    
    # 递归搜索
    for root, dirs, files in os.walk(kg_datasets_path):
        if 'raw' in dirs and 'train.txt' in os.listdir(os.path.join(root, 'raw')):
            # 检查是否匹配数据集名称
            dir_name = os.path.basename(root)
            if any(name.lower() in dir_name.lower() or dir_name.lower() in name.lower() 
                   for name in possible_names if name):
                possible_paths.append(os.path.join(root, 'raw'))
    
    # 去重并检查存在性
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "train.txt")):
            return path
    
    return None

def load_dataset_triples(dataset_name, dataset_type="transductive"):
    """
    加载数据集的三元组文件
    
    Args:
        dataset_name: 数据集名称
        dataset_type: 数据集类型 (transductive, inductive, etc.)
    
    Returns:
        train_triples, valid_triples, test_triples: 三元组列表 [(h, r, t), ...]
        entity_vocab, relation_vocab: 词汇表
    """
    flags = load_flags()
    kg_datasets_path = flags.get('kg_datasets_path', '/T20030104/ynj/semma/kg-datasets')
    
    raw_dir = find_dataset_raw_dir(kg_datasets_path, dataset_name)
    
    if raw_dir is None:
        print(f"⚠️  Warning: Cannot find raw directory for {dataset_name}")
        return None, None, None, None, None
    
    # 加载三元组文件
    def load_triples_file(filepath):
        if not os.path.exists(filepath):
            return []
        triples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    triples.append((h, r, t))
        return triples
    
    train_file = os.path.join(raw_dir, "train.txt")
    valid_file = os.path.join(raw_dir, "valid.txt")
    test_file = os.path.join(raw_dir, "test.txt")
    
    train_triples = load_triples_file(train_file)
    valid_triples = load_triples_file(valid_file)
    test_triples = load_triples_file(test_file)
    
    # 构建词汇表
    all_triples = train_triples + valid_triples + test_triples
    entities = set()
    relations = set()
    
    for h, r, t in all_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    
    entity_vocab = {e: i for i, e in enumerate(sorted(entities))}
    relation_vocab = {r: i for i, r in enumerate(sorted(relations))}
    
    return train_triples, valid_triples, test_triples, entity_vocab, relation_vocab

def calculate_gini_coefficient(values):
    """计算基尼系数（衡量分布的不均匀程度）"""
    if len(values) == 0:
        return 0.0
    values = np.array(values)
    values = values.flatten()
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

def calculate_entropy(values):
    """计算熵（衡量分布的均匀程度）"""
    if len(values) == 0:
        return 0.0
    values = np.array(values)
    values = values[values > 0]  # 只考虑非零值
    if len(values) == 0:
        return 0.0
    probs = values / np.sum(values)
    return -np.sum(probs * np.log2(probs + 1e-10))

def analyze_relation_structure(dataset_name, triples_list, entity_vocab, relation_vocab):
    """
    分析关系的结构化程度
    
    Returns:
        metrics: 包含各种指标的字典
    """
    all_triples = []
    for triples in triples_list:
        all_triples.extend(triples)
    
    if len(all_triples) == 0:
        return None
    
    # 1. 关系频率统计
    relation_counts = Counter([r for _, r, _ in all_triples])
    relation_frequencies = list(relation_counts.values())
    
    # 2. 关系频率分布指标
    gini_coefficient = calculate_gini_coefficient(relation_frequencies)
    entropy = calculate_entropy(relation_frequencies)
    
    # 3. 关系-实体比例
    num_relations = len(relation_vocab)
    num_entities = len(entity_vocab)
    relation_entity_ratio = num_relations / num_entities if num_entities > 0 else 0
    
    # 4. 关系的平均频率
    avg_relation_freq = np.mean(relation_frequencies) if len(relation_frequencies) > 0 else 0
    std_relation_freq = np.std(relation_frequencies) if len(relation_frequencies) > 0 else 0
    cv_relation_freq = std_relation_freq / avg_relation_freq if avg_relation_freq > 0 else 0  # 变异系数
    
    # 5. 关系的长尾分布程度（前10%的关系占总频率的比例）
    sorted_freqs = sorted(relation_frequencies, reverse=True)
    top_10_percent = int(max(1, len(sorted_freqs) * 0.1))
    top_10_percent_freq = sum(sorted_freqs[:top_10_percent])
    total_freq = sum(sorted_freqs)
    top_10_percent_ratio = top_10_percent_freq / total_freq if total_freq > 0 else 0
    
    # 6. 关系的共现模式（两个关系同时出现在同一个实体对上的频率）
    # 简化：计算关系的平均邻居关系数
    entity_relations = defaultdict(set)
    for h, r, t in all_triples:
        entity_relations[h].add(r)
        entity_relations[t].add(r)
    
    relation_cooccurrence = defaultdict(set)
    for entity, rels in entity_relations.items():
        for r1 in rels:
            for r2 in rels:
                if r1 != r2:
                    relation_cooccurrence[r1].add(r2)
    
    avg_cooccurrence = np.mean([len(rels) for rels in relation_cooccurrence.values()]) if len(relation_cooccurrence) > 0 else 0
    
    # 7. 图的密度
    num_edges = len(all_triples)
    max_possible_edges = num_entities * num_entities
    graph_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    # 8. 关系的平均度（每个关系平均连接多少实体对）
    relation_degrees = defaultdict(int)
    for _, r, _ in all_triples:
        relation_degrees[r] += 1
    avg_relation_degree = np.mean(list(relation_degrees.values())) if len(relation_degrees) > 0 else 0
    
    metrics = {
        'dataset_name': dataset_name,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'num_triples': len(all_triples),
        'relation_entity_ratio': relation_entity_ratio,
        'gini_coefficient': gini_coefficient,  # 越高越不均匀（可能更结构化）
        'entropy': entropy,  # 越高越均匀（可能更不结构化）
        'avg_relation_freq': avg_relation_freq,
        'cv_relation_freq': cv_relation_freq,  # 变异系数，越高越不均匀
        'top_10_percent_ratio': top_10_percent_ratio,  # 长尾分布程度
        'avg_cooccurrence': avg_cooccurrence,
        'graph_density': graph_density,
        'avg_relation_degree': avg_relation_degree,
    }
    
    return metrics

def classify_structure_level(metrics):
    """
    根据指标分类结构化程度
    
    Returns:
        structure_level: 'high', 'medium', 'low'
        reasoning: 分类理由
    """
    if metrics is None:
        return 'unknown', 'No data available'
    
    # 综合多个指标判断
    scores = []
    reasons = []
    
    # 1. Gini系数（越高越结构化，因为关系分布不均匀，说明有主导关系）
    gini = metrics['gini_coefficient']
    if gini > 0.7:
        scores.append(2)  # high
        reasons.append(f"High Gini coefficient ({gini:.3f}) indicates concentrated relation distribution")
    elif gini > 0.5:
        scores.append(1)  # medium
        reasons.append(f"Medium Gini coefficient ({gini:.3f})")
    else:
        scores.append(0)  # low
        reasons.append(f"Low Gini coefficient ({gini:.3f}) indicates uniform relation distribution")
    
    # 2. 变异系数（越高越结构化）
    cv = metrics['cv_relation_freq']
    if cv > 1.0:
        scores.append(2)
        reasons.append(f"High coefficient of variation ({cv:.3f})")
    elif cv > 0.5:
        scores.append(1)
        reasons.append(f"Medium coefficient of variation ({cv:.3f})")
    else:
        scores.append(0)
        reasons.append(f"Low coefficient of variation ({cv:.3f})")
    
    # 3. 长尾分布（越高越结构化）
    top10 = metrics['top_10_percent_ratio']
    if top10 > 0.6:
        scores.append(2)
        reasons.append(f"High top-10% ratio ({top10:.3f}) indicates long-tail distribution")
    elif top10 > 0.4:
        scores.append(1)
        reasons.append(f"Medium top-10% ratio ({top10:.3f})")
    else:
        scores.append(0)
        reasons.append(f"Low top-10% ratio ({top10:.3f})")
    
    # 4. 关系-实体比例（越低可能越结构化，因为关系类型集中）
    ratio = metrics['relation_entity_ratio']
    if ratio < 0.01:
        scores.append(2)
        reasons.append(f"Low relation-entity ratio ({ratio:.4f}) indicates concentrated relation types")
    elif ratio < 0.05:
        scores.append(1)
        reasons.append(f"Medium relation-entity ratio ({ratio:.4f})")
    else:
        scores.append(0)
        reasons.append(f"High relation-entity ratio ({ratio:.4f})")
    
    avg_score = np.mean(scores)
    
    if avg_score >= 1.5:
        level = 'high'
    elif avg_score >= 0.5:
        level = 'medium'
    else:
        level = 'low'
    
    reasoning = "; ".join(reasons)
    return level, reasoning

def analyze_multiple_datasets(dataset_names):
    """分析多个数据集"""
    results = []
    
    for dataset_name in dataset_names:
        print(f"\n📊 Analyzing {dataset_name}...")
        
        try:
            triples_list = load_dataset_triples(dataset_name)
            if triples_list[0] is None:
                print(f"   ⚠️  Skipping {dataset_name} (cannot load data)")
                continue
            
            train_triples, valid_triples, test_triples, entity_vocab, relation_vocab = triples_list
            
            metrics = analyze_relation_structure(
                dataset_name, 
                [train_triples, valid_triples, test_triples],
                entity_vocab,
                relation_vocab
            )
            
            if metrics:
                structure_level, reasoning = classify_structure_level(metrics)
                metrics['structure_level'] = structure_level
                metrics['reasoning'] = reasoning
                results.append(metrics)
                print(f"   ✅ Structure level: {structure_level}")
                print(f"   📈 Gini: {metrics['gini_coefficient']:.3f}, CV: {metrics['cv_relation_freq']:.3f}")
        except Exception as e:
            print(f"   ❌ Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def create_visualization(results_df):
    """创建可视化图表"""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    if len(results_df) == 0:
        print("⚠️  No data to visualize")
        return
    
    # 1. 结构化程度分类分布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 Gini系数 vs 变异系数
    ax1 = axes[0, 0]
    for level in ['high', 'medium', 'low']:
        data = results_df[results_df['structure_level'] == level]
        if len(data) > 0:
            ax1.scatter(data['gini_coefficient'], data['cv_relation_freq'], 
                       label=f'{level.title()} Structure', s=100, alpha=0.7)
            for _, row in data.iterrows():
                ax1.annotate(row['dataset_name'], 
                           (row['gini_coefficient'], row['cv_relation_freq']),
                           fontsize=8, alpha=0.7)
    ax1.set_xlabel('Gini Coefficient', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Coefficient of Variation', fontsize=11, fontweight='bold')
    ax1.set_title('Relation Structure: Gini vs CV', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1.2 长尾分布 vs 关系-实体比例
    ax2 = axes[0, 1]
    for level in ['high', 'medium', 'low']:
        data = results_df[results_df['structure_level'] == level]
        if len(data) > 0:
            ax2.scatter(data['top_10_percent_ratio'], data['relation_entity_ratio'], 
                       label=f'{level.title()} Structure', s=100, alpha=0.7)
            for _, row in data.iterrows():
                ax2.annotate(row['dataset_name'], 
                           (row['top_10_percent_ratio'], row['relation_entity_ratio']),
                           fontsize=8, alpha=0.7)
    ax2.set_xlabel('Top-10% Frequency Ratio', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relation-Entity Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Relation Structure: Long-tail vs Ratio', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 1.3 结构化程度分布
    ax3 = axes[1, 0]
    structure_counts = results_df['structure_level'].value_counts()
    colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
    bars = ax3.bar(structure_counts.index, structure_counts.values, 
                   color=[colors.get(x, '#95a5a6') for x in structure_counts.index],
                   alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Structure Level', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Structure Levels', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 1.4 关键指标对比
    ax4 = axes[1, 1]
    structure_levels = ['high', 'medium', 'low']
    metrics_to_compare = ['gini_coefficient', 'cv_relation_freq', 'top_10_percent_ratio']
    x = np.arange(len(structure_levels))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_compare):
        values = [results_df[results_df['structure_level'] == level][metric].mean() 
                 for level in structure_levels]
        ax4.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.7)
    
    ax4.set_xlabel('Structure Level', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average Metric Value', fontsize=11, fontweight='bold')
    ax4.set_title('Key Metrics by Structure Level', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(structure_levels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '25_dataset_structure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to {output_dir / '25_dataset_structure_analysis.png'}")

if __name__ == "__main__":
    # 从之前的分析中获取关键数据集名称（使用data.md中的实际名称）
    key_datasets = [
        'Metafam',
        'YAGO310-ht',
        'ConceptNet 100k-ht',
        'FB15K237',
        'WN18RR',
        'NELL995-ht',
        'CoDExSmall-ht',
        'CoDExLarge-ht',
        'NELL23k-ht',
        'WDsinger-ht',
        'AristoV4-ht',
    ]
    
    print("🔍 Analyzing dataset structure levels...")
    print(f"📋 Analyzing {len(key_datasets)} datasets...")
    
    results = analyze_multiple_datasets(key_datasets)
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        
        # 保存结果
        output_file = Path(__file__).parent / "dataset_structure_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to {output_file}")
        
        # 打印结果
        print("\n📊 Dataset Structure Analysis Results:")
        print("=" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['dataset_name']}:")
            print(f"  Structure Level: {row['structure_level'].upper()}")
            print(f"  Gini Coefficient: {row['gini_coefficient']:.3f}")
            print(f"  CV: {row['cv_relation_freq']:.3f}")
            print(f"  Top-10% Ratio: {row['top_10_percent_ratio']:.3f}")
            print(f"  Relation-Entity Ratio: {row['relation_entity_ratio']:.4f}")
            print(f"  Reasoning: {row['reasoning']}")
        
        # 创建可视化
        print("\n📈 Creating visualizations...")
        create_visualization(results_df)
        
        print("\n🎉 Analysis completed!")
    else:
        print("\n⚠️  No results to save. Please check dataset paths.")

