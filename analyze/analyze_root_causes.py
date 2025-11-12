#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析ARE性能显著上升或下降的根本原因
通过数据集特征和性能变化的关系可视化展示
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def parse_data():
    """解析数据"""
    data_file = Path(__file__).parent / "data.md"
    
    datasets = []
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = [
        ("Pre-training", r"## 1\. Pre-training datasets\n(.*?)\n\n##"),
        ("Transductive", r"## 2\. Transductive datasets\n(.*?)\n\n---"),
        ("Inductive(e)", r"## 3\. Inductive\(e\) datasets\n(.*?)\n\n---"),
        ("Inductive(e,r)", r"## 4\. Inductive\(e,r\) datasets\n(.*?)\n\n#"),
    ]
    
    for section_name, pattern in sections:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            table_content = match.group(1)
            lines = table_content.strip().split('\n')
            for line in lines[2:]:
                if '|' in line and not line.startswith('|---'):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 7:
                        dataset = parts[0]
                        try:
                            semma_mrr = float(parts[3])
                            semma_h10 = float(parts[4])
                            are_mrr = float(parts[5])
                            are_h10 = float(parts[6])
                            
                            datasets.append({
                                'dataset': dataset,
                                'type': section_name,
                                'semma_mrr': semma_mrr,
                                'semma_h10': semma_h10,
                                'are_mrr': are_mrr,
                                'are_h10': are_h10,
                                'mrr_diff': are_mrr - semma_mrr,
                                'h10_diff': are_h10 - semma_h10,
                                'mrr_improvement_pct': (are_mrr - semma_mrr) / semma_mrr * 100 if semma_mrr > 0 else 0,
                            })
                        except ValueError:
                            continue
    
    return pd.DataFrame(datasets)

def classify_dataset_features(df):
    """根据数据集名称和类型分类特征"""
    
    # 定义数据集特征
    features = {
        'relation_structure': [],  # 关系结构化程度: 'high', 'medium', 'low'
        'domain_type': [],  # 领域类型: 'biology', 'common_sense', 'general', 'domain_specific'
        'data_scale': [],  # 数据规模: 'large', 'medium', 'small'
        'relation_similarity': [],  # 关系相似性: 'high', 'medium', 'low'
        'pretrain_match': [],  # 与预训练数据匹配度: 'high', 'medium', 'low'
    }
    
    for _, row in df.iterrows():
        dataset = row['dataset'].lower()
        dataset_type = row['type']
        
        # 关系结构化程度
        if any(x in dataset for x in ['metafam', 'yago', 'fb15k', 'wn18', 'codex']):
            features['relation_structure'].append('high')
        elif any(x in dataset for x in ['conceptnet', 'nell']):
            features['relation_structure'].append('low')
        else:
            features['relation_structure'].append('medium')
        
        # 领域类型
        if 'metafam' in dataset:
            features['domain_type'].append('biology')
        elif 'conceptnet' in dataset:
            features['domain_type'].append('common_sense')
        elif any(x in dataset for x in ['wikitopics', 'wiktopics']):
            features['domain_type'].append('domain_specific')
        else:
            features['domain_type'].append('general')
        
        # 数据规模（基于数据集名称推断）
        if any(x in dataset for x in ['yago', 'large', '100k', '310']):
            features['data_scale'].append('large')
        elif any(x in dataset for x in ['small', '23k', '995']):
            features['data_scale'].append('small')
        else:
            features['data_scale'].append('medium')
        
        # 关系相似性（基于领域类型推断）
        if any(x in dataset for x in ['metafam', 'yago', 'wn18', 'fb15k']):
            features['relation_similarity'].append('high')
        elif 'conceptnet' in dataset:
            features['relation_similarity'].append('low')
        else:
            features['relation_similarity'].append('medium')
        
        # 与预训练数据匹配度（FB15K237, WN18RR, CoDExMedium是预训练数据）
        if any(x in dataset for x in ['fb15k', 'wn18', 'codex']):
            features['pretrain_match'].append('high')
        elif 'conceptnet' in dataset or 'wikitopics' in dataset:
            features['pretrain_match'].append('low')
        else:
            features['pretrain_match'].append('medium')
    
    for key in features:
        df[key] = features[key]
    
    return df

def create_root_cause_analysis(df):
    """创建根本原因分析图表"""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 添加特征分类
    df = classify_dataset_features(df)
    
    # 1. 关系结构化程度 vs 性能变化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    structure_order = ['high', 'medium', 'low']
    structure_labels = ['High (Structured)', 'Medium', 'Low (Unstructured)']
    
    # MRR差异
    ax1 = axes[0]
    structure_mrr = [df[df['relation_structure'] == s]['mrr_diff'].mean() for s in structure_order]
    structure_mrr_std = [df[df['relation_structure'] == s]['mrr_diff'].std() for s in structure_order]
    
    bars1 = ax1.bar(structure_labels, structure_mrr, yerr=structure_mrr_std, 
                   color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, capsize=5)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Average MRR Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Change by Relation Structure', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(structure_mrr, structure_mrr_std)):
        ax1.text(i, mean + std + 0.005 if mean >= 0 else mean - std - 0.005, 
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=10)
    
    # H@10差异
    ax2 = axes[1]
    structure_h10 = [df[df['relation_structure'] == s]['h10_diff'].mean() for s in structure_order]
    structure_h10_std = [df[df['relation_structure'] == s]['h10_diff'].std() for s in structure_order]
    
    bars2 = ax2.bar(structure_labels, structure_h10, yerr=structure_h10_std, 
                   color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, capsize=5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Average H@10 Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Change by Relation Structure', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(structure_h10, structure_h10_std)):
        ax2.text(i, mean + std + 0.005 if mean >= 0 else mean - std - 0.005, 
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '7_relation_structure_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 领域类型 vs 性能变化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    domain_order = ['biology', 'general', 'domain_specific', 'common_sense']
    domain_labels = ['Biology\n(Metafam)', 'General\n(FB15K, YAGO)', 'Domain-Specific\n(WikiTopics)', 'Common Sense\n(ConceptNet)']
    
    # MRR差异
    ax1 = axes[0]
    domain_mrr = [df[df['domain_type'] == d]['mrr_diff'].mean() for d in domain_order]
    domain_mrr_std = [df[df['domain_type'] == d]['mrr_diff'].std() for d in domain_order]
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars1 = ax1.bar(domain_labels, domain_mrr, yerr=domain_mrr_std, 
                   color=colors, alpha=0.7, capsize=5)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Average MRR Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Change by Domain Type', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    
    for i, (mean, std) in enumerate(zip(domain_mrr, domain_mrr_std)):
        ax1.text(i, mean + std + 0.01 if mean >= 0 else mean - std - 0.01, 
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=10)
    
    # H@10差异
    ax2 = axes[1]
    domain_h10 = [df[df['domain_type'] == d]['h10_diff'].mean() for d in domain_order]
    domain_h10_std = [df[df['domain_type'] == d]['h10_diff'].std() for d in domain_order]
    
    bars2 = ax2.bar(domain_labels, domain_h10, yerr=domain_h10_std, 
                   color=colors, alpha=0.7, capsize=5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Average H@10 Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Change by Domain Type', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)
    
    for i, (mean, std) in enumerate(zip(domain_h10, domain_h10_std)):
        ax2.text(i, mean + std + 0.01 if mean >= 0 else mean - std - 0.01, 
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '8_domain_type_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 预训练数据匹配度 vs 性能变化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    match_order = ['high', 'medium', 'low']
    match_labels = ['High Match\n(FB15K, WN18RR)', 'Medium Match', 'Low Match\n(ConceptNet, WikiTopics)']
    
    # MRR差异
    ax1 = axes[0]
    match_mrr = [df[df['pretrain_match'] == m]['mrr_diff'].mean() for m in match_order]
    match_mrr_std = [df[df['pretrain_match'] == m]['mrr_diff'].std() for m in match_order]
    
    bars1 = ax1.bar(match_labels, match_mrr, yerr=match_mrr_std, 
                   color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, capsize=5)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Average MRR Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Change by Pretrain Data Match', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(match_mrr, match_mrr_std)):
        ax1.text(i, mean + std + 0.005 if mean >= 0 else mean - std - 0.005, 
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=10)
    
    # H@10差异
    ax2 = axes[1]
    match_h10 = [df[df['pretrain_match'] == m]['h10_diff'].mean() for m in match_order]
    match_h10_std = [df[df['pretrain_match'] == m]['h10_diff'].std() for m in match_order]
    
    bars2 = ax2.bar(match_labels, match_h10, yerr=match_h10_std, 
                   color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, capsize=5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Average H@10 Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Change by Pretrain Data Match', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(match_h10, match_h10_std)):
        ax2.text(i, mean + std + 0.005 if mean >= 0 else mean - std - 0.005, 
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '9_pretrain_match_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 关键数据集特征对比雷达图
    key_datasets = ['Metafam', 'YAGO310-ht', 'ConceptNet 100k-ht', 'WikiTopicsMT3:infra', 
                    'FB15K237Inductive:v2', 'NELLInductive:v1']
    
    key_df = df[df['dataset'].isin(key_datasets)].copy()
    
    if len(key_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 定义特征维度
        categories = ['Relation\nStructure', 'Domain\nMatch', 'Data\nScale', 
                     'Relation\nSimilarity', 'Pretrain\nMatch']
        
        # 为每个数据集创建特征向量
        for idx, (_, row) in enumerate(key_df.iterrows()):
            dataset_name = row['dataset']
            mrr_diff = row['mrr_diff']
            
            # 特征值（0-1之间）
            structure_val = {'high': 1.0, 'medium': 0.5, 'low': 0.0}[row['relation_structure']]
            domain_val = {'biology': 1.0, 'general': 0.7, 'domain_specific': 0.3, 'common_sense': 0.0}[row['domain_type']]
            scale_val = {'large': 1.0, 'medium': 0.5, 'small': 0.0}[row['data_scale']]
            similarity_val = {'high': 1.0, 'medium': 0.5, 'low': 0.0}[row['relation_similarity']]
            match_val = {'high': 1.0, 'medium': 0.5, 'low': 0.0}[row['pretrain_match']]
            
            values = [structure_val, domain_val, scale_val, similarity_val, match_val]
            
            # 角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合
            angles += angles[:1]
            
            # 根据性能变化选择颜色
            color = 'green' if mrr_diff > 0.01 else 'red' if mrr_diff < -0.01 else 'gray'
            alpha = 0.6 if abs(mrr_diff) > 0.01 else 0.3
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f"{dataset_name}\n(MRR: {mrr_diff:+.3f})", 
                   color=color, alpha=alpha)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.set_title('Key Datasets Feature Comparison\n(Green=Improved, Red=Degraded, Gray=Stable)', 
                    fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / '10_key_datasets_features_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 性能变化 vs SEMMA基础性能
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MRR
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['semma_mrr'], df['mrr_diff'], 
                          c=df['mrr_improvement_pct'], cmap='RdYlGn', 
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('SEMMA MRR (Base Performance)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MRR Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Change vs Base Performance (MRR)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Improvement %')
    
    # 标注关键数据集
    key_datasets_list = ['Metafam', 'YAGO310-ht', 'ConceptNet 100k-ht', 'WikiTopicsMT3:infra', 
                         'NELLInductive:v1', 'FB15K237Inductive:v2']
    for dataset in key_datasets_list:
        row = df[df['dataset'] == dataset]
        if len(row) > 0:
            row = row.iloc[0]
            if np.isfinite(row['semma_mrr']) and np.isfinite(row['mrr_diff']):
                ax1.annotate(dataset, (row['semma_mrr'], row['mrr_diff']), 
                            fontsize=8, alpha=0.7)
    
    # H@10
    ax2 = axes[1]
    scatter2 = ax2.scatter(df['semma_h10'], df['h10_diff'], 
                          c=df['mrr_improvement_pct'], cmap='RdYlGn', 
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('SEMMA H@10 (Base Performance)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('H@10 Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Change vs Base Performance (H@10)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Improvement %')
    
    for dataset in key_datasets_list:
        row = df[df['dataset'] == dataset]
        if len(row) > 0:
            row = row.iloc[0]
            if np.isfinite(row['semma_h10']) and np.isfinite(row['h10_diff']):
                ax2.annotate(dataset, (row['semma_h10'], row['h10_diff']), 
                            fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / '11_performance_vs_base.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 综合特征重要性分析
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 计算每个特征对性能变化的影响
    feature_impacts = {}
    
    for feature in ['relation_structure', 'domain_type', 'pretrain_match', 'relation_similarity']:
        if feature == 'domain_type':
            groups = ['biology', 'general', 'domain_specific', 'common_sense']
        elif feature == 'relation_structure':
            groups = ['high', 'medium', 'low']
        elif feature == 'pretrain_match':
            groups = ['high', 'medium', 'low']
        else:
            groups = ['high', 'medium', 'low']
        
        impacts = []
        for group in groups:
            group_data = df[df[feature] == group]['mrr_diff']
            if len(group_data) > 0:
                impacts.append(group_data.mean())
            else:
                impacts.append(0.0)
        
        # 确保所有特征有相同数量的组（用于热力图）
        feature_impacts[feature] = {
            'groups': groups,
            'impacts': impacts,
            'range': max(impacts) - min(impacts) if impacts else 0.0
        }
    
    # 按影响范围排序
    sorted_features = sorted(feature_impacts.items(), key=lambda x: x[1]['range'], reverse=True)
    
    # 创建热力图
    heatmap_data = []
    feature_names = []
    group_names = []
    
    for feature, data in sorted_features:
        feature_label = feature.replace('_', ' ').title()
        for i, (group, impact) in enumerate(zip(data['groups'], data['impacts'])):
            heatmap_data.append([impact])
            feature_names.append(feature_label)
            group_names.append(group)
    
    # 重新组织数据 - 分别处理不同组数的特征
    # 只使用有3个组的特征（relation_structure, pretrain_match, relation_similarity）
    three_group_features = [(f, d) for f, d in sorted_features if len(d['groups']) == 3]
    
    if len(three_group_features) > 0:
        heatmap_matrix = np.array([data['impacts'] for _, data in three_group_features])
        row_labels = [f.replace('_', ' ').title() for f, _ in three_group_features]
        col_labels = three_group_features[0][1]['groups']
        
        sns.heatmap(heatmap_matrix, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                   xticklabels=col_labels, yticklabels=row_labels,
                   cbar_kws={'label': 'Average MRR Difference'}, ax=ax)
    else:
        # 如果没有3个组的特征，创建一个简单的条形图
        ax.text(0.5, 0.5, 'No suitable features for heatmap', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Feature Impact on Performance Change\n(Higher values = Better ARE performance)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Type', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '12_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All root cause analysis charts generated in {output_dir} directory")
    print(f"📊 Generated 6 sets of charts analyzing root causes of performance changes")

if __name__ == "__main__":
    print("📈 Generating root cause analysis charts...")
    df = parse_data()
    print(f"✅ Successfully parsed {len(df)} datasets")
    create_root_cause_analysis(df)
    print("🎉 Root cause analysis completed!")

