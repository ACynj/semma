#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于定量指标的关系结构化程度分析
计算数据集的定量特征并基于此进行分类
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")

def parse_data():
    """解析性能数据"""
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

def compute_structure_metrics(df):
    """
    基于性能数据计算结构化程度的定量指标
    使用多个代理指标来推断结构化程度
    """
    
    metrics = {
        'are_effectiveness': [],  # ARE有效性：性能提升幅度
        'performance_stability': [],  # 性能稳定性：SEMMA和ARE的差异
        'improvement_consistency': [],  # 提升一致性：MRR和H@10是否都提升
    }
    
    for _, row in df.iterrows():
        # 指标1: ARE有效性 - 基于性能提升幅度
        # 如果ARE提升大，说明关系可能更结构化（相似度增强有效）
        mrr_improvement = row['mrr_diff']
        h10_improvement = row['h10_diff']
        avg_improvement = (mrr_improvement + h10_improvement) / 2
        metrics['are_effectiveness'].append(avg_improvement)
        
        # 指标2: 性能稳定性 - SEMMA和ARE的差异
        # 如果差异小，说明数据集特征稳定
        mrr_stability = 1 - abs(row['semma_mrr'] - row['are_mrr']) / max(row['semma_mrr'], 0.001)
        h10_stability = 1 - abs(row['semma_h10'] - row['are_h10']) / max(row['semma_h10'], 0.001)
        metrics['performance_stability'].append((mrr_stability + h10_stability) / 2)
        
        # 指标3: 提升一致性 - MRR和H@10是否都提升
        # 如果都提升，说明增强机制一致有效
        both_improve = 1 if (mrr_improvement > 0 and h10_improvement > 0) else 0
        both_degrade = 1 if (mrr_improvement < 0 and h10_improvement < 0) else 0
        consistency = 1.0 if (both_improve or both_degrade) else 0.5
        metrics['improvement_consistency'].append(consistency)
    
    for key in metrics:
        df[f'metric_{key}'] = metrics[key]
    
    # 计算综合结构化得分
    # 归一化各个指标到0-1范围
    for key in metrics:
        col = f'metric_{key}'
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col + '_norm'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col + '_norm'] = 0.5
    
    # 综合得分：加权平均
    # ARE有效性权重最高（0.5），稳定性（0.3），一致性（0.2）
    df['structure_score'] = (
        0.5 * df['metric_are_effectiveness_norm'] +
        0.3 * df['metric_performance_stability_norm'] +
        0.2 * df['metric_improvement_consistency_norm']
    )
    
    return df

def classify_by_quantitative_metrics(df):
    """基于定量指标分类结构化程度"""
    
    # 使用三分位数进行分类
    q33 = df['structure_score'].quantile(0.33)
    q67 = df['structure_score'].quantile(0.67)
    
    def classify(score):
        if score >= q67:
            return 'high'
        elif score >= q33:
            return 'medium'
        else:
            return 'low'
    
    df['structure_quantitative'] = df['structure_score'].apply(classify)
    
    return df

def compare_classifications(df):
    """对比基于规则和基于定量指标的分类"""
    
    # 基于规则的分类（原有方法）
    def rule_based_classify(dataset):
        dataset_lower = dataset.lower()
        if any(x in dataset_lower for x in ['metafam', 'yago', 'fb15k', 'wn18', 'codex']):
            return 'high'
        elif any(x in dataset_lower for x in ['conceptnet', 'nell']):
            return 'low'
        else:
            return 'medium'
    
    df['structure_rule_based'] = df['dataset'].apply(rule_based_classify)
    
    return df

def create_comparison_visualizations(df):
    """创建对比可视化"""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 1. 结构化得分分布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 结构化得分分布
    ax1 = axes[0, 0]
    ax1.hist(df['structure_score'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(df['structure_score'].quantile(0.33), color='orange', linestyle='--', 
               label='33% quantile (Medium threshold)')
    ax1.axvline(df['structure_score'].quantile(0.67), color='green', linestyle='--', 
               label='67% quantile (High threshold)')
    ax1.set_xlabel('Structure Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Structure Scores', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 结构化得分 vs ARE性能提升
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['structure_score'], df['mrr_diff'], 
                          c=df['mrr_improvement_pct'], cmap='RdYlGn', 
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Structure Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MRR Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax2.set_title('Structure Score vs Performance Improvement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Improvement %')
    
    # 标注关键数据集
    key_datasets = ['Metafam', 'YAGO310-ht', 'ConceptNet 100k-ht', 'WikiTopicsMT3:infra']
    for dataset in key_datasets:
        row = df[df['dataset'] == dataset]
        if len(row) > 0:
            row = row.iloc[0]
            if np.isfinite(row['structure_score']) and np.isfinite(row['mrr_diff']):
                ax2.annotate(dataset, (row['structure_score'], row['mrr_diff']), 
                            fontsize=8, alpha=0.7)
    
    # 规则分类 vs 定量分类对比
    ax3 = axes[1, 0]
    comparison = pd.crosstab(df['structure_rule_based'], df['structure_quantitative'], 
                            normalize='index') * 100
    sns.heatmap(comparison, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3, 
               cbar_kws={'label': 'Percentage'})
    ax3.set_xlabel('Quantitative Classification', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Rule-Based Classification', fontsize=11, fontweight='bold')
    ax3.set_title('Rule-Based vs Quantitative Classification', fontsize=12, fontweight='bold')
    
    # 不同分类方法的性能差异对比
    ax4 = axes[1, 1]
    methods = ['Rule-Based', 'Quantitative']
    categories = ['High', 'Medium', 'Low']
    
    rule_means = [df[df['structure_rule_based'] == cat.lower()]['mrr_diff'].mean() 
                  for cat in categories]
    quant_means = [df[df['structure_quantitative'] == cat.lower()]['mrr_diff'].mean() 
                  for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, rule_means, width, label='Rule-Based', 
                   color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = ax4.bar(x + width/2, quant_means, width, label='Quantitative', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Structure Level', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average MRR Difference', fontsize=11, fontweight='bold')
    ax4.set_title('Performance by Classification Method', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '13_quantitative_structure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 详细指标对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['are_effectiveness', 'performance_stability', 'improvement_consistency']
    metric_labels = ['ARE Effectiveness', 'Performance Stability', 'Improvement Consistency']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # 按定量分类分组
        high_data = df[df['structure_quantitative'] == 'high'][f'metric_{metric}_norm']
        medium_data = df[df['structure_quantitative'] == 'medium'][f'metric_{metric}_norm']
        low_data = df[df['structure_quantitative'] == 'low'][f'metric_{metric}_norm']
        
        data_to_plot = [high_data, medium_data, low_data]
        bp = ax.boxplot(data_to_plot, labels=['High', 'Medium', 'Low'], 
                       patch_artist=True)
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Normalized Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} by Structure Level', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '14_structure_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 关键数据集详细对比
    key_datasets = ['Metafam', 'YAGO310-ht', 'ConceptNet 100k-ht', 'WikiTopicsMT3:infra', 
                    'FB15K237Inductive:v2', 'NELLInductive:v1']
    key_df = df[df['dataset'].isin(key_datasets)].copy()
    
    if len(key_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(key_df))
        width = 0.15
        
        metrics_to_plot = ['are_effectiveness_norm', 'performance_stability_norm', 
                          'improvement_consistency_norm', 'structure_score']
        metric_labels_short = ['ARE Eff.', 'Stability', 'Consistency', 'Overall']
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels_short)):
            if metric == 'structure_score':
                values = key_df[metric].values
            else:
                values = key_df[f'metric_{metric}'].values
            offset = (i - len(metrics_to_plot)/2) * width
            bars = ax.bar(x + offset, values, width, label=label, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for j, (bar, val) in enumerate(zip(bars, values)):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width()/2., val,
                           f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', 
                           fontsize=8)
        
        ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Score', fontsize=11, fontweight='bold')
        ax.set_title('Key Datasets: Quantitative Structure Metrics', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['dataset']}\n({row['structure_quantitative']})" 
                           for _, row in key_df.iterrows()], rotation=15, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / '15_key_datasets_quantitative_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Quantitative structure analysis charts generated in {output_dir} directory")

if __name__ == "__main__":
    print("📈 Computing quantitative structure metrics...")
    df = parse_data()
    print(f"✅ Successfully parsed {len(df)} datasets")
    
    print("🔢 Computing structure metrics...")
    df = compute_structure_metrics(df)
    
    print("📊 Classifying by quantitative metrics...")
    df = classify_by_quantitative_metrics(df)
    
    print("🔄 Comparing with rule-based classification...")
    df = compare_classifications(df)
    
    print("📈 Creating visualizations...")
    create_comparison_visualizations(df)
    
    # 保存结果
    output_file = Path(__file__).parent / "quantitative_structure_results.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")
    
    # 打印统计信息
    print("\n📊 Classification Statistics:")
    print("\nRule-Based Classification:")
    print(df['structure_rule_based'].value_counts().sort_index())
    print("\nQuantitative Classification:")
    print(df['structure_quantitative'].value_counts().sort_index())
    print("\n🎉 Analysis completed!")

