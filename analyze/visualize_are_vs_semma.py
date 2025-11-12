#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARE vs SEMMA 性能对比可视化脚本
生成多个图表证明分析结果
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re

# 设置字体（使用英文标签避免中文字体问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 数据解析
def parse_data():
    """从data.md解析数据"""
    data_file = Path(__file__).parent / "data.md"
    
    datasets = []
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析各个部分的数据
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
            # 解析表格行
            lines = table_content.strip().split('\n')
            for line in lines[2:]:  # 跳过表头和分隔线
                if '|' in line and not line.startswith('|---'):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 7:
                        dataset = parts[0]
                        try:
                            ultra_mrr = float(parts[1])
                            ultra_h10 = float(parts[2])
                            semma_mrr = float(parts[3])
                            semma_h10 = float(parts[4])
                            are_mrr = float(parts[5])
                            are_h10 = float(parts[6])
                            
                            datasets.append({
                                'dataset': dataset,
                                'type': section_name,
                                'ultra_mrr': ultra_mrr,
                                'ultra_h10': ultra_h10,
                                'semma_mrr': semma_mrr,
                                'semma_h10': semma_h10,
                                'are_mrr': are_mrr,
                                'are_h10': are_h10,
                                'mrr_diff': are_mrr - semma_mrr,
                                'h10_diff': are_h10 - semma_h10,
                                'mrr_improvement': (are_mrr - semma_mrr) / semma_mrr * 100 if semma_mrr > 0 else 0,
                                'h10_improvement': (are_h10 - semma_h10) / semma_h10 * 100 if semma_h10 > 0 else 0,
                            })
                        except ValueError:
                            continue
    
    return pd.DataFrame(datasets)

# 创建图表
def create_visualizations(df):
    """创建所有可视化图表"""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 1. ARE vs SEMMA 性能对比散点图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MRR对比
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['semma_mrr'], df['are_mrr'], 
                          c=df['mrr_diff'], cmap='RdYlGn', 
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x (No difference line)')
    ax1.set_xlabel('SEMMA MRR', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ARE MRR', fontsize=12, fontweight='bold')
    ax1.set_title('ARE vs SEMMA MRR Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1, label='Performance Difference (ARE - SEMMA)')
    
    # H@10对比
    ax2 = axes[1]
    scatter2 = ax2.scatter(df['semma_h10'], df['are_h10'], 
                          c=df['h10_diff'], cmap='RdYlGn', 
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x (No difference line)')
    ax2.set_xlabel('SEMMA H@10', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ARE H@10', fontsize=12, fontweight='bold')
    ax2.set_title('ARE vs SEMMA H@10 Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Performance Difference (ARE - SEMMA)')
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_performance_comparison_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 显著提升和下降的数据集
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 按MRR差异排序
    df_sorted = df.sort_values('mrr_diff', ascending=False)
    
    # 显著提升 (>0.01)
    top_improvements = df_sorted[df_sorted['mrr_diff'] > 0.01].head(10)
    # 显著下降 (<-0.01)
    top_degradations = df_sorted[df_sorted['mrr_diff'] < -0.01].head(10)
    
    # MRR提升
    ax1 = axes[0]
    if len(top_improvements) > 0:
        y_pos = np.arange(len(top_improvements))
        bars1 = ax1.barh(y_pos, top_improvements['mrr_diff'], color='green', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['dataset']}\n({row['type']})" for _, row in top_improvements.iterrows()], 
                           fontsize=9)
        ax1.set_xlabel('MRR Improvement (ARE - SEMMA)', fontsize=11, fontweight='bold')
        ax1.set_title('Top Improved Datasets (MRR Improvement > 0.01)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (idx, row) in enumerate(top_improvements.iterrows()):
            ax1.text(row['mrr_diff'] + 0.005, i, f"+{row['mrr_diff']:.3f}\n(+{row['mrr_improvement']:.1f}%)", 
                   va='center', fontsize=8)
    
    # MRR下降
    ax2 = axes[1]
    if len(top_degradations) > 0:
        y_pos = np.arange(len(top_degradations))
        bars2 = ax2.barh(y_pos, top_degradations['mrr_diff'], color='red', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{row['dataset']}\n({row['type']})" for _, row in top_degradations.iterrows()], 
                           fontsize=9)
        ax2.set_xlabel('MRR Degradation (ARE - SEMMA)', fontsize=11, fontweight='bold')
        ax2.set_title('Top Degraded Datasets (MRR Degradation > 0.01)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (idx, row) in enumerate(top_degradations.iterrows()):
            ax2.text(row['mrr_diff'] - 0.005, i, f"{row['mrr_diff']:.3f}\n({row['mrr_improvement']:.1f}%)", 
                   va='center', ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_top_improvements_degradations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 不同数据集类型的表现对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    type_summary = df.groupby('type').agg({
        'mrr_diff': ['mean', 'std', 'count'],
        'h10_diff': ['mean', 'std']
    }).round(4)
    
    types = type_summary.index.tolist()
    mrr_means = type_summary[('mrr_diff', 'mean')].values
    mrr_stds = type_summary[('mrr_diff', 'std')].values
    h10_means = type_summary[('h10_diff', 'mean')].values
    h10_stds = type_summary[('h10_diff', 'std')].values
    
    x = np.arange(len(types))
    width = 0.35
    
    # MRR对比
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, mrr_means, width, yerr=mrr_stds, 
                   label='Average MRR Difference', color='steelblue', alpha=0.7, capsize=5)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference line')
    ax1.set_xlabel('Dataset Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average MRR Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax1.set_title('Average MRR Difference by Dataset Type', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(mrr_means, mrr_stds)):
        ax1.text(i, mean + std + 0.005 if mean >= 0 else mean - std - 0.005, 
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=9)
    
    # H@10对比
    ax2 = axes[1]
    bars2 = ax2.bar(x - width/2, h10_means, width, yerr=h10_stds, 
                   label='Average H@10 Difference', color='coral', alpha=0.7, capsize=5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference line')
    ax2.set_xlabel('Dataset Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average H@10 Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax2.set_title('Average H@10 Difference by Dataset Type', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(types, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(h10_means, h10_stds)):
        y_pos = mean + std + 0.005 if mean >= 0 else mean - std - 0.005
        va_align = 'bottom' if mean >= 0 else 'top'
        ax2.text(i, y_pos, f'{mean:.4f}', ha='center', va=va_align, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_dataset_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 性能变化分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MRR差异分布
    ax1 = axes[0]
    ax1.hist(df['mrr_diff'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference line')
    ax1.axvline(x=df['mrr_diff'].mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {df["mrr_diff"].mean():.4f}')
    ax1.set_xlabel('MRR Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax1.set_title('MRR Performance Change Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # H@10差异分布
    ax2 = axes[1]
    ax2.hist(df['h10_diff'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference line')
    ax2.axvline(x=df['h10_diff'].mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {df["h10_diff"].mean():.4f}')
    ax2.set_xlabel('H@10 Difference (ARE - SEMMA)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Datasets', fontsize=11, fontweight='bold')
    ax2.set_title('H@10 Performance Change Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 关键数据集详细对比（Metafam, YAGO310, ConceptNet等）
    key_datasets = ['Metafam', 'YAGO310-ht', 'ConceptNet 100k-ht', 'NELLInductive:v1', 
                    'WikiTopicsMT3:infra', 'FB15K237Inductive:v2', 'WN18RRInductive:v3']
    
    key_df = df[df['dataset'].isin(key_datasets)].copy()
    if len(key_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        x = np.arange(len(key_df))
        width = 0.35
        
        # MRR对比
        ax1 = axes[0]
        bars1 = ax1.bar(x - width/2, key_df['semma_mrr'], width, label='SEMMA', 
                       color='lightblue', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, key_df['are_mrr'], width, label='ARE', 
                       color='lightcoral', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Dataset', fontsize=11, fontweight='bold')
        ax1.set_ylabel('MRR', fontsize=11, fontweight='bold')
        ax1.set_title('Key Datasets MRR Detailed Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{row['dataset']}\n({row['type']})" for _, row in key_df.iterrows()], 
                           rotation=15, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加差异标注
        for i, (idx, row) in enumerate(key_df.iterrows()):
            diff = row['mrr_diff']
            color = 'green' if diff > 0 else 'red'
            ax1.text(i, max(row['semma_mrr'], row['are_mrr']) + 0.02, 
                   f"{diff:+.3f}", ha='center', color=color, fontweight='bold', fontsize=9)
        
        # H@10对比
        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, key_df['semma_h10'], width, label='SEMMA', 
                       color='lightblue', alpha=0.8, edgecolor='black')
        bars4 = ax2.bar(x + width/2, key_df['are_h10'], width, label='ARE', 
                       color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Dataset', fontsize=11, fontweight='bold')
        ax2.set_ylabel('H@10', fontsize=11, fontweight='bold')
        ax2.set_title('Key Datasets H@10 Detailed Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{row['dataset']}\n({row['type']})" for _, row in key_df.iterrows()], 
                           rotation=15, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加差异标注
        for i, (idx, row) in enumerate(key_df.iterrows()):
            diff = row['h10_diff']
            color = 'green' if diff > 0 else 'red'
            ax2.text(i, max(row['semma_h10'], row['are_h10']) + 0.02, 
                   f"{diff:+.3f}", ha='center', color=color, fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / '5_key_datasets_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. 性能提升/下降统计饼图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MRR统计
    improved_mrr = len(df[df['mrr_diff'] > 0.01])
    degraded_mrr = len(df[df['mrr_diff'] < -0.01])
    stable_mrr = len(df[(df['mrr_diff'] >= -0.01) & (df['mrr_diff'] <= 0.01)])
    
    ax1 = axes[0]
    sizes = [improved_mrr, degraded_mrr, stable_mrr]
    labels = [f'Significantly Improved\n(>0.01)\n{improved_mrr} datasets', 
             f'Significantly Degraded\n(<-0.01)\n{degraded_mrr} datasets', 
             f'Stable\n(±0.01)\n{stable_mrr} datasets']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.1, 0.1, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('MRR Performance Change Statistics', fontsize=12, fontweight='bold')
    
    # H@10统计
    improved_h10 = len(df[df['h10_diff'] > 0.01])
    degraded_h10 = len(df[df['h10_diff'] < -0.01])
    stable_h10 = len(df[(df['h10_diff'] >= -0.01) & (df['h10_diff'] <= 0.01)])
    
    ax2 = axes[1]
    sizes = [improved_h10, degraded_h10, stable_h10]
    labels = [f'Significantly Improved\n(>0.01)\n{improved_h10} datasets', 
             f'Significantly Degraded\n(<-0.01)\n{degraded_h10} datasets', 
             f'Stable\n(±0.01)\n{stable_h10} datasets']
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('H@10 Performance Change Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '6_performance_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All charts generated in {output_dir} directory")
    print(f"📊 Generated 6 sets of charts showing detailed ARE vs SEMMA comparison")

if __name__ == "__main__":
    print("📈 Generating ARE vs SEMMA performance comparison charts...")
    df = parse_data()
    print(f"✅ Successfully parsed {len(df)} datasets")
    create_visualizations(df)
    print("🎉 Chart generation completed!")

