#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化Gini系数和CV的示例
帮助理解这两个指标的含义
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")

def calculate_gini(values):
    """计算Gini系数"""
    if len(values) == 0:
        return 0.0
    values = np.array(values)
    values = values.flatten()
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

def calculate_cv(values):
    """计算CV"""
    if len(values) == 0:
        return 0.0
    values = np.array(values)
    mean_val = np.mean(values)
    if mean_val == 0:
        return 0.0
    return np.std(values) / mean_val

def create_gini_cv_visualization():
    """创建Gini和CV的可视化示例"""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 创建4个示例
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 示例1: 高度结构化（高Gini，高CV）
    ax1 = axes[0, 0]
    freqs1 = [30000, 25000, 20000, 15000, 10000, 5000, 3000, 2000, 1000, 500, 300, 200, 100, 50, 30, 20, 10]
    gini1 = calculate_gini(freqs1)
    cv1 = calculate_cv(freqs1)
    
    ax1.bar(range(len(freqs1)), freqs1, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Relation Type (sorted by frequency)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title(f'High Structure\nGini={gini1:.3f} (High), CV={cv1:.3f} (High)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.5, 0.95, 'Few dominant relations\naccount for most occurrences', 
            transform=ax1.transAxes, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontsize=10)
    
    # 示例2: 低结构化（低Gini，低CV）
    ax2 = axes[0, 1]
    freqs2 = [500] * 17
    gini2 = calculate_gini(freqs2)
    cv2 = calculate_cv(freqs2)
    
    ax2.bar(range(len(freqs2)), freqs2, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Relation Type (sorted by frequency)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title(f'Low Structure\nGini={gini2:.3f} (Low), CV={cv2:.3f} (Low)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.5, 0.95, 'All relations have\nsimilar frequencies', 
            transform=ax2.transAxes, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
            fontsize=10)
    
    # 示例3: 中等结构化（中Gini，中CV）
    ax3 = axes[1, 0]
    freqs3 = [8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 800, 600, 400, 300, 200, 150, 100, 50]
    gini3 = calculate_gini(freqs3)
    cv3 = calculate_cv(freqs3)
    
    ax3.bar(range(len(freqs3)), freqs3, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Relation Type (sorted by frequency)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title(f'Medium Structure\nGini={gini3:.3f} (Medium), CV={cv3:.3f} (Medium)', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(0.5, 0.95, 'Moderate concentration\nof relation frequencies', 
            transform=ax3.transAxes, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
            fontsize=10)
    
    # 示例4: 实际数据对比（YAGO310 vs ConceptNet）
    ax4 = axes[1, 1]
    
    # YAGO310-ht (提升)
    yago_freqs = [80000, 60000, 50000, 40000, 30000, 20000, 15000, 10000, 8000, 5000, 
                  3000, 2000, 1000, 500, 300, 200, 100, 50, 30, 20, 10, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    yago_gini = 0.832
    yago_cv = 2.696
    
    # ConceptNet (下降)
    cnet_freqs = [15000, 12000, 10000, 8000, 6000, 5000, 4000, 3000, 2500, 2000,
                  1500, 1200, 1000, 800, 600, 500, 400, 300, 250, 200,
                  150, 120, 100, 80, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8]
    cnet_gini = 0.690
    cnet_cv = 1.455
    
    x = np.arange(max(len(yago_freqs), len(cnet_freqs)))
    width = 0.35
    
    # 只显示前20个关系
    show_n = 20
    ax4.bar(x[:show_n] - width/2, yago_freqs[:show_n], width, 
           label=f'YAGO310-ht (Gini={yago_gini:.3f}, CV={yago_cv:.3f})', 
           color='#2ecc71', alpha=0.7, edgecolor='black')
    ax4.bar(x[:show_n] + width/2, cnet_freqs[:show_n], width, 
           label=f'ConceptNet (Gini={cnet_gini:.3f}, CV={cnet_cv:.3f})', 
           color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel('Relation Type (sorted by frequency)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Real Dataset Comparison\nYAGO310-ht (Improved) vs ConceptNet (Degraded)', 
                 fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.text(0.5, 0.95, 'Both have high Gini, but\nConceptNet has low semantic clustering', 
            transform=ax4.transAxes, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '26_gini_cv_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to {output_dir / '26_gini_cv_explanation.png'}")

def create_gini_cv_scatter():
    """创建Gini和CV的散点图"""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 读取实际数据
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent / "comprehensive_quantitative_analysis.csv")
    
    improved = df[df['performance_category'] == 'significantly_improved']
    degraded = df[df['performance_category'] == 'significantly_degraded']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制散点图
    if len(improved) > 0:
        ax.scatter(improved['gini_coefficient'], improved['cv_relation_freq'], 
                  s=200, alpha=0.7, color='#2ecc71', edgecolor='black', linewidth=2,
                  label='Significantly Improved', zorder=3)
        for _, row in improved.iterrows():
            if pd.notna(row['matched_name']):
                ax.annotate(row['matched_name'], 
                          (row['gini_coefficient'], row['cv_relation_freq']),
                          fontsize=8, alpha=0.8, ha='left', va='bottom')
    
    if len(degraded) > 0:
        ax.scatter(degraded['gini_coefficient'], degraded['cv_relation_freq'], 
                  s=200, alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=2,
                  label='Significantly Degraded', zorder=3)
        for _, row in degraded.iterrows():
            if pd.notna(row['matched_name']):
                ax.annotate(row['matched_name'], 
                          (row['gini_coefficient'], row['cv_relation_freq']),
                          fontsize=8, alpha=0.8, ha='left', va='bottom')
    
    # 添加区域标注
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.text(0.85, 2.5, 'High Gini\nHigh CV\n→ High Structure', 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
           fontsize=10, ha='center')
    ax.text(0.35, 0.3, 'Low Gini\nLow CV\n→ Low Structure', 
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3),
           fontsize=10, ha='center')
    
    ax.set_xlabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
    ax.set_title('Gini vs CV: Relation Structure Analysis\n(All datasets with quantitative evidence)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '27_gini_cv_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Scatter plot saved to {output_dir / '27_gini_cv_scatter.png'}")

if __name__ == "__main__":
    print("📊 Creating Gini and CV visualizations...")
    
    create_gini_cv_visualization()
    create_gini_cv_scatter()
    
    print("\n✅ All visualizations completed!")

