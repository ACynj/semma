#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析显著提升和下降数据集的共性特征
找出模型适用性和不适用性的场景特征
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
from scipy import stats

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")

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
                                'h10_improvement_pct': (are_h10 - semma_h10) / semma_h10 * 100 if semma_h10 > 0 else 0,
                            })
                        except ValueError:
                            continue
    
    return pd.DataFrame(datasets)

def extract_dataset_features(df):
    """从数据集名称和性能数据中提取特征"""
    
    features = {
        'dataset_name_pattern': [],  # 数据集名称模式
        'dataset_type': [],  # 数据集类型
        'base_performance': [],  # SEMMA基础性能
        'performance_level': [],  # 性能水平: high/medium/low
        'domain_category': [],  # 领域类别
        'scale_indicator': [],  # 规模指示器
        'is_inductive': [],  # 是否归纳设置
        'is_pretrain': [],  # 是否预训练数据
    }
    
    for _, row in df.iterrows():
        dataset = row['dataset'].lower()
        dataset_type = row['type']
        base_mrr = row['semma_mrr']
        
        # 数据集名称模式
        if 'metafam' in dataset:
            features['dataset_name_pattern'].append('biology')
        elif 'conceptnet' in dataset:
            features['dataset_name_pattern'].append('common_sense')
        elif 'yago' in dataset:
            features['dataset_name_pattern'].append('structured_large')
        elif 'fb15k' in dataset or 'fb' in dataset:
            features['dataset_name_pattern'].append('fb_family')
        elif 'wn18' in dataset or 'wn' in dataset:
            features['dataset_name_pattern'].append('wordnet')
        elif 'nell' in dataset:
            features['dataset_name_pattern'].append('nell_family')
        elif 'wikitopics' in dataset or 'wiktopics' in dataset:
            features['dataset_name_pattern'].append('domain_specific')
        elif 'codex' in dataset:
            features['dataset_name_pattern'].append('codex_family')
        else:
            features['dataset_name_pattern'].append('other')
        
        # 数据集类型
        features['dataset_type'].append(dataset_type)
        
        # 基础性能
        features['base_performance'].append(base_mrr)
        
        # 性能水平
        if base_mrr >= 0.5:
            features['performance_level'].append('high')
        elif base_mrr >= 0.3:
            features['performance_level'].append('medium')
        else:
            features['performance_level'].append('low')
        
        # 领域类别
        if 'metafam' in dataset:
            features['domain_category'].append('biology')
        elif 'conceptnet' in dataset:
            features['domain_category'].append('common_sense')
        elif any(x in dataset for x in ['wikitopics', 'wiktopics']):
            features['domain_category'].append('domain_specific')
        else:
            features['domain_category'].append('general')
        
        # 规模指示器
        if any(x in dataset for x in ['large', '100k', '310', 'yago']):
            features['scale_indicator'].append('large')
        elif any(x in dataset for x in ['small', '23k', '995', '10', '20', '50']):
            features['scale_indicator'].append('small')
        else:
            features['scale_indicator'].append('medium')
        
        # 是否归纳设置
        features['is_inductive'].append(1 if 'inductive' in dataset_type.lower() else 0)
        
        # 是否预训练数据
        features['is_pretrain'].append(1 if dataset_type == 'Pre-training' else 0)
    
    for key in features:
        df[key] = features[key]
    
    return df

def classify_datasets(df):
    """分类数据集为显著提升、显著下降、基本持平"""
    
    # 定义显著提升和下降的阈值
    improvement_threshold = 0.01  # MRR提升 > 0.01
    degradation_threshold = -0.01  # MRR下降 < -0.01
    
    def classify_row(row):
        mrr_diff = row['mrr_diff']
        if mrr_diff > improvement_threshold:
            return 'significantly_improved'
        elif mrr_diff < degradation_threshold:
            return 'significantly_degraded'
        else:
            return 'stable'
    
    df['performance_category'] = df.apply(classify_row, axis=1)
    
    return df

def analyze_common_features(df):
    """分析提升和下降数据集的共性特征"""
    
    improved = df[df['performance_category'] == 'significantly_improved']
    degraded = df[df['performance_category'] == 'significantly_degraded']
    stable = df[df['performance_category'] == 'stable']
    
    print(f"\n📊 数据集分类统计:")
    print(f"  显著提升: {len(improved)} 个")
    print(f"  显著下降: {len(degraded)} 个")
    print(f"  基本持平: {len(stable)} 个")
    
    # 分析各个特征的分布
    feature_analysis = {}
    
    categorical_features = ['dataset_name_pattern', 'dataset_type', 'performance_level', 
                           'domain_category', 'scale_indicator']
    
    for feature in categorical_features:
        improved_dist = improved[feature].value_counts(normalize=True) * 100
        degraded_dist = degraded[feature].value_counts(normalize=True) * 100
        stable_dist = stable[feature].value_counts(normalize=True) * 100
        
        feature_analysis[feature] = {
            'improved': improved_dist,
            'degraded': degraded_dist,
            'stable': stable_dist
        }
    
    # 分析数值特征
    numerical_features = ['base_performance', 'semma_mrr', 'semma_h10']
    
    for feature in numerical_features:
        feature_analysis[feature] = {
            'improved_mean': improved[feature].mean() if len(improved) > 0 else 0,
            'improved_std': improved[feature].std() if len(improved) > 0 else 0,
            'degraded_mean': degraded[feature].mean() if len(degraded) > 0 else 0,
            'degraded_std': degraded[feature].std() if len(degraded) > 0 else 0,
            'stable_mean': stable[feature].mean() if len(stable) > 0 else 0,
            'stable_std': stable[feature].std() if len(stable) > 0 else 0,
        }
    
    return feature_analysis, improved, degraded, stable

def create_feature_comparison_charts(df, feature_analysis, improved, degraded, stable):
    """创建特征对比图表"""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 1. 分类特征分布对比（饼图）
    categorical_features = ['dataset_name_pattern', 'dataset_type', 'domain_category', 'scale_indicator']
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, feature in enumerate(categorical_features):
        ax = axes[idx]
        
        # 获取三类数据集的分布
        improved_dist = feature_analysis[feature]['improved']
        degraded_dist = feature_analysis[feature]['degraded']
        stable_dist = feature_analysis[feature]['stable']
        
        # 合并所有类别
        all_categories = set(improved_dist.index) | set(degraded_dist.index) | set(stable_dist.index)
        
        improved_values = [improved_dist.get(cat, 0) for cat in all_categories]
        degraded_values = [degraded_dist.get(cat, 0) for cat in all_categories]
        stable_values = [stable_dist.get(cat, 0) for cat in all_categories]
        
        x = np.arange(len(all_categories))
        width = 0.25
        
        bars1 = ax.bar(x - width, improved_values, width, label='Significantly Improved', 
                      color='#2ecc71', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, stable_values, width, label='Stable', 
                      color='#95a5a6', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, degraded_values, width, label='Significantly Degraded', 
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in all_categories], 
                          rotation=15, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / '16_categorical_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 数值特征分布对比（箱线图）
    numerical_features = ['base_performance', 'semma_mrr', 'semma_h10']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, feature in enumerate(numerical_features):
        ax = axes[idx]
        
        data_to_plot = [
            improved[feature].values if len(improved) > 0 else [],
            stable[feature].values if len(stable) > 0 else [],
            degraded[feature].values if len(degraded) > 0 else []
        ]
        
        bp = ax.boxplot(data_to_plot, tick_labels=['Improved', 'Stable', 'Degraded'], 
                       patch_artist=True)
        
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '17_numerical_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 关键特征重要性分析
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 计算每个特征在不同类别下的差异
    feature_importance = {}
    
    # 分类特征：使用卡方检验或简单的比例差异
    for feature in categorical_features:
        improved_dist = feature_analysis[feature]['improved']
        degraded_dist = feature_analysis[feature]['degraded']
        
        # 计算差异（使用最大差异）
        all_cats = set(improved_dist.index) | set(degraded_dist.index)
        max_diff = 0
        for cat in all_cats:
            imp_val = improved_dist.get(cat, 0)
            deg_val = degraded_dist.get(cat, 0)
            diff = abs(imp_val - deg_val)
            max_diff = max(max_diff, diff)
        
        feature_importance[feature] = max_diff
    
    # 数值特征：使用t检验或均值差异
    for feature in numerical_features:
        if len(improved) > 0 and len(degraded) > 0:
            imp_mean = improved[feature].mean()
            deg_mean = degraded[feature].mean()
            diff = abs(imp_mean - deg_mean)
            # 归一化到0-100范围
            feature_importance[feature] = diff * 100
    
    # 排序并绘制
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)
    
    bars = ax.barh(range(len(features)), importances, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=11)
    ax.set_xlabel('Feature Importance (Difference Score)', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance for Distinguishing Improved vs Degraded Datasets', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp + 0.5, i, f'{imp:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '18_feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 提升和下降数据集的详细特征对比表
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建对比表格
    table_data = []
    
    # 添加分类特征
    for feature in categorical_features:
        improved_dist = feature_analysis[feature]['improved']
        degraded_dist = feature_analysis[feature]['degraded']
        
        # 找到差异最大的类别
        all_cats = set(improved_dist.index) | set(degraded_dist.index)
        max_diff_cat = None
        max_diff = 0
        
        for cat in all_cats:
            imp_val = improved_dist.get(cat, 0)
            deg_val = degraded_dist.get(cat, 0)
            diff = abs(imp_val - deg_val)
            if diff > max_diff:
                max_diff = diff
                max_diff_cat = cat
        
        if max_diff_cat:
            imp_pct = improved_dist.get(max_diff_cat, 0)
            deg_pct = degraded_dist.get(max_diff_cat, 0)
            table_data.append([
                feature.replace('_', ' ').title(),
                max_diff_cat.replace('_', ' ').title(),
                f'{imp_pct:.1f}%',
                f'{deg_pct:.1f}%',
                f'{imp_pct - deg_pct:+.1f}%'
            ])
    
    # 添加数值特征
    for feature in numerical_features:
        imp_mean = feature_analysis[feature]['improved_mean']
        deg_mean = feature_analysis[feature]['degraded_mean']
        diff = imp_mean - deg_mean
        table_data.append([
            feature.replace('_', ' ').title(),
            'Mean Value',
            f'{imp_mean:.4f}',
            f'{deg_mean:.4f}',
            f'{diff:+.4f}'
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Feature', 'Category/Value', 'Improved (%)', 'Degraded (%)', 'Difference'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if j == 4:  # Difference列
                val = float(table_data[i-1][4].replace('%', '').replace('+', ''))
                if val > 0:
                    table[(i, j)].set_facecolor('#d5f4e6')
                elif val < 0:
                    table[(i, j)].set_facecolor('#fadbd8')
                else:
                    table[(i, j)].set_facecolor('#f8f9fa')
    
    plt.title('Common Features Comparison: Improved vs Degraded Datasets', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / '19_detailed_feature_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 适用性场景总结图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：适用场景特征
    ax1 = axes[0]
    
    # 从提升数据集中提取共性特征
    improved_features = {
        'High Base Performance': len(improved[improved['base_performance'] >= 0.4]),
        'Inductive Setting': len(improved[improved['is_inductive'] == 1]),
        'Structured Domain': len(improved[improved['domain_category'] == 'general']),
        'FB/WordNet Family': len(improved[improved['dataset_name_pattern'].isin(['fb_family', 'wordnet'])]),
    }
    
    categories = list(improved_features.keys())
    values = list(improved_features.values())
    total_improved = len(improved)
    percentages = [v / total_improved * 100 if total_improved > 0 else 0 for v in values]
    
    bars1 = ax1.barh(categories, percentages, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Percentage of Improved Datasets (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Applicability: Common Features of Improved Datasets', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, pct) in enumerate(zip(bars1, percentages)):
        ax1.text(pct + 2, i, f'{pct:.1f}% ({values[i]}/{total_improved})', 
                va='center', fontsize=10)
    
    # 右图：不适用场景特征
    ax2 = axes[1]
    
    # 从下降数据集中提取共性特征
    degraded_features = {
        'Low Base Performance': len(degraded[degraded['base_performance'] < 0.3]),
        'Common Sense Domain': len(degraded[degraded['domain_category'] == 'common_sense']),
        'Domain Specific': len(degraded[degraded['domain_category'] == 'domain_specific']),
        'Non-Structured': len(degraded[degraded['dataset_name_pattern'] == 'common_sense']),
    }
    
    categories = list(degraded_features.keys())
    values = list(degraded_features.values())
    total_degraded = len(degraded)
    percentages = [v / total_degraded * 100 if total_degraded > 0 else 0 for v in values]
    
    bars2 = ax2.barh(categories, percentages, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Percentage of Degraded Datasets (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Model Non-Applicability: Common Features of Degraded Datasets', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, pct) in enumerate(zip(bars2, percentages)):
        ax2.text(pct + 2, i, f'{pct:.1f}% ({values[i]}/{total_degraded})', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '20_applicability_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All feature analysis charts generated in {output_dir} directory")

if __name__ == "__main__":
    print("📈 Analyzing common features of improved and degraded datasets...")
    
    # 解析数据
    df = parse_data()
    print(f"✅ Successfully parsed {len(df)} datasets")
    
    # 提取特征
    print("🔍 Extracting dataset features...")
    df = extract_dataset_features(df)
    
    # 分类数据集
    print("📊 Classifying datasets...")
    df = classify_datasets(df)
    
    # 分析共性特征
    print("🔬 Analyzing common features...")
    feature_analysis, improved, degraded, stable = analyze_common_features(df)
    
    # 创建可视化
    print("📈 Creating visualizations...")
    create_feature_comparison_charts(df, feature_analysis, improved, degraded, stable)
    
    # 保存结果
    output_file = Path(__file__).parent / "common_features_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")
    
    # 打印关键发现
    print("\n🎯 Key Findings:")
    print(f"\n显著提升的数据集 ({len(improved)}个):")
    if len(improved) > 0:
        print(f"  - 平均基础性能: {improved['base_performance'].mean():.4f}")
        print(f"  - 平均MRR提升: {improved['mrr_diff'].mean():.4f}")
        print(f"  - 最常见的领域: {improved['domain_category'].mode()[0] if len(improved['domain_category'].mode()) > 0 else 'N/A'}")
    
    print(f"\n显著下降的数据集 ({len(degraded)}个):")
    if len(degraded) > 0:
        print(f"  - 平均基础性能: {degraded['base_performance'].mean():.4f}")
        print(f"  - 平均MRR下降: {degraded['mrr_diff'].mean():.4f}")
        print(f"  - 最常见的领域: {degraded['domain_category'].mode()[0] if len(degraded['domain_category'].mode()) > 0 else 'N/A'}")
    
    print("\n🎉 Analysis completed!")

