#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验分析脚本
分析 Abl1 (similarity_enhancer), Abl2 (prompt_enhancer) 和完整模型 ARE 的性能对比
"""

import re

def parse_table(file_path):
    """解析Markdown表格数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        line = line.strip()
        if line.startswith('|') and not line.startswith('|---'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 6 and parts[0] != 'Dataset':
                try:
                    dataset = parts[0]
                    abl1_mrr = float(parts[1])
                    abl1_h10 = float(parts[2])
                    abl2_mrr = float(parts[3])
                    abl2_h10 = float(parts[4])
                    are_mrr = float(parts[5])
                    are_h10 = float(parts[6])
                    data.append({
                        'dataset': dataset,
                        'abl1_mrr': abl1_mrr,
                        'abl1_h10': abl1_h10,
                        'abl2_mrr': abl2_mrr,
                        'abl2_h10': abl2_h10,
                        'are_mrr': are_mrr,
                        'are_h10': are_h10
                    })
                except (ValueError, IndexError):
                    continue
    return data

def analyze_ablation(data):
    """分析消融实验结果"""
    total = len(data)
    are_better_abl1_mrr = 0
    are_better_abl1_h10 = 0
    are_better_abl2_mrr = 0
    are_better_abl2_h10 = 0
    are_best_mrr = 0
    are_best_h10 = 0
    are_worst_mrr = 0
    are_worst_h10 = 0
    
    improvements = []
    degradations = []
    
    for item in data:
        dataset = item['dataset']
        abl1_mrr, abl1_h10 = item['abl1_mrr'], item['abl1_h10']
        abl2_mrr, abl2_h10 = item['abl2_mrr'], item['abl2_h10']
        are_mrr, are_h10 = item['are_mrr'], item['are_h10']
        
        # MRR 比较
        if are_mrr > abl1_mrr:
            are_better_abl1_mrr += 1
        if are_mrr > abl2_mrr:
            are_better_abl2_mrr += 1
        
        # H@10 比较
        if are_h10 > abl1_h10:
            are_better_abl1_h10 += 1
        if are_h10 > abl2_h10:
            are_better_abl2_h10 += 1
        
        # 是否是最好的
        best_mrr = max(abl1_mrr, abl2_mrr, are_mrr)
        best_h10 = max(abl1_h10, abl2_h10, are_h10)
        if are_mrr >= best_mrr - 0.001:  # 考虑浮点误差
            are_best_mrr += 1
        if are_h10 >= best_h10 - 0.001:
            are_best_h10 += 1
        
        # 是否是最差的
        worst_mrr = min(abl1_mrr, abl2_mrr, are_mrr)
        worst_h10 = min(abl1_h10, abl2_h10, are_h10)
        if are_mrr <= worst_mrr + 0.001:
            are_worst_mrr += 1
        if are_h10 <= worst_h10 + 0.001:
            are_worst_h10 += 1
        
        # 记录改进和退化
        max_abl_mrr = max(abl1_mrr, abl2_mrr)
        max_abl_h10 = max(abl1_h10, abl2_h10)
        if are_mrr > max_abl_mrr + 0.001:
            improvements.append((dataset, 'MRR', are_mrr - max_abl_mrr))
        elif are_mrr < max_abl_mrr - 0.001:
            degradations.append((dataset, 'MRR', max_abl_mrr - are_mrr))
        
        if are_h10 > max_abl_h10 + 0.001:
            improvements.append((dataset, 'H@10', are_h10 - max_abl_h10))
        elif are_h10 < max_abl_h10 - 0.001:
            degradations.append((dataset, 'H@10', max_abl_h10 - are_h10))
    
    return {
        'total': total,
        'are_better_abl1_mrr': are_better_abl1_mrr,
        'are_better_abl1_h10': are_better_abl1_h10,
        'are_better_abl2_mrr': are_better_abl2_mrr,
        'are_better_abl2_h10': are_better_abl2_h10,
        'are_best_mrr': are_best_mrr,
        'are_best_h10': are_best_h10,
        'are_worst_mrr': are_worst_mrr,
        'are_worst_h10': are_worst_h10,
        'improvements': improvements,
        'degradations': degradations
    }

def print_analysis(results):
    """打印分析结果"""
    print("=" * 80)
    print("消融实验分析结果")
    print("=" * 80)
    print(f"\n总数据集数量: {results['total']}")
    print("\n" + "-" * 80)
    print("ARE vs Abl1 (similarity_enhancer only):")
    print(f"  MRR: ARE 优于 Abl1 的数据集: {results['are_better_abl1_mrr']}/{results['total']} ({results['are_better_abl1_mrr']/results['total']*100:.1f}%)")
    print(f"  H@10: ARE 优于 Abl1 的数据集: {results['are_better_abl1_h10']}/{results['total']} ({results['are_better_abl1_h10']/results['total']*100:.1f}%)")
    print("\n" + "-" * 80)
    print("ARE vs Abl2 (prompt_enhancer only):")
    print(f"  MRR: ARE 优于 Abl2 的数据集: {results['are_better_abl2_mrr']}/{results['total']} ({results['are_better_abl2_mrr']/results['total']*100:.1f}%)")
    print(f"  H@10: ARE 优于 Abl2 的数据集: {results['are_better_abl2_h10']}/{results['total']} ({results['are_better_abl2_h10']/results['total']*100:.1f}%)")
    print("\n" + "-" * 80)
    print("ARE 是否是最佳性能:")
    print(f"  MRR: ARE 是最佳或并列最佳: {results['are_best_mrr']}/{results['total']} ({results['are_best_mrr']/results['total']*100:.1f}%)")
    print(f"  H@10: ARE 是最佳或并列最佳: {results['are_best_h10']}/{results['total']} ({results['are_best_h10']/results['total']*100:.1f}%)")
    print("\n" + "-" * 80)
    print("ARE 是否是最差性能:")
    print(f"  MRR: ARE 是最差或并列最差: {results['are_worst_mrr']}/{results['total']} ({results['are_worst_mrr']/results['total']*100:.1f}%)")
    print(f"  H@10: ARE 是最差或并列最差: {results['are_worst_h10']}/{results['total']} ({results['are_worst_h10']/results['total']*100:.1f}%)")
    
    if results['improvements']:
        print("\n" + "-" * 80)
        print("ARE 相比单个组件有明显改进的数据集 (>0.001):")
        improvements_sorted = sorted(results['improvements'], key=lambda x: x[2], reverse=True)
        for dataset, metric, diff in improvements_sorted[:10]:  # 显示前10个
            print(f"  {dataset} ({metric}): +{diff:.4f}")
    
    if results['degradations']:
        print("\n" + "-" * 80)
        print("ARE 相比单个组件有明显退化的数据集 (>0.001):")
        degradations_sorted = sorted(results['degradations'], key=lambda x: x[2], reverse=True)
        for dataset, metric, diff in degradations_sorted[:10]:  # 显示前10个
            print(f"  {dataset} ({metric}): -{diff:.4f}")
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    
    # 判断消融实验是否成功
    success_rate_mrr = results['are_best_mrr'] / results['total']
    success_rate_h10 = results['are_best_h10'] / results['total']
    
    if success_rate_mrr >= 0.5 and success_rate_h10 >= 0.5:
        print("✓ 消融实验基本成功！")
        print(f"  - ARE 在 {success_rate_mrr*100:.1f}% 的数据集上 MRR 达到最佳或并列最佳")
        print(f"  - ARE 在 {success_rate_h10*100:.1f}% 的数据集上 H@10 达到最佳或并列最佳")
        print("\n建议:")
        print("  - 两个组件（similarity_enhancer 和 prompt_enhancer）都对模型有贡献")
        print("  - 完整模型（两个组件都开启）在大多数情况下表现最好或相当")
    else:
        print("⚠ 消融实验结果需要进一步分析")
        print(f"  - ARE 只在 {success_rate_mrr*100:.1f}% 的数据集上 MRR 达到最佳")
        print(f"  - ARE 只在 {success_rate_h10*100:.1f}% 的数据集上 H@10 达到最佳")
        print("\n可能的原因:")
        print("  - 两个组件可能存在冲突或冗余")
        print("  - 某些数据集可能更适合单个组件")
        print("  - 需要检查实验设置或超参数")

if __name__ == '__main__':
    data = parse_table('data1.md')
    results = analyze_ablation(data)
    print_analysis(results)

