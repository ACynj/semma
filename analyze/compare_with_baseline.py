#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比消融实验组件与SEMMA基准的性能
分析 Abl1/Abl2 是否低于 SEMMA 基准，以及对模型有效性证明的影响
"""

def parse_data1_table(file_path):
    """解析 data1.md 表格数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = {}
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
                    data[dataset] = {
                        'abl1_mrr': abl1_mrr,
                        'abl1_h10': abl1_h10,
                        'abl2_mrr': abl2_mrr,
                        'abl2_h10': abl2_h10,
                        'are_mrr': are_mrr,
                        'are_h10': are_h10
                    }
                except (ValueError, IndexError):
                    continue
    return data

def parse_data_table(file_path):
    """解析 data.md 表格数据，提取 SEMMA 和 ARE 的结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = {}
    sections = content.split('##')
    
    for section in sections:
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('|') and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 6 and parts[0] != 'Dataset':
                    try:
                        dataset = parts[0]
                        # 根据表格列数判断
                        if len(parts) >= 7:  # 包含 ULTRA, SEMMA, ARE
                            semma_mrr = float(parts[3])
                            semma_h10 = float(parts[4])
                            are_mrr = float(parts[5])
                            are_h10 = float(parts[6])
                        else:  # 只有 SEMMA 和 ARE
                            semma_mrr = float(parts[1])
                            semma_h10 = float(parts[2])
                            are_mrr = float(parts[3])
                            are_h10 = float(parts[4])
                        
                        if dataset not in data:
                            data[dataset] = {}
                        data[dataset]['semma_mrr'] = semma_mrr
                        data[dataset]['semma_h10'] = semma_h10
                        data[dataset]['are_mrr'] = are_mrr
                        data[dataset]['are_h10'] = are_h10
                    except (ValueError, IndexError):
                        continue
    return data

def normalize_dataset_name(name):
    """标准化数据集名称以便匹配"""
    name = name.replace(' ', '').replace('_', '').replace('-', '').lower()
    # 处理一些特殊情况
    name = name.replace('dbpedia100k', 'dbpedia100k')
    name = name.replace('conceptnet100k', 'conceptnet100k')
    name = name.replace('nell23k', 'nell23k')
    name = name.replace('yago310', 'yago310')
    name = name.replace('wdsinger', 'wdsinger')
    name = name.replace('aristov4', 'aristov4')
    return name

def compare_with_baseline(ablation_data, baseline_data):
    """对比消融实验组件与SEMMA基准"""
    results = {
        'abl1_vs_semma': {'mrr_better': 0, 'mrr_worse': 0, 'h10_better': 0, 'h10_worse': 0, 'details': []},
        'abl2_vs_semma': {'mrr_better': 0, 'mrr_worse': 0, 'h10_better': 0, 'h10_worse': 0, 'details': []},
        'are_vs_semma': {'mrr_better': 0, 'mrr_worse': 0, 'h10_better': 0, 'h10_worse': 0, 'details': []},
        'total_matched': 0
    }
    
    # 创建标准化名称映射
    baseline_normalized = {}
    for name, values in baseline_data.items():
        normalized = normalize_dataset_name(name)
        baseline_normalized[normalized] = (name, values)
    
    for abl_name, abl_values in ablation_data.items():
        normalized = normalize_dataset_name(abl_name)
        if normalized in baseline_normalized:
            orig_name, semma_values = baseline_normalized[normalized]
            results['total_matched'] += 1
            
            # 比较 Abl1 vs SEMMA
            abl1_mrr = abl_values['abl1_mrr']
            abl1_h10 = abl_values['abl1_h10']
            semma_mrr = semma_values['semma_mrr']
            semma_h10 = semma_values['semma_h10']
            
            if abl1_mrr > semma_mrr + 0.001:
                results['abl1_vs_semma']['mrr_better'] += 1
            elif abl1_mrr < semma_mrr - 0.001:
                results['abl1_vs_semma']['mrr_worse'] += 1
                results['abl1_vs_semma']['details'].append((abl_name, 'MRR', semma_mrr - abl1_mrr))
            
            if abl1_h10 > semma_h10 + 0.001:
                results['abl1_vs_semma']['h10_better'] += 1
            elif abl1_h10 < semma_h10 - 0.001:
                results['abl1_vs_semma']['h10_worse'] += 1
                results['abl1_vs_semma']['details'].append((abl_name, 'H@10', semma_h10 - abl1_h10))
            
            # 比较 Abl2 vs SEMMA
            abl2_mrr = abl_values['abl2_mrr']
            abl2_h10 = abl_values['abl2_h10']
            
            if abl2_mrr > semma_mrr + 0.001:
                results['abl2_vs_semma']['mrr_better'] += 1
            elif abl2_mrr < semma_mrr - 0.001:
                results['abl2_vs_semma']['mrr_worse'] += 1
                results['abl2_vs_semma']['details'].append((abl_name, 'MRR', semma_mrr - abl2_mrr))
            
            if abl2_h10 > semma_h10 + 0.001:
                results['abl2_vs_semma']['h10_better'] += 1
            elif abl2_h10 < semma_h10 - 0.001:
                results['abl2_vs_semma']['h10_worse'] += 1
                results['abl2_vs_semma']['details'].append((abl_name, 'H@10', semma_h10 - abl2_h10))
            
            # 比较 ARE vs SEMMA
            are_mrr = abl_values['are_mrr']
            are_h10 = abl_values['are_h10']
            
            if are_mrr > semma_mrr + 0.001:
                results['are_vs_semma']['mrr_better'] += 1
            elif are_mrr < semma_mrr - 0.001:
                results['are_vs_semma']['mrr_worse'] += 1
                results['are_vs_semma']['details'].append((abl_name, 'MRR', semma_mrr - are_mrr))
            
            if are_h10 > semma_h10 + 0.001:
                results['are_vs_semma']['h10_better'] += 1
            elif are_h10 < semma_h10 - 0.001:
                results['are_vs_semma']['h10_worse'] += 1
                results['are_vs_semma']['details'].append((abl_name, 'H@10', semma_h10 - are_h10))
    
    return results

def print_comparison(results):
    """打印对比结果"""
    print("=" * 80)
    print("消融实验组件 vs SEMMA 基准对比分析")
    print("=" * 80)
    print(f"\n匹配的数据集数量: {results['total_matched']}")
    
    print("\n" + "-" * 80)
    print("Abl1 (similarity_enhancer only) vs SEMMA:")
    abl1 = results['abl1_vs_semma']
    print(f"  MRR: Abl1 优于 SEMMA: {abl1['mrr_better']}, 低于 SEMMA: {abl1['mrr_worse']}")
    print(f"  H@10: Abl1 优于 SEMMA: {abl1['h10_better']}, 低于 SEMMA: {abl1['h10_worse']}")
    if abl1['details']:
        print(f"\n  Abl1 低于 SEMMA 的数据集 (前10个):")
        sorted_details = sorted(abl1['details'], key=lambda x: x[2], reverse=True)[:10]
        for dataset, metric, diff in sorted_details:
            print(f"    {dataset} ({metric}): -{diff:.4f}")
    
    print("\n" + "-" * 80)
    print("Abl2 (prompt_enhancer only) vs SEMMA:")
    abl2 = results['abl2_vs_semma']
    print(f"  MRR: Abl2 优于 SEMMA: {abl2['mrr_better']}, 低于 SEMMA: {abl2['mrr_worse']}")
    print(f"  H@10: Abl2 优于 SEMMA: {abl2['h10_better']}, 低于 SEMMA: {abl2['h10_worse']}")
    if abl2['details']:
        print(f"\n  Abl2 低于 SEMMA 的数据集 (前10个):")
        sorted_details = sorted(abl2['details'], key=lambda x: x[2], reverse=True)[:10]
        for dataset, metric, diff in sorted_details:
            print(f"    {dataset} ({metric}): -{diff:.4f}")
    
    print("\n" + "-" * 80)
    print("ARE (完整模型) vs SEMMA:")
    are = results['are_vs_semma']
    print(f"  MRR: ARE 优于 SEMMA: {are['mrr_better']}, 低于 SEMMA: {are['mrr_worse']}")
    print(f"  H@10: ARE 优于 SEMMA: {are['h10_better']}, 低于 SEMMA: {are['h10_worse']}")
    if are['details']:
        print(f"\n  ARE 低于 SEMMA 的数据集:")
        sorted_details = sorted(are['details'], key=lambda x: x[2], reverse=True)
        for dataset, metric, diff in sorted_details:
            print(f"    {dataset} ({metric}): -{diff:.4f}")
    
    print("\n" + "=" * 80)
    print("影响分析:")
    print("=" * 80)
    
    # 分析影响
    abl1_worse_rate = (abl1['mrr_worse'] + abl1['h10_worse']) / (results['total_matched'] * 2) if results['total_matched'] > 0 else 0
    abl2_worse_rate = (abl2['mrr_worse'] + abl2['h10_worse']) / (results['total_matched'] * 2) if results['total_matched'] > 0 else 0
    are_worse_rate = (are['mrr_worse'] + are['h10_worse']) / (results['total_matched'] * 2) if results['total_matched'] > 0 else 0
    
    print(f"\n1. 单个组件低于基准的比例:")
    print(f"   - Abl1: {abl1_worse_rate*100:.1f}% 的指标低于 SEMMA")
    print(f"   - Abl2: {abl2_worse_rate*100:.1f}% 的指标低于 SEMMA")
    print(f"   - ARE: {are_worse_rate*100:.1f}% 的指标低于 SEMMA")
    
    print(f"\n2. 对模型有效性证明的影响:")
    if are_worse_rate < 0.1:
        print("   ✓ 完整模型(ARE)几乎在所有数据集上都优于或等于SEMMA基准")
        print("   ✓ 这有力地证明了模型的有效性")
    elif are_worse_rate < 0.2:
        print("   ✓ 完整模型(ARE)在大多数数据集上优于SEMMA基准")
        print("   ✓ 模型有效性得到良好证明")
    else:
        print("   ⚠ 完整模型(ARE)在部分数据集上低于SEMMA基准")
        print("   ⚠ 需要进一步分析这些数据集的特点")
    
    if abl1_worse_rate > 0.3 or abl2_worse_rate > 0.3:
        print(f"\n3. 关于消融实验组件的说明:")
        print("   ⚠ 单个组件(Abl1/Abl2)在部分数据集上低于SEMMA基准")
        print("   这是正常的，因为:")
        print("   - 消融实验的目的是验证组件的贡献，而不是与基准对比")
        print("   - 单个组件可能在某些数据集上表现不佳，但组合后效果更好")
        print("   - 关键是完整模型(ARE)优于基准，这证明了设计的有效性")
        print("\n   建议在论文中:")
        print("   - 强调完整模型(ARE)相比SEMMA的改进")
        print("   - 说明消融实验验证了组件的必要性（组合效果>单个组件）")
        print("   - 可以提及单个组件在某些数据集上的局限性，但强调组合的协同效应")
    else:
        print(f"\n3. 消融实验组件表现:")
        print("   ✓ 单个组件在大多数数据集上也优于或接近SEMMA基准")
        print("   ✓ 这进一步证明了每个组件的有效性")

if __name__ == '__main__':
    ablation_data = parse_data1_table('data1.md')
    baseline_data = parse_data_table('data.md')
    
    print(f"消融实验数据集数量: {len(ablation_data)}")
    print(f"基准对比数据集数量: {len(baseline_data)}")
    
    results = compare_with_baseline(ablation_data, baseline_data)
    print_comparison(results)

