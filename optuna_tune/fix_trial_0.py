#!/usr/bin/env python
"""
手动修复第一个trial的结果
从评估日志中提取指标，计算分数，并更新Optuna数据库
"""

import os
import sys
import re
import json
import optuna
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_metrics_from_log(log_file):
    """从日志文件中解析MRR和Hits@10"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 查找test集的MRR和Hits@10（通常是最后一个）
    mrr_pattern = r'mrr(?:-tail)?[:\s]+(\d+\.\d+)'
    hits10_pattern = r'hits@10(?:-tail)?[:\s]+(\d+\.\d+)'
    
    mrr_matches = re.findall(mrr_pattern, content, re.IGNORECASE)
    hits10_matches = re.findall(hits10_pattern, content, re.IGNORECASE)
    
    if mrr_matches and hits10_matches:
        # 取最后一个匹配（通常是test集的结果）
        mrr = float(mrr_matches[-1])
        hits10 = float(hits10_matches[-1])
        return {'mrr': mrr, 'hits@10': hits10}
    
    return None

def find_evaluation_logs():
    """查找所有评估日志文件"""
    eval_dir = "/T20030104/ynj/semma/v3_vip_output/Ultra"
    if not os.path.exists(eval_dir):
        return {}
    
    # 代表性数据集列表（与调参脚本中一致）
    representative_datasets = [
        ("FB15k237", None, "transductive"),
        ("WN18RR", None, "transductive"),
        ("CoDExSmall", None, "transductive"),
        ("FB15k237Inductive", "v1", "inductive"),
        ("WN18RRInductive", "v1", "inductive"),
        ("NELLInductive", "v1", "inductive"),
    ]
    
    results = {}
    
    for dataset_name, version, dataset_type in representative_datasets:
        dataset_dir = os.path.join(eval_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"  ⚠ {dataset_name}: 目录不存在")
            continue
        
        # 查找最新的日志文件
        log_files = list(Path(dataset_dir).rglob("log.txt"))
        if not log_files:
            print(f"  ⚠ {dataset_name}: 没有找到日志文件")
            continue
        
        # 获取最新的日志文件（按修改时间）
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        
        metrics = parse_metrics_from_log(str(latest_log))
        if metrics:
            results[dataset_name] = metrics
            print(f"  ✓ {dataset_name}: MRR={metrics['mrr']:.4f}, Hits@10={metrics['hits@10']:.4f}")
        else:
            print(f"  ✗ {dataset_name}: 无法解析指标")
    
    return results

def calculate_score(eval_results):
    """计算综合分数"""
    if not eval_results:
        return 0.0
    
    total_mrr = 0.0
    total_hits10 = 0.0
    count = 0
    
    for dataset_name, metrics in eval_results.items():
        if isinstance(metrics, dict) and 'mrr' in metrics and 'hits@10' in metrics:
            total_mrr += metrics['mrr']
            total_hits10 += metrics['hits@10']
            count += 1
    
    if count == 0:
        return 0.0
    
    avg_mrr = total_mrr / count
    avg_hits10 = total_hits10 / count
    
    # 加权平均：MRR权重0.6，Hits@10权重0.4
    score = 0.6 * avg_mrr + 0.4 * avg_hits10
    
    return score

def fix_trial_0():
    """修复第一个trial的结果"""
    study_db = "/T20030104/ynj/semma/optuna_tune/trials/study.db"
    
    if not os.path.exists(study_db):
        print(f"❌ Optuna数据库不存在: {study_db}")
        return False
    
    print("="*70)
    print("🔧 修复Trial 0的结果")
    print("="*70)
    
    # 1. 加载study
    try:
        study = optuna.load_study(
            study_name="enhancement_params_tuning",
            storage=f"sqlite:///{study_db}"
        )
    except Exception as e:
        print(f"❌ 加载study失败: {e}")
        return False
    
    # 2. 检查trial 0的状态
    if len(study.trials) == 0:
        print("❌ 没有找到任何trials")
        return False
    
    trial_0 = study.trials[0]
    print(f"\n📋 Trial 0当前状态:")
    print(f"  状态: {trial_0.state}")
    print(f"  值: {trial_0.value}")
    print(f"  参数: {trial_0.params}")
    
    if trial_0.state == optuna.trial.TrialState.COMPLETE and trial_0.value != float('inf'):
        print("\n✅ Trial 0已经成功完成，无需修复")
        return True
    
    # 3. 从评估日志中提取结果
    print(f"\n📊 从评估日志中提取指标...")
    eval_results = find_evaluation_logs()
    
    if not eval_results:
        print("❌ 没有找到有效的评估结果")
        return False
    
    # 4. 计算综合分数
    score = calculate_score(eval_results)
    print(f"\n📈 计算结果:")
    print(f"  评估数据集数: {len(eval_results)}")
    avg_mrr = sum(m['mrr'] for m in eval_results.values()) / len(eval_results)
    avg_hits10 = sum(m['hits@10'] for m in eval_results.values()) / len(eval_results)
    print(f"  平均MRR: {avg_mrr:.4f}")
    print(f"  平均Hits@10: {avg_hits10:.4f}")
    print(f"  综合分数: {score:.4f}")
    
    # 5. 更新trial 0的结果
    print(f"\n💾 更新Optuna数据库...")
    try:
        # 使用study的内部API来更新trial
        # 注意：这需要直接操作数据库，因为Optuna不提供直接修改已完成trial的API
        
        import sqlite3
        conn = sqlite3.connect(study_db)
        cursor = conn.cursor()
        
        # 更新trial的值
        # Optuna使用负值因为我们最小化（但实际是最大化score）
        new_value = -score
        
        # 查找trial_id
        cursor.execute("SELECT trial_id FROM trials WHERE number = 0")
        trial_id_result = cursor.fetchone()
        if not trial_id_result:
            print("❌ 找不到trial 0的ID")
            conn.close()
            return False
        
        trial_id = trial_id_result[0]
        
        # 更新或插入trial值（使用FINITE类型）
        cursor.execute("""
            UPDATE trial_values 
            SET value = ?, value_type = 'FINITE'
            WHERE trial_id = ? AND objective = 0
        """, (new_value, trial_id))
        
        # 如果trial_values表中没有记录，插入一条
        if cursor.rowcount == 0:
            cursor.execute("""
                INSERT INTO trial_values (trial_id, objective, value, value_type)
                VALUES (?, 0, ?, 'FINITE')
            """, (trial_id, new_value))
        
        # 更新trial状态为COMPLETE
        cursor.execute("""
            UPDATE trials 
            SET state = 'COMPLETE' 
            WHERE trial_id = ?
        """, (trial_id,))
        
        conn.commit()
        conn.close()
        
        print(f"  ✓ 成功更新trial 0:")
        print(f"    新值: {new_value:.4f} (对应分数: {score:.4f})")
        print(f"    状态: COMPLETE")
        
    except Exception as e:
        print(f"❌ 更新数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 保存trial结果文件
    print(f"\n📁 保存trial结果文件...")
    trial_dir = "/T20030104/ynj/semma/optuna_tune/trials/trial_0"
    os.makedirs(trial_dir, exist_ok=True)
    
    result = {
        'trial_number': 0,
        'score': score,
        'params': trial_0.params,
        'eval_results': {
            **eval_results,
            'avg_mrr': avg_mrr,
            'avg_hits10': avg_hits10,
            'score': score
        },
        'timestamp': datetime.now().isoformat(),
        'fixed': True  # 标记为手动修复
    }
    
    result_file = os.path.join(trial_dir, 'result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✓ 结果已保存到: {result_file}")
    
    # 7. 验证更新
    print(f"\n🔍 验证更新...")
    study = optuna.load_study(
        study_name="enhancement_params_tuning",
        storage=f"sqlite:///{study_db}"
    )
    
    trial_0_updated = study.trials[0]
    print(f"  Trial 0新状态:")
    print(f"    状态: {trial_0_updated.state}")
    print(f"    值: {trial_0_updated.value:.4f} (对应分数: {-trial_0_updated.value:.4f})")
    
    if study.best_trial:
        print(f"\n🏆 当前最佳trial:")
        print(f"  Trial ID: {study.best_trial.number}")
        print(f"  最佳值: {-study.best_trial.value:.4f}")
        print(f"  参数: {study.best_trial.params}")
    
    print("\n" + "="*70)
    print("✅ Trial 0修复完成！")
    print("="*70)
    print("\n💡 提示:")
    print("  - 修复后的结果已保存到数据库")
    print("  - 后续trials会基于这个结果继续优化")
    print("  - 可以使用 optuna-dashboard 查看更新后的结果")
    
    return True

if __name__ == "__main__":
    success = fix_trial_0()
    sys.exit(0 if success else 1)

