#!/usr/bin/env python
"""
补充Trial 2缺失的两个数据集评估（FB15k237和WN18RR）
评估完成后更新完整结果到数据库
"""

import os
import sys
import subprocess
import re
import json
import optuna
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_metrics_from_output(output):
    """从输出中解析MRR和Hits@10"""
    metrics = {}
    
    # 查找MRR
    mrr_patterns = [
        r'mrr(?:-tail)?[:\s]+(\d+\.\d+)',
        r'mrr[:\s]+(\d+\.\d+)',
    ]
    
    for pattern in mrr_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                metrics['mrr'] = float(matches[-1])
                break
            except:
                pass
    
    # 查找Hits@10
    hits10_patterns = [
        r'hits@10(?:-tail)?[:\s]+(\d+\.\d+)',
        r'hits@10[:\s]+(\d+\.\d+)',
    ]
    
    for pattern in hits10_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                metrics['hits@10'] = float(matches[-1])
                break
            except:
                pass
    
    return metrics if 'mrr' in metrics and 'hits@10' in metrics else None

def evaluate_dataset(checkpoint_path, dataset_name, dataset_type="transductive"):
    """使用checkpoint在数据集上评估"""
    project_root = "/T20030104/ynj/semma"
    
    if dataset_type == "transductive":
        config_path = "config/transductive/inference-fb.yaml"
        cmd = [
            "python", "script/run.py",
            "-c", config_path,
            "--dataset", dataset_name,
            "--ckpt", checkpoint_path,
            "--gpus", "[0]",
            "--epochs", "0",
            "--bpe", "null"
        ]
    else:  # inductive
        config_path = "config/inductive/inference.yaml"
        cmd = [
            "python", "script/run.py",
            "-c", config_path,
            "--dataset", dataset_name,
            "--version", "v1",
            "--ckpt", checkpoint_path,
            "--gpus", "[0]",
            "--epochs", "0",
            "--bpe", "null"
        ]
    
    try:
        print(f"  评估 {dataset_name}...")
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )
        
        if result.returncode == 0:
            metrics = parse_metrics_from_output(result.stdout + result.stderr)
            if metrics:
                print(f"    ✓ MRR: {metrics['mrr']:.4f}, Hits@10: {metrics['hits@10']:.4f}")
                return metrics
            else:
                print(f"    ⚠ 无法解析指标")
                print(f"    输出: {result.stdout[-500:]}")
        else:
            print(f"    ✗ 评估失败")
            print(f"    错误: {result.stderr[-500:]}")
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ 评估超时")
    except Exception as e:
        print(f"    ✗ 评估异常: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def complete_trial2_evaluation():
    """补充Trial 2缺失的数据集评估"""
    checkpoint_path = "/T20030104/ynj/semma/output/Ultra/JointDataset/2025-11-07-22-27-49/model_epoch_9.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return False
    
    print("="*70)
    print("🔧 补充Trial 2缺失的数据集评估")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # 读取现有的Trial 2结果
    result_file = "/T20030104/ynj/semma/optuna_tune/trials/trial_2/result.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            existing_result = json.load(f)
        existing_eval_results = existing_result.get('eval_results', {})
        print(f"\n📋 现有评估结果 ({len(existing_eval_results)}个数据集):")
        for dataset, metrics in existing_eval_results.items():
            if isinstance(metrics, dict) and 'mrr' in metrics:
                print(f"  ✓ {dataset}: MRR={metrics['mrr']:.4f}, Hits@10={metrics['hits@10']:.4f}")
    else:
        existing_eval_results = {}
        print("\n⚠ 没有找到现有的评估结果")
    
    # 需要评估的数据集
    missing_datasets = [
        ("FB15k237", "transductive"),
        ("WN18RR", "transductive"),
    ]
    
    # 检查哪些数据集缺失
    datasets_to_evaluate = []
    for dataset_name, dataset_type in missing_datasets:
        if dataset_name not in existing_eval_results or not isinstance(existing_eval_results[dataset_name], dict) or 'mrr' not in existing_eval_results[dataset_name]:
            datasets_to_evaluate.append((dataset_name, dataset_type))
            print(f"  ⚠ {dataset_name}: 缺失")
        else:
            print(f"  ✓ {dataset_name}: 已有结果")
    
    if not datasets_to_evaluate:
        print("\n✅ 所有数据集都已评估完成！")
        return True
    
    # 运行评估
    print(f"\n📊 开始评估缺失的数据集 ({len(datasets_to_evaluate)}个)...")
    new_results = {}
    
    for dataset_name, dataset_type in datasets_to_evaluate:
        metrics = evaluate_dataset(checkpoint_path, dataset_name, dataset_type)
        if metrics:
            new_results[dataset_name] = metrics
    
    if not new_results:
        print("\n❌ 没有获得任何新的评估结果")
        return False
    
    # 合并结果
    all_eval_results = {**existing_eval_results, **new_results}
    
    # 移除统计信息，只保留数据集结果
    dataset_results = {k: v for k, v in all_eval_results.items() 
                      if isinstance(v, dict) and 'mrr' in v and 'hits@10' in v}
    
    if len(dataset_results) < 6:
        print(f"\n⚠ 警告: 只有{len(dataset_results)}个数据集的结果，期望6个")
        print(f"  数据集: {list(dataset_results.keys())}")
    
    # 计算综合分数
    total_mrr = sum(m['mrr'] for m in dataset_results.values())
    total_hits10 = sum(m['hits@10'] for m in dataset_results.values())
    count = len(dataset_results)
    
    avg_mrr = total_mrr / count
    avg_hits10 = total_hits10 / count
    score = 0.6 * avg_mrr + 0.4 * avg_hits10
    
    print(f"\n📈 完整评估结果:")
    print(f"  评估数据集数: {len(dataset_results)}/6")
    for dataset, metrics in sorted(dataset_results.items()):
        print(f"  {dataset}: MRR={metrics['mrr']:.4f}, Hits@10={metrics['hits@10']:.4f}")
    print(f"  平均MRR: {avg_mrr:.4f}")
    print(f"  平均Hits@10: {avg_hits10:.4f}")
    print(f"  综合分数: {score:.4f}")
    
    # 更新Optuna数据库
    study_db = "/T20030104/ynj/semma/optuna_tune/trials/study.db"
    print(f"\n💾 更新Optuna数据库...")
    
    try:
        study = optuna.load_study(
            study_name="enhancement_params_tuning",
            storage=f"sqlite:///{study_db}"
        )
        
        trial_2 = study.trials[2]
        params = trial_2.params
        
        conn = sqlite3.connect(study_db)
        cursor = conn.cursor()
        
        new_value = -score
        
        cursor.execute("SELECT trial_id FROM trials WHERE number = 2")
        trial_id = cursor.fetchone()[0]
        
        # 更新trial值
        cursor.execute("""
            UPDATE trial_values 
            SET value = ?, value_type = 'FINITE'
            WHERE trial_id = ? AND objective = 0
        """, (new_value, trial_id))
        
        if cursor.rowcount == 0:
            cursor.execute("""
                INSERT INTO trial_values (trial_id, objective, value, value_type)
                VALUES (?, 0, ?, 'FINITE')
            """, (trial_id, new_value))
        
        conn.commit()
        conn.close()
        
        print(f"  ✓ 成功更新trial 2:")
        print(f"    值: {new_value:.4f} (对应分数: {score:.4f})")
        print(f"    参数: {params}")
        
    except Exception as e:
        print(f"❌ 更新数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存完整结果文件
    print(f"\n📁 保存完整trial结果文件...")
    trial_dir = "/T20030104/ynj/semma/optuna_tune/trials/trial_2"
    os.makedirs(trial_dir, exist_ok=True)
    
    result = {
        'trial_number': 2,
        'score': score,
        'params': params,
        'eval_results': {
            **dataset_results,
            'avg_mrr': avg_mrr,
            'avg_hits10': avg_hits10,
            'score': score
        },
        'timestamp': datetime.now().isoformat(),
        'manually_added': True,
        'note': '使用epoch 9 checkpoint评估（epoch 10未完成），已补充完整6个数据集'
    }
    
    result_file = os.path.join(trial_dir, 'result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✓ 完整结果已保存到: {result_file}")
    
    # 验证更新
    print(f"\n🔍 验证更新...")
    study = optuna.load_study(
        study_name="enhancement_params_tuning",
        storage=f"sqlite:///{study_db}"
    )
    
    if len(study.trials) > 2:
        trial_2_updated = study.trials[2]
        print(f"  Trial 2新状态:")
        print(f"    状态: {trial_2_updated.state}")
        print(f"    值: {trial_2_updated.value:.4f} (对应分数: {-trial_2_updated.value:.4f})")
        print(f"    参数: {trial_2_updated.params}")
    
    if study.best_trial:
        print(f"\n🏆 当前最佳trial:")
        print(f"  Trial ID: {study.best_trial.number}")
        print(f"  最佳值: {-study.best_trial.value:.4f}")
        print(f"  参数: {study.best_trial.params}")
    
    print("\n" + "="*70)
    print("✅ Trial 2完整评估完成！")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = complete_trial2_evaluation()
    sys.exit(0 if success else 1)

