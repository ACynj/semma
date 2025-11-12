#!/usr/bin/env python
"""
使用Trial 2的epoch 9 checkpoint运行评估，并更新结果
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

def evaluate_with_checkpoint(checkpoint_path, dataset_name, version=None, dataset_type="transductive"):
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
            "--version", version,
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
        else:
            print(f"    ✗ 评估失败")
            print(f"    错误: {result.stderr[-200:]}")
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ 评估超时")
    except Exception as e:
        print(f"    ✗ 评估异常: {e}")
    
    return None

def evaluate_trial2_with_epoch9():
    """使用Trial 2的epoch 9 checkpoint运行评估"""
    checkpoint_path = "/T20030104/ynj/semma/output/Ultra/JointDataset/2025-11-07-22-27-49/model_epoch_9.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return False
    
    print("="*70)
    print("🔧 使用Trial 2的Epoch 9 Checkpoint运行评估")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # 代表性数据集列表
    representative_datasets = [
        ("FB15k237", None, "transductive"),
        ("WN18RR", None, "transductive"),
        ("CoDExSmall", None, "transductive"),
        ("FB15k237Inductive", "v1", "inductive"),
        ("WN18RRInductive", "v1", "inductive"),
        ("NELLInductive", "v1", "inductive"),
    ]
    
    # 运行评估
    print(f"\n📊 开始评估...")
    eval_results = {}
    
    for dataset_name, version, dataset_type in representative_datasets:
        metrics = evaluate_with_checkpoint(checkpoint_path, dataset_name, version, dataset_type)
        if metrics:
            eval_results[dataset_name] = metrics
    
    if not eval_results:
        print("\n❌ 没有获得任何评估结果")
        return False
    
    # 计算综合分数
    total_mrr = sum(m['mrr'] for m in eval_results.values())
    total_hits10 = sum(m['hits@10'] for m in eval_results.values())
    count = len(eval_results)
    
    avg_mrr = total_mrr / count
    avg_hits10 = total_hits10 / count
    score = 0.6 * avg_mrr + 0.4 * avg_hits10
    
    print(f"\n📈 评估结果:")
    print(f"  评估数据集数: {len(eval_results)}")
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
        
        # 更新trial状态
        cursor.execute("""
            UPDATE trials 
            SET state = 'COMPLETE' 
            WHERE trial_id = ?
        """, (trial_id,))
        
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
    
    # 保存结果文件
    print(f"\n📁 保存trial结果文件...")
    trial_dir = "/T20030104/ynj/semma/optuna_tune/trials/trial_2"
    os.makedirs(trial_dir, exist_ok=True)
    
    result = {
        'trial_number': 2,
        'score': score,
        'params': params,
        'eval_results': {
            **eval_results,
            'avg_mrr': avg_mrr,
            'avg_hits10': avg_hits10,
            'score': score
        },
        'timestamp': datetime.now().isoformat(),
        'manually_added': True,
        'note': '使用epoch 9 checkpoint评估（epoch 10未完成）'
    }
    
    result_file = os.path.join(trial_dir, 'result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✓ 结果已保存到: {result_file}")
    
    print("\n" + "="*70)
    print("✅ Trial 2评估完成！")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = evaluate_trial2_with_epoch9()
    sys.exit(0 if success else 1)


