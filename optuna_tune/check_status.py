#!/usr/bin/env python
"""
检查当前调参状态和诊断问题
"""

import os
import sys
import sqlite3
import json
import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_optuna_study():
    """检查Optuna study状态"""
    study_db = "/T20030104/ynj/semma/optuna_tune/trials/study.db"
    if not os.path.exists(study_db):
        print("❌ Optuna数据库不存在")
        return
    
    try:
        import optuna
        study = optuna.load_study(study_name="enhancement_params_tuning", 
                                  storage=f"sqlite:///{study_db}")
        
        print(f"\n📊 Optuna Study状态:")
        print(f"  总trials数: {len(study.trials)}")
        print(f"  完成trials: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)}")
        print(f"  失败trials: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)}")
        print(f"  运行中trials: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING)}")
        
        if study.best_trial:
            print(f"\n🏆 最佳trial:")
            print(f"  Trial ID: {study.best_trial.number}")
            print(f"  最佳值: {-study.best_trial.value:.4f}" if study.best_trial.value != float('inf') else "  Best value: inf (失败)")
            print(f"  参数: {study.best_trial.params}")
        
        print(f"\n📋 所有trials:")
        for trial in study.trials[:5]:  # 只显示前5个
            state_str = "✓" if trial.state == optuna.trial.TrialState.COMPLETE else \
                       "✗" if trial.state == optuna.trial.TrialState.FAIL else \
                       "⏳" if trial.state == optuna.trial.TrialState.RUNNING else "?"
            value_str = f"{-trial.value:.4f}" if trial.value != float('inf') else "inf"
            print(f"  {state_str} Trial {trial.number}: value={value_str}, params={trial.params}")
            
    except Exception as e:
        print(f"❌ 读取Optuna study失败: {e}")

def check_trial_results():
    """检查trial结果文件"""
    trials_dir = "/T20030104/ynj/semma/optuna_tune/trials"
    trial_dirs = glob.glob(os.path.join(trials_dir, "trial_*"))
    
    print(f"\n📁 Trial结果文件:")
    print(f"  找到 {len(trial_dirs)} 个trial目录")
    
    for trial_dir in sorted(trial_dirs)[:5]:  # 只显示前5个
        result_file = os.path.join(trial_dir, "result.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                print(f"\n  ✓ {os.path.basename(trial_dir)}:")
                print(f"    分数: {result.get('score', 'N/A')}")
                print(f"    参数: {result.get('params', {})}")
                if 'eval_results' in result:
                    eval_res = result['eval_results']
                    if isinstance(eval_res, dict):
                        datasets = [k for k in eval_res.keys() if k not in ['avg_mrr', 'avg_hits10', 'score']]
                        print(f"    评估数据集数: {len(datasets)}")
            except Exception as e:
                print(f"  ✗ {os.path.basename(trial_dir)}: 读取失败 - {e}")
        else:
            print(f"  ⚠ {os.path.basename(trial_dir)}: 没有result.json")

def check_checkpoints():
    """检查checkpoint文件"""
    output_dir = "/T20030104/ynj/semma/output/Ultra/JointDataset"
    if not os.path.exists(output_dir):
        print(f"\n❌ 输出目录不存在: {output_dir}")
        return
    
    print(f"\n💾 Checkpoint文件:")
    checkpoints = glob.glob(os.path.join(output_dir, "**/model_epoch_*.pth"), recursive=True)
    checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)
    
    print(f"  找到 {len(checkpoints)} 个checkpoint文件")
    if checkpoints:
        print(f"  最新的checkpoint:")
        for cp in checkpoints[:3]:
            mtime = os.path.getmtime(cp)
            from datetime import datetime
            print(f"    {cp} (修改时间: {datetime.fromtimestamp(mtime)})")

def check_evaluation_results():
    """检查评估结果"""
    eval_dir = "/T20030104/ynj/semma/v3_vip_output/Ultra"
    if not os.path.exists(eval_dir):
        print(f"\n❌ 评估目录不存在: {eval_dir}")
        return
    
    print(f"\n📈 评估结果:")
    datasets = [d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))]
    print(f"  找到 {len(datasets)} 个数据集评估目录")
    
    for dataset in sorted(datasets)[:5]:  # 只显示前5个
        log_files = glob.glob(os.path.join(eval_dir, dataset, "**/log.txt"), recursive=True)
        if log_files:
            latest_log = max(log_files, key=os.path.getmtime)
            print(f"  ✓ {dataset}: 有日志文件")
            # 尝试读取最后几行
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    # 查找MRR和Hits@10
                    for line in reversed(lines[-50:]):
                        if 'mrr' in line.lower() and 'test' in line.lower():
                            print(f"    最新结果: {line.strip()}")
                            break
            except:
                pass

if __name__ == "__main__":
    print("="*70)
    print("🔍 EnhancedUltra调参状态检查")
    print("="*70)
    
    check_optuna_study()
    check_trial_results()
    check_checkpoints()
    check_evaluation_results()
    
    print("\n" + "="*70)
    print("✅ 检查完成")
    print("="*70)

