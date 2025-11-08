#!/usr/bin/env python
"""
验证Trial 2的采样是否基于正确的历史数据
如果发现问题，提供修复建议
"""

import os
import sys
import optuna
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_trial2_sampling():
    """验证Trial 2的采样是否正确"""
    study_db = "/T20030104/ynj/semma/optuna_tune/trials/study.db"
    
    if not os.path.exists(study_db):
        print(f"❌ Optuna数据库不存在: {study_db}")
        return False
    
    print("="*70)
    print("🔍 验证Trial 2的采样逻辑")
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
    
    # 2. 检查已完成的trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n📊 已完成的trials ({len(completed_trials)}个):")
    for t in completed_trials:
        value = -t.value if t.value != float('inf') else 'inf'
        print(f"  Trial {t.number}: threshold={t.params['similarity_threshold_init']:.2f}, "
              f"strength={t.params['enhancement_strength_init']:.2f}, value={value:.4f}")
    
    # 3. 检查Trial 2的状态
    if len(study.trials) < 3:
        print("\n⚠ Trial 2还不存在")
        return True
    
    trial_2 = study.trials[2]
    print(f"\n📋 Trial 2状态:")
    print(f"  状态: {trial_2.state}")
    print(f"  参数: {trial_2.params}")
    
    # 4. 检查时间戳
    conn = sqlite3.connect(study_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT datetime_start FROM trials WHERE number = 2")
    trial2_start = cursor.fetchone()[0]
    
    cursor.execute("SELECT datetime_complete FROM trials WHERE number = 1")
    trial1_complete = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n⏰ 时间戳分析:")
    print(f"  Trial 1完成时间: {trial1_complete}")
    print(f"  Trial 2开始时间: {trial2_start}")
    
    # 5. 分析采样逻辑
    print(f"\n🔬 采样逻辑分析:")
    
    if trial2_start and trial1_complete:
        if trial2_start < trial1_complete:
            print(f"  ⚠ 警告: Trial 2在Trial 1修复之前开始！")
            print(f"     这意味着Trial 2的参数可能基于错误的历史数据采样")
            print(f"     但是，Optuna的TPESampler在每次suggest时会重新加载study")
            print(f"     所以如果Trial 2的参数是在Trial 1修复之后才被suggest的，就没问题")
            
            # 检查Trial 2的参数是否合理
            t2_threshold = trial_2.params['similarity_threshold_init']
            t2_strength = trial_2.params['enhancement_strength_init']
            
            # 基于正确的历史数据，TPESampler应该会：
            # - 探索最佳参数附近的区域（Trial 1: 0.85, 0.09）
            # - 或者探索其他有希望的区域
            
            best_trial = study.best_trial
            best_threshold = best_trial.params['similarity_threshold_init']
            best_strength = best_trial.params['enhancement_strength_init']
            
            print(f"\n  📈 基于正确的历史数据:")
            print(f"     最佳参数: threshold={best_threshold:.2f}, strength={best_strength:.2f}")
            print(f"     Trial 2参数: threshold={t2_threshold:.2f}, strength={t2_strength:.2f}")
            
            # 计算参数距离
            threshold_diff = abs(t2_threshold - best_threshold)
            strength_diff = abs(t2_strength - best_strength)
            
            if threshold_diff > 0.3 or strength_diff > 0.1:
                print(f"\n  ⚠ Trial 2的参数与最佳参数差异较大")
                print(f"     这可能表明它基于了错误的历史数据，或者是在探索阶段")
                print(f"     建议: 如果Trial 2还在预训练阶段，可以考虑重新采样")
            else:
                print(f"\n  ✓ Trial 2的参数看起来合理（在探索最佳参数附近）")
        else:
            print(f"  ✓ Trial 2在Trial 1修复之后开始")
            print(f"     应该基于正确的历史数据采样")
    
    # 6. 检查TPESampler的配置
    print(f"\n⚙️ TPESampler配置:")
    print(f"  n_startup_trials: 2 (前2个trial随机采样)")
    print(f"  当前完成trial数: {len(completed_trials)}")
    
    if len(completed_trials) >= 2:
        print(f"  ✓ 有足够的已完成trials，TPESampler应该正常工作")
    else:
        print(f"  ⚠ 只有{len(completed_trials)}个完成的trial，TPESampler可能还在随机采样阶段")
    
    # 7. 结论和建议
    print(f"\n" + "="*70)
    print("📝 结论和建议")
    print("="*70)
    
    if trial2_start and trial1_complete and trial2_start < trial1_complete:
        print("⚠ 发现潜在问题:")
        print("  - Trial 2在Trial 1修复之前开始")
        print("  - 但Optuna的TPESampler在每次suggest时会重新加载study")
        print("  - 所以如果参数是在修复之后suggest的，应该没问题")
        print("\n💡 建议:")
        print("  1. 检查Trial 2的参数是否合理（已完成）")
        print("  2. 如果Trial 2还在预训练阶段，可以继续观察")
        print("  3. 如果Trial 2的参数明显不合理，可以考虑:")
        print("     - 等待Trial 2完成，然后评估结果")
        print("     - 如果结果不好，可以在后续trials中纠正")
    else:
        print("✓ 没有发现明显问题")
        print("  - Trial 2应该基于正确的历史数据采样")
        print("  - 可以继续运行，无需担心")
    
    return True

if __name__ == "__main__":
    success = verify_trial2_sampling()
    sys.exit(0 if success else 1)

