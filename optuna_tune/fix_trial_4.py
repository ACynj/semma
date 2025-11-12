#!/usr/bin/env python
"""
手动修复第五次实验（Trial 4）的结果
从评估日志中提取指标，计算分数，并更新Optuna数据库
不会中断当前正在运行的程序
"""

import os
import sys
import re
import json
import optuna
import sqlite3
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

def find_evaluation_logs_for_trial4():
    """查找Trial 4的评估日志文件（第五次实验）"""
    eval_dir = "/T20030104/ynj/semma/v3_vip_output/Ultra"
    if not os.path.exists(eval_dir):
        return {}
    
    # Trial 4的预训练完成时间（用于确认评估日志）
    pretrain_checkpoint = "/T20030104/ynj/semma/output/Ultra/JointDataset/2025-11-08-17-48-54/model_epoch_10.pth"
    if os.path.exists(pretrain_checkpoint):
        pretrain_end_time = os.path.getmtime(pretrain_checkpoint)
    else:
        pretrain_end_time = 0
    
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
        
        # 查找在预训练完成后的评估日志（Trial 4的评估应该在03:08之后）
        log_files = sorted(Path(dataset_dir).rglob("log.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print(f"  ⚠ {dataset_name}: 没有找到日志文件")
            continue
        
        # 查找在预训练完成时间之后的日志（Trial 4的评估）
        latest_log = None
        for log_file in log_files:
            log_time = os.path.getmtime(log_file)
            # 在预训练完成后2小时内
            if log_time > pretrain_end_time - 3600 and log_time < pretrain_end_time + 7200:
                latest_log = log_file
                break
        
        if latest_log is None:
            # 如果没有找到，使用最新的（可能是Trial 4的）
            latest_log = log_files[0]
        
        metrics = parse_metrics_from_log(str(latest_log))
        if metrics:
            results[dataset_name] = metrics
            log_time_str = datetime.fromtimestamp(os.path.getmtime(latest_log)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  ✓ {dataset_name}: MRR={metrics['mrr']:.4f}, Hits@10={metrics['hits@10']:.4f} (日志时间: {log_time_str})")
        else:
            print(f"  ✗ {dataset_name}: 无法解析指标")
    
    return results

def calculate_score(eval_results):
    """计算综合分数"""
    if not eval_results:
        return 0.0, 0.0, 0.0
    
    total_mrr = 0.0
    total_hits10 = 0.0
    count = 0
    
    for dataset_name, metrics in eval_results.items():
        if isinstance(metrics, dict) and 'mrr' in metrics and 'hits@10' in metrics:
            total_mrr += metrics['mrr']
            total_hits10 += metrics['hits@10']
            count += 1
    
    if count == 0:
        return 0.0, 0.0, 0.0
    
    avg_mrr = total_mrr / count
    avg_hits10 = total_hits10 / count
    
    # 加权平均：MRR权重0.6，Hits@10权重0.4
    score = 0.6 * avg_mrr + 0.4 * avg_hits10
    
    return score, avg_mrr, avg_hits10

def fix_trial_4(result_json_path=None):
    """修复Trial 4的结果（第五次实验）
    
    Args:
        result_json_path: 可选，包含trial 4结果的JSON文件路径
                          格式: {"params": {...}, "eval_results": {...}}
    """
    study_db = "/T20030104/ynj/semma/optuna_tune/trials/study.db"
    
    if not os.path.exists(study_db):
        print(f"❌ Optuna数据库不存在: {study_db}")
        return False
    
    print("="*70)
    print("🔧 修复Trial 4的结果（第五次实验）")
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
    
    # 2. 检查trial 4的状态
    if len(study.trials) < 5:
        print("❌ Trial 4还不存在")
        return False
    
    trial_4 = study.trials[4]
    print(f"\n📋 Trial 4当前状态:")
    print(f"  状态: {trial_4.state}")
    print(f"  值: {trial_4.value}")
    print(f"  参数: {trial_4.params}")
    
    # 确认这是第五次实验
    print(f"\n🔍 确认这是第五次实验:")
    print(f"  参数: similarity_threshold_init={trial_4.params['similarity_threshold_init']}, "
          f"enhancement_strength_init={trial_4.params['enhancement_strength_init']}")
    
    # 检查预训练目录
    pretrain_dir = "/T20030104/ynj/semma/output/Ultra/JointDataset/2025-11-08-17-48-54"
    if os.path.exists(pretrain_dir):
        print(f"  ✓ 找到预训练目录: {pretrain_dir}")
        checkpoint = os.path.join(pretrain_dir, "model_epoch_10.pth")
        if os.path.exists(checkpoint):
            print(f"  ✓ 找到epoch 10 checkpoint（预训练完成）")
        else:
            print(f"  ⚠ 未找到epoch 10 checkpoint")
    else:
        print(f"  ⚠ 未找到预训练目录")
    
    if trial_4.state == optuna.trial.TrialState.COMPLETE and trial_4.value != float('inf'):
        print("\n⚠ Trial 4已经完成且有值，将覆盖现有结果")
    
    # 3. 获取参数和评估结果
    params = None
    eval_results = None
    
    # 如果提供了JSON文件，从中读取
    if result_json_path and os.path.exists(result_json_path):
        print(f"\n📂 从JSON文件读取结果: {result_json_path}")
        try:
            with open(result_json_path, 'r') as f:
                data = json.load(f)
            params = data.get('params')
            eval_results = data.get('eval_results')
            print(f"  ✓ 成功读取参数和评估结果")
        except Exception as e:
            print(f"  ✗ 读取JSON文件失败: {e}")
            result_json_path = None
    
    # 如果参数未知，从现有trial获取
    if params is None:
        params = trial_4.params
        print(f"\n📝 从数据库读取的参数: {params}")
    
    # 4. 从评估日志中提取结果（如果还没有）
    if eval_results is None:
        print(f"\n📊 尝试从评估日志中提取指标（Trial 4的评估）...")
        eval_results = find_evaluation_logs_for_trial4()
    
    if not eval_results:
        print("\n⚠ 无法从日志中自动提取结果")
        print("  请手动输入评估结果:")
        
        representative_datasets = [
            "FB15k237", "WN18RR", "CoDExSmall",
            "FB15k237Inductive", "WN18RRInductive", "NELLInductive"
        ]
        
        eval_results = {}
        for dataset_name in representative_datasets:
            print(f"\n  {dataset_name}:")
            mrr = input("    MRR: ").strip()
            hits10 = input("    Hits@10: ").strip()
            
            try:
                eval_results[dataset_name] = {
                    'mrr': float(mrr),
                    'hits@10': float(hits10)
                }
            except ValueError:
                print(f"    ⚠ 跳过 {dataset_name}（格式错误）")
    
    if not eval_results:
        print("❌ 没有有效的评估结果")
        return False
    
    # 5. 计算综合分数
    score, avg_mrr, avg_hits10 = calculate_score(eval_results)
    print(f"\n📈 计算结果:")
    print(f"  评估数据集数: {len(eval_results)}")
    print(f"  平均MRR: {avg_mrr:.4f}")
    print(f"  平均Hits@10: {avg_hits10:.4f}")
    print(f"  综合分数: {score:.4f}")
    
    # 6. 更新Optuna数据库
    print(f"\n💾 更新Optuna数据库...")
    try:
        conn = sqlite3.connect(study_db)
        cursor = conn.cursor()
        
        # Optuna使用负值因为我们最小化（但实际是最大化score）
        new_value = -score
        
        # 获取trial_id
        cursor.execute("SELECT trial_id FROM trials WHERE number = 4")
        trial_id_result = cursor.fetchone()
        if not trial_id_result:
            print("❌ 找不到trial 4的ID")
            conn.close()
            return False
        
        trial_id = trial_id_result[0]
        
        # 更新trial值
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
        
        # 更新参数（如果需要）
        cursor.execute("""
            UPDATE trial_params 
            SET param_value = ?
            WHERE trial_id = ? AND param_name = 'similarity_threshold_init'
        """, (str(params['similarity_threshold_init']), trial_id))
        
        cursor.execute("""
            UPDATE trial_params 
            SET param_value = ?
            WHERE trial_id = ? AND param_name = 'enhancement_strength_init'
        """, (str(params['enhancement_strength_init']), trial_id))
        
        conn.commit()
        conn.close()
        
        print(f"  ✓ 成功更新trial 4:")
        print(f"    值: {new_value:.4f} (对应分数: {score:.4f})")
        print(f"    状态: COMPLETE")
        print(f"    参数: {params}")
        
    except Exception as e:
        print(f"❌ 更新数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. 保存trial结果文件
    print(f"\n📁 保存trial结果文件...")
    trial_dir = "/T20030104/ynj/semma/optuna_tune/trials/trial_4"
    os.makedirs(trial_dir, exist_ok=True)
    
    result = {
        'trial_number': 4,
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
        'note': '第五次实验（Trial 4）'
    }
    
    result_file = os.path.join(trial_dir, 'result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✓ 结果已保存到: {result_file}")
    
    # 8. 验证更新
    print(f"\n🔍 验证更新...")
    study = optuna.load_study(
        study_name="enhancement_params_tuning",
        storage=f"sqlite:///{study_db}"
    )
    
    if len(study.trials) > 4:
        trial_4_updated = study.trials[4]
        print(f"  Trial 4新状态:")
        print(f"    状态: {trial_4_updated.state}")
        print(f"    值: {trial_4_updated.value:.4f} (对应分数: {-trial_4_updated.value:.4f})")
        print(f"    参数: {trial_4_updated.params}")
    
    if study.best_trial:
        print(f"\n🏆 当前最佳trial:")
        print(f"  Trial ID: {study.best_trial.number}")
        print(f"  最佳值: {-study.best_trial.value:.4f}")
        print(f"  参数: {study.best_trial.params}")
    
    print("\n" + "="*70)
    print("✅ Trial 4修复完成！")
    print("="*70)
    print("\n💡 提示:")
    print("  - 结果已保存到数据库，不会影响当前正在运行的程序")
    print("  - 后续trials会基于这个结果继续优化")
    print("  - 可以使用 optuna-dashboard 查看更新后的结果")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='修复Trial 4的结果（第五次实验）')
    parser.add_argument('--result_json', type=str, default=None,
                       help='包含trial 4结果的JSON文件路径（格式: {"params": {...}, "eval_results": {...}}）')
    args = parser.parse_args()
    
    success = fix_trial_4(result_json_path=args.result_json)
    sys.exit(0 if success else 1)

