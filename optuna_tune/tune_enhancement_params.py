#!/usr/bin/env python
"""
EnhancedUltra相似度增强参数调优脚本
专门调优 similarity_threshold_init 和 enhancement_strength_init 两个参数
"""

import optuna
import optuna.visualization as vis
import os
import sys
import subprocess
import json
import yaml
import shutil
from pathlib import Path
import torch
from datetime import datetime
import re
import glob

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultra import util

class EnhancementParamsTuner:
    def __init__(self, 
                 pretrain_config="config/transductive/pretrain_semma.yaml",
                 flags_path="flags.yaml",
                 output_dir="./optuna_tune/trials"):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pretrain_config = os.path.join(self.project_root, pretrain_config)
        self.flags_path = os.path.join(self.project_root, flags_path)
        self.output_dir = os.path.join(self.project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 可视化输出目录
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 备份原始flags.yaml
        self.flags_backup = os.path.join(self.output_dir, "flags_backup.yaml")
        shutil.copy(self.flags_path, self.flags_backup)
        
        # 代表性数据集列表（用于快速评估）
        # 包含转导和归纳数据集，覆盖不同类型
        self.representative_datasets = [
            # 转导数据集（3个）
            ("FB15k237", None, "transductive"),
            ("WN18RR", None, "transductive"),
            ("CoDExSmall", None, "transductive"),
            # 归纳数据集（3个）
            ("FB15k237Inductive", "v1", "inductive"),
            ("WN18RRInductive", "v1", "inductive"),
            ("NELLInductive", "v1", "inductive"),
        ]
        
        self.current_trial = 0
        self.total_trials = 0
        self.start_time = None
        
    def objective(self, trial):
        """Optuna目标函数"""
        self.current_trial = trial.number + 1
        
        # 采样两个参数
        similarity_threshold = trial.suggest_float(
            'similarity_threshold_init', 
            0.5,  # 最小值
            0.95,  # 最大值
            step=0.05  # 步长
        )
        
        enhancement_strength = trial.suggest_float(
            'enhancement_strength_init', 
            0.01,  # 最小值
            0.15,  # 最大值（保持较小，因为最终会映射到0-0.2）
            step=0.01  # 步长
        )
        
        params = {
            'similarity_threshold_init': similarity_threshold,
            'enhancement_strength_init': enhancement_strength
        }
        
        # 打印进度
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
        eta = (elapsed / max(1, self.current_trial - 1)) * (self.total_trials - self.current_trial) if self.current_trial > 1 else 0
        
        print(f"\n{'='*70}")
        print(f"Trial {trial.number+1}/{self.total_trials}")
        print(f"已用时间: {elapsed:.1f}小时 | 预计剩余: {eta:.1f}小时")
        print(f"参数: similarity_threshold_init={similarity_threshold:.3f}, enhancement_strength_init={enhancement_strength:.3f}")
        print(f"{'='*70}")
        
        try:
            # 1. 更新flags.yaml
            self._update_flags(params)
            
            # 2. 运行预训练（10小时）
            print(f"\n[步骤1/3] 开始预训练...")
            checkpoint_path = self._run_pretrain()
            if checkpoint_path is None:
                print(f"[Trial {trial.number}] ✗ 预训练失败")
                return float('inf')
            
            print(f"✓ 预训练完成，checkpoint: {checkpoint_path}")
            
            # 3. 快速评估（代表性数据集，约30-40分钟）
            print(f"\n[步骤2/3] 在代表性数据集上快速评估...")
            eval_results = self._fast_evaluate(checkpoint_path)
            
            if not eval_results:
                print(f"[Trial {trial.number}] ✗ 评估失败: 没有获得任何评估结果")
                print(f"  检查checkpoint路径: {checkpoint_path}")
                print(f"  检查评估日志以获取更多信息")
                return float('inf')
            
            # 检查是否有有效的指标
            valid_results = {k: v for k, v in eval_results.items() if isinstance(v, dict) and 'mrr' in v and 'hits@10' in v}
            if not valid_results:
                print(f"[Trial {trial.number}] ✗ 评估失败: 没有有效的指标结果")
                print(f"  评估结果: {eval_results}")
                return float('inf')
            
            # 4. 计算综合分数
            score = self._calculate_score(eval_results)
            
            print(f"\n[步骤3/3] 评估完成")
            print(f"✓ 综合分数: {score:.4f}")
            print(f"  平均MRR: {eval_results.get('avg_mrr', 0):.4f}")
            print(f"  平均Hits@10: {eval_results.get('avg_hits10', 0):.4f}")
            
            if trial.number > 0 and hasattr(trial.study, 'best_value'):
                best_so_far = -trial.study.best_value
                print(f"  当前最佳: {best_so_far:.4f} (Trial {trial.study.best_trial.number})")
            
            # 5. 保存结果
            self._save_trial(trial.number, params, score, eval_results)
            
            # 6. 每5个trial生成一次可视化
            if (trial.number + 1) % 5 == 0:
                self._generate_realtime_plots(trial.study)
            
            # 返回负分数（Optuna最小化，我们最大化score）
            return -score
            
        except Exception as e:
            print(f"[Trial {trial.number}] ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
        finally:
            # 清理临时文件
            self._cleanup()
    
    def _update_flags(self, params):
        """更新flags.yaml中的参数"""
        with open(self.flags_path, 'r') as f:
            lines = f.readlines()
        
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('similarity_threshold_init:'):
                lines[i] = f"similarity_threshold_init: {params['similarity_threshold_init']}\n"
                updated = True
            elif line.startswith('enhancement_strength_init:'):
                lines[i] = f"enhancement_strength_init: {params['enhancement_strength_init']}\n"
                updated = True
        
        if updated:
            with open(self.flags_path, 'w') as f:
                f.writelines(lines)
    
    def _run_pretrain(self):
        """运行预训练"""
        cmd = [
            "python", "script/pretrain.py",
            "-c", self.pretrain_config,
            "--gpus", "[0]",
            "--seed", "42"
        ]
        
        try:
            print(f"  执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=39600  # 11小时超时（10小时预训练+缓冲）
            )
            
            if result.returncode == 0:
                checkpoint = self._parse_checkpoint_path(result.stdout, result.stderr)
                return checkpoint
            else:
                print(f"  错误输出: {result.stderr[-500:]}")
                return None
                
        except subprocess.TimeoutExpired:
            print("  ✗ 预训练超时")
            return None
        except Exception as e:
            print(f"  ✗ 预训练异常: {e}")
            return None
    
    def _parse_checkpoint_path(self, stdout, stderr):
        """从输出中解析checkpoint路径"""
        # 从配置读取output_dir
        config = util.load_config(self.pretrain_config, context={'gpus': '[0]'})
        
        output_dir = getattr(config, 'output_dir', './output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.project_root, output_dir)
        
        # 查找最新的checkpoint
        patterns = [
            os.path.join(output_dir, "**", "*.pt"),
            os.path.join(output_dir, "**", "*.pth"),
        ]
        
        latest_checkpoint = None
        latest_time = 0
        
        for pattern in patterns:
            for file in glob.glob(pattern, recursive=True):
                if os.path.getmtime(file) > latest_time:
                    latest_time = os.path.getmtime(file)
                    latest_checkpoint = file
        
        if latest_checkpoint:
            return latest_checkpoint
        
        # 从输出中解析
        all_output = stdout + stderr
        for line in all_output.split('\n'):
            if 'checkpoint' in line.lower() or 'saved' in line.lower():
                paths = re.findall(r'[\w/.-]+\.(?:pt|pth)', line)
                if paths:
                    return paths[-1]
        
        return None
    
    def _fast_evaluate(self, checkpoint_path):
        """在代表性数据集上快速评估"""
        results = {}
        
        for dataset_name, version, dataset_type in self.representative_datasets:
            print(f"\n  评估数据集: {dataset_name}" + (f" (v{version})" if version else ""))
            
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
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30分钟超时（每个数据集）
                )
                
                if result.returncode == 0:
                    metrics = self._parse_metrics(result.stdout)
                    if metrics:
                        results[dataset_name] = metrics
                        print(f"    ✓ MRR: {metrics.get('mrr', 0):.4f}, Hits@10: {metrics.get('hits@10', 0):.4f}")
                    else:
                        print(f"    ⚠ 无法解析指标")
                else:
                    print(f"    ✗ 评估失败")
                    
            except subprocess.TimeoutExpired:
                print(f"    ✗ 评估超时")
            except Exception as e:
                print(f"    ✗ 评估异常: {e}")
        
        return results
    
    def _parse_metrics(self, output):
        """从输出中解析MRR和Hits@10"""
        metrics = {}
        
        # 查找MRR（支持 mrr, mrr-tail 等格式）
        mrr_patterns = [
            r'mrr(?:-tail)?[:\s]+(\d+\.\d+)',  # 匹配 mrr: 或 mrr-tail:
            r'mrr[:\s]+(\d+\.\d+)',  # 备用模式
            r'mrr[:\s]+(\d+\.\d+e[+-]?\d+)',  # 科学计数法
        ]
        
        for pattern in mrr_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    # 取最后一个匹配（通常是test集的结果）
                    metrics['mrr'] = float(matches[-1])
                    break
                except:
                    pass
        
        # 查找Hits@10（支持 hits@10, hits@10-tail 等格式）
        hits10_patterns = [
            r'hits@10(?:-tail)?[:\s]+(\d+\.\d+)',  # 匹配 hits@10: 或 hits@10-tail:
            r'hits@10[:\s]+(\d+\.\d+)',  # 备用模式
            r'hits@10[:\s]+(\d+\.\d+e[+-]?\d+)',  # 科学计数法
        ]
        
        for pattern in hits10_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    # 取最后一个匹配（通常是test集的结果）
                    metrics['hits@10'] = float(matches[-1])
                    break
                except:
                    pass
        
        return metrics
    
    def _calculate_score(self, eval_results):
        """计算综合分数"""
        if not eval_results:
            return 0.0
        
        total_mrr = 0.0
        total_hits10 = 0.0
        count = 0
        
        for dataset_name, metrics in eval_results.items():
            if 'mrr' in metrics and 'hits@10' in metrics:
                total_mrr += metrics['mrr']
                total_hits10 += metrics['hits@10']
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_mrr = total_mrr / count
        avg_hits10 = total_hits10 / count
        
        # 加权平均：MRR权重0.6，Hits@10权重0.4
        score = 0.6 * avg_mrr + 0.4 * avg_hits10
        
        # 保存到eval_results中
        eval_results['avg_mrr'] = avg_mrr
        eval_results['avg_hits10'] = avg_hits10
        eval_results['score'] = score
        
        return score
    
    def _save_trial(self, trial_num, params, score, eval_results):
        """保存试验结果"""
        trial_dir = os.path.join(self.output_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)
        
        result = {
            'trial_number': trial_num,
            'score': score,
            'params': params,
            'eval_results': eval_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(trial_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
    
    def _generate_realtime_plots(self, study):
        """实时生成可视化图表"""
        try:
            print(f"\n[可视化] 生成实时图表...")
            
            # 1. 优化历史
            fig = vis.plot_optimization_history(study)
            fig.write_html(os.path.join(self.viz_dir, "optimization_history.html"))
            
            # 2. 参数重要性
            if len(study.trials) > 3:
                try:
                    fig = vis.plot_param_importances(study)
                    fig.write_html(os.path.join(self.viz_dir, "param_importances.html"))
                except:
                    pass
            
            # 3. 参数关系（等高线图）
            if len(study.trials) > 5:
                try:
                    fig = vis.plot_contour(
                        study, 
                        params=['similarity_threshold_init', 'enhancement_strength_init']
                    )
                    fig.write_html(os.path.join(self.viz_dir, "contour_plot.html"))
                except:
                    pass
            
            # 4. 平行坐标图
            if len(study.trials) > 5:
                try:
                    fig = vis.plot_parallel_coordinate(study)
                    fig.write_html(os.path.join(self.viz_dir, "parallel_coordinate.html"))
                except:
                    pass
            
            print(f"  ✓ 图表已保存到: {self.viz_dir}")
            print(f"    打开 {os.path.join(self.viz_dir, 'optimization_history.html')} 查看进度")
            
        except Exception as e:
            print(f"  ⚠ 生成图表时出错: {e}")
    
    def _cleanup(self):
        """清理临时文件"""
        # 恢复原始flags.yaml
        if os.path.exists(self.flags_backup):
            shutil.copy(self.flags_backup, self.flags_path)
    
    def run_study(self, n_trials=10):
        """运行调参研究"""
        print("\n" + "="*70)
        print("🚀 EnhancedUltra 相似度增强参数调优")
        print("="*70)
        print(f"调优参数:")
        print(f"  - similarity_threshold_init: [0.5, 0.95], step=0.05")
        print(f"  - enhancement_strength_init: [0.01, 0.15], step=0.01")
        print(f"\n评估策略:")
        print(f"  - 预训练: 完整10个epoch (~10小时)")
        print(f"  - 快速评估: {len(self.representative_datasets)}个代表性数据集 (~30-40分钟)")
        print(f"  - 总时间估算: {n_trials} trials × (~10.5小时) = {n_trials * 10.5:.1f}小时")
        print("="*70)
        
        self.total_trials = n_trials
        self.start_time = datetime.now()
        
        # 创建study
        study = optuna.create_study(
            direction='minimize',  # 我们返回负分数，所以最小化
            study_name='enhancement_params_tuning',
            storage=f"sqlite:///{self.output_dir}/study.db",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=2,  # 至少运行2个trial才开始剪枝
                n_warmup_steps=1,
                interval_steps=1
            )
        )
        
        # 运行优化
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # 生成最终可视化
        print(f"\n[可视化] 生成最终图表...")
        self._generate_final_plots(study)
        
        # 保存最终结果
        final_result = {
            'best_params': study.best_params,
            'best_value': -study.best_value,  # 转回正值
            'n_trials': len(study.trials),
            'total_time_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
        
        with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
            json.dump(final_result, f, indent=2)
        
        # 恢复原始flags.yaml
        self._cleanup()
        
        print("\n" + "="*70)
        print("✅ 调参完成！")
        print("="*70)
        print(f"最佳参数:")
        print(f"  similarity_threshold_init: {study.best_params['similarity_threshold_init']:.3f}")
        print(f"  enhancement_strength_init: {study.best_params['enhancement_strength_init']:.3f}")
        print(f"最佳分数: {-study.best_value:.4f}")
        print(f"\n结果已保存到: {self.output_dir}")
        print(f"可视化图表: {self.viz_dir}")
        print("\n💡 提示: 使用以下命令启动Optuna Dashboard查看详细结果:")
        print(f"   optuna-dashboard sqlite:///{self.output_dir}/study.db")
        print("="*70)
        
        return study.best_params
    
    def _generate_final_plots(self, study):
        """生成最终可视化图表"""
        try:
            plots = {
                'optimization_history': vis.plot_optimization_history(study),
                'param_importances': None,
                'parallel_coordinate': None,
                'contour': None,
            }
            
            if len(study.trials) > 3:
                try:
                    plots['param_importances'] = vis.plot_param_importances(study)
                except:
                    pass
            
            if len(study.trials) > 5:
                try:
                    plots['parallel_coordinate'] = vis.plot_parallel_coordinate(study)
                    plots['contour'] = vis.plot_contour(
                        study, 
                        params=['similarity_threshold_init', 'enhancement_strength_init']
                    )
                except:
                    pass
            
            # 保存所有图表
            for name, fig in plots.items():
                if fig is not None:
                    fig.write_html(os.path.join(self.viz_dir, f"final_{name}.html"))
            
            print(f"  ✓ 所有图表已保存到: {self.viz_dir}")
            
        except Exception as e:
            print(f"  ⚠ 生成图表时出错: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EnhancedUltra相似度增强参数调优')
    parser.add_argument('--n_trials', type=int, default=10, help='试验次数')
    parser.add_argument('--pretrain_config', type=str, 
                       default='config/transductive/pretrain_semma.yaml',
                       help='预训练配置文件路径')
    args = parser.parse_args()
    
    tuner = EnhancementParamsTuner(pretrain_config=args.pretrain_config)
    best_params = tuner.run_study(n_trials=args.n_trials)
    
    print(f"\n🎯 最终最优参数:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

