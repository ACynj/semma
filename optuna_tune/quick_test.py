#!/usr/bin/env python
"""
快速测试脚本 - 验证调参环境是否正确配置
只运行1个trial，使用较短的预训练时间（2个epoch）来快速验证
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tune_enhancement_params import EnhancementParamsTuner
from ultra import util
import yaml
import shutil
import subprocess

class QuickTestTuner(EnhancementParamsTuner):
    """快速测试版本 - 使用更少的epoch和数据集"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 只使用2个代表性数据集进行快速测试
        self.representative_datasets = [
            ("FB15k237", None, "transductive"),
            ("WN18RR", None, "transductive"),
        ]
    
    def _run_pretrain(self):
        """运行快速预训练（2个epoch）"""
        # 创建临时配置文件，使用更少的epoch
        temp_config_path = os.path.join(self.output_dir, "temp_pretrain_test.yaml")
        
        # 直接读取原始文件并修改特定行
        with open(self.pretrain_config, 'r') as f:
            lines = f.readlines()
        
        # 修改num_epoch和batch_per_epoch
        modified_lines = []
        for line in lines:
            if line.strip().startswith('num_epoch:'):
                modified_lines.append('  num_epoch: 2  # Quick test: reduced from 10\n')
            elif line.strip().startswith('batch_per_epoch:'):
                modified_lines.append('  batch_per_epoch: 5000  # Quick test: reduced from 20000\n')
            else:
                modified_lines.append(line)
        
        with open(temp_config_path, 'w') as f:
            f.writelines(modified_lines)
        
        cmd = [
            "python", "script/pretrain.py",
            "-c", temp_config_path,
            "--gpus", "[0]",
            "--seed", "42"
        ]
        
        try:
            print(f"  执行命令: {' '.join(cmd)}")
            print(f"  ⚠ 快速测试模式: 只运行2个epoch（正常为10个）")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=7200  # 2小时超时
            )
            
            if result.returncode == 0:
                checkpoint = self._parse_checkpoint_path(result.stdout, result.stderr)
                # 清理临时配置文件
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                return checkpoint
            else:
                print(f"  错误输出: {result.stderr[-500:]}")
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                return None
                
        except Exception as e:
            print(f"  ✗ 预训练异常: {e}")
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 快速测试 - EnhancedUltra相似度增强参数调优环境验证")
    print("="*70)
    print("⚠️  注意: 这是快速测试模式")
    print("   - 只运行1个trial")
    print("   - 预训练只运行2个epoch（正常为10个）")
    print("   - 只评估2个代表性数据集")
    print("   - 预计时间: ~30-40分钟")
    print("="*70)
    
    tuner = QuickTestTuner(pretrain_config='config/transductive/pretrain_semma.yaml')
    best_params = tuner.run_study(n_trials=1)
    
    print(f"\n✅ 快速测试完成！")
    print(f"如果测试成功，可以运行完整调参:")
    print(f"  python optuna_tune/tune_enhancement_params.py --n_trials 10")

