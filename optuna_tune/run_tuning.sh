#!/bin/bash
# EnhancedUltra相似度增强参数调优启动脚本

# 激活虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh  # 根据你的conda路径调整
conda activate semma

# 进入项目目录
cd /T20030104/ynj/semma

# 运行调参脚本
# 默认10个trials，可以通过命令行参数修改
python optuna_tune/tune_enhancement_params.py --n_trials ${1:-10}


