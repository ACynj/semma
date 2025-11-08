# Trial 2补充评估指南

## 当前状态

Trial 2的评估已完成4/6个数据集：
- ✅ CoDExSmall
- ✅ FB15k237Inductive  
- ✅ WN18RRInductive
- ✅ NELLInductive
- ❌ FB15k237（缺失，GPU内存不足）
- ❌ WN18RR（缺失，GPU内存不足）

当前综合分数：**0.7023**（基于4个数据集）

## 补充评估方法

### 方法1：使用自动等待脚本（推荐）

```bash
cd /T20030104/ynj/semma
bash optuna_tune/run_trial2_completion.sh --wait
```

这个脚本会：
1. 自动检测GPU是否空闲
2. 如果GPU被占用，会每30秒检查一次
3. GPU空闲后自动运行评估
4. 评估完成后自动更新数据库和结果文件

### 方法2：手动运行

当GPU空闲时（Trial 3完成或暂停后），运行：

```bash
conda activate semma
cd /T20030104/ynj/semma
python optuna_tune/complete_trial2_evaluation.py
```

### 方法3：后台运行（不推荐，但可以）

```bash
cd /T20030104/ynj/semma
nohup bash optuna_tune/run_trial2_completion.sh --wait > optuna_tune/trial2_completion.log 2>&1 &
```

## 检查GPU状态

```bash
nvidia-smi
```

或者检查GPU内存：

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

需要至少10GB空闲内存才能运行评估。

## 评估完成后

评估完成后，脚本会自动：
1. ✅ 更新Optuna数据库中的Trial 2结果
2. ✅ 保存完整的6个数据集评估结果到 `optuna_tune/trials/trial_2/result.json`
3. ✅ 重新计算综合分数（基于6个数据集）

## 验证结果

评估完成后，可以运行：

```bash
conda activate semma
python optuna_tune/check_status.py
```

或者查看结果文件：

```bash
cat optuna_tune/trials/trial_2/result.json | python -m json.tool
```

## 注意事项

1. **不要中断Trial 3**：当前Trial 3正在运行，建议等待其完成后再运行补充评估
2. **GPU内存**：评估需要约10GB GPU内存，确保有足够空间
3. **时间**：每个数据集评估需要约5-10分钟，2个数据集总共约10-20分钟
4. **不影响其他trials**：补充评估不会影响正在运行的Trial 3或其他进程

## 当前Trial 2状态

- **参数**: similarity_threshold_init=0.55, enhancement_strength_init=0.03
- **当前分数**: 0.7023（基于4个数据集）
- **Checkpoint**: epoch 9（epoch 10未完成）
- **状态**: 已更新到数据库，但缺少2个数据集

补充评估完成后，分数可能会有所变化（基于完整的6个数据集）。

