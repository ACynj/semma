# 使用指南

## 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
conda activate semma

# 安装依赖（如果还没安装）
pip install optuna optuna-dashboard plotly
```

### 2. 快速测试（推荐先运行）

在运行完整调参之前，建议先运行快速测试验证环境：

```bash
cd /T20030104/ynj/semma
python optuna_tune/quick_test.py
```

**快速测试特点：**
- 只运行1个trial
- 预训练只运行2个epoch（约30分钟）
- 只评估2个代表性数据集
- 总时间：~30-40分钟

### 3. 完整调参

如果快速测试成功，运行完整调参：

```bash
# 方式1：直接运行Python脚本
python optuna_tune/tune_enhancement_params.py --n_trials 10

# 方式2：使用shell脚本
bash optuna_tune/run_tuning.sh 10

# 方式3：后台运行（推荐）
nohup python optuna_tune/tune_enhancement_params.py --n_trials 10 > optuna_tune/tuning.log 2>&1 &
```

## 时间估算

### 单次试验时间分解

| 步骤 | 时间 | 说明 |
|------|------|------|
| 预训练 | ~10小时 | 10个epoch |
| 快速评估 | ~30-40分钟 | 6个代表性数据集 |
| **总计** | **~10.5小时** | 每次试验 |

### 总时间估算

| Trials数 | 总时间 | 预计完成时间 |
|----------|--------|--------------|
| 5 | ~52.5小时 | 约2.2天 |
| 10 | ~105小时 | 约4.4天 |
| 15 | ~157.5小时 | 约6.6天 |

## 监控进度

### 方式1：查看日志文件

```bash
# 如果使用nohup后台运行
tail -f optuna_tune/tuning.log

# 或者查看实时输出
tail -f optuna_tune/trials/trial_*/result.json
```

### 方式2：Optuna Dashboard（推荐）

```bash
# 在另一个终端运行
conda activate semma
optuna-dashboard sqlite:///optuna_tune/trials/study.db
```

然后在浏览器打开 `http://localhost:8080`

### 方式3：查看HTML可视化

```bash
# 打开生成的可视化图表
firefox optuna_tune/trials/visualizations/optimization_history.html
```

## 参数搜索空间

### similarity_threshold_init
- **范围**: 0.5 - 0.95
- **步长**: 0.05
- **说明**: 相似度阈值初始值，控制哪些相似关系会被参考

### enhancement_strength_init
- **范围**: 0.01 - 0.15
- **步长**: 0.01
- **说明**: 增强强度初始值，控制增强的强度（最终会映射到0-0.2范围）

## 评估策略

### 快速评估数据集（6个）

为了节省时间，我们使用代表性数据集进行快速评估：

**转导数据集（3个）：**
- FB15k237
- WN18RR
- CoDExSmall

**归纳数据集（3个）：**
- FB15k237Inductive (v1)
- WN18RRInductive (v1)
- NELLInductive (v1)

### 完整评估

如果需要，可以在找到最佳参数后，手动运行完整54个数据集的评估。

## 结果文件

所有结果保存在 `optuna_tune/trials/` 目录：

```
optuna_tune/trials/
├── study.db                    # Optuna数据库
├── final_results.json          # 最终最优参数
├── flags_backup.yaml           # flags.yaml备份
├── trial_0/
│   └── result.json            # 第0个trial的详细结果
├── trial_1/
│   └── result.json
└── visualizations/             # 可视化图表
    ├── optimization_history.html
    ├── param_importances.html
    ├── contour_plot.html
    └── parallel_coordinate.html
```

## 查看结果

### 查看最终最优参数

```bash
cat optuna_tune/trials/final_results.json | python -m json.tool
```

### 查看某个trial的详细结果

```bash
cat optuna_tune/trials/trial_0/result.json | python -m json.tool
```

### 应用最佳参数

找到最佳参数后，手动更新 `flags.yaml`：

```yaml
similarity_threshold_init: 0.85  # 最佳值
enhancement_strength_init: 0.08  # 最佳值
```

## 故障排除

### 问题1：预训练失败

**可能原因：**
- GPU不可用
- 配置文件路径错误
- 内存不足

**解决方法：**
```bash
# 检查GPU
nvidia-smi

# 检查配置文件
cat config/transductive/pretrain_semma.yaml
```

### 问题2：评估失败

**可能原因：**
- Checkpoint路径错误
- 数据集不存在
- 配置文件错误

**解决方法：**
```bash
# 检查checkpoint是否存在
ls -lh output/**/*.pt

# 检查数据集
ls -lh kg-datasets/
```

### 问题3：可视化无法生成

**可能原因：**
- plotly未安装
- 权限问题

**解决方法：**
```bash
pip install plotly
chmod -R 755 optuna_tune/trials/visualizations/
```

### 问题4：中断后如何恢复

Optuna会自动从数据库恢复，无需特殊操作。直接重新运行命令即可：

```bash
python optuna_tune/tune_enhancement_params.py --n_trials 10
```

## 最佳实践

1. **先运行快速测试**：确认环境配置正确
2. **使用nohup后台运行**：避免终端断开导致中断
3. **定期监控**：使用Optuna Dashboard查看进度
4. **保存结果**：定期备份 `optuna_tune/trials/` 目录
5. **验证最佳参数**：在完整54个数据集上验证最佳参数的效果

## 联系与支持

如有问题，请检查：
1. 日志文件：`optuna_tune/tuning.log`
2. Trial结果：`optuna_tune/trials/trial_*/result.json`
3. Optuna Dashboard：查看详细错误信息


