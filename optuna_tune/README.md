# EnhancedUltra 相似度增强参数调优

## 概述

本目录包含用于调优 EnhancedUltra 模型中相似度增强模块初始参数的 Optuna 调参脚本。

### 调优参数

- `similarity_threshold_init`: 相似度阈值初始值 (范围: 0.5-0.95, 步长: 0.05)
- `enhancement_strength_init`: 增强强度初始值 (范围: 0.01-0.15, 步长: 0.01)

### 评估指标

- **MRR** (Mean Reciprocal Rank)
- **Hits@10**

综合分数 = 0.6 × 平均MRR + 0.4 × 平均Hits@10

## 使用方法

### 1. 安装依赖

```bash
conda activate semma
pip install optuna optuna-dashboard plotly
```

### 2. 运行调参

```bash
cd /T20030104/ynj/semma
python optuna_tune/tune_enhancement_params.py --n_trials 10
```

参数说明：
- `--n_trials`: 试验次数（默认10）
- `--pretrain_config`: 预训练配置文件路径（默认: config/transductive/pretrain_semma.yaml）

### 3. 查看可视化

#### 方式A：实时查看HTML图表

运行过程中会自动生成HTML图表，保存在 `optuna_tune/trials/visualizations/` 目录：

- `optimization_history.html` - 优化历史曲线
- `param_importances.html` - 参数重要性
- `contour_plot.html` - 参数等高线图
- `parallel_coordinate.html` - 参数关系平行坐标图

用浏览器打开这些HTML文件即可查看。

#### 方式B：Optuna Dashboard（推荐）

在另一个终端运行：

```bash
optuna-dashboard sqlite:///optuna_tune/trials/study.db
```

然后在浏览器打开 `http://localhost:8080`，可以看到：
- 实时更新的试验进度
- 交互式参数图表
- 试验详情和参数对比
- 参数重要性分析

## 时间估算

**单次试验时间：**
- 预训练：~10小时（10个epoch）
- 快速评估：~30-40分钟（6个代表性数据集）
- **总计：~10.5小时/试验**

**总时间估算：**
- 10个trials：~105小时（约4.4天）
- 5个trials：~52.5小时（约2.2天）

## 评估策略

为了节省时间，我们使用**代表性数据集快速评估**，而不是完整的54个数据集：

### 代表性数据集（6个）

**转导数据集（3个）：**
- FB15k237
- WN18RR
- CoDExSmall

**归纳数据集（3个）：**
- FB15k237Inductive (v1)
- WN18RRInductive (v1)
- NELLInductive (v1)

这些数据集覆盖了不同类型和规模，能够快速评估参数效果。

### 完整评估

如果需要，可以手动运行完整54个数据集的评估：

```bash
# 使用最佳参数更新flags.yaml后，运行完整评估
python script/run_many.py ...
```

## 结果文件

调参结果保存在 `optuna_tune/trials/` 目录：

- `study.db` - Optuna数据库（包含所有试验数据）
- `final_results.json` - 最终最优参数
- `trial_*/result.json` - 每个试验的详细结果
- `visualizations/` - 可视化图表

## 注意事项

1. **确保虚拟环境正确**：使用 `semma` 环境
2. **GPU资源**：确保GPU可用，脚本使用 `--gpus [0]`
3. **磁盘空间**：确保有足够空间存储checkpoint和输出
4. **中断恢复**：如果中断，Optuna会自动从 `study.db` 恢复，使用 `load_if_exists=True`
5. **参数范围**：如果需要调整参数搜索范围，修改 `tune_enhancement_params.py` 中的 `suggest_float` 参数

## 最佳实践

1. **先运行少量trials**：先用 `--n_trials 3-5` 测试
2. **监控进度**：使用Optuna Dashboard实时监控
3. **分析结果**：查看可视化图表，了解参数影响
4. **验证最佳参数**：在完整54个数据集上验证最佳参数

## 故障排除

### 预训练失败
- 检查配置文件路径
- 检查GPU是否可用
- 查看日志文件

### 评估失败
- 检查checkpoint路径
- 检查数据集是否完整
- 检查配置文件

### 可视化无法生成
- 确保安装了 `plotly`
- 检查 `visualizations/` 目录权限


