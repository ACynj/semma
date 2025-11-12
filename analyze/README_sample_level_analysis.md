# 样本级别相似关系参考分析

## 概述

这个脚本用于分析每个测试样本的相似关系参考情况，统计：
1. **有多少可以参考的相似关系**：对于每个测试样本，找到相似度超过阈值的关系
2. **有多少是参考有效的**：这些相似关系是否在训练数据中出现，是否能帮助预测
3. **多少引入了噪音**：相似关系在训练数据中不存在或不能帮助预测

## 功能说明

### 核心指标

- **参考率 (Reference Rate)**: 有相似关系参考的样本比例
- **有效性率 (Effectiveness Rate)**: 有效参考的比例
- **噪音率 (Noise Rate)**: 只有噪音参考的样本比例

### 有效性判断标准

一个相似关系被认为是**有效的**，如果：
- 对于tail预测：`(h, similar_rel)` 在训练数据中出现，且能提供上下文信息
- 对于head预测：`(similar_rel, t)` 在训练数据中出现，且能提供上下文信息

一个相似关系被认为是**噪音**，如果：
- 在训练数据中不存在
- 或者存在但不能提供有用的上下文信息

## 使用方法

```bash
cd /T20030104/ynj/semma
conda run -n semma python analyze/analyze_sample_level_reference.py
```

## Checkpoint配置

脚本会自动查找checkpoint：
1. 首先使用 `ckpts/optuna_1.pth`（基础checkpoint）
2. 如果不存在，会在 `optuna_1_output/Ultra/{dataset_name}/` 中查找数据集特定的checkpoint
3. 如果都找不到，会使用随机初始化的模型（结果可能不准确）

## 输出文件

1. **sample_level_reference_analysis.csv**: 详细统计结果
2. **figures/28_sample_level_reference_analysis.png**: 可视化图表

## 注意事项

- 脚本使用 **EnhancedUltra** 模型（因为ARE就是EnhanceUltra）
- 默认分析每个数据集的500个样本（可在代码中调整 `num_samples` 参数）
- 相似度阈值从 `flags.yaml` 中的 `similarity_threshold_init` 读取（默认0.85）
- 脚本会自动从 `common_features_analysis.csv` 读取显著提升和下降的数据集

## 分析的数据集

脚本会自动从 `common_features_analysis.csv` 中读取显著提升和下降的数据集，并选择代表性的数据集进行分析。

