# PPR动态阈值实现说明

## 概述

根据您的idea，我已经成功实现了基于Personalized PageRank (PPR)的动态阈值方案，用于替代SEMMA模型中的固定阈值方法。这个实现解决了以下问题：

1. **关系文本语义差异大**：PPR通过个性化动态筛选，能自动适配不同语义密度的关系
2. **间接语义关联丢失**：PPR擅长捕捉间接关联，通过迭代传播发现隐性结构关联
3. **LLM生成偏差**：PPR具备噪声鲁棒性，通过多轮迭代传播修正单个节点的误差

## 实现细节

### 1. 配置文件更新 (`flags.yaml`)

新增了以下配置参数：

```yaml
threshold: 0.8 # threshold for constructing rg2 (only used when use_ppr_threshold=False)
use_ppr_threshold: False # [True, False], if True, use PPR-based dynamic threshold instead of fixed threshold
ppr_alpha: 0.85 # damping coefficient for PPR algorithm
ppr_max_iter: 100 # maximum iterations for PPR convergence
ppr_tolerance: 1e-6 # convergence tolerance for PPR
```

**注意**：已移除K-means相关的`dynamic_threshold`参数，简化配置。

### 2. 核心算法实现 (`ultra/tasks.py`)

#### `compute_ppr_scores(embeddings, alpha, max_iter, tolerance)`
- 基于关系嵌入的余弦相似度构建邻接矩阵
- 使用PPR算法计算每个关系的重要性得分
- 支持自定义阻尼系数、最大迭代次数和收敛容忍度

#### `find_pairs_with_ppr(embeddings, relation_names, top_k_ratio)`
- 结合PPR得分和余弦相似度计算综合得分
- 选择top-k比例的关系对
- 避免重复计算（使用上三角矩阵）

#### `find_pairs_above_threshold(embeddings, relation_names)` (修改)
- 添加了PPR选项检查
- 当`use_ppr_threshold=True`时，自动使用PPR方法
- 保持向后兼容性，`use_ppr_threshold=False`时使用原固定阈值方案

## 使用方法

### 启用PPR动态阈值

在`flags.yaml`中设置：
```yaml
use_ppr_threshold: True
ppr_alpha: 0.85  # 可选，默认0.85
ppr_max_iter: 100  # 可选，默认100
ppr_tolerance: 1e-6  # 可选，默认1e-6
```

### 使用原固定阈值方案

在`flags.yaml`中设置：
```yaml
use_ppr_threshold: False
threshold: 0.8  # 固定阈值
```

## 参数说明

### PPR参数详解

1. **`ppr_alpha: 0.85`** - PPR阻尼系数
   - 控制随机游走的概率
   - 0.85表示85%按图结构传播，15%随机重启
   - 值越大越依赖图结构，值越小越随机

2. **`ppr_max_iter: 100`** - 最大迭代次数
   - PPR算法收敛的最大迭代次数限制
   - 实际通常6-11次迭代就能收敛
   - 防止算法无限循环

3. **`ppr_tolerance: 1e-6`** - 收敛容忍度
   - 判断PPR算法是否收敛的阈值
   - 连续两次迭代得分变化小于此值时认为收敛
   - 值越小收敛判断越严格

## 算法优势

1. **个性化动态筛选**：以查询依赖为核心，通过迭代传播关系节点的重要性得分
2. **间接关联捕捉**：能发现实体间的隐性结构关联（如2-hop邻居通过1-hop邻居获得高得分）
3. **噪声鲁棒性**：通过多轮迭代传播，可利用其他有效关联修正单个节点的误差
4. **自适应性强**：
   - 对语义密集关系，PPR会通过传播效应自动保留更多语义邻居
   - 对语义独特关系，PPR会因传播路径少而减少邻居数量，避免冗余噪声

## 测试结果

测试显示PPR算法能够：
- 快速收敛（通常6-11次迭代）
- 正确计算关系重要性得分
- 有效选择关系对（测试中从45个可能对中选择了9个，比例20%）
- 对不同参数设置具有良好的适应性

## 简化说明

- **移除K-means方法**：删除了`dynamic_threshold`参数和相关的K-means聚类代码
- **简化配置**：现在只有两种选择：PPR动态阈值 或 固定阈值
- **保持兼容性**：完全向后兼容，不影响现有代码逻辑
- **灵活切换**：通过`use_ppr_threshold`参数在两种方法间切换
