# 语义增强的 RDG (Semantic-Enhanced RDG)

## 概述

语义增强的 RDG 将结构关系依赖与 SEMMA 的语义相似性信息相结合，创建更鲁棒的关系依赖图。这种方法可以：

1. **过滤噪声依赖**：只保留既有结构依赖又有语义相关性的边
2. **增强信号强度**：通过语义相似性调整依赖边的权重
3. **提高泛化能力**：结合结构和语义信息，在不同知识图谱中更稳定

## 核心功能

### 1. 语义增强函数 (`enhance_rdg_with_semantics`)

将结构依赖边与语义相似性矩阵结合，支持三种模式：

- **filter 模式**：只保留语义相似度 >= 阈值的边
- **weight 模式**：根据语义相似度调整边的权重
- **both 模式**：先过滤，再调整权重

### 2. 语义增强的 RDG 构建 (`build_semantic_enhanced_rdg_edges`)

完整的语义增强 RDG 构建流程，自动从 `graph.relation_graph2` 获取语义信息。

## 配置选项

在 `flags.yaml` 中配置：

```yaml
# 启用 RDG
use_rdg: True

# 启用语义增强
rdg_use_semantic_enhancement: True

# 语义相似性阈值（归一化到 [0,1]）
rdg_semantic_similarity_threshold: 0.5

# 结构-语义权重组合参数
# 最终权重 = (1-alpha) * 结构权重 + alpha * 语义相似度
rdg_semantic_weight_alpha: 0.5

# 过滤模式：filter, weight, both
rdg_semantic_filter_mode: both
```

## 使用方法

### 基本使用

```python
from ultra.rdg import build_semantic_enhanced_rdg_edges, RDGConfig

# 创建配置
config = RDGConfig(
    enabled=True,
    use_semantic_enhancement=True,
    semantic_similarity_threshold=0.5,
    semantic_weight_alpha=0.5,
    semantic_filter_mode='both'
)

# 构建语义增强的 RDG
# 如果 graph.relation_graph2 存在，会自动获取语义信息
edge_index, edge_weights, tau, deps = build_semantic_enhanced_rdg_edges(
    graph, semantic_similarity_matrix=None, config=config
)
```

### 在 SEMMA 中使用

语义增强的 RDG 会自动集成到 `build_relation_graph_exp` 函数中：

1. 当 `use_rdg=True` 且 `rdg_use_semantic_enhancement=True` 时
2. 系统会自动从 `graph.relation_graph2` 获取语义相似性矩阵
3. 如果 `relation_graph2` 不存在，会回退到纯结构 RDG

## 工作原理

### 1. 结构依赖提取

首先从知识图谱中提取结构依赖：
- 对于边 `(h, r1, t)`，查找以 `t` 为头实体的边 `(t, r2, t2)`
- 这创建依赖关系：`r1 -> r2`
- 统计依赖路径频率作为权重

### 2. 语义增强

然后使用语义相似性增强依赖边：

**Filter 模式**：
```python
if normalized_semantic >= threshold:
    keep_edge()
```

**Weight 模式**：
```python
enhanced_weight = (1-alpha) * structural_weight + alpha * normalized_semantic
```

**Both 模式**：
```python
if normalized_semantic >= threshold:
    enhanced_weight = (1-alpha) * structural_weight + alpha * normalized_semantic
    keep_edge_with_enhanced_weight()
```

### 3. 优先级计算

基于增强后的依赖边计算关系优先级 τ(r)。

## 优势

1. **噪声过滤**：语义不相关但结构上偶然连接的依赖会被过滤
2. **信号增强**：语义相关且结构依赖的边权重更高
3. **跨 KG 泛化**：结合结构和语义信息，在不同知识图谱中更稳定
4. **向后兼容**：可以禁用语义增强，使用纯结构 RDG

## 测试

运行测试脚本验证功能：

```bash
python ultra/rdg/test_semantic_enhanced_rdg.py
```

测试包括：
- 语义增强函数的基本功能
- 不同过滤模式
- 与 SEMMA 语义信息的集成
- 完整流程测试（CPU 模式）

## 注意事项

1. **语义信息可用性**：语义增强需要 `graph.relation_graph2` 存在，否则会回退到纯结构 RDG
2. **阈值设置**：`semantic_similarity_threshold` 是归一化到 [0,1] 的值（原始余弦相似度 [-1,1] 会被归一化）
3. **性能影响**：语义增强会增加少量计算开销，但通常可以忽略

## 实验建议

为了验证语义增强 RDG 的效果，建议进行以下对比实验：

1. **基线**：SEMMA（不使用 RDG）
2. **结构 RDG**：SEMMA + 结构 RDG（`use_rdg=True`, `rdg_use_semantic_enhancement=False`）
3. **语义增强 RDG**：SEMMA + 语义增强 RDG（`use_rdg=True`, `rdg_use_semantic_enhancement=True`）

评估指标：
- Link prediction 性能（MRR, Hits@K）
- 跨数据集泛化能力
- 未见关系推理能力

