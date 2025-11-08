# RDG (Relation Dependency Graph) Module

## 概述

RDG模块实现了基于GRAPHORACLE论文的关系依赖图构建功能。该模块从知识图谱中提取关系之间的依赖关系，并建立层次化的关系优先级结构。

## 核心功能

### 1. 关系依赖提取 (`extract_relation_dependencies`)

从知识图谱中提取关系依赖路径。

**工作原理：**
- 对于每条边 `(h, r1, t)`，查找以 `t` 为头实体的边 `(t, r2, t2)`
- 这创建了依赖关系：`r1 -> r2`（r1的尾实体是r2的头实体）
- 统计每条依赖路径的出现频率作为权重

**输入：**
- `graph`: 包含 `edge_index` [2, num_edges] 和 `edge_type` [num_edges] 的图对象
- `config`: RDG配置对象

**输出：**
- `List[Tuple[int, int, float]]`: `(r_i, r_j, weight)` 依赖边列表

**示例：**
```python
# 知识图谱：
# (Alice, bornIn, Beijing) -> (Beijing, locatedIn, China)
# 提取依赖：bornIn -> locatedIn (权重: 0.1667)
```

### 2. 关系优先级计算 (`compute_relation_precedence`)

计算每个关系的优先级值 τ(r)。

**工作原理：**
- 基于入度（indegree）方法：
  - 计算每个关系被其他关系依赖的总权重（入度）
  - 入度越高，关系越"基础"（τ值越低）
  - 入度越低，关系越"复合"（τ值越高）
- 归一化到 [0, 1] 范围

**公式：**
```
in_degree[r] = Σ weight for all edges (r_i -> r)
τ[r] = 1.0 - (in_degree[r] / max_in_degree)
```

**输出：**
- `Dict[int, float]`: 关系ID到优先级值的映射

**示例：**
```
locatedIn: τ = 0.0000 (最基础，被多个关系依赖)
bornIn: τ = 0.5000 (中等)
livesIn: τ = 1.0000 (最复合，不依赖其他关系)
```

### 3. 前驱关系获取 (`get_preceding_relations`)

获取关系 r_v 的前驱关系集合 N^past(r_v)。

**条件：**
1. 存在依赖边 `r_i -> r_v`
2. `τ[r_i] < τ[r_v]`（r_i 的优先级更低，更基础）

**输出：**
- `List[int]`: 前驱关系ID列表

### 4. 完整RDG构建 (`build_rdg_edges`)

构建完整的RDG边和元数据。

**输出：**
- `edge_index`: [2, num_rdg_edges] 依赖边索引
- `edge_weights`: [num_rdg_edges] 边权重
- `tau`: 关系优先级字典
- `dependency_edges`: 依赖边列表

## 数据流和Shape说明

### 输入阶段

```
知识图谱 (graph):
├── edge_index: [2, num_edges]
│   ├── [0, :]: 头实体索引
│   └── [1, :]: 尾实体索引
├── edge_type: [num_edges]
│   └── 每条边的关系类型
└── num_relations: 关系总数
```

### 处理阶段

**步骤1：依赖提取**
```
遍历所有边 (h, r1, t)
  └─> 查找以 t 为头的边 (t, r2, t2)
      └─> 记录依赖: r1 -> r2 (权重 += 1)
      
输出: dependency_edges = [(r_i, r_j, weight), ...]
```

**步骤2：优先级计算**
```
对于每个关系 r:
  in_degree[r] = Σ weight for (r_i -> r)
  τ[r] = 1.0 - (in_degree[r] / max_in_degree)
  
输出: tau = {r: τ_value, ...}
```

**步骤3：构建张量**
```
edge_index = [[r_i, r_i, ...],    # [2, num_deps]
              [r_j, r_j, ...]]
              
edge_weights = [w1, w2, ...]      # [num_deps]
```

### 输出阶段

```
RDG输出:
├── edge_index: [2, num_rdg_edges]
│   └── 依赖边的源-目标关系对
├── edge_weights: [num_rdg_edges]
│   └── 每条依赖边的权重
├── tau: Dict[int, float]
│   └── 关系优先级值
└── dependency_edges: List[Tuple]
    └── 原始依赖边列表
```

### 集成到关系图

```
原始关系图:
├── edge_index: [2, num_existing_edges]
├── edge_type: [num_existing_edges]  (类型: 0,1,2,3 = hh,tt,ht,th)
└── num_relations: 4

添加RDG后:
├── edge_index: [2, num_existing + num_rdg]
│   └── 拼接: [existing_edges, rdg_edges]
├── edge_type: [num_existing + num_rdg]
│   └── 拼接: [existing_types, rdg_types(全为4)]
└── num_relations: 5 (新增类型4 = RDG依赖边)
```

## 配置选项

在 `flags.yaml` 中配置：

```yaml
use_rdg: False                    # 是否启用RDG
rdg_min_weight: 0.001             # 最小依赖权重阈值
rdg_precedence_method: indegree    # 优先级计算方法
rdg_normalize_weights: True       # 是否归一化权重
```

## 使用示例

### 基本使用

```python
from ultra.rdg import build_rdg_edges, RDGConfig

# 创建配置
config = RDGConfig(
    enabled=True,
    min_dependency_weight=0.001,
    precedence_method='indegree'
)

# 构建RDG
edge_index, edge_weights, tau, deps = build_rdg_edges(graph, config)
```

### 在关系图中使用

RDG会自动集成到 `build_relation_graph` 函数中（当 `use_rdg: True` 时）：

```python
# 在 tasks.py 的 build_relation_graph 中
# 如果 flags.use_rdg == True，会自动：
# 1. 提取依赖关系
# 2. 计算优先级
# 3. 添加为第5种边类型
# 4. 存储元数据到 graph.rdg_precedence
```

## 与ULTRA的区别

| 特性 | ULTRA th边 | RDG依赖边 |
|------|-----------|----------|
| **结构** | 对称（双向） | 有向（单向） |
| **权重** | 无或等权重 | 基于频率加权 |
| **优先级** | 无 | 有（τ函数） |
| **语义** | 共现连接 | 逻辑依赖 |
| **用途** | 直接消息传递 | 层次化推理 |

## 潜在提升

1. **跨KG泛化**：依赖模式在不同KG中更稳定
2. **层次化推理**：通过优先级建立关系层次
3. **查询依赖注意力**：为阶段二做准备
4. **未见关系处理**：通过依赖模式推断新关系

## 测试

运行测试脚本验证功能：

```bash
python test_rdg.py
```

测试包括：
- 依赖提取正确性
- Shape一致性
- 优先级计算
- 集成测试
- 潜力分析

## 文件结构

```
ultra/rdg/
├── __init__.py          # 模块导出
├── rdg_builder.py       # 核心实现
└── README.md            # 本文档
```

## 未来扩展

- [ ] 支持多跳依赖路径
- [ ] 拓扑排序方法
- [ ] 查询依赖的注意力机制（阶段二）
- [ ] 动态依赖更新

