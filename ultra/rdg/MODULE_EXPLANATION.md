# RDG模块工作原理详解

本文档通过具体的shape和例子详细说明RDG模块是如何工作的。

## 一、整体流程概览

```
知识图谱 (KG)
    ↓
[步骤1] 提取关系依赖路径
    ↓
[步骤2] 计算关系优先级 τ
    ↓
[步骤3] 构建RDG边和权重
    ↓
[步骤4] 集成到关系图（作为第5种边类型）
```

## 二、详细步骤说明

### 步骤1：关系依赖提取

#### 输入数据Shape

```python
graph.edge_index: torch.Tensor
    Shape: [2, num_edges]
    ├── [0, :]: 头实体索引 (head entities)
    └── [1, :]: 尾实体索引 (tail entities)
    
graph.edge_type: torch.Tensor
    Shape: [num_edges]
    └── 每条边的关系类型ID

示例：
edge_index = [[0, 1, 0, 3, 0, 4, 4, 1, 5],    # 9条边
              [1, 2, 3, 2, 4, 1, 1, 2, 3]]
              
edge_type = [0, 1, 2, 1, 3, 1, 1, 0, 1]      # 9个关系类型
```

#### 处理过程

**1.1 构建实体-关系映射**

```python
# 遍历所有边，建立映射
entity_to_outgoing = {
    0: [0, 2, 3],  # 实体0作为头的关系: bornIn, livesIn, worksAt
    1: [1],        # 实体1作为头的关系: locatedIn
    3: [1],        # 实体3作为头的关系: locatedIn
    4: [1],        # 实体4作为头的关系: locatedIn
    5: [0]         # 实体5作为头的关系: bornIn
}

entity_to_incoming = {
    1: [0, 1],     # 实体1作为尾的关系: bornIn, locatedIn
    2: [1, 1, 1],  # 实体2作为尾的关系: locatedIn (多次)
    3: [2, 0],     # 实体3作为尾的关系: livesIn, bornIn
    4: [3]         # 实体4作为尾的关系: worksAt
}
```

**1.2 提取依赖路径**

```python
# 对于每条边 (h, r1, t)
for i in range(num_edges):
    h = edge_index[0, i]  # 头实体
    t = edge_index[1, i]  # 尾实体
    r1 = edge_type[i]     # 关系类型
    
    # 查找从 t 出发的边 (t, r2, t2)
    if t in entity_to_outgoing:
        for r2 in entity_to_outgoing[t]:
            # 发现依赖: r1 -> r2
            dependency_count[(r1, r2)] += 1

示例路径：
边0: (0, bornIn, 1) -> 查找实体1的出边
     -> 找到 (1, locatedIn, 2)
     -> 依赖: bornIn -> locatedIn (count += 1)

边2: (0, livesIn, 3) -> 查找实体3的出边
     -> 找到 (3, locatedIn, 2)
     -> 依赖: livesIn -> locatedIn (count += 1)

边4: (0, worksAt, 4) -> 查找实体4的出边
     -> 找到 (4, locatedIn, 1)
     -> 依赖: worksAt -> locatedIn (count += 1)
```

**1.3 计算权重**

```python
total_paths = sum(dependency_count.values())  # 例如: 6

dependency_edges = []
for (r_i, r_j), count in dependency_count.items():
    weight = count / total_paths  # 归一化权重
    if weight >= min_threshold:
        dependency_edges.append((r_i, r_j, weight))

输出示例：
dependency_edges = [
    (0, 1, 0.1667),  # bornIn -> locatedIn
    (2, 1, 0.1667),  # livesIn -> locatedIn
    (3, 1, 0.3333),  # worksAt -> locatedIn
    (1, 0, 0.3333)   # locatedIn -> bornIn (反向)
]
```

### 步骤2：关系优先级计算

#### 计算入度

```python
in_degree = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}  # 初始化

# 对于每条依赖边 (r_i, r_j, weight)
for r_i, r_j, weight in dependency_edges:
    in_degree[r_j] += weight  # r_j 被 r_i 依赖

结果：
in_degree = {
    0: 0.3333,  # bornIn 被 locatedIn 依赖 (权重0.3333)
    1: 0.6667,  # locatedIn 被 bornIn, livesIn, worksAt 依赖
    2: 0.0,     # livesIn 不被依赖
    3: 0.0      # worksAt 不被依赖
}
```

#### 计算优先级 τ

```python
max_degree = max(in_degree.values())  # 0.6667

tau = {}
for r in range(num_relations):
    # 归一化并反转：入度越高，τ越低（更基础）
    tau[r] = 1.0 - (in_degree[r] / max_degree)

结果：
tau = {
    0: 0.5000,  # bornIn: 中等优先级
    1: 0.0000,  # locatedIn: 最低优先级（最基础）
    2: 1.0000,  # livesIn: 最高优先级（最复合）
    3: 1.0000   # worksAt: 最高优先级（最复合）
}
```

### 步骤3：构建RDG张量

#### 构建edge_index

```python
edges_list = [(r_i, r_j) for r_i, r_j, _ in dependency_edges]
# edges_list = [(0,1), (2,1), (3,1), (1,0)]

edge_index = torch.tensor(edges_list, dtype=torch.long).T

Shape: [2, 4]
[[0, 2, 3, 1],   # 源关系 (r_i)
 [1, 1, 1, 0]]   # 目标关系 (r_j)
```

#### 构建edge_weights

```python
weights_list = [weight for _, _, weight in dependency_edges]
# weights_list = [0.1667, 0.1667, 0.3333, 0.3333]

edge_weights = torch.tensor(weights_list, dtype=torch.float)

Shape: [4]
[0.1667, 0.1667, 0.3333, 0.3333]
```

### 步骤4：集成到关系图

#### 原始关系图Shape

```python
# 原有的4种边类型 (hh, tt, ht, th)
rel_graph.edge_index: [2, num_existing_edges]
rel_graph.edge_type: [num_existing_edges]  # 值: 0,1,2,3
rel_graph.num_relations: 4

示例：
edge_index = [[0, 1],    # 假设有2条原有边
              [1, 2]]
edge_type = [0, 1]       # hh, tt
```

#### 添加RDG边

```python
# RDG边作为第5种边类型 (edge_type = 4)
rdg_edge_types = torch.full((4,), 4, dtype=torch.long)
# Shape: [4]
# [4, 4, 4, 4]

# 拼接
combined_edge_index = torch.cat([
    rel_graph.edge_index,    # [2, 2]
    rdg_edge_index           # [2, 4]
], dim=1)

# Shape: [2, 6]
# [[0, 1, 0, 2, 3, 1],   # 原有边 + RDG边
#  [1, 2, 1, 1, 1, 0]]

combined_edge_type = torch.cat([
    rel_graph.edge_type,     # [2]
    rdg_edge_types           # [4]
], dim=0)

# Shape: [6]
# [0, 1, 4, 4, 4, 4]     # 原有类型 + RDG类型(4)
```

#### 最终关系图

```python
rel_graph.edge_index: [2, 6]      # 原有2条 + RDG 4条
rel_graph.edge_type: [6]         # 类型: 0,1,4,4,4,4
rel_graph.num_relations: 5       # 新增类型4

# 存储元数据
graph.rdg_precedence: Dict       # {0: 0.5, 1: 0.0, 2: 1.0, 3: 1.0}
graph.rdg_dependency_edges: List # [(0,1,0.1667), ...]
graph.rdg_edge_weights: [4]     # [0.1667, 0.1667, 0.3333, 0.3333]
```

## 三、完整数据流示例

### 输入：知识图谱

```python
实体: 0=Alice, 1=Beijing, 2=China, 3=Shanghai, 4=Company, 5=Bob
关系: 0=bornIn, 1=locatedIn, 2=livesIn, 3=worksAt

三元组:
(0, 0, 1)  # Alice bornIn Beijing
(1, 1, 2)  # Beijing locatedIn China
(0, 2, 3)  # Alice livesIn Shanghai
(3, 1, 2)  # Shanghai locatedIn China
(0, 3, 4)  # Alice worksAt Company
(4, 1, 1)  # Company locatedIn Beijing
(5, 0, 3)  # Bob bornIn Shanghai
```

### 处理：依赖提取

```python
路径1: (0, bornIn, 1) -> (1, locatedIn, 2)
      → 依赖: bornIn -> locatedIn

路径2: (0, livesIn, 3) -> (3, locatedIn, 2)
      → 依赖: livesIn -> locatedIn

路径3: (0, worksAt, 4) -> (4, locatedIn, 1)
      → 依赖: worksAt -> locatedIn

路径4: (1, locatedIn, 2) -> (2, ?, ?)  # 无出边，跳过
路径5: (4, locatedIn, 1) -> (1, locatedIn, 2)
      → 依赖: locatedIn -> locatedIn (自环，通常跳过)
```

### 输出：RDG结构

```python
依赖边 (4条):
- bornIn -> locatedIn (权重: 0.1667)
- livesIn -> locatedIn (权重: 0.1667)
- worksAt -> locatedIn (权重: 0.3333)
- locatedIn -> bornIn (权重: 0.3333)

优先级 τ:
- locatedIn: 0.0000 (最基础)
- bornIn: 0.5000 (中等)
- livesIn: 1.0000 (最复合)
- worksAt: 1.0000 (最复合)

前驱关系:
- bornIn的前驱: [locatedIn]
- locatedIn的前驱: [] (无，因为它是基础)
- livesIn的前驱: [] (无，因为τ值相同或更高)
- worksAt的前驱: [] (无，因为τ值相同或更高)
```

## 四、Shape总结表

| 阶段 | 张量/变量 | Shape/类型 | 说明 |
|------|----------|-----------|------|
| **输入** | `graph.edge_index` | `[2, num_edges]` | 边索引 |
| | `graph.edge_type` | `[num_edges]` | 关系类型 |
| | `graph.num_relations` | `int` | 关系总数 |
| **处理1** | `entity_to_outgoing` | `Dict[int, List[int]]` | 实体出边映射 |
| | `entity_to_incoming` | `Dict[int, List[int]]` | 实体入边映射 |
| | `dependency_count` | `Dict[Tuple, int]` | 依赖计数 |
| **处理2** | `dependency_edges` | `List[Tuple]` | 依赖边列表 |
| | `in_degree` | `Dict[int, float]` | 入度字典 |
| | `tau` | `Dict[int, float]` | 优先级字典 |
| **输出** | `rdg_edge_index` | `[2, num_deps]` | RDG边索引 |
| | `rdg_edge_weights` | `[num_deps]` | RDG边权重 |
| | `tau` | `Dict[int, float]` | 优先级值 |
| **集成** | `combined_edge_index` | `[2, num_existing + num_deps]` | 合并边索引 |
| | `combined_edge_type` | `[num_existing + num_deps]` | 合并边类型 |
| | `rel_graph.num_relations` | `int` (5) | 关系类型数 |

## 五、关键洞察

1. **依赖方向性**：`r1 -> r2` 表示 r1 的尾实体是 r2 的头实体，建立了有向依赖
2. **权重归一化**：依赖权重基于路径频率，归一化后表示相对重要性
3. **优先级层次**：τ值建立了关系的层次结构，低τ值表示更基础的关系
4. **前驱约束**：只有优先级更低的关系才能作为前驱，确保层次一致性

## 六、与消息传递的集成

在后续的消息传递中（阶段二），RDG边会被用于：

```python
# 在关系图的消息传递中
for layer in range(num_layers):
    for relation in relations:
        # 只从优先级更低的前驱关系聚合信息
        preceding = get_preceding_relations(relation, deps, tau)
        messages = aggregate([h[r] for r in preceding])
        h[relation] = update(h[relation], messages)
```

这确保了信息从基础关系流向复合关系，符合逻辑依赖的层次结构。

