# 逆关系向量嵌入实现说明

## 概述

根据您的需求，我们已经成功实现了基于关系类型分类的逆关系向量嵌入生成方法。该实现根据关系的类型（对称、反对称、非对称）采用不同的策略来生成逆关系的嵌入向量。

## 实现的功能

### 1. 配置项添加
- 在 `flags.yaml` 中添加了 `is-inverse-relation-classify` 配置项
- 当设置为 `True` 时，启用基于关系类型分类的逆关系嵌入生成

### 2. 关系类型分类
- 实现了 `load_relation_types()` 函数，从JSON文件加载关系类型信息
- 实现了 `get_relation_type()` 函数，获取特定关系的类型
- 支持三种关系类型：Symmetric（对称）、Asymmetric（非对称）、Antisymmetric（反对称）

### 3. 逆关系语义信息提取
- 实现了 `get_inverse_relation_semantics()` 函数，从JSON文件提取逆关系的名称和描述
- 支持从 `cleaned_inverse_relations` 和 `inverse_relations_descriptions` 字段获取信息

### 4. 嵌入生成策略
根据关系类型采用不同的策略：

#### 对称关系 (Symmetric)
- 逆关系嵌入 = 原关系嵌入
- 例如：`sibling` 关系的逆关系嵌入与原关系相同

#### 反对称关系 (Antisymmetric)  
- 逆关系嵌入 = -原关系嵌入
- 例如：`child` 关系的逆关系嵌入为原关系的负值

#### 非对称关系 (Asymmetric)
- 逆关系嵌入 = 基于语义信息生成的嵌入
- 使用 `generate_inverse_embeddings_for_asymmetric()` 函数
- 结合逆关系的名称和描述，通过jina-embeddings生成嵌入

### 5. 核心函数修改

#### `order_embeddings()` 函数
- 添加了 `dataset_name` 参数
- 根据配置决定是否使用分类策略
- 保持与原有方案的兼容性

#### `build_relation_graph_exp()` 函数
- 添加了 `dataset_name` 参数
- 在调用 `order_embeddings()` 时传递数据集名称

#### 数据集调用修改
- 修改了 `datasets.py` 中的相关调用
- 确保 `dataset_name` 参数正确传递

## 技术细节

### 嵌入维度一致性
- 生成的嵌入向量维度为64（与原有方案一致）
- 使用相同的jina-embeddings模型
- 保持与现有代码的兼容性

### 错误处理
- 当JSON文件不存在时，提供警告信息
- 当关系类型未知时，默认为非对称关系
- 当逆关系语义信息缺失时，使用原关系名称作为备选

### 性能考虑
- 只在启用分类功能时才进行额外的语义处理
- 保持原有逻辑的向后兼容性
- 避免不必要的计算开销

## 测试验证

我们创建了完整的测试套件，验证了以下功能：
1. ✅ 关系类型加载（CoDExMedium: 51个关系，FB15k237: 237个关系，WN18RR: 11个关系）
2. ✅ 关系类型分类（正确识别对称、非对称、反对称关系）
3. ✅ 逆关系语义信息提取（正确提取名称和描述）
4. ✅ 嵌入生成（生成64维嵌入向量）

## 使用方法

1. 在 `flags.yaml` 中设置 `is-inverse-relation-classify: True`
2. 确保相应的JSON文件存在于 `openrouter/relations_type/gpt-4o-2024-11-20/` 目录
3. 运行训练或推理代码，系统会自动使用新的逆关系嵌入生成策略

## 文件修改清单

1. `flags.yaml` - 添加配置项
2. `ultra/tasks.py` - 核心实现逻辑
3. `ultra/datasets.py` - 数据集调用修改

## 兼容性

- 完全向后兼容，不影响现有功能
- 当 `is-inverse-relation-classify` 为 `False` 时，使用原有逻辑
- 生成的嵌入向量维度与原有方案完全一致
