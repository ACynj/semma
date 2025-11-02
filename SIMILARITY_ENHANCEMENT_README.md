# 基于余弦相似度的关系增强模块

## 概述

在 EnhancedUltra 模型基础上，新增了基于余弦相似度的关系增强模块（`SimilarityBasedRelationEnhancer`），该模块能够根据查询关系与其他关系的相似度，智能地参考相似关系来增强查询关系的表示。

## 核心设计思路

1. **相似度计算**：对每个查询关系，计算它与所有关系的余弦相似度
2. **阈值过滤**：只参考相似度超过可学习阈值的关系
3. **加权融合**：根据相似度大小加权融合相似关系的表示
4. **可学习参数**：包含多个可学习参数，允许模型自适应调整增强策略
5. **保守增强**：使用较小的增强强度（默认0.05），确保不影响原SEMMA模型的性能

## 可学习参数

模块包含以下可学习参数：

1. **similarity_threshold_raw**：相似度阈值（经过sigmoid映射到0-1范围）
   - 初始值：0.5
   - 作用：控制哪些相似关系会被参考

2. **enhancement_strength_raw**：增强强度（经过sigmoid映射到0-0.2范围）
   - 初始值：0.05
   - 作用：控制增强的强度，保持较小以避免过度影响原模型

3. **similarity_weight_scale**：相似度权重缩放因子
   - 初始值：1.0
   - 作用：调整相似度对权重的影响程度

4. **temperature**：温度参数（用于softmax缩放）
   - 初始值：1.0
   - 作用：控制相似度分布的平滑度

## 实现细节

### SimilarityBasedRelationEnhancer 模块

```python
class SimilarityBasedRelationEnhancer(nn.Module):
    def forward(self, final_relation_representations, query_rels):
        # final_relation_representations: [batch_size, num_relations, embedding_dim]
        # query_rels: [batch_size] 查询关系索引
        
        # 1. 对每个查询关系，计算与所有关系的余弦相似度
        # 2. 找到相似度超过阈值的关系
        # 3. 根据相似度加权融合
        # 4. 应用小的增强强度更新查询关系表示
```

### 工作流程

1. **获取查询关系表示**：从 `final_relation_representations` 中提取查询关系的表示
2. **计算余弦相似度**：与所有关系的表示计算余弦相似度
3. **阈值过滤**：筛选相似度超过阈值的关系
4. **加权融合**：
   - 使用温度缩放的softmax计算基础权重
   - 使用可学习的缩放因子进一步调整权重
   - 计算加权平均的相似关系表示
5. **应用增强**：`enhanced = (1 - strength) * original + strength * weighted_similar`

## 集成到 EnhancedUltra

在 `EnhancedUltra` 的 `forward` 方法中：

```python
# 获取基础关系表示（SEMMA标准流程）
self.final_relation_representations = self.combiner(
    self.relation_representations_structural, 
    self.relation_representations_semantic
)

# 应用基于相似度的增强
self.enhanced_relation_representations = self.similarity_enhancer(
    self.final_relation_representations, 
    query_rels
)

# 使用增强后的表示进行实体推理
score = self.entity_model(data, self.enhanced_relation_representations, batch)
```

## 优势

1. **最小化对原模型的影响**：增强强度默认很小（0.05），不会显著改变原SEMMA模型的行为
2. **自适应性**：所有关键参数都是可学习的，模型可以自动调整最佳策略
3. **相似度驱动**：只参考真正相似的关系，避免引入噪声
4. **灵活性**：可以根据相似度动态调整参考的程度

## 使用方法

### 运行 EnhancedUltra

确保 `flags.yaml` 中 `run: EnhancedUltra`：

```yaml
run: EnhancedUltra
```

### 训练

```bash
python script/pretrain.py -c config/transductive/pretrain_3g.yaml --gpus [0]
```

### 推理/评估

```bash
python script/run.py -c config/transductive/run_3g.yaml --gpus [0]
```

## 参数调整建议

如果需要调整增强强度：

1. **增加增强强度**：在初始化时设置 `enhancement_strength_init` 更大的值（但仍建议 < 0.2）
2. **调整相似度阈值**：修改 `similarity_threshold_init`（建议范围 0.3-0.7）
3. **训练过程中**：参数会自动学习，无需手动调整

## 预期效果

- **保持SEMMA性能**：由于增强强度较小，原模型性能不应下降
- **小幅提升**：通过参考相似关系，可能在部分查询上获得提升
- **更好的泛化**：对于相似关系的查询，能够利用相关关系的知识

## 注意事项

1. 增强模块在训练和推理时都会使用，但强度较小，不会显著影响训练过程
2. 如果某些查询关系没有找到相似关系（相似度都低于阈值），将保持原表示不变
3. 可学习参数会在训练过程中自动优化，初始值只是起点

## 技术细节

- **余弦相似度计算**：使用归一化的向量点积计算，保证相似度在[-1, 1]范围
- **阈值过滤**：只考虑相似度 > 阈值的正相关关系
- **加权策略**：使用softmax确保权重和为1，相似度越高权重越大
- **增强公式**：线性插值，`(1-α)*original + α*similar`，其中α为增强强度


