# EnhancedUltra 消融实验设计文档

## 一、实验目标

通过消融实验验证 EnhancedUltra 模型中各个增强组件的贡献，理解每个组件对模型性能的影响。

## 二、模型组件分析

### 2.1 EnhancedUltra 架构

EnhancedUltra 在 SEMMA 基础上新增了三个增强组件：

1. **SimilarityBasedRelationEnhancer（相似度关系增强器）**
   - 功能：基于余弦相似度，利用相似关系来增强查询关系表示
   - 关键参数：
     - `similarity_threshold_init`: 0.85（相似度阈值）
     - `enhancement_strength_init`: 0.09（增强强度）
   - 工作原理：
     - 计算查询关系与所有关系的余弦相似度
     - 筛选相似度 > threshold 的关系
     - 使用 softmax 对相似关系进行加权
     - 混合：`enhanced = (1 - strength) * original + strength * weighted_similar`

2. **AdaptiveEnhancementGate（自适应增强门控）**
   - 功能：基于查询特征学习决定是否应该使用增强
   - 输入特征：
     - 查询关系嵌入（64维）
     - 查询实体嵌入（64维）
     - 图统计特征（4维：关系频率、实体度、相似度、图密度）
   - 输出：门控权重（0-1），控制增强强度
   - 注意：需要配合 SimilarityEnhancer 使用

3. **OptimizedPromptGraph（优化提示图增强）**
   - 功能：基于提示图上下文增强关系表示
   - 参数：
     - `max_hops`: 2（最大跳数）
     - `num_prompt_samples`: 3（提示样本数）
   - 工作原理：
     - 生成查询关系的提示图
     - 编码提示图上下文
     - 自适应融合上下文信息

### 2.2 基础组件（SEMMA）

- **RelNBFNet**: 结构关系模型
- **SemRelNBFNet**: 语义关系模型
- **CombineEmbeddings**: 融合器（MLP/Attention/Concat）

## 三、消融实验设计

### 3.1 实验配置

我们设计了 6 种消融实验配置：

| 配置ID | 名称 | SimilarityEnhancer | AdaptiveGate | PromptGraph | 描述 |
|--------|------|-------------------|--------------|-------------|------|
| `baseline` | Baseline (SEMMA) | ❌ | ❌ | ❌ | 无任何增强，作为基线 |
| `similarity_only` | +SimilarityEnhancer | ✅ | ❌ | ❌ | 只使用相似度增强 |
| `similarity_gate` | +SimilarityEnhancer+Gate | ✅ | ✅ | ❌ | 相似度增强 + 自适应门控 |
| `prompt_only` | +PromptGraph | ❌ | ❌ | ✅ | 只使用提示图增强 |
| `similarity_prompt` | +SimilarityEnhancer+PromptGraph | ✅ | ❌ | ✅ | 相似度增强 + 提示图增强 |
| `full` | Full (EnhancedUltra) | ✅ | ✅ | ✅ | 所有组件 |

### 3.2 实验假设

1. **假设1**: SimilarityEnhancer 是核心组件，应该带来最大性能提升
2. **假设2**: AdaptiveGate 能够自适应控制增强强度，在合适的数据集上提升性能
3. **假设3**: PromptGraph 提供额外的上下文信息，可能在某些场景下有用
4. **假设4**: 组件组合可能产生协同效应

### 3.3 实验数据集选择

建议在以下类型的数据集上运行消融实验：

1. **提升显著的数据集**（验证组件有效性）：
   - Metafam (Inductive(e,r)): +74.4% MRR
   - YAGO310-ht (Transductive): +20.9% MRR
   - WN18RRInductive:v3 (Inductive(e)): +5.2% MRR

2. **下降显著的数据集**（验证组件负面影响）：
   - ConceptNet 100k-ht (Transductive): -15.4% MRR
   - WikiTopicsMT3:infra (Inductive(e,r)): -5.1% MRR
   - AristoV4-ht (Transductive): -7.7% MRR

3. **代表性数据集**：
   - FB15k237 (Transductive, 标准数据集)
   - WN18RR (Transductive, 标准数据集)

## 四、实验流程

### 4.1 运行消融实验

#### 方法1: 使用运行脚本（推荐）

```bash
# 运行所有消融实验配置
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237,Metafam,ConceptNet100k" \
    -a all \
    -tr \
    -reps 3

# 运行特定消融实验配置
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a "baseline,similarity_only,full" \
    -tr \
    -reps 3

# 使用预训练检查点进行微调
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -ckpt /path/to/checkpoint.pth \
    -ft \
    -reps 3

# 只进行推理（使用已训练的模型）
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -ckpt /path/to/checkpoint.pth \
    -reps 3
```

#### 方法2: 手动修改 flags.yaml

1. 修改 `flags.yaml` 中的消融实验配置：
```yaml
run: EnhancedUltra  # 或 semma（对于 baseline）
use_similarity_enhancer: True  # 或 False
use_prompt_enhancer: True  # 或 False
use_adaptive_gate: True  # 或 False
```

2. 运行标准训练/测试脚本

### 4.2 结果分析

实验完成后，结果会保存在 `ablation_results_YYYY-MM-DD-HH-MM-SS.csv` 文件中，包含以下指标：

- MR (Mean Rank)
- MRR (Mean Reciprocal Rank)
- Hits@1, Hits@3, Hits@10, Hits@10_50

#### 分析步骤：

1. **计算性能变化**：
   - 相对于 baseline 的性能提升/下降
   - 各组件的独立贡献
   - 组件组合的协同效应

2. **统计分析**：
   - 多次运行的平均值和标准差
   - 显著性检验（如需要）

3. **可视化**：
   - 绘制各组件的性能对比图
   - 绘制不同数据集上的性能变化

## 五、预期结果分析

### 5.1 基于现有分析报告的预期

根据 `are_vs_semma_root_cause_analysis.md` 的分析：

#### 在提升显著的数据集上（如 Metafam）：
- **Baseline**: 基准性能
- **+SimilarityEnhancer**: 应该带来显著提升（核心组件）
- **+AdaptiveGate**: 可能进一步提升（自适应控制）
- **+PromptGraph**: 可能带来额外提升（上下文信息）
- **Full**: 应该达到最佳性能

#### 在下降显著的数据集上（如 ConceptNet）：
- **Baseline**: 基准性能
- **+SimilarityEnhancer**: 可能带来负面影响（关系语义模糊）
- **+AdaptiveGate**: 可能缓解负面影响（自适应控制）
- **+PromptGraph**: 影响可能较小
- **Full**: 可能仍然有负面影响

### 5.2 关键洞察

通过消融实验，我们希望回答以下问题：

1. **SimilarityEnhancer 是否总是有益的？**
   - 在哪些数据集上有益？
   - 在哪些数据集上有害？
   - 为什么？

2. **AdaptiveGate 是否能够缓解负面影响？**
   - 是否能够学习到"何时不使用增强"的策略？
   - 在哪些场景下最有效？

3. **PromptGraph 的贡献如何？**
   - 是否提供额外的有用信息？
   - 与 SimilarityEnhancer 是否有协同效应？

4. **组件组合的协同效应如何？**
   - 是否有 1+1>2 的效果？
   - 是否有相互干扰的情况？

## 六、实验注意事项

### 6.1 实验设置

1. **随机种子**: 使用固定的随机种子列表 `[1024, 42, 1337, 512, 256]` 以确保可重复性
2. **训练配置**: 使用与原始实验相同的训练配置（epochs, batch_per_epoch）
3. **评估指标**: 使用相同的评估指标（MR, MRR, Hits@K）

### 6.2 计算资源

- 消融实验需要运行 6 个配置 × N 个数据集 × M 次重复 = 6×N×M 次实验
- 建议先在小规模数据集上测试，再扩展到大规模数据集
- 可以使用预训练检查点进行微调，以节省时间

### 6.3 结果记录

- 每次实验都会生成详细日志
- 结果会自动保存到 CSV 文件
- 建议定期备份结果文件

## 七、后续分析

完成消融实验后，建议进行以下分析：

1. **组件贡献分析**: 量化每个组件的独立贡献
2. **数据集特征分析**: 分析哪些数据集特征与组件效果相关
3. **参数敏感性分析**: 分析关键参数（如 similarity_threshold_init）的影响
4. **改进建议**: 基于消融实验结果提出模型改进建议

## 八、参考文献

- EnhancedUltra 模型实现: `ultra/enhanced_models.py`
- 根本原因分析: `analyze/are_vs_semma_root_cause_analysis.md`
- 性能对比分析: `analyze/are_vs_semma_analysis.md`

