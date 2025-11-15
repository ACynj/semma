# 可学习融合效果差的原因总结

## ✅ 诊断完成

### 检查结果

1. **Checkpoint检查**：
   - ✅ Checkpoint中有 `fusion_weights_logits` 参数
   - ✅ 权重形状正确：`[2]`（增量融合）
   - ✅ 权重值：similarity=0.2250, prompt=0.7750
   - ✅ 权重正确加载，没有警告

2. **权重对比**：
   - 训练后权重：similarity=0.2250, prompt=0.7750
   - 固定权重：similarity=0.2000, prompt=0.8000
   - 差异：similarity增加了12.5%，prompt减少了3.1%

3. **训练配置**：
   - ✅ 训练时使用：`use_learnable_fusion: True`
   - ✅ 初始权重：similarity=0.2, prompt=0.8
   - ✅ 训练任务：MultiGraphPretraining

## 🎯 根本原因

### 问题：训练后的权重不是最优的

**核心原因**：

1. **预训练任务不适合学习融合权重**
   - 预训练在多个数据集上联合训练（FB15K237, WN18RR, CoDExMedium）
   - 融合权重学习到了"通用"但"次优"的值
   - 在特定数据集上推理时，这个通用权重不是最优的

2. **权重学习不充分**
   - 初始权重：similarity=0.2, prompt=0.8
   - 训练后权重：similarity=0.225, prompt=0.775
   - 变化很小（12.5%），说明可能没有充分学习

3. **权重可能学习到了次优值**
   - 训练后增加了similarity的权重
   - 但从消融实验看，similarity增强在某些数据集上效果不如prompt增强
   - 固定权重（0.2/0.8）是经过调优的，更适合大多数数据集

## 📊 证据

### 从消融实验看：
- **Abl1** (similarity only): 在部分数据集上效果不如ARE
- **Abl2** (prompt only): 在部分数据集上效果不如ARE  
- **ARE** (完整模型，固定权重): 在大多数数据集上效果最好

这说明固定权重（0.2/0.8）是经过调优的，比较适合。

### 从推理结果看：
- FBNELL: 可学习融合 MRR=0.4387 vs ARE MRR=0.483（下降9.2%）
- 其他数据集也有类似下降

## 🔧 解决方案

### 方案1：使用固定权重（推荐，立即修复）

```yaml
# flags.yaml
use_learnable_fusion: False
similarity_enhancer_weight: 0.2
prompt_enhancer_weight: 0.8
```

**优点**：
- ✅ 立即修复，不需要重新训练
- ✅ 基于消融实验调优的权重，效果更好
- ✅ 简单可靠

### 方案2：使用训练后的权重作为固定权重

```yaml
# flags.yaml
use_learnable_fusion: False
similarity_enhancer_weight: 0.225
prompt_enhancer_weight: 0.775
```

**优点**：
- ✅ 如果训练后的权重确实学到了有用信息
- ✅ 可以作为固定权重使用

### 方案3：重新训练（长期方案）

如果确实想使用可学习融合：
1. **在特定数据集上微调**，而不是在预训练时学习
2. **调整学习率**，给融合权重单独的学习率
3. **调整初始权重**，基于消融实验结果

## 📝 立即行动

### 步骤1：验证固定权重效果

修改 `flags.yaml`：
```yaml
use_learnable_fusion: False
similarity_enhancer_weight: 0.2
prompt_enhancer_weight: 0.8
```

然后重新运行推理，对比效果。

### 步骤2：如果固定权重效果好

说明问题确实在于训练后的权重，可以：
- 使用固定权重模式（推荐）
- 或者尝试方案2（使用训练后的权重值）

## 🎯 结论

**根本原因**：预训练任务不适合学习融合权重，训练后的权重（0.225/0.775）不是最优的。

**解决方案**：使用固定权重模式（`use_learnable_fusion: False`），固定权重（0.2/0.8）是基于消融实验调优的，效果更好。

**建议**：立即修改 `flags.yaml`，使用固定权重模式验证效果。

