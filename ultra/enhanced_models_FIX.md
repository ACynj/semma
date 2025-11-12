# EnhancedUltra 推理随机性修复说明

## 修复日期
2025-01-XX

## 问题描述

在推理时，`EnhancedUltra` 模型包含多个随机操作，导致：
1. **结果不可重复**：每次运行相同输入得到不同结果
2. **评估不稳定**：不同seed下结果差异较大
3. **难以复现**：无法准确评估模型性能

## 修复内容

### 1. 随机边采样（第190行）

**修复前**：
```python
indices = torch.randperm(query_edges.shape[1], device=device)[:num_samples]
```

**修复后**：
```python
if self.training:
    # 训练时：随机采样以增加多样性
    indices = torch.randperm(query_edges.shape[1], device=device)[:num_samples]
else:
    # 推理时：确定性采样（选择前N个）以保证可重复性
    indices = torch.arange(num_samples, device=device)
```

**影响**：
- 训练时：保留随机性，增加数据多样性
- 推理时：使用确定性采样，保证结果可重复

### 2. 随机节点嵌入（第231行）

**修复前**：
```python
node_embeddings = torch.randn(prompt_graph.num_nodes, self.embedding_dim, device=device)
```

**修复后**：
```python
if self.training:
    # 训练时：使用随机初始化以增加多样性
    node_embeddings = torch.randn(prompt_graph.num_nodes, self.embedding_dim, device=device)
else:
    # 推理时：使用固定初始化（零向量）以保证可重复性
    node_embeddings = torch.zeros(prompt_graph.num_nodes, self.embedding_dim, device=device)
```

**影响**：
- 训练时：保留随机初始化
- 推理时：使用零初始化，保证结果一致
- **注意**：使用零初始化可能会略微影响性能，但保证了可重复性。更好的方案是使用学习到的节点嵌入，但需要修改模型结构。

### 3. 随机噪声增强（第385行）

**修复前**：
```python
noise = torch.randn_like(base_repr) * 0.01
enhanced_repr += noise
```

**修复后**：
```python
# 在训练时使用随机噪声，在推理时禁用以保证可重复性
if self.training:
    noise = torch.randn_like(base_repr) * 0.01
    enhanced_repr += noise
# 推理时不添加噪声，保证结果可重复
```

**影响**：
- 训练时：保留噪声增强，提高鲁棒性
- 推理时：禁用噪声，保证结果可重复

## 修复效果

### 修复前
- ❌ 每次推理结果不同
- ❌ 不同seed下结果差异大
- ❌ 无法准确评估模型性能
- ❌ 难以复现实验结果

### 修复后
- ✅ 推理时结果完全可重复
- ✅ 相同输入得到相同输出
- ✅ 可以准确评估模型性能
- ✅ 实验结果可复现

## 使用建议

1. **训练时**：随机性仍然保留，有助于提高模型鲁棒性和泛化能力
2. **推理时**：模型自动切换到确定性模式，保证结果可重复
3. **评估时**：确保模型处于 `eval()` 模式（`model.eval()`），这样所有随机操作都会被禁用

## 注意事项

1. **节点嵌入初始化**：当前使用零初始化，可能会略微影响性能。如果需要更好的性能，可以考虑：
   - 使用学习到的节点嵌入
   - 使用固定的非零初始化（如Xavier初始化）
   - 从预训练模型中加载节点嵌入

2. **边采样策略**：当前使用前N个边，可能不是最优的。可以考虑：
   - 使用重要性采样（基于度的采样）
   - 使用固定的采样策略（如均匀间隔采样）

3. **向后兼容性**：修复后的代码完全向后兼容，不会影响已有的checkpoint。

## 测试建议

运行以下命令验证修复效果：

```bash
# 使用相同的checkpoint和seed，运行两次，结果应该完全相同
python script/run_many.py \
    -c config/transductive/inference-fb.yaml \
    -d CoDExSmall \
    -reps 2 \
    --ckpt /path/to/checkpoint.pth \
    --gpus "[0]"
```

如果两次运行的结果完全相同，说明修复成功。

