# 可学习融合详细说明（use_learnable_fusion: True）

## 概述

当 `use_learnable_fusion: True` 时，模型使用**可学习融合（方案3）**，融合权重会在训练过程中自动学习优化。

## 融合公式

### 基本融合公式

```
weights = softmax(fusion_weights_logits)  # [w_r, w_sim, w_prompt]
final = weights[0] * r + weights[1] * r1 + weights[2] * r2
```

其中：
- `r`: 原始SEMMA融合后的嵌入 `[batch_size, num_relations, embedding_dim]`
- `r1 = r + r1_delta`: similarity_enhancer增强后的完整表示
- `r2 = r + r2_delta`: prompt_enhancer增强后的完整表示
- `weights[0]`: 原始表示r的权重
- `weights[1]`: similarity_enhancer的权重
- `weights[2]`: prompt_enhancer的权重

### 权重归一化

使用 **softmax** 归一化，确保：
- 所有权重和为 1：`weights[0] + weights[1] + weights[2] = 1`
- 所有权重非负：`weights[i] >= 0`
- 所有权重在 [0, 1] 范围内

## 权重初始化

### 初始化过程

1. **读取初始权重**（从flags.yaml）：
   ```python
   initial_r_weight = 1.0
   initial_sim_weight = similarity_enhancer_weight  # 默认 0.2
   initial_prompt_weight = prompt_enhancer_weight   # 默认 0.8
   ```

2. **归一化初始权重**：
   ```python
   total = 1.0 + 0.2 + 0.8 = 2.0
   initial_weights = [1.0/2.0, 0.2/2.0, 0.8/2.0] = [0.5, 0.1, 0.4]
   ```

3. **转换为logits**（用于softmax）：
   ```python
   logits = log(initial_weights)
   # 例如：logits = [-0.6931, -2.3026, -0.9163]
   ```

3. **注册为可学习参数**：
   ```python
   self.fusion_weights_logits = nn.Parameter(logits)
   ```

### 初始权重示例

基于当前flags.yaml配置：
- `similarity_enhancer_weight: 0.2`
- `prompt_enhancer_weight: 0.8`

初始权重（softmax后）：
- `weights[0]` (原始r): **0.5000** (50%)
- `weights[1]` (similarity): **0.1000** (10%)
- `weights[2]` (prompt): **0.4000** (40%)

## 训练过程中的权重更新

### 自动学习

1. **梯度计算**：融合权重参与前向传播，损失函数会计算梯度
2. **权重更新**：优化器（如Adam）根据梯度更新权重
3. **自动优化**：模型自动学习最优的权重组合

### 权重变化示例

训练过程中，权重可能会从初始值：
```
初始: [0.5000, 0.1000, 0.4000]
```

学习到更优的组合，例如：
```
训练后: [0.3000, 0.2000, 0.5000]  # prompt增强更重要
或
训练后: [0.6000, 0.1500, 0.2500]  # 原始表示更重要
```

## 代码实现流程

### 1. 初始化阶段（__init__）

```python
if self.use_learnable_fusion:
    # 基于flags.yaml中的权重初始化
    initial_r_weight = 1.0
    initial_sim_weight = self.similarity_enhancer_weight  # 0.2
    initial_prompt_weight = self.prompt_enhancer_weight   # 0.8
    
    # 归一化
    total = initial_r_weight + initial_sim_weight + initial_prompt_weight
    initial_weights = torch.tensor([
        initial_r_weight / total,      # 0.5
        initial_sim_weight / total,    # 0.1
        initial_prompt_weight / total  # 0.4
    ])
    
    # 转换为logits
    logits = torch.log(initial_weights)
    
    # 注册为可学习参数
    self.fusion_weights_logits = nn.Parameter(logits)
```

### 2. 前向传播阶段（forward）

```python
if self.use_learnable_fusion:
    # 1. 归一化可学习权重
    fusion_weights = F.softmax(self.fusion_weights_logits, dim=0)
    # fusion_weights = [w_r, w_sim, w_prompt]，和为1
    
    # 2. 计算增强后的完整表示
    r1 = r + r1_delta  # similarity增强后的表示
    r2 = r + r2_delta  # prompt增强后的表示
    
    # 3. 加权融合
    final = (
        fusion_weights[0] * r +      # 原始表示
        fusion_weights[1] * r1 +     # similarity增强
        fusion_weights[2] * r2       # prompt增强
    )
```

### 3. 反向传播阶段

```python
# 损失函数计算
loss = criterion(predictions, targets)

# 反向传播（自动计算融合权重的梯度）
loss.backward()

# 优化器更新（包括融合权重）
optimizer.step()
```

## 与固定权重模式的对比

| 特性 | 可学习融合 | 固定权重 |
|------|----------|---------|
| **权重来源** | 模型自动学习 | 手动设置（flags.yaml） |
| **权重更新** | 训练过程中自动更新 | 固定不变 |
| **融合公式** | `w[0]*r + w[1]*r1 + w[2]*r2` | `r + u*r1_delta + θ*r2_delta` |
| **权重归一化** | ✅ softmax归一化（和为1） | ❌ 不归一化 |
| **权重范围** | [0, 1] | 任意值（建议[0, 1]） |
| **调参需求** | 无需手动调参 | 需要手动调参 |
| **适用场景** | 需要模型自动学习最优权重 | 已知最优权重组合 |

## 配置示例

### 启用可学习融合

```yaml
use_learnable_fusion: True  # 启用可学习融合

# 这两个参数仅用于初始化，训练过程中会被学习更新
similarity_enhancer_weight: 0.2  # 初始权重
prompt_enhancer_weight: 0.8      # 初始权重
```

### 初始权重的影响

虽然权重会在训练中更新，但初始权重仍然重要：
- **好的初始权重**：帮助模型更快收敛
- **差的初始权重**：可能导致训练不稳定或收敛到次优解

建议：
- 基于消融实验结果设置初始权重
- 如果prompt_enhancer效果好，设置较大的初始权重
- 如果similarity_enhancer效果好，设置较大的初始权重

## 权重监控建议

在训练过程中，建议监控融合权重的变化：

```python
# 在训练循环中
if epoch % 10 == 0:
    weights = F.softmax(model.fusion_weights_logits, dim=0)
    print(f"Epoch {epoch} - Fusion weights: r={weights[0]:.4f}, "
          f"sim={weights[1]:.4f}, prompt={weights[2]:.4f}")
```

这样可以：
1. 了解模型学习到的融合策略
2. 发现训练异常（如权重突然变化）
3. 分析不同数据集的最优权重组合

## 优势

1. **自适应学习**：模型自动学习最优权重，无需手动调参
2. **归一化保证**：softmax确保权重和为1，数值稳定
3. **灵活性**：不同数据集可能学习到不同的最优权重
4. **可解释性**：训练后的权重可以反映各组件的相对重要性

## 注意事项

1. **学习率**：融合权重会随模型一起训练，可能需要调整学习率
2. **初始化**：初始权重影响训练，建议基于实验结果设置
3. **权重监控**：建议在训练过程中监控权重变化
4. **收敛性**：如果训练不稳定，可以尝试降低学习率或调整初始权重

