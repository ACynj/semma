# 固定权重配置说明（use_learnable_fusion: False）

## 配置方式

当 `use_learnable_fusion: False` 时，使用**固定权重模式（方案1）**，权重通过以下两个参数配置：

```yaml
# Parallel Fusion Settings
use_learnable_fusion: False  # 禁用可学习融合

# Enhancement Module Weights (控制两个增强器的贡献权重)
similarity_enhancer_weight: 0.2  # similarity_enhancer的权重 u
prompt_enhancer_weight: 0.8     # prompt_enhancer的权重 θ
```

## 融合公式

固定权重模式下的融合公式：

```
final = r + u * r1_delta + θ * r2_delta
```

其中：
- `r`: 原始SEMMA融合后的嵌入 `[batch_size, num_relations, embedding_dim]`
- `r1_delta`: similarity_enhancer的增量（基于r计算）
- `r2_delta`: prompt_enhancer的增量（基于r计算）
- `u = similarity_enhancer_weight`: 从flags.yaml读取，默认0.2
- `θ = prompt_enhancer_weight`: 从flags.yaml读取，默认0.8

## 权重范围

- **权重范围**: `[0.0, 1.0]` 或更大（可以超过1.0，但建议在合理范围内）
- **权重含义**: 
  - `0.0`: 完全禁用该增强器
  - `1.0`: 使用该增强器的完整输出
  - `>1.0`: 增强该组件的贡献（可能不稳定，需谨慎使用）

## 配置示例

### 示例1：只使用prompt_enhancer
```yaml
use_learnable_fusion: False
similarity_enhancer_weight: 0.0  # 禁用similarity_enhancer
prompt_enhancer_weight: 1.0     # 完全使用prompt_enhancer
```

### 示例2：只使用similarity_enhancer
```yaml
use_learnable_fusion: False
similarity_enhancer_weight: 1.0  # 完全使用similarity_enhancer
prompt_enhancer_weight: 0.0      # 禁用prompt_enhancer
```

### 示例3：平衡使用两个增强器
```yaml
use_learnable_fusion: False
similarity_enhancer_weight: 0.5  # 中等权重
prompt_enhancer_weight: 0.5      # 中等权重
```

### 示例4：当前推荐配置（prompt更强）
```yaml
use_learnable_fusion: False
similarity_enhancer_weight: 0.2  # 较小权重
prompt_enhancer_weight: 0.8      # 较大权重
```

## 与可学习融合的对比

| 特性 | 固定权重模式 | 可学习融合模式 |
|------|------------|--------------|
| 配置参数 | `similarity_enhancer_weight`, `prompt_enhancer_weight` | `use_learnable_fusion: True` |
| 权重来源 | 手动设置（flags.yaml） | 模型自动学习 |
| 权重更新 | 固定不变 | 训练过程中自动更新 |
| 融合公式 | `r + u*r1_delta + θ*r2_delta` | `w[0]*r + w[1]*r1 + w[2]*r2` |
| 权重归一化 | 无（可以任意值） | 有（softmax归一化，和为1） |
| 适用场景 | 已知最优权重组合 | 需要模型自动学习最优权重 |

## 代码实现位置

在 `ultra/enhanced_models.py` 的 `forward` 方法中：

```python
if self.use_learnable_fusion and self.fusion_weights_logits is not None:
    # 可学习融合模式
    ...
else:
    # 固定权重模式
    self.enhanced_relation_representations = (
        r + 
        self.similarity_enhancer_weight * r1_delta + 
        self.prompt_enhancer_weight * r2_delta
    )
```

## 注意事项

1. **权重不归一化**：固定权重模式下，权重不需要归一化，可以任意设置
2. **权重可以超过1.0**：理论上可以设置大于1.0的权重，但可能导致数值不稳定
3. **需要手动调参**：固定权重需要根据实验结果手动调整，找到最优组合
4. **与自适应门控的配合**：如果启用 `use_adaptive_gate`，similarity_enhancer的权重会再乘以门控权重

