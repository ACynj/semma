# 继续训练门控网络 - 使用说明

## 配置文件

`continue_train_gate.yaml` - 用于在现有checkpoint基础上继续训练门控网络

## 配置说明

### 关键设置

1. **Checkpoint路径**：
   ```yaml
   checkpoint: ./ckpts/optuna_1.pth
   ```
   - 从现有checkpoint加载预训练权重
   - 门控网络的参数会使用随机初始化（初始权重≈0.7）
   - 其他模块保持预训练权重

2. **学习率**：
   ```yaml
   optimizer:
     lr: 1.0e-4  # 较小的学习率，用于微调
   ```
   - 使用较小的学习率（1e-4）避免破坏预训练权重
   - 适合微调门控网络

3. **训练轮数**：
   ```yaml
   train:
     num_epoch: 5  # 训练5个epoch
   ```
   - 通常3-5个epoch足够让门控网络学习
   - 可以根据验证集性能调整

4. **模型类型**：
   - 由 `flags.yaml` 中的 `run: EnhancedUltra` 决定
   - 确保 `use_adaptive_gate: True` 启用门控机制

## 使用方法

### 1. 确保flags.yaml配置正确

```yaml
run: EnhancedUltra
use_adaptive_gate: True
```

### 2. 运行训练脚本

```bash
cd /T20030104/ynj/semma
python script/pretrain.py --config config/transductive/continue_train_gate.yaml --gpus [0]
```

### 3. 监控训练过程

训练过程中可以监控：
- 损失函数变化
- 验证集性能（MRR, Hits@10）
- 门控权重分布（如果添加了日志）

### 4. 验证门控机制效果

训练完成后，可以：
- 检查不同数据集上的门控权重分布
- 对比性能提升（与SEMMA和原EnhancedUltra对比）

## 训练时间估算

- **5个epoch**：取决于数据集大小
- **JointDataset (3个图)**：约几小时到一天
- **单GPU (batch_size=64)**：每个epoch约1-2小时

## 预期效果

### 训练前（门控权重随机）：
- 门控权重 ≈ 0.7（初始值）
- 性能可能不稳定

### 训练后（门控权重学习）：
- 在增强有帮助的数据集上：门控权重接近1.0
- 在增强有害的数据集上：门控权重接近0.0
- 整体性能优于固定增强策略

## 调整建议

### 如果训练不稳定：
- 减小学习率：`lr: 5.0e-5`
- 增加训练轮数：`num_epoch: 10`

### 如果门控权重学习太慢：
- 增大学习率：`lr: 2.0e-4`
- 检查特征提取是否正常

### 如果只想训练门控网络（冻结其他参数）：
可以修改训练脚本，添加：
```python
# 冻结其他参数，只训练门控网络
for name, param in model.named_parameters():
    if 'enhancement_gate' not in name:
        param.requires_grad = False
```

## 检查点

训练前检查：
- ✅ checkpoint文件存在：`ckpts/optuna_1.pth`
- ✅ flags.yaml中 `run: EnhancedUltra`
- ✅ flags.yaml中 `use_adaptive_gate: True`
- ✅ 配置文件路径正确

训练后检查：
- ✅ 模型保存成功
- ✅ 验证集性能提升
- ✅ 门控权重分布合理

## 故障排除

### 问题1：Checkpoint加载失败
- 检查路径是否正确（相对路径或绝对路径）
- 检查checkpoint文件是否存在
- 查看日志中的错误信息

### 问题2：门控权重不学习
- 检查学习率是否太小
- 检查门控网络是否被正确初始化
- 检查梯度是否正常流动

### 问题3：性能下降
- 学习率可能太大，破坏了预训练权重
- 尝试更小的学习率或只训练门控网络

## 下一步

训练完成后：
1. 在多个数据集上评估性能
2. 分析门控权重分布
3. 对比SEMMA和原EnhancedUltra的性能
4. 根据结果调整配置或改进门控机制

