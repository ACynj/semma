# 性能分析工具使用指南

## ✅ 已添加的性能分析工具

我已经在代码的关键位置添加了性能计时器，可以帮助你定位最慢的部分。

## 📊 性能分析输出

运行模型时，日志中会显示类似以下的信息：

```
[性能] 收集查询实体: 0.15ms
[性能] 计算2跳邻居 (种子数=1): 25.30ms
[性能] [Entity Enhancer] 构建邻接表 (边数=123456): 15.20ms
[性能] [Entity Enhancer] BFS遍历2跳 (种子数=1): 10.10ms
[性能] 计算实体度并排序 (实体数=500): 5.50ms
[性能] 计算实体特征 (实体数=500): 2500.00ms
[性能] Relation Model (structural): 120.50ms
[性能] Semantic Model: 150.30ms
[性能] Combiner: 5.20ms
[性能] Similarity Enhancer: 10.50ms
[性能] Prompt Enhancer (batch_size=32): 500.00ms
[性能] Prompt Enhancer batch 0: 15.60ms
```

## 🔍 如何定位性能瓶颈

### 1. 查看日志输出

运行模型时，查看日志中 `[性能]` 标记的输出，找出耗时最长的操作。

### 2. 重点关注的操作

根据经验，以下操作通常是最慢的：

#### **最可能慢的操作（按优先级）**：

1. **计算实体特征 (实体数=500)**
   - 如果这个操作耗时很长（>1000ms），说明实体特征计算是瓶颈
   - **优化建议**：
     - 进一步减少 `max_entities_to_compute`（从500降到200-300）
     - 或者禁用实体增强（设置 `use_entity_enhancement: False`）

2. **Prompt Enhancer (batch_size=32)**
   - 如果这个操作耗时很长（>500ms），说明提示图增强是瓶颈
   - **优化建议**：
     - 减少 `num_prompt_samples`（从3降到2或1）
     - 或者禁用提示图增强（设置 `use_prompt_enhancer: False`）

3. **构建邻接表 (边数=123456)**
   - 如果边数很大（>100万），这个操作可能很慢
   - **优化建议**：
     - 考虑缓存邻接表（在初始化时构建一次）
     - 或者使用更高效的图数据结构

4. **Relation Model / Semantic Model**
   - 如果这些操作很慢，说明基础模型本身是瓶颈
   - **优化建议**：
     - 检查模型配置
     - 考虑使用更小的模型

### 3. 分析步骤

1. **运行模型并收集日志**
   ```bash
   # 确保日志级别设置为WARNING或更低，以显示性能信息
   python your_training_script.py 2>&1 | grep "\[性能\]"
   ```

2. **找出最慢的操作**
   - 查看所有 `[性能]` 输出，找出耗时最长的操作

3. **根据结果优化**
   - 如果 "计算实体特征" 最慢 → 减少实体数量或禁用实体增强
   - 如果 "Prompt Enhancer" 最慢 → 减少采样数量或禁用提示图增强
   - 如果 "构建邻接表" 最慢 → 考虑缓存或优化图数据结构

## 🎯 快速优化建议

### 如果 "计算实体特征" 最慢：

1. **减少实体数量限制**
   ```python
   # 在 flags.yaml 或代码中
   max_entities_to_compute = 200  # 从500降到200
   ```

2. **或者完全禁用实体增强**
   ```yaml
   # flags.yaml
   use_entity_enhancement: False
   ```

### 如果 "Prompt Enhancer" 最慢：

1. **减少提示样本数量**
   ```python
   # 在 EnhancedUltra.__init__ 中
   num_prompt_samples = 1  # 从3降到1
   ```

2. **或者完全禁用提示图增强**
   ```yaml
   # flags.yaml
   use_prompt_enhancer: False
   ```

### 如果 "构建邻接表" 最慢：

1. **考虑缓存邻接表**
   - 在模型初始化时构建一次，后续复用

2. **使用更高效的图数据结构**
   - 考虑使用 `torch_geometric` 的内置图操作

## 📝 性能分析工具说明

### timer 函数

```python
@contextmanager
def timer(description, logger=None, min_time_ms=10):
    """性能计时器上下文管理器"""
    start = time.time()
    yield
    elapsed_ms = (time.time() - start) * 1000
    if logger is None:
        logger = logging.getLogger(__name__)
    if elapsed_ms >= min_time_ms:  # 只记录超过阈值的操作
        logger.warning(f"[性能] {description}: {elapsed_ms:.2f}ms")
```

### 使用方式

```python
with timer("操作名称", logger):
    # 要计时的代码
    do_something()
```

## ✅ 下一步

1. **运行模型**，查看性能日志
2. **找出最慢的操作**
3. **根据上述建议进行优化**
4. **重新运行**，验证优化效果

如果仍然很慢，请将性能日志发给我，我可以帮你进一步分析！

