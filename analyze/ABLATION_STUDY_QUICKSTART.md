# EnhancedUltra 消融实验快速使用指南

## 一、快速开始

### 1.1 基本用法

运行所有消融实验配置（推荐用于小规模测试）：

```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -tr \
    -reps 1
```

### 1.2 运行特定消融实验

只运行部分消融实验配置：

```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237,Metafam" \
    -a "baseline,similarity_only,full" \
    -tr \
    -reps 3
```

### 1.3 使用预训练检查点进行微调

```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -ckpt /path/to/pretrained_checkpoint.pth \
    -ft \
    -reps 3
```

### 1.4 只进行推理（使用已训练的模型）

```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -ckpt /path/to/trained_checkpoint.pth \
    -reps 3
```

## 二、参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `-c, --config` | YAML配置文件路径 | `config/ablation/ablation_config.yaml` | `-c config/ablation/ablation_config.yaml` |
| `-d, --datasets` | 目标数据集（逗号分隔） | 必需 | `-d "FB15k237,Metafam"` |
| `-a, --ablations` | 消融实验配置（逗号分隔，或'all'） | `all` | `-a "baseline,similarity_only,full"` |
| `-ckpt, --checkpoint` | 检查点路径 | `None` | `-ckpt /path/to/checkpoint.pth` |
| `-ft, --finetune` | 微调检查点 | `False` | `-ft` |
| `-tr, --train` | 从头训练 | `False` | `-tr` |
| `-reps, --repeats` | 每个实验的重复次数 | `1` | `-reps 3` |

## 三、消融实验配置

可用的消融实验配置：

| 配置ID | 名称 | 描述 |
|--------|------|------|
| `baseline` | Baseline (SEMMA) | 无任何增强，作为基线 |
| `similarity_only` | +SimilarityEnhancer | 只使用相似度增强 |
| `similarity_gate` | +SimilarityEnhancer+Gate | 相似度增强 + 自适应门控 |
| `prompt_only` | +PromptGraph | 只使用提示图增强 |
| `similarity_prompt` | +SimilarityEnhancer+PromptGraph | 相似度增强 + 提示图增强 |
| `full` | Full (EnhancedUltra) | 所有组件 |

## 四、结果文件

实验结果会自动保存到 CSV 文件：

- 文件名格式：`ablation_results_YYYY-MM-DD-HH-MM-SS.csv`
- 位置：项目根目录
- 包含字段：
  - `ablation`: 消融实验配置ID
  - `name`: 消融实验名称
  - `dataset`: 数据集名称
  - `seed`: 随机种子
  - `mr`: Mean Rank
  - `mrr`: Mean Reciprocal Rank
  - `hits@1`, `hits@3`, `hits@10`, `hits@10_50`: Hits@K 指标

## 五、示例工作流

### 5.1 完整消融实验（推荐）

1. **小规模测试**（验证配置正确）：
```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -tr \
    -reps 1
```

2. **完整实验**（多个数据集，多次重复）：
```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237,Metafam,ConceptNet100k,YAGO310" \
    -a all \
    -tr \
    -reps 3
```

### 5.2 使用预训练模型进行快速评估

1. **使用预训练检查点进行微调**：
```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -ckpt /path/to/pretrained.pth \
    -ft \
    -reps 3
```

2. **只进行推理**（如果已有训练好的模型）：
```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a all \
    -ckpt /path/to/trained.pth \
    -reps 3
```

## 六、注意事项

1. **计算资源**：消融实验需要运行 6 个配置 × N 个数据集 × M 次重复 = 6×N×M 次实验，请确保有足够的计算资源。

2. **时间成本**：建议先在小规模数据集上测试，再扩展到大规模数据集。

3. **随机种子**：使用固定的随机种子列表 `[1024, 42, 1337, 512, 256]` 以确保可重复性。

4. **flags.yaml 修改**：脚本会自动修改 `flags.yaml` 文件以应用消融实验配置。建议在运行前备份原始文件。

5. **结果分析**：完成实验后，使用 CSV 文件进行结果分析和可视化。

## 七、故障排除

### 7.1 常见错误

1. **找不到配置文件**：
   - 确保配置文件路径正确
   - 检查 `config/ablation/ablation_config.yaml` 是否存在

2. **找不到数据集**：
   - 检查数据集路径配置
   - 确保数据集已正确下载

3. **CUDA 内存不足**：
   - 减少 batch_size
   - 使用更少的 GPU
   - 减少并行实验数量

### 7.2 调试建议

1. **先运行单个实验**：
```bash
python script/run_ablation.py \
    -c config/ablation/ablation_config.yaml \
    -d "FB15k237" \
    -a "baseline" \
    -tr \
    -reps 1
```

2. **检查日志输出**：查看控制台输出和日志文件

3. **验证 flags.yaml**：检查消融实验配置是否正确应用

## 八、更多信息

- 详细设计文档：`analyze/ABLATION_STUDY_DESIGN.md`
- 模型实现：`ultra/enhanced_models.py`
- 根本原因分析：`analyze/are_vs_semma_root_cause_analysis.md`

