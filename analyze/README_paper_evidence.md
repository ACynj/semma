# 论文证据文件索引

## 概述

本目录包含用于解释ARE模型性能变化（显著提升和下降）的数据集构造特征证据。所有证据都经过量化分析，可直接用于论文写作。

---

## 核心证据文件

### 1. 证据总结（推荐首先阅读）
- **`paper_evidence_summary.md`** ⭐
  - 包含所有关键证据的总结
  - 提供论文写作模板和关键数字
  - **建议**: 这是写论文时的主要参考文档

### 2. 详细证据报告
- **`paper_evidence_report.md`**
  - 完整的证据分析报告
  - 包含所有数据集的详细分析
  - 提供典型案例的深入解释

### 3. 数据文件
- **`construction_features_analysis.csv`**
  - 包含所有57个数据集的构造特征数据
  - 可用于进一步分析或验证

---

## 可视化证据（图表）

### 构造特征对比图表

1. **图表21**: `21_construction_features_comparison.png`
   - **内容**: 4个关键构造特征在提升和下降数据集中的分布对比
   - **用途**: 展示语义聚类、类型多样性、层次性、领域特异性的差异
   - **论文使用**: 可用于Results或Analysis部分

2. **图表22**: `22_key_datasets_construction_features.png`
   - **内容**: 关键数据集（Metafam, ConceptNet等）的构造特征热力图
   - **用途**: 直观展示极端案例的构造特征差异
   - **论文使用**: 可用于Case Study部分

3. **图表23**: `23_construction_feature_importance.png`
   - **内容**: 构造特征重要性排序
   - **用途**: 展示哪些构造特征最重要
   - **论文使用**: 可用于Discussion部分

4. **图表24**: `24_construction_characteristics_summary.png`
   - **内容**: 提升和下降数据集的典型构造特征总结
   - **用途**: 总结性展示两类数据集的构造特征差异
   - **论文使用**: 可用于Conclusion部分

---

## 关键发现速查表

### 提升数据集特征（11个）

| 特征 | 占比 | 关键数字 |
|------|------|---------|
| **High语义聚类** | 63.6% | 7/11 ⭐ |
| **Low类型多样性** | 18.2% | 2/11 |
| **Hierarchical层次** | 18.2% | 2/11 |
| **General领域** | 90.9% | 10/11 |

### 下降数据集特征（8个）

| 特征 | 占比 | 关键数字 |
|------|------|---------|
| **High语义聚类** | 0% | 0/8 ⚠️ |
| **High类型多样性** | 12.5% | 1/8 |
| **Flat层次** | 12.5% | 1/8 |
| **Domain Specific** | 25.0% | 2/8 |

### 关键差异

| 特征 | 差异 | 重要性 |
|------|------|--------|
| **语义聚类质量** | 63.6% | ⭐⭐⭐ 最重要 |
| **领域特异性** | 25.0% | ⭐⭐ |
| **类型多样性** | 18.2% | ⭐ |
| **层次性** | 18.2% | ⭐ |

---

## 论文写作建议

### 1. Introduction/Related Work
- 引用数据集构造特征的重要性
- 说明为什么ARE可能在不同构造的数据集上表现不同

### 2. Methodology
- 说明ARE的相似度增强机制
- 解释为什么构造特征会影响ARE效果

### 3. Results
- 使用图表21展示构造特征对比
- 使用图表22展示关键案例

### 4. Analysis/Discussion
- 使用图表23展示特征重要性
- 引用关键数字（63.6%, 25.0%等）
- 解释为什么语义聚类质量最重要

### 5. Conclusion
- 使用图表24总结构造特征
- 总结ARE的适用和不适用场景

---

## 典型案例引用

### 最佳提升案例：Metafam
- **提升**: MRR +74.4%
- **构造特征**: High语义聚类 + Low类型多样性 + Hierarchical层次
- **解释**: 所有构造特征都指向高度结构化

### 最差下降案例：ConceptNet
- **下降**: MRR -15.4%
- **构造特征**: Low语义聚类 + High类型多样性 + Flat层次
- **解释**: 构造特征与Metafam完全相反

---

## 论文表述模板

### 提升原因
> "Our analysis reveals that **63.6% of significantly improved datasets** (7 out of 11) exhibit **high semantic clustering** of relations, compared to **0% of degraded datasets**. This indicates that structured relations form clear clusters in the embedding space, enabling ARE's similarity-based enhancement mechanism to effectively identify and leverage similar relations."

### 下降原因
> "Conversely, datasets with **low semantic clustering** and **high relation type diversity** (e.g., ConceptNet with commonsense relations) show performance degradation. The diverse and unstructured nature of relations prevents the similarity enhancement mechanism from finding meaningful similar relations."

---

## 文件结构

```
analyze/
├── paper_evidence_summary.md          # ⭐ 主要参考文档
├── paper_evidence_report.md            # 详细报告
├── construction_features_analysis.csv # 数据文件
├── extract_dataset_construction_evidence.py # 分析脚本
└── figures/
    ├── 21_construction_features_comparison.png
    ├── 22_key_datasets_construction_features.png
    ├── 23_construction_feature_importance.png
    └── 24_construction_characteristics_summary.png
```

---

## 使用建议

1. **写论文前**: 先阅读 `paper_evidence_summary.md`
2. **写Results部分**: 使用图表21和22
3. **写Analysis部分**: 使用图表23和关键数字
4. **写Conclusion部分**: 使用图表24和总结
5. **需要详细数据**: 查看 `construction_features_analysis.csv`

---

## 关键数字速查（可直接复制）

- **63.6%** - 提升数据集中High语义聚类的占比
- **0%** - 下降数据集中High语义聚类的占比
- **25.0%** - 下降数据集中Domain Specific的占比
- **18.2%** - 提升数据集中Low类型多样性的占比
- **+74.4%** - Metafam的MRR提升（最佳案例）
- **-15.4%** - ConceptNet的MRR下降（最差案例）

---

## 联系信息

如有问题或需要进一步分析，请参考：
- `README_common_features_analysis.md` - 共性特征分析说明
- `README_complete_analysis.md` - 完整分析索引

