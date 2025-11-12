# ARE vs SEMMA 可视化图表说明

本目录包含用于证明ARE（EnhanceUltra）与SEMMA性能对比分析的可视化图表。

## 生成的图表

### 1. `1_performance_comparison_scatter.png`
**ARE vs SEMMA 性能对比散点图**

- **左图**: MRR性能对比散点图
- **右图**: H@10性能对比散点图
- **说明**: 
  - 对角线（红色虚线）表示无差异线（y=x）
  - 点在对角线上方 = ARE表现更好
  - 点在对角线下方 = SEMMA表现更好
  - 颜色表示性能差异（绿色=提升，红色=下降）

**证明点**: 直观展示ARE和SEMMA在所有数据集上的性能对比关系

---

### 2. `2_top_improvements_degradations.png`
**显著提升和下降的数据集**

- **上图**: 显著提升的数据集（MRR提升 > 0.01）
- **下图**: 显著下降的数据集（MRR下降 > 0.01）
- **说明**: 
  - 绿色柱状图表示提升
  - 红色柱状图表示下降
  - 标注了提升/下降的数值和百分比

**证明点**: 
- **Metafam**: 最大提升（MRR +0.192, +74.4%）
- **YAGO310-ht**: 显著提升（MRR +0.082, +20.9%）
- **ConceptNet 100k-ht**: 最大下降（MRR -0.025, -15.4%）
- **WikiTopicsMT3:infra**: 显著下降（MRR -0.033, -5.1%）

---

### 3. `3_dataset_type_comparison.png`
**不同数据集类型的平均性能差异**

- **左图**: 不同数据集类型的平均MRR差异
- **右图**: 不同数据集类型的平均H@10差异
- **说明**: 
  - 误差棒表示标准差
  - 红色虚线表示无差异线（y=0）
  - 柱状图在0线上方 = ARE平均表现更好
  - 柱状图在0线下方 = SEMMA平均表现更好

**证明点**: 
- **Inductive(e,r)**: ARE平均表现最好（MRR +0.009）
- **Transductive**: ARE略好（MRR +0.005）
- **Inductive(e)**: ARE略好（MRR +0.002）
- **Pre-training**: ARE略差（MRR -0.002）

---

### 4. `4_performance_distribution.png`
**性能变化分布直方图**

- **左图**: MRR性能变化分布
- **右图**: H@10性能变化分布
- **说明**: 
  - 红色虚线 = 无差异线（x=0）
  - 绿色虚线 = 平均值
  - 分布偏向右侧 = 更多数据集提升
  - 分布偏向左侧 = 更多数据集下降

**证明点**: 
- 性能变化呈现正态分布，但略微偏向提升
- 平均值略大于0，说明ARE整体略好于SEMMA
- 但存在显著下降的异常值（如ConceptNet）

---

### 5. `5_key_datasets_comparison.png`
**关键数据集详细对比**

- **左图**: 关键数据集的MRR详细对比
- **右图**: 关键数据集的H@10详细对比
- **关键数据集**: 
  - Metafam（最大提升）
  - YAGO310-ht（显著提升）
  - ConceptNet 100k-ht（最大下降）
  - NELLInductive:v1（已表现很好，下降）
  - WikiTopicsMT3:infra（领域特定，下降）
  - FB15K237Inductive:v2（归纳设置，提升）
  - WN18RRInductive:v3（词汇关系，提升）

**证明点**: 
- 清晰展示关键数据集的性能差异
- 绿色标注 = 提升，红色标注 = 下降
- 证明ARE在结构化关系上表现好，在常识关系上表现差

---

### 6. `6_performance_statistics.png`
**性能变化统计饼图**

- **左图**: MRR性能变化统计
- **右图**: H@10性能变化统计
- **分类**: 
  - 绿色：显著提升（>0.01）
  - 红色：显著下降（<-0.01）
  - 灰色：基本持平（±0.01）

**证明点**: 
- 展示提升/下降/持平的数据集数量分布
- 证明ARE在大多数数据集上表现稳定或提升
- 但仍有部分数据集显著下降

---

## 如何运行脚本

```bash
cd /T20030104/ynj/semma
python analyze/visualize_are_vs_semma.py
```

## 依赖包

- matplotlib
- pandas
- numpy
- seaborn

## 图表证明的核心观点

### 1. ARE在结构化关系上表现优异
- **证据**: Metafam（生物关系）提升74.4%，YAGO310（结构化关系）提升20.9%
- **图表**: `2_top_improvements_degradations.png`, `5_key_datasets_comparison.png`

### 2. ARE在常识关系上表现较差
- **证据**: ConceptNet（常识关系）下降15.4%
- **图表**: `2_top_improvements_degradations.png`, `5_key_datasets_comparison.png`

### 3. ARE在归纳设置下表现良好
- **证据**: FB15K237Inductive系列普遍提升
- **图表**: `5_key_datasets_comparison.png`

### 4. ARE在已表现很好的数据集上可能引入干扰
- **证据**: NELLInductive:v1（SEMMA MRR=0.796）下降2.0%
- **图表**: `5_key_datasets_comparison.png`

### 5. ARE整体略好于SEMMA，但存在显著异常值
- **证据**: 平均MRR差异略大于0，但存在显著下降的数据集
- **图表**: `4_performance_distribution.png`, `6_performance_statistics.png`

---

## 总结

这些可视化图表从多个角度证明了分析报告中的核心观点：

1. ✅ **ARE在结构化关系上有效** - 通过Metafam和YAGO310的提升证明
2. ✅ **ARE在常识关系上失效** - 通过ConceptNet的下降证明
3. ✅ **ARE在归纳设置下有效** - 通过FB15K237Inductive系列证明
4. ✅ **ARE可能引入干扰** - 通过NELLInductive:v1的下降证明
5. ✅ **ARE整体略好但存在异常** - 通过分布图和统计图证明

所有图表都保存在 `analyze/figures/` 目录中，可以直接用于论文或报告。

