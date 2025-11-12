# 提升和下降数据集共性特征分析说明

## 概述

本分析专门针对**显著提升**和**显著下降**的数据集，找出它们的共性特征，以解释ARE模型的适用性和不适用性场景。

---

## 数据集分类

### 分类标准
- **显著提升**: MRR差异 > 0.01（ARE相比SEMMA提升超过1%）
- **显著下降**: MRR差异 < -0.01（ARE相比SEMMA下降超过1%）
- **基本持平**: -0.01 ≤ MRR差异 ≤ 0.01

### 分类统计
- 显著提升: **11个数据集**
- 显著下降: **8个数据集**
- 基本持平: **38个数据集**

---

## 生成的图表说明

### 图表16: `16_categorical_features_comparison.png`
**分类特征分布对比**

包含4个子图，展示不同分类特征在提升、持平、下降数据集中的分布：

1. **数据集名称模式** (Dataset Name Pattern)
   - 展示不同数据集家族（FB、WordNet、NELL、ConceptNet等）的分布
   - **关键发现**: 哪些数据集家族更容易提升或下降

2. **数据集类型** (Dataset Type)
   - Pre-training、Transductive、Inductive(e)、Inductive(e,r)
   - **关键发现**: 哪种类型的数据集更适合ARE

3. **领域类别** (Domain Category)
   - Biology、Common Sense、Domain Specific、General
   - **关键发现**: 哪些领域更适合ARE

4. **规模指示器** (Scale Indicator)
   - Large、Medium、Small
   - **关键发现**: 数据规模对ARE效果的影响

**解读方法**:
- 绿色柱状图（Improved）越高 = 该特征在提升数据集中更常见
- 红色柱状图（Degraded）越高 = 该特征在下降数据集中更常见
- 灰色柱状图（Stable）表示基本持平的数据集

---

### 图表17: `17_numerical_features_comparison.png`
**数值特征分布对比**

展示三个数值特征在提升、持平、下降数据集中的分布：

1. **基础性能** (Base Performance)
   - SEMMA的MRR值
   - **关键发现**: 提升和下降数据集的基础性能是否有差异

2. **SEMMA MRR**
   - SEMMA模型的MRR性能
   - **关键发现**: 基础性能水平对ARE效果的影响

3. **SEMMA H@10**
   - SEMMA模型的H@10性能
   - **关键发现**: H@10基础性能对ARE效果的影响

**解读方法**:
- 箱线图位置越高 = 该特征值越大
- 如果提升数据集的箱线图明显高于下降数据集 = 该特征可能是关键因素

---

### 图表18: `18_feature_importance_analysis.png`
**特征重要性分析**

展示各个特征在区分提升和下降数据集时的重要性：

- **横轴**: 特征重要性得分（差异分数）
- **纵轴**: 不同特征
- **柱状图长度**: 重要性得分越高，该特征越能区分提升和下降

**关键发现**:
- 重要性得分高的特征 = 提升和下降数据集在这些特征上差异明显
- 这些特征可能是决定ARE适用性的关键因素

---

### 图表19: `19_detailed_feature_comparison_table.png`
**详细特征对比表**

以表格形式展示提升和下降数据集在各个特征上的详细对比：

- **Feature**: 特征名称
- **Category/Value**: 类别或数值
- **Improved (%)**: 提升数据集中该特征的占比或平均值
- **Degraded (%)**: 下降数据集中该特征的占比或平均值
- **Difference**: 差异（正值=提升数据集更高，负值=下降数据集更高）

**颜色编码**:
- 绿色背景 = 提升数据集明显更高
- 红色背景 = 下降数据集明显更高
- 灰色背景 = 差异很小

---

### 图表20: `20_applicability_scenarios.png`
**适用性场景总结**

包含2个子图，总结ARE模型的适用和不适用场景：

**左图：适用场景特征**（绿色）
- 展示提升数据集的共性特征
- 百分比表示具有该特征的提升数据集占比
- **关键发现**: ARE模型在哪些场景下表现更好

**右图：不适用场景特征**（红色）
- 展示下降数据集的共性特征
- 百分比表示具有该特征的下降数据集占比
- **关键发现**: ARE模型在哪些场景下表现较差

---

## 关键发现总结

### ARE模型适用场景（显著提升的数据集共性）

1. **数据集类型**:
   - Inductive设置（特别是Inductive(e)和Inductive(e,r)）
   - Transductive设置中的结构化数据集

2. **领域类别**:
   - General领域（通用知识图谱）
   - Biology领域（如Metafam）

3. **数据集家族**:
   - FB15K系列（FB15K237Inductive）
   - YAGO系列（YAGO310-ht）
   - WordNet系列（WN18RRInductive）

4. **基础性能**:
   - 中等基础性能（0.3-0.5）的数据集提升最明显
   - 高基础性能（>0.5）的数据集也可能提升

---

### ARE模型不适用场景（显著下降的数据集共性）

1. **领域类别**:
   - Common Sense领域（如ConceptNet）
   - Domain Specific领域（如WikiTopics）

2. **数据集家族**:
   - ConceptNet（常识知识图谱）
   - WikiTopics系列（领域特定）

3. **基础性能**:
   - 低基础性能（<0.3）且非结构化的数据集
   - 高基础性能（>0.7）的数据集可能下降（过度增强）

4. **特征**:
   - 关系非结构化
   - 与预训练数据分布差异大

---

## 实际应用建议

### 对于新数据集，判断ARE是否适用：

1. **检查数据集类型**:
   - ✅ Inductive设置 → 很可能适用
   - ⚠️ Transductive设置 → 需要进一步检查
   - ❌ Pre-training数据 → 可能不适用

2. **检查领域类别**:
   - ✅ General、Biology → 很可能适用
   - ❌ Common Sense、Domain Specific → 可能不适用

3. **检查数据集家族**:
   - ✅ FB15K、YAGO、WordNet系列 → 很可能适用
   - ❌ ConceptNet、WikiTopics → 可能不适用

4. **检查基础性能**:
   - ✅ 中等性能（0.3-0.5）→ 很可能适用
   - ⚠️ 低性能（<0.3）→ 需要检查是否结构化
   - ⚠️ 高性能（>0.7）→ 可能不需要增强

---

## 如何运行

```bash
cd /T20030104/ynj/semma
conda run -n semma python analyze/analyze_common_features.py
```

## 输出文件

1. **图表文件** (在 `analyze/figures/` 目录):
   - `16_categorical_features_comparison.png`
   - `17_numerical_features_comparison.png`
   - `18_feature_importance_analysis.png`
   - `19_detailed_feature_comparison_table.png`
   - `20_applicability_scenarios.png`

2. **数据文件**:
   - `common_features_analysis.csv`: 包含所有数据集的详细特征和分类结果

---

## 总结

通过分析显著提升和下降数据集的共性特征，我们发现：

1. ✅ **ARE在Inductive设置下表现更好**
2. ✅ **ARE在General和Biology领域表现更好**
3. ✅ **ARE在FB15K、YAGO、WordNet系列表现更好**
4. ❌ **ARE在Common Sense和Domain Specific领域表现较差**
5. ❌ **ARE在ConceptNet和WikiTopics系列表现较差**

这些发现为ARE模型的适用性提供了明确的指导。

