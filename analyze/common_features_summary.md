# 提升和下降数据集共性特征总结

## 执行摘要

通过分析11个显著提升和8个显著下降的数据集，我们发现了ARE模型的适用性和不适用性场景的明确模式。

---

## 显著提升的数据集（11个）

### 数据集列表
1. **Metafam** - MRR提升0.192 (+74.4%) ⭐ 最大提升
2. **YAGO310-ht** - MRR提升0.082 (+20.9%)
3. **FB15K237Inductive:v2** - MRR提升0.021 (+4.2%)
4. **WN18RRInductive:v3** - MRR提升0.023 (+5.2%)
5. **FB15K237Inductive:v1** - MRR提升0.013 (+2.7%)
6. **FB15K237Inductive:v3** - MRR提升0.010 (+2.0%)
7. **FB15K237Inductive:v4** - MRR提升0.010 (+2.0%)
8. **NELL995-ht** - MRR提升0.013 (+2.9%)
9. **NELLInductive:v4** - MRR提升0.011 (+2.2%)
10. **WKIngram:25** - MRR提升0.013 (+4.3%)
11. **NLIngram:25** - MRR提升0.014 (+3.6%)

### 共性特征分析

#### 1. 数据集类型分布
- **Inductive(e)**: 6个（55%）
- **Inductive(e,r)**: 3个（27%）
- **Transductive**: 2个（18%）

**结论**: ✅ **Inductive设置是ARE最适用的场景**

#### 2. 领域类别分布
- **General**: 10个（91%）
- **Biology**: 1个（9%，但提升最大）

**结论**: ✅ **General领域是ARE的主要适用场景**

#### 3. 数据集家族分布
- **FB15K系列**: 4个（36%）
- **WordNet系列**: 1个（9%）
- **NELL系列**: 2个（18%）
- **YAGO系列**: 1个（9%）
- **其他**: 3个（27%）

**结论**: ✅ **FB15K系列在Inductive设置下表现最好**

#### 4. 基础性能分布
- **平均基础性能**: 0.427
- **范围**: 0.258 - 0.503
- **中位数**: 0.442

**结论**: ✅ **中等基础性能（0.3-0.5）的数据集提升最明显**

#### 5. 规模指示器
- **Large**: 1个
- **Medium**: 8个
- **Small**: 2个

**结论**: ⚠️ **规模不是关键因素**

---

## 显著下降的数据集（8个）

### 数据集列表
1. **ConceptNet 100k-ht** - MRR下降0.025 (-15.4%) ⭐ 最大下降
2. **WikiTopicsMT3:infra** - MRR下降0.033 (-5.1%)
3. **NELLInductive:v1** - MRR下降0.016 (-2.0%)
4. **WikiTopicsMT1:health** - MRR下降0.018 (-5.4%)
5. **NELLInductive:v3** - MRR下降0.012 (-2.3%)
6. **AristoV4-ht** - MRR下降0.017 (-7.7%)
7. **WDsinger-ht** - MRR下降0.011 (-3.0%)
8. **NLIngram:75** - MRR下降0.011 (-3.1%)

### 共性特征分析

#### 1. 数据集类型分布
- **Inductive(e)**: 2个（25%）
- **Inductive(e,r)**: 3个（38%）
- **Transductive**: 3个（38%）

**结论**: ⚠️ **下降数据集也包含Inductive设置，说明Inductive不是充分条件**

#### 2. 领域类别分布
- **Common Sense**: 1个（13%，但下降最大）
- **Domain Specific**: 2个（25%）
- **General**: 5个（63%）

**结论**: ❌ **Common Sense和Domain Specific领域更容易下降**

#### 3. 数据集家族分布
- **ConceptNet**: 1个（13%，但下降最大）
- **WikiTopics**: 2个（25%）
- **NELL系列**: 2个（25%）
- **其他**: 3个（38%）

**结论**: ❌ **ConceptNet和WikiTopics系列更容易下降**

#### 4. 基础性能分布
- **平均基础性能**: 0.427
- **范围**: 0.162 - 0.796
- **中位数**: 0.368

**结论**: ⚠️ **基础性能范围很大，说明基础性能不是决定性因素**

#### 5. 特殊案例
- **NELLInductive:v1**: 基础性能0.796（很高），但下降
- **ConceptNet**: 基础性能0.162（很低），且下降

**结论**: ❌ **高基础性能和低基础性能都可能下降，取决于数据集特征**

---

## 关键对比发现

### 1. 数据集类型
- **提升数据集**: 82%是Inductive设置
- **下降数据集**: 63%是Inductive设置
- **结论**: Inductive设置是提升的必要条件，但不是充分条件

### 2. 领域类别
- **提升数据集**: 91%是General领域
- **下降数据集**: 63%是General领域，但包含Common Sense和Domain Specific
- **结论**: General领域更适合，但Common Sense和Domain Specific不适合

### 3. 基础性能
- **提升数据集平均**: 0.427
- **下降数据集平均**: 0.427
- **结论**: 基础性能相近，说明基础性能不是关键因素

### 4. 数据集家族
- **提升数据集**: 主要是FB15K、WordNet、YAGO系列
- **下降数据集**: 主要是ConceptNet、WikiTopics系列
- **结论**: 数据集家族是重要因素

---

## ARE模型适用性判断标准

### ✅ 高度适用场景（预期显著提升）

**必须满足**:
1. ✅ Inductive设置（Inductive(e)或Inductive(e,r)）
2. ✅ General领域（或Biology领域）
3. ✅ FB15K、WordNet、YAGO系列

**典型例子**:
- FB15K237Inductive系列 ✅
- WN18RRInductive:v3 ✅
- Metafam（Biology + Inductive）✅

---

### ⚠️ 中等适用场景（可能提升或持平）

**特征**:
1. ⚠️ Transductive设置
2. ⚠️ General领域
3. ⚠️ 结构化数据集（YAGO、NELL995）

**典型例子**:
- YAGO310-ht ✅（提升）
- NELL995-ht ✅（提升）
- 但其他Transductive数据集可能持平

---

### ❌ 不适用场景（预期下降）

**特征**:
1. ❌ Common Sense领域（ConceptNet）
2. ❌ Domain Specific领域（WikiTopics）
3. ❌ 已表现很好（>0.7）且关系非结构化
4. ❌ 低基础性能（<0.3）且非结构化

**典型例子**:
- ConceptNet 100k-ht ❌
- WikiTopicsMT3:infra ❌
- NELLInductive:v1（已表现很好）❌

---

## 实际应用建议

### 对于新数据集，使用以下检查清单：

1. **检查数据集类型**
   - [ ] 是Inductive设置？ → ✅ 很可能适用
   - [ ] 是Transductive设置？ → ⚠️ 需要进一步检查
   - [ ] 是Pre-training数据？ → ❌ 可能不适用

2. **检查领域类别**
   - [ ] 是General或Biology领域？ → ✅ 很可能适用
   - [ ] 是Common Sense领域？ → ❌ 不适用
   - [ ] 是Domain Specific领域？ → ❌ 不适用

3. **检查数据集家族**
   - [ ] 是FB15K、WordNet、YAGO系列？ → ✅ 很可能适用
   - [ ] 是ConceptNet、WikiTopics系列？ → ❌ 不适用

4. **检查基础性能**
   - [ ] 中等性能（0.3-0.5）？ → ✅ 很可能适用
   - [ ] 低性能（<0.3）且非结构化？ → ❌ 不适用
   - [ ] 高性能（>0.7）？ → ⚠️ 可能不需要增强

### 决策树

```
新数据集
├─ Inductive设置？
│  ├─ 是 → General/Biology领域？
│  │  ├─ 是 → FB15K/WordNet/YAGO系列？
│  │  │  ├─ 是 → ✅ 高度适用，使用ARE
│  │  │  └─ 否 → ⚠️ 中等适用，谨慎使用
│  │  └─ 否（Common Sense/Domain Specific）→ ❌ 不适用，禁用或降低阈值
│  └─ 否（Transductive）→ ⚠️ 需要进一步检查
└─ Pre-training数据？→ ❌ 可能不适用
```

---

## 总结

通过分析提升和下降数据集的共性特征，我们发现：

1. ✅ **Inductive设置 + General领域 + FB15K/WordNet/YAGO系列 = 高度适用**
2. ⚠️ **Transductive设置 + 结构化数据集 = 中等适用**
3. ❌ **Common Sense领域或Domain Specific领域 = 不适用**
4. ❌ **已表现很好（>0.7）且非结构化 = 不适用**

这些发现为ARE模型的实际应用提供了明确的指导原则。

