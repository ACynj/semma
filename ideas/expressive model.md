# 1 引言（Introduction）
知识图谱（KG）推理旨在从现有关系事实中推断新的关系事实。早期研究主要聚焦于** transduction （转导）设置**下的静态KG推理，但缺乏处理KG中新实体或关系的泛化能力。近年研究（[1-4]）通过考虑已见与未见实体间的关系模式实现了** inductive（归纳）推理**，但由于预训练KG与未见KG间的实体、关系词汇不共享且无关联，这些方法仍无法迁移到未见KG上进行推理。

现有方法在泛化到新实体、新关系甚至不同KG时，核心挑战在于**未见数据的表示问题**：
- 部分方法（[1-4]）通过聚合查询条件的关系结构表示实体，可实现未见实体的归纳推理，但无法处理未见关系；
- 后续方法（[5,6]）通过查询条件的关系图（节点为关系，边表示关系共享实体）建模关系交互，但该关系图仅描述KG中关系的连通性，忽略了查询中实体与关系的**局部上下文**，易引入噪声（如推断“parentOf”时，“teach”关系可能作为噪声干扰模型）。

为此，本文提出基于**上下文学习（in-context learning）** 的知识图谱推理基础模型** KG-ICL **。上下文学习使预训练模型无需更新参数，仅通过少量示例即可学习任务，其成功依赖三大核心：提示设计、统一分词、上下文理解与利用。KG-ICL的核心设计包括：
1. **提示图（Prompt Graph）**：以查询关系的示例事实（subject, query relation, object）为中心，融合实体上下文（示例实体的邻域）与关系上下文（示例实体间的关系路径）构建子图；
2. **统一分词器（Unified Tokenizer）**：将不同KG的实体（按到示例实体的最短路径长度分组）与关系（按是否为查询关系分组）映射到预定义令牌，实现跨KG的“统一语言”；
3. **双消息传递网络**：分别用于提示图编码（获取提示表示）与KG推理（结合提示初始化KG表示并完成推理）。

本文的主要贡献如下：
- 提出首个基于上下文学习的KG推理基础模型，支持跨多样KG的关系推理；
- 设计提示图作为上下文支持上下文学习，并提出统一分词器映射实体与关系到预定义令牌；
- 提出两个消息传递网络分别用于提示图编码与KG推理，模型可在特定KG上微调以提升性能；
- 在43个KG的转导与归纳设置下验证，证明模型的通用推理能力。


# 2 相关工作（Related Work）
## 2.1 知识图谱推理（KG Reasoning）
KG推理主要分为三种设置，现有工作存在明显局限：
- **转导设置（Transductive）**：早期研究（[10-14]）假设KG静态，为每个实体/关系学习独立嵌入，无法处理新实体；
- **归纳设置（Inductive）**：针对动态KG，允许新实体出现（[1-4,15-23]），但仍局限于同一KG；
- **全归纳设置（Fully-inductive）**：允许查询中出现新实体与新关系（[5,24-26]），但同样未突破单一KG的限制。

本文的KG-ICL模型打破上述设置的壁垒，通过上下文学习实现跨KG的通用推理。

## 2.2 图预训练中的提示与上下文学习（Prompt and In-Context Learning in Graph Pre-training）
- **图预训练范式演变**：受NLP（[27]）和CV（[28]）预训练启发，早期图预训练模型（[29-33]）遵循“预训练-微调”范式；近年研究（[8,35-45]）转向“预训练-提示-微调”范式，利用任务提示增强知识迁移能力（如[6]的关系图可视为特殊提示）。
- **上下文学习的局限**：受GPT（[7]）启发，部分工作（如PRODIGY [46]）通过少量示例实现图分类任务的上下文学习，但PRODIGY仅适用于关系分类，无法处理候选实体数量庞大的KG推理任务。


# 3 问题定义（Problem Definition）
## 3.1 知识图谱推理（KG Reasoning）
定义知识图谱为 \( K=(E, R, T) \)，其中：
- \( E \)：实体集合，\( R \)：关系集合，\( T \)：事实集合；
- 事实 \( (s, r, o) \in T \) 由主语实体 \( s \in E \)、关系 \( r \in R \)、宾语实体 \( o \in E \) 组成。

推理任务：给定KG和查询事实 \( (s, q, ?) \)（\( q \in R \) 为**查询关系**），从 \( E \) 中预测缺失的宾语实体。为简化推理，本文引入**逆关系**：对每个 \( r \in R \)，添加逆关系 \( r^- \in R \) 及反向事实 \( (o, r^-, s) \in T \)。

## 3.2 上下文知识图谱推理（In-Context KG Reasoning）
- **预训练阶段**：使用一组源KG \( \{K_1, ..., K_n\} \) 预训练模型；
- **推理阶段**：在新出现的KG上推理时，仅需提供少量与查询相关的示例作为上下文，**无需更新模型参数**。

### 3.2.1 提示图（Prompt Graph）
给定KG \( K=(E,R,T) \) 中的示例事实 \( c=(u, q, v) \in T \)（\( q \) 为查询关系），提示图定义为 \( K \) 的子图 \( P_c=(E_{pmt} \subseteq E, R_{pmt} \subseteq R, T_{pmt} \subseteq T) \)，用于捕捉查询相关的局部上下文。

### 3.2.2 统一分词器（Unified Tokenizer）
为实现跨KG的表示统一，设计**多对一映射函数**，将不同提示图的实体与关系映射到预定义令牌：
- **实体分词**：基于实体到示例事实中主语 \( u \) 和宾语 \( v \) 的最短路径长度，即 \( \text{tokenize}(e) \leftarrow [\text{dist}(u,e), \text{dist}(v,e)] \)，其中 \( \text{dist}(\cdot) \) 表示两实体间最短路径长度；
- **关系分词**：基于关系是否与查询关系 \( q \) 一致，即 \( \text{tokenize}(r) \leftarrow [\text{same}(r,q)] \)，其中 \( \text{same}(r,q)=1 \)（若 \( r=q \)），否则为0。

每个令牌分配可学习的向量表示（见4.2节）。


# 4 知识图谱的上下文推理（In-context Reasoning over KGs）
KG-ICL的整体流程分为三步：生成查询关系的提示图 → 编码提示图并提取提示表示 → 将提示融入KG推理过程（如图1所示）。

## 4.1 提示图生成（Prompt Graph Generation）
针对“如何使提示图跨KG通用”与“如何提供有效推理信息”两大挑战，设计两阶段生成流程：

### 4.1.1 示例采样（Example Sampling）
对查询关系 \( q \)，随机采样 \( M \) 个示例事实：
\[
\mathcal{S}_q = \{c_i\}_{i=1}^M, \quad c_i=(u, q, v) \sim \text{Uniform}(\mathcal{N}_q)
\]
其中 \( \mathcal{N}_q \) 表示所有包含 \( q \) 的事实集合，\( c_i \) 为 \( q \) 专属的示例事实。

### 4.1.2 提示图提取（Prompt Graph Extraction）
提示图需突出查询关系推理的关键信息，因此包含两类实体：
1. 示例事实中主语 \( u \) 和宾语 \( v \) 的1跳邻域实体；
2. \( u \) 与 \( v \) 间的 \( k \)-跳路径上的实体（\( k \) 为超参数，控制路径最大长度）。

实体集合 \( E_{pmt} \) 定义为：
\[
\mathcal{E}_{pmt} = \{x | \exists(x,r,u) \in \mathcal{T}\} \cup \{x | \exists(x,r,v) \in \mathcal{T}\} \cup \{x | \text{dist}(x,u)+\text{dist}(x,v) \leq k\}
\]
进一步提取 \( E_{pmt} \) 间的事实与关系，即 \( T_{pmt} = \{(s,r,o) | s,o \in E_{pmt}, (s,r,o) \in T\} \)，\( R_{pmt} = \{r | \exists(s,r,o) \in T_{pmt}\} \)，最终构成提示图 \( P_c=(E_{pmt}, R_{pmt}, T_{pmt}) \)。


## 4.2 提示编码（Prompt Encoding）
设计消息传递神经网络实现提示编码，包含**令牌表示、消息传递、读出**三个子模块：

### 4.2.1 令牌表示（Token Representations）
为每个预定义令牌分配可学习向量：
- **实体令牌**：根据最短路径长度的可能组合，实体令牌总数为 \( \frac{(k+1)(k+2)}{2} - 2(k-1) \)，对应表示矩阵 \( T \in \mathbb{R}^{(\frac{(k+1)(k+2)}{2}-2(k-1)) \times d} \)（\( d \) 为嵌入维度）；
- **关系令牌**：令牌 \( [z] \)（\( z \in \{0,1\} \)）的表示初始化为 \( q^{\text{token}} \cdot z \)，其中 \( q^{\text{token}} \) 为可学习向量。

提示图的实体与关系输入表示矩阵分别记为 \( H_E^{(0)} \) 和 \( H_R^{(0)} \)。

### 4.2.2 提示图的消息传递（Message Passing for Prompt Graph）
采用 \( L \)-层消息传递网络，每一层包含**实体中心聚合**与**关系中心聚合**：
1. **实体中心聚合**：更新实体表示（基于包含该实体的事实）：
\[
H_E^{(l+1)} \leftarrow \underset{\forall e \in \mathcal{E}_{pmt}, \forall n \in \mathcal{N}_e}{\text{Aggregation}_E} \left( \left\{ \text{Message}(H_E^{(l)}, H_R^{(l)}, n, q) \right\} \right)
\]
其中 \( \mathcal{N}_e \subseteq T_{pmt} \) 为包含实体 \( e \) 的事实集合，\( q \) 为查询关系。

2. **关系中心聚合**：基于更新后的实体表示更新关系表示（基于包含该关系的事实）：
\[
H_R^{(l+1)} \leftarrow \underset{\forall r \in \mathcal{R}_{pmt}, \forall n \in \mathcal{N}_r}{\text{Aggregation}_R} \left( \left\{ \text{Message}(H_E^{(l+1)}, H_R^{(l)}, n, q) \right\} \right)
\]
其中 \( \mathcal{N}_r \subseteq T_{pmt} \) 为包含关系 \( r \) 的事实集合。

聚合过程中融入**查询感知注意力**（Query-aware Attention）与**残差连接**（Residual Connection）、**层归一化**（Layer Normalization）以提升学习稳定性。

### 4.2.3 读出（Readout）
- 单提示图的提示表示：对 \( L \)-层后的关系表示拼接并线性变换：
\[
H_{\mathcal{P}} = W_{\text{Readout}} \left( H_R^{(1)} \| H_R^{(2)} \| \cdots \| H_R^{(L)} \right)
\]
其中 \( W_{\text{Readout}} \in \mathbb{R}^{d \times Ld} \) 为可学习矩阵，“\( \| \)”表示拼接。对提示图中未出现的关系，用零向量填充得到 \( \hat{H}_{\mathcal{P}} \in \mathbb{R}^{|R| \times d} \)。

- 多提示图的信息聚合：对 \( M \) 个提示图的表示进行平均池化：
\[
\overline{H}_{pmt} = \frac{1}{|\mathcal{S}_q|} \sum_{c \in \mathcal{S}_q} \hat{H}_{\mathcal{P}_c}
\]
其中 \( \overline{H}_{pmt} \in \mathbb{R}^{|R| \times d} \) 为最终的**提示关系表示矩阵**，\( \mathcal{S}_q \) 为 \( q \) 的示例事实集合。


## 4.3 上下文KG编码与推理（In-Context KG Encoding and Reasoning）
基于提示表示实现KG推理，模块包含**初始化、KG编码、推理**：

### 4.3.1 初始化（Initialization）
- **关系表示初始化**：KG中关系的初始表示直接采用提示关系表示，即 \( V_R^{(0)} = \overline{H}_{pmt} \)；
- **实体表示初始化**：查询事实 \( (s, q, ?) \) 中，主语实体 \( s \) 的初始表示设为查询关系 \( q \) 的表示（\( s = q \)），其他实体初始化为零向量，实体输入表示矩阵记为 \( V_E^{(0)} \)。

### 4.3.2 KG的消息传递（Message Passing for KG）
采用 \( N \)-层消息传递网络聚合多跳信息，融入**距离感知归纳偏置**（Distance-based Inductive Bias）：
1. **关系表示更新**：
\[
V_R^{(l+1)} = \text{LN} \left( V_R^{(l)} + W_R^{(l)} V_R^{(l)} \right)
\]
其中 \( \text{LN} \) 为层归一化，\( W_R^{(l)} \in \mathbb{R}^{d \times d} \) 为可学习矩阵。

2. **实体表示更新**：从主语实体开始，逐层扩展1跳邻域并更新（\( \mathcal{L}^{(l)} \) 为 \( s \) 的 \( l \)-跳邻域实体集合）：
\[
V_E^{(l+1)} \leftarrow \underset{\forall e \in \mathcal{L}^{(l+1)}, \forall n \in \mathcal{N}_e}{\text{Aggregation}_E} \left( \left\{ \text{Message}(V_E^{(l)}, V_R^{(l+1)}, n, q) \right\} \right)
\]
其中 \( \mathcal{L}^{(0)} = \{s\} \)，\( \mathcal{L}^{(l+1)} = \mathcal{L}^{(l)} \cup \{e | \exists(x,y,e) \in T, x \in \mathcal{L}^{(l)}\} \)。

### 4.3.3 推理（Reasoning）
对候选实体的最终表示打分，预测缺失实体：
\[
f(s, q, e) = W_{\text{score}} \cdot e_{s,q}^{(N)}
\]
其中 \( e_{s,q}^{(N)} \in V_E^{(N)} \) 为实体 \( e \) 的 \( N \)-层输出表示，\( W_{\text{score}} \in \mathbb{R}^{1 \times d} \) 为可学习矩阵。仅对主语 \( s \) 的 \( N \)-跳邻域实体打分，其余实体得分为0。


## 4.4 预训练目标（Pre-training Objective）
使用**多分类对数损失**（Multi-class Log-loss）预训练模型，最小化正事实得分与负事实得分的差距：
\[
\mathcal{L} = -\frac{1}{|\mathcal{T}_{\text{train}}|} \sum_{(s,r,o) \in \mathcal{T}_{\text{train}}} \log \frac{\exp(f(s,r,o))}{\sum_{e \in E \setminus \{o\}} \exp(f(s,r,e))}
\]
其中 \( \mathcal{T}_{\text{train}} \) 为源KG的训练事实集合，\( f(\cdot) \) 为4.3.3节的得分函数。