# How Expressive are Knowledge Graph Foundation Models?
Xingyue Huang ¹ Pablo Barceló ² ³ ⁴ Michael M. Bronstein ¹ ⁵ İsmail İlkan Ceylan ¹ Mikhail Galkin ⁶ Juan L Reutter ² ³ ⁴ Miguel Romero Orth ² ⁴  
¹University of Oxford ²Universidad Católica de Chile ³IMFD ⁴CENIA ⁵AITHYRA ⁶Google. Correspondence to: Xingyue Huang <University of Oxford>.  
Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).  
arXiv:2502.13339v2 [cs.LG] 9 Jun 2025


## Abstract
Knowledge Graph Foundation Models (KGFMs) are at the frontier for deep learning on knowledge graphs (KGs), as they can generalize to completely novel knowledge graphs with different relational vocabularies. Despite their empirical success, our theoretical understanding of KGFMs remains very limited. In this paper, we conduct a rigorous study of the expressive power of KGFMs. Specifically, we show that the expressive power of KGFMs directly depends on the motifs that are used to learn the relation representations. We then observe that the most typical motifs used in the existing literature are binary, as the representations are learned based on how pairs of relations interact, which limits the model’s expressiveness. As part of our study, we design more expressive KGFMs using richer motifs, which necessitate learning relation representations based on, e.g., how triples of relations interact with each other. Finally, we empirically validate our theoretical findings, showing that the use of richer motifs results in better performance on a wide range of datasets drawn from different domains.


## 1. Introduction
Knowledge Graph Foundation Models (KGFMs) gained significant attention (Galkin et al., 2024; Mao et al., 2024) for their success in link prediction tasks involving unseen entities and relations on knowledge graphs (KGs). These models aim to generalize across different KGs by effectively learning **relation invariants**: properties of relations that are transferable across KGs of different relational vocabularies (Gao et al., 2023). KGFMs learn representations of relations by relying on their structural roles in the KG, which can be transferred to novel relations based on their “structurally similarities” to the observed relations.

### Example 1.1
Consider the two KGs from Figure 1 which use disjoint sets of relations. Suppose that the model is trained on \(G_{train}\) and the goal is to predict the missing link \(produce(Intel, SemiConductors)\) in \(G_{test}\). Ideally, the model will predict the link by learning relation invariants that map \(produce \leftrightarrow research\) and \(supply \leftrightarrow provide\), as the structural role of these relations are similar in the respective graphs, even if the relations themselves differ. △

![Figure 1. Training is over relations provide and research, and testing is over structurally similar relations supply and produce.](https://example.com/figure1)  
*Figure 1. Training is over relations provide and research, and testing is over structurally similar relations supply and produce.*

The dominant approach (Geng et al., 2023; Lee et al., 2023; Galkin et al., 2024) for computing relation invariants is by learning relational embeddings over a **graph of relations**. In this graph, each node represents a relation type from the original KG and each edge represents a connection between two relations. One prominent choice of these connections is to identify whether two relations in the original KG match a set of shared patterns known as **graph motifs**: small, recurring subgraphs that capture essential relational structures within the KG. Figure 2 depicts a simple motif: a path of length two, connecting three entities via two relations (i.e., a binary motif). For instance, the relations \(provide\) and \(research\) are connected via the entity \(Oxford\) in Figure 1, and hence match this motif with the map \(\alpha \mapsto provide, \beta \mapsto research\).

![Figure 2. A simple motif of a path of length two.](https://example.com/figure2)  
*Figure 2. A simple motif of a path of length two.*

We saliently argue that existing models limit themselves to **binary motifs**, which restricts their ability to capture complex interactions involving more relations. We investigate the expressive power of these models, specifically, how the choice of motifs impacts the model’s ability to capture relation invariants and help distinguish between different relational structures. Prior works have extensively studied relational message-passing neural networks on KGs (Barceló et al., 2022; Huang et al., 2023), but these results do not apply to KGFMs, which is the focus of our work.

### Approach
We introduce **MOTif-Induced Framework for KGFMs (MOTIF)** based on learning relation invariants from arbitrary (not necessarily binary) graph motifs. MOTIF first constructs a **relational hypergraph**, where each node corresponds to a relation in the original KG, and each hyperedge represents a match of relations to a motif in the original KG. Then, it performs message passing over the relational hypergraph to generate relation representations, which are used in message passing over the original KG to enable link prediction over novel relations. Importantly, MOTIF is a general framework that strictly subsumes existing KGFMs such as ULTRA (Galkin et al., 2024) and INGRAM (Lee et al., 2023) as detailed in Section 5, and provides a principled approach for enhancing the expressive power of other KGFMs such as TRIX (Zhang et al., 2024) and RMPI (Geng et al., 2023).

### Contributions
Our contributions can be summarized as follows:
- We introduce MOTIF as a general framework capable of integrating arbitrary graph motifs into KGFMs, subsuming existing KGFMs such as ULTRA and INGRAM.
- To the best of our knowledge, we provide the first rigorous analysis on the expressive power of KGFMs. Our study explains the capabilities and limitations of existing KGFMs in capturing relational invariants.
- We identify a sufficient condition under which a newly introduced motif enhances the expressiveness of a KGFM within the framework of MOTIF. Using this theoretical recipe, we derive a new KGFM that is provably more expressive than ULTRA.
- Empirically, we conduct an extensive analysis on 54 KGs from diverse domains and validate our theoretical results through synthetic and real-world experiments.

All proofs of the technical results can be found in the appendix of the paper, along with the additional experiments.


## 2. Related work
### Transductive and inductive (on node) link prediction
Link prediction on KGs has been extensively studied in the literature. Early approaches like TransE (Bordes et al., 2013), RotatE (Sun et al., 2019), and BoxE (Abboud et al., 2020) focus on the **transductive setting**, where the learned entity embeddings are fixed, and thus inapplicable to unseen entities at test time. Multi-relational GNNs such as RGCN (Schlichtkrull et al., 2018) and CompGCN (Vashishth et al., 2020) remain transductive as they store entity and relation embeddings as parameters.

To overcome this limitation, Teru et al. (2020) introduce GraIL, which enables inductive link prediction via the labeling trick. NBFNet (Zhu et al., 2021), A*Net (Zhu et al., 2023), RED-GNN (Zhang & Yao, 2022), and AdaProp (Zhang et al., 2023) provide improvements by leveraging conditional message-passing, which is provably more expressive (Huang et al., 2023). These models, once trained, can only be applied to KGs with the same relational vocabulary, limiting their applicability to graphs with unseen relations.

### Inductive (on node and relation) link prediction
INGRAM (Lee et al., 2023) was one of the first approaches to study inductive link prediction over both new nodes and unseen relations by constructing a weighted relation graph to learn new relation representations. Galkin et al. (2024) extended this idea with the ULTRA architecture, which constructs a multi-relational graph of fundamental relations and leverages conditional message passing to enhance performance. ULTRA was among the first KGFMs to inspire an entire field of research (Mao et al., 2024).

Concurrently, RMPI (Geng et al., 2023) explored generating multirelational graphs through local subgraph extraction while also incorporating ontological schema. Gao et al. (2023) introduced the concept of double-equivariant GNNs, which establish invariants on nodes and relations by leveraging subgraph GNNs in the proposed ISDEA framework to enforce double equivariance precisely. MTDEA (Zhou et al., 2023a) expands this framework with an adaptation procedure for multi-task generalization. Further, TRIX (Zhang et al., 2024) expands on ULTRA with recursive updates of relation and entity embeddings, and is the first work to compare expressivity among KGFMs. Finally, KG-ICL (Cui et al., 2024) introduced a new KGFM utilizing in-context learning with a unified tokenizer for entities and relations.

### Link prediction on relational hypergraphs
Relational hypergraphs are a generalization of KGs used to represent higher-arity relational data. Work on link prediction in relational hypergraphs first focused on shallow embeddings (Wen et al., 2016; Liu et al., 2020; Fatemi et al., 2020), and later G-MPNN (Yadati, 2020) and RD-MPNNs (Zhou et al., 2023b) advanced by incorporating message passing. Recently, Huang et al. (2024) conducted an in-depth expressivity study on these models and proposed HCNets, extending conditional message-passing to relational hypergraphs and achieving strong results on inductive link prediction.


## 3. Preliminaries
### Knowledge graphs
A knowledge graph (KG) is a tuple \(G=(V, E, R)\), where:
- \(V\) is a set of nodes (entities),
- \(R\) is the set of relation types,
- \(E \subseteq V \times R \times V\) is a set of labeled edges (facts).

We write \(r(u, v)\) to denote a labeled edge (fact) where \(r \in R\) and \(u, v \in V\). A **potential link** in \(G\) is a triple \(r(u, v) \in V \times R \times V\) that may or may not be a fact in \(G\). The neighborhood of node \(v \in V\) relative to a relation \(r \in R\) is \(N_{r}(v) := \{u \mid r(u, v) \in E\}\).

### Relational hypergraphs
A relational hypergraph is a tuple \(G=(V, E, R)\), where:
- \(V\) is a set of nodes,
- \(R\) is a set of relation types,
- \(E\) is a set of hyperedges (facts) of the form \(e=r(u_1, \dots, u_k) \in E\) (where \(r \in R\), \(u_1, \dots, u_k \in V\), and \(k=ar(r)\) is the **arity** of relation \(r\)).

We write \(\rho(e)\) as the relation type \(r \in R\) of hyperedge \(e \in E\), and \(e(i)\) to refer to the node in the \(i\)-th arity position of \(e\).

### Homomorphisms
A **(node-relation) homomorphism** from a KG \(G=(V, E, R)\) to a KG \(G'=(V', E', R')\) is a pair of mappings \(h=(\pi, \phi)\), where \(\pi: V \to V'\) (node mapping) and \(\phi: R \to R'\) (relation mapping), such that for every fact \(r(u, v) \in G\), the fact \(\phi(r)(\pi(u), \pi(v)) \in G'\). The image \(h(G)\) of \(h\) is the KG \((\pi(V), h(E), \phi(R))\), where \(h(E) = \{\phi(r)(\pi(u), \pi(v)) \mid r(u, v) \in E\}\).

### Isomorphism, relation invariants, link invariants
An **isomorphism** from KG \(G=(V, E, R)\) to KG \(G'=(V', E', R')\) is a pair of bijections \((\pi, \phi)\) (node and relation bijections) such that \(r(u, v) \in G \iff \phi(r)(\pi(u), \pi(v)) \in G'\). Two graphs are isomorphic if there exists an isomorphism between them.

- For \(k \geq 1\), a **\(k\)-ary relation invariant** is a function \(\xi\) associating to each KG \(G\) a function \(\xi(G)\) with domain \(R^k\), such that for isomorphic \(G, G'\), every isomorphism \((\pi, \phi)\), and every \(\bar{r} \in R^k\), \(\xi(G)(\bar{r}) = \xi(G')(\phi(\bar{r}))\).
- A **link invariant** is a function \(\omega\) associating to each KG \(G\) a function \(\omega(G)\) with domain \(V \times R \times V\), such that for isomorphic \(G, G'\), every isomorphism \((\pi, \phi)\), and every link \(q(u, v) \in G\), \(\omega(G)(q(u, v)) = \omega(G')(\phi(q)(\pi(u), \pi(v)))\).

For instance, a unary relation invariant \(\xi\) evaluated on \(G_{train}\) and \(G_{test}\) (Figure 1) may satisfy \(\xi(G_{train})(research) = \xi(G_{test})(produce)\) if the two relations play analogous structural roles. Similarly, a link invariant \(\omega\) may assign equal values to \(research(Oxford, Finance)\) (in \(G_{train}\)) and \(produce(Intel, SemiConductors)\) (in \(G_{test}\)) under a structure-preserving mapping.


## 4. MOTIF: A general framework for KGFMs
We present MOTIF, a general framework for KGFMs. Given a KG \(G=(V, E, R)\), MOTIF computes an encoding for each potential link \(q(u, v) \in G\) via three steps:

1. **LIFT**: Use a set \(F\) of motifs to compute a relational hypergraph \(LIFT_F(G)=(V_{LIFT}, E_{LIFT}, R_{LIFT})\).
2. **RELATION ENCODER**: Apply a relation encoder on \(LIFT_F(G)\) to obtain relation representations.
3. **ENTITY ENCODER**: Use relation representations (Step 2) and apply an entity encoder on \(G\) to obtain final link encodings.

The overall process is illustrated in Figure 3. We detail each step below.

![Figure 3. Overall framework of MOTIF(F) over the motif set F: Given a query \(r_1(u, v)\), MOTIF(F) first applies the LIFT_F operation to generate the relational hypergraph \((V_{LIFT}, E_{LIFT}, R_{LIFT})\) over F. Then a relation encoder generates a relation representation (color-coded) conditioned on query \(r_1\), followed by an entity encoder that conducts conditional message passing.](https://example.com/figure3)  
*Figure 3. Overall framework of MOTIF(F) over the motif set F: Given a query \(r_1(u, v)\), MOTIF(F) first applies the LIFT_F operation to generate the relational hypergraph \((V_{LIFT}, E_{LIFT}, R_{LIFT})\) over F. Then a relation encoder generates a relation representation (color-coded) conditioned on query \(r_1\), followed by an entity encoder that conducts conditional message passing.*

### 4.1. LIFT
A **graph motif** is a pair \(P=(G_M, \bar{r})\), where:
- \(G_M=(V_M, E_M, R_M)\) is a connected KG,
- \(\bar{r}\) is a tuple defining an order on \(R_M\).

A **\(k\)-ary motif** has \(|R_M|=k\) (exactly \(k\) relation types in the motif graph). For \(k=2\), it is a **binary motif**. The information extracted by a motif is defined via homomorphism:

#### Definition 4.1
Let \(G\) be a KG and \(P=(G_M, \bar{r})\) be a motif with \(\bar{r}=(r_1, \dots, r_n)\). The **evaluation** \(Eval(P, G)\) of \(P\) over \(G\) is the set of tuples \((\phi(r_1), \dots, \phi(r_n))\) for each homomorphism \(h=(\pi, \phi)\) from \(G_M\) to \(G\). □

#### The LIFT operation
Given a KG \(G=(V, E, R)\) and a set of motifs \(F\), \(LIFT_F(G)\) constructs a relational hypergraph \((V_{LIFT}, E_{LIFT}, R_{LIFT})\) as follows:
- **Node set**: \(V_{LIFT}=R\) (each node corresponds to a relation type from \(G\)).
- **Relation set**: \(R_{LIFT}=F\) (each motif \(P \in F\) is a distinct relation type in the hypergraph).
- **Hyperedge set**: 
  \[
  E_{LIFT} = \left\{ P(r_1, \dots, r_k) \mid P \in F, (r_1, \dots, r_k) \in Eval(P, G) \right\}
  \]
  where \(Eval(P, G)\) is the set of all \(k\)-tuples of relations in \(G\) that match motif \(P\).

### 4.2. Relation encoder
A relation encoder is a tuple \((F, Enc_{LIFT})\), where \(F\) is a set of motifs and \(Enc_{LIFT}\) computes relation representations for \(R=V_{LIFT}\) (nodes of \(LIFT_F(G)\)). Since we aim to encode links \(q(u, v)\), the encoding of relation \(r \in R\) is **conditioned on query relation \(q \in R\)**.

The encoder defines a sequence of features \(h_{r|q}^{(t)} \in \mathbb{R}^{d(t)}\) (for \(0 \leq t \leq T\)) iteratively:
\[
h_{r|q}^{(0)} = INIT_1(q, r), \quad h_{r|q}^{(t+1)} = UP_1\left(h_{r|q}^{(t)}, AGG_1(M)\right)
\]
where:
- \(INIT_1\), \(UP_1\), \(AGG_1\) are differentiable initialization, update, and aggregation functions,
- \(M = \left\{ MSG_{\rho(e)}\left( \left\{ (h_{r'|q}^{(t)}, j) \mid (r', j) \in \mathcal{N}^i(e) \right\} \right) \mid (e, i) \in E_{LIFT}(r) \right\}\),
- \(MSG_{\rho(e)}\) is a motif-specific message function,
- \(E_{LIFT}(r) = \{ (e, i) \mid e(i)=r, e \in E_{LIFT}, 1 \leq i \leq ar(\rho(e)) \}\) (edge-position pairs of node \(r\)),
- \(\mathcal{N}^i(e) = \{ (e(j), j) \mid j \neq i, 1 \leq j \leq ar(\rho(e)) \}\) (positional neighborhood of hyperedge \(e\) at position \(i\)).

We denote \(Enc_{LIFT, q}[r] = h_{r|q}^{(T)}\) (final relation encoding of \(r\) conditioned on \(q\)).

#### Remark 1
The relation encoder computes binary relation invariants if \(INIT_1\) is an invariant.

### 4.3. Entity encoder
MOTIF uses an entity encoder to compute link-level representations of \(G\) (regardless of relational vocabulary). The encoder is defined as a tuple \((Enc_{KG}, (F, Enc_{LIFT}))\), where \(Enc_{KG}\) computes node representations over \(G=(V, E, R)\) **conditioned on source node \(u \in V\) and query relation \(q\)**.

It defines a sequence of features \(h_{v|u,q}^{(\ell)} \in \mathbb{R}^{d(\ell)}\) (for \(0 \leq \ell \leq L\)) iteratively:
\[
h_{v|u,q}^{(0)} = INIT_2(u, v, q), \quad h_{v|u,q}^{(\ell+1)} = UP_2\left(h_{v|u,q}^{(\ell)}, AGG_2(N)\right)
\]
where:
- \(INIT_2\), \(UP_2\), \(AGG_2\), \(MSG\) are differentiable initialization, update, aggregation, and message functions,
- \(v\) is an arbitrary node in \(V\),
- \(N = \left\{ MSG\left(h_{w|u,q}^{(\ell)}, Enc_{LIFT, q}[r]\right) \mid w \in \mathcal{N}_r(v), r \in R \right\}\).

We denote \(Enc_{KG, u, q}[v] = h_{v|u,q}^{(L)}\) (encoding of \(v\) conditioned on \(u\) and \(q\)). Finally, a unary decoder \(Dec: \mathbb{R}^{d(L)} \mapsto [0,1]\) maps \(Enc_{KG, u, q}[v]\) to the probability score of link \(q(u, v)\).

#### Remark 2
The entity encoder computes link invariants if \(INIT_2\) is an invariant.


## 5. KGFMs captured by MOTIF
We revisit existing KGFMs and show how they fit into the MOTIF framework.

### ULTRA (Galkin et al., 2024)
The four fundamental relations in ULTRA (Figure 4) can be interpreted as graph motifs. Thus, any ULTRA architecture is a MOTIF instance, with \(Enc_{LIFT}\) and \(Enc_{KG}\) being specific variants of NBFNets (Zhu et al., 2021).

![Figure 4. The 4 motifs used by ULTRA: \(h2t\) (head-to-tail), \(t2h\) (tail-to-head), \(t2t\) (tail-to-tail), \(h2h\) (head-to-head). Note \(h2t\) and \(t2h\) are isomorphic but with different relation ordering.](https://example.com/figure4)  
*Figure 4. The 4 motifs used by ULTRA: \(h2t\) (head-to-tail), \(t2h\) (tail-to-head), \(t2t\) (tail-to-tail), \(h2h\) (head-to-head). Note \(h2t\) and \(t2h\) are isomorphic but with different relation ordering.*

### INGRAM (Lee et al., 2023)
A slight variant of INGRAM is captured by MOTIF when replacing its weighted relation graph with a KG. Here, \(LIFT\) uses motif set \(F=\{h2h, t2t\}\), \(Enc_{LIFT}\) and \(Enc_{KG}\) are variants of GATv2 (Brody et al., 2022), and \(INIT_1\) is an invariant function (replacing random initialization).

### Other KGFMs
- **RMPI (Geng et al., 2023)**: Constructs a relation graph inspired by line graphs, using ULTRA’s motifs plus two additional motifs (PARA, LOOP). Though not strictly in MOTIF, it relies on a LIFT-like operation.
- **TRIX (Zhang et al., 2024)**: Uses ULTRA’s motifs but adds alternating updates of entity and relation embeddings. See Appendix B for a detailed expressivity comparison with MOTIF.

MOTIF also supports **unconditional message passing** (Huang et al., 2023) when \(INIT_1\) and \(INIT_2\) are agnostic to query \(q\).


## 6. Expressive power
MOTIF encodes links in KGs; we aim to understand: (1) which links MOTIF can separate, and (2) whether richer motifs increase separation power. We denote \(MOTIF(F)\) as the set of all MOTIF instances using motif set \(F\) (i.e., instances of the form \((Enc_{KG}, (F, Enc_{LIFT}))\)).

### Definition 6.1
Let \(F\) and \(F'\) be sets of graph motifs.
- \(F\) **refines** \(F'\) (denoted \(F \preceq F'\)) if: for every KG \(G\) and links \(q(u, v), q'(u', v') \in G\), if all \(MOTIF(F)\) instances encode \(q(u, v)\) and \(q'(u', v')\) identically, then all \(MOTIF(F')\) instances also encode them identically.
- \(F\) and \(F'\) have **same separation power** (denoted \(F \sim F'\)) if \(F \preceq F'\) and \(F' \preceq F\). □

This definition captures the ability to distinguish links: \(F \preceq F'\) means \(F'\) cannot separate links that \(F\) cannot; \(F \sim F'\) means \(F\) and \(F'\) are interchangeable for separation.

### Proposition 6.2
Let \(P=(G_M, \bar{r})\) and \(P'=(G_M', \bar{r}')\) be isomorphic motifs ( \(G_M \cong G_M'\) ). For any motif set \(F\):
\[
(F \cup \{P\}) \sim (F \cup \{P'\}) \sim (F \cup \{P, P'\})
\]
*Proof: See Appendix D.1.*

For example, ULTRA’s \(h2t\) and \(t2h\) are isomorphic (Figure 4), so \(F=\{h2t, t2h, h2h, t2t\} \sim \{h2t, h2h, t2t\} \sim \{t2h, h2h, t2t\}\). However, ULTRA cannot drop \(h2t\) or \(t2h\) directly: ULTRA uses NBFNets (Zhu et al., 2021) for relation encoding, while MOTIF uses HCNets (Huang et al., 2024) (which natively support inverse relations; see Appendices A and F).

### 6.1. When a new motif enhances separation?
We first introduce key terminology:
- A homomorphism \(h=(\pi, \phi)\) from \(G\) to \(G'\) is **relation-preserving** if \(R \subseteq R'\) and \(\phi(R)=R\).
- A KG \(H\) is a **relation-preserving core** if every relation-preserving homomorphism from \(H\) to \(H\) is onto ( \(h(H)=H\) ).
- A homomorphism \(h\) from \(G\) to \(G'\) is **core-onto** if \(h(G) \cong\) relation-preserving core of \(G'\). We write \(G \to^{co} G'\) if such a homomorphism exists.
- A motif is **trivial** if its relation-preserving core is isomorphic to a single fact \(r(u, v)\).

#### Proposition 6.3
For every KG \(G\), there exists a unique KG \(H\) (up to isomorphism) such that:
1. \(H\) is a relation-preserving core,
2. There exist relation-preserving homomorphisms from \(G\) to \(H\) and from \(H\) to \(G\).

\(H\) is called the **relation-preserving core** of \(G\). *Proof: See Appendix D.2.*

#### Theorem 6.4
Let \(F, F'\) be motif sets with \(F \preceq F'\). For every non-trivial \(P' \in F'\), there exists \(P \in F\) such that \(P \to^{co} P'\).  
*Proof: See Appendix D.3.*

This theorem provides a necessary condition for refinement: if \(F'\) has a non-trivial motif not covered by any \(P \in F\) (via \(P \to^{co} P'\)), then \(F \npreceq F'\) (i.e., \(F'\) has stronger separation power).

### 6.2. Consequences for MOTIF design
Theorem 6.4 implies: if a new motif \(P\) cannot be covered by any \(P' \in F\) (via \(P' \to^{co} P\)), then adding \(P\) to \(F\) enhances expressivity. We illustrate this with two motif classes:

#### 1. \(k\)-path motifs
A \(k\)-path motif is a path of length \(k\):
\[
r_1(u_1, u_2), r_2(u_2, u_3), \dots, r_k(u_k, u_{k+1})
\]
Let \(F_n^{path}\) be the set of all \(k\)-path motifs (for \(k \leq n\)) with arbitrary edge orientations (up to isomorphism). By Theorem 6.4:
\[
F_n^{path} \npreceq F_m^{path} \quad \text{if } n < m
\]
Thus, longer path motifs add separation power. Notably, ULTRA’s motif set is \(F=\{h2t, t2h, h2h, t2t\}\), which (by Proposition 6.2) has the same separation power as \(F_2^{path}=\{h2t, h2h, t2t\}\). This leads to:

#### Theorem 6.5
ULTRA has the same expressive power as \(MOTIF(F_2^{path})\).  
*Proof: See Appendix D.4.*

#### 2. \(k\)-star motifs
A \(k\)-star motif is a set of \(k\) relations sharing a common entity:
\[
r_1(v_1, u), r_2(v_2, u), \dots, r_k(v_k, u)
\]
Let \(F_n^{star}\) be the set of all \(k\)-star motifs (for \(k \leq n\)). By Theorem 6.4:
\[
F_n^{star} \npreceq F_m^{star} \quad \text{if } n < m
\]
Wider star motifs add separation power (Figure 5).

![Figure 5. The 4-star motif: relations \(\alpha, \beta, \gamma, \delta\) all point to a common entity \(u\).](https://example.com/figure5)  
*Figure 5. The 4-star motif: relations \(\alpha, \beta, \gamma, \delta\) all point to a common entity \(u\).*

#### A more expressive KGFM
Adding 3-path motifs (Figure 6) to ULTRA’s 4 motifs enhances expressivity. For example, consider the KG in Figure 7: ULTRA constructs a complete relation graph and cannot distinguish \(r_1\) and \(r_2\), but \(MOTIF(F_3^{path})\) constructs a hyperedge \((r_2, r_3, r_1)\) (not \((r_1, r_3, r_2)\)), thus separating \(r_1\) and \(r_2\). *See Appendix K for proof.*

![Figure 6. The 3-path motifs (\(F_3^{path}\)): tfh (tail-forward-head), tft (tail-forward-tail), hfh (head-forward-head), hft (head-forward-tail).](https://example.com/figure6)  
*Figure 6. The 3-path motifs (\(F_3^{path}\)): tfh (tail-forward-head), tft (tail-forward-tail), hfh (head-forward-head), hft (head-forward-tail).*

![Figure 7. ULTRA cannot distinguish \(r_3(u, v_1)\) and \(r_3(u, v_2)\), but \(MOTIF(F_3^{path})\) can.](https://example.com/figure7)  
*Figure 7. ULTRA cannot distinguish \(r_3(u, v_1)\) and \(r_3(u, v_2)\), but \(MOTIF(F_3^{path})\) can.*

This core idea (augmenting KGFMs with richer motifs) applies to INGRAM, RMPI, and TRIX, providing an orthogonal way to boost their expressivity.

### 6.3. Comparison with existing link prediction models
- **Inductive models (C-MPNNs, R-MPNNs)**: These are MOTIF instances with empty motif set (\(F=\emptyset\)) and \(INIT_1\) as one-hot relation encoding. However, this breaks relation invariance, so they cannot generalize to unseen relations.
- **NBFNet (Zhu et al., 2021)**: Strictly more expressive than ULTRA for KGs with fixed relations, but cannot generalize to unseen relations (unlike ULTRA).


## 7. Experimental analysis
We evaluate MOTIF on 54 KGs to answer four questions:
- **Q1**: Does MOTIF with richer motifs perform better?
- **Q2**: Does ULTRA match \(MOTIF(F_2^{path})\)’s expressivity?
- **Q3**: How does refined relation invariance help link prediction?
- **Q4**: What is the expressivity-scalability trade-off? (See Appendices H and I.)

Our base architecture uses:
- **Relation encoder**: HCNets (Huang et al., 2024),
- **Entity encoder**: Modified NBFNet (Zhu et al., 2021) (see Appendix E).

### 7.1. Synthetic experiments: ConnectHub
We construct synthetic datasets \(ConnectHub(k)\) to validate Q1 (enhanced expressivity with richer motifs).

#### Task & Setup
Each \(ConnectHub(k)\) KG has:
- Relations: Positive class \(P\) (size \(k+1\)), negative class \(N\) (size \(k+1\)), query relation \(q\),
- Structure: A \((k+1)\)-star hub (positive relations), plus \(k\)-star communities (positive/negative subsets of \(P/N\)).

The task: Predict if \(q\) links the hub center to positive (not negative) communities. Success requires distinguishing \(P\) and \(N\). We compare ULTRA and \(MOTIF(F_m^{star})\) (varying \(m\)).

#### Results (Table 1)
| Model               | \(k=2\) | \(k=3\) | \(k=4\) | \(k=5\) | \(k=6\) |
|---------------------|---------|---------|---------|---------|---------|
| ULTRA               | 0.50    | 0.50    | 0.50    | 0.50    | 0.50    |
| \(MOTIF(F_2^{star})\) | 0.50    | 0.50    | 0.50    | 0.50    | 0.50    |
| \(MOTIF(F_3^{star})\) | 1.00    | 0.50    | 0.50    | 0.50    | 0.50    |
| \(MOTIF(F_4^{star})\) | 1.00    | 1.00    | 0.50    | 0.50    | 0.50    |
| \(MOTIF(F_5^{star})\) | 1.00    | 1.00    | 1.00    | 0.50    | 0.50    |
| \(MOTIF(F_6^{star})\) | 1.00    | 1.00    | 1.00    | 1.00    | 0.50    |
| \(MOTIF(F_7^{star})\) | 1.00    | 1.00    | 1.00    | 1.00    | 1.00    |

*Table 1. Accuracy for \(ConnectHub(k)\) datasets.*

- **Ineffective motifs**: ULTRA and \(MOTIF(F_m^{star})\) ( \(m \leq k\) ) cannot detect the \((k+1)\)-star hub, so \(LIFT_F(G)\) has isomorphic hypergraphs for \(P\) and \(N\) (50% accuracy, random guess).
- **Effective motifs**: \(MOTIF(F_m^{star})\) ( \(m = k+1\) ) detects the positive hub (hyperedge over \(P\), no hyperedge over \(N\)), achieving 100% accuracy. This validates Q1.

### 7.2. Pretraining and fine-tuning experiments
#### Datasets & Evaluation
- **Pretraining**: FB15k237 (Toutanova & Chen, 2015), WN18RR (Dettmers et al., 2018), CoDEx Medium (Safavi & Koutra, 2020).
- **Inference**: 51 KGs across 3 settings:
  1. Inductive (nodes + relations) (\(Inductive_{e,r}\)),
  2. Inductive (nodes only) (\(Inductive_e\)),
  3. Transductive.

We use inverse relation augmentation ( \(r(u, v) \to r^{-1}(v, u)\) ) and report **MRR** (Mean Reciprocal Rank) and **Hits@10** (filtered ranking; Bordes et al., 2013). Code: https://github.com/HxyScotthuang/MOTIF/.

#### Setup
We test four MOTIF variants:
- \(F_3^{path}\) (3-path motifs),
- \(F_2^{path}\) (2-path motifs),
- \(\{h2t\}\) (only head-to-tail),
- \(\emptyset\) (no motifs).

We compare with ULTRA (Galkin et al., 2024).

#### Results (Table 2)
| Model               | Zero-shot Inference (MRR/H@10) |          |          |          | Fine-tuned Inference (MRR/H@10) |          |          |          |
|---------------------|---------------------------------|----------|----------|----------|----------------------------------|----------|----------|----------|
|                     | \(Inductive_{e,r}\)             | \(Inductive_e\) | Transductive | Avg      | \(Inductive_{e,r}\)              | \(Inductive_e\) | Transductive | Avg      |
| ULTRA               | 0.339/0.436                     | 0.374/0.529 | 0.378/0.537 | -        | 0.335/0.492                      | 0.370/0.527 | 0.378/0.537 | -        |
| \(MOTIF(F_3^{path})\) | 0.345/0.431                     | 0.374/0.529 | 0.378/0.537 | 0.366/0.498 | 0.397/0.556                      | 0.410/0.563 | 0.415/0.569 | 0.407/0.563 |
| \(MOTIF(F_2^{path})\) | 0.349/0.431                     | 0.374/0.529 | 0.378/0.537 | 0.367/0.499 | 0.401/0.558                      | 0.419/0.569 | 0.415/0.565 | 0.412/0.564 |
| \(MOTIF(\{h2t\})\)   | 0.337/0.422                     | 0.361/0.518 | 0.361/0.518 | 0.353/0.486 | 0.384/0.543                      | 0.394/0.563 | 0.394/0.563 | 0.391/0.556 |
| \(MOTIF(\emptyset)\)  | 0.074/0.107                     | 0.082/0.134 | 0.082/0.134 | 0.080/0.125 | 0.107/0.169                      | 0.101/0.134 | 0.101/0.134 | 0.103/0.146 |

*Table 2. Average zero-shot and fine-tuned MRR/Hits@10 over 51 KGs.*

Key findings:
1. **Q1 validated**: \(MOTIF(F_3^{path})\) outperforms \(MOTIF(F_2^{path})\) in zero-shot/fine-tuned settings (richer motifs learn better relation invariants).
2. **Q2 validated**: \(MOTIF(F_2^{path})\) matches ULTRA’s performance (theoretical equivalence holds empirically).
3. **Motif impact**: Removing motifs ( \(\{h2t\} \to \emptyset\) ) degrades performance; \(\emptyset\) fails to learn non-trivial invariants (no generalization).

#### Domain-specific results (Q1)
- **Metafam (Zhou et al., 2023a)**: \(MOTIF(F_3^{path})\) improves MRR by 45% over ULTRA (zero-shot), capturing conflicting/compositional patterns.
- **WIKITOPICS-MT (Zhou et al., 2023a)**: \(MOTIF(F_3^{path})\) improves MRR (0.331→0.358) and Hits@10 (0.442→0.481) averaged over 8 datasets; 58% relative gain on MT1-tax.

### 7.3. End-to-end experiments
We train \(MOTIF(F_3^{path})\) from scratch on each dataset’s training set and evaluate on validation/test sets.

#### Results (Table 3)
| Model               | Avg MRR | Avg Hits@10 |
|---------------------|---------|-------------|
| ULTRA               | 0.394   | 0.552       |
| \(MOTIF(F_3^{path})\) | 0.409   | 0.561       |

*Table 3. Average end-to-end MRR/Hits@10 over 54 KGs.*

\(MOTIF(F_3^{path})\) outperforms ULTRA (Q1 validated).

#### Case study: WN-v2 (Q3)
WN-v2 (Teru et al., 2020) has 20 relations (inverse-augmented). ULTRA’s MRR (0.296) is far lower than \(MOTIF(F_3^{path})\)’s (0.684). Figure 9 shows cosine similarities of relation embeddings: ULTRA produces highly similar embeddings, while \(MOTIF(F_3^{path})\) generates distinguishing ones (aids link prediction).

![Figure 9. Cosine similarities between relation embeddings (queried relations: \(r_0\)=derivationally related form, \(r_2\)=hypernym) in WN-v2 test set. ULTRA’s embeddings are more similar; MOTIF’s are distinct.](https://example.com/figure9)  
*Figure 9. Cosine similarities between relation embeddings (queried relations: \(r_0\)=derivationally related form, \(r_2\)=hypernym) in WN-v2 test set. ULTRA’s embeddings are more similar; MOTIF’s are distinct.*

Notably, pretraining (vs. end-to-end) avoids this degradation, suggesting diverse motifs or multi-KG training improve relation learning.


## 8. Conclusions
We introduced MOTIF, a general framework for KGFMs that integrates arbitrary graph motifs. We rigorously analyzed KGFM expressive power, identified conditions for motif-driven expressivity gains, and validated findings on 54 KGs.

### Limitations & Future Work
While richer motifs enhance expressivity, they increase computational cost (time/memory; Appendices H and I). Future work will improve MOTIF’s scalability for large-scale KGs.


## Acknowledgment
Bronstein is supported by EPSRC Turing AI World-Leading Research Fellowship No. EP/X040062/1 and EPSRC AI Hub on Mathematical Foundations of Intelligence: An “Erlangen Programme” for AI No. EP/Y028872/1. Reutter is funded by Fondecyt grant 1221799. Barceló and Reutter are funded by ANID–Millennium Science Initiative Program - Code ICN17002. Barceló and Romero are funded by the National Center for Artificial Intelligence CENIA FB210017, Basal ANID.


## Impact Statement
This work advances machine learning for knowledge graphs, with potential applications in recommendation systems and KG completion. These applications may improve information retrieval and personalization but carry risks (bias amplification, misinformation). No specific immediate concerns require special attention.


## References
Abboud, R., Ceylan, İ. İ., Lukasiewicz, T., and Salvatori, T. Boxe: A box embedding model for knowledge base completion. In NeurIPS, 2020.  
Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. In arXiv, 2016.  
Barceló, P., Galkin, M., Morris, C., and Romero, M. Weisfeiler and leman go relational. In LoG, 2022.  
Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., and Yakhnenko, O. Translating embeddings for modeling multi-relational data. In NIPS, 2013.  
Brody, S., Alon, U., and Yahav, E. How attentive are graph attention networks? In ICLR, 2022.  
Cui, Y., Sun, Z., and Hu, W. A prompt-based knowledge graph foundation model for universal in-context reasoning. In NeurIPS, 2024.  
Dettmers, T., Pasquale, M., Pontus, S., and Riedel, S. Convolutional 2D knowledge graph embeddings. In AAAI, 2018.  
Fatemi, B., Taslakian, P., Vazquez, D., and Poole, D. Knowledge hypergraphs: Prediction beyond binary relations. In IJCAI, 2020.  
Fey, M. and Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
Galárraga, L. A., Teflioudi, C., Hose, K., and Suchanek, F. AMIE: Association rule mining under incomplete evidence in ontological knowledge bases. In WWW, 2013.  
Galkin, M., Denis, E., Wu, J., and Hamilton, W. L. Nodepiece: Compositional and parameter-efficient representations of large knowledge graphs. In ICLR, 2022.  
Galkin, M., Yuan, X., Mostafa, H., Tang, J., and Zhu, Z. Towards foundation models for knowledge graph reasoning. In ICLR, 2024.  
Gao, J., Zhou, Y., Zhou, J., and Ribeiro, B. Double equivariance for inductive link prediction for both new nodes and new relation types. In arXiv, 2023.  
Geng, Y., Chen, J., Pan, J. Z., Chen, M., Jiang, S., Zhang, W., and Chen, H. Relational message passing for fully inductive knowledge graph completion. In ICDE, 2023.  
Grohe, M. The logic of graph neural networks. In LICS, 2021.  
Himmelstein, D. S., Lizee, A., Hessler, C., Brueggeman, L., Chen, S. L., Hadley, D., Green, A., Khankhanian, P., and Baranzini, S. E. Systematic integration of biomedical knowledge prioritizes drugs for repurposing. Elife, 2017.  
Huang, X., Orth, M. R., Ceylan, İ. İ., and Barceló, P. A theory of link prediction via relational weisfeiler-leman on knowledge graphs. In NeurIPS, 2023.  
Huang, X., Orth, M. R., Barceló, P., Bronstein, M. M., and İsmail İlkan Ceylan. Link prediction with relational hypergraphs. In arXiv, 2024.  
Lee, J., Chung, C., and Whang, J. J. Ingram: Inductive knowledge graph embedding via relation graphs. In ICML, 2023.  
Liu, S., Grau, B., Horrocks, I., and Kostylev, E. Indigo: Gnn-based inductive knowledge graph completion using pair-wise encoding. In NeurIPS, 2021.  
Liu, Y., Yao, Q., and Li, Y. Generalizing tensor decomposition for n-ary relational knowledge bases. In WWW, 2020.  
Lv, X., Xu Han, L. H., Li, J., Liu, Z., Zhang, W., Zhang, Y., Kong, H., and Wu, S. Dynamic anticipation and completion for multi-hop reasoning over sparse knowledge graph. In EMNLP, 2020.  
Mahdisoltani, F., Biega, J. A., and Suchanek, F. M. Yago3: A knowledge base from multilingual wikipedias. In CIDR, 2015.  
Mao, H., Chen, Z., Tang, W., Zhao, J., Ma, Y., Zhao, T., Shah, N., Galkin, M., and Tang, J. Position: Graph foundation models are already here. In ICML, 2024.  
Safavi, T. and Koutra, D. CoDEx: A Comprehensive Knowledge Graph Completion Benchmark. In EMNLP, 2020.  
Schlichtkrull, M. S., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., and Welling, M. Modeling relational data with graph convolutional networks. In ESWC, 2018.  
Sun, Z., Deng, Z.-H., Nie, J.-Y., and Tang, J. Rotate: Knowledge graph embedding by relational rotation in complex space. In ICLR, 2019.  
Teru, K. K., Denis, E. G., and Hamilton, W. L. Inductive relation prediction by subgraph reasoning. In ICML, 2020.  
Toutanova, K. and Chen, D. Observed versus latent features for knowledge base and text inference. In Workshop on Continuous Vector Space Models and their Compositionality, 2015.  
Vashishth, S., Sanyal, S., Nitin, V., and Talukdar, P. Composition-based multi-relational graph convolutional networks. In ICLR, 2020.  
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In NIPS, 2017.  
Wen, J., Li, J., Mao, Y., Chen, S., and Zhang, R. On the representation and embedding of knowledge bases beyond binary relations. In IJCAI, 2016.  
Xiong, W., Hoang, T., and Wang, W. Y. Deeppath: A reinforcement learning method for knowledge graph reasoning. In EMNLP, 2017.  
Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? In ICLR, 2019.  
Yadati, N. Neural message passing for multi-relational ordered and recursive hypergraphs. In NeurIPS, 2020.  
Zhang, M., Li, P., Xia, Y., Wang, K., and Jin, L. Labeling trick: A theory of using graph neural networks for multinode representation learning. In NeurIPS, 2021.  
Zhang, Y. and Yao, Q. Knowledge graph reasoning with relational digraph. In WebConf, 2022.  
Zhang, Y., Zhou, Z., Yao, Q., Chu, X., and Han, B. Adaprop: Learning adaptive propagation for graph neural network based knowledge graph reasoning. In KDD, 2023.  
Zhang, Y., Bevilacqua, B., Galkin, M., and Ribeiro, B. TRIX: A more expressive model for zero-shot domain transfer in knowledge graphs. In LoG, 2024.  
Zhou, J., Bevilacqua, B., and Ribeiro, B. A multi-task perspective for link prediction with new relation types and nodes. In NeurIPS GLFrontiers, 2023a.  
Zhou, X., Hui, B., Zeira, I., Wu, H., and Tian, L. Dynamic relation learning for link prediction in knowledge hypergraphs. In Appl Intell, 2023b.  
Zhu, Z., Zhang, Z., Xhonneux, L.-P., and Tang, J. Neural bellman-ford networks: A general graph neural network framework for link prediction. In NeurIPS, 2021.  
Zhu, Z., Yuan, X., Galkin, M., Xhonneux, S., Zhang, M., Gazeau, M., and Tang, J. A*net: A scalable path-based reasoning approach for knowledge graphs. In NeurIPS, 2023.


## Appendix A. C-MPNNs and HC-MPNNs
We define **Conditional Message-Passing Neural Networks (C-MPNNs)** (Huang et al., 2023) and **Hypergraph Conditional MPNNs (HC-MPNNs)** (Huang et al., 2024), plus their corresponding Weisfeiler-Leman (WL) tests for expressivity.

### A.1. Model definitions
#### C-MPNNs
For a KG \(G=(V, E, R)\), a C-MPNN computes node representations **conditioned on query \(q \in R\) and source node \(u \in V\)** iteratively:
\[
h_{v|u,q}^{(0)} = INIT(u, v, q)
\]
\[
h_{v|u,q}^{(\ell+1)} = UP\left( h_{v|u,q}^{(\ell)}, AGG\left( \left\{ MSG_r\left(h_{w|u,q}^{(\ell)}, z_q\right) \mid w \in \mathcal{N}_r(v), r \in R \right\} \right) \right)
\]
where:
- \(INIT, UP, AGG, MSG_r\): Differentiable functions (initialization, update, aggregation, relation-specific message),
- \(z_q\): Learnable vector for query \(q\),
- \(h_q^{(\ell)}(u, v) = h_{v|u,q}^{(\ell)}\) (pair representation),
- \(INIT\) satisfies **target node distinguishability**: \(INIT(u, u, q) \neq INIT(u, v, q)\) for \(u \neq v\).

A unary decoder \(Dec: \mathbb{R}^{d(L)} \to \mathbb{R}\) predicts the likelihood of \(q(u, v)\).

#### HC-MPNNs
For a relational hypergraph \(H=(V, E, R)\) and query \(q=(q, \tilde{u}, t)\) (missing the \(t\)-th node in \(q(u_1, \dots, u_{t-1}, ?, u_{t+1}, \dots, u_m)\)), an HC-MPNN computes node representations iteratively:
\[
h_{v|q}^{(0)} = INIT(v, q)
\]
\[
h_{v|q}^{(\ell+1)} = UP\left( h_{v|q}^{(\ell)}, AGG\left( \left\{ MSG_{\rho(e)}\left( \left\{ (h_{w|q}^{(\ell)}, j) \mid (w, j) \in \mathcal{N}_i(e) \right\}, q \right) \mid (e, i) \in E(v) \right\} \right) \right)
\]
where:
- \(E(v) = \{ (e, i) \mid e(i)=v, e \in E \}\),
- \(\mathcal{N}_i(e) = \{ (e(j), j) \mid j \neq i \}\) (positional neighborhood),
- \(h_q^{(\ell)}(v) = h_{v|q}^{(\ell)}\) (node representation).

#### HCNets
HCNets are an HC-MPNN variant with specific functions:
\[
h_{v|q}^{(0)} = \sum_{i \neq t} \mathbb{1}_{v=u_i} * (p_i + z_q)
\]
\[
h_{v|q}^{(\ell+1)} = \sigma\left( W^{(\ell)} \left[ h_{v|q}^{(\ell)} \parallel \sum_{(e,i) \in E(v)} g_{\rho(e),q}^{(\ell)} \left( \odot_{j \neq i} (\alpha^{(\ell)} h_{e(j)|q}^{(\ell)} + (1-\alpha^{(\ell)}) p_j \right) \right) \right] + b^{(\ell)} \right)
\]
where:
- \(g_{\rho(e),q}^{(\ell)}\): Learnable diagonal message function,
- \(\sigma\): Activation function (e.g., ReLU),
- \(W^{(\ell)}, b^{(\ell)}\): Learnable weights/biases,
- \(p_i\): Sinusoidal positional encoding (Vaswani et al., 2017),
- \(\mathbb{1}_C\): Indicator function.

### A.2. Relational WL tests
#### Relational Asymmetric Local 2-WL (rawl₂)
Captures C-MPNN expressivity (Huang et al., 2023). For a KG \(G=(V, E, R)\) and initial pairwise coloring \(\eta: V \times V \to D\) (satisfies target node distinguishability):
\[
rawl_2^{(0)}(u, v) = \eta(u, v)
\]
\[
rawl_2^{(\ell+1)}(u, v) = HASH\left( rawl_2^{(\ell)}(u, v), \left\{ (rawl_2^{(\ell)}(u, w), r) \mid w \in \mathcal{N}_r(v), r \in R \right\} \right)
\]
where \(HASH\) is injective. Theorem 5.1 (Huang et al., 2023) shows \(rawl_2^{(\ell)}\) characterizes C-MPNN expressivity.

#### Hypergraph Relational Local 1-WL (hrwl₁)
Captures HC-MPNN expressivity (Huang et al., 2024). For a hypergraph \(H=(V, E, R)\) and initial node coloring \(c: V \to D\):
\[
hrwl_1^{(0)}(v) = c(v)
\]
\[
hrwl_1^{(\ell+1)}(v) = HASH\left( hrwl_1^{(\ell)}(v), \left\{ \left( \left\{ (hrwl_1^{(\ell)}(w), j) \mid (w, j) \in \mathcal{N}_i(e) \right\}, \rho(e) \right) \mid (e, i) \in E(v) \right\} \right)
\]
For queries \(q\), \(c\) must satisfy **generalized target node distinguishability**: \(c(u) \neq c(v)\) for \(u \in \tilde{u}, v \notin \tilde{u}\) and \(c(u_i) \neq c(u_j)\) for \(u_i \neq u_j \in \tilde{u}\). Theorem 5.1 (Huang et al., 2024) links \(hrwl_1\) to HC-MPNN expressivity.

#### Hypergraph Relational Local 2-WL (hcwl₂)
For binary hypergraphs (arity-2 edges) and tail prediction, use pairwise coloring \(\eta: V \times V \to D\) (target node distinguishability):
\[
hcwl_2^{(0)}(u, v) = \eta(u, v)
\]
\[
hcwl_2^{(\ell+1)}(u, v) = HASH\left( hcwl_2^{(\ell)}(u, v), \left\{ \left( \left\{ (hcwl_2^{(\ell)}(u, w), j) \mid (w, j) \in \mathcal{N}_i(e) \right\}, \rho(e) \right) \mid (e, i) \in E(v) \right\} \right)
\]


## Appendix B. Differences between TRIX and MOTIF
TRIX (Zhang et al., 2024) and MOTIF are **incomparable** in expressivity:
- **TRIX**: Distinguishes relations via **frequency counts** of homomorphism matches. For example, in \(E=\{r_1(u_1,v_1), r_2(v_1,w_1), r_3(u_2,v_2), r_4(v_2,w_2), r_3(u_3,v_3), r_4(v_3,w_3)\}\), TRIX separates \((r_1,r_2)\) (1 match) and \((r_3,r_4)\) (2 matches).
- **MOTIF**: Distinguishes relations via **existence of higher-order motifs**. For example, in \(E=\{r_1(x_1,x_2), r_2(x_1,x_2), r_1(x_3,x_4), r_2(x_3,x_4), r_3(y_1,y_2), r_4(y_1,y_4), r_3(y_3,y_2), r_4(y_3,y_4)\}\), MOTIF uses the PARA motif (\(\alpha(x,y), \beta(x,y)\)) to separate \((r_1,r_2)\) (matches PARA) and \((r_3,r_4)\) (does not).

TRIX excels at frequency-based distinction; MOTIF at structural motif-based distinction. Augmenting TRIX with higher-order motifs could boost its expressivity (aligning with MOTIF’s framework).


## Appendix C. A WL test for MOTIF
We define a two-stage WL test that matches \(MOTIF(F)\)’s separation power. This simplifies proofs and is of independent interest.

### Test Overview
For a KG \(G=(V, E, R)\), the test assigns colors to links \(q(u, v)\) via two stages:
1. **Stage 1**: Color relations \(r \in R\) (nodes of \(LIFT_F(G)\)) conditioned on query \(q \in R\) (denoted \(hcwl_F^{(t)}(q, r)\)).
2. **Stage 2**: Color nodes \(v \in V\) (of \(G\)) conditioned on \(q\) and source \(u\) (final link color \(col_{F,T}^{(\ell)}(q(u, v))\)).

### Formal Definitions
#### Stage 1: Relation Coloring (\(hcwl_F^{(t)}\))
\[
hcwl_F^{(0)}(q, r) = \mathbb{1}_{r=q} * 1
\]
\[
hcwl_F^{(t+1)}(q, r) = HASH\left( hcwl_F^{(t)}(q, r), \left\{ \left( \left\{ (hcwl_F^{(t)}(q, s), j) \mid (s, j) \in \mathcal{N}^i(e) \right\}, \rho(e) \right) \mid (e, i) \in E_{LIFT}(r) \right\} \right)
\]
where:
- \(E_{LIFT}(r) = \{ (e, i) \mid e(i)=r, e \in E_{LIFT} \}\),
- \(\mathcal{N}^i(e) = \{ (e(j), j) \mid j \neq i \}\),
- \(\mathbb{1}_{r=q}\): 1 if \(r=q\), else 0.

#### Stage 2: Link Coloring (\(col_{F,T}^{(\ell)}\))
\[
col_{F,T}^{(0)}(q(u, v)) = \mathbb{1}_{v=u} * hcwl_F^{(T)}(q, q)
\]
\[
col_{F,T}^{(\ell+1)}(q(u, v)) = HASH\left( col_{F,T}^{(\ell)}(q(u, v)), \left\{ (col_{F,T}^{(\ell)}(q(u, w)), hcwl_F^{(T)}(q, r)) \mid w \in \mathcal{N}_r(v), r \in R \right\} \right)
\]

### Key Propositions
#### Proposition C.1
For any \(MOTIF(F)\) instance ( \(T\) relation layers, \(L\) entity layers) and links \(q(u, v), q'(u', v')\):
\[
col_{F,T}^{(L)}(q(u, v)) = col_{F,T}^{(L)}(q'(u', v')) \implies Enc_{KG,u,q}[v] = Enc_{KG,u',q'}[v']
\]
*Proof: See Appendix C.*

#### Proposition C.2
For any \(T, L \geq 0\), there exists a \(MOTIF(F)\) instance such that:
\[
col_{F,T}^{(L)}(q(u, v)) = col_{F,T}^{(L)}(q'(u', v')) \iff Enc_{KG,u,q}[v] = Enc_{KG,u',q'}[v']
\]
*Proof: See Appendix C.*

#### Corollary C.3
\(F \preceq F'\) iff for every \(G\), links \(q(u, v), q'(u', v')\), and \(T, L \geq 0\):
\[
col_{F,T}^{(L)}(q(u, v)) = col_{F,T}^{(L)}(q'(u', v')) \implies col_{F',T}^{(L)}(q(u, v)) = col_{F',T}^{(L)}(q'(u', v'))
\]


## Appendix D. Missing proofs in the paper
### D.1. Proof of Proposition 6.2
Let \(P=(G_M, \bar{r})\) and \(P'=(G_M', \bar{r}')\) be isomorphic motifs. For any \(F\):
\[
(F \cup \{P\}) \sim (F \cup \{P'\}) \sim (F \cup \{P, P'\})
\]

**Proof**: Let \(F_1=F \cup \{P\}\), \(F_2=F \cup \{P, P'\}\). We show \(F_1 \sim F_2\) (the rest is analogous). By Corollary C.3, we need:
\[
col_{F_1,T}^{(L)}(q(u, v)) = col_{F_1,T}^{(L)}(q'(u', v')) \iff col_{F_2,T}^{(L)}(q(u, v)) = col_{F_2,T}^{(L)}(q'(u', v'))
\]

Since Stage 2 colorings depend only on Stage 1, we focus on \(hcwl_{F_1}^{(t)}(q, r)\) and \(hcwl_{F_2}^{(t)}(q, r)\). For \(t=0\), colorings are identical. Assume \(t>0\):

- \(hcwl_{F_2}^{(t)}(q, r)\) includes an extra term for \(P'\): \(\left\{ \left( \left\{ (hcwl_{F_2}^{(t-1)}(q, x), j) \mid (x, j) \in \mathcal{N}^i(P') \right\}, \rho(e) \right) \mid (P', i) \in