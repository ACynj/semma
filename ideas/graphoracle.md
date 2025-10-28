# 4 The Proposed Method
In order to enable fully inductive KG reasoning and improve the generalization ability of models across KGs, the key is to generalize the dependencies among relations for different KGs. To achieve this goal, we firstly introduce Relational Dependency Graph (RDG), which explicitly models how relations depend on each other, in Section 4.1. Based on RDG, we propose a query-dependent multi-head attention mechanism to learn relation representations from a weighted combination of precedent relations in Section 4.2. Subsequently, in Section 4.3, we introduce the approach where entity representations are represented with the recursive function (1) in the original KGs by using the relation representations just obtained. The overview of our approach is shown in Fig 2.

![Figure 2](Note: Fig 2 content description: Overview of the GRAPHORACLE process for predicting the answer entity \(e_{a}\) from a given query \((e_{1}, r_{1}, ?)\): Given a Knowledge Graph, we first construct the Relation Dependency Graph (RDG). Then, a multi-head attention mechanism combined with a GNN is used to propagate messages among RDG to obtain relation representations \(R_{q}\), which are then used in a second GNN for message passing over entity representations. Finally, a scoring function evaluates candidate entities based on the aggregated entity representations, ranking them for answer entity prediction. The figure includes components labeled as "Relational-Dependency Graph", "Query = \(e_1, r_1, ?\)", "Knowledge Graph \(G=(V, R, F)\)", "RDG construction", "Multi-head GNN Model 1", "GNN Model 2", "Score Function", "Candidates", and "Answer Prediction".)

## 4.1 Relation-Dependency Graph (RDG) Construction
To build an effective KGFM capable of cross KG generalization, we must capture the fundamental dependent patterns through which one relation can be represented by a combination of others. Our key innovation is transforming the entity-relation interaction space into a relation-dependency interaction manifold that explicitly models how relations influence each other.

Given a KG \(G=(V, R, F)\), we construct a RDG \(G^{R}\) through a structural transformation. First, we define a relation adjacency operator \(\Phi: F \to R × R\) that extracts transitive relation dependencies:

Then, the RDG is defined as \(G^{R}=(R, E^{R})\), where the node set \(R\) contains all the relations and the edge sets \(E^{R}=\Phi(F)\) includes relation dependencies induced by entity-mediated pathways.

This transformation alters the conceptual framework, shifting from an entity-centric perspective to a relation-interaction manifold where compositional connections between relations become explicit. Each directed edge \((r_{i}, r_{j})\) in \(G^{R}\) represents a potential relation-dependency pathway, indicating that relation \(r_{i}\) preconditions relation \(r_{j}\) through their sequential interaction mediated by a common entity substrate. The edge structure encodes compositional relational semantics, capturing how one relation may influence the probability or applicability of another relation in sequence.

To incorporate the hierarchical and compositional nature of relation interactions, we define a partial ordering function \(\tau: R \to \mathbb{R}\) that assigns each relation a position in a relation precedence structure. This ordering is derived from the KG’s inherent structure through rigorous topological analysis of relation co-occurrence patterns and functional dependencies. Relations that serve as logical precursors in inference chains are assigned lower \(\tau\) values, thereby establishing a directed acyclic structure in the relation graph that reflects the natural flow of information propagation. Using this ordering, we define the set of preceding relations for any relation \(r_{v}\) as:

This formulation enables us to capture the directional dependency patterns where relations with lower positions in the hierarchy systematically precede and inform relations with higher \(\tau\). By explicitly modeling these precedence relationships, our framework can identify and leverage compositional reasoning patterns that remain invariant across domains, enhancing the generalization capabilities.

## 4.2 Relation Representation Learning on Relation-Dependency Graph
Building on the constructed RDG \(G^{R}\), we develop a representation mechanism that captures the contextualized semantics of relations conditioned on a specific query. Given a query relation \(r_{q}\), we introduce an RDG aggregation mechanism to compute \(d\)-dimensional relation-node representations \(R_{q} \in \mathbb{R}^{|R| × d}\) conditional on \(r_{q}\).

Following Eq. (1), we apply a labeling initialization to distinguish the query relation node \(r_{q}\) in \(G^{R}\). Then employ multi-head attention relation-dependency message passing over the graph: where \(\delta_{r_{v}, r_{q}}=1\) if \(v=q\), and 0 otherwise. \(H\) is the number of attention heads, and \(W_{1}^{\ell, h}\), \(W_{2}^{\ell, h} \in \mathbb{R}^{d × d}\) are head-specific parameter matrices. The relation-dependency attention weight \(\hat{\alpha}_{r_{u} r_{v}}^{\ell, h}\) captures the directional influence of relation \(r_{u}\) on relation \(r_{v}\), computed as:

\[
\hat{\alpha}_{r_{u}r_{v}}^{\ell, h} = \frac{exp\left(a^{T}\left[W_{\alpha }^{h}h_{r_{u}}^{\ell -1} \parallel W_{\alpha }^{h}h_{r_{v}}^{\ell -1}\right]\right)}{\sum _{r_{w}\in \mathcal{N}^{past}(r_{v})}exp\left(a^{T}\left[W_{\alpha }^{h}h_{r_{w}}^{\ell -1} \parallel W_{\alpha }^{h}h_{r_{v}}^{\ell -1}\right]\right)},
\] (6)

where \(a \in \mathbb{R}^{2d}\) is a learnable attention parameter vector, "∥" denotes vector concatenation, and \(W_{\alpha}^{h} \in \mathbb{R}^{d × d}\) are head-specific trainable projection matrix. The neighborhood function enforces the relation-dependency ordering of relations as defined in Eq. (4).

\[
h_{e|q}^{\ell} = \delta\left(W^{\ell} \cdot \sum_{(e_{s},r,e)\in \mathcal{F}_{train}}\alpha_{e_{s},r|q}^{\ell}\left(h_{e_{s}|q}^{\ell -1} + h_{r|_{r_{q}}}^{L_{r}}\right)\right),
\]

After \(L_r\) layers of message passing, the final relation representation incorporates both local and higher-order dependencies \(R_{q}=\{h_{r | r_{q}}^{L_{r}} | r \in R\}\).

## 4.3 Entity Representation Learning on the Original KG
After obtaining the relation representations \(R_{q}\) for \(r_{q}\) from RDG, we obtain query-dependent entity representations by conducting message passing over the original KG structures. This approach enables effective reasoning across both seen and unseen entities and relations.

For a given query \((e_{q}, r_{q}, ?)\), we compute entity representations recursively with Eq. (1) through the KG \(G\). The initial representations \(h_{e | q}^{0}=1\) if \(e=e_{q}\), and otherwise 0. At each layer \(\ell\), the representation of an entity \(e\) is computed as: where \(\delta(\cdot)\) is a non-linear activation, and the attention weight \(\alpha_{e_{s}, r | q}^{\ell}\) is computed as:

where \(w_{\alpha}^{\ell} \in \mathbb{R}^{d}\) and \(W_{\alpha}^{\ell} \in \mathbb{R}^{d × 3d}\) are learnable parameters, and \(\sigma\) is the sigmoid function¹.

We iterate Eq. (7) for \(L_{e}\) steps and use the final layer representation \(h_{e | q}^{L_{e}}\) for scoring each entity \(e \in V\). The critical idea here is replacing the learnable relation embeddings \(r\) with the contextualized relation embedding \(h_{r | r_{q}}^{L_{r}}\) from our relation-dependency relation graph, enabling fully inductive reasoning.

## 4.4 Training Details
All the learnable parameters such as \(W_{O}^{h}\), \(W_{2}^{\ell, h}\), \(W_{\alpha}^{h}\), \(a\), \(W^{\ell}\), \(W_{\alpha}^{\ell}\), \(w_{\alpha}^{\ell}\), \(w_{s}\) are trained end-to-end by minimizing the loss function Eq. (2). GRAPHORACLE adopts a sequential multi-dataset pre-train → fine-tune paradigm to acquire a general relation-dependency graph representation across KGs \(\{G_{1}, ..., G_{K}\}\). For each graph \(G_{k}\), we optimize the regularized objective \(L^{(k)}=L_{task}^{(k)}+\lambda_{k}\|\Theta\|_{2}^{2}\), where \(L_{task}^{(k)}\) denotes the task-specific loss on \(G_{k}\), \(\Theta\) represents all learnable parameters, and \(\lambda_{k}\) controls the strength of L2 regularization. Early stopping technique is used for each graph once validation MRR fails to improve for several epochs. This iterative pre-train process, together with our relation-dependency graph encoder, equips GRAPHORACLE with strong cross-domain generalization. (Full details are reported in Appendix D.) When adapting GRAPHORACLE to new KGs, we firstly build the relation-dependency graph and then support two inference paradigms:
- **Zero-shot Inference**: The pre-trained model is directly applied to unseen KGs without tuning.
- **Fine-tuning**: For more challenging domains, we fine-tune the pre-trained parameters on the target KG \(G_{target}\) for a limited number of epochs \(E_{fine-tune} \ll E_{train}\).


¹Theoretical analysis of the GRAPHORACLE model is given in Appendix H, and time complexity is given in Appendix B.1.
