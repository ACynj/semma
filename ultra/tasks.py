from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import re
import os
import numpy as np
import json

from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from ultra import parse

mydir = os.getcwd()
flags = parse.load_flags(os.path.join(mydir, "flags.yaml"))

def load_relation_types(dataset_name):
    """
    Load relation types from JSON file based on dataset name.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'CoDExMedium', 'FB15k237', 'WN18RR')
    
    Returns:
        dict: Dictionary mapping relation names to their types ('Symmetric', 'Asymmetric', 'Antisymmetric')
    """
    json_path = f"/T20030104/ynj/semma/openrouter/relations_type/gpt-4o-2024-11-20/{dataset_name}.json"
    
    if not os.path.exists(json_path):
        print(f"Warning: Relation types file not found at {json_path}")
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('relations_type', {})

def get_relation_type(relation_name, relation_types_dict):
    """
    Get the type of a relation from the loaded types dictionary.
    
    Args:
        relation_name (str): Name of the relation
        relation_types_dict (dict): Dictionary mapping relation names to types
    
    Returns:
        str: Relation type ('Symmetric', 'Asymmetric', 'Antisymmetric') or 'Asymmetric' as default
    """
    return relation_types_dict.get(relation_name, 'Asymmetric')

def get_inverse_relation_semantics(dataset_name, relation_name):
    """
    Get inverse relation semantics (name and description) from JSON file.
    
    Args:
        dataset_name (str): Name of the dataset
        relation_name (str): Name of the original relation
    
    Returns:
        tuple: (inverse_relation_name, inverse_relation_description)
    """
    json_path = f"/T20030104/ynj/semma/openrouter/relations_type/gpt-4o-2024-11-20/{dataset_name}.json"
    
    if not os.path.exists(json_path):
        print(f"Warning: Relation types file not found at {json_path}")
        return "", ""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_inverse_relations = data.get('cleaned_inverse_relations', {})
    inverse_relations_descriptions = data.get('inverse_relations_descriptions', {})
    
    inverse_name = cleaned_inverse_relations.get(relation_name, "")
    inverse_desc = inverse_relations_descriptions.get(relation_name, "")
    
    return inverse_name, inverse_desc

def generate_inverse_embeddings_for_asymmetric(relation_names, dataset_name, model_embed="jinaai"):
    """
    Generate embeddings for inverse relations of asymmetric relations using semantic information.
    
    Args:
        relation_names (list): List of relation names
        dataset_name (str): Name of the dataset
        model_embed (str): Embedding model to use
    
    Returns:
        torch.Tensor: Embeddings for inverse relations
    """
    device = torch.device(f"cuda:{flags.gpus}" if torch.cuda.is_available() else "cpu")
    
    print(f"🧠 [语义嵌入生成] 为非对称关系生成逆关系嵌入")
    print(f"   - 嵌入模型: {model_embed}")
    print(f"   - 设备: {device}")
    
    # Load relation types
    relation_types_dict = load_relation_types(dataset_name)
    
    # Count asymmetric relations
    asymmetric_count = 0
    for relation_name in relation_names:
        relation_type = get_relation_type(relation_name, relation_types_dict)
        if relation_type == 'Asymmetric':
            asymmetric_count += 1
    
    print(f"   - 发现 {asymmetric_count} 个非对称关系需要语义嵌入")
    print(f"   - 总关系数: {len(relation_names)}")
    
    # Prepare semantic information for asymmetric relations
    inverse_semantic_texts = []
    
    for relation_name in relation_names:
        relation_type = get_relation_type(relation_name, relation_types_dict)
        
        if relation_type == 'Asymmetric':
            # Get inverse relation semantics
            inverse_name, inverse_desc = get_inverse_relation_semantics(dataset_name, relation_name)
            
            # Combine name and description for embedding
            if inverse_name and inverse_desc:
                semantic_text = f"{inverse_name}: {inverse_desc}"
            elif inverse_name:
                semantic_text = inverse_name
            elif inverse_desc:
                semantic_text = inverse_desc
            else:
                # Fallback to original relation name if no inverse semantics available
                semantic_text = relation_name
        else:
            # For symmetric and antisymmetric relations, use original relation name
            semantic_text = relation_name
        
        inverse_semantic_texts.append(semantic_text)
    
    # Generate embeddings using the same model as original relations
    embeddings = get_relation_embeddings(inverse_semantic_texts, model_embed)
    print(f"   - 成功生成 {len(embeddings)} 个逆关系嵌入向量")
    
    return embeddings

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)

def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch

def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask

def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking

def get_relation_embeddings(relation_names, model_embed = None):
    
    device = torch.device(f"cuda:{flags.gpus}" if torch.cuda.is_available() else "cpu")

    if(model_embed == "sentbert"):
        # Load pre-trained language model from local path and ensure it's on the specified device
        # Note: You may need to download all-mpnet-base-v2 model locally first
        # For now, using jina-embeddings-v3 as fallback since it's available locally
        cache_dir = r'/T20030104/ynj/ULTRA/models/jina-embeddings-v3'
        model = AutoModel.from_pretrained(cache_dir, trust_remote_code=True, torch_dtype=torch.float32, local_files_only=True)
        model.to(device)
        # Use jina model for sentence transformer functionality
        raw_embeddings = model.encode(relation_names, task="text-matching", truncate_dim=64)
        
        if isinstance(raw_embeddings, np.ndarray):
            embeddings = torch.from_numpy(raw_embeddings)
        elif isinstance(raw_embeddings, list) and len(raw_embeddings) > 0 and isinstance(raw_embeddings[0], np.ndarray):
            embeddings = torch.from_numpy(np.stack(raw_embeddings))
        elif isinstance(raw_embeddings, torch.Tensor):
            embeddings = raw_embeddings
        else:
            raise TypeError(f"Unexpected embedding type from model.encode: {type(raw_embeddings)}")

    else: # Assuming jinaai or other models
        # Use local model path instead of downloading from huggingface
        cache_dir = r'/T20030104/ynj/ULTRA/models/jina-embeddings-v3'
        model = AutoModel.from_pretrained(cache_dir, trust_remote_code=True, torch_dtype=torch.float32, local_files_only=True)
        model.to(device)
        # model.encode for jinaai typically returns numpy arrays
        raw_embeddings = model.encode(relation_names, task="text-matching", truncate_dim=64)
        
        if isinstance(raw_embeddings, np.ndarray):
            embeddings = torch.from_numpy(raw_embeddings)
        elif isinstance(raw_embeddings, list) and len(raw_embeddings) > 0 and isinstance(raw_embeddings[0], np.ndarray):
            # Handle list of numpy arrays by stacking, e.g. if batch processing in encode returned list
            embeddings = torch.from_numpy(np.stack(raw_embeddings))
        elif isinstance(raw_embeddings, torch.Tensor):
            embeddings = raw_embeddings # Already a tensor
        else:
            raise TypeError(f"Unexpected embedding type from model.encode: {type(raw_embeddings)}")

    # Ensure the final embeddings tensor is on the correct device
    if not isinstance(embeddings, torch.Tensor):
        # This might happen if the model.encode path for 'else' resulted in an unexpected type
        # that wasn't converted to a tensor above.
        # For safety, convert if it's a common type like numpy array, otherwise raise error.
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        else:
            raise TypeError(f"Embeddings are not a PyTorch Tensor before final device placement: {type(embeddings)}")

    if embeddings.device != device:
        embeddings = embeddings.to(device)
        
    return embeddings

def find_top_k_similar_relations(embeddings, k, relation_names):
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    top_k_similarities = []

    for i in range(similarity_matrix.size(0)):
        top_k_with_self = similarity_matrix[i].topk(k + 1)
        indices = top_k_with_self.indices.tolist()  # Indices of top-k+1 similarities
        values = top_k_with_self.values.tolist()    # Similarity values (if needed)

        # Filter out self-similarity (where index == i)
        top_k = [(j, values[indices.index(j)]) for j in indices if j != i]

        # Collect the results
        top_k_similarities.extend([(i, j) for j, _ in top_k])

    return top_k_similarities  # List of (relation_i, relation_j) pairs

def find_top_x_percent(embeddings, relation_names):
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
 
    # Mask the diagonal (self-similarity) by setting it to -inf
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
 
    # Flatten the matrix and get all valid pairs with their similarity
    num_relations = similarity_matrix.size(0)
    total_possible_pairs = (num_relations * (num_relations - 1)) / 2  # Ordered pairs
 
    indices = torch.triu_indices(num_relations, num_relations, offset=1)
    similarities = similarity_matrix[indices[0], indices[1]]
 
    # Calculate number of pairs to select (top X%)
    here = flags.topx / 100
    num_pairs = int(total_possible_pairs * here)
    num_pairs = min(num_pairs, similarities.numel())
 
    top_similarities, top_indices = torch.topk(similarities, num_pairs)
 
    # Convert back to relation index pairs
    top_x_pairs = [(indices[0][i].item(), indices[1][i].item()) for i in top_indices]
    # add the inverse edges to the pairs
    top_x_pairs_inv = [(indices[1][i].item(), indices[0][i].item()) for i in top_indices]
    top_x_pairs.extend(top_x_pairs_inv)
    
    print("====================================")
    print("Number of similar relation pairs: ", len(top_x_pairs))
    
    # Uncomment if you want to print the actual relations
    # for i, (rel1, rel2) in enumerate(top_x_pairs):
    #     if rel1 < num_relations // 2 and rel2 < num_relations // 2:
    #         print(relation_names[rel1], "---", relation_names[rel2])
 
    return top_x_pairs


def compute_ppr_scores(embeddings, alpha=0.85, max_iter=100, tolerance=1e-6):
    # 确保参数类型正确
    alpha = float(alpha)
    max_iter = int(max_iter)
    tolerance = float(tolerance)
    """
    使用Personalized PageRank (PPR)计算关系的重要性得分
    基于关系嵌入的余弦相似度构建邻接矩阵，然后使用PPR算法计算每个关系的重要性
    
    Args:
        embeddings: 关系嵌入张量 [num_relations, embedding_dim]
        alpha: PPR的阻尼系数，控制随机游走的概率
        max_iter: 最大迭代次数
        tolerance: 收敛容忍度
    
    Returns:
        ppr_scores: 每个关系的PPR得分 [num_relations]
    """
    device = embeddings.device
    num_relations = embeddings.size(0)
    
    # 计算余弦相似度矩阵
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    # 将对角线设为0（自相似度）
    mask = torch.eye(num_relations, device=device).bool()
    similarity_matrix.masked_fill_(mask, 0.0)
    
    # 将相似度矩阵转换为邻接矩阵（保留正相似度）
    adjacency_matrix = torch.where(similarity_matrix > 0, similarity_matrix, torch.zeros_like(similarity_matrix))
    
    # 计算度矩阵（每个节点的出度）
    degree_matrix = torch.sum(adjacency_matrix, dim=1)
    
    # 避免除零错误，将度数为0的节点设为1
    degree_matrix = torch.where(degree_matrix > 0, degree_matrix, torch.ones_like(degree_matrix))
    
    # 构建转移概率矩阵 P = D^(-1) * A
    transition_matrix = adjacency_matrix / degree_matrix.unsqueeze(1)
    
    # 初始化PPR得分向量（均匀分布）
    ppr_scores = torch.ones(num_relations, device=device) / num_relations
    
    # PPR迭代更新
    for iteration in range(max_iter):
        # 保存上一次的得分
        prev_scores = ppr_scores.clone()
        
        # PPR更新公式: p = α * s + (1-α) * P^T * p
        # 这里s是均匀分布，P^T是转移矩阵的转置
        uniform_restart = torch.ones(num_relations, device=device) / num_relations
        ppr_scores = alpha * uniform_restart + (1 - alpha) * torch.matmul(transition_matrix.T, ppr_scores)
        
        # 检查收敛
        if torch.norm(ppr_scores - prev_scores) < tolerance:
            print(f"PPR converged after {iteration + 1} iterations")
            break
    
    return ppr_scores

def find_pairs_with_ppr(embeddings, relation_names, top_k_ratio=0.1):
    """
    使用PPR得分来选择关系对，替代固定阈值方法
    
    Args:
        embeddings: 关系嵌入张量
        relation_names: 关系名称列表
        top_k_ratio: 选择top-k比例的关系对
    
    Returns:
        selected_pairs: 选中的关系对列表
    """
    device = embeddings.device
    num_relations = embeddings.size(0)
    
    # 计算PPR得分
    ppr_scores = compute_ppr_scores(embeddings, 
                                  alpha=float(flags.ppr_alpha), 
                                  max_iter=int(flags.ppr_max_iter), 
                                  tolerance=float(flags.ppr_tolerance))
    
    # 计算余弦相似度矩阵
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    # 将对角线设为-inf（避免自相似度）
    mask = torch.eye(num_relations, device=device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    # 计算关系对的综合得分：结合PPR得分和余弦相似度
    # 对于关系对(i,j)，综合得分 = PPR_i * PPR_j * similarity_ij
    ppr_weighted_similarity = ppr_scores.unsqueeze(1) * ppr_scores.unsqueeze(0) * similarity_matrix
    
    # 获取上三角矩阵的索引（避免重复计算）
    indices = torch.triu_indices(num_relations, num_relations, offset=1)
    upper_tri_scores = ppr_weighted_similarity[indices[0], indices[1]]
    
    # 选择top-k比例的关系对
    num_pairs_to_select = max(1, int(len(upper_tri_scores) * top_k_ratio))
    _, top_indices = torch.topk(upper_tri_scores, num_pairs_to_select)
    
    # 将索引转换回关系对
    selected_pairs = []
    for idx in top_indices:
        row_idx = indices[0][idx].item()
        col_idx = indices[1][idx].item()
        selected_pairs.append((row_idx, col_idx))
    
    print("====================================")
    print(f"PPR-based relation pair selection:")
    print(f"  - Total possible pairs: {len(upper_tri_scores)}")
    print(f"  - Selected pairs: {len(selected_pairs)}")
    print(f"  - Selection ratio: {len(selected_pairs) / len(upper_tri_scores):.4f}")
    print(f"  - PPR alpha: {flags.ppr_alpha}")
    print(f"  - PPR max iterations: {flags.ppr_max_iter}")
    
    return selected_pairs


def find_pairs_above_threshold(embeddings, relation_names):
    # Check if PPR-based threshold is enabled
    if hasattr(flags, 'use_ppr_threshold') and flags.use_ppr_threshold:
        # Use PPR-based dynamic selection
        return find_pairs_with_ppr(embeddings, relation_names, top_k_ratio=0.1)
    
    # Original fixed threshold approach
    # Compute the cosine similarity matrix
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # Mask the diagonal (self-similarity)
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    # Use fixed threshold
    threshold = flags.threshold
    
    num_relations = similarity_matrix.size(0)
    row_indices, col_indices = torch.where(similarity_matrix > threshold)
    
    # Convert to relation index pairs
    selected_pairs = [(row.item(), col.item()) for row, col in zip(row_indices, col_indices)]
    
    print("====================================")
    print(f"Number of relation pairs with cosine similarity > {threshold:.4f}: {len(selected_pairs)}")
    # Uncomment if you want to print the actual relations
    # for row, col in zip(row_indices, col_indices):
    #     rel1, rel2 = row.item(), col.item()
    #     if(rel1 < num_relations // 2 and rel2 < num_relations // 2):
    #         sim_value = similarity_matrix[rel1, rel2].item()
    #         print(relation_names[rel1], "---", relation_names[rel2], "---", sim_value)

    return selected_pairs

def load_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def create_harder_rg1(graph, original_rel_graph, test_relation, inverse_relation, head=None, tail=None):
    
    device = graph.edge_index.device
    num_rels = graph.num_relations
    
    # Copy the original relation graph edges
    new_edge_index = original_rel_graph.edge_index.clone()
    new_edge_type = original_rel_graph.edge_type.clone()
    
    test_rel_node = test_relation  # New node ID
    inverse_rel_node = inverse_relation  # New node ID
    
    # Lists to collect new edges and types
    new_edges = []
    new_types = []
    
    # Case 1: Only head is known
    if head is not None and tail is None:
        # Find all relations where this entity appears as head (h2h connections)
        head_mask = (graph.edge_index[0] == head)
        head_relations = graph.edge_type[head_mask].unique()
        
        # Add h2h connections
        for rel in head_relations:
            if rel != test_relation:  # Avoid self-loops
                new_edges.append([test_rel_node, rel.item()])
                new_types.append(0)  # h2h type
                
                # Reverse connection: rel -> test_relation (h2h)
                new_edges.append([rel.item(), test_rel_node])
                new_types.append(0)  # h2h type
        
        # Find all relations where this entity appears as tail (t2h connections)
        tail_mask = (graph.edge_index[1] == head)
        tail_relations = graph.edge_type[tail_mask].unique()
        
        # Add t2h connections
        for rel in tail_relations:
            new_edges.append([test_rel_node, rel.item()])
            new_types.append(2)  # h2t type
            
            # Reverse connection: rel -> test_relation (t2h)
            new_edges.append([rel.item(), test_rel_node])
            new_types.append(3)  # t2h type

        # For the inverse relation (tail is known for the inverse)
        # For inverse relation, the head in original becomes tail
        # t2h connections for inverse relation
        for rel in head_relations:
            new_edges.append([inverse_rel_node, rel.item()])
            new_types.append(3)  # t2h type
            
            new_edges.append([rel.item(), inverse_rel_node])
            new_types.append(2)  # h2t type
        
        # t2t connections for inverse relation
        for rel in tail_relations:
            if rel != inverse_relation:  # Avoid self-loops
                new_edges.append([inverse_rel_node, rel.item()])
                new_types.append(1)  # t2t type
                
                # Reverse connection: rel -> inverse_relation (h2t)
                new_edges.append([rel.item(), inverse_rel_node])
                new_types.append(1)  # t2t type
        
        # Connect test relation with its inverse
        new_edges.append([test_rel_node, inverse_rel_node])
        new_types.append(2)  # h2t type - original to inverse
        
        new_edges.append([inverse_rel_node, test_rel_node])
        new_types.append(3)  # t2h type - inverse to original
    
    # Case 2: Only tail is known
    elif tail is not None and head is None:
        
        tail_mask = (graph.edge_index[1] == tail)
        tail_relations = graph.edge_type[tail_mask].unique()
        
        # Add t2t connections
        for rel in tail_relations:
            if rel != test_relation:  # Avoid self-loops
                new_edges.append([test_rel_node, rel.item()])
                new_types.append(1)  # t2t type
                
                # Reverse connection: rel -> test_relation (t2t)
                new_edges.append([rel.item(), test_rel_node])
                new_types.append(1)  # t2t type
        
        # Find all relations where this entity appears as head (h2t connections)
        head_mask = (graph.edge_index[0] == tail)
        head_relations = graph.edge_type[head_mask].unique()
        
        # Add h2t connections
        for rel in head_relations:
            new_edges.append([test_rel_node, rel.item()])
            new_types.append(3)  # t2h type
            
            # Reverse connection: rel -> test_relation (h2t)
            new_edges.append([rel.item(), test_rel_node])
            new_types.append(2)  # h2h type
    
        # For the inverse relation (head is known for the inverse)
        # For inverse relation, the tail in original becomes head
        # h2h connections for inverse relation
        for rel in tail_relations:
            new_edges.append([inverse_rel_node, rel.item()])
            new_types.append(2)  # h2t type
            
            # Reverse connection: rel -> inverse_relation (h2h)
            new_edges.append([rel.item(), inverse_rel_node])
            new_types.append(3)  # t2h type
        
        # t2h connections for inverse relation
        for rel in head_relations:
            if rel != inverse_relation:  # Avoid self-loops
                new_edges.append([inverse_rel_node, rel.item()])
                new_types.append(0)  # h2h type
            
                # Reverse connection: rel -> inverse_relation (t2h)
                new_edges.append([rel.item(), inverse_rel_node])
                new_types.append(0)  # h2h type
        
        # Connect test relation with its inverse
        new_edges.append([test_rel_node, inverse_rel_node])
        new_types.append(2)  # h2t type - original to inverse
        
        new_edges.append([inverse_rel_node, test_rel_node])
        new_types.append(3)  # t2h type - inverse to original
    
    # Convert to tensors if there are new edges
    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).T
        new_type_tensor = torch.tensor(new_types, dtype=torch.long, device=device)
        
        # Add to existing edges
        new_edge_index = torch.cat([new_edge_index, new_edge_tensor], dim=1)
        new_edge_type = torch.cat([new_edge_type, new_type_tensor])
    
    # Create the new relation graph
    test_rel_graph = Data(
        edge_index=new_edge_index,
        edge_type=new_edge_type,
        num_nodes=num_rels,  # Original relations + test relation + inverse test relation
        num_relations=4  # Same relation types as before
    )

    return test_rel_graph

def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head

    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph

    if(flags.harder_setting == True):
        if hasattr(graph, "is_harder") and graph.is_harder == True:
            graph.harder_head_rg1 = {}
            graph.harder_tail_rg1 = {}
            for i in range(graph.target_edge_index.size(1)):
                head = graph.target_edge_index[0, i].item()
                relation = num_rels // 2 - 1
                tail = graph.target_edge_index[1, i].item()

                inverse_relation = relation + (num_rels // 2) 
                
                # Create a new relation graph with the test relation as a new node
                # For head known case
                head_known_rel_graph = create_harder_rg1(
                    graph, rel_graph, relation, inverse_relation, head=head, tail=None
                )
                
                # For tail known case
                tail_known_rel_graph = create_harder_rg1(
                    graph, rel_graph, relation, inverse_relation, head=None, tail=tail
                )
                
                # Store these relation graphs
                graph.harder_head_rg1[(head, graph.target_edge_type[i].item())] = head_known_rel_graph
                graph.harder_tail_rg1[(tail, graph.target_edge_type[i].item())] = tail_known_rel_graph
    
    return graph

def order_embeddings(embeddings, relation_names, graph, num_rels, inv_embeddings = None, dataset_name = None):
    """
    Order embeddings for both original and inverse relations based on relation types.
    
    Args:
        embeddings: Original relation embeddings
        relation_names: List of relation names
        graph: Graph object
        num_rels: Number of relations
        inv_embeddings: Pre-computed inverse embeddings (optional)
        dataset_name: Name of the dataset for relation type classification
    
    Returns:
        List of ordered embeddings for both original and inverse relations
    """
    # Check if inverse relation classification is enabled
    if hasattr(flags, 'is-inverse-relation-classify') and getattr(flags, 'is-inverse-relation-classify', False) and dataset_name:
        print(f"🔍 [逆关系嵌入] 使用语义嵌入方式生成逆关系嵌入")
        print(f"   - 数据集: {dataset_name}")
        print(f"   - 关系类型分类: 已启用")
        print(f"   - 将为非对称关系生成基于语义的逆关系嵌入")
        return order_embeddings_with_classification(embeddings, relation_names, graph, num_rels, dataset_name)
    else:
        print(f"🔄 [逆关系嵌入] 使用直接翻转方式生成逆关系嵌入")
        print(f"   - 数据集: {dataset_name if dataset_name else '未知'}")
        print(f"   - 关系类型分类: 已禁用")
        print(f"   - 所有逆关系将使用直接翻转 (-1 * 原关系)")
        # Original logic
        return order_embeddings_original(embeddings, relation_names, graph, num_rels, inv_embeddings)

def order_embeddings_original(embeddings, relation_names, graph, num_rels, inv_embeddings = None):
    """Original embedding ordering logic."""
    print(f"📝 [原始逻辑] 使用直接翻转方式处理 {len(embeddings)} 个关系")
    print(f"   - 所有逆关系将使用: 逆关系 = -1 * 原关系")
    print(f"   - 预计算逆关系嵌入: {'是' if inv_embeddings is not None else '否'}")
    
    ordered_embeddings = {}
    for i in range(len(embeddings)):
        if(relation_names[i] in graph.edge2id):
            ordered_embeddings[graph.edge2id[relation_names[i]]] = embeddings[i]
            if inv_embeddings is not None:
                ordered_embeddings[graph.edge2id[relation_names[i]] + len(graph.edge2id)] = inv_embeddings[i]
            else:
                ordered_embeddings[graph.edge2id[relation_names[i]] + len(graph.edge2id)] = -embeddings[i]

    embeddings = []
    for i in range(2*len(graph.edge2id)):
        if i in ordered_embeddings:
            embeddings.append(ordered_embeddings[i])
        else:
            # 如果索引不存在，使用零向量或默认值
            # 这里使用与embeddings[0]相同维度的零向量
            if len(embeddings) > 0:
                embeddings.append(torch.zeros_like(embeddings[0]))
            else:
                # 如果这是第一个元素且缺失，创建一个默认的零向量
                # 假设embedding维度为64（根据代码中的truncate_dim=64）
                embeddings.append(torch.zeros(64))

    return embeddings

def order_embeddings_with_classification(embeddings, relation_names, graph, num_rels, dataset_name):
    """
    Order embeddings based on relation type classification.
    
    Strategy:
    - Symmetric relations: inverse embedding = original embedding
    - Antisymmetric relations: inverse embedding = -original embedding  
    - Asymmetric relations: inverse embedding = semantic-based embedding
    """
    # Load relation types
    relation_types_dict = load_relation_types(dataset_name)
    
    # Count relation types for statistics
    type_counts = {'Symmetric': 0, 'Antisymmetric': 0, 'Asymmetric': 0, 'Unknown': 0}
    for relation_name in relation_names:
        if relation_name in graph.edge2id:
            relation_type = get_relation_type(relation_name, relation_types_dict)
            if relation_type in type_counts:
                type_counts[relation_type] += 1
            else:
                type_counts['Unknown'] += 1
    
    print(f"📊 [关系类型分布统计]")
    print(f"   - 对称关系: {type_counts['Symmetric']} 个 (逆关系 = 原关系)")
    print(f"   - 反对称关系: {type_counts['Antisymmetric']} 个 (逆关系 = -原关系)")
    print(f"   - 非对称关系: {type_counts['Asymmetric']} 个 (逆关系 = 语义嵌入)")
    print(f"   - 未知类型: {type_counts['Unknown']} 个 (默认为非对称)")
    
    # Generate inverse embeddings for asymmetric relations using semantic information
    asymmetric_inverse_embeddings = generate_inverse_embeddings_for_asymmetric(
        relation_names, dataset_name, flags.model_embed
    )
    
    ordered_embeddings = {}
    for i in range(len(embeddings)):
        if(relation_names[i] in graph.edge2id):
            relation_name = relation_names[i]
            relation_type = get_relation_type(relation_name, relation_types_dict)
            
            # Store original embedding
            ordered_embeddings[graph.edge2id[relation_name]] = embeddings[i]
            
            # Generate inverse embedding based on relation type
            if relation_type == 'Symmetric':
                # Symmetric: inverse embedding = original embedding
                inverse_embedding = embeddings[i]
            elif relation_type == 'Antisymmetric':
                # Antisymmetric: inverse embedding = -original embedding
                inverse_embedding = -embeddings[i]
            else:  # Asymmetric
                # Asymmetric: use semantic-based embedding
                inverse_embedding = asymmetric_inverse_embeddings[i]
            
            ordered_embeddings[graph.edge2id[relation_name] + len(graph.edge2id)] = inverse_embedding

    # Create final embeddings list
    final_embeddings = []
    for i in range(2*len(graph.edge2id)):
        if i in ordered_embeddings:
            final_embeddings.append(ordered_embeddings[i])
        else:
            # 如果索引不存在，使用零向量或默认值
            if len(final_embeddings) > 0:
                final_embeddings.append(torch.zeros_like(final_embeddings[0]))
            else:
                # 如果这是第一个元素且缺失，创建一个默认的零向量
                # 假设embedding维度为64（根据代码中的truncate_dim=64）
                final_embeddings.append(torch.zeros(64))

    return final_embeddings

def build_relation_graph_exp(graph, dataset_name=None):
    # Extract existing edge indices and types
    k = flags.k
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head

    # Step 4: Add fifth edge type for semantically similar relations
    # Generate embeddings for relation names based on the edge2id dictionary keys

    if(flags.run == "semma"):
        rel_graph = Data(
            edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
            edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
            num_nodes=num_rels, 
            num_relations=4
        )
        graph.relation_graph = rel_graph

        if(flags.harder_setting == True):
            if hasattr(graph, "is_harder") and graph.is_harder == True:
                graph.harder_head_rg1 = {}
                graph.harder_tail_rg1 = {}
                for i in range(graph.target_edge_index.size(1)):
                    head = graph.target_edge_index[0, i].item()
                    relation = num_rels // 2 - 1
                    tail = graph.target_edge_index[1, i].item()

                    inverse_relation = relation + (num_rels // 2) 
                    
                    # Create a new relation graph with the test relation as a new node
                    # For head known case
                    head_known_rel_graph = create_harder_rg1(
                        graph, rel_graph, relation, inverse_relation, head=head, tail=None
                    )
                    
                    # For tail known case
                    tail_known_rel_graph = create_harder_rg1(
                        graph, rel_graph, relation, inverse_relation, head=None, tail=tail
                    )
                    
                    # Store these relation graphs
                    graph.harder_head_rg1[(head, graph.target_edge_type[i].item())] = head_known_rel_graph
                    graph.harder_tail_rg1[(tail, graph.target_edge_type[i].item())] = tail_known_rel_graph
    
    # 确保dataset_name正确设置
    if dataset_name is None and hasattr(graph, 'dataset'):
        dataset_name = graph.dataset
    elif dataset_name is None:
        dataset_name = "Unknown"
    
    print(f"🔍 [调试信息] dataset_name: {dataset_name}")
    
    file_path = None

    if flags.LLM == "gpt4o":
        file_path = os.path.join(mydir, "openrouter/descriptions/gpt-4o-2024-11-20", graph.dataset + ".json")
    elif flags.LLM == "qwen3-32b":
        file_path = os.path.join(mydir, "openrouter/descriptions/qwen3-32b", graph.dataset + ".json")
    elif flags.LLM == "deepseekv3":
        file_path = os.path.join(mydir, "openrouter/descriptions/deepseek-chat-v3-0324", graph.dataset + ".json")

    if file_path is not None:
        dict = load_file(file_path)
    else:
        raise ValueError("LLM not supported")

    cleaned_relations = dict['cleaned_relations']
    relation_descriptions = dict['relation_descriptions']

    relation_names = list(cleaned_relations.keys())

    cleaned_relations = list(cleaned_relations.values())
    relation_descriptions = list(relation_descriptions.values())
    # relation_descriptions[i][0] -> description of relation i and [i][1] -> description of inverse of relation i
    rel_desc = []
    for i in range(len(relation_descriptions)):
        rel_desc.append(relation_descriptions[i][0])
    inv_desc = []
    for i in range(len(relation_descriptions)):
        inv_desc.append(relation_descriptions[i][1])

    # traverse through edge_type and get the relation names

    if(flags.rg2_embedding == "combined"):
        embeddings_nollm = get_relation_embeddings(relation_names, flags.model_embed)
        embeddings_llm_name = get_relation_embeddings(cleaned_relations, flags.model_embed)
        embeddings_llm_desc = get_relation_embeddings(rel_desc, flags.model_embed)
        embeddings_llm_inv_desc = get_relation_embeddings(inv_desc, flags.model_embed)
        embeddings_nollm = order_embeddings(embeddings_nollm, relation_names, graph, num_rels, dataset_name=dataset_name)
        embeddings_llm_name = order_embeddings(embeddings_llm_name, relation_names, graph, num_rels, dataset_name=dataset_name)
        embeddings_llm_desc = order_embeddings(embeddings_llm_desc, relation_names, graph, num_rels, embeddings_llm_inv_desc, dataset_name=dataset_name)
        embeddings = [
            (a + b + c) / 3
            for a, b, c in zip(embeddings_nollm, embeddings_llm_name, embeddings_llm_desc)
        ]

    elif(flags.rg2_embedding == "combined-sum"):
        embeddings_nollm = get_relation_embeddings(relation_names, flags.model_embed)
        embeddings_llm_name = get_relation_embeddings(cleaned_relations, flags.model_embed)
        embeddings_llm_desc = get_relation_embeddings(rel_desc, flags.model_embed)
        embeddings_llm_inv_desc = get_relation_embeddings(inv_desc, flags.model_embed)
        embeddings_nollm = order_embeddings(embeddings_nollm, relation_names, graph, num_rels, dataset_name=dataset_name)
        embeddings_llm_name = order_embeddings(embeddings_llm_name, relation_names, graph, num_rels, dataset_name=dataset_name)
        embeddings_llm_desc = order_embeddings(embeddings_llm_desc, relation_names, graph, num_rels, embeddings_llm_inv_desc, dataset_name=dataset_name)
        embeddings = [
            (a + b + c)
            for a, b, c in zip(embeddings_nollm, embeddings_llm_name, embeddings_llm_desc)
        ]

    elif(flags.rg2_embedding == "no llm"):
        embeddings = get_relation_embeddings(relation_names, flags.model_embed)
        embeddings = order_embeddings(embeddings, relation_names, graph, num_rels, dataset_name=dataset_name)
    elif(flags.rg2_embedding == "llm name"):
        embeddings = get_relation_embeddings(cleaned_relations, flags.model_embed)
        embeddings = order_embeddings(embeddings, relation_names, graph, num_rels, dataset_name=dataset_name)
    elif(flags.rg2_embedding == "llm description"):
        embeddings = get_relation_embeddings(rel_desc, flags.model_embed) # (num_relations, embedding size)
        inv_embeddings = get_relation_embeddings(inv_desc, flags.model_embed)
        embeddings = order_embeddings(embeddings, relation_names, graph, num_rels, inv_embeddings, dataset_name=dataset_name)

    if flags.harder_setting == True:
        if hasattr(graph, "is_harder") and graph.is_harder == True:            
            graph.harder_head_rg2 = {}
            graph.harder_tail_rg2 = {}
            for i in range(graph.target_edge_index.size(1)): 
                head = graph.target_edge_index[0, i].item()
                relation = num_rels // 2 - 1
                tail = graph.target_edge_index[1, i].item()

                inverse_relation = relation + (num_rels // 2)

                embeddings_here = []
                for j in range(num_rels // 2 - 1):
                    embeddings_here.append(embeddings[j])
                embeddings_here.append(embeddings[graph.target_edge_type[i].item()])
                for j in range(num_rels // 2 - 1):
                    embeddings_here.append(embeddings[j + len(graph.edge2id)])
                embeddings_here.append(embeddings[graph.target_edge_type[i].item() + len(graph.edge2id)])
                
                embeddings_here = torch.stack(embeddings_here).to(device)
                relation_similarity_matrix = F.cosine_similarity(embeddings_here.unsqueeze(1), embeddings_here.unsqueeze(0), dim=2)
                
                if(flags.k != 0):
                    similar_pairs = find_top_k_similar_relations(embeddings_here, k, relation_names)
                elif(flags.topx != 0):
                    similar_pairs = find_top_x_percent(embeddings_here, relation_names)
                else:
                    similar_pairs = find_pairs_above_threshold(embeddings_here, relation_names)

                if len(similar_pairs) == 0:
                    sem_sim_edges = torch.empty((0, 2), dtype=torch.long, device=device)
                    sem_sim_type = torch.empty((0,), dtype=torch.long, device=device)
                else:
                    # Add new edges for semantically similar relations with type 0
                    sem_sim_edges = torch.tensor(similar_pairs, dtype=torch.long, device=device)
                    sem_sim_type = torch.full((sem_sim_edges.shape[0],), 0, dtype=torch.long, device=device)
                    
                if flags.use_cos_sim_weights:
                    rel_graph2_here = Data(
                        edge_index=sem_sim_edges.T,
                        edge_type=sem_sim_type,
                        num_nodes=num_rels,
                        num_relations=1,  
                        relation_embeddings = embeddings_here,
                        relation_similarity_matrix = relation_similarity_matrix
                    )
                else:
                    rel_graph2_here = Data(
                        edge_index=sem_sim_edges.T,
                        edge_type=sem_sim_type,
                        num_nodes=num_rels,
                        num_relations=1, 
                        relation_embeddings = embeddings_here
                    )
                graph.harder_head_rg2[(head, graph.target_edge_type[i].item())] = rel_graph2_here
                graph.harder_tail_rg2[(tail, graph.target_edge_type[i].item())] = rel_graph2_here
            
    embeddings = torch.stack(embeddings).to(device)
    # Calculate full cosine similarity matrix
    relation_similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

    # Find top-k similar relations
    if(flags.k != 0):
        similar_pairs = find_top_k_similar_relations(embeddings, k, relation_names)
    elif(flags.topx != 0):
        similar_pairs = find_top_x_percent(embeddings, relation_names)
    else:
        similar_pairs = find_pairs_above_threshold(embeddings, relation_names)

    if len(similar_pairs) == 0:
        sem_sim_edges = torch.empty((0, 2), dtype=torch.long, device=device)
        sem_sim_type = torch.empty((0,), dtype=torch.long, device=device)
    else:
        # Add new edges for semantically similar relations with type 0
        sem_sim_edges = torch.tensor(similar_pairs, dtype=torch.long, device=device)
        sem_sim_type = torch.full((sem_sim_edges.shape[0],), 0, dtype=torch.long, device=device)

    if flags.use_cos_sim_weights:
        rel_graph2 = Data(
            edge_index=sem_sim_edges.T,
            edge_type=sem_sim_type,
            num_nodes=num_rels,
            num_relations=1,  # Updated for 5 relation types
            relation_embeddings = embeddings,
            relation_similarity_matrix = relation_similarity_matrix
        )
    else:
        rel_graph2 = Data(
            edge_index=sem_sim_edges.T,
            edge_type=sem_sim_type,
            num_nodes=num_rels,
            num_relations=1,  # Updated for 5 relation types
            relation_embeddings = embeddings
        )
    graph.relation_graph2 = rel_graph2

    return graph