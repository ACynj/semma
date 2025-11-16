import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import logging
import warnings
import time
from contextlib import contextmanager
import numpy as np
from collections import defaultdict


# 性能分析工具
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


class EntityRelationJointEnhancer(nn.Module):
    """
    实体-关系联合增强模块（方案3）
    同时增强实体和关系，使用实体-关系交互信息
    """
    
    def __init__(self, embedding_dim=64):
        super(EntityRelationJointEnhancer, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 实体-关系交互网络
        # 输入：实体特征 + 关系特征
        self.entity_relation_interaction = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 实体上下文聚合网络
        # 用于聚合实体的邻居信息
        self.entity_context_aggregator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 实体增强强度控制（可学习）
        self.enhancement_strength = nn.Parameter(torch.tensor(0.1))
        
        # 缓存邻接表（避免重复构建）
        self._adj_list_cache = None
        self._cached_edge_index = None
        self._cached_edge_index_hash = None
    
    def _get_k_hop_neighbors(self, data, seed_entities, max_hops=1):
        """
        获取实体的k跳邻居（包括实体本身）
        
        Args:
            data: 图数据
            seed_entities: 种子实体集合（set或list）
            max_hops: 最大跳数（默认1跳）
        
        Returns:
            relevant_entities: 相关实体集合（包括种子实体和k跳邻居）
        """
        if data.edge_index.numel() == 0:
            return set(seed_entities)
        
        relevant_entities = set(seed_entities)
        current_frontier = set(seed_entities)
        
        logger = logging.getLogger(__name__)
        
        # 检查是否可以复用缓存的邻接表
        edge_index_hash = hash((data.edge_index.shape[0], data.edge_index.shape[1], 
                               data.edge_index.data_ptr() if hasattr(data.edge_index, 'data_ptr') else id(data.edge_index)))
        
        if (self._adj_list_cache is not None and 
            self._cached_edge_index_hash == edge_index_hash and
            self._cached_edge_index is not None and
            self._cached_edge_index.shape == data.edge_index.shape):
            # 快速检查：只比较前几个元素（避免完整比较）
            if torch.equal(self._cached_edge_index[:2, :min(100, data.edge_index.shape[1])], 
                          data.edge_index[:2, :min(100, data.edge_index.shape[1])]):
                # 复用缓存的邻接表
                adj_list = self._adj_list_cache
                logger.debug(f"[Entity Enhancer] 复用缓存的邻接表 (边数={data.edge_index.shape[1]})")
            else:
                adj_list = None
        else:
            adj_list = None
        
        # 如果需要构建新的邻接表
        if adj_list is None:
            with timer(f"[Entity Enhancer] 构建邻接表 (边数={data.edge_index.shape[1]})", logger):
                # 使用numpy向量化操作（比Python循环快很多）
                edge_index_np = data.edge_index.cpu().numpy()
                src_nodes = edge_index_np[0]
                dst_nodes = edge_index_np[1]
                
                # 使用defaultdict提高效率
                adj_list = defaultdict(set)
                
                # 向量化构建双向邻接表
                for src, dst in zip(src_nodes, dst_nodes):
                    adj_list[src].add(dst)
                    adj_list[dst].add(src)  # 无向图
                
                # 转换为普通dict
                adj_list = dict(adj_list)
                
                # 缓存邻接表和边索引
                self._adj_list_cache = adj_list
                self._cached_edge_index = data.edge_index.clone()
                self._cached_edge_index_hash = edge_index_hash
        
        # BFS遍历k跳（优化：使用批量集合操作）
        with timer(f"[Entity Enhancer] BFS遍历{max_hops}跳 (种子数={len(seed_entities)})", logger):
            for hop in range(max_hops):
                # 批量获取当前frontier的所有邻居
                all_neighbors = set()
                for entity in current_frontier:
                    if entity in adj_list:
                        all_neighbors.update(adj_list[entity])
                
                # 批量添加新邻居（使用集合差集操作，比逐个检查快）
                new_neighbors = all_neighbors - relevant_entities
                relevant_entities.update(new_neighbors)
                current_frontier = new_neighbors
                
                if not current_frontier:  # 如果没有新邻居，提前结束
                    break
        
        return relevant_entities
    
    def compute_entity_relation_features(self, entity_id, data, relation_embeddings):
        """
        计算实体-关系联合特征
        
        Args:
            entity_id: 实体ID
            data: 图数据
            relation_embeddings: 关系嵌入矩阵 [num_relations, embedding_dim]
        
        Returns:
            entity_relation_feat: 实体-关系联合特征 [embedding_dim]
        """
        device = relation_embeddings.device
        
        # 1. 获取实体相关的所有关系
        # 性能优化：使用向量化操作而不是循环
        entity_edge_mask = (data.edge_index[0] == entity_id) | (data.edge_index[1] == entity_id)
        
        if entity_edge_mask.any():
            # 获取这些边的关系类型
            entity_rels = data.edge_type[entity_edge_mask]
            # 获取这些关系的嵌入
            valid_rels = entity_rels[entity_rels < relation_embeddings.shape[0]]
            
            if len(valid_rels) > 0:
                # 实体相关的所有关系的平均嵌入
                # 确保valid_rels是张量且不为空
                if isinstance(valid_rels, torch.Tensor) and valid_rels.numel() > 0:
                    if valid_rels.dim() == 0:  # 标量
                        entity_relation_avg = relation_embeddings[valid_rels.item()].unsqueeze(0)
                    else:
                        entity_relation_avg = relation_embeddings[valid_rels].mean(dim=0)  # [embedding_dim]
                else:
                    entity_relation_avg = relation_embeddings.mean(dim=0)
                
                # 2. 计算实体度（连接的边数）作为权重
                entity_degree = len(valid_rels)
                degree_weight = torch.sigmoid(torch.tensor(entity_degree / 10.0, device=device))  # 归一化到0-1
                
                # 3. 实体-关系交互特征
                # 使用实体相关的所有关系的平均嵌入作为实体特征
                entity_feat = entity_relation_avg
                # 使用全局关系的平均嵌入作为关系上下文
                relation_context = relation_embeddings.mean(dim=0)
                
                # 拼接实体特征和关系上下文
                combined = torch.cat([entity_feat, relation_context], dim=-1)  # [embedding_dim * 2]
                
                # 通过交互网络
                interaction_feat = self.entity_relation_interaction(combined)  # [embedding_dim]
                
                # 4. 实体上下文聚合（简化版：直接使用实体自己的关系嵌入作为上下文，避免查找邻居）
                # 优化：不再查找邻居的边，直接使用实体自己的关系嵌入作为上下文
                # 这样可以避免O(实体数 * 邻居数 * 边查找)的复杂度，大幅提升计算速度
                # 获取邻居实体集合（仅用于判断是否有邻居，不用于查找邻居的边）
                neighbor_entities = set()
                if entity_edge_mask.any():
                    connected_edges = data.edge_index[:, entity_edge_mask]
                    neighbor_entities.update(connected_edges[0].tolist())
                    neighbor_entities.update(connected_edges[1].tolist())
                    neighbor_entities.discard(entity_id)  # 移除自己
                
                if len(neighbor_entities) > 0:
                    # 简化：直接使用实体自己的关系嵌入作为邻居上下文（避免重复查找）
                    # 邻居上下文 = 实体自己的关系嵌入（已经包含了邻居信息）
                    neighbor_context = entity_relation_avg  # 使用实体自己的关系嵌入
                    # 结合实体特征和邻居上下文
                    context_combined = torch.cat([entity_feat, neighbor_context], dim=-1)
                    context_feat = self.entity_context_aggregator(context_combined)  # [embedding_dim]
                else:
                    context_feat = interaction_feat
                
                # 5. 加权融合
                final_feat = (1.0 - torch.clamp(self.enhancement_strength, 0, 0.3)) * entity_feat + \
                            torch.clamp(self.enhancement_strength, 0, 0.3) * context_feat
                
                return final_feat
            else:
                # 回退：使用所有关系的平均嵌入
                return relation_embeddings.mean(dim=0)
        else:
            # 回退：如果实体没有边，使用所有关系的平均嵌入
            return relation_embeddings.mean(dim=0)
    
    def _get_k_hop_neighbors(self, data, seed_entities, max_hops=1):
        """
        获取实体的k跳邻居（包括实体本身）
        
        Args:
            data: 图数据
            seed_entities: 种子实体集合（set或list）
            max_hops: 最大跳数（默认1跳）
        
        Returns:
            relevant_entities: 相关实体集合（包括种子实体和k跳邻居）
        """
        if data.edge_index.numel() == 0:
            return set(seed_entities)
        
        relevant_entities = set(seed_entities)
        current_frontier = set(seed_entities)
        
        # 构建邻接表（双向）
        adj_list = {}
        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            if src not in adj_list:
                adj_list[src] = set()
            if dst not in adj_list:
                adj_list[dst] = set()
            adj_list[src].add(dst)
            adj_list[dst].add(src)  # 无向图
        
        # BFS遍历k跳
        for hop in range(max_hops):
            next_frontier = set()
            for entity in current_frontier:
                if entity in adj_list:
                    neighbors = adj_list[entity]
                    for neighbor in neighbors:
                        if neighbor not in relevant_entities:
                            relevant_entities.add(neighbor)
                            next_frontier.add(neighbor)
            current_frontier = next_frontier
            if not current_frontier:  # 如果没有新邻居，提前结束
                break
        
        return relevant_entities
    
    def compute_enhanced_boundary(self, data, h_index, r_index, relation_representations, prompt_entities=None):
        """
        计算增强的boundary条件（为所有实体提供初始特征）
        优化：优先使用提示图中的实体（如果提供），否则计算1跳邻居
        
        Args:
            data: 图数据
            h_index: 源实体索引 [batch_size]
            r_index: 关系索引 [batch_size]
            relation_representations: 关系表示 [batch_size, num_relations, embedding_dim] 或 [num_relations, embedding_dim]
            prompt_entities: 提示图中的实体集合（可选），如果提供则优先使用，避免重复计算
        
        Returns:
            enhanced_boundary: 增强的boundary条件 [batch_size, num_nodes, embedding_dim]
        """
        batch_size = len(h_index)
        device = h_index.device
        
        # 处理relation_representations的维度
        if relation_representations.dim() == 3:
            # [batch_size, num_relations, embedding_dim]
            # 对于batch中的每个样本，使用对应的关系表示
            # 但为了效率，我们使用第一个batch的关系嵌入（所有batch共享图结构）
            relation_embeddings = relation_representations[0]  # 使用第一个batch的关系嵌入
        elif relation_representations.dim() == 2:
            # [num_relations, embedding_dim]
            relation_embeddings = relation_representations
        else:
            raise ValueError(f"relation_representations维度不正确: {relation_representations.shape}")
        
        # 初始化boundary（所有实体初始化为零）
        # 内存优化：如果节点数很大，使用更节省内存的方式初始化
        if data.num_nodes > 10000:
            # 对于大图，分块初始化以减少峰值内存
            enhanced_boundary = torch.zeros(batch_size, data.num_nodes, self.embedding_dim, device=device, dtype=torch.float32)
        else:
            enhanced_boundary = torch.zeros(batch_size, data.num_nodes, self.embedding_dim, device=device)
        
        logger = logging.getLogger(__name__)
        
        # 优化：只计算查询实体的增强boundary（大幅减少计算量）
        # 1. 收集所有查询实体（源实体）- 这些是Bellman-Ford的起点，需要增强boundary
        with timer("收集查询实体", logger):
            query_entities = set()
            if isinstance(h_index, torch.Tensor):
                if h_index.dim() == 0:
                    query_entities.add(h_index.item())
                else:
                    query_entities.update(h_index.tolist())
            else:
                query_entities.add(h_index)
        
        # 2. 优化策略：只增强查询实体 + 最重要的6个实体（按度排序，并按权重增强）
        # 原因：Bellman-Ford算法中，查询实体是起点最重要，但少量重要邻居的增强可以提供更好的初始上下文
        # 如果提供了prompt_entities，优先使用并按重要性排序选择最重要的6个，然后按权重增强
        MAX_PROMPT_ENTITIES = 6  # 只选择最重要的6个实体（快速且精准）
        entity_weights = {}  # 存储每个实体的权重（用于加权增强）
        
        if prompt_entities is not None and len(prompt_entities) > 0:
            # 使用prompt_entities，但限制数量（按度排序选择最重要的6个）
            prompt_entities_set = set(prompt_entities)
            prompt_entities_set.update(query_entities)  # 确保包含查询实体
            
            # 计算所有实体的度（用于排序和权重计算）
            with timer("计算实体度并选择最重要的6个", logger):
                # 快速计算所有实体的度（使用向量化操作，非常快）
                all_nodes = torch.cat([data.edge_index[0], data.edge_index[1]])
                node_degrees = torch.bincount(all_nodes, minlength=data.num_nodes)
                
                # 计算每个prompt_entity的度
                prompt_entities_list = sorted([e for e in prompt_entities_set if e < data.num_nodes])
                entity_degrees_dict = {e: node_degrees[e].item() if e < len(node_degrees) else 0 
                                      for e in prompt_entities_list}
                
                # 优先保留查询实体，然后按度排序（度高的更重要）
                query_entities_set = set(query_entities)
                sorted_entities = sorted(
                    prompt_entities_list,
                    key=lambda e: (e not in query_entities_set, -entity_degrees_dict.get(e, 0))
                )
                
                # 只选择前6个最重要的实体
                selected_entities = sorted_entities[:MAX_PROMPT_ENTITIES]
                relevant_entities = set(selected_entities)
                
                # 计算每个选中实体的权重（基于度，查询实体权重最高）
                max_degree = max(entity_degrees_dict.values()) if entity_degrees_dict else 1
                for entity_id in selected_entities:
                    if entity_id in query_entities_set:
                        # 查询实体权重最高（1.0）
                        entity_weights[entity_id] = 1.0
                    else:
                        # 其他实体权重基于度（归一化到0.3-0.8之间）
                        degree = entity_degrees_dict.get(entity_id, 0)
                        # 使用sigmoid函数将度映射到0.3-0.8之间
                        normalized_degree = degree / max(max_degree, 1)  # 归一化到0-1
                        entity_weights[entity_id] = 0.3 + 0.5 * normalized_degree  # 映射到0.3-0.8
                
                logger.debug(f"[Entity Enhancer] prompt_entities数量({len(prompt_entities_set)})，"
                           f"按度排序选择前{MAX_PROMPT_ENTITIES}个最重要的实体，权重范围: "
                           f"{min(entity_weights.values()):.2f}-{max(entity_weights.values()):.2f}")
        else:
            # 如果没有prompt_entities，只使用查询实体（最快）
            relevant_entities = query_entities
            # 查询实体权重为1.0
            for entity_id in query_entities:
                entity_weights[entity_id] = 1.0
            logger.debug(f"[Entity Enhancer] 快速模式：只计算查询实体的增强boundary (数量={len(relevant_entities)})")
        
        # 3. 实体数量限制（通常查询实体数量很少，这个限制基本不会触发）
        MAX_ENTITIES_TO_COMPUTE = 100  # 限制最多计算100个实体（快速模式：只计算查询实体）
        if len(relevant_entities) > MAX_ENTITIES_TO_COMPUTE:
            logger.warning(f"[Entity Enhancer] 相关实体数量({len(relevant_entities)})超过限制({MAX_ENTITIES_TO_COMPUTE})，"
                          f"优先保留查询实体，然后按度排序选择前{MAX_ENTITIES_TO_COMPUTE}个实体")
            
            # 批量计算所有实体的度（使用向量化操作）
            relevant_entities_list = sorted([e for e in relevant_entities if e < data.num_nodes])
            if len(relevant_entities_list) > 0:
                relevant_entities_tensor = torch.tensor(relevant_entities_list, device=device, dtype=torch.long)
                
                # 向量化：一次性找到所有相关实体的边
                src_mask = torch.isin(data.edge_index[0], relevant_entities_tensor)
                dst_mask = torch.isin(data.edge_index[1], relevant_entities_tensor)
                relevant_edge_mask = src_mask | dst_mask
                
                # 批量计算每个实体的度
                entity_degrees = {}
                if relevant_edge_mask.any():
                    # 使用bincount快速统计每个实体作为src或dst出现的次数
                    src_nodes = data.edge_index[0][relevant_edge_mask]
                    dst_nodes = data.edge_index[1][relevant_edge_mask]
                    
                    # 统计每个实体在边中出现的次数
                    all_nodes = torch.cat([src_nodes, dst_nodes])
                    node_counts = torch.bincount(all_nodes, minlength=data.num_nodes)
                    
                    for entity_id in relevant_entities_list:
                        entity_degrees[entity_id] = node_counts[entity_id].item()
                else:
                    for entity_id in relevant_entities_list:
                        entity_degrees[entity_id] = 0
            
            # 优先保留查询实体，然后按度排序
            query_entities_set = set(query_entities)
            sorted_entities = sorted(
                relevant_entities,
                key=lambda e: (e not in query_entities_set, -entity_degrees.get(e, 0))
            )
            relevant_entities = set(sorted_entities[:MAX_ENTITIES_TO_COMPUTE])
        
        logger.debug(f"[Entity Enhancer] 查询实体数={len(query_entities)}, "
                    f"需要计算增强boundary的实体数={len(relevant_entities)}, batch_size={batch_size}")
        
        # 4. 批量计算所有实体的特征（大幅优化性能）
        with timer(f"批量计算实体特征 (实体数={len(relevant_entities)})", logger):
            # 批量获取所有实体相关的边
            relevant_entities_list = sorted([e for e in relevant_entities if e < data.num_nodes])
            if len(relevant_entities_list) > 0:
                relevant_entities_tensor = torch.tensor(relevant_entities_list, device=device, dtype=torch.long)
                
                # 向量化：一次性找到所有相关实体的边
                src_mask = torch.isin(data.edge_index[0], relevant_entities_tensor)
                dst_mask = torch.isin(data.edge_index[1], relevant_entities_tensor)
                relevant_edge_mask = src_mask | dst_mask
                
                if relevant_edge_mask.any():
                    # 获取所有相关边的关系类型（只获取有效的）
                    relevant_rels = data.edge_type[relevant_edge_mask]
                    valid_rel_mask = relevant_rels < relation_embeddings.shape[0]
                    valid_rels = relevant_rels[valid_rel_mask]
                    
                    # 获取相关边的源节点和目标节点
                    relevant_src = data.edge_index[0][relevant_edge_mask]
                    relevant_dst = data.edge_index[1][relevant_edge_mask]
                    
                    # 为每个实体批量计算特征（使用字典分组）
                    entity_to_rels = defaultdict(list)
                    for i in range(len(relevant_src)):
                        if valid_rel_mask[i]:
                            src_id = relevant_src[i].item()
                            dst_id = relevant_dst[i].item()
                            rel_id = valid_rels[i].item()
                            
                            if src_id in relevant_entities:
                                entity_to_rels[src_id].append(rel_id)
                            if dst_id in relevant_entities:
                                entity_to_rels[dst_id].append(rel_id)
                    
                    # 批量计算特征（按权重增强）
                    global_avg = relation_embeddings.mean(dim=0)
                    for entity_id in relevant_entities_list:
                        if entity_id in entity_to_rels and len(entity_to_rels[entity_id]) > 0:
                            # 使用该实体相关的所有关系的平均嵌入
                            rel_ids = torch.tensor(entity_to_rels[entity_id], device=device, dtype=torch.long)
                            entity_feat = relation_embeddings[rel_ids].mean(dim=0)
                        else:
                            entity_feat = global_avg
                        
                        # 根据实体重要性设置权重（查询实体权重1.0，其他实体0.3-0.8）
                        weight = entity_weights.get(entity_id, 0.5)  # 默认权重0.5
                        
                        # 加权增强：基础特征 + 权重 * 增强特征
                        # 对于查询实体，权重高，增强明显；对于其他实体，权重较低，增强较弱
                        enhanced_feat = entity_feat * weight
                        
                        # 批量设置特征（使用加权后的特征）
                        enhanced_boundary[:, entity_id, :] = enhanced_feat.unsqueeze(0).expand(batch_size, -1)
                else:
                    # 如果没有相关边，使用全局平均（按权重）
                    global_avg = relation_embeddings.mean(dim=0)
                    for entity_id in relevant_entities_list:
                        # 根据实体重要性设置权重
                        weight = entity_weights.get(entity_id, 0.5)  # 默认权重0.5
                        enhanced_feat = global_avg * weight
                        enhanced_boundary[:, entity_id, :] = enhanced_feat.unsqueeze(0).expand(batch_size, -1)
        
        # 5. 确保源实体有特征（使用关系嵌入，叠加到已有特征上，查询实体权重最高）
        # 处理r_index可能是标量或1D张量的情况
        if isinstance(r_index, torch.Tensor):
            if r_index.dim() == 0:  # 标量
                query = relation_embeddings[r_index.item()].unsqueeze(0)  # [1, embedding_dim]
                h_index_expanded = h_index.unsqueeze(0) if h_index.dim() == 0 else h_index
                for i in range(min(len(query), len(h_index_expanded))):
                    h_idx = h_index_expanded[i].item()
                    # 查询实体权重为1.0，直接叠加关系嵌入（增强效果）
                    enhanced_boundary[i, h_idx, :] += query[i]
            else:
                query = relation_embeddings[r_index]  # [batch_size, embedding_dim]
                for i in range(batch_size):
                    h_idx = h_index[i].item()
                    # 查询实体权重为1.0，直接叠加关系嵌入（增强效果）
                    enhanced_boundary[i, h_idx, :] += query[i]
        else:
            # r_index是Python标量
            query = relation_embeddings[r_index].unsqueeze(0)  # [1, embedding_dim]
            h_idx = h_index.item() if isinstance(h_index, torch.Tensor) and h_index.dim() == 0 else h_index[0]
            # 查询实体权重为1.0，直接叠加关系嵌入（增强效果）
            enhanced_boundary[0, h_idx, :] += query[0]
        
        return enhanced_boundary


class EnhancedEntityNBFNet(nn.Module):
    """
    增强版EntityNBFNet，使用实体-关系联合增强的boundary条件
    包装原始EntityNBFNet，但使用增强的boundary初始化
    """
    
    def __init__(self, entity_model, entity_enhancer):
        super(EnhancedEntityNBFNet, self).__init__()
        self.entity_model = entity_model
        self.entity_enhancer = entity_enhancer
    
    def forward(self, data, relation_representations, batch, prompt_entities=None):
        """
        使用增强的boundary条件进行实体推理
        
        Args:
            data: 图数据
            relation_representations: 关系表示
            batch: 批次数据
            prompt_entities: 提示图中的实体集合（可选），用于实体增强
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"[EnhancedEntityNBFNet] Forward开始，batch形状={batch.shape}")
        
        h_index, t_index, r_index = batch.unbind(-1)
        
        from ultra import parse
        try:
            flags = parse.load_flags('flags.yaml')
        except:
            # 如果无法加载flags，使用默认值
            class DefaultFlags:
                harder_setting = False
            flags = DefaultFlags()
        
        if flags.harder_setting == True:
            r_index = torch.ones_like(r_index) * (data.num_relations // 2 - 1)
        
        # 设置query和relation（保持与原始EntityNBFNet一致）
        # 注意：relation_representations可能是[batch_size, num_relations, embedding_dim]或[num_relations, embedding_dim]
        self.entity_model.query = relation_representations
        for layer in self.entity_model.layers:
            layer.relation = relation_representations
        
        if self.entity_model.training:
            data = self.entity_model.remove_easy_edges(data, h_index, t_index, r_index)
        
        shape = h_index.shape
        # negative_sample_to_tail期望2D输入，如果是1D需要先扩展
        if h_index.dim() == 1:
            h_index = h_index.unsqueeze(1)
            t_index = t_index.unsqueeze(1)
            r_index = r_index.unsqueeze(1)
        
        h_index, t_index, r_index = self.entity_model.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        
        # negative_sample_to_tail后，h_index和r_index应该是2D的
        assert h_index.dim() == 2, f"h_index应该是2D的，但得到: {h_index.shape}"
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        h_index_flat = h_index[:, 0]
        r_index_flat = r_index[:, 0]
        
        # 使用增强的boundary条件
        logger.debug(f"[EnhancedEntityNBFNet] 计算增强的boundary条件")
        enhanced_boundary = self.entity_enhancer.compute_enhanced_boundary(
            data, h_index_flat, r_index_flat, relation_representations,
            prompt_entities=prompt_entities  # 传递提示图中的实体
        )  # [batch_size, num_nodes, embedding_dim]
        logger.debug(f"[EnhancedEntityNBFNet] enhanced_boundary形状={enhanced_boundary.shape}")
        
        # 调用增强的bellmanford
        output = self._enhanced_bellmanford(
            data, h_index_flat, r_index_flat, enhanced_boundary
        )
        
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        
        score = self.entity_model.mlp(feature).squeeze(-1)
        logger.debug(f"[EnhancedEntityNBFNet] Forward完成，score形状={score.shape}")
        return score.view(shape)
    
    def _enhanced_bellmanford(self, data, h_index, r_index, enhanced_boundary):
        """
        使用增强的boundary条件执行Bellman-Ford算法
        """
        batch_size = len(r_index)
        
        # 获取query（关系嵌入）
        # 处理query的维度：可能是[batch_size, num_relations, embedding_dim]或[num_relations, embedding_dim]
        if self.entity_model.query.dim() == 3:
            # [batch_size, num_relations, embedding_dim]
            query = self.entity_model.query[torch.arange(batch_size, device=r_index.device), r_index]
        elif self.entity_model.query.dim() == 2:
            # [num_relations, embedding_dim]
            query = self.entity_model.query[r_index]  # [batch_size, embedding_dim]
        else:
            raise ValueError(f"query维度不正确: {self.entity_model.query.shape}")
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)
        
        hiddens = []
        edge_weights = []
        layer_input = enhanced_boundary  # 使用增强的boundary而不是零向量
        
        for layer in self.entity_model.layers:
            hidden = layer(layer_input, query, enhanced_boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.entity_model.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        
        # 添加query信息
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.entity_model.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        
        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }


class SimilarityBasedRelationEnhancer(nn.Module):
    """
    基于余弦相似度的关系增强模块
    根据查询关系与所有关系的相似度，加权参考相似关系来增强查询关系表示
    """
    
    def __init__(self, embedding_dim=64, similarity_threshold_init=0.8, enhancement_strength_init=0.05, max_similar_relations=20):
        super(SimilarityBasedRelationEnhancer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_similar_relations = max_similar_relations  # 最多使用多少个最相似的关系
        
        # 可学习的相似度阈值（经过sigmoid后映射到0-1范围）
        self.similarity_threshold_raw = nn.Parameter(torch.tensor(similarity_threshold_init * 2.0 - 1.0))
        
        # 可学习的增强强度参数（经过sigmoid后映射到0-0.2范围，保持较小以避免过度影响）
        self.enhancement_strength_raw = nn.Parameter(torch.tensor(enhancement_strength_init * 5.0 - 1.0))
        
        # 可学习的相似度加权参数（用于调整相似度对权重的影响）
        self.similarity_weight_scale = nn.Parameter(torch.tensor(1.0))
        
        # 可学习的温度参数（用于softmax缩放，控制相似度分布的平滑度）
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def get_similarity_threshold(self):
        """获取当前的可学习阈值（sigmoid映射到0-1）"""
        return torch.sigmoid(self.similarity_threshold_raw)
    
    def get_enhancement_strength(self):
        """获取当前的可学习增强强度（sigmoid映射到0-0.2）"""
        return torch.sigmoid(self.enhancement_strength_raw) * 0.2
    
    def forward(self, final_relation_representations, query_rels, return_enhancement_only=False):
        """
        基于相似度增强关系表示
        
        Args:
            final_relation_representations: [batch_size, num_relations, embedding_dim]
            query_rels: [batch_size] 查询关系索引
            return_enhancement_only: 如果为True，只返回增强增量（不进行内部混合），
                                    由外部门控机制控制混合；如果为False，使用内部strength进行混合
        
        Returns:
            enhanced_representations: [batch_size, num_relations, embedding_dim]
            如果 return_enhancement_only=True，返回的是增强增量（weighted_similar_repr），
            需要与原始表示混合；如果为False，返回的是已经混合后的表示
        """
        batch_size, num_relations, embedding_dim = final_relation_representations.shape
        device = final_relation_representations.device
        
        # 获取可学习参数
        threshold = self.get_similarity_threshold()
        strength = self.get_enhancement_strength()
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)  # 限制温度范围
        
        # 定期打印参数值（每100个batch打印一次）
        logger = logging.getLogger(__name__)
        if not hasattr(self, '_sim_log_counter'):
            self._sim_log_counter = 0
        self._sim_log_counter += 1
        if self._sim_log_counter % 100 == 0 or self._sim_log_counter == 1:
            logger.debug(f"[Similarity Enhancer] 参数: threshold={threshold.item():.3f}, strength={strength.item():.4f}, temp={temp.item():.2f}")
        
        # 初始化增强后的表示
        if return_enhancement_only:
            # 如果只返回增强增量，初始化为零（只更新查询关系的位置）
            enhanced_reprs = torch.zeros_like(final_relation_representations)
        else:
            # 如果返回混合后的表示，初始化为原始表示
            enhanced_reprs = final_relation_representations.clone()
        
        # 改进：批量归一化所有关系表示
        all_norms = F.normalize(final_relation_representations, p=2, dim=2)  # [batch, num_rels, dim]
        
        # 批量提取查询关系表示
        batch_indices = torch.arange(batch_size, device=device)  # [batch]
        query_indices_expanded = query_rels.unsqueeze(1).unsqueeze(2).expand(-1, -1, embedding_dim)  # [batch, 1, dim]
        query_reprs = torch.gather(all_norms, 1, query_indices_expanded).squeeze(1)  # [batch, dim]
        
        # 批量计算相似度矩阵
        # query_reprs: [batch, dim], all_norms: [batch, num_rels, dim]
        similarities = torch.bmm(query_reprs.unsqueeze(1), all_norms.transpose(1, 2)).squeeze(1)  # [batch, num_rels]
        
        # 批量排除查询关系本身（设置为-1，确保不被选择）
        similarities[batch_indices, query_rels] = -1.0
        
        # 批量计算阈值权重
        similarity_weights = torch.sigmoid((similarities - threshold) * 10.0)  # [batch, num_rels]
        above_threshold = similarity_weights > 0.5  # [batch, num_rels]
        
        # 统计信息
        total_valid_relations = 0
        total_used_relations = 0
        
        # 优化：批量处理top-k选择（使用批量top-k和gather）
        # 批量选择top-k个最相似的关系
        # similarities: [batch_size, num_relations]
        # 排除查询关系本身（已经设置为-1）
        # 批量top-k选择
        top_k_values, top_k_indices = torch.topk(similarities, k=min(self.max_similar_relations, num_relations), dim=1)  # [batch_size, k]
        
        # 批量过滤：只保留超过阈值的（top_k_values > threshold）
        # 注意：由于我们已经设置了查询关系本身为-1，top-k可能包含-1
        valid_mask = top_k_values > threshold  # [batch_size, k]
        
        # 对每个batch处理（由于每个batch的有效关系数不同，仍需要部分循环，但已大大简化）
        for i in range(batch_size):
            query_rel_idx = query_rels[i].item()
            if query_rel_idx >= num_relations or query_rel_idx < 0:
                continue
            
            # 获取当前batch的有效top-k索引
            batch_valid_mask = valid_mask[i]  # [k]
            batch_top_k_indices = top_k_indices[i]  # [k]
            batch_top_k_values = top_k_values[i]  # [k]
            
            # 过滤出有效的索引（超过阈值且不是查询关系本身）
            valid_top_k_mask = batch_valid_mask & (batch_top_k_indices != query_rel_idx)
            valid_indices = batch_top_k_indices[valid_top_k_mask]  # [num_valid, <=k]
            valid_similarities_raw = batch_top_k_values[valid_top_k_mask]  # [num_valid]
            
            total_valid_relations += len(valid_indices)
            total_used_relations += len(valid_indices)
            
            if len(valid_indices) == 0:
                # 如果没有超过阈值的，保持原样（零增量或原始表示）
                if not return_enhancement_only:
                    enhanced_reprs[i, query_rel_idx, :] = final_relation_representations[i, query_rel_idx, :]
                continue
            
            # 使用平滑的权重而不是硬阈值
            valid_weights_smooth = similarity_weights[i, valid_indices]  # [num_valid]
            
            # 获取有效关系的表示（使用批量归一化后的表示）
            valid_reprs = all_norms[i, valid_indices, :]  # [num_valid, embedding_dim]
            
            # 使用有效关系的相似度和平滑权重
            valid_similarities = valid_similarities_raw  # [num_valid]
            
            # 使用softmax根据相似度加权（应用温度参数）
            # 相似度越高，权重越大
            scaled_similarities = valid_similarities / temp
            weights = F.softmax(scaled_similarities, dim=0)  # [num_valid]
            
            # 结合平滑的阈值权重（让阈值也参与梯度计算）
            # weights来自softmax（基于相似度），valid_weights_smooth来自阈值过滤
            combined_weights = weights * valid_weights_smooth  # [num_valid]
            
            # 进一步根据相似度强度调整权重（可学习的缩放因子）
            adjusted_weights = combined_weights * (1.0 + self.similarity_weight_scale * valid_similarities)  # [num_valid]
            adjusted_weights = adjusted_weights / (adjusted_weights.sum() + 1e-8)  # 重新归一化 [num_valid]
            
            # 计算加权平均的相似关系表示
            # valid_reprs: [num_valid, embedding_dim], adjusted_weights: [num_valid]
            # [num_valid, embedding_dim] * [num_valid, 1] -> [embedding_dim]
            weighted_similar_repr = torch.sum(valid_reprs * adjusted_weights.unsqueeze(1), dim=0)  # [embedding_dim]
            
            # 根据 return_enhancement_only 决定返回什么
            query_rel_repr = final_relation_representations[i, query_rel_idx, :]  # [embedding_dim]
            if return_enhancement_only:
                # 只返回增强增量（weighted_similar_repr - query_rel_repr），由外部门控机制控制混合
                enhancement_delta = weighted_similar_repr - query_rel_repr
                enhanced_reprs[i, query_rel_idx, :] = enhancement_delta
            else:
                # 使用内部strength进行混合（原有逻辑）
                # enhanced = original + strength * (weighted_similar - original)
                # 这等价于：enhanced = (1 - strength) * original + strength * weighted_similar
                enhanced_query_repr = (1.0 - strength) * query_rel_repr + strength * weighted_similar_repr
                enhanced_reprs[i, query_rel_idx, :] = enhanced_query_repr
        
        # 打印统计信息（每100个batch打印一次）
        if self._sim_log_counter % 100 == 0 or self._sim_log_counter == 1:
            avg_valid = total_valid_relations / batch_size if batch_size > 0 else 0
            avg_used = total_used_relations / batch_size if batch_size > 0 else 0
            logger.debug(f"[Similarity Enhancer] 统计: 平均有效相似关系={avg_valid:.1f}, 平均使用={avg_used:.1f}")
        
        return enhanced_reprs


class OptimizedPromptGraph(nn.Module):
    """
    优化版自适应提示图增强模块
    减少计算开销，提高运行效率
    """
    
    def __init__(self, embedding_dim=64, max_hops=1, num_prompt_samples=3):
        super(OptimizedPromptGraph, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_hops = max_hops
        self.num_prompt_samples = num_prompt_samples
        
        # EntityNBFNet特征投影层（解决维度不匹配问题）
        # 优化：动态创建投影层，只创建实际使用的（节省约53,504参数）
        # EntityNBFNet的feature_dim可能是128（concat_hidden=False）或448（concat_hidden=True）
        # 使用字典存储，按需创建投影层
        self.entity_feature_proj = nn.ModuleDict()  # 空字典，动态创建投影层
        
        # EntityNBFNet结果缓存（优化：避免重复计算bellmanford）
        # 缓存key: (relation_representations_hash, data_hash)
        # 缓存value: entity_features_dict
        self._bfnet_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 10  # 最多缓存10个结果（内存优化：减少缓存大小以避免OOM）
        
        # 简化的提示图编码器（用于无边的图）
        self.prompt_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 简单的GCN层（用于编码图结构）
        # 使用MessagePassing实现简单的图卷积
        from torch_geometric.nn import MessagePassing
        
        class SimpleGCNLayer(MessagePassing):
            """简单的GCN层（带自环和归一化）"""
            def __init__(self, embedding_dim):
                super().__init__(aggr='add')
                self.lin = nn.Linear(embedding_dim, embedding_dim)
                self.norm = nn.LayerNorm(embedding_dim)
            
            def forward(self, x, edge_index):
                # x: [num_nodes, embedding_dim]
                # edge_index: [2, num_edges]
                # 添加自环（每个节点连接到自身）
                from torch_geometric.utils import add_self_loops
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
                
                # 计算度归一化
                from torch_geometric.utils import degree
                row, col = edge_index
                deg = degree(col, x.size(0), dtype=x.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                
                # 传播消息
                out = self.propagate(edge_index, x=x, norm=norm)
                out = self.lin(out)
                out = self.norm(out)
                return out
            
            def message(self, x_j, norm):
                return norm.view(-1, 1) * x_j
        
        # 使用1层GCN编码图结构
        self.gnn_layers = nn.ModuleList([
            SimpleGCNLayer(embedding_dim) for _ in range(1)
        ])
        
        # 注意力池化（替代简单平均池化）
        self.attention_pooling = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # 简化的自适应权重网络
        self.adaptive_weights = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 简化的上下文融合网络
        self.context_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 缓存邻接表（避免重复构建）
        self._adj_list_cache = None
        self._cached_edge_index = None
        self._cached_edge_index_hash = None
        
    def _get_k_hop_neighbors(self, data, seed_entities, max_hops=1):
        """
        获取实体的k跳邻居（包括实体本身）
        
        Args:
            data: 图数据
            seed_entities: 种子实体集合（set或list）
            max_hops: 最大跳数（默认1跳）
        
        Returns:
            relevant_entities: 相关实体集合（包括种子实体和k跳邻居）
        """
        if data.edge_index.numel() == 0:
            return set(seed_entities)
        
        relevant_entities = set(seed_entities)
        current_frontier = set(seed_entities)
        
        logger = logging.getLogger(__name__)
        
        # 检查是否可以复用缓存的邻接表
        # 使用hash来快速比较（避免每次都比较整个tensor）
        edge_index_hash = hash((data.edge_index.shape[0], data.edge_index.shape[1], 
                               data.edge_index.data_ptr() if hasattr(data.edge_index, 'data_ptr') else id(data.edge_index)))
        
        if (self._adj_list_cache is not None and 
            self._cached_edge_index_hash == edge_index_hash and
            self._cached_edge_index is not None and
            self._cached_edge_index.shape == data.edge_index.shape):
            # 快速检查：只比较前几个元素（避免完整比较）
            if torch.equal(self._cached_edge_index[:2, :min(100, data.edge_index.shape[1])], 
                          data.edge_index[:2, :min(100, data.edge_index.shape[1])]):
                # 复用缓存的邻接表
                adj_list = self._adj_list_cache
                logger.debug(f"[Prompt Enhancer] 复用缓存的邻接表 (边数={data.edge_index.shape[1]})")
            else:
                # 图数据已变化，需要重新构建
                adj_list = None
        else:
            adj_list = None
        
        # 如果需要构建新的邻接表
        if adj_list is None:
            with timer(f"[Prompt Enhancer] 构建邻接表 (边数={data.edge_index.shape[1]})", logger):
                # 使用numpy向量化操作（比Python循环快很多）
                edge_index_np = data.edge_index.cpu().numpy()
                src_nodes = edge_index_np[0]
                dst_nodes = edge_index_np[1]
                
                # 使用defaultdict提高效率
                adj_list = defaultdict(set)
                
                # 向量化构建双向邻接表
                # 同时处理src->dst和dst->src
                for src, dst in zip(src_nodes, dst_nodes):
                    adj_list[src].add(dst)
                    adj_list[dst].add(src)  # 无向图
                
                # 转换为普通dict（避免后续访问defaultdict的开销）
                adj_list = dict(adj_list)
                
                # 缓存邻接表和边索引
                self._adj_list_cache = adj_list
                self._cached_edge_index = data.edge_index.clone()
                self._cached_edge_index_hash = edge_index_hash
        
        # BFS遍历k跳（优化：使用批量集合操作）
        with timer(f"[Prompt Enhancer] BFS遍历{max_hops}跳 (种子数={len(seed_entities)})", logger):
            for hop in range(max_hops):
                # 批量获取当前frontier的所有邻居
                all_neighbors = set()
                for entity in current_frontier:
                    if entity in adj_list:
                        all_neighbors.update(adj_list[entity])
                
                # 批量添加新邻居（使用集合差集操作，比逐个检查快）
                new_neighbors = all_neighbors - relevant_entities
                relevant_entities.update(new_neighbors)
                current_frontier = new_neighbors
                
                if not current_frontier:  # 如果没有新邻居，提前结束
                    break
        
        return relevant_entities
        
    def generate_prompt_graph(self, data, query_relation, query_entity, num_samples=None):
        """简化版：直接找到与查询关系相关的边，选择中心度最高的几个
        
        Returns:
            selected_edges: 选中的边 [2, num_selected]，或None
            selected_edge_types: 选中的边类型 [num_selected]，或None
        """
        logger = logging.getLogger(__name__)
        if num_samples is None:
            num_samples = self.num_prompt_samples
            
        device = query_entity.device
        query_entity_id = query_entity.item() if isinstance(query_entity, torch.Tensor) else query_entity
        query_rel_idx = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
        logger.debug(f"[Prompt Enhancer] 简化模式：查找查询关系相关的边: query_entity={query_entity_id}, query_rel={query_rel_idx}")
        
        # 1. 找到包含查询关系的边
        edge_mask = (data.edge_type == query_relation)
        query_edges = data.edge_index[:, edge_mask]
        query_edge_types = data.edge_type[edge_mask]
        
        if query_edges.shape[1] == 0:
            logger.debug(f"[Prompt Enhancer] 未找到查询关系的边")
            return None, None
        
        # 2. 如果边数超过num_samples，选择中心度最高的几个
        if query_edges.shape[1] > num_samples:
            # 计算每条边的重要性（使用源实体和目标实体的度之和作为中心度）
            head_nodes = query_edges[0]
            tail_nodes = query_edges[1]
            
            # 快速计算实体度（使用bincount）
            all_nodes = torch.cat([data.edge_index[0], data.edge_index[1]])
            node_degrees = torch.bincount(all_nodes, minlength=data.num_nodes)
            
            # 计算每条边的重要性（源实体度 + 目标实体度）
            edge_importance = node_degrees[head_nodes] + node_degrees[tail_nodes]
            
            # 选择重要性最高的num_samples条边
            _, top_indices = torch.topk(edge_importance, k=min(num_samples, query_edges.shape[1]), dim=0)
            selected_edges = query_edges[:, top_indices]
            selected_edge_types = query_edge_types[top_indices]
        else:
            selected_edges = query_edges
            selected_edge_types = query_edge_types
        
        logger.debug(f"[Prompt Enhancer] 选中 {selected_edges.shape[1]} 条边（总共有 {query_edges.shape[1]} 条查询关系的边）")
        
        return selected_edges, selected_edge_types
    
    def generate_prompt_graph_batch(self, data, query_rels, query_entities, num_samples=None):
        """批量版：为整个batch找到查询关系相关的边，选择中心度最高的几个
        
        Args:
            data: 图数据
            query_rels: [batch_size] 查询关系索引
            query_entities: [batch_size] 查询实体索引
            num_samples: 每个查询关系选择的边数
            
        Returns:
            selected_edges_list: List[selected_edges] 每个batch的选中边，或None
            selected_edge_types_list: List[selected_edge_types] 每个batch的选中边类型，或None
        """
        logger = logging.getLogger(__name__)
        if num_samples is None:
            num_samples = self.num_prompt_samples
        
        batch_size = len(query_rels)
        device = query_rels.device
        
        # 预计算实体度（只需要计算一次）
        all_nodes = torch.cat([data.edge_index[0], data.edge_index[1]])
        node_degrees = torch.bincount(all_nodes, minlength=data.num_nodes)
        
        selected_edges_list = []
        selected_edge_types_list = []
        
        # 批量处理：按查询关系分组（相同查询关系可以共享边查找结果）
        unique_query_rels = query_rels.unique()
        rel_to_edges = {}
        
        # 为每个唯一的查询关系找到边
        for rel in unique_query_rels:
            edge_mask = (data.edge_type == rel)
            query_edges = data.edge_index[:, edge_mask]
            query_edge_types = data.edge_type[edge_mask]
            
            if query_edges.shape[1] == 0:
                rel_to_edges[rel.item()] = (None, None)
                continue
            
            # 如果边数超过num_samples，选择中心度最高的几个
            if query_edges.shape[1] > num_samples:
                head_nodes = query_edges[0]
                tail_nodes = query_edges[1]
                edge_importance = node_degrees[head_nodes] + node_degrees[tail_nodes]
                _, top_indices = torch.topk(edge_importance, k=min(num_samples, query_edges.shape[1]), dim=0)
                selected_edges = query_edges[:, top_indices]
                selected_edge_types = query_edge_types[top_indices]
            else:
                selected_edges = query_edges
                selected_edge_types = query_edge_types
            
            rel_to_edges[rel.item()] = (selected_edges, selected_edge_types)
        
        # 为每个batch样本分配对应的边
        for i in range(batch_size):
            rel = query_rels[i].item()
            selected_edges, selected_edge_types = rel_to_edges[rel]
            selected_edges_list.append(selected_edges)
            selected_edge_types_list.append(selected_edge_types)
        
        logger.debug(f"[Prompt Enhancer] 批量处理完成，batch_size={batch_size}, 唯一查询关系数={len(unique_query_rels)}")
        
        return selected_edges_list, selected_edge_types_list
    
    def encode_prompt_context(self, selected_edges, selected_edge_types, query_relation, relation_embeddings=None, query_entity=None, data=None):
        """改进版：使用选中的边信息增强context（中期优化）
        
        Args:
            selected_edges: 选中的边 [2, num_selected]，或None
            selected_edge_types: 选中的边类型 [num_selected]，或None
            query_relation: 查询关系索引
            relation_embeddings: 关系嵌入矩阵 [num_relations, embedding_dim]
            query_entity: 查询实体索引
            data: 图数据（用于获取实体信息）
        """
        logger = logging.getLogger(__name__)
        
        if selected_edges is None or selected_edge_types is None:
            device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
            return torch.zeros(self.embedding_dim, device=device)
        
        device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
        
        if relation_embeddings is None or relation_embeddings.shape[0] == 0:
            logger.warning(f"[Prompt Enhancer] 未提供关系嵌入，返回零向量")
            return torch.zeros(self.embedding_dim, device=device)
        
        query_rel_idx = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
        
        # 基础：使用查询关系的嵌入
        if query_rel_idx >= relation_embeddings.shape[0]:
            logger.warning(f"[Prompt Enhancer] 查询关系索引 {query_rel_idx} 超出范围，使用全局平均")
            base_context = relation_embeddings.mean(dim=0)  # [embedding_dim]
        else:
            base_context = relation_embeddings[query_rel_idx]  # [embedding_dim]
        
        # 中期优化：利用选中的边信息增强context
        # 使用选中边的实体对信息（通过关系嵌入的加权平均）
        num_selected = selected_edges.shape[1]
        if num_selected > 0:
            # 提取选中边的实体对
            head_entities = selected_edges[0]  # [num_selected]
            tail_entities = selected_edges[1]  # [num_selected]
            
            # 方案1：使用选中边的数量作为权重（边越多，增强越强）
            # 方案2：使用选中边的中心度作为权重（已经在generate_prompt_graph中计算过）
            # 简化：使用均匀权重，但根据边数调整增强强度
            edge_weight = min(1.0, num_selected / self.num_prompt_samples)  # 归一化到[0,1]
            
            # 使用选中边的关系嵌入（都是查询关系类型）的加权平均
            # 由于所有边都是查询关系类型，可以直接使用base_context
            # 但我们可以根据边数调整：边数越多，context越"丰富"
            enhanced_context = base_context * (1.0 + 0.1 * edge_weight)  # 轻微增强
            
            logger.debug(f"[Prompt Enhancer] 使用选中边增强context: 边数={num_selected}, 增强权重={edge_weight:.3f}")
            return enhanced_context
        else:
            return base_context
    
    def encode_prompt_context_batch(self, selected_edges_list, selected_edge_types_list, query_rels, relation_embeddings, query_entities, data):
        """批量版：为整个batch编码提示图上下文
        
        Args:
            selected_edges_list: List[selected_edges] 每个batch的选中边
            selected_edge_types_list: List[selected_edge_types] 每个batch的选中边类型
            query_rels: [batch_size] 查询关系索引
            relation_embeddings: [batch_size, num_relations, embedding_dim] 或 [num_relations, embedding_dim] 关系嵌入矩阵
            query_entities: [batch_size] 查询实体索引
            data: 图数据
            
        Returns:
            context_embeddings: [batch_size, embedding_dim] 每个batch的context嵌入
        """
        batch_size = len(query_rels)
        device = query_rels.device
        embedding_dim = self.embedding_dim
        
        context_embeddings = torch.zeros(batch_size, embedding_dim, device=device)
        
        # 处理relation_embeddings的维度
        if relation_embeddings.dim() == 3:
            # [batch_size, num_relations, embedding_dim] - 每个batch有自己的关系嵌入
            use_batch_specific = True
        else:
            # [num_relations, embedding_dim] - 共享的关系嵌入
            use_batch_specific = False
        
        for i in range(batch_size):
            selected_edges = selected_edges_list[i]
            selected_edge_types = selected_edge_types_list[i]
            query_rel = query_rels[i]
            query_entity = query_entities[i]
            
            # 选择对应的关系嵌入
            if use_batch_specific:
                rel_emb = relation_embeddings[i]  # [num_relations, embedding_dim]
            else:
                rel_emb = relation_embeddings  # [num_relations, embedding_dim]
            
            context = self.encode_prompt_context(
                selected_edges, selected_edge_types, query_rel, 
                rel_emb, query_entity, data
            )
            context_embeddings[i] = context
        
        return context_embeddings
    
    def _improved_entity_embedding(self, prompt_graph, query_relation, query_entity, relation_embeddings, device):
        """改进方案：基于节点在图中的位置和连接关系初始化嵌入"""
        # 获取查询关系嵌入
        query_rel_idx = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
        
        if query_rel_idx < relation_embeddings.shape[0]:
            base_embedding = relation_embeddings[query_rel_idx]  # [embedding_dim]
        else:
            # 如果查询关系索引无效，使用全局平均
            base_embedding = relation_embeddings.mean(dim=0)  # [embedding_dim]
        
        num_nodes = prompt_graph.num_nodes
        node_embeddings = torch.zeros(num_nodes, self.embedding_dim, device=device)
        
        # 计算每个节点的特征
        query_entity_id = query_entity.item() if isinstance(query_entity, torch.Tensor) else query_entity
        
        # 计算节点度（如果图有边）
        if prompt_graph.edge_index.numel() > 0:
            # 计算每个节点的度（在prompt_graph中的度）
            src_nodes = prompt_graph.edge_index[0]
            node_degrees = torch.bincount(src_nodes, minlength=num_nodes).float()
            # 归一化度
            max_degree = node_degrees.max()
            if max_degree > 0:
                normalized_degrees = node_degrees / (max_degree + 1e-8)
            else:
                normalized_degrees = torch.ones(num_nodes, device=device) * 0.5
        else:
            normalized_degrees = torch.ones(num_nodes, device=device) * 0.5
        
        # 为每个节点生成不同的嵌入
        for i in range(num_nodes):
            # 基于节点度生成变化（度高的节点更接近base_embedding，度低的节点有更多变化）
            degree_factor = normalized_degrees[i].item()
            
            # 使用节点索引和查询实体的哈希值生成确定性的变化
            import hashlib
            seed = int(hashlib.md5(f"{i}_{query_entity_id}_{query_rel_idx}".encode()).hexdigest()[:8], 16) % (2**32)
            torch.manual_seed(seed)
            variation = torch.randn(self.embedding_dim, device=device) * (1 - degree_factor) * 0.1
            torch.manual_seed(torch.initial_seed())
            
            # 结合base_embedding和变化
            node_embeddings[i] = base_embedding + variation
        
        # 训练时添加小的随机噪声以增加多样性
        if self.training:
            noise_scale = 0.02  # 很小的噪声，避免过度影响
            node_embeddings = node_embeddings + torch.randn_like(node_embeddings) * noise_scale
        
        return node_embeddings
    
    def forward(self, data, query_relation, query_entity, base_embeddings, return_enhancement_only=False, relation_embeddings=None, entity_model=None, relation_representations=None):
        """
        优化的前向传播
        
        Args:
            data: 图数据
            query_relation: 查询关系索引
            query_entity: 查询实体索引
            base_embeddings: 基础嵌入表示
            return_enhancement_only: 如果为True，只返回增强增量（不进行内部混合），
                                    由外部权重控制混合；如果为False，使用内部adaptive_weight进行混合
            relation_embeddings: 关系嵌入矩阵 [num_relations, embedding_dim]，用于初始化提示图节点嵌入
            entity_model: EntityNBFNet模型，用于计算实体特征（方案2：最优方案）
            relation_representations: 关系表示 [num_relations, embedding_dim]，用于EntityNBFNet计算
        
        Returns:
            如果 return_enhancement_only=True，返回增强增量
            如果 return_enhancement_only=False，返回增强后的完整表示
        """
        logger = logging.getLogger(__name__)
        
        # 快速生成提示图（同时获取实体列表）
        with timer("generate_prompt_graph", logger):
            prompt_graph, prompt_entities = self.generate_prompt_graph(data, query_relation, query_entity)
        
        # 快速编码上下文（方案2：使用EntityNBFNet计算实体特征，最优方案）
        # 传入entity_model和relation_representations以使用EntityNBFNet计算实体特征
        with timer("encode_prompt_context", logger):
            prompt_context = self.encode_prompt_context(
                prompt_graph, query_relation, relation_embeddings, prompt_entities, data,
                entity_model=entity_model, relation_representations=relation_representations
            )
        
        # 计算自适应权重
        # 兼容两种输入：
        # - base_embeddings 为 [embedding_dim] 的单个向量（推荐用法）
        # - base_embeddings 为 [num_relations, embedding_dim] 的矩阵（回退兼容）
        if isinstance(base_embeddings, torch.Tensor) and base_embeddings.dim() >= 2:
            # 从矩阵中取出对应关系的向量
            query_embedding = base_embeddings[query_relation]
        else:
            # 已经是对应关系的向量
            query_embedding = base_embeddings
        
        # 防御性检查：确保都是 1D 向量 [embedding_dim]
        if query_embedding.dim() == 0:
            # 避免 0 维张量导致 cat 报错
            query_embedding = query_embedding.unsqueeze(0)
        if prompt_context.dim() == 0:
            prompt_context = prompt_context.unsqueeze(0)
        
        weight_input = torch.cat([query_embedding, prompt_context], dim=-1)
        adaptive_weight = self.adaptive_weights(weight_input)
        
        # 融合上下文信息
        fusion_input = torch.cat([query_embedding, prompt_context], dim=-1)
        enhanced_embedding = self.context_fusion(fusion_input)
        
        if return_enhancement_only:
            # 只返回增强增量，由外部权重控制混合
            enhancement_delta = enhanced_embedding  # 返回融合后的增强嵌入（不含自适应权重）
            return enhancement_delta
        else:
            # 应用自适应权重（原有逻辑）
            final_embedding = query_embedding + adaptive_weight * enhanced_embedding
            return final_embedding


class AdaptiveEnhancementGate(nn.Module):
    """
    自适应增强门控网络
    基于查询特征学习决定是否应该使用增强
    """
    
    def __init__(self, embedding_dim=64):
        super(AdaptiveEnhancementGate, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 特征提取网络：输入关系嵌入、实体嵌入、图统计特征
        # 特征维度：关系嵌入(64) + 实体嵌入(64) + 统计特征(4) = 132
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU()
        )
        
        # 门控网络：输出增强权重 (0-1之间)
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()  # 输出0-1之间的增强权重
        )
        
        # 初始化：默认倾向于使用增强（初始权重0.7）
        # 这样模型可以从一个相对积极的增强策略开始学习
        with torch.no_grad():
            self.gate_network[-2].weight.data.fill_(0.0)
            self.gate_network[-2].bias.data.fill_(0.85)  # sigmoid(0.85) ≈ 0.7
    
    def extract_graph_statistics(self, data, query_rels, query_entities):
        """
        提取图统计特征
        Args:
            data: 图数据对象
            query_rels: [batch_size] 查询关系索引
            query_entities: [batch_size] 查询实体索引
        Returns:
            stats: [batch_size, 4] 统计特征
        """
        batch_size = len(query_rels)
        device = query_rels.device
        stats = []
        
        # 预计算总边数，避免重复计算
        total_edges = data.edge_index.shape[1] if data.edge_index.numel() > 0 else 1
        total_relations = data.num_relations if hasattr(data, 'num_relations') else 1
        
        for i in range(batch_size):
            rel_idx = query_rels[i].item()
            entity_idx = query_entities[i].item()
            
            # 特征1: 关系频率（归一化）
            if rel_idx < total_relations and total_edges > 0:
                rel_mask = (data.edge_type == rel_idx)
                rel_freq = rel_mask.sum().item()
                rel_freq_norm = min(rel_freq / max(total_edges, 1), 1.0)
            else:
                rel_freq_norm = 0.0
            
            # 特征2: 实体度（归一化）
            if entity_idx < data.num_nodes and total_edges > 0:
                entity_mask = (data.edge_index[0] == entity_idx) | (data.edge_index[1] == entity_idx)
                entity_degree = entity_mask.sum().item()
                entity_degree_norm = min(entity_degree / max(total_edges, 1), 1.0)
            else:
                entity_degree_norm = 0.0
            
            # 特征3: 查询关系的平均相似度（与所有其他关系的相似度）
            # 这里简化处理，使用关系频率作为代理
            avg_similarity = rel_freq_norm
            
            # 特征4: 图的稀疏度（边数/节点数）
            max_possible_edges = data.num_nodes * data.num_nodes if data.num_nodes > 0 else 1
            graph_density = min(total_edges / max(max_possible_edges, 1), 1.0)
            
            stats.append([rel_freq_norm, entity_degree_norm, avg_similarity, graph_density])
        
        return torch.tensor(stats, device=device, dtype=torch.float32)
    
    def forward(self, relation_embeddings, query_rels, query_entities, data):
        """
        计算增强门控权重
        Args:
            relation_embeddings: [batch_size, num_relations, embedding_dim] 关系嵌入
            query_rels: [batch_size] 查询关系索引
            query_entities: [batch_size] 查询实体索引
            data: 图数据对象
        Returns:
            gate_weights: [batch_size] 增强权重 (0-1之间)
        """
        batch_size = len(query_rels)
        device = query_rels.device
        
        # 提取查询关系的嵌入
        query_rel_embeddings = []
        query_entity_embeddings = []
        
        for i in range(batch_size):
            rel_idx = query_rels[i].item()
            entity_idx = query_entities[i].item()
            
            # 获取查询关系嵌入（使用平均池化作为实体嵌入的代理）
            if rel_idx < relation_embeddings.shape[1]:
                rel_emb = relation_embeddings[i, rel_idx, :]  # [embedding_dim]
            else:
                rel_emb = torch.zeros(self.embedding_dim, device=device)
            
            # 对于实体嵌入，我们使用与该实体相关的所有关系的平均嵌入作为代理
            # 这是一个简化，因为实体嵌入可能不在relation_embeddings中
            if entity_idx < data.num_nodes:
                entity_edge_mask = (data.edge_index[0] == entity_idx) | (data.edge_index[1] == entity_idx)
                if entity_edge_mask.any():
                    entity_rels = data.edge_type[entity_edge_mask]
                    # 获取这些关系的嵌入并平均
                    valid_rels = entity_rels[entity_rels < relation_embeddings.shape[1]]
                    if len(valid_rels) > 0:
                        entity_emb = relation_embeddings[i, valid_rels, :].mean(dim=0)
                    else:
                        entity_emb = torch.zeros(self.embedding_dim, device=device)
                else:
                    entity_emb = torch.zeros(self.embedding_dim, device=device)
            else:
                entity_emb = torch.zeros(self.embedding_dim, device=device)
            
            query_rel_embeddings.append(rel_emb)
            query_entity_embeddings.append(entity_emb)
        
        query_rel_embeddings = torch.stack(query_rel_embeddings, dim=0)  # [batch_size, embedding_dim]
        query_entity_embeddings = torch.stack(query_entity_embeddings, dim=0)  # [batch_size, embedding_dim]
        
        # 提取图统计特征
        graph_stats = self.extract_graph_statistics(data, query_rels, query_entities)  # [batch_size, 4]
        
        # 拼接特征
        features = torch.cat([
            query_rel_embeddings,  # [batch_size, embedding_dim]
            query_entity_embeddings,  # [batch_size, embedding_dim]
            graph_stats  # [batch_size, 4]
        ], dim=-1)  # [batch_size, embedding_dim * 2 + 4]
        
        # 提取特征
        extracted_features = self.feature_extractor(features)  # [batch_size, embedding_dim // 2]
        
        # 计算门控权重
        gate_weights = self.gate_network(extracted_features).squeeze(-1)  # [batch_size]
        
        return gate_weights


class EnhancedUltra(nn.Module):
    """
    增强版Ultra模型
    集成提示图增强功能，提高知识图谱推理性能
    """
    
    def __init__(self, rel_model_cfg, entity_model_cfg, sem_model_cfg=None):
        super(EnhancedUltra, self).__init__()
        
        # 导入原始模型类
        from ultra.models import RelNBFNet, EntityNBFNet, SemRelNBFNet, CombineEmbeddings
        from ultra import parse
        
        # 使用更可靠的方法找到项目根目录
        import os
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        # 在__init__中加载flags一次，避免在forward中重复加载
        self.flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        flags = self.flags
        
        # 原始模型组件
        self.relation_model = RelNBFNet(**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)
        
        if flags.run == "semma" or flags.run == "EnhancedUltra":
            self.semantic_model = SemRelNBFNet(**sem_model_cfg)
            self.combiner = CombineEmbeddings(embedding_dim=64)
        
        # 消融实验配置：控制各组件的启用/禁用
        self.use_similarity_enhancer = getattr(flags, 'use_similarity_enhancer', True)
        self.use_prompt_enhancer = getattr(flags, 'use_prompt_enhancer', False)
        # 注意：use_adaptive_gate已移除，功能由use_learnable_fusion替代（更灵活，可以学习两个增强器的权重）
        # use_learnable_fusion可以通过学习权重为0来"关闭"增强，功能更强大
        self.use_adaptive_gate = False  # 已废弃，保留为False以保持兼容性
        
        # 增强器权重配置：仅用于固定权重模式（当use_learnable_fusion=False时）
        # 当use_learnable_fusion=True时，这些权重不会被使用，融合权重会自动学习
        self.similarity_enhancer_weight = getattr(flags, 'similarity_enhancer_weight', 1.0)
        self.prompt_enhancer_weight = getattr(flags, 'prompt_enhancer_weight', 1.0)
        
        # 可学习的融合权重（2权重归一化方案，更清晰且无冗余）
        # 融合公式：enhanced = w_base*r + w_enhance*(r + w_sim*r1_delta + w_prompt*r2_delta)
        # 其中 w_base + w_enhance = 1 (softmax归一化)
        # w_sim 和 w_prompt 是增强器内部的权重分配（归一化）
        # 这样设计避免了3权重方案中的冗余问题
        use_learnable_fusion = getattr(flags, 'use_learnable_fusion', True)  # 默认启用可学习融合
        self.use_learnable_fusion = use_learnable_fusion
        
        if self.use_learnable_fusion:
            # 智能初始化：2个主权重 [原始r的权重, 增强的权重]
            # 基于增强器的启用状态自动计算合理的初始值，不依赖配置文件
            
            epsilon = 1e-3  # 禁用时的最小权重（避免完全为0导致梯度问题）
            
            # 计算启用的增强器数量
            num_enabled_enhancers = sum([self.use_similarity_enhancer, self.use_prompt_enhancer])
            
            if num_enabled_enhancers == 0:
                # 没有启用任何增强器：原始r占主导
                initial_base_weight = 0.95
                initial_enhance_weight = 0.05
            elif num_enabled_enhancers == 1:
                # 只启用一个增强器：原始r占70%，增强占30%
                initial_base_weight = 0.7
                initial_enhance_weight = 0.3
            else:
                # 两个增强器都启用：原始r占60%，增强占40%
                initial_base_weight = 0.6
                initial_enhance_weight = 0.4
            
            # 归一化两个主权重（确保和为1）
            total = initial_base_weight + initial_enhance_weight
            initial_main_weights = torch.tensor([
                initial_base_weight / total,
                initial_enhance_weight / total
            ])
            
            # 增强器内部权重分配（用于分配r1_delta和r2_delta的权重）
            if num_enabled_enhancers == 0:
                initial_sim_ratio = 0.5  # 未使用，但需要初始化
                initial_prompt_ratio = 0.5
            elif num_enabled_enhancers == 1:
                if self.use_similarity_enhancer:
                    initial_sim_ratio = 1.0
                    initial_prompt_ratio = epsilon
                else:
                    initial_sim_ratio = epsilon
                    initial_prompt_ratio = 1.0
            else:
                # 两个都启用：平分权重
                initial_sim_ratio = 0.5
                initial_prompt_ratio = 0.5
            
            # 归一化增强器内部权重
            total_enhance = initial_sim_ratio + initial_prompt_ratio
            initial_enhance_weights = torch.tensor([
                initial_sim_ratio / total_enhance,
                initial_prompt_ratio / total_enhance
            ])
            
            # 转换为logits（使用log-softmax的逆变换）
            initial_main_weights = torch.clamp(initial_main_weights, min=epsilon, max=1.0-epsilon)
            initial_enhance_weights = torch.clamp(initial_enhance_weights, min=epsilon, max=1.0-epsilon)
            main_logits = torch.log(initial_main_weights)
            enhance_logits = torch.log(initial_enhance_weights)
            
            # 注册为可学习参数
            # main_weights: [w_base, w_enhance]
            # enhance_weights: [w_sim, w_prompt] (用于分配r1_delta和r2_delta的权重)
            self.fusion_main_weights_logits = nn.Parameter(main_logits)
            self.fusion_enhance_weights_logits = nn.Parameter(enhance_logits)
            
            # 记录初始权重（用于调试）
            logger = logging.getLogger(__name__)
            logger.info(f"[EnhancedUltra] 融合权重初始化 (2权重方案): "
                       f"原始r={initial_main_weights[0]:.3f}, "
                       f"增强={initial_main_weights[1]:.3f}, "
                       f"增强内部分配: 相似度={initial_enhance_weights[0]:.3f} (启用={self.use_similarity_enhancer}), "
                       f"提示图={initial_enhance_weights[1]:.3f} (启用={self.use_prompt_enhancer})")
        else:
            self.fusion_main_weights_logits = None
            self.fusion_enhance_weights_logits = None
        
        # 提示图增强模块（保留原有功能，可通过配置禁用）
        if self.use_prompt_enhancer:
            num_prompt_samples = getattr(flags, 'num_prompt_samples', 3)  # 使用3个最重要的示例
            max_hops = getattr(flags, 'max_hops', 1)  # 使用1跳邻居
            self.prompt_enhancer = OptimizedPromptGraph(
                embedding_dim=64,
                max_hops=max_hops,  # 使用1跳邻居
                num_prompt_samples=num_prompt_samples  # 使用最重要的几个示例
            )
        else:
            self.prompt_enhancer = None
        
        # 基于相似度的关系增强模块（新增，可通过配置禁用）
        if self.use_similarity_enhancer:
            max_similar_relations = getattr(flags, 'max_similar_relations', 10)  # 只使用top-10个最相似的关系
            
            # 使用合理的默认初始值（不依赖配置文件）
            # similarity_threshold: 初始化为0.75（中等阈值，经过sigmoid映射）
            # enhancement_strength: 初始化为0.05（较小的增强强度，避免一开始就过度影响）
            self.similarity_enhancer = SimilarityBasedRelationEnhancer(
                embedding_dim=64,
                similarity_threshold_init=0.75,  # 合理的默认值：中等相似度阈值
                enhancement_strength_init=0.05,  # 合理的默认值：较小的增强强度
                max_similar_relations=max_similar_relations  # 只使用最重要的几个关系
            )
        else:
            self.similarity_enhancer = None
        
        # 自适应增强门控网络（根据配置决定是否启用）
        # 注意：门控网络需要配合相似度增强器使用
        if self.use_adaptive_gate and self.use_similarity_enhancer:
            self.enhancement_gate = AdaptiveEnhancementGate(embedding_dim=64)
        else:
            self.enhancement_gate = None
            # 如果启用了门控但没有启用相似度增强，发出警告
            if self.use_adaptive_gate and not self.use_similarity_enhancer:
                import warnings
                warnings.warn("use_adaptive_gate=True but use_similarity_enhancer=False. "
                            "Adaptive gate requires similarity enhancer, disabling gate.")
                self.use_adaptive_gate = False
        
        # 实体增强已移除，直接使用原始entity_model
        self.use_entity_enhancement = False
        self.entity_enhancer = None
        
        # 增强置信度机制（用于评估增强增量的可靠性）
        # 如果置信度低，则减少增强的影响
        embedding_dim = 64  # 固定为64维（与模型其他部分一致）
        self.use_enhancement_confidence = getattr(flags, 'use_enhancement_confidence', True)  # 默认启用
        if self.use_enhancement_confidence:
            self.enhancement_confidence = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim // 2),  # 输入：r和combined_delta
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()  # 输出0-1之间的置信度
            )
        else:
            self.enhancement_confidence = None
        
        # 存储表示
        self.relation_representations_structural = None
        self.relation_representations_semantic = None
        self.final_relation_representations = None
        self.enhanced_relation_representations = None
        
        # 打印模型配置信息
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("[EnhancedUltra] 模型初始化完成")
        logger.info(f"  - 相似度增强器: {'启用' if self.use_similarity_enhancer else '禁用'}")
        logger.info(f"  - 提示图增强器: {'启用' if self.use_prompt_enhancer else '禁用'}")
        logger.info(f"  - 可学习融合: {'启用' if self.use_learnable_fusion else '禁用'}")
        logger.info(f"  - 增强置信度: {'启用' if self.use_enhancement_confidence else '禁用'}")
        if self.use_prompt_enhancer:
            logger.info(f"  - 提示图参数: max_hops={getattr(flags, 'max_hops', 1)}, num_samples={getattr(flags, 'num_prompt_samples', 3)}")
        if self.use_similarity_enhancer:
            logger.info(f"  - 相似度参数: max_similar_relations={getattr(flags, 'max_similar_relations', 10)}")
        logger.info("=" * 80)
        
    def forward(self, data, batch, is_tail=False):
        """增强版前向传播 - 根据配置使用自适应门控机制"""
        logger = logging.getLogger(__name__)
        batch_size = len(batch)
        
        # 减少日志打印频率：每50个batch打印一次
        if not hasattr(self, '_forward_log_counter'):
            self._forward_log_counter = 0
        self._forward_log_counter += 1
        if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
            logger.info(f"[EnhancedUltra] 开始前向传播，batch_size={batch_size}, 训练模式={self.training} (batch #{self._forward_log_counter})")
        
        query_rels = batch[:, 0, 2]
        query_rels_traverse = batch[:, 0, :]
        query_entities = batch[:, 0, 0]  # 查询实体（head）
        
        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.debug(f"[EnhancedUltra] GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
        
        # 获取基础关系表示（使用原始逻辑）
        # 使用在__init__中已加载的flags，避免重复加载
        flags = self.flags
        
        if flags.run == "semma" or flags.run == "EnhancedUltra":
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info("[EnhancedUltra] 步骤1/4: 获取基础关系表示 (SEMMA模式)")
            with timer("Relation Model (structural)", logger):
                self.relation_representations_structural = self.relation_model(data, query=query_rels)
            logger.debug(f"  - 结构关系表示形状: {self.relation_representations_structural.shape}")
            
            with timer("Semantic Model", logger):
                self.relation_representations_semantic = self.semantic_model(data, query=query_rels)
            logger.debug(f"  - 语义关系表示形状: {self.relation_representations_semantic.shape}")
            
            with timer("Combiner", logger):
                self.final_relation_representations = self.combiner(
                    self.relation_representations_structural, 
                    self.relation_representations_semantic
                )
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 融合后关系表示形状: {self.final_relation_representations.shape}")
        else:
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info("[EnhancedUltra] 步骤1/4: 获取基础关系表示 (单模型模式)")
            with timer("Relation Model", logger):
                self.relation_representations_structural = self.relation_model(data, query=query_rels)
            self.final_relation_representations = self.relation_representations_structural
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 关系表示形状: {self.final_relation_representations.shape}")
        
        # 应用增强模块（并行融合方式）
        # 原始表示 r (SEMMA融合后的嵌入)
        r = self.final_relation_representations  # [batch_size, num_relations, embedding_dim]
        batch_size = len(query_rels)
        
        if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
            logger.info(f"[EnhancedUltra] 步骤2/4: 应用增强模块，r形状={r.shape}")
        
        # 并行获取两个增强器的增量（都基于原始表示r）
        # r1: similarity_enhancer的增量
        if self.use_similarity_enhancer and self.similarity_enhancer is not None:
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 应用相似度增强器 (max_similar_relations={self.similarity_enhancer.max_similar_relations})")
            with timer("Similarity Enhancer", logger):
                r1_delta = self.similarity_enhancer(
                    r, 
                    query_rels,
                    return_enhancement_only=True  # 只返回增强增量
                )  # [batch_size, num_relations, embedding_dim]
            r1_delta_norm = torch.norm(r1_delta, dim=-1).mean().item()
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 相似度增强完成，r1_delta形状={r1_delta.shape}，平均L2范数={r1_delta_norm:.4f}")
        else:
            r1_delta = torch.zeros_like(r)
            logger.debug("  - 相似度增强器未启用，使用零增量")
        
        # r2: prompt_enhancer的增量（只增强查询关系）- 批量处理优化
        if self.use_prompt_enhancer and self.prompt_enhancer is not None:
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 应用提示图增强器 (max_hops={self.prompt_enhancer.max_hops}, num_samples={self.prompt_enhancer.num_prompt_samples}) [批量处理模式]")
            r2_delta = torch.zeros_like(r)
            prompt_success_count = 0
            prompt_fail_count = 0
            
            with timer(f"Prompt Enhancer (batch_size={batch_size})", logger):
                try:
                    # 批量处理：一次性为整个batch找到查询关系的边
                    selected_edges_list, selected_edge_types_list = self.prompt_enhancer.generate_prompt_graph_batch(
                        data, query_rels, query_entities
                    )
                    
                    # 批量编码提示图上下文
                    # 注意：每个batch使用自己的关系嵌入r[i]，但encode_prompt_context_batch需要整个关系嵌入矩阵
                    # 这里我们传递r[0]作为参考（实际上每个样本会使用自己的query_rel对应的嵌入）
                    prompt_contexts = self.prompt_enhancer.encode_prompt_context_batch(
                        selected_edges_list, selected_edge_types_list, query_rels, r, query_entities, data
                    )  # [batch_size, embedding_dim]
                    
                    # 批量计算增强增量
                    for i in range(batch_size):
                        query_rel = query_rels[i]
                        base_repr = r[i, query_rel, :]  # [embedding_dim]
                        prompt_context = prompt_contexts[i]  # [embedding_dim]
                        
                        if selected_edges_list[i] is not None:
                            # 计算增强增量（与r1_delta保持一致，都是增量形式）
                            fusion_input = torch.cat([base_repr, prompt_context], dim=-1)  # [2*embedding_dim]
                            enhanced_embedding = self.prompt_enhancer.context_fusion(fusion_input)  # [embedding_dim]
                            # 统一为增量形式：enhanced_embedding - base_repr
                            prompt_delta = enhanced_embedding - base_repr
                            r2_delta[i, query_rel, :] = prompt_delta
                            prompt_success_count += 1
                            logger.debug(f"    batch {i}: 选中边数={selected_edges_list[i].shape[1]}")
                        else:
                            prompt_delta = torch.zeros_like(base_repr)
                            r2_delta[i, query_rel, :] = prompt_delta
                            prompt_fail_count += 1
                            logger.debug(f"    batch {i}: 未找到查询关系的边")
                            
                except Exception as e:
                    logger.warning(f"[EnhancedUltra] prompt_enhancer批量处理失败，回退到逐个处理: {e}")
                    # 回退到逐个处理
                    for i in range(batch_size):
                        query_rel = query_rels[i]
                        query_entity = query_entities[i]
                        base_repr = r[i, query_rel, :]
                        try:
                            selected_edges, selected_edge_types = self.prompt_enhancer.generate_prompt_graph(
                                data, query_rel, query_entity
                            )
                            if selected_edges is not None:
                                prompt_context = self.prompt_enhancer.encode_prompt_context(
                                    selected_edges, selected_edge_types, query_rel, r[i], query_entity, data
                                )
                                fusion_input = torch.cat([base_repr, prompt_context], dim=-1)
                                enhanced_embedding = self.prompt_enhancer.context_fusion(fusion_input)
                                prompt_delta = enhanced_embedding - base_repr
                                r2_delta[i, query_rel, :] = prompt_delta
                                prompt_success_count += 1
                            else:
                                r2_delta[i, query_rel, :] = torch.zeros_like(base_repr)
                                prompt_fail_count += 1
                        except Exception as e2:
                            prompt_fail_count += 1
                            logger.warning(f"[EnhancedUltra] prompt_enhancer在batch {i}失败: {e2}")
                            r2_delta[i, query_rel, :] = torch.zeros_like(base_repr)
            
            r2_delta_norm = torch.norm(r2_delta, dim=-1).mean().item()
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 提示图增强完成，r2_delta形状={r2_delta.shape}，平均L2范数={r2_delta_norm:.4f}，成功={prompt_success_count}/{batch_size}")
        else:
            r2_delta = torch.zeros_like(r)
            logger.debug("  - 提示图增强器未启用，使用零增量")
        
        # 并行融合：使用可学习权重（2权重归一化方案）
        if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
            logger.info(f"[EnhancedUltra] 步骤3/4: 融合增强表示")
        if self.use_learnable_fusion and self.fusion_main_weights_logits is not None:
            # 使用softmax归一化主权重（2个权重归一化）
            main_weights = F.softmax(self.fusion_main_weights_logits, dim=0)  # [2]
            # main_weights[0]: 原始r的权重 (w_base)
            # main_weights[1]: 增强的权重 (w_enhance)
            
            # 增强器内部权重分配（用于分配r1_delta和r2_delta的权重）
            enhance_weights = F.softmax(self.fusion_enhance_weights_logits, dim=0)  # [2]
            # enhance_weights[0]: similarity增强的权重 (w_sim)
            # enhance_weights[1]: prompt增强的权重 (w_prompt)
            
            # 定期打印融合权重（每100个batch打印一次，避免日志过多）
            if not hasattr(self, '_fusion_log_counter'):
                self._fusion_log_counter = 0
            self._fusion_log_counter += 1
            if self._fusion_log_counter % 100 == 0 or self._fusion_log_counter == 1:
                logger.info(f"  - 融合权重 (可学习): base={main_weights[0]:.3f}, enhance={main_weights[1]:.3f}, "
                           f"sim={enhance_weights[0]:.3f}, prompt={enhance_weights[1]:.3f}")
            
            # 2权重融合公式：enhanced = w_base*r + w_enhance*(r + w_sim*r1_delta + w_prompt*r2_delta)
            # 展开后：enhanced = (w_base + w_enhance)*r + w_enhance*(w_sim*r1_delta + w_prompt*r2_delta)
            # 因为权重归一化，w_base + w_enhance = 1，所以等价于：
            # enhanced = r + w_enhance*(w_sim*r1_delta + w_prompt*r2_delta)
            # 这样设计更清晰，避免了3权重方案中的冗余问题
            
            # 计算增强增量（加权组合两个增强器的增量）
            combined_delta = enhance_weights[0] * r1_delta + enhance_weights[1] * r2_delta
            
            # 使用置信度机制评估增强增量的可靠性
            if self.use_enhancement_confidence and self.enhancement_confidence is not None:
                # 计算置信度：基于r和combined_delta的相似性
                # 对于每个关系，计算置信度
                confidence_input = torch.cat([r, combined_delta], dim=-1)  # [batch, num_rels, 2*dim]
                confidence = self.enhancement_confidence(confidence_input)  # [batch, num_rels, 1]
                confidence = confidence.squeeze(-1)  # [batch, num_rels]
                
                # 打印置信度统计信息
                avg_confidence = confidence.mean().item()
                min_confidence = confidence.min().item()
                max_confidence = confidence.max().item()
                if self._fusion_log_counter % 100 == 0 or self._fusion_log_counter == 1:
                    logger.info(f"  - 增强置信度: 平均={avg_confidence:.3f}, 最小={min_confidence:.3f}, 最大={max_confidence:.3f}")
                
                # 应用置信度：低置信度时减少增强的影响
                combined_delta = combined_delta * confidence.unsqueeze(-1)  # [batch, num_rels, dim]
            
            if self.use_adaptive_gate and self.enhancement_gate is not None and self.use_similarity_enhancer:
                # 使用自适应门控机制：计算门控权重
                gate_weights = self.enhancement_gate(
                    r,
                    query_rels,
                    query_entities,
                    data
                )  # [batch_size]
                gate_weights_expanded = gate_weights.view(batch_size, 1, 1)
                
                # 2权重融合 + 自适应门控：w_base*r + gate*w_enhance*(r + combined_delta)
                # 注意：门控只影响增强部分，不影响原始表示
                self.enhanced_relation_representations = (
                    main_weights[0] * r + 
                    gate_weights_expanded * main_weights[1] * (r + combined_delta)
                )
            else:
                # 2权重融合：w_base*r + w_enhance*(r + combined_delta)
                self.enhanced_relation_representations = (
                    main_weights[0] * r + 
                    main_weights[1] * (r + combined_delta)
                )
            
            # 计算增强前后的差异统计
            diff = torch.abs(self.enhanced_relation_representations - r).mean().item()
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 融合完成，增强前后平均差异={diff:.4f}")
        else:
            # 固定权重融合（回退到方案1）：r + u*r1 + θ*r2
            # 其中 u = similarity_enhancer_weight, θ = prompt_enhancer_weight
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 使用固定权重融合: sim_weight={self.similarity_enhancer_weight:.3f}, prompt_weight={self.prompt_enhancer_weight:.3f}")
            
            # 使用置信度机制评估增强增量的可靠性
            combined_delta_fixed = self.similarity_enhancer_weight * r1_delta + self.prompt_enhancer_weight * r2_delta
            if self.use_enhancement_confidence and self.enhancement_confidence is not None:
                confidence_input = torch.cat([r, combined_delta_fixed], dim=-1)  # [batch, num_rels, 2*dim]
                confidence = self.enhancement_confidence(confidence_input).squeeze(-1)  # [batch, num_rels]
                avg_confidence = confidence.mean().item()
                logger.info(f"  - 增强置信度: 平均={avg_confidence:.3f}")
                combined_delta_fixed = combined_delta_fixed * confidence.unsqueeze(-1)  # [batch, num_rels, dim]
            
            if self.use_adaptive_gate and self.enhancement_gate is not None and self.use_similarity_enhancer:
                # 使用自适应门控机制：计算门控权重
                gate_weights = self.enhancement_gate(
                    r,
                    query_rels,
                    query_entities,
                    data
                )  # [batch_size]
                gate_weights_expanded = gate_weights.view(batch_size, 1, 1)
                
                # 并行融合：r + gate_weight * combined_delta（已应用置信度）
                self.enhanced_relation_representations = (
                    r + 
                    gate_weights_expanded * combined_delta_fixed
                )
            else:
                # 不使用门控机制：直接并行融合 r + combined_delta（已应用置信度）
                self.enhanced_relation_representations = (
                    r + 
                    combined_delta_fixed
                )
        
        # 使用最终的关系表示进行实体推理
        if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
            logger.info(f"[EnhancedUltra] 步骤4/4: 实体推理，enhanced_relation_representations形状={self.enhanced_relation_representations.shape}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - memory_reserved
            logger.debug(f"[EnhancedUltra] 实体推理前GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB, 空闲={memory_free:.2f}GB")
            
            # 内存优化：如果内存紧张，清理GPU缓存
            if memory_free < 1.0:  # 如果空闲内存小于1GB
                torch.cuda.empty_cache()  # 清理GPU缓存
                logger.warning(f"[EnhancedUltra] 内存紧张，清理GPU缓存")
        
        try:
            # 使用原始entity_model进行实体推理（不传递实体增强相关参数）
            with timer("Entity Model", logger):
                score = self.entity_model(data, self.enhanced_relation_representations, batch)
            if self._forward_log_counter % 50 == 0 or self._forward_log_counter == 1:
                logger.info(f"  - 实体推理完成，score形状={score.shape}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"[EnhancedUltra] GPU内存不足: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"[EnhancedUltra] 已清理GPU缓存")
            # 所有RuntimeError都重新抛出
            raise
    
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.debug(f"[EnhancedUltra] Forward完成，GPU内存: {memory_allocated:.2f}GB")
        else:
            logger.info(f"[EnhancedUltra] 前向传播完成")
    
        return score
    