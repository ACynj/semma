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
        
        # 初始化增强后的表示
        if return_enhancement_only:
            # 如果只返回增强增量，初始化为零（只更新查询关系的位置）
            enhanced_reprs = torch.zeros_like(final_relation_representations)
        else:
            # 如果返回混合后的表示，初始化为原始表示
            enhanced_reprs = final_relation_representations.clone()
        
        for i in range(batch_size):
            query_rel_idx = query_rels[i].item()
            if query_rel_idx >= num_relations or query_rel_idx < 0:
                continue
            
            # 获取查询关系的表示
            query_rel_repr = final_relation_representations[i, query_rel_idx, :]  # [embedding_dim]
            
            # 获取所有关系的表示
            all_rel_reprs = final_relation_representations[i, :, :]  # [num_relations, embedding_dim]
            
            # 计算与所有关系的余弦相似度
            # 归一化
            query_norm = F.normalize(query_rel_repr, p=2, dim=0)  # [embedding_dim]
            all_norms = F.normalize(all_rel_reprs, p=2, dim=1)  # [num_relations, embedding_dim]
            
            # 计算余弦相似度 (使用矩阵乘法更高效)
            similarities = torch.matmul(query_norm.unsqueeze(0), all_norms.t()).squeeze(0)  # [num_relations]
            
            # 排除查询关系本身（设置为-1，确保不被选择）
            similarities[query_rel_idx] = -1.0
            
            # 找到相似度大于阈值的关系
            # 使用soft thresholding让阈值也参与梯度计算（使用sigmoid平滑）
            # 这比硬阈值更好，因为可以计算梯度
            threshold_scaled = threshold.unsqueeze(0).expand_as(similarities)  # [num_relations]
            # 使用sigmoid实现平滑的阈值过滤，保留梯度
            # similarity_weight = sigmoid((similarity - threshold) * temperature_scale)
            # 这样相似度高的会有较大权重，相似度低的会有较小权重
            similarity_weights = torch.sigmoid((similarities - threshold) * 10.0)  # [num_relations]
            
            # 找到权重足够大的关系（>0.5相当于相似度>阈值）
            above_threshold = similarity_weights > 0.5
            valid_indices = torch.where(above_threshold)[0]
            
            if len(valid_indices) == 0:
                # 如果没有超过阈值的，保持原样（零增量或原始表示）
                if not return_enhancement_only:
                    enhanced_reprs[i, query_rel_idx, :] = query_rel_repr
                continue
            
            # 优化：只使用top-k个最相似的关系（大幅减少计算量）
            valid_similarities_raw = similarities[valid_indices]  # [num_valid]
            
            # 选择top-k个最相似的关系
            if len(valid_indices) > self.max_similar_relations:
                # 获取top-k个最相似的关系的索引
                top_k_values, top_k_local_indices = torch.topk(valid_similarities_raw, 
                                                               k=min(self.max_similar_relations, len(valid_indices)),
                                                               dim=0)
                # 将局部索引转换为全局索引
                top_k_indices = valid_indices[top_k_local_indices]
                valid_indices = top_k_indices
                valid_similarities_raw = top_k_values
            
            # 使用平滑的权重而不是硬阈值
            valid_weights_smooth = similarity_weights[valid_indices]  # [num_valid]
            
            # 获取有效关系的表示
            valid_reprs = all_rel_reprs[valid_indices, :]  # [num_valid, embedding_dim]
            
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
        
        # 简化的提示图编码器
        self.prompt_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
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
        """快速生成提示图（优化：使用1跳邻居）
        
        Returns:
            prompt_graph: 提示图对象，或None
            prompt_entities: 提示图中的实体列表（用于节点初始化）
        """
        if num_samples is None:
            num_samples = self.num_prompt_samples
            
        # 优化的提示图生成 - 使用1跳邻域
        device = query_entity.device
        
        # 1. 找到包含查询关系的边（用于采样示例）
        with timer("查找查询关系的边", logger=None, min_time_ms=50):
            edge_mask = (data.edge_type == query_relation)
            query_edges = data.edge_index[:, edge_mask]
        
        # 2. 从查询实体开始，获取1跳邻居
        query_entity_id = query_entity.item() if isinstance(query_entity, torch.Tensor) else query_entity
        seed_entities = {query_entity_id}
        
        # 3. 获取查询实体及其1跳邻居
        prompt_entities = self._get_k_hop_neighbors(data, seed_entities, max_hops=self.max_hops)
        
        # 4. 如果有查询关系的边，只选择最重要的几个示例（优化：按重要性排序）
        if query_edges.shape[1] > 0:
            # 优化：只使用最重要的几个示例（按度排序，选择连接度高的实体）
            if query_edges.shape[1] > num_samples:
                # 计算每条边的重要性（使用源实体和目标实体的度之和）
                head_nodes = query_edges[0]
                tail_nodes = query_edges[1]
                
                # 快速计算实体度（使用bincount）
                all_nodes = torch.cat([data.edge_index[0], data.edge_index[1]])
                node_degrees = torch.bincount(all_nodes, minlength=data.num_nodes)
                
                # 计算每条边的重要性（源实体度 + 目标实体度）
                edge_importance = node_degrees[head_nodes] + node_degrees[tail_nodes]
                
                # 选择重要性最高的num_samples条边
                _, top_indices = torch.topk(edge_importance, k=min(num_samples, query_edges.shape[1]), dim=0)
                sampled_edges = query_edges[:, top_indices]
            else:
                sampled_edges = query_edges
            
            # 添加采样边的实体（直接添加，不找邻居）
            with timer("添加采样边的实体", logger=None, min_time_ms=50):
                # 使用向量化操作
                head_nodes = sampled_edges[0]
                tail_nodes = sampled_edges[1]
                prompt_entities.update(head_nodes.cpu().tolist())
                prompt_entities.update(tail_nodes.cpu().tolist())
        
        # 5. 转换为有序列表（保证可重复性）
        prompt_entities_list = sorted(list(prompt_entities))
        
        if len(prompt_entities_list) == 0:
            return None, []
        
        # 6. 构建简化的子图（只包含相关实体）
        # 过滤边，只保留连接prompt_entities的边（使用向量化操作优化）
        logger = logging.getLogger(__name__)
        with timer(f"过滤边 (总边数={data.edge_index.shape[1]}, 实体数={len(prompt_entities_list)})", logger):
            if data.edge_index.numel() > 0:
                prompt_entities_set = set(prompt_entities_list)
                # 使用向量化操作替代循环（快很多）
                # 将prompt_entities转换为tensor以便向量化比较
                prompt_entities_tensor = torch.tensor(list(prompt_entities_set), device=device, dtype=torch.long)
                
                # 向量化检查：src和dst是否都在prompt_entities中
                src_nodes = data.edge_index[0]  # [num_edges]
                dst_nodes = data.edge_index[1]  # [num_edges]
                
                # 使用广播和isin操作（比循环快很多）
                src_in_prompt = torch.isin(src_nodes, prompt_entities_tensor)
                dst_in_prompt = torch.isin(dst_nodes, prompt_entities_tensor)
                edge_mask = src_in_prompt & dst_in_prompt
                
                filtered_edge_index = data.edge_index[:, edge_mask]
                filtered_edge_type = data.edge_type[edge_mask]
            else:
                filtered_edge_index = data.edge_index
                filtered_edge_type = data.edge_type
        
        # 构建提示图
        prompt_graph = Data(
            edge_index=filtered_edge_index,
            edge_type=filtered_edge_type,
            num_nodes=len(prompt_entities_list)
        )
        
        return prompt_graph, prompt_entities_list
    
    def encode_prompt_context(self, prompt_graph, query_relation, relation_embeddings=None, prompt_entities=None, data=None, entity_model=None, relation_representations=None):
        """快速编码提示图上下文
        
        Args:
            prompt_graph: 提示图
            query_relation: 查询关系索引
            relation_embeddings: 关系嵌入矩阵 [num_relations, embedding_dim]，可选
            prompt_entities: 提示图中的实体列表，用于为每个节点生成不同的初始化
            data: 图数据对象，用于获取实体相关的边信息或计算实体特征
            entity_model: EntityNBFNet模型，用于计算实体特征（方案2：最优方案）
            relation_representations: 关系表示 [num_relations, embedding_dim]，用于EntityNBFNet计算
        """
        if prompt_graph is None:
            device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
            return torch.zeros(self.embedding_dim, device=device)
            
        # 简化的编码 - 使用平均池化
        device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
        
        # 方案2：使用EntityNBFNet计算实体特征（最优方案，有语义意义且考虑图结构）
        # 优化：限制实体数量以提高速度，只计算最重要的实体
        # 优化：使用缓存避免重复计算bellmanford（大幅提升速度）
        # 快速模式：如果实体数量很少，直接使用关系平均嵌入（更快）
        USE_FAST_MODE = True  # 快速模式：实体数<=10时跳过EntityNBFNet，直接使用关系平均嵌入
        use_entity_nbfnet = True  # 尝试启用，如果失败会自动回退
        MAX_ENTITIES_FOR_NBFNET = 30  # 限制最多计算30个实体（快速模式：从100降到30，大幅提升速度）
        node_embeddings = None
        if use_entity_nbfnet and entity_model is not None and relation_representations is not None and data is not None and prompt_entities is not None and len(prompt_entities) == prompt_graph.num_nodes:
            try:
                # 优化：检查缓存（避免重复计算bellmanford）
                # 使用relation_representations的hash作为缓存key
                import hashlib
                import logging
                logger = logging.getLogger(__name__)
                cache_key = None
                if relation_representations is not None and isinstance(relation_representations, torch.Tensor):
                    # 优化：使用内容hash而不是data_ptr，提高缓存命中率
                    # 使用前100个元素和形状作为hash（快速且能区分不同的关系嵌入）
                    rel_repr_sample = relation_representations.flatten()[:100].detach().cpu().numpy()
                    import hashlib
                    rel_repr_bytes = rel_repr_sample.tobytes() + str(tuple(relation_representations.shape)).encode()
                    rel_repr_hash = int(hashlib.md5(rel_repr_bytes).hexdigest()[:16], 16)  # 使用MD5的前16位作为hash
                    # 使用data的hash（边数和节点数）
                    data_hash = hash((
                        data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0,
                        data.num_nodes if hasattr(data, 'num_nodes') else 0
                    ))
                    cache_key = (rel_repr_hash, data_hash)
                
                # 检查缓存
                cached_result = None
                if cache_key is not None and cache_key in self._bfnet_cache:
                    cached_result = self._bfnet_cache[cache_key]
                    self._cache_hits += 1
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[Prompt Enhancer] 缓存命中！已复用EntityNBFNet结果 (缓存命中率: {self._cache_hits}/{self._cache_hits + self._cache_misses})")
                
                # 优化：限制实体数量以提高速度
                prompt_entities_list = list(prompt_entities)
                
                # 快速模式：如果实体数量很少（<=10），直接使用关系平均嵌入（更快）
                if USE_FAST_MODE and len(prompt_entities_list) <= 10 and cached_result is None:
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[Prompt Enhancer] 快速模式：实体数({len(prompt_entities_list)})<=10，跳过EntityNBFNet，使用关系平均嵌入")
                    use_entity_nbfnet = False  # 跳过EntityNBFNet计算
                
                if len(prompt_entities_list) > MAX_ENTITIES_FOR_NBFNET:
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[Prompt Enhancer] 实体数量({len(prompt_entities_list)})超过限制({MAX_ENTITIES_FOR_NBFNET})，"
                                f"优先保留查询实体，然后按度排序选择前{MAX_ENTITIES_FOR_NBFNET}个实体")
                    
                    # 快速计算实体度（使用bincount）
                    all_nodes = torch.cat([data.edge_index[0], data.edge_index[1]])
                    node_degrees = torch.bincount(all_nodes, minlength=data.num_nodes)
                    
                    # 优先保留查询实体，然后按度排序
                    query_entity_id = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
                    query_entity_set = {query_entity_id} if query_entity_id in prompt_entities_list else set()
                    
                    entity_degrees_dict = {e: node_degrees[e].item() if e < len(node_degrees) else 0 
                                          for e in prompt_entities_list}
                    sorted_entities = sorted(
                        prompt_entities_list,
                        key=lambda e: (e not in query_entity_set, -entity_degrees_dict.get(e, 0))
                    )
                    prompt_entities_list = sorted_entities[:MAX_ENTITIES_FOR_NBFNET]
                
                # 使用EntityNBFNet的bellmanford计算实体特征（如果快速模式未启用）
                # 为所有实体批量计算（使用查询关系作为虚拟关系）
                if not use_entity_nbfnet:
                    # 快速模式：直接使用关系平均嵌入
                    node_embeddings = self._fallback_entity_embedding(prompt_entities_list, query_relation, relation_representations, data, device)
                else:
                    query_rel_idx = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
                    
                    # 准备批量输入：为每个实体创建一个查询
                    num_entities = len(prompt_entities_list)
                    h_indices = torch.tensor(prompt_entities_list, device=device, dtype=torch.long)  # [num_entities]
                    # 使用查询关系作为虚拟关系（或者使用0作为默认关系）
                    r_indices = torch.full((num_entities,), query_rel_idx, device=device, dtype=torch.long)  # [num_entities]
                
                    # 获取实际的EntityNBFNet实例（处理EnhancedEntityNBFNet包装器）
                    actual_entity_model = entity_model
                    if hasattr(entity_model, 'entity_model'):
                        # 如果是EnhancedEntityNBFNet包装器，获取内部的entity_model
                        actual_entity_model = entity_model.entity_model
                    
                    # 检查actual_entity_model是否有bellmanford方法
                    if not hasattr(actual_entity_model, 'bellmanford'):
                        raise ValueError(f"entity_model没有bellmanford方法，类型: {type(actual_entity_model)}")
                
                    # 设置entity_model的query为relation_representations
                    # EntityNBFNet的forward方法中会设置self.query = relation_representations
                    # 但bellmanford需要query是[batch_size, num_relations, embedding_dim]格式
                    # 我们需要临时设置query，然后批量计算
                    original_query = None
                    if hasattr(actual_entity_model, 'query'):
                        original_query = actual_entity_model.query
                    
                    # 使用relation_representations作为query
                    # relation_representations应该是[num_relations, embedding_dim]
                    # 需要扩展为[batch_size, num_relations, embedding_dim]以匹配bellmanford的期望
                    # 检查relation_representations是否为None或无效
                    if relation_representations is None:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"[Prompt Enhancer] relation_representations为None，跳过EntityNBFNet计算")
                        raise ValueError("relation_representations不能为None")
                    
                    if not isinstance(relation_representations, torch.Tensor):
                        logger = logging.getLogger(__name__)
                        logger.warning(f"[Prompt Enhancer] relation_representations不是Tensor，类型: {type(relation_representations)}")
                        raise ValueError(f"relation_representations必须是Tensor，实际类型: {type(relation_representations)}")
                    
                    # 检查relation_representations是否有有效的数据
                    if relation_representations.numel() == 0:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"[Prompt Enhancer] relation_representations为空张量")
                        raise ValueError("relation_representations不能为空张量")
                    
                    if relation_representations.dim() == 2:
                        # 扩展为[batch_size, num_relations, embedding_dim]
                        # batch_size = num_entities（每个实体一个查询）
                        query_tensor = relation_representations.unsqueeze(0).expand(num_entities, -1, -1)
                        actual_entity_model.query = query_tensor
                    elif relation_representations.dim() == 3:
                        # 如果已经是3D，检查batch_size是否匹配
                        if relation_representations.shape[0] == 1:
                            actual_entity_model.query = relation_representations.expand(num_entities, -1, -1)
                        else:
                            actual_entity_model.query = relation_representations
                    else:
                        # 如果格式不对，回退到方案1
                        logger = logging.getLogger(__name__)
                        logger.warning(f"[Prompt Enhancer] relation_representations维度不正确: {relation_representations.shape}")
                        raise ValueError(f"relation_representations维度不正确: {relation_representations.shape}")
                    
                    # 验证query是否成功设置
                    if not hasattr(actual_entity_model, 'query') or actual_entity_model.query is None:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"[Prompt Enhancer] 无法设置entity_model.query")
                        raise ValueError("无法设置entity_model.query")
                    
                    # 设置layers的relation（EntityNBFNet的forward方法中会设置这个）
                    # 这对于bellmanford中的layers正常工作很重要
                    if hasattr(actual_entity_model, 'layers'):
                        # 使用relation_representations设置每个layer的relation
                        # 注意：当project_relations=True时，layer.relation应该是[num_relations, embedding_dim]
                        # 但需要确保num_relations >= data.edge_type.max() + 1（因为edge_type可能包含逆关系）
                        # 如果relation_representations的num_relations不够，需要扩展
                        max_edge_type = data.edge_type.max().item() if data.edge_type.numel() > 0 else 0
                        required_num_relations = max_edge_type + 1
                        actual_num_relations = relation_representations.shape[0]
                        
                        logger = logging.getLogger(__name__)
                        if actual_num_relations < required_num_relations:
                            # 需要扩展relation_representations以匹配edge_type的最大值
                            logger.warning(f"[Prompt Enhancer] relation_representations的关系数({actual_num_relations}) "
                                         f"小于edge_type的最大值({max_edge_type})，需要扩展到{required_num_relations}")
                            # 使用零向量填充
                            padding_size = required_num_relations - actual_num_relations
                            padding = torch.zeros(padding_size, relation_representations.shape[1], 
                                                device=relation_representations.device, 
                                                dtype=relation_representations.dtype)
                            extended_relation_representations = torch.cat([relation_representations, padding], dim=0)
                        else:
                            extended_relation_representations = relation_representations
                        
                        logger.debug(f"[Prompt Enhancer] 设置layers的relation: shape={extended_relation_representations.shape}, "
                                   f"required_num_relations={required_num_relations}, max_edge_type={max_edge_type}")
                        
                        # 设置每个layer的relation
                        for layer in actual_entity_model.layers:
                            if hasattr(layer, 'relation'):
                                layer.relation = extended_relation_representations
                    
                    # 计算实体特征（批量计算）
                    # 注意：这里使用torch.no_grad()以避免影响主模型的梯度
                    # 但如果是在训练时，可能需要保留梯度（取决于需求）
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[Prompt Enhancer] 使用EntityNBFNet计算{num_entities}个实体的特征，"
                                f"query形状={actual_entity_model.query.shape if hasattr(actual_entity_model, 'query') else 'None'}")
                    
                    # 优化：如果缓存命中，直接使用缓存结果
                    if cached_result is not None:
                        entity_features = cached_result["node_feature"]  # [num_entities, num_nodes, feature_dim]
                        logger.info(f"[Prompt Enhancer] ✅ 使用缓存的EntityNBFNet结果，特征形状: {entity_features.shape}")
                    else:
                        # 缓存未命中，需要计算
                        self._cache_misses += 1
                        logger.debug(f"[Prompt Enhancer] 缓存未命中，计算EntityNBFNet (缓存命中率: {self._cache_hits}/{self._cache_hits + self._cache_misses})")
                        
                        # 优化：使用torch.no_grad()和eval模式以提高速度并减少内存占用
                        actual_entity_model.eval()  # 临时设置为eval模式，避免dropout等
                        was_training = actual_entity_model.training
                        try:
                            with torch.no_grad():  # 推理时不需要梯度，避免影响主模型，同时提升速度
                                entity_features_dict = actual_entity_model.bellmanford(data, h_indices, r_indices)
                                entity_features = entity_features_dict["node_feature"]  # [num_entities, num_nodes, feature_dim]
                                logger.info(f"[Prompt Enhancer] ✅ EntityNBFNet计算成功！特征形状: {entity_features.shape}, "
                                           f"实体数={num_entities}, feature_dim={entity_features.shape[-1]}")
                                
                                # 缓存结果（如果缓存未满）
                                # 内存优化：如果缓存已满，清理最旧的缓存项（FIFO策略）
                                if cache_key is not None:
                                    if len(self._bfnet_cache) >= self._max_cache_size:
                                        # 清理最旧的缓存项（删除第一个）
                                        oldest_key = next(iter(self._bfnet_cache))
                                        del self._bfnet_cache[oldest_key]
                                        logger.debug(f"[Prompt Enhancer] 缓存已满，清理最旧缓存项 (当前大小: {len(self._bfnet_cache)})")
                                    self._bfnet_cache[cache_key] = entity_features_dict
                                    logger.debug(f"[Prompt Enhancer] 已缓存EntityNBFNet结果 (缓存大小: {len(self._bfnet_cache)})")
                        except Exception as e:
                            logger.warning(f"[Prompt Enhancer] EntityNBFNet计算失败: {e}")
                            raise
                        finally:
                            # 恢复原始状态
                            if original_query is not None:
                                actual_entity_model.query = original_query
                        actual_entity_model.train(was_training)
                    
                    # 优化：批量提取和投影实体特征（向量化操作）
                    # entity_features形状: [num_entities, num_nodes, feature_dim]
                    # 对于实体i，它在图中的节点ID是prompt_entities_list[i]，特征在entity_features[i, prompt_entities_list[i], :]
                    feature_dim = entity_features.shape[-1]
                    
                    # 批量提取特征：使用索引选择
                    entity_indices = torch.arange(num_entities, device=device)
                    node_indices = torch.tensor(prompt_entities_list, device=device, dtype=torch.long)
                    
                    # 确保索引在有效范围内
                    valid_mask = (entity_indices < entity_features.shape[0]) & (node_indices < entity_features.shape[1])
                    valid_entity_indices = entity_indices[valid_mask]
                    valid_node_indices = node_indices[valid_mask]
                    
                    # 批量提取特征 [num_valid, feature_dim]
                    if len(valid_entity_indices) > 0:
                        extracted_features = entity_features[valid_entity_indices, valid_node_indices, :]  # [num_valid, feature_dim]
                        
                        # 使用投影层将feature_dim映射到embedding_dim
                        feature_dim_str = str(feature_dim)
                        if feature_dim_str in self.entity_feature_proj:
                            proj_layer = self.entity_feature_proj[feature_dim_str]
                        else:
                            # 动态创建投影层（优化：只创建实际使用的，节省参数）
                            # 对于常见维度（128, 448），使用单层Linear（更高效）
                            # 对于其他维度，使用两层MLP（更灵活）
                            if feature_dim in [128, 448]:
                                # 单层Linear：更高效，参数更少
                                proj_layer = nn.Linear(feature_dim, self.embedding_dim).to(device)
                            else:
                                # 两层MLP：适用于其他维度
                                proj_layer = nn.Sequential(
                                    nn.Linear(feature_dim, self.embedding_dim * 2),
                                    nn.ReLU(),
                                    nn.Linear(self.embedding_dim * 2, self.embedding_dim)
                                ).to(device)
                            # 保存到ModuleDict中（需要先注册为子模块）
                            self.add_module(f'entity_feature_proj_{feature_dim_str}', proj_layer)
                            self.entity_feature_proj[feature_dim_str] = proj_layer
                        
                        # 批量投影 [num_valid, feature_dim] -> [num_valid, embedding_dim]
                        projected_features = proj_layer(extracted_features)  # [num_valid, embedding_dim]
                        
                        # 内存优化：释放中间结果（extracted_features不再需要）
                        del extracted_features
                        
                        # 初始化所有节点的嵌入（使用全局平均作为默认值）
                        global_avg = relation_embeddings.mean(dim=0) if relation_embeddings is not None and relation_embeddings.shape[0] > 0 else None
                        node_embeddings = torch.zeros(prompt_graph.num_nodes, self.embedding_dim, device=device)
                        
                        if global_avg is not None:
                            node_embeddings[:] = global_avg.unsqueeze(0).expand(prompt_graph.num_nodes, -1)
                        
                        # 将投影后的特征赋值给对应的节点
                        valid_prompt_indices = torch.arange(len(prompt_entities_list), device=device)[valid_mask]
                        node_embeddings[valid_prompt_indices] = projected_features
                    else:
                        # 如果没有有效特征，使用全局平均
                        global_avg = relation_embeddings.mean(dim=0) if relation_embeddings is not None and relation_embeddings.shape[0] > 0 else None
                        node_embeddings = torch.zeros(prompt_graph.num_nodes, self.embedding_dim, device=device)
                        if global_avg is not None:
                            node_embeddings[:] = global_avg.unsqueeze(0).expand(prompt_graph.num_nodes, -1)
                
            except Exception as e:
                # 如果EntityNBFNet计算失败，回退到方案1（使用关系平均嵌入）
                logger = logging.getLogger(__name__)
                logger.warning(f"[Prompt Enhancer] EntityNBFNet计算失败，回退到关系平均嵌入方案: {e}")
                import warnings
                # 只在DEBUG模式下显示警告，避免日志过多
                if logger.level <= logging.DEBUG:
                    warnings.warn(f"EntityNBFNet计算失败，回退到关系平均嵌入方案: {e}")
                node_embeddings = None  # 标记为失败，将使用方案1
        
        # 方案1：使用实体相关的所有关系的平均嵌入（回退方案）
        # 只有在方案2失败或未启用时才使用
        if node_embeddings is None:
            if relation_embeddings is not None and relation_embeddings.shape[0] > 0:
                node_embeddings = self._fallback_entity_embedding(
                    prompt_graph, query_relation, relation_embeddings, prompt_entities, data, device
                )
            else:
                # 回退：如果没有关系嵌入，使用改进的初始化策略
                if self.training:
                    # 训练时：使用随机初始化以增加多样性
                    node_embeddings = torch.randn(prompt_graph.num_nodes, self.embedding_dim, device=device)
                else:
                    # 推理时：使用小的固定值而不是零向量（比零向量稍好）
                    # 使用查询关系索引的哈希值作为种子，保证可重复性
                    import hashlib
                    seed = int(hashlib.md5(str(query_relation.item()).encode()).hexdigest()[:8], 16) % (2**32)
                    torch.manual_seed(seed)
                    node_embeddings = torch.randn(prompt_graph.num_nodes, self.embedding_dim, device=device) * 0.01
                    torch.manual_seed(torch.initial_seed())  # 恢复随机种子
        
        # 使用简化的编码器
        encoded_embeddings = self.prompt_encoder(node_embeddings)
        
        # 全局平均池化
        context_embedding = torch.mean(encoded_embeddings, dim=0)
        
        return context_embedding
    
    def _fallback_entity_embedding(self, prompt_graph, query_relation, relation_embeddings, prompt_entities, data, device):
        """回退方案：使用实体相关的所有关系的平均嵌入"""
        try:
            node_embeddings = torch.zeros(prompt_graph.num_nodes, self.embedding_dim, device=device)
        except RuntimeError as e:
            # 如果CUDA错误，尝试在CPU上创建然后移到GPU
            logger = logging.getLogger(__name__)
            logger.warning(f"[Prompt Enhancer] 在GPU上创建零向量失败，尝试在CPU上创建: {e}")
            node_embeddings = torch.zeros(prompt_graph.num_nodes, self.embedding_dim, device='cpu')
            if device.type == 'cuda':
                node_embeddings = node_embeddings.to(device)
        
        # 如果有实体列表和图数据，使用实体相关的所有关系的平均嵌入
        # 优化：批量计算所有实体的嵌入，避免循环
        if prompt_entities is not None and len(prompt_entities) == prompt_graph.num_nodes and data is not None:
            # 批量计算：为所有实体一次性找到相关的边
            prompt_entities_tensor = torch.tensor(prompt_entities, device=device, dtype=torch.long)
            
            # 向量化查找：找到所有与prompt_entities相关的边
            src_mask = torch.isin(data.edge_index[0], prompt_entities_tensor)
            dst_mask = torch.isin(data.edge_index[1], prompt_entities_tensor)
            relevant_edge_mask = src_mask | dst_mask
            
            if relevant_edge_mask.any():
                # 获取所有相关边的关系类型
                relevant_rels = data.edge_type[relevant_edge_mask]
                valid_rels = relevant_rels[relevant_rels < relation_embeddings.shape[0]]
                
                # 为每个实体计算其相关关系的平均嵌入
                for i, entity_id in enumerate(prompt_entities):
                    # 找到包含该实体的边（在相关边中）
                    entity_edge_mask = ((data.edge_index[0] == entity_id) | (data.edge_index[1] == entity_id)) & relevant_edge_mask
                    if entity_edge_mask.any():
                        # 获取这些边的关系类型
                        entity_rels = data.edge_type[entity_edge_mask]
                        valid_entity_rels = entity_rels[entity_rels < relation_embeddings.shape[0]]
                        if len(valid_entity_rels) > 0:
                            entity_emb = relation_embeddings[valid_entity_rels].mean(dim=0)
                        else:
                            entity_emb = relation_embeddings.mean(dim=0)
                    else:
                        entity_emb = relation_embeddings.mean(dim=0)
                    
                    node_embeddings[i] = entity_emb
            else:
                # 如果没有相关边，所有实体都使用全局平均
                global_avg = relation_embeddings.mean(dim=0)
                node_embeddings[:] = global_avg
        elif prompt_entities is not None and len(prompt_entities) == prompt_graph.num_nodes:
            # 如果没有图数据，使用查询关系嵌入 + 基于实体ID的小变化（回退方案）
            query_rel_idx = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
            if query_rel_idx < relation_embeddings.shape[0]:
                base_embedding = relation_embeddings[query_rel_idx]
            else:
                base_embedding = relation_embeddings.mean(dim=0)
            
            for i, entity_id in enumerate(prompt_entities):
                # 使用实体ID的哈希值生成确定性的变化（保证可重复性）
                import hashlib
                seed = int(hashlib.md5(str(entity_id).encode()).hexdigest()[:8], 16) % (2**32)
                torch.manual_seed(seed)
                variation = torch.randn(self.embedding_dim, device=device) * 0.1
                torch.manual_seed(torch.initial_seed())
                node_embeddings[i] = base_embedding + variation
        else:
            # 如果没有实体列表，使用查询关系嵌入 + 基于索引的变化
            query_rel_idx = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
            if query_rel_idx < relation_embeddings.shape[0]:
                base_embedding = relation_embeddings[query_rel_idx]
            else:
                base_embedding = relation_embeddings.mean(dim=0)
            
            for i in range(prompt_graph.num_nodes):
                import hashlib
                seed = int(hashlib.md5(str(i).encode()).hexdigest()[:8], 16) % (2**32)
                torch.manual_seed(seed)
                variation = torch.randn(self.embedding_dim, device=device) * 0.1
                torch.manual_seed(torch.initial_seed())
                node_embeddings[i] = base_embedding + variation
        
        # 训练时添加额外的随机噪声以增加多样性
        if self.training:
            noise_scale = 0.05  # 减小噪声幅度，因为已经有了基于实体的变化
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
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        
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
        
        # 增强器权重配置：控制两个增强器的贡献强度（用于固定权重模式）
        self.similarity_enhancer_weight = getattr(flags, 'similarity_enhancer_weight', 1.0)
        self.prompt_enhancer_weight = getattr(flags, 'prompt_enhancer_weight', 1.0)
        
        # 可学习的融合权重（方案3：可学习融合 - 增量融合方式）
        # 增量融合：r + w[0]*r1_delta + w[1]*r2_delta
        # 只学习两个增强器的权重，原始r直接保留（不加权）
        use_learnable_fusion = getattr(flags, 'use_learnable_fusion', True)  # 默认启用可学习融合
        self.use_learnable_fusion = use_learnable_fusion
        
        if self.use_learnable_fusion:
            # 初始化：基于flags.yaml中的权重，只初始化两个增强器的权重
            # 初始权重：[similarity_enhancer的权重, prompt_enhancer的权重]
            initial_sim_weight = self.similarity_enhancer_weight
            initial_prompt_weight = self.prompt_enhancer_weight
            
            # 归一化两个增强器的初始权重（使它们和为1）
            total = initial_sim_weight + initial_prompt_weight
            if total > 0:
                initial_weights = torch.tensor([
                    initial_sim_weight / total,
                    initial_prompt_weight / total
                ])
            else:
                # 如果总权重为0，使用均匀初始化
                initial_weights = torch.tensor([0.5, 0.5])
            
            # 转换为logits（使用log-softmax的逆变换）
            # 为了避免log(0)，添加小的epsilon
            epsilon = 1e-8
            initial_weights = torch.clamp(initial_weights, min=epsilon, max=1.0-epsilon)
            logits = torch.log(initial_weights)
            
            # 注册为可学习参数（只有2个权重：w[0]=similarity, w[1]=prompt）
            self.fusion_weights_logits = nn.Parameter(logits)
        else:
            self.fusion_weights_logits = None
        
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
            # 从flags.yaml读取初始值，如果没有则使用默认值
            similarity_threshold_init = getattr(flags, 'similarity_threshold_init', 0.8)
            enhancement_strength_init = getattr(flags, 'enhancement_strength_init', 0.05)
            max_similar_relations = getattr(flags, 'max_similar_relations', 10)  # 只使用top-10个最相似的关系
            
            self.similarity_enhancer = SimilarityBasedRelationEnhancer(
                embedding_dim=64,
                similarity_threshold_init=similarity_threshold_init,
                enhancement_strength_init=enhancement_strength_init,
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
        
        # 实体-关系联合增强模块（方案3：实体增强）
        use_entity_enhancement = getattr(flags, 'use_entity_enhancement', True)  # 默认启用
        self.use_entity_enhancement = use_entity_enhancement
        if self.use_entity_enhancement:
            self.entity_enhancer = EntityRelationJointEnhancer(embedding_dim=64)
            # 使用增强版EntityNBFNet包装原始entity_model
            self.entity_model = EnhancedEntityNBFNet(self.entity_model, self.entity_enhancer)
        else:
            self.entity_enhancer = None
        
        # 存储表示
        self.relation_representations_structural = None
        self.relation_representations_semantic = None
        self.final_relation_representations = None
        self.enhanced_relation_representations = None
        
    def forward(self, data, batch, is_tail=False):
        """增强版前向传播 - 根据配置使用自适应门控机制"""
        logger = logging.getLogger(__name__)
        logger.debug(f"[EnhancedUltra] Forward开始，batch_size={len(batch)}")
        
        query_rels = batch[:, 0, 2]
        query_rels_traverse = batch[:, 0, :]
        query_entities = batch[:, 0, 0]  # 查询实体（head）
        
        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.debug(f"[EnhancedUltra] GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
        
        # 获取基础关系表示（使用原始逻辑）
        from ultra import parse
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        
        if flags.run == "semma" or flags.run == "EnhancedUltra":
            with timer("Relation Model (structural)", logger):
                self.relation_representations_structural = self.relation_model(data, query=query_rels)
            with timer("Semantic Model", logger):
                self.relation_representations_semantic = self.semantic_model(data, query=query_rels)
            with timer("Combiner", logger):
                self.final_relation_representations = self.combiner(
                    self.relation_representations_structural, 
                    self.relation_representations_semantic
                )
        else:
            with timer("Relation Model", logger):
                self.relation_representations_structural = self.relation_model(data, query=query_rels)
            self.final_relation_representations = self.relation_representations_structural
        
        # 应用增强模块（并行融合方式）
        # 原始表示 r (SEMMA融合后的嵌入)
        r = self.final_relation_representations  # [batch_size, num_relations, embedding_dim]
        batch_size = len(query_rels)
        
        logger.debug(f"[EnhancedUltra] 开始并行增强，r形状={r.shape}")
        
        # 并行获取两个增强器的增量（都基于原始表示r）
        # r1: similarity_enhancer的增量
        if self.use_similarity_enhancer and self.similarity_enhancer is not None:
            logger.debug(f"[EnhancedUltra] 应用similarity_enhancer")
            with timer("Similarity Enhancer", logger):
                r1_delta = self.similarity_enhancer(
                    r, 
                    query_rels,
                    return_enhancement_only=True  # 只返回增强增量
                )  # [batch_size, num_relations, embedding_dim]
            logger.debug(f"[EnhancedUltra] similarity_enhancer完成，r1_delta形状={r1_delta.shape}")
        else:
            r1_delta = torch.zeros_like(r)
        
        # r2: prompt_enhancer的增量（只增强查询关系）
        # 同时收集提示图中的实体，用于实体增强（避免重复计算）
        all_prompt_entities = None  # 收集所有batch的提示图实体
        if self.use_prompt_enhancer and self.prompt_enhancer is not None:
            logger.debug(f"[EnhancedUltra] 应用prompt_enhancer，batch_size={batch_size}")
            r2_delta = torch.zeros_like(r)
            all_prompt_entities = set()  # 收集所有提示图中的实体
            
            with timer(f"Prompt Enhancer (batch_size={batch_size})", logger):
                for i in range(batch_size):
                    query_rel = query_rels[i]
                    query_entity = query_entities[i]
                    base_repr = r[i, query_rel, :]  # 使用原始表示r，而不是enhanced_relation_representations
                    
                    # 获取提示图增强增量（传入关系嵌入、entity_model和relation_representations以使用方案2）
                    try:
                        with timer(f"Prompt Enhancer batch {i}", logger, min_time_ms=50):
                            # 先生成提示图以获取实体列表
                            prompt_graph, prompt_entities_list = self.prompt_enhancer.generate_prompt_graph(
                                data, query_rel, query_entity
                            )
                            
                            # 收集提示图中的实体（用于实体增强）
                            if prompt_entities_list is not None and len(prompt_entities_list) > 0:
                                all_prompt_entities.update(prompt_entities_list)
                            
                            # 编码提示图上下文并获取增强增量
                            # 传递entity_model和relation_representations以启用更好的节点初始化
                            if prompt_graph is not None:
                                # 确保relation_representations是有效的Tensor
                                # r[i]是[num_relations, embedding_dim]，这是正确的格式
                                relation_repr_for_nbfnet = r[i] if r[i] is not None and isinstance(r[i], torch.Tensor) else None
                                
                                prompt_context = self.prompt_enhancer.encode_prompt_context(
                                    prompt_graph, query_rel, r[i], prompt_entities_list, data,
                                    entity_model=self.entity_model if hasattr(self, 'entity_model') else None,
                                    relation_representations=relation_repr_for_nbfnet  # 传递当前batch的关系表示
                                )
                                
                                # 计算增强增量
                                weight_input = torch.cat([base_repr, prompt_context], dim=-1)
                                adaptive_weight = self.prompt_enhancer.adaptive_weights(weight_input)
                                fusion_input = torch.cat([base_repr, prompt_context], dim=-1)
                                enhanced_embedding = self.prompt_enhancer.context_fusion(fusion_input)
                                prompt_delta = enhanced_embedding  # 返回增强增量
                            else:
                                prompt_delta = torch.zeros_like(base_repr)
                        
                        r2_delta[i, query_rel, :] = prompt_delta
                    except Exception as e:
                        logger.warning(f"[EnhancedUltra] prompt_enhancer在batch {i}失败: {e}")
                        # 失败时使用零增量，避免CUDA错误传播
                        try:
                            r2_delta[i, query_rel, :] = torch.zeros_like(base_repr)
                        except RuntimeError as cuda_err:
                            # 如果CUDA错误，尝试在CPU上创建然后移到GPU
                            logger.error(f"[EnhancedUltra] CUDA错误，尝试在CPU上创建零向量: {cuda_err}")
                            cpu_zero = torch.zeros_like(base_repr.cpu())
                            r2_delta[i, query_rel, :] = cpu_zero.to(base_repr.device)
            
            logger.debug(f"[EnhancedUltra] prompt_enhancer完成，r2_delta形状={r2_delta.shape}，收集到{len(all_prompt_entities)}个提示图实体")
            
            # 内存优化：Prompt Enhancer处理完后，如果内存紧张，清理部分缓存
            if torch.cuda.is_available():
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                memory_free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - memory_reserved
                if memory_free < 2.0 and hasattr(self, 'prompt_enhancer') and self.prompt_enhancer is not None:
                    # 保留最新的5个缓存项，清理其他
                    cache_size = len(self.prompt_enhancer._bfnet_cache)
                    if cache_size > 5:
                        # 只保留最后5个（转换为list后取最后5个）
                        cache_items = list(self.prompt_enhancer._bfnet_cache.items())
                        self.prompt_enhancer._bfnet_cache.clear()
                        self.prompt_enhancer._bfnet_cache.update(cache_items[-5:])
                        torch.cuda.empty_cache()
                        logger.debug(f"[EnhancedUltra] Prompt Enhancer处理完成，清理缓存 (保留最新5项，清理前: {cache_size}项)")
        else:
            r2_delta = torch.zeros_like(r)
        
        # 并行融合：使用可学习权重（方案3 - 增量融合方式）
        if self.use_learnable_fusion and self.fusion_weights_logits is not None:
            # 使用softmax归一化可学习权重（只归一化两个增强器的权重）
            enhancement_weights = F.softmax(self.fusion_weights_logits, dim=0)  # [2]
            # enhancement_weights[0]: similarity_enhancer的权重
            # enhancement_weights[1]: prompt_enhancer的权重
            
            # 增量融合：r + w[0]*r1_delta + w[1]*r2_delta
            # 原始r直接保留，只加权增强增量
            if self.use_adaptive_gate and self.enhancement_gate is not None and self.use_similarity_enhancer:
                # 使用自适应门控机制：计算门控权重
                gate_weights = self.enhancement_gate(
                    r,
                    query_rels,
                    query_entities,
                    data
                )  # [batch_size]
                gate_weights_expanded = gate_weights.view(batch_size, 1, 1)
                
                # 增量融合 + 自适应门控：r + gate*w[0]*r1_delta + w[1]*r2_delta
                self.enhanced_relation_representations = (
                    r + 
                    gate_weights_expanded * enhancement_weights[0] * r1_delta + 
                    enhancement_weights[1] * r2_delta
                )
            else:
                # 增量融合：r + w[0]*r1_delta + w[1]*r2_delta
                self.enhanced_relation_representations = (
                    r + 
                    enhancement_weights[0] * r1_delta + 
                    enhancement_weights[1] * r2_delta
                )
        else:
            # 固定权重融合（回退到方案1）：r + u*r1 + θ*r2
            # 其中 u = similarity_enhancer_weight, θ = prompt_enhancer_weight
            if self.use_adaptive_gate and self.enhancement_gate is not None and self.use_similarity_enhancer:
                # 使用自适应门控机制：计算门控权重
                gate_weights = self.enhancement_gate(
                    r,
                    query_rels,
                    query_entities,
                    data
                )  # [batch_size]
                gate_weights_expanded = gate_weights.view(batch_size, 1, 1)
                
                # 并行融合：r + gate_weight * u * r1 + θ * r2
                self.enhanced_relation_representations = (
                    r + 
                    gate_weights_expanded * self.similarity_enhancer_weight * r1_delta + 
                    self.prompt_enhancer_weight * r2_delta
                )
            else:
                # 不使用门控机制：直接并行融合 r + u*r1 + θ*r2
                self.enhanced_relation_representations = (
                    r + 
                    self.similarity_enhancer_weight * r1_delta + 
                    self.prompt_enhancer_weight * r2_delta
                )
        
        # 使用最终的关系表示进行实体推理
        logger.debug(f"[EnhancedUltra] 开始实体推理，enhanced_relation_representations形状={self.enhanced_relation_representations.shape}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - memory_reserved
            logger.debug(f"[EnhancedUltra] 实体推理前GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB, 空闲={memory_free:.2f}GB")
            
            # 内存优化：如果内存紧张，清理Prompt Enhancer的缓存
            if memory_free < 1.0:  # 如果空闲内存小于1GB
                if hasattr(self, 'prompt_enhancer') and self.prompt_enhancer is not None:
                    cache_size_before = len(self.prompt_enhancer._bfnet_cache)
                    self.prompt_enhancer._bfnet_cache.clear()  # 清空缓存
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    logger.warning(f"[EnhancedUltra] 内存紧张，清理Prompt Enhancer缓存 (清理前: {cache_size_before}项)")
        
        try:
            # 传递提示图中的实体给实体增强器（如果已收集）
            prompt_entities_for_entity = all_prompt_entities if all_prompt_entities is not None and len(all_prompt_entities) > 0 else None
            if prompt_entities_for_entity is not None:
                logger.debug(f"[EnhancedUltra] 使用提示图中的{len(prompt_entities_for_entity)}个实体进行实体增强")
            
            # 如果entity_model是EnhancedEntityNBFNet，传递prompt_entities
            if hasattr(self.entity_model, 'forward') and prompt_entities_for_entity is not None:
                score = self.entity_model(data, self.enhanced_relation_representations, batch, 
                                         prompt_entities=prompt_entities_for_entity)
            else:
                score = self.entity_model(data, self.enhanced_relation_representations, batch)
            logger.debug(f"[EnhancedUltra] 实体推理完成，score形状={score.shape}")
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
    
        return score
    