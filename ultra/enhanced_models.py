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
        
        # BFS遍历k跳
        with timer(f"[Entity Enhancer] BFS遍历{max_hops}跳 (种子数={len(seed_entities)})", logger):
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
    
    def compute_enhanced_boundary(self, data, h_index, r_index, relation_representations):
        """
        计算增强的boundary条件（为所有实体提供初始特征）
        优化：只计算查询相关实体及其1跳邻居，减少不必要的计算
        
        Args:
            data: 图数据
            h_index: 源实体索引 [batch_size]
            r_index: 关系索引 [batch_size]
            relation_representations: 关系表示 [batch_size, num_relations, embedding_dim] 或 [num_relations, embedding_dim]
        
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
        enhanced_boundary = torch.zeros(batch_size, data.num_nodes, self.embedding_dim, device=device)
        
        logger = logging.getLogger(__name__)
        
        # 优化：只计算查询相关实体及其1跳邻居
        # 1. 收集所有查询实体（源实体）
        with timer("收集查询实体", logger):
            query_entities = set()
            if isinstance(h_index, torch.Tensor):
                if h_index.dim() == 0:
                    query_entities.add(h_index.item())
                else:
                    query_entities.update(h_index.tolist())
            else:
                query_entities.add(h_index)
        
        # 2. 获取查询实体的1跳邻居
        with timer(f"计算1跳邻居 (种子数={len(query_entities)})", logger):
            relevant_entities = self._get_k_hop_neighbors(data, query_entities, max_hops=1)
        
        # 3. 移除实体数量限制，计算所有相关实体以最大化模型效果
        # 由于实体特征计算很快（~500ms for 500 entities），可以计算所有相关实体
        # 不再限制实体数量，让所有1跳邻居都参与计算
        
        logger.debug(f"[Entity Enhancer] 查询实体数={len(query_entities)}, "
                    f"1跳邻居实体数={len(relevant_entities)}, batch_size={batch_size}")
        
        # 4. 只为相关实体计算特征
        with timer(f"计算实体特征 (实体数={len(relevant_entities)})", logger):
            for idx, entity_id in enumerate(relevant_entities):
                if entity_id < data.num_nodes:
                    entity_feat = self.compute_entity_relation_features(
                        entity_id, data, relation_embeddings
                    )  # [embedding_dim]
                    # 为所有batch设置相同的实体特征
                    enhanced_boundary[:, entity_id, :] = entity_feat.unsqueeze(0).expand(batch_size, -1)
                    
                    # 每100个实体输出一次进度
                    if (idx + 1) % 100 == 0:
                        logger.debug(f"[Entity Enhancer] 已计算 {idx + 1}/{len(relevant_entities)} 个实体特征")
        
        # 5. 确保源实体有特征（使用关系嵌入，叠加到已有特征上）
        # 处理r_index可能是标量或1D张量的情况
        if isinstance(r_index, torch.Tensor):
            if r_index.dim() == 0:  # 标量
                query = relation_embeddings[r_index.item()].unsqueeze(0)  # [1, embedding_dim]
                h_index_expanded = h_index.unsqueeze(0) if h_index.dim() == 0 else h_index
                for i in range(min(len(query), len(h_index_expanded))):
                    enhanced_boundary[i, h_index_expanded[i], :] += query[i]
            else:
                query = relation_embeddings[r_index]  # [batch_size, embedding_dim]
                for i in range(batch_size):
                    enhanced_boundary[i, h_index[i], :] += query[i]
        else:
            # r_index是Python标量
            query = relation_embeddings[r_index].unsqueeze(0)  # [1, embedding_dim]
            h_idx = h_index.item() if isinstance(h_index, torch.Tensor) and h_index.dim() == 0 else h_index[0]
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
    
    def forward(self, data, relation_representations, batch):
        """
        使用增强的boundary条件进行实体推理
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
            data, h_index_flat, r_index_flat, relation_representations
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
    
    def __init__(self, embedding_dim=64, similarity_threshold_init=0.8, enhancement_strength_init=0.05):
        super(SimilarityBasedRelationEnhancer, self).__init__()
        
        self.embedding_dim = embedding_dim
        
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
            
            # 使用平滑的权重而不是硬阈值
            valid_similarities_raw = similarities[valid_indices]  # [num_valid]
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
        
        # BFS遍历k跳
        with timer(f"[Prompt Enhancer] BFS遍历{max_hops}跳 (种子数={len(seed_entities)})", logger):
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
        prompt_entities = self._get_k_hop_neighbors(data, seed_entities, max_hops=1)
        
        # 4. 如果有查询关系的边，也添加这些边的实体（但不作为种子再找邻居）
        if query_edges.shape[1] > 0:
            # 采样少量示例
            if query_edges.shape[1] > num_samples:
                if self.training:
                    # 训练时：随机采样以增加多样性
                    indices = torch.randperm(query_edges.shape[1], device=device)[:num_samples]
                else:
                    # 推理时：确定性采样（选择前N个）以保证可重复性
                    indices = torch.arange(num_samples, device=device)
                sampled_edges = query_edges[:, indices]
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
        # 注意：由于维度匹配问题，暂时禁用方案2，直接使用方案1
        use_entity_nbfnet = False  # 暂时禁用，避免维度不匹配问题
        if use_entity_nbfnet and entity_model is not None and relation_representations is not None and data is not None and prompt_entities is not None and len(prompt_entities) == prompt_graph.num_nodes:
            try:
                # 使用EntityNBFNet的bellmanford计算实体特征
                # 为所有实体批量计算（使用查询关系作为虚拟关系）
                query_rel_idx = query_relation.item() if isinstance(query_relation, torch.Tensor) else query_relation
                
                # 准备批量输入：为每个实体创建一个查询
                num_entities = len(prompt_entities)
                h_indices = torch.tensor(prompt_entities, device=device, dtype=torch.long)  # [num_entities]
                # 使用查询关系作为虚拟关系（或者使用0作为默认关系）
                r_indices = torch.full((num_entities,), query_rel_idx, device=device, dtype=torch.long)  # [num_entities]
                
                # 获取实际的EntityNBFNet实例（处理EnhancedEntityNBFNet包装器）
                actual_entity_model = entity_model
                if hasattr(entity_model, 'entity_model'):
                    # 如果是EnhancedEntityNBFNet包装器，获取内部的entity_model
                    actual_entity_model = entity_model.entity_model
                
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
                if relation_representations.dim() == 2:
                    # 扩展为[batch_size, num_relations, embedding_dim]
                    # batch_size = num_entities（每个实体一个查询）
                    actual_entity_model.query = relation_representations.unsqueeze(0).expand(num_entities, -1, -1)
                elif relation_representations.dim() == 3:
                    # 如果已经是3D，检查batch_size是否匹配
                    if relation_representations.shape[0] == 1:
                        actual_entity_model.query = relation_representations.expand(num_entities, -1, -1)
                    else:
                        actual_entity_model.query = relation_representations
                else:
                    # 如果格式不对，回退到方案1
                    raise ValueError(f"relation_representations维度不正确: {relation_representations.shape}")
                
                # 计算实体特征（批量计算）
                # 注意：这里使用torch.no_grad()以避免影响主模型的梯度
                # 但如果是在训练时，可能需要保留梯度（取决于需求）
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[Prompt Enhancer] 使用EntityNBFNet计算{num_entities}个实体的特征")
                
                actual_entity_model.eval()  # 临时设置为eval模式，避免dropout等
                was_training = actual_entity_model.training
                try:
                    with torch.no_grad():  # 推理时不需要梯度，避免影响主模型
                        entity_features_dict = actual_entity_model.bellmanford(data, h_indices, r_indices)
                        entity_features = entity_features_dict["node_feature"]  # [num_entities, num_nodes, feature_dim]
                        logger.debug(f"[Prompt Enhancer] EntityNBFNet计算成功，特征形状: {entity_features.shape}")
                except Exception as e:
                    logger.warning(f"[Prompt Enhancer] EntityNBFNet计算失败: {e}")
                    raise
                finally:
                    # 恢复原始状态
                    if original_query is not None:
                        actual_entity_model.query = original_query
                    actual_entity_model.train(was_training)
                
                # 提取每个实体对应的节点特征
                # entity_features形状: [num_entities, num_nodes, feature_dim]
                # 对于实体i，它在图中的节点ID是prompt_entities[i]，特征在entity_features[i, prompt_entities[i], :]
                node_embeddings = torch.zeros(prompt_graph.num_nodes, self.embedding_dim, device=device)
                for i, entity_id in enumerate(prompt_entities):
                    if i < entity_features.shape[0] and entity_id < entity_features.shape[1]:
                        # 提取实体对应的节点特征
                        # entity_features[i, entity_id, :] 是实体i（节点ID=entity_id）的特征
                        entity_feat = entity_features[i, entity_id, :]  # [feature_dim]
                        
                        # 处理特征维度不匹配的情况
                        if entity_feat.shape[0] > self.embedding_dim:
                            # 如果特征维度大于embedding_dim，使用前embedding_dim维
                            entity_feat = entity_feat[:self.embedding_dim]
                        elif entity_feat.shape[0] < self.embedding_dim:
                            # 如果特征维度小于embedding_dim，需要填充
                            padding = torch.zeros(self.embedding_dim - entity_feat.shape[0], device=device)
                            entity_feat = torch.cat([entity_feat, padding])
                        
                        node_embeddings[i] = entity_feat
                    else:
                        # 回退：使用关系嵌入的平均值
                        if relation_embeddings is not None and relation_embeddings.shape[0] > 0:
                            node_embeddings[i] = relation_embeddings.mean(dim=0)
                
            except Exception as e:
                # 如果EntityNBFNet计算失败，回退到方案1（使用关系平均嵌入）
                logger = logging.getLogger(__name__)
                logger.warning(f"[Prompt Enhancer] EntityNBFNet计算失败，回退到关系平均嵌入方案: {e}")
                import warnings
                # 只在DEBUG模式下显示警告，避免日志过多
                if logger.level <= logging.DEBUG:
                    warnings.warn(f"EntityNBFNet计算失败，回退到关系平均嵌入方案: {e}")
                node_embeddings = self._fallback_entity_embedding(
                    prompt_graph, query_relation, relation_embeddings, prompt_entities, data, device
                )
        
        # 方案1：使用实体相关的所有关系的平均嵌入（主要方案，避免维度不匹配问题）
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
        self.use_adaptive_gate = getattr(flags, 'use_adaptive_gate', False)
        
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
            self.prompt_enhancer = OptimizedPromptGraph(
                embedding_dim=64,
                max_hops=1,  # 使用1跳邻居，提升计算速度
                num_prompt_samples=3  # 减少到3，提升计算速度（从5降低）
            )
        else:
            self.prompt_enhancer = None
        
        # 基于相似度的关系增强模块（新增，可通过配置禁用）
        if self.use_similarity_enhancer:
            # 从flags.yaml读取初始值，如果没有则使用默认值
            similarity_threshold_init = getattr(flags, 'similarity_threshold_init', 0.8)
            enhancement_strength_init = getattr(flags, 'enhancement_strength_init', 0.05)
            
            self.similarity_enhancer = SimilarityBasedRelationEnhancer(
                embedding_dim=64,
                similarity_threshold_init=similarity_threshold_init,
                enhancement_strength_init=enhancement_strength_init
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
        if self.use_prompt_enhancer and self.prompt_enhancer is not None:
            logger.debug(f"[EnhancedUltra] 应用prompt_enhancer，batch_size={batch_size}")
            r2_delta = torch.zeros_like(r)
            with timer(f"Prompt Enhancer (batch_size={batch_size})", logger):
                for i in range(batch_size):
                    query_rel = query_rels[i]
                    query_entity = query_entities[i]
                    base_repr = r[i, query_rel, :]  # 使用原始表示r，而不是enhanced_relation_representations
                    
                    # 获取提示图增强增量（传入关系嵌入、entity_model和relation_representations以使用方案2）
                    try:
                        with timer(f"Prompt Enhancer batch {i}", logger, min_time_ms=50):
                            prompt_delta = self.prompt_enhancer(
                                data, query_rel, query_entity, base_repr,
                                return_enhancement_only=True,  # 只返回增强增量
                                relation_embeddings=r[i],  # 传入当前batch的关系嵌入 [num_relations, embedding_dim]
                                entity_model=None,  # 暂时禁用EntityNBFNet方案，避免维度不匹配问题
                                relation_representations=None  # 暂时禁用EntityNBFNet方案
                            )  # [embedding_dim]
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
            logger.debug(f"[EnhancedUltra] prompt_enhancer完成，r2_delta形状={r2_delta.shape}")
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
            logger.debug(f"[EnhancedUltra] 实体推理前GPU内存: {memory_allocated:.2f}GB")
        
        try:
            score = self.entity_model(data, self.enhanced_relation_representations, batch)
            logger.debug(f"[EnhancedUltra] 实体推理完成，score形状={score.shape}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"[EnhancedUltra] GPU内存不足: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"[EnhancedUltra] 已清理GPU缓存")
                raise
            else:
                raise
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.debug(f"[EnhancedUltra] Forward完成，GPU内存: {memory_allocated:.2f}GB")
        
        return score
    
    def _apply_enhancement(self, data, batch, query_rels_traverse):
        """应用提示图增强"""
        batch_size = len(query_rels_traverse)
        enhanced_reprs = []
        
        for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
            # 获取基础表示
            base_repr = self.final_relation_representations[i]
            
            # 应用提示图增强
            enhanced_repr = self.prompt_enhancer(
                data, query_rel, head, base_repr
            )
            
            enhanced_reprs.append(enhanced_repr)
        
        return torch.stack(enhanced_reprs, dim=0)
    
    def _apply_simple_enhancement(self, data, batch, query_rels_traverse):
        """保守调优增强策略 - 轻微增强，保持原始性能"""
        batch_size = len(query_rels_traverse)
        enhanced_reprs = []

        for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
            base_repr = self.final_relation_representations[i]
            
            # 策略1: 轻微全局增强 - 只使用很小的权重
            if batch_size > 1:
                # 计算全局平均
                global_context = self.final_relation_representations.mean(dim=0)
                # 非常轻微的增强
                enhanced_repr = base_repr + 0.05 * global_context
            else:
                enhanced_repr = base_repr
            
            # 策略2: 轻微噪声增强 - 增加鲁棒性
            # 在训练时使用随机噪声，在推理时禁用以保证可重复性
            if self.training:
                noise = torch.randn_like(base_repr) * 0.01
                enhanced_repr += noise
            # 推理时不添加噪声，保证结果可重复
            
            # 策略3: 残差保护 - 确保不偏离太远
            enhanced_repr = 0.95 * enhanced_repr + 0.05 * base_repr
            
            enhanced_reprs.append(enhanced_repr)

        return torch.stack(enhanced_reprs, dim=0)
    
    def _enhanced_entity_model_forward(self, data, relation_representations, batch):
        """突破性增强：直接增强EntityNBFNet的query嵌入"""
        h_index, t_index, r_index = batch.unbind(-1)

        # 导入flags
        from ultra import parse
        import os
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        
        if flags.harder_setting == True:
            r_index = torch.ones_like(r_index) * (data.num_relations // 2 - 1)

        # 突破性增强：对query嵌入进行深度增强
        enhanced_query = self._deep_enhance_query(relation_representations, batch)
        
        # 设置增强后的query
        self.entity_model.query = enhanced_query

        # 初始化每层的relation
        for layer in self.entity_model.layers:
            layer.relation = enhanced_query

        if self.entity_model.training:
            data = self.entity_model.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        h_index, t_index, r_index = self.entity_model.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # 使用增强后的query进行Bellman-Ford
        output = self.entity_model.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"]

        # 计算最终得分
        score = self.entity_model.mlp(feature).squeeze(-1)
        return score.view(shape)
    
    def _deep_enhance_query(self, relation_representations, batch):
        """深度增强query嵌入"""
        batch_size = relation_representations.shape[0]
        enhanced_queries = []
        
        for i in range(batch_size):
            base_query = relation_representations[i]
            
            # 策略1: 自注意力增强
            attention_weights = torch.softmax(
                torch.sum(base_query * relation_representations, dim=1), dim=0
            )
            attended_query = torch.sum(
                attention_weights.unsqueeze(1) * relation_representations, dim=0
            )
            
            # 策略2: 残差连接
            enhanced_query = base_query + 0.2 * attended_query
            
            # 策略3: 层归一化
            enhanced_query = F.layer_norm(enhanced_query, enhanced_query.shape)
            
            # 策略4: 门控机制
            gate = torch.sigmoid(torch.sum(base_query * enhanced_query))
            enhanced_query = gate * enhanced_query + (1 - gate) * base_query
            
            enhanced_queries.append(enhanced_query)
        
        return torch.stack(enhanced_queries, dim=0)
    
    def _revolutionary_enhancement(self, data, batch, query_rels_traverse):
        """基于相似度的关系增强（已弃用，现在使用similarity_enhancer）"""
        # 这个方法保留是为了向后兼容，实际已使用similarity_enhancer
        return self.final_relation_representations
    
    def _get_relation_frequency(self, data, rel_id):
        """获取关系频率"""
        # 确保rel_id在有效范围内
        if rel_id >= data.num_relations or rel_id < 0:
            return 1
        edge_mask = (data.edge_type == rel_id)
        return edge_mask.sum().item()
    
    def _get_entity_degree(self, data, entity_id):
        """获取实体度"""
        # 确保entity_id在有效范围内
        if entity_id >= data.num_nodes or entity_id < 0:
            return 1
        edge_mask = (data.edge_index[0] == entity_id) | (data.edge_index[1] == entity_id)
        return edge_mask.sum().item()
    
    def _get_semantic_boost(self, data, rel_id):
        """获取语义增强因子"""
        # 基于关系类型的语义增强 - 增强力度
        if rel_id < data.num_relations // 2:
            # 正向关系
            return 1.12
        else:
            # 反向关系
            return 1.08
    
    
    def _post_process_enhancement(self, score, data, batch):
        """后处理增强（已弃用，现在通过相似度增强模块完成）"""
        # 这个方法保留是为了向后兼容，实际增强已在前面的similarity_enhancer中完成
        return score
    