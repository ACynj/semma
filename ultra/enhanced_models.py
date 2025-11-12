import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os

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
    
    def __init__(self, embedding_dim=64, max_hops=2, num_prompt_samples=3):
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
        
    def generate_prompt_graph(self, data, query_relation, query_entity, num_samples=None):
        """快速生成提示图"""
        if num_samples is None:
            num_samples = self.num_prompt_samples
            
        # 简化的提示图生成 - 只使用1跳邻域
        device = query_entity.device
        
        # 找到包含查询关系的边
        edge_mask = (data.edge_type == query_relation)
        query_edges = data.edge_index[:, edge_mask]
        
        if query_edges.shape[1] == 0:
            return None
            
        # 采样少量示例
        # 在训练时使用随机采样，在推理时使用确定性采样（选择前N个）
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
            
        # 收集提示图中的实体
        prompt_entities = set()
        for i in range(sampled_edges.shape[1]):
            head, tail = sampled_edges[0, i], sampled_edges[1, i]
            prompt_entities.add(head.item())
            prompt_entities.add(tail.item())
            
        # 添加查询实体的1跳邻域
        query_neighbors = set([query_entity.item()])
        edge_mask = (data.edge_index[0] == query_entity) | (data.edge_index[1] == query_entity)
        if edge_mask.any():
            connected_edges = data.edge_index[:, edge_mask]
            query_neighbors.update(connected_edges[0].tolist())
            query_neighbors.update(connected_edges[1].tolist())
            
        prompt_entities.update(query_neighbors)
        
        # 构建简化的子图
        prompt_graph = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            num_nodes=len(prompt_entities)
        )
        
        return prompt_graph
    
    def encode_prompt_context(self, prompt_graph, query_relation):
        """快速编码提示图上下文"""
        if prompt_graph is None:
            device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
            return torch.zeros(self.embedding_dim, device=device)
            
        # 简化的编码 - 使用平均池化
        device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
        
        # 使用固定初始化而不是随机初始化，以保证推理时的可重复性
        # 使用零初始化作为基础，这样在推理时是确定性的
        # 如果需要学习到的嵌入，应该从模型的嵌入层获取，但这里为了简化使用固定值
        if self.training:
            # 训练时：使用随机初始化以增加多样性
            node_embeddings = torch.randn(prompt_graph.num_nodes, self.embedding_dim, device=device)
        else:
            # 推理时：使用固定初始化（零向量）以保证可重复性
            # 注意：这可能会影响性能，但保证了结果的一致性
            # 更好的方案是使用学习到的节点嵌入，但需要修改模型结构
            node_embeddings = torch.zeros(prompt_graph.num_nodes, self.embedding_dim, device=device)
        
        # 使用简化的编码器
        encoded_embeddings = self.prompt_encoder(node_embeddings)
        
        # 全局平均池化
        context_embedding = torch.mean(encoded_embeddings, dim=0)
        
        return context_embedding
    
    def forward(self, data, query_relation, query_entity, base_embeddings):
        """优化的前向传播"""
        # 快速生成提示图
        prompt_graph = self.generate_prompt_graph(data, query_relation, query_entity)
        
        # 快速编码上下文
        prompt_context = self.encode_prompt_context(prompt_graph, query_relation)
        
        # 计算自适应权重
        query_embedding = base_embeddings[query_relation]
        weight_input = torch.cat([query_embedding, prompt_context], dim=-1)
        adaptive_weight = self.adaptive_weights(weight_input)
        
        # 融合上下文信息
        fusion_input = torch.cat([query_embedding, prompt_context], dim=-1)
        enhanced_embedding = self.context_fusion(fusion_input)
        
        # 应用自适应权重
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
        
        # 提示图增强模块（保留原有功能）
        self.prompt_enhancer = OptimizedPromptGraph(
            embedding_dim=64,
            max_hops=2,  # 减少跳数
            num_prompt_samples=3  # 减少样本数
        )
        
        # 基于相似度的关系增强模块（新增）
        # 从flags.yaml读取初始值，如果没有则使用默认值
        similarity_threshold_init = getattr(flags, 'similarity_threshold_init', 0.8)
        enhancement_strength_init = getattr(flags, 'enhancement_strength_init', 0.05)
        
        self.similarity_enhancer = SimilarityBasedRelationEnhancer(
            embedding_dim=64,
            similarity_threshold_init=similarity_threshold_init,
            enhancement_strength_init=enhancement_strength_init
        )
        
        # 自适应增强门控网络（根据配置决定是否启用）
        self.use_adaptive_gate = getattr(flags, 'use_adaptive_gate', True)
        if self.use_adaptive_gate:
            self.enhancement_gate = AdaptiveEnhancementGate(embedding_dim=64)
        else:
            self.enhancement_gate = None
        
        # 存储表示
        self.relation_representations_structural = None
        self.relation_representations_semantic = None
        self.final_relation_representations = None
        self.enhanced_relation_representations = None
        
    def forward(self, data, batch, is_tail=False):
        """增强版前向传播 - 根据配置使用自适应门控机制"""
        query_rels = batch[:, 0, 2]
        query_rels_traverse = batch[:, 0, :]
        query_entities = batch[:, 0, 0]  # 查询实体（head）
        
        # 获取基础关系表示（使用原始逻辑）
        from ultra import parse
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        flags = parse.load_flags(os.path.join(project_root, "flags.yaml"))
        
        if flags.run == "semma" or flags.run == "EnhancedUltra":
            self.relation_representations_structural = self.relation_model(data, query=query_rels)
            self.relation_representations_semantic = self.semantic_model(data, query=query_rels)
            self.final_relation_representations = self.combiner(
                self.relation_representations_structural, 
                self.relation_representations_semantic
            )
        else:
            self.relation_representations_structural = self.relation_model(data, query=query_rels)
            self.final_relation_representations = self.relation_representations_structural
        
        # 根据配置决定是否使用门控机制
        if self.use_adaptive_gate and self.enhancement_gate is not None:
            # 使用自适应门控机制：只获取增强增量，由门控机制控制混合
            # 这样可以避免双重混合（消除 enhancement_strength 和 gate_weight 的冗余）
            enhancement_delta = self.similarity_enhancer(
                self.final_relation_representations, 
                query_rels,
                return_enhancement_only=True  # 只返回增强增量，不进行内部混合
            )  # [batch_size, num_relations, embedding_dim]
            
            # 计算门控权重
            gate_weights = self.enhancement_gate(
                self.final_relation_representations,
                query_rels,
                query_entities,
                data
            )  # [batch_size]
            
            # 使用门控权重混合原始表示和增强增量
            # final = original + gate_weight * enhancement_delta
            # 这样门控权重直接控制增强强度，消除了与 enhancement_strength 的冗余
            batch_size = len(query_rels)
            gate_weights_expanded = gate_weights.view(batch_size, 1, 1)
            
            # 混合：final = original + gate * delta
            self.enhanced_relation_representations = (
                self.final_relation_representations +
                gate_weights_expanded * enhancement_delta
            )
        else:
            # 不使用门控机制：使用内部 enhancement_strength 进行混合（原有逻辑）
            self.enhanced_relation_representations = self.similarity_enhancer(
                self.final_relation_representations, 
                query_rels,
                return_enhancement_only=False  # 使用内部strength进行混合
            )
        
        # 使用最终的关系表示进行实体推理
        score = self.entity_model(data, self.enhanced_relation_representations, batch)
        
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
    