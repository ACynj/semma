import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
import numpy as np
from collections import defaultdict
import networkx as nx

class AdaptivePromptGraph(nn.Module):
    """
    自适应提示图增强模块
    基于KG-ICL论文的提示图机制，为Ultra模型提供动态上下文增强
    """
    
    def __init__(self, embedding_dim=64, max_hops=3, num_prompt_samples=5):
        super(AdaptivePromptGraph, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_hops = max_hops
        self.num_prompt_samples = num_prompt_samples
        
        # 提示图编码器
        self.prompt_encoder = PromptGraphEncoder(embedding_dim)
        
        # 自适应权重网络
        self.adaptive_weights = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # 上下文融合网络
        self.context_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 确保所有组件在相同设备上
        self.device = torch.device('cpu')
        
    def generate_prompt_graph(self, data, query_relation, query_entity, num_samples=None):
        """
        为查询关系生成提示图
        
        Args:
            data: 图数据对象
            query_relation: 查询关系ID
            query_entity: 查询实体ID
            num_samples: 提示样本数量
            
        Returns:
            prompt_graph: 提示图数据对象
        """
        if num_samples is None:
            num_samples = self.num_prompt_samples
            
        # 找到包含查询关系的所有三元组
        query_triples = self._find_query_triples(data, query_relation)
        
        if len(query_triples) == 0:
            return None
            
        # 采样示例三元组
        sampled_triples = self._sample_prompt_triples(query_triples, num_samples)
        
        # 构建提示图
        prompt_graph = self._build_prompt_graph(data, sampled_triples, query_entity)
        
        return prompt_graph
    
    def _find_query_triples(self, data, query_relation):
        """找到包含查询关系的所有三元组"""
        edge_index = data.edge_index
        edge_type = data.edge_type
        
        # 找到查询关系对应的边
        query_mask = (edge_type == query_relation)
        query_edges = edge_index[:, query_mask]
        
        triples = []
        for i in range(query_edges.shape[1]):
            head, tail = query_edges[0, i], query_edges[1, i]
            triples.append((head.item(), query_relation.item(), tail.item()))
            
        return triples
    
    def _sample_prompt_triples(self, triples, num_samples):
        """采样提示三元组"""
        if len(triples) <= num_samples:
            return triples
        
        # 随机采样
        indices = np.random.choice(len(triples), num_samples, replace=False)
        return [triples[i] for i in indices]
    
    def _build_prompt_graph(self, data, sampled_triples, query_entity):
        """构建提示图"""
        # 收集提示图中的实体
        prompt_entities = set()
        for head, rel, tail in sampled_triples:
            prompt_entities.add(head)
            prompt_entities.add(tail)
            
        # 添加查询实体的邻域
        query_neighbors = self._get_entity_neighbors(data, query_entity, self.max_hops)
        prompt_entities.update(query_neighbors)
        
        # 构建子图
        prompt_graph = self._extract_subgraph(data, list(prompt_entities))
        
        return prompt_graph
    
    def _get_entity_neighbors(self, data, entity, max_hops):
        """获取实体的多跳邻域"""
        neighbors = set([entity])
        current_level = set([entity])
        
        for hop in range(max_hops):
            next_level = set()
            for node in current_level:
                # 找到与当前节点相连的所有节点
                edge_mask = (data.edge_index[0] == node) | (data.edge_index[1] == node)
                connected_nodes = data.edge_index[:, edge_mask]
                next_level.update(connected_nodes[0].tolist())
                next_level.update(connected_nodes[1].tolist())
            
            next_level -= neighbors  # 移除已访问的节点
            neighbors.update(next_level)
            current_level = next_level
            
            if len(next_level) == 0:
                break
                
        return neighbors
    
    def _extract_subgraph(self, data, entities):
        """提取子图"""
        entity_set = set(entities)
        
        # 找到子图中的边
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        for i in range(data.edge_index.shape[1]):
            head, tail = data.edge_index[0, i], data.edge_index[1, i]
            if head.item() in entity_set and tail.item() in entity_set:
                edge_mask[i] = True
        
        # 构建子图
        subgraph = Data(
            edge_index=data.edge_index[:, edge_mask],
            edge_type=data.edge_type[edge_mask],
            num_nodes=len(entity_set)
        )
        
        return subgraph
    
    def encode_prompt_context(self, prompt_graph, query_relation):
        """编码提示图上下文"""
        if prompt_graph is None:
            device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
            return torch.zeros(self.embedding_dim, device=device)
            
        # 使用提示图编码器
        context_embedding = self.prompt_encoder(prompt_graph, query_relation)
        return context_embedding
    
    def forward(self, data, query_relation, query_entity, base_embeddings):
        """
        前向传播
        
        Args:
            data: 图数据
            query_relation: 查询关系
            query_entity: 查询实体
            base_embeddings: 基础嵌入
            
        Returns:
            enhanced_embeddings: 增强后的嵌入
        """
        # 生成提示图
        prompt_graph = self.generate_prompt_graph(data, query_relation, query_entity)
        
        # 编码提示上下文
        prompt_context = self.encode_prompt_context(prompt_graph, query_relation)
        
        # 计算自适应权重
        query_embedding = base_embeddings[query_relation]
        weight_input = torch.cat([query_embedding, prompt_context], dim=-1)
        adaptive_weight = self.adaptive_weights(weight_input)
        
        # 融合上下文信息
        fusion_input = torch.cat([
            base_embeddings[query_relation],
            prompt_context,
            adaptive_weight * prompt_context
        ], dim=-1)
        
        enhanced_embedding = self.context_fusion(fusion_input)
        
        return enhanced_embedding


class PromptGraphEncoder(nn.Module):
    """提示图编码器"""
    
    def __init__(self, embedding_dim=64, num_layers=2):
        super(PromptGraphEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        
        # 关系感知注意力
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 读出层
        self.readout = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, prompt_graph, query_relation):
        """编码提示图"""
        if prompt_graph is None or prompt_graph.num_nodes == 0:
            device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
            return torch.zeros(self.embedding_dim, device=device)
        
        # 获取设备信息
        device = query_relation.device if hasattr(query_relation, 'device') else torch.device('cpu')
        
        # 初始化节点嵌入
        node_embeddings = torch.randn(prompt_graph.num_nodes, self.embedding_dim, device=device)
        
        # 图卷积编码
        for layer in self.gcn_layers:
            node_embeddings = F.relu(layer(node_embeddings))
        
        # 关系感知注意力
        # 这里简化处理，实际应该根据边类型进行更复杂的注意力计算
        attended_embeddings, _ = self.relation_attention(
            node_embeddings.unsqueeze(0),
            node_embeddings.unsqueeze(0),
            node_embeddings.unsqueeze(0)
        )
        
        # 全局读出
        graph_embedding = torch.mean(attended_embeddings.squeeze(0), dim=0)
        final_embedding = self.readout(graph_embedding)
        
        return final_embedding


class MultiScaleContextFusion(nn.Module):
    """多尺度上下文融合模块"""
    
    def __init__(self, embedding_dim=64):
        super(MultiScaleContextFusion, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 不同尺度的编码器
        self.local_encoder = nn.Linear(embedding_dim, embedding_dim)
        self.global_encoder = nn.Linear(embedding_dim, embedding_dim)
        
        # 尺度融合网络
        self.scale_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, local_context, global_context):
        """融合多尺度上下文"""
        # 编码不同尺度的上下文
        local_encoded = self.local_encoder(local_context)
        global_encoded = self.global_encoder(global_context)
        
        # 尺度融合
        fused_context = self.scale_fusion(
            torch.cat([local_encoded, global_encoded], dim=-1)
        )
        
        # 注意力加权
        contexts = torch.stack([local_encoded, global_encoded, fused_context], dim=0).unsqueeze(0)
        attended_context, _ = self.attention(contexts, contexts, contexts)
        
        return attended_context.squeeze(0).mean(dim=0)
