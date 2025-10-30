"""
KG-ICL风格的Prompt机制 - 简化版
简化版本，减少内存占用和计算开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from collections import defaultdict


class SimplePromptEncoder(nn.Module):
    """简化的Prompt编码器 - 单层消息传递"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(SimplePromptEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 简化的消息网络
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 简化的更新网络
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, edge_index, edge_type, num_relations, query_relations, batch_size):
        """
        简化的消息传递
        Args:
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            num_relations: int
            query_relations: [batch_size]
            batch_size: int
        Returns:
            relation_embeddings: [batch_size, num_relations, hidden_dim]
        """
        device = edge_index.device
        total_relations = num_relations * batch_size
        
        # 初始化关系嵌入
        relation_embeddings = torch.randn(
            total_relations, self.hidden_dim, 
            device=device
        ) * 0.1
        
        # 为查询关系设置特殊标记
        for i in range(batch_size):
            rel_idx = query_relations[i] + i * num_relations
            relation_embeddings[rel_idx] = torch.ones(self.hidden_dim, device=device)
        
        # 单层消息传递（简化）
        head_embeds = relation_embeddings[edge_type]
        tail_embeds = relation_embeddings[edge_type]
        
        msg = torch.cat([head_embeds, tail_embeds], dim=-1)
        msg = self.msg_net(msg)
        
        # 聚合消息
        new_embeddings = scatter_add(
            msg, edge_type, dim=0, dim_size=total_relations
        )
        relation_embeddings = self.update_net(new_embeddings)
        
        # 重塑为 [batch_size, num_relations, hidden_dim]
        relation_embeddings = relation_embeddings.view(batch_size, num_relations, self.hidden_dim)
        
        return relation_embeddings


class SimplePromptConstructor:
    """简化的Prompt图构造器"""
    def __init__(self, num_examples=2, max_hops=1):
        self.num_examples = num_examples
        self.max_hops = max_hops
        
    def construct_prompt_graph(self, data, query_relation, query_head):
        """
        简化的prompt图构造
        Args:
            data: 图数据对象
            query_relation: 查询关系ID
            query_head: 查询头实体ID
        Returns:
            prompt_data: 简化的prompt图数据
        """
        edge_index = data.edge_index
        edge_type = data.edge_type
        
        # 找到query_relation的边
        relation_mask = (edge_type == query_relation)
        relation_edges = edge_index[:, relation_mask]
        
        if relation_edges.size(1) == 0:
            return None
        
        # 采样示例（简化版）
        num_available = min(self.num_examples, relation_edges.size(1))
        if num_available < relation_edges.size(1):
            indices = torch.randperm(relation_edges.size(1))[:num_available]
            sampled_edges = relation_edges[:, indices]
        else:
            sampled_edges = relation_edges
        
        # 收集实体
        entities = set()
        for i in range(sampled_edges.size(1)):
            entities.add(sampled_edges[0, i].item())
            entities.add(sampled_edges[1, i].item())
        
        # 添加查询头实体的1跳邻居（简化）
        head_mask = (edge_index[0] == query_head)
        if head_mask.any():
            neighbors = edge_index[1, head_mask][:5]  # 限制最多5个邻居
            entities.update(neighbors.tolist())
        
        # 构建子图边
        entity_list = list(entities)
        entity_map = {old: new for new, old in enumerate(entity_list)}
        
        new_edges = []
        new_types = []
        
        for i in range(edge_index.size(1)):
            h, t = edge_index[0, i].item(), edge_index[1, i].item()
            if h in entity_map and t in entity_map:
                new_edges.append([entity_map[h], entity_map[t]])
                new_types.append(edge_type[i].item())
                if len(new_edges) >= 20:  # 限制边数
                    break
        
        if len(new_edges) == 0:
            return None
        
        return {
            'edge_index': torch.tensor(new_edges, dtype=torch.long).t(),
            'edge_type': torch.tensor(new_types, dtype=torch.long),
            'num_nodes': len(entity_list)
        }


class KGICLPromptEnhancer(nn.Module):
    """
    简化版的KG-ICL Prompt增强模块
    减少参数量和计算开销
    """
    def __init__(self, hidden_dim=64, num_prompt_layers=1, 
                 num_examples=2, max_hops=1):
        super(KGICLPromptEnhancer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_examples = num_examples
        
        # 简化的核心组件
        self.prompt_encoder = SimplePromptEncoder(hidden_dim)
        self.constructor = SimplePromptConstructor(num_examples, max_hops)
        
        # 简化的融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 门控权重（简化）
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def enhance_relation_representations(self, data, query_relations, 
                                        query_heads, base_relation_reprs):
        """
        简化的关系表示增强
        Args:
            data: 图数据
            query_relations: [batch_size]
            query_heads: [batch_size]
            base_relation_reprs: [batch_size, num_relations, hidden_dim]
        Returns:
            enhanced_reprs: [batch_size, num_relations, hidden_dim]
        """
        batch_size = query_relations.size(0)
        device = query_relations.device
        
        enhanced_reprs = []
        
        for i in range(batch_size):
            query_rel = query_relations[i]
            query_head = query_heads[i]
            base_repr = base_relation_reprs[i]
            
            # 构建简化的prompt图
            prompt_data = self.constructor.construct_prompt_graph(
                data, query_rel, query_head
            )
            
            if prompt_data is None:
                enhanced_reprs.append(base_repr)
                continue
            
            # 移动到设备
            edge_index = prompt_data['edge_index'].to(device)
            edge_type = prompt_data['edge_type'].to(device)
            
            # 简化的编码
            num_relations = base_repr.size(0)
            try:
                prompt_repr = self.prompt_encoder(
                    edge_index, edge_type,
                    num_relations, query_rel.unsqueeze(0), 1
                ).squeeze(0)
                
                # 融合
                combined = torch.cat([base_repr, prompt_repr], dim=-1)
                fused = self.fusion(combined)
                
                # 门控融合（简化）
                gate_weight = self.gate(combined)
                enhanced = gate_weight * fused + (1 - gate_weight) * base_repr
                
                enhanced_reprs.append(enhanced)
            except Exception as e:
                # 如果失败，使用原始表示
                enhanced_reprs.append(base_repr)
        
        enhanced_reprs = torch.stack(enhanced_reprs, dim=0)
        return enhanced_reprs
    
    def forward(self, data, query_relations, query_heads, base_relation_reprs):
        """前向传播"""
        return self.enhance_relation_representations(
            data, query_relations, query_heads, base_relation_reprs
        )
