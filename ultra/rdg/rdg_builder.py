"""
Relation Dependency Graph (RDG) Builder

This module implements the core functionality for building Relation Dependency Graphs
based on entity-mediated pathways in knowledge graphs.
"""

import torch
from torch_scatter import scatter_add
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RDGConfig:
    """Configuration for RDG construction"""
    enabled: bool = False
    min_dependency_weight: float = 0.001  # Minimum weight threshold for dependency edges
    precedence_method: str = 'indegree'  # 'indegree' or 'topological'
    normalize_weights: bool = True  # Whether to normalize dependency weights
    include_multi_hop: bool = False  # Whether to include multi-hop dependencies
    max_path_length: int = 2  # Maximum path length for multi-hop dependencies


def extract_relation_dependencies(
    graph,
    config: Optional[RDGConfig] = None
) -> List[Tuple[int, int, float]]:
    """
    Extract relation dependencies from a knowledge graph.
    
    A dependency edge (r_i, r_j) exists if there exists a path:
    (h, r_i, e) -> (e, r_j, t)
    meaning r_i's tail entity is r_j's head entity.
    
    Args:
        graph: Graph object with edge_index and edge_type
        config: RDG configuration (optional)
    
    Returns:
        List of (r_i, r_j, weight) tuples representing dependency edges
        where r_i is the predecessor and r_j is the successor
    """
    if config is None:
        config = RDGConfig()
    
    edge_index = graph.edge_index  # [2, num_edges]
    edge_type = graph.edge_type    # [num_edges]
    num_relations = graph.num_relations
    device = edge_index.device
    
    # Build entity-to-relation mappings
    # entity_to_outgoing: maps entity_id -> list of relations where entity is head
    # entity_to_incoming: maps entity_id -> list of relations where entity is tail
    entity_to_outgoing = {}
    entity_to_incoming = {}
    
    # Also track edge counts for weight calculation
    entity_relation_pairs = {}  # {(entity, relation): count}
    
    for i in range(edge_index.size(1)):
        h = edge_index[0, i].item()
        t = edge_index[1, i].item()
        r = edge_type[i].item()
        
        # Track outgoing relations (entity as head)
        if h not in entity_to_outgoing:
            entity_to_outgoing[h] = []
        entity_to_outgoing[h].append(r)
        
        # Track incoming relations (entity as tail)
        if t not in entity_to_incoming:
            entity_to_incoming[t] = []
        entity_to_incoming[t].append(r)
        
        # Track entity-relation pairs for weight calculation
        key = (h, r, 'out')
        entity_relation_pairs[key] = entity_relation_pairs.get(key, 0) + 1
        key = (t, r, 'in')
        entity_relation_pairs[key] = entity_relation_pairs.get(key, 0) + 1
    
    # Extract dependency paths
    # For each edge (h, r1, t), find edges (t, r2, t2) where t is the head
    # This creates dependency: r1 -> r2 (r1's tail is r2's head)
    dependency_count = {}  # {(r_i, r_j): count}
    
    for i in range(edge_index.size(1)):
        h = edge_index[0, i].item()
        t = edge_index[1, i].item()
        r1 = edge_type[i].item()
        
        # Find relations where t (r1's tail) is the head entity
        # This creates dependency: r1 -> r2
        if t in entity_to_outgoing:
            for r2 in entity_to_outgoing[t]:
                if r1 != r2:  # Avoid self-loops
                    key = (r1, r2)
                    dependency_count[key] = dependency_count.get(key, 0) + 1
        
        # Also consider reverse: find relations where h (r1's head) is the tail entity
        # This creates dependency: r2 -> r1
        if h in entity_to_incoming:
            for r2 in entity_to_incoming[h]:
                if r1 != r2:  # Avoid self-loops
                    key = (r2, r1)
                    dependency_count[key] = dependency_count.get(key, 0) + 1
    
    # Convert counts to weighted edges
    dependency_edges = []
    total_paths = sum(dependency_count.values()) if dependency_count else 1.0
    
    for (r_i, r_j), count in dependency_count.items():
        # Weight is the frequency of this dependency path
        weight = count / total_paths if config.normalize_weights else count
        
        # Filter by minimum weight threshold
        if weight >= config.min_dependency_weight:
            dependency_edges.append((r_i, r_j, weight))
    
    return dependency_edges


def compute_relation_precedence(
    dependency_edges: List[Tuple[int, int, float]],
    num_relations: int,
    config: Optional[RDGConfig] = None
) -> Dict[int, float]:
    """
    Compute relation precedence values τ(r) for each relation.
    
    Relations with lower τ values are more "basic" (precursors),
    while relations with higher τ values are more "composite" (depend on others).
    
    Args:
        dependency_edges: List of (r_i, r_j, weight) dependency edges
        num_relations: Total number of relations
        config: RDG configuration (optional)
    
    Returns:
        Dictionary mapping relation_id -> precedence_value (τ)
    """
    if config is None:
        config = RDGConfig()
    
    if config.precedence_method == 'indegree':
        # Method 1: Based on in-degree (how many relations depend on this one)
        # Higher in-degree means more basic (lower τ value)
        in_degree = {r: 0.0 for r in range(num_relations)}
        
        for r_i, r_j, weight in dependency_edges:
            # r_j is depended upon by r_i, so r_j's in-degree increases
            in_degree[r_j] += weight
        
        # Normalize to [0, 1] range
        # Lower in-degree -> higher τ (more composite)
        # Higher in-degree -> lower τ (more basic)
        max_degree = max(in_degree.values()) if in_degree.values() else 1.0
        if max_degree > 0:
            tau = {r: 1.0 - (in_degree[r] / max_degree) for r in range(num_relations)}
        else:
            tau = {r: 0.5 for r in range(num_relations)}  # Default: all equal
        
    elif config.precedence_method == 'topological':
        # Method 2: Topological ordering (simplified)
        # This would require a DAG structure
        # For now, fall back to indegree
        return compute_relation_precedence(
            dependency_edges, num_relations,
            RDGConfig(precedence_method='indegree')
        )
    else:
        raise ValueError(f"Unknown precedence method: {config.precedence_method}")
    
    return tau


def get_preceding_relations(
    r_v: int,
    dependency_edges: List[Tuple[int, int, float]],
    tau: Dict[int, float]
) -> List[int]:
    """
    Get the set of preceding relations N^past(r_v) for relation r_v.
    
    Preceding relations are those that:
    1. Have a dependency edge to r_v (r_i -> r_v)
    2. Have lower precedence (lower τ value) than r_v
    
    Args:
        r_v: Target relation ID
        dependency_edges: List of dependency edges
        tau: Precedence values for all relations
    
    Returns:
        List of relation IDs that precede r_v
    """
    preceding_rels = []
    
    for r_i, r_j, weight in dependency_edges:
        if r_j == r_v:  # r_i is a predecessor of r_v
            # Only include if r_i has lower precedence (is more basic)
            if tau[r_i] < tau[r_v]:
                preceding_rels.append(r_i)
    
    return preceding_rels


def build_rdg_edges(
    graph,
    config: Optional[RDGConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, float], List[Tuple[int, int, float]]]:
    """
    Build RDG edges and compute precedence values.
    
    Args:
        graph: Graph object with edge_index and edge_type
        config: RDG configuration (optional)
    
    Returns:
        Tuple of:
        - edge_index: [2, num_rdg_edges] tensor of dependency edges
        - edge_weights: [num_rdg_edges] tensor of edge weights
        - tau: Dictionary mapping relation_id -> precedence_value
        - dependency_edges: List of (r_i, r_j, weight) tuples
    """
    if config is None:
        config = RDGConfig()
    
    # Extract dependency edges
    dependency_edges = extract_relation_dependencies(graph, config)
    
    if not dependency_edges:
        # No dependencies found, return empty tensors
        device = graph.edge_index.device
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weights = torch.empty((0,), dtype=torch.float, device=device)
        tau = {r: 0.5 for r in range(graph.num_relations)}
        return edge_index, edge_weights, tau, dependency_edges
    
    # Compute precedence values
    tau = compute_relation_precedence(
        dependency_edges,
        graph.num_relations,
        config
    )
    
    # Convert to tensors
    device = graph.edge_index.device
    edges_list = [(r_i, r_j) for r_i, r_j, _ in dependency_edges]
    weights_list = [weight for _, _, weight in dependency_edges]
    
    edge_index = torch.tensor(edges_list, dtype=torch.long, device=device).T  # [2, num_edges]
    edge_weights = torch.tensor(weights_list, dtype=torch.float, device=device)  # [num_edges]
    
    return edge_index, edge_weights, tau, dependency_edges

