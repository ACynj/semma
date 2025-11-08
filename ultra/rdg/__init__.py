"""
Relation Dependency Graph (RDG) Module

This module implements the Relation Dependency Graph construction as proposed in GRAPHORACLE.
It extracts relation dependencies from knowledge graphs and builds a hierarchical relation graph
with precedence ordering.
"""

from .rdg_builder import (
    extract_relation_dependencies,
    compute_relation_precedence,
    get_preceding_relations,
    build_rdg_edges,
    RDGConfig
)

__all__ = [
    'extract_relation_dependencies',
    'compute_relation_precedence',
    'get_preceding_relations',
    'build_rdg_edges',
    'RDGConfig'
]

