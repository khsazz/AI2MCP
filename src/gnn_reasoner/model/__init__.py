"""GNN model implementations."""

from gnn_reasoner.model.scene_gnn import SceneGNN, GraphEncoder
from gnn_reasoner.model.relational_gnn import (
    RelationalGNN,
    PredicateOutput,
    ALL_PREDICATES,
    SPATIAL_PREDICATES,
    INTERACTION_PREDICATES,
    NodeEncoder,
    EdgeEncoder,
    PredicateHead,
    ConditionalPredicateHead,
)
from gnn_reasoner.model.multimodal_gnn import (
    MultiModalGNN,
    VisionEncoder,
    CrossAttentionFusion,
    MockVisionEncoder,
)

__all__ = [
    # Scene GNN (legacy)
    "SceneGNN",
    "GraphEncoder",
    # Relational GNN (Option A baseline)
    "RelationalGNN",
    "PredicateOutput",
    "ALL_PREDICATES",
    "SPATIAL_PREDICATES",
    "INTERACTION_PREDICATES",
    "NodeEncoder",
    "EdgeEncoder",
    "PredicateHead",
    "ConditionalPredicateHead",
    # Multi-Modal GNN (Option C)
    "MultiModalGNN",
    "VisionEncoder",
    "CrossAttentionFusion",
    "MockVisionEncoder",
]

