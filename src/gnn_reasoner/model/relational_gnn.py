"""Relational GNN for predicate prediction.

Graph Neural Network that processes robot state graphs for spatial
and interaction predicate prediction from LeRobot trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_dense_adj

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

    class Data:  # type: ignore
        pass

    class Batch:  # type: ignore
        pass


if TYPE_CHECKING:
    from torch import Tensor


# Predicate definitions
SPATIAL_PREDICATES = [
    "is_near",  # Two nodes are within proximity threshold
    "is_above",  # Node i is above node j
    "is_below",  # Node i is below node j
    "is_left_of",  # Node i is left of node j
    "is_right_of",  # Node i is right of node j
]

INTERACTION_PREDICATES = [
    "is_holding",  # Gripper is holding an object
    "is_contacting",  # Two nodes are in contact
    "is_approaching",  # Node i is moving toward node j
    "is_retracting",  # Node i is moving away from node j
]

ALL_PREDICATES = SPATIAL_PREDICATES + INTERACTION_PREDICATES


@dataclass
class PredicateOutput:
    """Structured output from predicate prediction."""

    predicate_name: str
    source_node: int
    target_node: int
    probability: float
    active: bool  # True if prob > threshold


class NodeEncoder(nn.Module):
    """Encodes node features for relational reasoning."""

    def __init__(
        self,
        input_dim: int = 5,  # [angle, velocity, x, y, z]
        hidden_dim: int = 64,
        num_node_types: int = 3,  # joint, end_effector, object
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Node type embedding
        self.type_embed = nn.Embedding(num_node_types, hidden_dim // 4)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim - hidden_dim // 4),
        )

    def forward(self, x: Tensor, node_types: Tensor | None = None) -> Tensor:
        """Encode node features.

        Args:
            x: Node features (num_nodes, input_dim)
            node_types: Node type indices (num_nodes,)

        Returns:
            Encoded features (num_nodes, hidden_dim)
        """
        feat_embed = self.feature_encoder(x)

        if node_types is not None:
            type_embed = self.type_embed(node_types)
            return torch.cat([feat_embed, type_embed], dim=-1)

        # Pad if no node types provided
        padding = torch.zeros(x.size(0), self.hidden_dim // 4, device=x.device)
        return torch.cat([feat_embed, padding], dim=-1)


class EdgeEncoder(nn.Module):
    """Encodes edge attributes for relational reasoning."""

    def __init__(
        self,
        input_dim: int = 2,  # [distance, is_kinematic]
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        """Encode edge attributes."""
        return self.encoder(edge_attr)


class PredicateHead(nn.Module):
    """Predicts binary predicates between node pairs."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_predicates: int = len(ALL_PREDICATES),
    ):
        super().__init__()
        self.num_predicates = num_predicates

        # Pairwise predicate prediction (source + target concat)
        self.pairwise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_predicates),
        )

    def forward(self, node_embeddings: Tensor, edge_index: Tensor) -> Tensor:
        """Predict predicates for edge pairs.

        Args:
            node_embeddings: (num_nodes, hidden_dim)
            edge_index: (2, num_edges)

        Returns:
            Predicate logits (num_edges, num_predicates)
        """
        src_embed = node_embeddings[edge_index[0]]
        tgt_embed = node_embeddings[edge_index[1]]
        pair_embed = torch.cat([src_embed, tgt_embed], dim=-1)
        return self.pairwise_predictor(pair_embed)


class RelationalGNN(nn.Module):
    """Relational Graph Neural Network for predicate prediction.

    Takes graph representations of robot state and outputs:
    - Node embeddings for each joint/object
    - Predicate logits for spatial/interaction relationships
    - Graph-level embedding for action prediction

    Designed for imitation learning from LeRobot datasets.
    """

    def __init__(
        self,
        node_input_dim: int = 5,
        edge_input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_predicates: int = len(ALL_PREDICATES),
    ):
        """Initialize RelationalGNN.

        Args:
            node_input_dim: Input dimension for node features
            edge_input_dim: Input dimension for edge features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN message passing layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_predicates: Number of predicates to predict
        """
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for RelationalGNN")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_predicates = num_predicates

        # Encoders
        self.node_encoder = NodeEncoder(node_input_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(edge_input_dim, hidden_dim // 4)

        # GNN layers using GATv2 for improved attention
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim // 4,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Predicate prediction head
        self.predicate_head = PredicateHead(hidden_dim, num_predicates)

        # Graph-level output (for action prediction)
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> dict[str, Tensor]:
        """Forward pass through the relational GNN.

        Args:
            data: PyG Data object with:
                - x: Node features (num_nodes, node_input_dim)
                - edge_index: Edge connectivity (2, num_edges)
                - edge_attr: Edge features (num_edges, edge_input_dim)
                - node_types: Optional node type indices (num_nodes,)
                - batch: Optional batch indices (num_nodes,)

        Returns:
            Dictionary with:
                - node_embeddings: (num_nodes, hidden_dim)
                - predicate_logits: (num_edges, num_predicates)
                - graph_embedding: (batch_size, hidden_dim)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        node_types = data.node_types if hasattr(data, "node_types") else None
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        # Encode inputs
        x = self.node_encoder(x, node_types)

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None

        # Message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x + x_residual)
            x = F.relu(x)
            x = self.dropout(x)

        # Predicate prediction on edges
        predicate_logits = self.predicate_head(x, edge_index)

        # Graph-level embedding (concat mean + max pooling)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_embed = self.graph_head(torch.cat([mean_pool, max_pool], dim=-1))

        return {
            "node_embeddings": x,
            "predicate_logits": predicate_logits,
            "graph_embedding": graph_embed,
        }

    def predict_predicates(
        self,
        data: Data,
        threshold: float = 0.5,
    ) -> list[PredicateOutput]:
        """Predict active predicates for a graph.

        Args:
            data: PyG Data object
            threshold: Probability threshold for predicate activation

        Returns:
            List of PredicateOutput for all edges
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
            probs = torch.sigmoid(outputs["predicate_logits"])

            results = []
            edge_index = data.edge_index

            for edge_idx in range(probs.size(0)):
                src = edge_index[0, edge_idx].item()
                tgt = edge_index[1, edge_idx].item()

                for pred_idx, pred_name in enumerate(ALL_PREDICATES):
                    prob = probs[edge_idx, pred_idx].item()
                    results.append(
                        PredicateOutput(
                            predicate_name=pred_name,
                            source_node=src,
                            target_node=tgt,
                            probability=prob,
                            active=prob > threshold,
                        )
                    )

            return results

    def get_active_predicates(
        self,
        data: Data,
        threshold: float = 0.5,
    ) -> list[PredicateOutput]:
        """Get only active predicates above threshold.

        Args:
            data: PyG Data object
            threshold: Probability threshold

        Returns:
            List of active PredicateOutput
        """
        all_preds = self.predict_predicates(data, threshold)
        return [p for p in all_preds if p.active]

    def to_world_context(self, data: Data, threshold: float = 0.5) -> dict:
        """Convert predictions to structured world context for MCP.

        Args:
            data: PyG Data object
            threshold: Probability threshold

        Returns:
            Dictionary suitable for JSON serialization
        """
        outputs = self.forward(data)
        active_predicates = self.get_active_predicates(data, threshold)

        # Group predicates by type
        spatial = [p for p in active_predicates if p.predicate_name in SPATIAL_PREDICATES]
        interaction = [
            p for p in active_predicates if p.predicate_name in INTERACTION_PREDICATES
        ]

        return {
            "num_nodes": data.x.size(0),
            "num_edges": data.edge_index.size(1),
            "graph_embedding": outputs["graph_embedding"].tolist(),
            "spatial_predicates": [
                {
                    "predicate": p.predicate_name,
                    "source": p.source_node,
                    "target": p.target_node,
                    "confidence": round(p.probability, 3),
                }
                for p in spatial
            ],
            "interaction_predicates": [
                {
                    "predicate": p.predicate_name,
                    "source": p.source_node,
                    "target": p.target_node,
                    "confidence": round(p.probability, 3),
                }
                for p in interaction
            ],
        }

    def graph_to_json(self, data: Data) -> dict:
        """Convert graph structure to JSON-serializable format.

        Args:
            data: PyG Data object

        Returns:
            JSON-serializable dictionary with graph structure
        """
        nodes = []
        for i in range(data.x.size(0)):
            node_type = "joint"
            if hasattr(data, "node_types"):
                type_idx = data.node_types[i].item()
                node_type = ["joint", "end_effector", "object"][min(type_idx, 2)]

            nodes.append(
                {
                    "id": i,
                    "type": node_type,
                    "features": data.x[i].tolist(),
                }
            )

        edges = []
        for i in range(data.edge_index.size(1)):
            edge_data = {
                "source": data.edge_index[0, i].item(),
                "target": data.edge_index[1, i].item(),
            }
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                edge_data["attributes"] = data.edge_attr[i].tolist()
            edges.append(edge_data)

        return {
            "nodes": nodes,
            "edges": edges,
            "num_joints": data.num_joints if hasattr(data, "num_joints") else None,
        }

