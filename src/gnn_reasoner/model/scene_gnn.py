"""Scene GNN for semantic understanding.

Graph Neural Network that processes world graphs for scene classification
and entity relationship reasoning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Mock classes for development
    class Data:  # type: ignore
        pass
    class Batch:  # type: ignore
        pass

if TYPE_CHECKING:
    from torch import Tensor
    from gnn_reasoner.graph_builder import WorldGraph


# Scene classes for indoor environments
SCENE_CLASSES = [
    "corridor",
    "room",
    "doorway",
    "open_space",
    "cluttered",
    "unknown",
]

# Node type embeddings
NODE_TYPES = {
    "robot": 0,
    "obstacle": 1,
    "object": 2,
    "landmark": 3,
    "wall": 4,
}

# Edge relation embeddings
EDGE_RELATIONS = {
    "near": 0,
    "blocking": 1,
    "visible": 2,
    "adjacent": 3,
    "reachable": 4,
}


class GraphEncoder(nn.Module):
    """Encodes world graph nodes and edges into feature tensors."""

    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 4,
        position_dim: int = 2,
    ):
        """Initialize graph encoder.
        
        Args:
            node_dim: Output dimension for node features
            edge_dim: Output dimension for edge features
            position_dim: Dimension of position coordinates (2 for 2D)
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Embeddings
        self.node_type_embed = nn.Embedding(len(NODE_TYPES), node_dim // 2)
        self.edge_type_embed = nn.Embedding(len(EDGE_RELATIONS), edge_dim // 2)
        
        # Position encoder
        self.position_encoder = nn.Linear(position_dim, node_dim // 2)
        
        # Edge attribute encoder (distance, weight)
        self.edge_attr_encoder = nn.Linear(2, edge_dim // 2)

    def encode_nodes(self, graph: WorldGraph) -> tuple[Tensor, dict[str, int]]:
        """Encode graph nodes to feature tensor.
        
        Returns:
            node_features: (num_nodes, node_dim) tensor
            node_id_map: mapping from node id to index
        """
        num_nodes = len(graph.nodes)
        if num_nodes == 0:
            return torch.zeros(0, self.node_dim), {}

        node_id_map = {n.id: i for i, n in enumerate(graph.nodes)}
        
        # Extract features
        type_indices = torch.tensor([
            NODE_TYPES.get(n.node_type, len(NODE_TYPES) - 1)
            for n in graph.nodes
        ])
        positions = torch.tensor([
            list(n.position) for n in graph.nodes
        ], dtype=torch.float32)

        # Encode
        type_embed = self.node_type_embed(type_indices)
        pos_embed = self.position_encoder(positions)

        # Concatenate
        node_features = torch.cat([type_embed, pos_embed], dim=1)
        
        return node_features, node_id_map

    def encode_edges(
        self,
        graph: WorldGraph,
        node_id_map: dict[str, int],
    ) -> tuple[Tensor, Tensor]:
        """Encode graph edges to index and feature tensors.
        
        Returns:
            edge_index: (2, num_edges) tensor of node indices
            edge_attr: (num_edges, edge_dim) tensor of edge features
        """
        if not graph.edges:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, self.edge_dim)

        edge_list = []
        relation_indices = []
        edge_attrs = []

        for e in graph.edges:
            if e.source not in node_id_map or e.target not in node_id_map:
                continue

            src_idx = node_id_map[e.source]
            tgt_idx = node_id_map[e.target]
            
            # Add both directions for undirected graph
            edge_list.append([src_idx, tgt_idx])
            edge_list.append([tgt_idx, src_idx])
            
            rel_idx = EDGE_RELATIONS.get(e.relation, len(EDGE_RELATIONS) - 1)
            relation_indices.extend([rel_idx, rel_idx])
            
            dist = e.attributes.get("distance", 1.0)
            attr = [dist, e.weight]
            edge_attrs.extend([attr, attr])

        if not edge_list:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, self.edge_dim)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        relation_embed = self.edge_type_embed(torch.tensor(relation_indices))
        attr_embed = self.edge_attr_encoder(torch.tensor(edge_attrs, dtype=torch.float32))
        edge_attr = torch.cat([relation_embed, attr_embed], dim=1)

        return edge_index, edge_attr


class SceneGNN(nn.Module):
    """Graph Neural Network for scene understanding.
    
    Takes a world graph and outputs:
    - Node embeddings for each entity
    - Scene classification logits
    - Navigation-relevant features
    """

    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize SceneGNN.
        
        Args:
            node_dim: Input node feature dimension
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout probability
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for SceneGNN")

        self.encoder = GraphEncoder(node_dim=node_dim, edge_dim=edge_dim)
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # GNN layers (GAT for attention-based message passing)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output heads
        self.scene_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(SCENE_CLASSES)),
        )
        
        self.nav_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # [can_move_forward, clearance, complexity, risk]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        data: Data,
    ) -> dict[str, Tensor]:
        """Forward pass through the GNN.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
            
        Returns:
            Dictionary with:
            - node_embeddings: (num_nodes, hidden_dim)
            - scene_logits: (batch_size, num_scene_classes)
            - nav_features: (batch_size, 4)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x + x_residual)  # Residual connection
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling for graph-level features
        graph_embed = global_mean_pool(x, batch)

        # Output heads
        scene_logits = self.scene_classifier(graph_embed)
        nav_features = torch.sigmoid(self.nav_head(graph_embed))

        return {
            "node_embeddings": x,
            "scene_logits": scene_logits,
            "nav_features": nav_features,
            "graph_embedding": graph_embed,
        }

    def predict_scene(self, data: Data) -> tuple[str, float]:
        """Predict scene class for a single graph.
        
        Returns:
            scene_class: Predicted scene name
            confidence: Prediction confidence
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
            probs = F.softmax(outputs["scene_logits"], dim=-1)
            confidence, pred_idx = probs.max(dim=-1)
            
            return SCENE_CLASSES[pred_idx.item()], confidence.item()

    @classmethod
    def from_world_graph(cls, graph: WorldGraph, model: SceneGNN | None = None) -> Data:
        """Convert WorldGraph to PyG Data object.
        
        Args:
            graph: WorldGraph to convert
            model: Optional model to use for encoding (uses default encoder if None)
            
        Returns:
            PyG Data object ready for forward pass
        """
        if model is None:
            encoder = GraphEncoder()
        else:
            encoder = model.encoder

        node_features, node_id_map = encoder.encode_nodes(graph)
        edge_index, edge_attr = encoder.encode_edges(graph, node_id_map)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

