"""Spatiotemporal GNN for temporal predicate stability and future prediction.

Phase 11: Predictive Temporal Verifiers (ST-GNN)

Architecture (Option C: Hybrid):
- Base: RelationalGNN encoder (frozen or trainable)
- Temporal: GRU layer after graph-level pooling
- Future: Action-conditioned future prediction head

This addresses the documented limitation of frame-by-frame predicate flicker
by maintaining temporal state across frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import global_mean_pool, global_max_pool

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

    class Data:  # type: ignore
        pass

    class Batch:  # type: ignore
        pass


if TYPE_CHECKING:
    from torch import Tensor
    from gnn_reasoner.model.relational_gnn import RelationalGNN

logger = logging.getLogger(__name__)


@dataclass
class TemporalPrediction:
    """Future predicate prediction result."""

    step: int  # Steps into future (1, 2, 3, ...)
    predicate_logits: torch.Tensor  # (num_edges, num_predicates)
    confidence: float  # Overall prediction confidence [0, 1]
    graph_embedding: torch.Tensor  # (hidden_dim,)


class TemporalGRU(nn.Module):
    """GRU layer for temporal graph embedding processing.
    
    Maintains hidden state across frames to enable temporal consistency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """Initialize temporal GRU.
        
        Args:
            input_dim: Input dimension (graph embedding size)
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        graph_embeddings: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process sequence of graph embeddings.
        
        Args:
            graph_embeddings: (batch, seq_len, input_dim) or (batch, input_dim) for single step
            hidden_state: Previous hidden state (num_layers, batch, hidden_dim) or None
            
        Returns:
            output: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            hidden_state: (num_layers, batch, hidden_dim)
        """
        # Handle single step (add sequence dimension)
        if graph_embeddings.dim() == 2:
            graph_embeddings = graph_embeddings.unsqueeze(1)  # (batch, 1, input_dim)
            single_step = True
        else:
            single_step = False
            
        # Forward through GRU
        output, hidden = self.gru(graph_embeddings, hidden_state)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Remove sequence dimension if single step
        if single_step:
            output = output.squeeze(1)  # (batch, hidden_dim)
            
        return output, hidden


class FutureProjectionHead(nn.Module):
    """Predicts future predicates from current state + action.
    
    Architecture:
    - Input: [temporal_embedding, action_embedding]
    - Output: Predicate logits for future steps
    """
    
    def __init__(
        self,
        temporal_dim: int,
        action_dim: int = 14,  # ALOHA 14-DoF
        hidden_dim: int = 128,
        num_predicates: int = 9,
        num_future_steps: int = 3,
    ):
        """Initialize future projection head.
        
        Args:
            temporal_dim: Dimension of temporal graph embedding
            action_dim: Dimension of action vector
            hidden_dim: Hidden dimension for projection network
            num_predicates: Number of predicates to predict
            num_future_steps: Number of future steps to predict
        """
        super().__init__()
        self.num_future_steps = num_future_steps
        self.num_predicates = num_predicates
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        
        # Combined projection network
        input_dim = temporal_dim + hidden_dim // 2
        self.projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_future_steps)
        ])
        
        # Predicate prediction heads (one per future step)
        self.predicate_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_predicates)
            for _ in range(num_future_steps)
        ])
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        temporal_embedding: torch.Tensor,
        action: torch.Tensor,
        num_edges: int,
    ) -> list[TemporalPrediction]:
        """Predict future predicates.
        
        Args:
            temporal_embedding: (batch, temporal_dim) temporal graph embedding
            action: (batch, action_dim) or (action_dim,) action vector
            num_edges: Number of edges in the graph (for predicate logits shape)
            
        Returns:
            List of TemporalPrediction for each future step
        """
        # Ensure batch dimension
        if temporal_embedding.dim() == 1:
            temporal_embedding = temporal_embedding.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        batch_size = temporal_embedding.size(0)
        
        # Encode action
        action_embed = self.action_encoder(action)  # (batch, hidden_dim // 2)
        
        # Combine temporal + action
        combined = torch.cat([temporal_embedding, action_embed], dim=-1)  # (batch, input_dim)
        
        predictions = []
        
        # Predict each future step
        for step_idx in range(self.num_future_steps):
            # Project through step-specific network
            features = self.projection[step_idx](combined)  # (batch, hidden_dim)
            
            # Predict predicates (broadcast to all edges)
            # Shape: (batch, num_predicates) -> (batch, num_edges, num_predicates)
            pred_logits = self.predicate_heads[step_idx](features)  # (batch, num_predicates)
            pred_logits = pred_logits.unsqueeze(1).expand(-1, num_edges, -1)  # (batch, num_edges, num_predicates)
            
            # Predict confidence
            confidence = self.confidence_head(features).squeeze(-1)  # (batch,)
            
            # For single graph, squeeze batch dimension
            if batch_size == 1:
                pred_logits = pred_logits.squeeze(0)  # (num_edges, num_predicates)
                confidence_val = confidence.item()
                graph_embed = features.squeeze(0)  # (hidden_dim,)
            else:
                confidence_val = confidence.mean().item()
                graph_embed = features.mean(dim=0)  # (hidden_dim,)
            
            predictions.append(
                TemporalPrediction(
                    step=step_idx + 1,
                    predicate_logits=pred_logits,
                    confidence=confidence_val,
                    graph_embedding=graph_embed,
                )
            )
            
        return predictions


class SpatiotemporalGNN(nn.Module):
    """Spatiotemporal GNN with temporal memory and future prediction.
    
    Architecture (Option C: Hybrid):
    1. RelationalGNN base encoder (processes single frame)
    2. Temporal GRU (maintains state across frames)
    3. Future projection head (predicts future predicates from action)
    
    This addresses predicate flicker by maintaining temporal consistency
    and enables proactive AI through future state prediction.
    """
    
    def __init__(
        self,
        base_gnn: nn.Module | None = None,
        hidden_dim: int = 128,
        temporal_hidden_dim: int = 128,
        num_temporal_layers: int = 1,
        action_dim: int = 14,
        num_predicates: int = 9,
        num_future_steps: int = 3,
        freeze_base: bool = False,
        dropout: float = 0.1,
    ):
        """Initialize SpatiotemporalGNN.
        
        Args:
            base_gnn: Pre-trained RelationalGNN encoder. If None, creates new one.
            hidden_dim: Hidden dimension for base GNN and temporal layers
            temporal_hidden_dim: GRU hidden dimension (can differ from base)
            num_temporal_layers: Number of GRU layers
            action_dim: Action vector dimension (14 for ALOHA)
            num_predicates: Number of predicates to predict
            num_future_steps: Number of future steps for projection
            freeze_base: If True, freeze base GNN weights
            dropout: Dropout probability
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for SpatiotemporalGNN")
        
        self.hidden_dim = hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.num_predicates = num_predicates
        self.num_future_steps = num_future_steps
        self.freeze_base = freeze_base
        
        # Base GNN encoder (RelationalGNN)
        if base_gnn is not None:
            self.base_gnn = base_gnn
            if freeze_base:
                for param in self.base_gnn.parameters():
                    param.requires_grad = False
                self.base_gnn.eval()
        else:
            # Create new RelationalGNN if not provided
            from gnn_reasoner.model.relational_gnn import RelationalGNN
            self.base_gnn = RelationalGNN(
                hidden_dim=hidden_dim,
                num_predicates=num_predicates,
            )
        
        # Get graph embedding dimension from base GNN
        # RelationalGNN outputs graph_embedding of size hidden_dim
        graph_embed_dim = hidden_dim
        
        # Temporal GRU layer
        self.temporal_gru = TemporalGRU(
            input_dim=graph_embed_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )
        
        # Future projection head
        self.future_head = FutureProjectionHead(
            temporal_dim=temporal_hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_predicates=num_predicates,
            num_future_steps=num_future_steps,
        )
        
        # Temporal predicate head (learnable even when base is frozen)
        # Uses temporal embedding to produce predicate logits
        self.temporal_predicate_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates),
        )
        
        # Store hidden state for temporal continuity
        self._hidden_state: torch.Tensor | None = None
        
    def reset_hidden_state(self):
        """Reset temporal hidden state (e.g., on frame jump)."""
        self._hidden_state = None
        
    def forward(
        self,
        data: Data,
        hidden_state: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through spatiotemporal GNN.
        
        Args:
            data: PyG Data object (single frame)
            hidden_state: Previous GRU hidden state (optional, uses internal if None)
            return_hidden: If True, return updated hidden state
            
        Returns:
            Dictionary with:
                - node_embeddings: (num_nodes, hidden_dim)
                - predicate_logits: (num_edges, num_predicates)
                - graph_embedding: (hidden_dim,)
                - temporal_embedding: (temporal_hidden_dim,)
                - hidden_state: (num_layers, 1, temporal_hidden_dim) if return_hidden
        """
        # Use internal hidden state if not provided
        if hidden_state is None:
            hidden_state = self._hidden_state
        
        # Base GNN forward (process single frame)
        with torch.set_grad_enabled(not self.freeze_base):
            base_outputs = self.base_gnn(data)
        
        # Extract graph embedding
        graph_embed = base_outputs["graph_embedding"]  # (1, hidden_dim) or (hidden_dim,)
        
        # Ensure batch dimension for GRU
        if graph_embed.dim() == 1:
            graph_embed = graph_embed.unsqueeze(0)  # (1, hidden_dim)
        
        # Temporal processing through GRU
        temporal_embed, updated_hidden = self.temporal_gru(graph_embed, hidden_state)
        
        # Update internal hidden state
        self._hidden_state = updated_hidden
        
        # Compute predicate logits using temporal embedding (ensures gradients flow)
        # The temporal predicate head is always learnable even when base GNN is frozen
        
        # Get edge info
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Handle both single graphs and batched graphs (from Batch.from_data_list)
        if hasattr(data, 'batch') and data.batch is not None:
            # Batched graphs: temporal_embed is (batch_size, hidden_dim)
            # We need to broadcast to edges based on which graph each edge belongs to
            
            # Get edge-to-graph mapping (which graph each edge belongs to)
            # For each edge, find which graph the source node belongs to
            edge_batch = data.batch[edge_index[0]]  # (num_edges,)
            
            # Gather temporal embedding for each edge based on its graph
            temporal_broadcast = temporal_embed[edge_batch]  # (num_edges, hidden_dim)
            temporal_embed_out = temporal_embed  # (batch_size, hidden_dim)
        else:
            # Single graph: broadcast temporal embedding to all edges
            temporal_embed_squeezed = temporal_embed.squeeze(0) if temporal_embed.dim() > 1 else temporal_embed
            temporal_broadcast = temporal_embed_squeezed.unsqueeze(0).expand(num_edges, -1)  # (num_edges, hidden_dim)
            temporal_embed_out = temporal_embed_squeezed
        
        # Use temporal embedding directly for predicate prediction
        # This ensures gradient flows through the temporal layers
        predicate_logits = self.temporal_predicate_head(temporal_broadcast)  # (num_edges, num_predicates)
        
        # Prepare output
        output = {
            "node_embeddings": base_outputs["node_embeddings"],
            "predicate_logits": predicate_logits,
            "graph_embedding": base_outputs["graph_embedding"],
            "temporal_embedding": temporal_embed_out,
        }
        
        if return_hidden:
            output["hidden_state"] = updated_hidden
            
        return output
    
    def predict_future(
        self,
        data: Data,
        action: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[list[TemporalPrediction], dict[str, torch.Tensor]]:
        """Predict future predicates given current state + action.
        
        Args:
            data: Current frame graph
            action: Action vector (14-DoF for ALOHA)
            hidden_state: Previous GRU hidden state (optional)
            
        Returns:
            Tuple of:
                - List of TemporalPrediction for each future step
                - Current frame outputs (same as forward())
        """
        # Get current frame prediction
        current_outputs = self.forward(data, hidden_state, return_hidden=False)
        
        # Extract temporal embedding
        temporal_embed = current_outputs["temporal_embedding"]
        
        # Get number of edges
        num_edges = data.edge_index.size(1) if hasattr(data, "edge_index") else 0
        
        # Predict future
        future_predictions = self.future_head(
            temporal_embed,
            action,
            num_edges,
        )
        
        return future_predictions, current_outputs
    
    def predict_predicates(
        self,
        data: Data,
        threshold: float = 0.5,
    ) -> list:
        """Predict active predicates (delegates to base GNN).
        
        Args:
            data: PyG Data object
            threshold: Probability threshold
            
        Returns:
            List of PredicateOutput
        """
        # Use base GNN's predicate prediction (with temporal smoothing)
        return self.base_gnn.predict_predicates(data, threshold)
    
    def to_world_context(
        self,
        data: Data,
        threshold: float = 0.5,
    ) -> dict:
        """Convert predictions to structured world context for MCP.
        
        Args:
            data: PyG Data object
            threshold: Probability threshold
            
        Returns:
            Dictionary suitable for JSON serialization
        """
        # Use base GNN's world context (temporal consistency handled internally)
        return self.base_gnn.to_world_context(data, threshold)

