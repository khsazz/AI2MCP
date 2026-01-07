"""Forward Dynamics Model for Pre-Execution Simulation.

Predicts future world state given current state and action sequence.
This enables LLM plan verification before physical execution.

Architecture:
    1. Current graph → RelationalGNN encoder → graph embedding
    2. Action vector → action encoder → action embedding
    3. Concat(graph_embed, action_embed) → dynamics network → predicted state delta
    4. Uncertainty head → confidence score

This is Phase 10.3: "Pre-Execution Simulation" for the MCP-GNN Physicality Filter.
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
    from torch_geometric.nn import global_mean_pool

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

    class Data:  # type: ignore
        pass

    class Batch:  # type: ignore
        pass


if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of forward simulation."""

    predicted_positions: Tensor  # (num_nodes, 3) predicted next positions
    position_delta: Tensor  # (num_nodes, 3) predicted change
    confidence: float  # Overall prediction confidence [0, 1]
    node_uncertainties: Tensor  # (num_nodes,) per-node uncertainty
    is_feasible: bool  # Whether predicted state is physically plausible


class ActionEncoder(nn.Module):
    """Encodes action vectors for dynamics prediction.
    
    ALOHA actions are 14-DoF joint velocity commands.
    """

    def __init__(
        self,
        action_dim: int = 14,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, action: Tensor) -> Tensor:
        """Encode action vector.
        
        Args:
            action: Action tensor (batch, action_dim) or (action_dim,)
            
        Returns:
            Action embedding (batch, hidden_dim) or (hidden_dim,)
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.encoder(action)


class DynamicsNetwork(nn.Module):
    """Predicts state transitions from graph + action embeddings.
    
    Takes concatenated [graph_embedding, action_embedding] and predicts:
    1. State delta (change in node positions)
    2. Uncertainty estimate (epistemic + aleatoric)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_nodes: int = 16,  # ALOHA: 14 joints + 2 grippers
        output_dim: int = 3,  # x, y, z position delta
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # Input: graph_embed (hidden_dim) + action_embed (hidden_dim)
        input_dim = hidden_dim * 2

        # Main dynamics network
        self.dynamics = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # State delta prediction head (per-node positions)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * output_dim),
        )

        # Uncertainty head (per-node uncertainty + global confidence)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes + 1),  # per-node + global
        )

        # Feasibility classifier (is the transition physically plausible?)
        self.feasibility_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        graph_embed: Tensor,
        action_embed: Tensor,
    ) -> dict[str, Tensor]:
        """Predict state transition.
        
        Args:
            graph_embed: Graph embedding (batch, hidden_dim)
            action_embed: Action embedding (batch, hidden_dim)
            
        Returns:
            Dictionary with:
                - delta: (batch, num_nodes, output_dim) position deltas
                - uncertainty: (batch, num_nodes) per-node uncertainties
                - confidence: (batch, 1) global confidence
                - feasibility: (batch, 1) feasibility logit
        """
        # Ensure batch dimension
        if graph_embed.dim() == 1:
            graph_embed = graph_embed.unsqueeze(0)
        if action_embed.dim() == 1:
            action_embed = action_embed.unsqueeze(0)

        # Concatenate embeddings
        combined = torch.cat([graph_embed, action_embed], dim=-1)

        # Forward through dynamics network
        features = self.dynamics(combined)

        # Predict state delta
        delta_flat = self.delta_head(features)
        delta = delta_flat.view(-1, self.num_nodes, self.output_dim)

        # Predict uncertainty
        uncertainty_out = self.uncertainty_head(features)
        node_uncertainty = F.softplus(uncertainty_out[:, :-1])  # Ensure positive
        global_confidence = torch.sigmoid(uncertainty_out[:, -1:])

        # Predict feasibility
        feasibility = self.feasibility_head(features)

        return {
            "delta": delta,
            "uncertainty": node_uncertainty,
            "confidence": global_confidence,
            "feasibility": feasibility,
        }


class ForwardDynamicsModel(nn.Module):
    """Complete forward dynamics model for pre-execution simulation.
    
    Combines:
    1. Graph encoder (from RelationalGNN, can be frozen)
    2. Action encoder
    3. Dynamics network
    
    Used for LLM plan verification via MCP `simulate_action` tool.
    
    Example:
        >>> model = ForwardDynamicsModel(gnn_encoder=pretrained_gnn)
        >>> result = model.simulate(current_graph, action_sequence, num_steps=5)
        >>> if result.confidence > 0.8 and result.is_feasible:
        ...     execute_action()
    """

    def __init__(
        self,
        gnn_encoder: nn.Module | None = None,
        hidden_dim: int = 128,
        action_dim: int = 14,
        num_nodes: int = 16,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize forward dynamics model.
        
        Args:
            gnn_encoder: Pre-trained GNN encoder (RelationalGNN or similar).
                         If None, creates a simple node aggregation encoder.
            hidden_dim: Hidden dimension for all components
            action_dim: Dimension of action vector (14 for ALOHA)
            num_nodes: Number of nodes in the graph (16 for ALOHA)
            freeze_encoder: If True, freeze the GNN encoder weights
            dropout: Dropout probability
        """
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for ForwardDynamicsModel")

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_nodes = num_nodes
        self.freeze_encoder = freeze_encoder

        # Graph encoder
        if gnn_encoder is not None:
            self.gnn_encoder = gnn_encoder
            if freeze_encoder:
                for param in self.gnn_encoder.parameters():
                    param.requires_grad = False
                self.gnn_encoder.eval()
            self._use_external_encoder = True
        else:
            # Simple fallback encoder
            self.gnn_encoder = self._build_simple_encoder(hidden_dim)
            self._use_external_encoder = False

        # Action encoder
        self.action_encoder = ActionEncoder(action_dim, hidden_dim)

        # Dynamics network
        self.dynamics = DynamicsNetwork(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            output_dim=3,
            dropout=dropout,
        )

        # Feasibility thresholds (learnable or fixed)
        self.register_buffer(
            "max_delta_threshold",
            torch.tensor(0.1),  # Max 10cm movement per step
        )
        self.register_buffer(
            "confidence_threshold",
            torch.tensor(0.7),  # Min confidence for feasible
        )

    def _build_simple_encoder(self, hidden_dim: int) -> nn.Module:
        """Build a simple encoder when no GNN is provided."""
        return nn.Sequential(
            nn.Linear(5, hidden_dim),  # node features: angle, vel, x, y, z
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode_graph(self, data: Data) -> Tensor:
        """Encode graph to embedding.
        
        Args:
            data: PyG Data object
            
        Returns:
            Graph embedding (batch, hidden_dim)
        """
        if self._use_external_encoder:
            # Use pre-trained GNN encoder
            with torch.set_grad_enabled(not self.freeze_encoder):
                outputs = self.gnn_encoder(data)
                return outputs["graph_embedding"]
        else:
            # Simple mean pooling of node features
            x = data.x
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            # Encode nodes
            encoded = self.gnn_encoder(x)

            # Global mean pooling
            return global_mean_pool(encoded, batch)

    def forward(
        self,
        data: Data,
        action: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass: predict next state from current state + action.
        
        Args:
            data: Current state as PyG Data object
            action: Action vector (batch, action_dim) or (action_dim,)
            
        Returns:
            Dictionary with predictions (see DynamicsNetwork.forward)
        """
        # Encode current state
        graph_embed = self.encode_graph(data)

        # Encode action
        action_embed = self.action_encoder(action)

        # Predict dynamics
        return self.dynamics(graph_embed, action_embed)

    def predict_next_state(
        self,
        data: Data,
        action: Tensor,
    ) -> tuple[Tensor, float, bool]:
        """Predict the next state positions.
        
        Args:
            data: Current state graph
            action: Action to apply
            
        Returns:
            Tuple of (predicted_positions, confidence, is_feasible)
        """
        outputs = self.forward(data, action)

        # Get current positions from graph
        current_positions = data.x[:, 2:5]  # x, y, z columns

        # Apply predicted delta
        delta = outputs["delta"].squeeze(0)  # (num_nodes, 3)

        # Handle node count mismatch (e.g., if objects detected)
        actual_nodes = current_positions.size(0)
        if delta.size(0) < actual_nodes:
            # Pad delta with zeros for extra nodes
            padding = torch.zeros(
                actual_nodes - delta.size(0), 3, device=delta.device
            )
            delta = torch.cat([delta, padding], dim=0)
        elif delta.size(0) > actual_nodes:
            # Truncate delta
            delta = delta[:actual_nodes]

        predicted_positions = current_positions + delta

        # Get confidence and feasibility
        confidence = outputs["confidence"].squeeze().item()
        feasibility_logit = outputs["feasibility"].squeeze().item()
        is_feasible = (
            feasibility_logit > 0
            and confidence > self.confidence_threshold.item()
            and delta.abs().max().item() < self.max_delta_threshold.item()
        )

        return predicted_positions, confidence, is_feasible

    def simulate(
        self,
        data: Data,
        actions: list[torch.Tensor] | torch.Tensor,
        return_trajectory: bool = False,
    ) -> SimulationResult | list[SimulationResult]:
        """Simulate a sequence of actions.
        
        Args:
            data: Initial state graph
            actions: List of action tensors or stacked tensor (T, action_dim)
            return_trajectory: If True, return all intermediate states
            
        Returns:
            SimulationResult or list of SimulationResults
        """
        self.eval()

        if isinstance(actions, torch.Tensor):
            if actions.dim() == 1:
                actions = [actions]
            else:
                actions = [actions[i] for i in range(actions.size(0))]

        results = []
        current_data = data

        with torch.no_grad():
            for action in actions:
                outputs = self.forward(current_data, action)

                # Get current positions
                current_positions = current_data.x[:, 2:5]
                delta = outputs["delta"].squeeze(0)

                # Handle size mismatch
                actual_nodes = current_positions.size(0)
                if delta.size(0) != actual_nodes:
                    if delta.size(0) < actual_nodes:
                        padding = torch.zeros(
                            actual_nodes - delta.size(0), 3, device=delta.device
                        )
                        delta = torch.cat([delta, padding], dim=0)
                    else:
                        delta = delta[:actual_nodes]

                predicted_positions = current_positions + delta

                confidence = outputs["confidence"].squeeze().item()
                node_uncertainties = outputs["uncertainty"].squeeze()

                # Handle uncertainty size mismatch
                if node_uncertainties.size(0) != actual_nodes:
                    if node_uncertainties.size(0) < actual_nodes:
                        padding = torch.ones(
                            actual_nodes - node_uncertainties.size(0),
                            device=node_uncertainties.device,
                        )
                        node_uncertainties = torch.cat(
                            [node_uncertainties, padding], dim=0
                        )
                    else:
                        node_uncertainties = node_uncertainties[:actual_nodes]

                feasibility_logit = outputs["feasibility"].squeeze().item()
                is_feasible = (
                    feasibility_logit > 0
                    and confidence > self.confidence_threshold.item()
                    and delta.abs().max().item() < self.max_delta_threshold.item()
                )

                result = SimulationResult(
                    predicted_positions=predicted_positions,
                    position_delta=delta,
                    confidence=confidence,
                    node_uncertainties=node_uncertainties,
                    is_feasible=is_feasible,
                )
                results.append(result)

                # Update graph for next step (autoregressive)
                if len(actions) > 1:
                    # Clone and update positions
                    new_x = current_data.x.clone()
                    new_x[:, 2:5] = predicted_positions
                    current_data = Data(
                        x=new_x,
                        edge_index=current_data.edge_index,
                        edge_attr=current_data.edge_attr
                        if hasattr(current_data, "edge_attr")
                        else None,
                        node_types=current_data.node_types
                        if hasattr(current_data, "node_types")
                        else None,
                    )

        if return_trajectory or len(results) > 1:
            return results
        return results[0]

    def to_mcp_response(
        self,
        results: SimulationResult | list[SimulationResult],
    ) -> dict:
        """Convert simulation results to MCP-compatible JSON response.
        
        Args:
            results: Single result or list of results
            
        Returns:
            JSON-serializable dictionary for MCP tool response
        """
        if isinstance(results, SimulationResult):
            results = [results]

        trajectory = []
        overall_feasible = True
        min_confidence = 1.0

        for i, result in enumerate(results):
            step_data = {
                "step": i + 1,
                "confidence": round(result.confidence, 3),
                "is_feasible": result.is_feasible,
                "max_delta": round(result.position_delta.abs().max().item(), 4),
                "mean_uncertainty": round(result.node_uncertainties.mean().item(), 4),
            }
            trajectory.append(step_data)

            overall_feasible = overall_feasible and result.is_feasible
            min_confidence = min(min_confidence, result.confidence)

        return {
            "num_steps": len(results),
            "overall_feasible": overall_feasible,
            "min_confidence": round(min_confidence, 3),
            "trajectory": trajectory,
            "recommendation": (
                "EXECUTE" if overall_feasible and min_confidence > 0.7 else "REPLAN"
            ),
        }


class ForwardDynamicsLoss(nn.Module):
    """Loss function for training forward dynamics model.
    
    Combines:
    1. State prediction loss (MSE on position delta)
    2. Uncertainty calibration loss (NLL with predicted variance)
    3. Feasibility classification loss (BCE)
    4. Temporal consistency regularization
    """

    def __init__(
        self,
        delta_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        feasibility_weight: float = 0.5,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.delta_weight = delta_weight
        self.uncertainty_weight = uncertainty_weight
        self.feasibility_weight = feasibility_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        predictions: dict[str, Tensor],
        target_delta: Tensor,
        target_feasible: Tensor | None = None,
        prev_delta: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute training loss.
        
        Args:
            predictions: Output from ForwardDynamicsModel
            target_delta: Ground truth position delta (batch, num_nodes, 3)
            target_feasible: Ground truth feasibility (batch,) or None
            prev_delta: Previous step's delta for consistency (batch, num_nodes, 3)
            
        Returns:
            Dictionary with individual losses and total
        """
        pred_delta = predictions["delta"]
        pred_uncertainty = predictions["uncertainty"]
        pred_feasibility = predictions["feasibility"]

        # 1. Delta prediction loss (MSE)
        delta_loss = F.mse_loss(pred_delta, target_delta)

        # 2. Uncertainty-aware loss (negative log likelihood)
        # Treat uncertainty as log variance
        variance = pred_uncertainty.unsqueeze(-1).expand_as(pred_delta)
        nll = 0.5 * (
            torch.log(variance + 1e-6)
            + (pred_delta - target_delta).pow(2) / (variance + 1e-6)
        )
        uncertainty_loss = nll.mean()

        # 3. Feasibility loss (BCE)
        if target_feasible is not None:
            feasibility_loss = F.binary_cross_entropy_with_logits(
                pred_feasibility.squeeze(-1),
                target_feasible.float(),
            )
        else:
            # Self-supervised: feasible if delta is small
            pseudo_feasible = (target_delta.abs().max(dim=-1)[0].max(dim=-1)[0] < 0.1).float()
            feasibility_loss = F.binary_cross_entropy_with_logits(
                pred_feasibility.squeeze(-1),
                pseudo_feasible,
            )

        # 4. Temporal consistency (smooth predictions)
        if prev_delta is not None:
            consistency_loss = F.mse_loss(pred_delta, prev_delta)
        else:
            consistency_loss = torch.tensor(0.0, device=pred_delta.device)

        # Total loss
        total_loss = (
            self.delta_weight * delta_loss
            + self.uncertainty_weight * uncertainty_loss
            + self.feasibility_weight * feasibility_loss
            + self.consistency_weight * consistency_loss
        )

        return {
            "total": total_loss,
            "delta": delta_loss,
            "uncertainty": uncertainty_loss,
            "feasibility": feasibility_loss,
            "consistency": consistency_loss,
        }

