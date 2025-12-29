"""LeRobot State Resources.

Exposes LeRobot dataset state, world graph, and predicted predicates
as MCP resources for AI agent consumption.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.types import Resource, TextContent

if TYPE_CHECKING:
    from gnn_reasoner.data_manager import DataManager
    from gnn_reasoner.lerobot_transformer import LeRobotGraphTransformer
    from gnn_reasoner.model.relational_gnn import RelationalGNN


class LeRobotResourceManager:
    """Manages LeRobot dataset resources for MCP exposure."""

    def __init__(
        self,
        data_manager: DataManager,
        graph_transformer: LeRobotGraphTransformer,
        gnn_model: RelationalGNN,
    ):
        """Initialize the resource manager.

        Args:
            data_manager: DataManager instance for LeRobot dataset access
            graph_transformer: Transformer for state-to-graph conversion
            gnn_model: RelationalGNN model for predicate prediction
        """
        self.data_manager = data_manager
        self.graph_transformer = graph_transformer
        self.gnn_model = gnn_model
        self._predicate_threshold: float = 0.5

    def get_current_state(self) -> dict:
        """Get the current observation state as JSON-serializable dict."""
        frame = self.data_manager.get_current_frame()

        state_data = {
            "frame_index": self.data_manager._current_frame_idx,
            "total_frames": len(self.data_manager),
            "repo_id": self.data_manager.repo_id,
        }

        # Extract observation state
        if "observation.state" in frame:
            state = frame["observation.state"]
            if hasattr(state, "tolist"):
                state = state.tolist()
            state_data["observation_state"] = state

        # Extract action
        if "action" in frame:
            action = frame["action"]
            if hasattr(action, "tolist"):
                action = action.tolist()
            state_data["action"] = action

        # List available cameras
        cameras = [k.replace("observation.images.", "") for k in frame.keys()
                   if k.startswith("observation.images.")]
        state_data["available_cameras"] = cameras

        return state_data

    def get_world_graph(self) -> dict:
        """Get GNN-processed world graph from current state."""
        import torch

        frame = self.data_manager.get_current_frame()
        state = frame.get("observation.state")

        if state is None:
            return {
                "error": "No observation state available",
                "frame_index": self.data_manager._current_frame_idx,
            }

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Convert to graph
        graph_data = self.graph_transformer.to_graph(state)

        # Get graph structure as JSON
        graph_json = self.gnn_model.graph_to_json(graph_data)
        graph_json["frame_index"] = self.data_manager._current_frame_idx
        graph_json["source"] = "lerobot"

        return graph_json

    def get_predicates(self) -> dict:
        """Get predicted predicates from GNN analysis."""
        import torch

        frame = self.data_manager.get_current_frame()
        state = frame.get("observation.state")

        if state is None:
            return {
                "error": "No observation state available",
                "frame_index": self.data_manager._current_frame_idx,
            }

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Convert to graph and get world context
        graph_data = self.graph_transformer.to_graph(state)
        world_context = self.gnn_model.to_world_context(graph_data, self._predicate_threshold)

        world_context["frame_index"] = self.data_manager._current_frame_idx
        world_context["threshold"] = self._predicate_threshold

        return world_context

    def set_predicate_threshold(self, threshold: float) -> None:
        """Set the predicate activation threshold."""
        self._predicate_threshold = max(0.0, min(1.0, threshold))


def register_lerobot_resources(
    server: Server,
    resource_manager: LeRobotResourceManager,
) -> None:
    """Register LeRobot resources with MCP server.

    Args:
        server: MCP Server instance
        resource_manager: LeRobotResourceManager for data access
    """
    existing_list = server._resource_handlers.get("list_resources")

    @server.list_resources()
    async def list_lerobot_resources() -> list[Resource]:
        """List LeRobot-specific resources."""
        existing_resources = await existing_list() if existing_list else []

        lerobot_resources = [
            Resource(
                uri="robot://lerobot/current_state",
                name="LeRobot Current State",
                description=(
                    "Current observation frame from LeRobot dataset including "
                    "joint states, actions, and available camera feeds."
                ),
                mimeType="application/json",
            ),
            Resource(
                uri="robot://lerobot/world_graph",
                name="LeRobot World Graph",
                description=(
                    "GNN-processed relational graph representation of the robot state. "
                    "Includes nodes (joints, end-effectors, objects) and edges (kinematic links, proximity)."
                ),
                mimeType="application/json",
            ),
            Resource(
                uri="robot://lerobot/predicates",
                name="LeRobot Predicates",
                description=(
                    "Predicted spatial and interaction predicates from GNN analysis. "
                    "Includes is_near, is_holding, is_contacting, etc."
                ),
                mimeType="application/json",
            ),
            Resource(
                uri="robot://lerobot/dataset_info",
                name="LeRobot Dataset Info",
                description="Metadata about the loaded LeRobot dataset.",
                mimeType="application/json",
            ),
        ]

        return existing_resources + lerobot_resources

    @server.read_resource()
    async def read_lerobot_resource(uri: str) -> list[TextContent]:
        """Read LeRobot resources."""
        if uri == "robot://lerobot/current_state":
            data = resource_manager.get_current_state()

        elif uri == "robot://lerobot/world_graph":
            data = resource_manager.get_world_graph()

        elif uri == "robot://lerobot/predicates":
            data = resource_manager.get_predicates()

        elif uri == "robot://lerobot/dataset_info":
            dm = resource_manager.data_manager
            data = {
                "repo_id": dm.repo_id,
                "total_frames": len(dm),
                "num_episodes": dm.num_episodes,
                "state_dim": dm.state_dim,
                "action_dim": dm.action_dim,
                "streaming": dm.streaming,
            }

        else:
            data = {"error": f"Unknown resource: {uri}"}

        return [TextContent(type="text", text=json.dumps(data, indent=2))]

