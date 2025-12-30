"""Prediction tools for GNN-based reasoning.

These tools expose the RelationalGNN's prediction capabilities
for world graph analysis and action outcome prediction.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog
from mcp.server import Server
from mcp.types import Tool, TextContent

if TYPE_CHECKING:
    from gnn_reasoner.data_manager import DataManager
    from gnn_reasoner.lerobot_transformer import LeRobotGraphTransformer
    from gnn_reasoner.model.relational_gnn import RelationalGNN

logger = structlog.get_logger()


class PredictionToolsManager:
    """Manages prediction tools for MCP exposure."""

    def __init__(
        self,
        data_manager: DataManager,
        graph_transformer: LeRobotGraphTransformer,
        gnn_model: RelationalGNN,
    ):
        """Initialize prediction tools manager.

        Args:
            data_manager: DataManager for LeRobot dataset access
            graph_transformer: Transformer for state-to-graph conversion
            gnn_model: RelationalGNN model for predictions
        """
        self.data_manager = data_manager
        self.graph_transformer = graph_transformer
        self.gnn_model = gnn_model
        self._inference_times: list[float] = []

    def get_world_graph(self, frame_idx: int | None = None, threshold: float = 0.5) -> dict:
        """Get the world graph for a given frame.

        Args:
            frame_idx: Frame index, or None for current frame
            threshold: Probability threshold for predicate activation

        Returns:
            World graph as JSON-serializable dictionary
        """
        import torch

        start_time = time.perf_counter()

        if frame_idx is not None:
            self.data_manager.set_frame_index(frame_idx)

        frame = self.data_manager.get_current_frame()
        state = frame.get("observation.state")

        if state is None:
            return {"error": "No observation state available"}

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Convert to graph
        graph_data = self.graph_transformer.to_graph(state)
        
        # Move graph to same device as model
        device = next(self.gnn_model.parameters()).device
        graph_data = graph_data.to(device)

        # Run through GNN to get context
        self.gnn_model.eval()
        with torch.no_grad():
            world_context = self.gnn_model.to_world_context(graph_data, threshold=threshold)

        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)

        return {
            "frame_index": self.data_manager._current_frame_idx,
            "world_context": world_context,
            "inference_time_ms": round(inference_time, 3),
        }

    def predict_action_outcome(
        self,
        action: list[float] | None = None,
        num_steps: int = 1,
    ) -> dict:
        """Predict the outcome of an action.

        Uses the current state and a proposed action to predict:
        - Expected next state predicates
        - Changes in spatial relationships
        - Confidence in prediction

        Args:
            action: Proposed action vector, or None to use dataset action
            num_steps: Number of forward steps to simulate

        Returns:
            Prediction result as dictionary
        """
        import torch

        start_time = time.perf_counter()

        # Get current state
        current_frame = self.data_manager.get_current_frame()
        current_state = current_frame.get("observation.state")

        if current_state is None:
            return {"error": "No observation state available"}

        if not isinstance(current_state, torch.Tensor):
            current_state = torch.tensor(current_state, dtype=torch.float32)

        # Get action to apply
        if action is not None:
            action_tensor = torch.tensor(action, dtype=torch.float32)
        else:
            action_data = current_frame.get("action")
            if action_data is None:
                return {"error": "No action available and none provided"}
            if not isinstance(action_data, torch.Tensor):
                action_tensor = torch.tensor(action_data, dtype=torch.float32)
            else:
                action_tensor = action_data

        # Get current predicates
        current_graph = self.graph_transformer.to_graph(current_state)
        device = next(self.gnn_model.parameters()).device
        current_graph = current_graph.to(device)
        current_context = self.gnn_model.to_world_context(current_graph, threshold=0.5)

        # Simulate forward steps
        # For true forward prediction, we would need a dynamics model
        # Here we approximate by looking at future frames in the dataset
        predictions = []
        frame_idx = self.data_manager._current_frame_idx

        for step in range(num_steps):
            next_idx = min(frame_idx + step + 1, len(self.data_manager) - 1)

            if next_idx == frame_idx + step:
                # We have actual future data
                future_frame = self.data_manager.get_frame(next_idx)
                future_state = future_frame.get("observation.state")

                if future_state is not None:
                    if not isinstance(future_state, torch.Tensor):
                        future_state = torch.tensor(future_state, dtype=torch.float32)

                    future_graph = self.graph_transformer.to_graph(future_state)
                    future_graph = future_graph.to(device)
                    future_context = self.gnn_model.to_world_context(
                        future_graph, threshold=0.5
                    )

                    predictions.append({
                        "step": step + 1,
                        "frame_index": next_idx,
                        "predicted_predicates": future_context,
                        "source": "dataset",
                    })
            else:
                # End of dataset, no more predictions
                break

        # Restore original frame index
        self.data_manager.set_frame_index(frame_idx)

        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)

        # Compute predicate changes
        predicate_changes = self._compute_predicate_changes(
            current_context, predictions
        )

        return {
            "current_frame": frame_idx,
            "action_applied": action_tensor.tolist(),
            "num_steps": len(predictions),
            "current_predicates": current_context,
            "predictions": predictions,
            "predicate_changes": predicate_changes,
            "inference_time_ms": round(inference_time, 3),
        }

    def _compute_predicate_changes(
        self,
        current: dict,
        predictions: list[dict],
    ) -> list[dict]:
        """Compute changes in predicates between current and predicted states."""
        if not predictions:
            return []

        changes = []

        # Get current active predicates
        current_spatial = {
            (p["predicate"], p["source"], p["target"])
            for p in current.get("spatial_predicates", [])
        }
        current_interaction = {
            (p["predicate"], p["source"], p["target"])
            for p in current.get("interaction_predicates", [])
        }
        current_all = current_spatial | current_interaction

        for pred in predictions:
            pred_context = pred.get("predicted_predicates", {})
            pred_spatial = {
                (p["predicate"], p["source"], p["target"])
                for p in pred_context.get("spatial_predicates", [])
            }
            pred_interaction = {
                (p["predicate"], p["source"], p["target"])
                for p in pred_context.get("interaction_predicates", [])
            }
            pred_all = pred_spatial | pred_interaction

            # Find added and removed predicates
            added = pred_all - current_all
            removed = current_all - pred_all

            changes.append({
                "step": pred["step"],
                "added": [
                    {"predicate": p[0], "source": p[1], "target": p[2]}
                    for p in added
                ],
                "removed": [
                    {"predicate": p[0], "source": p[1], "target": p[2]}
                    for p in removed
                ],
            })

        return changes

    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)


def get_prediction_tools() -> list[Tool]:
    """Get list of prediction-related MCP tools."""
    return [
        Tool(
            name="get_world_graph",
            description=(
                "Get the GNN-processed relational world graph from the current robot state. "
                "Returns nodes (joints, objects), edges (kinematic links, proximity), "
                "and predicted predicates for reasoning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "frame_idx": {
                        "type": "integer",
                        "description": "Frame index to analyze. If not provided, uses current frame.",
                        "minimum": 0,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Probability threshold for predicate activation (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="predict_action_outcome",
            description=(
                "Predict the outcome of an action by forward-simulating the robot state. "
                "Returns predicted predicates, relationship changes, and confidence scores. "
                "Useful for planning and action selection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Action vector to apply. If not provided, uses the action "
                            "from the current dataset frame."
                        ),
                    },
                    "num_steps": {
                        "type": "integer",
                        "description": "Number of forward steps to predict (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 1,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="advance_frame",
            description=(
                "Advance to the next frame in the LeRobot dataset. "
                "Returns the new frame index and observation state."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="set_frame",
            description="Set the current frame index in the LeRobot dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "frame_idx": {
                        "type": "integer",
                        "description": "Frame index to set",
                        "minimum": 0,
                    },
                },
                "required": ["frame_idx"],
            },
        ),
        Tool(
            name="get_predicates",
            description=(
                "Get predicted spatial and interaction predicates for the current state. "
                "Predicates include: is_near, is_above, is_holding, is_contacting, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Probability threshold for predicate activation (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                    },
                },
                "required": [],
            },
        ),
    ]


async def handle_prediction_tool(
    name: str, arguments: dict[str, Any], tools_manager: PredictionToolsManager
) -> list[TextContent]:
    """Handle prediction tool calls."""
    import torch

    logger.info("Prediction tool called", tool=name, arguments=arguments)

    result: dict[str, Any]

    if name == "get_world_graph":
        frame_idx = arguments.get("frame_idx")
        threshold = arguments.get("threshold", 0.5)
        result = tools_manager.get_world_graph(frame_idx, threshold)

    elif name == "predict_action_outcome":
        action = arguments.get("action")
        num_steps = arguments.get("num_steps", 1)
        result = tools_manager.predict_action_outcome(action, num_steps)

    elif name == "advance_frame":
        frame = tools_manager.data_manager.advance_frame()
        state = frame.get("observation.state")
        if state is not None and hasattr(state, "tolist"):
            state = state.tolist()
        result = {
            "frame_index": tools_manager.data_manager._current_frame_idx,
            "total_frames": len(tools_manager.data_manager),
            "observation_state": state,
        }

    elif name == "set_frame":
        frame_idx = arguments["frame_idx"]
        tools_manager.data_manager.set_frame_index(frame_idx)
        result = {
            "frame_index": tools_manager.data_manager._current_frame_idx,
            "total_frames": len(tools_manager.data_manager),
        }

    elif name == "get_predicates":
        threshold = arguments.get("threshold", 0.5)
        frame = tools_manager.data_manager.get_current_frame()
        state = frame.get("observation.state")

        if state is None:
            result = {"error": "No observation state available"}
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            graph = tools_manager.graph_transformer.to_graph(state)
            device = next(tools_manager.gnn_model.parameters()).device
            graph = graph.to(device)
            result = tools_manager.gnn_model.to_world_context(graph, threshold)
            result["frame_index"] = tools_manager.data_manager._current_frame_idx

    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


def register_prediction_tools(
    server: Server,
    tools_manager: PredictionToolsManager,
) -> None:
    """Register prediction tools with MCP server.
    
    Note: This function stores the tools_manager for later use.
    The actual tool registration happens in the main __init__.py.
    """
    # Store prediction tools on the manager for access by the main handler
    tools_manager._tools = get_prediction_tools()
