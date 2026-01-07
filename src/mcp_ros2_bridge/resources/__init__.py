"""MCP Resources for robot state.

Resources represent readable state that the AI can query to understand
the robot's current situation and environment.

IMPORTANT: The @server.read_resource() decorator expects:
    Return type: list[TextResourceContents]
    
The previous bug was returning plain `str` which caused:
    TypeError: 'NoneType' object is not callable
in the Starlette SSE transport layer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import Resource

from mcp_ros2_bridge.resources.pose import get_pose_resources, handle_pose_resource
from mcp_ros2_bridge.resources.scan import get_scan_resources, handle_scan_resource
from mcp_ros2_bridge.resources.world_graph import get_world_graph_resources, handle_world_graph_resource
from mcp_ros2_bridge.resources.lerobot_state import LeRobotResourceManager

if TYPE_CHECKING:
    from mcp_ros2_bridge.ros_node import ROS2Bridge


def get_lerobot_resources() -> list[Resource]:
    """Get LeRobot-specific resources."""
    return [
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


async def handle_lerobot_resource(
    uri: str, resource_manager: LeRobotResourceManager
) -> str | None:
    """Handle LeRobot resource reads. Returns None if URI not handled."""
    if uri == "robot://lerobot/current_state":
        data = resource_manager.get_current_state()
    elif uri == "robot://lerobot/world_graph":
        data = resource_manager.get_world_graph()
    elif uri == "robot://lerobot/predicates":
        data = resource_manager.get_predicates()
    elif uri == "robot://lerobot/dataset_info":
        data = {
            "repo_id": resource_manager.data_manager.repo_id,
            "total_frames": len(resource_manager.data_manager),
            "current_frame": resource_manager.data_manager._current_frame_idx,
            "predicate_threshold": resource_manager._predicate_threshold,
        }
    else:
        return None

    return json.dumps(data, indent=2)


def register_resources(
    server: Server,
    ros_bridge: ROS2Bridge,
    lerobot_resource_manager: LeRobotResourceManager | None = None,
) -> None:
    """Register all MCP resources with the server.
    
    Args:
        server: MCP Server instance
        ros_bridge: ROS2Bridge for robot state
        lerobot_resource_manager: Optional LeRobotResourceManager for LeRobot integration
    """
    # Collect all resources from each module
    pose_resources = get_pose_resources()
    scan_resources = get_scan_resources()
    world_graph_resources = get_world_graph_resources()
    all_resources = pose_resources + scan_resources + world_graph_resources

    # Add LeRobot resources if enabled
    # NOTE: Resource.uri is AnyUrl type, but handlers receive str.
    # Convert to strings for comparison.
    lerobot_uris: set[str] = set()
    if lerobot_resource_manager is not None:
        lerobot_resources = get_lerobot_resources()
        all_resources = all_resources + lerobot_resources
        lerobot_uris = {str(r.uri) for r in lerobot_resources}

    @server.list_resources()
    async def list_all_resources() -> list[Resource]:
        """List all available resources."""
        return all_resources

    @server.read_resource()
    async def read_resource(uri) -> list[ReadResourceContents]:
        """Route resource reads to appropriate handler.
        
        Returns list[ReadResourceContents] - the correct SDK helper type.
        
        NOTE: The MCP SDK passes AnyUrl (not str) and expects ReadResourceContents
        from mcp.server.lowlevel.helper_types (with .content and .mime_type),
        NOT TextResourceContents from mcp.types (with .text and .mimeType).
        """
        # Convert AnyUrl to string for comparison
        uri_str = str(uri)
        json_text: str | None = None
        
        # Try LeRobot handler first if enabled
        if lerobot_resource_manager is not None and uri_str in lerobot_uris:
            json_text = await handle_lerobot_resource(uri_str, lerobot_resource_manager)

        # Try each core handler
        if json_text is None:
            json_text = await handle_pose_resource(uri_str, ros_bridge)
        
        if json_text is None:
            json_text = await handle_scan_resource(uri_str, ros_bridge)

        if json_text is None:
            json_text = await handle_world_graph_resource(uri_str, ros_bridge)

        # Default error response
        if json_text is None:
            json_text = json.dumps({"error": f"Unknown resource: {uri_str}"})

        # Return proper MCP SDK helper type
        return [
            ReadResourceContents(
                content=json_text,
                mime_type="application/json",
            )
        ]


def register_lerobot_resources(
    server: Server,
    resource_manager: LeRobotResourceManager,
) -> None:
    """Deprecated: Use register_resources with lerobot_resource_manager parameter instead."""
    pass  # No-op, kept for backwards compatibility


__all__ = [
    "register_resources",
    "register_lerobot_resources",
    "LeRobotResourceManager",
]
