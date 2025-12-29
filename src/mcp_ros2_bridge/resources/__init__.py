"""MCP Resources for robot state.

Resources represent readable state that the AI can query to understand
the robot's current situation and environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_ros2_bridge.resources.pose import register_pose_resource
from mcp_ros2_bridge.resources.scan import register_scan_resource
from mcp_ros2_bridge.resources.world_graph import register_world_graph_resource
from mcp_ros2_bridge.resources.lerobot_state import (
    LeRobotResourceManager,
    register_lerobot_resources,
)

if TYPE_CHECKING:
    from mcp.server import Server
    from mcp_ros2_bridge.ros_node import ROS2Bridge


def register_resources(server: Server, ros_bridge: ROS2Bridge) -> None:
    """Register all MCP resources with the server."""
    register_pose_resource(server, ros_bridge)
    register_scan_resource(server, ros_bridge)
    register_world_graph_resource(server, ros_bridge)


__all__ = [
    "register_resources",
    "register_lerobot_resources",
    "LeRobotResourceManager",
]

