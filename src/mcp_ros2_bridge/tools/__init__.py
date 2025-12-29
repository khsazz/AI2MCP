"""MCP Tools for robot control.

Tools represent actions the AI can take to interact with the physical world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_ros2_bridge.tools.motion import register_motion_tools
from mcp_ros2_bridge.tools.perception import register_perception_tools
from mcp_ros2_bridge.tools.prediction import (
    PredictionToolsManager,
    register_prediction_tools,
)

if TYPE_CHECKING:
    from mcp.server import Server
    from mcp_ros2_bridge.ros_node import ROS2Bridge


def register_tools(server: Server, ros_bridge: ROS2Bridge) -> None:
    """Register all MCP tools with the server."""
    register_motion_tools(server, ros_bridge)
    register_perception_tools(server, ros_bridge)


__all__ = [
    "register_tools",
    "register_prediction_tools",
    "PredictionToolsManager",
]

