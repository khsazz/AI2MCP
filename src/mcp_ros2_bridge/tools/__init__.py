"""MCP Tools for robot control.

Tools represent actions the AI can take to interact with the physical world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from mcp_ros2_bridge.tools.motion import get_motion_tools, handle_motion_tool
from mcp_ros2_bridge.tools.perception import get_perception_tools, handle_perception_tool
from mcp_ros2_bridge.tools.prediction import (
    PredictionToolsManager,
    get_prediction_tools,
    handle_prediction_tool,
    register_prediction_tools,
)

if TYPE_CHECKING:
    from mcp_ros2_bridge.ros_node import ROS2Bridge


def register_tools(
    server: Server,
    ros_bridge: ROS2Bridge,
    prediction_tools_manager: PredictionToolsManager | None = None,
) -> None:
    """Register all MCP tools with the server.
    
    Args:
        server: MCP Server instance
        ros_bridge: ROS2Bridge for robot control
        prediction_tools_manager: Optional PredictionToolsManager for LeRobot integration
    """
    # Collect all tools from each module
    motion_tools = get_motion_tools()
    perception_tools = get_perception_tools()
    all_tools = motion_tools + perception_tools
    
    # Add prediction tools if LeRobot is enabled
    prediction_tool_names = set()
    if prediction_tools_manager is not None:
        prediction_tools = get_prediction_tools()
        all_tools = all_tools + prediction_tools
        prediction_tool_names = {t.name for t in prediction_tools}

    @server.list_tools()
    async def list_all_tools() -> list[Tool]:
        """List all available tools."""
        return all_tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
        """Route tool calls to appropriate handler."""
        arguments = arguments or {}
        
        # Motion tools
        motion_tool_names = {t.name for t in motion_tools}
        if name in motion_tool_names:
            return await handle_motion_tool(name, arguments, ros_bridge)
        
        # Perception tools
        perception_tool_names_set = {t.name for t in perception_tools}
        if name in perception_tool_names_set:
            return await handle_perception_tool(name, arguments, ros_bridge)
        
        # Prediction tools (if available)
        if prediction_tools_manager is not None and name in prediction_tool_names:
            return await handle_prediction_tool(name, arguments, prediction_tools_manager)
        
        # Unknown tool
        return [TextContent(type="text", text=f'{{"error": "Unknown tool: {name}"}}')]


__all__ = [
    "register_tools",
    "register_prediction_tools",
    "PredictionToolsManager",
    "get_prediction_tools",
    "handle_prediction_tool",
]
