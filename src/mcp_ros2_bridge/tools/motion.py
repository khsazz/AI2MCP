"""Motion control tools for robot movement.

These tools allow the AI to control the robot's movement through
velocity commands and navigation actions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog
from mcp.types import Tool, TextContent

if TYPE_CHECKING:
    from mcp_ros2_bridge.ros_node import ROS2Bridge

logger = structlog.get_logger()


def get_motion_tools() -> list[Tool]:
    """Get list of motion-related MCP tools."""
    return [
        Tool(
            name="move",
            description=(
                "Move the robot with specified linear and angular velocity for a duration. "
                "Linear velocity is in m/s (positive=forward, negative=backward). "
                "Angular velocity is in rad/s (positive=left, negative=right)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "linear_x": {
                        "type": "number",
                        "description": "Linear velocity in m/s (-1.0 to 1.0)",
                        "minimum": -1.0,
                        "maximum": 1.0,
                    },
                    "angular_z": {
                        "type": "number",
                        "description": "Angular velocity in rad/s (-2.0 to 2.0)",
                        "minimum": -2.0,
                        "maximum": 2.0,
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration in milliseconds (100-10000)",
                        "minimum": 100,
                        "maximum": 10000,
                    },
                },
                "required": ["linear_x", "angular_z", "duration_ms"],
            },
        ),
        Tool(
            name="set_velocity",
            description=(
                "Set continuous velocity. Robot will keep moving until stop() is called. "
                "Use with caution - always follow with stop() when done."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "linear_x": {
                        "type": "number",
                        "description": "Linear velocity in m/s",
                        "minimum": -1.0,
                        "maximum": 1.0,
                    },
                    "angular_z": {
                        "type": "number",
                        "description": "Angular velocity in rad/s",
                        "minimum": -2.0,
                        "maximum": 2.0,
                    },
                },
                "required": ["linear_x", "angular_z"],
            },
        ),
        Tool(
            name="stop",
            description="Emergency stop. Immediately halts all robot movement.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="rotate",
            description="Rotate the robot by a specified angle.",
            inputSchema={
                "type": "object",
                "properties": {
                    "angle_degrees": {
                        "type": "number",
                        "description": "Rotation angle in degrees (positive=left, negative=right)",
                        "minimum": -360,
                        "maximum": 360,
                    },
                    "speed": {
                        "type": "number",
                        "description": "Rotation speed in rad/s (0.1-2.0)",
                        "minimum": 0.1,
                        "maximum": 2.0,
                        "default": 0.5,
                    },
                },
                "required": ["angle_degrees"],
            },
        ),
        Tool(
            name="move_forward",
            description="Move the robot forward by a specified distance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "distance_meters": {
                        "type": "number",
                        "description": "Distance to move in meters",
                        "minimum": 0.1,
                        "maximum": 10.0,
                    },
                    "speed": {
                        "type": "number",
                        "description": "Movement speed in m/s (0.1-1.0)",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.3,
                    },
                },
                "required": ["distance_meters"],
            },
        ),
    ]


async def handle_motion_tool(
    name: str, arguments: dict[str, Any], ros_bridge: ROS2Bridge
) -> list[TextContent]:
    """Handle motion tool calls."""
    logger.info("Motion tool called", tool=name, arguments=arguments)

    result: dict[str, Any]

    if name == "move":
        result = await ros_bridge.move(
            linear_x=arguments["linear_x"],
            angular_z=arguments["angular_z"],
            duration_ms=arguments["duration_ms"],
        )
    elif name == "set_velocity":
        result = await ros_bridge.set_velocity(
            linear_x=arguments["linear_x"],
            angular_z=arguments["angular_z"],
        )
    elif name == "stop":
        result = await ros_bridge.stop()
    elif name == "rotate":
        angle_rad = arguments["angle_degrees"] * 3.14159 / 180.0
        speed = arguments.get("speed", 0.5)
        duration_ms = int(abs(angle_rad / speed) * 1000)
        angular_z = speed if angle_rad > 0 else -speed
        result = await ros_bridge.move(
            linear_x=0.0,
            angular_z=angular_z,
            duration_ms=duration_ms,
        )
    elif name == "move_forward":
        distance = arguments["distance_meters"]
        speed = arguments.get("speed", 0.3)
        duration_ms = int((distance / speed) * 1000)
        result = await ros_bridge.move(
            linear_x=speed,
            angular_z=0.0,
            duration_ms=duration_ms,
        )
    else:
        result = {"success": False, "error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]
