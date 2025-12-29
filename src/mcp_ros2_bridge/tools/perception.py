"""Perception tools for querying robot sensors.

These tools allow the AI to request sensor data and processed
perception information from the robot.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import structlog
from mcp.server import Server
from mcp.types import Tool, TextContent

if TYPE_CHECKING:
    from mcp_ros2_bridge.ros_node import ROS2Bridge

logger = structlog.get_logger()


def register_perception_tools(server: Server, ros_bridge: ROS2Bridge) -> None:
    """Register perception-related MCP tools."""

    # Store existing list_tools handler and extend it
    existing_list_tools = server._tool_handlers.get("list_tools")

    @server.list_tools()
    async def list_perception_tools() -> list[Tool]:
        """List available perception tools."""
        motion_tools = await existing_list_tools() if existing_list_tools else []
        perception_tools = [
            Tool(
                name="get_obstacle_distances",
                description=(
                    "Get distances to obstacles in specified directions. "
                    "Returns distances from LiDAR scan at requested angles."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directions": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["front", "left", "right", "back", "front_left", "front_right"],
                            },
                            "description": "Directions to check for obstacles",
                            "default": ["front", "left", "right"],
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="check_path_clear",
                description=(
                    "Check if the path ahead is clear for a specified distance. "
                    "Returns whether robot can safely move forward."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "distance_meters": {
                            "type": "number",
                            "description": "Distance to check ahead (meters)",
                            "minimum": 0.1,
                            "maximum": 10.0,
                            "default": 1.0,
                        },
                        "width_meters": {
                            "type": "number",
                            "description": "Width of path to check (meters)",
                            "minimum": 0.2,
                            "maximum": 2.0,
                            "default": 0.5,
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="scan_surroundings",
                description=(
                    "Perform a 360-degree scan and return obstacle summary. "
                    "Identifies clear sectors and closest obstacles."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "num_sectors": {
                            "type": "integer",
                            "description": "Number of sectors to divide the scan into",
                            "minimum": 4,
                            "maximum": 36,
                            "default": 8,
                        },
                    },
                    "required": [],
                },
            ),
        ]
        return motion_tools + perception_tools

    @server.call_tool()
    async def call_perception_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle perception tool calls."""
        logger.info("Perception tool called", tool=name, arguments=arguments)

        state = ros_bridge.state
        result: dict[str, Any]

        if name == "get_obstacle_distances":
            directions = arguments.get("directions", ["front", "left", "right"])
            result = _get_obstacle_distances(state.scan_ranges, directions, 
                                             state.scan_angle_min, state.scan_angle_increment)
        
        elif name == "check_path_clear":
            distance = arguments.get("distance_meters", 1.0)
            width = arguments.get("width_meters", 0.5)
            result = _check_path_clear(state.scan_ranges, distance, width,
                                       state.scan_angle_min, state.scan_angle_increment)
        
        elif name == "scan_surroundings":
            num_sectors = arguments.get("num_sectors", 8)
            result = _scan_surroundings(state.scan_ranges, num_sectors,
                                        state.scan_angle_min, state.scan_angle_max,
                                        state.scan_angle_increment)
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]


def _get_obstacle_distances(
    ranges: list[float],
    directions: list[str],
    angle_min: float,
    angle_increment: float,
) -> dict[str, Any]:
    """Calculate distances to obstacles in specified directions."""
    if not ranges:
        return {"success": False, "error": "No scan data available"}

    direction_angles = {
        "front": 0.0,
        "left": math.pi / 2,
        "right": -math.pi / 2,
        "back": math.pi,
        "front_left": math.pi / 4,
        "front_right": -math.pi / 4,
    }

    distances = {}
    for direction in directions:
        if direction not in direction_angles:
            continue
        
        target_angle = direction_angles[direction]
        index = int((target_angle - angle_min) / angle_increment)
        
        if 0 <= index < len(ranges):
            distance = ranges[index]
            # Handle inf/nan
            if math.isinf(distance) or math.isnan(distance):
                distance = float("inf")
            distances[direction] = round(distance, 3)
        else:
            distances[direction] = None

    return {
        "success": True,
        "distances": distances,
        "unit": "meters",
    }


def _check_path_clear(
    ranges: list[float],
    distance: float,
    width: float,
    angle_min: float,
    angle_increment: float,
) -> dict[str, Any]:
    """Check if path ahead is clear for specified distance and width."""
    if not ranges:
        return {"success": False, "error": "No scan data available"}

    # Calculate angular span for the width at given distance
    half_angle = math.atan2(width / 2, distance)
    
    # Find indices for the angular span around front (0 degrees)
    start_angle = -half_angle
    end_angle = half_angle
    
    start_idx = int((start_angle - angle_min) / angle_increment)
    end_idx = int((end_angle - angle_min) / angle_increment)
    
    # Clamp indices
    start_idx = max(0, min(start_idx, len(ranges) - 1))
    end_idx = max(0, min(end_idx, len(ranges) - 1))
    
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Check all ranges in the span
    min_distance = float("inf")
    for i in range(start_idx, end_idx + 1):
        r = ranges[i]
        if not math.isinf(r) and not math.isnan(r):
            min_distance = min(min_distance, r)

    is_clear = min_distance >= distance

    return {
        "success": True,
        "is_clear": is_clear,
        "min_obstacle_distance": round(min_distance, 3) if not math.isinf(min_distance) else None,
        "requested_distance": distance,
        "path_width": width,
    }


def _scan_surroundings(
    ranges: list[float],
    num_sectors: int,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
) -> dict[str, Any]:
    """Perform 360-degree scan and summarize obstacles by sector."""
    if not ranges:
        return {"success": False, "error": "No scan data available"}

    total_angle = angle_max - angle_min
    sector_angle = total_angle / num_sectors
    
    sectors = []
    for i in range(num_sectors):
        sector_start = angle_min + i * sector_angle
        sector_end = sector_start + sector_angle
        sector_center = (sector_start + sector_end) / 2
        
        # Get indices for this sector
        start_idx = int((sector_start - angle_min) / angle_increment)
        end_idx = int((sector_end - angle_min) / angle_increment)
        
        start_idx = max(0, min(start_idx, len(ranges) - 1))
        end_idx = max(0, min(end_idx, len(ranges) - 1))
        
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Calculate sector statistics
        sector_ranges = [r for r in ranges[start_idx:end_idx + 1] 
                        if not math.isinf(r) and not math.isnan(r)]
        
        if sector_ranges:
            min_dist = min(sector_ranges)
            avg_dist = sum(sector_ranges) / len(sector_ranges)
        else:
            min_dist = float("inf")
            avg_dist = float("inf")

        # Convert center angle to degrees and direction name
        center_deg = math.degrees(sector_center)
        direction = _angle_to_direction(center_deg)

        sectors.append({
            "sector": i,
            "direction": direction,
            "angle_degrees": round(center_deg, 1),
            "min_distance": round(min_dist, 3) if not math.isinf(min_dist) else None,
            "avg_distance": round(avg_dist, 3) if not math.isinf(avg_dist) else None,
            "is_clear": min_dist > 1.0,  # Threshold for "clear"
        })

    # Find best direction (clearest path)
    clear_sectors = [s for s in sectors if s["is_clear"]]
    best_direction = max(sectors, key=lambda s: s["min_distance"] or 0)

    return {
        "success": True,
        "num_sectors": num_sectors,
        "sectors": sectors,
        "clear_sectors": len(clear_sectors),
        "best_direction": best_direction["direction"],
        "best_direction_distance": best_direction["min_distance"],
    }


def _angle_to_direction(angle_degrees: float) -> str:
    """Convert angle in degrees to cardinal direction."""
    # Normalize to -180 to 180
    while angle_degrees > 180:
        angle_degrees -= 360
    while angle_degrees < -180:
        angle_degrees += 360

    if -22.5 <= angle_degrees < 22.5:
        return "front"
    elif 22.5 <= angle_degrees < 67.5:
        return "front-left"
    elif 67.5 <= angle_degrees < 112.5:
        return "left"
    elif 112.5 <= angle_degrees < 157.5:
        return "back-left"
    elif angle_degrees >= 157.5 or angle_degrees < -157.5:
        return "back"
    elif -157.5 <= angle_degrees < -112.5:
        return "back-right"
    elif -112.5 <= angle_degrees < -67.5:
        return "right"
    else:
        return "front-right"

