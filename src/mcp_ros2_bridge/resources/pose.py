"""Robot pose resource.

Exposes the robot's current position and orientation as an MCP resource.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.types import Resource

if TYPE_CHECKING:
    from mcp_ros2_bridge.ros_node import ROS2Bridge


def get_pose_resources() -> list[Resource]:
    """Get list of pose-related resources."""
    return [
        Resource(
            uri="robot://pose",
            name="Robot Pose",
            description="Current robot position (x, y) and orientation (theta) in the world frame",
            mimeType="application/json",
        ),
        Resource(
            uri="robot://velocity",
            name="Robot Velocity",
            description="Current robot linear and angular velocity",
            mimeType="application/json",
        ),
        Resource(
            uri="robot://status",
            name="Robot Status",
            description="Overall robot status including connection state and movement status",
            mimeType="application/json",
        ),
    ]


async def handle_pose_resource(uri: str, ros_bridge: ROS2Bridge) -> str | None:
    """Handle pose-related resource reads. Returns None if URI not handled."""
    state = ros_bridge.state

    if uri == "robot://pose":
        data = {
            "x": round(state.pose_x, 4),
            "y": round(state.pose_y, 4),
            "theta": round(state.pose_theta, 4),
            "theta_degrees": round(state.pose_theta * 180 / 3.14159, 2),
            "frame": "odom",
        }
    elif uri == "robot://velocity":
        data = {
            "linear_x": round(state.linear_velocity, 4),
            "angular_z": round(state.angular_velocity, 4),
            "is_moving": state.is_moving,
        }
    elif uri == "robot://status":
        data = {
            "connected": ros_bridge.is_connected,
            "is_moving": state.is_moving,
            "has_scan_data": len(state.scan_ranges) > 0,
            "has_image": state.last_image is not None,
            "pose": {
                "x": round(state.pose_x, 4),
                "y": round(state.pose_y, 4),
                "theta": round(state.pose_theta, 4),
            },
        }
    else:
        return None

    return json.dumps(data, indent=2)
