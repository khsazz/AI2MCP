"""LiDAR scan resource.

Exposes processed laser scan data as an MCP resource.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from mcp.types import Resource

if TYPE_CHECKING:
    from mcp_ros2_bridge.ros_node import ROS2Bridge


def get_scan_resources() -> list[Resource]:
    """Get list of scan-related resources."""
    return [
        Resource(
            uri="robot://scan/summary",
            name="Scan Summary",
            description="Summarized LiDAR scan with min/max/avg distances per quadrant",
            mimeType="application/json",
        ),
        Resource(
            uri="robot://scan/obstacles",
            name="Detected Obstacles",
            description="List of detected obstacles with positions relative to robot",
            mimeType="application/json",
        ),
        Resource(
            uri="robot://scan/raw",
            name="Raw Scan Data",
            description="Raw LiDAR range data (may be large)",
            mimeType="application/json",
        ),
    ]


async def handle_scan_resource(uri: str, ros_bridge: ROS2Bridge) -> str | None:
    """Handle scan-related resource reads. Returns None if URI not handled."""
    state = ros_bridge.state

    if uri == "robot://scan/summary":
        data = _create_scan_summary(
            state.scan_ranges,
            state.scan_angle_min,
            state.scan_angle_max,
            state.scan_angle_increment,
        )
    elif uri == "robot://scan/obstacles":
        data = _detect_obstacles(
            state.scan_ranges,
            state.scan_angle_min,
            state.scan_angle_increment,
        )
    elif uri == "robot://scan/raw":
        # Limit raw data size
        ranges = state.scan_ranges[:360] if len(state.scan_ranges) > 360 else state.scan_ranges
        data = {
            "ranges": [round(r, 3) if not math.isinf(r) else None for r in ranges],
            "angle_min": state.scan_angle_min,
            "angle_max": state.scan_angle_max,
            "angle_increment": state.scan_angle_increment,
            "num_readings": len(state.scan_ranges),
        }
    else:
        return None

    return json.dumps(data, indent=2)


def _create_scan_summary(
    ranges: list[float],
    angle_min: float,
    angle_max: float,
    angle_increment: float,
) -> dict:
    """Create summarized scan data by quadrant."""
    if not ranges:
        return {"error": "No scan data available", "quadrants": []}

    quadrants = [
        {"name": "front", "start": -45, "end": 45},
        {"name": "left", "start": 45, "end": 135},
        {"name": "back", "start": 135, "end": -135},
        {"name": "right", "start": -135, "end": -45},
    ]

    results = []
    for q in quadrants:
        start_rad = q["start"] * math.pi / 180
        end_rad = q["end"] * math.pi / 180

        # Handle wrap-around for back quadrant
        if q["name"] == "back":
            indices = []
            for i, angle in enumerate(_get_angles(angle_min, angle_increment, len(ranges))):
                if angle >= start_rad or angle <= end_rad:
                    indices.append(i)
        else:
            indices = [
                i for i, angle in enumerate(_get_angles(angle_min, angle_increment, len(ranges)))
                if start_rad <= angle <= end_rad
            ]

        if indices:
            quadrant_ranges = [
                ranges[i] for i in indices
                if not math.isinf(ranges[i]) and not math.isnan(ranges[i])
            ]
            if quadrant_ranges:
                results.append({
                    "quadrant": q["name"],
                    "min_distance": round(min(quadrant_ranges), 3),
                    "max_distance": round(max(quadrant_ranges), 3),
                    "avg_distance": round(sum(quadrant_ranges) / len(quadrant_ranges), 3),
                    "num_readings": len(quadrant_ranges),
                })
            else:
                results.append({
                    "quadrant": q["name"],
                    "min_distance": None,
                    "max_distance": None,
                    "avg_distance": None,
                    "num_readings": 0,
                })

    # Overall closest obstacle
    valid_ranges = [r for r in ranges if not math.isinf(r) and not math.isnan(r)]
    closest = min(valid_ranges) if valid_ranges else None

    return {
        "quadrants": results,
        "closest_obstacle": round(closest, 3) if closest else None,
        "total_readings": len(ranges),
    }


def _detect_obstacles(
    ranges: list[float],
    angle_min: float,
    angle_increment: float,
    distance_threshold: float = 2.0,
    cluster_threshold: float = 0.3,
) -> dict:
    """Detect and cluster obstacles from scan data."""
    if not ranges:
        return {"error": "No scan data available", "obstacles": []}

    obstacles = []
    current_cluster: list[tuple[float, float, float]] = []

    for i, distance in enumerate(ranges):
        if math.isinf(distance) or math.isnan(distance) or distance > distance_threshold:
            # End current cluster if exists
            if current_cluster:
                obstacles.append(_finalize_cluster(current_cluster))
                current_cluster = []
            continue

        angle = angle_min + i * angle_increment
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)

        if not current_cluster:
            current_cluster = [(x, y, distance)]
        else:
            # Check if point belongs to current cluster
            last_x, last_y, _ = current_cluster[-1]
            if math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2) < cluster_threshold:
                current_cluster.append((x, y, distance))
            else:
                obstacles.append(_finalize_cluster(current_cluster))
                current_cluster = [(x, y, distance)]

    # Finalize last cluster
    if current_cluster:
        obstacles.append(_finalize_cluster(current_cluster))

    return {
        "obstacles": obstacles,
        "count": len(obstacles),
        "detection_range": distance_threshold,
    }


def _finalize_cluster(cluster: list[tuple[float, float, float]]) -> dict:
    """Convert cluster points to obstacle representation."""
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    distances = [p[2] for p in cluster]

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    min_dist = min(distances)

    # Estimate size from cluster spread
    width = max(xs) - min(xs)
    depth = max(ys) - min(ys)

    return {
        "position": {"x": round(center_x, 3), "y": round(center_y, 3)},
        "distance": round(min_dist, 3),
        "size": {"width": round(max(width, 0.1), 3), "depth": round(max(depth, 0.1), 3)},
        "num_points": len(cluster),
    }


def _get_angles(angle_min: float, angle_increment: float, count: int) -> list[float]:
    """Generate list of angles for scan readings."""
    return [angle_min + i * angle_increment for i in range(count)]
