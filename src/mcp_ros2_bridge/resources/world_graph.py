"""World Graph resource.

Exposes the GNN-processed semantic scene graph as an MCP resource.
This is the key integration point between perception and reasoning.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from mcp.types import Resource, TextContent

if TYPE_CHECKING:
    from mcp_ros2_bridge.ros_node import ROS2Bridge


def get_world_graph_resources() -> list[Resource]:
    """Get list of world graph resources."""
    return [
        Resource(
            uri="robot://world_graph",
            name="World Graph",
            description=(
                "Semantic scene graph with entities (nodes) and relations (edges). "
                "Processed by GNN for structured environment understanding."
            ),
            mimeType="application/json",
        ),
        Resource(
            uri="robot://world_graph/entities",
            name="Detected Entities",
            description="List of detected entities (robot, obstacles, landmarks) with attributes",
            mimeType="application/json",
        ),
        Resource(
            uri="robot://world_graph/relations",
            name="Entity Relations",
            description="Spatial and semantic relations between entities",
            mimeType="application/json",
        ),
    ]


async def handle_world_graph_resource(uri: str, ros_bridge: ROS2Bridge) -> list[TextContent] | None:
    """Handle world graph resource reads. Returns None if URI not handled."""
    state = ros_bridge.state

    if uri == "robot://world_graph":
        data = _build_world_graph(state, ros_bridge.is_connected)
    elif uri == "robot://world_graph/entities":
        graph = _build_world_graph(state, ros_bridge.is_connected)
        data = {"entities": graph.get("nodes", []), "count": len(graph.get("nodes", []))}
    elif uri == "robot://world_graph/relations":
        graph = _build_world_graph(state, ros_bridge.is_connected)
        data = {"relations": graph.get("edges", []), "count": len(graph.get("edges", []))}
    else:
        return None

    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def _build_world_graph(state: Any, is_connected: bool) -> dict:
    """Build semantic world graph from current robot state.
    
    In a full implementation, this would call the GNN reasoner service.
    For now, we build a simple graph from scan data.
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    # Add robot node
    robot_node = {
        "id": "robot",
        "type": "robot",
        "position": [round(state.pose_x, 3), round(state.pose_y, 3)],
        "attributes": {
            "theta": round(state.pose_theta, 3),
            "is_moving": state.is_moving,
            "linear_velocity": round(state.linear_velocity, 3),
            "angular_velocity": round(state.angular_velocity, 3),
        },
    }
    nodes.append(robot_node)

    if not state.scan_ranges:
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "source": "mcp_ros2_bridge",
                "gnn_processed": False,
                "connected": is_connected,
            },
        }

    # Detect obstacles from scan and add as nodes
    obstacles = _cluster_obstacles(
        state.scan_ranges,
        state.scan_angle_min,
        state.scan_angle_increment,
        state.pose_x,
        state.pose_y,
        state.pose_theta,
    )

    for i, obs in enumerate(obstacles):
        obs_id = f"obstacle_{i}"
        obs_node = {
            "id": obs_id,
            "type": "obstacle",
            "position": [obs["world_x"], obs["world_y"]],
            "attributes": {
                "size": obs["size"],
                "distance_to_robot": obs["distance"],
                "relative_angle": obs["angle"],
            },
        }
        nodes.append(obs_node)

        # Add edge from robot to obstacle
        relation = _infer_relation(obs["distance"], obs["angle"])
        edge = {
            "source": "robot",
            "target": obs_id,
            "relation": relation,
            "weight": round(1.0 / max(obs["distance"], 0.1), 3),
        }
        edges.append(edge)

    # Add edges between nearby obstacles
    for i, obs1 in enumerate(obstacles):
        for j, obs2 in enumerate(obstacles):
            if i >= j:
                continue
            dist = math.sqrt(
                (obs1["world_x"] - obs2["world_x"]) ** 2 +
                (obs1["world_y"] - obs2["world_y"]) ** 2
            )
            if dist < 1.0:  # Within 1 meter
                edges.append({
                    "source": f"obstacle_{i}",
                    "target": f"obstacle_{j}",
                    "relation": "near",
                    "weight": round(1.0 / max(dist, 0.1), 3),
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "source": "mcp_ros2_bridge",
            "gnn_processed": False,  # Will be True when GNN service is integrated
            "connected": is_connected,
            "num_entities": len(nodes),
            "num_relations": len(edges),
        },
    }


def _cluster_obstacles(
    ranges: list[float],
    angle_min: float,
    angle_increment: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    max_range: float = 3.0,
    cluster_threshold: float = 0.3,
) -> list[dict]:
    """Cluster scan points into obstacles with world coordinates."""
    obstacles = []
    current_cluster: list[tuple[float, float, float, float]] = []  # (local_x, local_y, dist, angle)

    for i, distance in enumerate(ranges):
        if math.isinf(distance) or math.isnan(distance) or distance > max_range:
            if current_cluster:
                obstacles.append(_finalize_obstacle_cluster(
                    current_cluster, robot_x, robot_y, robot_theta
                ))
                current_cluster = []
            continue

        angle = angle_min + i * angle_increment
        local_x = distance * math.cos(angle)
        local_y = distance * math.sin(angle)

        if not current_cluster:
            current_cluster = [(local_x, local_y, distance, angle)]
        else:
            last_x, last_y, _, _ = current_cluster[-1]
            if math.sqrt((local_x - last_x) ** 2 + (local_y - last_y) ** 2) < cluster_threshold:
                current_cluster.append((local_x, local_y, distance, angle))
            else:
                obstacles.append(_finalize_obstacle_cluster(
                    current_cluster, robot_x, robot_y, robot_theta
                ))
                current_cluster = [(local_x, local_y, distance, angle)]

    if current_cluster:
        obstacles.append(_finalize_obstacle_cluster(
            current_cluster, robot_x, robot_y, robot_theta
        ))

    return obstacles


def _finalize_obstacle_cluster(
    cluster: list[tuple[float, float, float, float]],
    robot_x: float,
    robot_y: float,
    robot_theta: float,
) -> dict:
    """Convert cluster to obstacle with world coordinates."""
    local_xs = [p[0] for p in cluster]
    local_ys = [p[1] for p in cluster]
    distances = [p[2] for p in cluster]
    angles = [p[3] for p in cluster]

    # Center in local frame
    center_local_x = sum(local_xs) / len(local_xs)
    center_local_y = sum(local_ys) / len(local_ys)

    # Transform to world frame
    world_x = robot_x + center_local_x * math.cos(robot_theta) - center_local_y * math.sin(robot_theta)
    world_y = robot_y + center_local_x * math.sin(robot_theta) + center_local_y * math.cos(robot_theta)

    return {
        "world_x": round(world_x, 3),
        "world_y": round(world_y, 3),
        "distance": round(min(distances), 3),
        "angle": round(sum(angles) / len(angles), 3),
        "size": round(max(max(local_xs) - min(local_xs), max(local_ys) - min(local_ys), 0.1), 3),
    }


def _infer_relation(distance: float, angle: float) -> str:
    """Infer semantic relation based on distance and angle."""
    # Angle in front arc
    is_front = -math.pi / 4 <= angle <= math.pi / 4

    if distance < 0.5:
        return "blocking" if is_front else "adjacent"
    elif distance < 1.0:
        return "near"
    else:
        return "visible"
