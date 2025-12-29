"""Graph builder for semantic world representation.

Transforms sensor data and detections into a graph structure
suitable for GNN processing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from gnn_reasoner.detector import Detection, DetectionResult

logger = structlog.get_logger()


@dataclass
class GraphNode:
    """Node in the world graph representing an entity."""
    
    id: str
    node_type: str  # "robot", "obstacle", "object", "landmark"
    position: tuple[float, float]  # (x, y) in world frame
    attributes: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge in the world graph representing a relation."""
    
    source: str
    target: str
    relation: str  # "near", "blocking", "visible", "reachable", etc.
    weight: float = 1.0
    attributes: dict = field(default_factory=dict)


@dataclass
class WorldGraph:
    """Complete world graph with nodes and edges."""
    
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type,
                    "position": list(n.position),
                    "attributes": n.attributes,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "weight": round(e.weight, 3),
                    "attributes": e.attributes,
                }
                for e in self.edges
            ],
            "metadata": self.metadata,
        }

    def add_node(self, node: GraphNode) -> None:
        """Add node if not already present."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add edge if not already present."""
        if not any(
            e.source == edge.source and e.target == edge.target and e.relation == edge.relation
            for e in self.edges
        ):
            self.edges.append(edge)


class SemanticGraphBuilder:
    """Builds semantic world graphs from sensor data and detections."""

    def __init__(
        self,
        spatial_threshold: float = 2.0,
        near_threshold: float = 1.0,
        blocking_threshold: float = 0.5,
    ):
        """Initialize graph builder.
        
        Args:
            spatial_threshold: Max distance to consider for graph edges
            near_threshold: Distance threshold for "near" relation
            blocking_threshold: Distance threshold for "blocking" relation
        """
        self.spatial_threshold = spatial_threshold
        self.near_threshold = near_threshold
        self.blocking_threshold = blocking_threshold

    def build_graph(
        self,
        robot_pose: tuple[float, float, float],  # (x, y, theta)
        scan_obstacles: list[dict] | None = None,
        detections: DetectionResult | None = None,
        depth_image: np.ndarray | None = None,
        camera_intrinsics: dict | None = None,
    ) -> WorldGraph:
        """Build complete world graph from available sensor data.
        
        Args:
            robot_pose: Robot position (x, y, theta) in world frame
            scan_obstacles: Clustered obstacles from LiDAR
            detections: Object detections from camera
            depth_image: Depth image for 3D projection
            camera_intrinsics: Camera parameters for projection
            
        Returns:
            WorldGraph with entities and relations
        """
        graph = WorldGraph()
        graph.metadata = {
            "robot_pose": list(robot_pose),
            "has_lidar": scan_obstacles is not None,
            "has_vision": detections is not None,
        }

        # Add robot node
        robot_node = GraphNode(
            id="robot",
            node_type="robot",
            position=(robot_pose[0], robot_pose[1]),
            attributes={"theta": robot_pose[2], "is_ego": True},
        )
        graph.add_node(robot_node)

        # Add obstacle nodes from LiDAR
        if scan_obstacles:
            self._add_lidar_obstacles(graph, scan_obstacles, robot_pose)

        # Add detected object nodes from camera
        if detections and depth_image is not None and camera_intrinsics:
            self._add_detected_objects(
                graph, detections, depth_image, camera_intrinsics, robot_pose
            )

        # Compute inter-entity edges
        self._compute_edges(graph)

        graph.metadata["num_nodes"] = len(graph.nodes)
        graph.metadata["num_edges"] = len(graph.edges)

        return graph

    def _add_lidar_obstacles(
        self,
        graph: WorldGraph,
        obstacles: list[dict],
        robot_pose: tuple[float, float, float],
    ) -> None:
        """Add obstacle nodes from LiDAR clusters."""
        for i, obs in enumerate(obstacles):
            # Obstacles should already be in world coordinates
            world_x = obs.get("world_x", obs.get("x", 0))
            world_y = obs.get("world_y", obs.get("y", 0))

            node = GraphNode(
                id=f"obstacle_{i}",
                node_type="obstacle",
                position=(world_x, world_y),
                attributes={
                    "source": "lidar",
                    "distance": obs.get("distance", 0),
                    "size": obs.get("size", 0.1),
                },
            )
            graph.add_node(node)

    def _add_detected_objects(
        self,
        graph: WorldGraph,
        detections: DetectionResult,
        depth_image: np.ndarray,
        camera_intrinsics: dict,
        robot_pose: tuple[float, float, float],
    ) -> None:
        """Add object nodes from camera detections with depth."""
        fx = camera_intrinsics.get("fx", 500)
        fy = camera_intrinsics.get("fy", 500)
        cx = camera_intrinsics.get("cx", depth_image.shape[1] / 2)
        cy = camera_intrinsics.get("cy", depth_image.shape[0] / 2)

        for i, det in enumerate(detections.detections):
            # Get depth at detection center
            u, v = det.center
            if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                depth = depth_image[v, u]
                if depth <= 0 or np.isnan(depth) or np.isinf(depth):
                    continue

                # Project to 3D camera frame
                x_cam = (u - cx) * depth / fx
                y_cam = (v - cy) * depth / fy
                z_cam = depth

                # Transform to robot frame (assuming camera at robot center, facing forward)
                x_robot = z_cam
                y_robot = -x_cam

                # Transform to world frame
                rx, ry, rtheta = robot_pose
                x_world = rx + x_robot * math.cos(rtheta) - y_robot * math.sin(rtheta)
                y_world = ry + x_robot * math.sin(rtheta) + y_robot * math.cos(rtheta)

                node = GraphNode(
                    id=f"object_{det.label}_{i}",
                    node_type="object",
                    position=(round(x_world, 3), round(y_world, 3)),
                    attributes={
                        "label": det.label,
                        "confidence": det.confidence,
                        "source": "camera",
                        "depth": round(depth, 3),
                    },
                )
                graph.add_node(node)

    def _compute_edges(self, graph: WorldGraph) -> None:
        """Compute edges between all nodes based on spatial relations."""
        for i, n1 in enumerate(graph.nodes):
            for j, n2 in enumerate(graph.nodes):
                if i >= j:
                    continue

                dist = math.sqrt(
                    (n1.position[0] - n2.position[0]) ** 2 +
                    (n1.position[1] - n2.position[1]) ** 2
                )

                if dist > self.spatial_threshold:
                    continue

                relation = self._infer_relation(n1, n2, dist)
                weight = 1.0 / max(dist, 0.1)

                edge = GraphEdge(
                    source=n1.id,
                    target=n2.id,
                    relation=relation,
                    weight=round(weight, 3),
                    attributes={"distance": round(dist, 3)},
                )
                graph.add_edge(edge)

    def _infer_relation(
        self,
        n1: GraphNode,
        n2: GraphNode,
        distance: float,
    ) -> str:
        """Infer semantic relation between two nodes."""
        # Check if one is robot
        is_robot_involved = n1.node_type == "robot" or n2.node_type == "robot"

        if distance < self.blocking_threshold:
            if is_robot_involved:
                return "blocking"
            return "adjacent"
        elif distance < self.near_threshold:
            return "near"
        else:
            return "visible"


def build_from_scan_data(
    robot_pose: tuple[float, float, float],
    scan_ranges: list[float],
    angle_min: float,
    angle_increment: float,
    max_range: float = 3.0,
) -> WorldGraph:
    """Convenience function to build graph from raw scan data.
    
    Args:
        robot_pose: Robot (x, y, theta) in world frame
        scan_ranges: LiDAR range readings
        angle_min: Starting angle of scan
        angle_increment: Angle between readings
        max_range: Maximum range to consider
        
    Returns:
        WorldGraph with robot and detected obstacles
    """
    from mcp_ros2_bridge.resources.world_graph import _cluster_obstacles

    obstacles = _cluster_obstacles(
        scan_ranges,
        angle_min,
        angle_increment,
        robot_pose[0],
        robot_pose[1],
        robot_pose[2],
        max_range=max_range,
    )

    builder = SemanticGraphBuilder()
    return builder.build_graph(
        robot_pose=robot_pose,
        scan_obstacles=obstacles,
    )

