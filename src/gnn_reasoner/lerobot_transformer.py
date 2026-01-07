"""LeRobot Graph Transformer.

Converts LeRobot observation states into torch_geometric graph structures
suitable for GNN-based relational reasoning.

Supports both kinematic-only graphs and hybrid graphs with detected objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    from gnn_reasoner.camera import CameraIntrinsics, Object3D
    from gnn_reasoner.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class JointConfig:
    """Configuration for a single robotic joint."""

    name: str
    parent_idx: int | None  # None for root joint
    position_indices: tuple[int, int, int] | None  # Indices into state for x, y, z position


# Default kinematic chain for ALOHA-style bimanual robots
# (left_arm: 0-6, right_arm: 7-13, left_gripper: 14, right_gripper: 15)
ALOHA_KINEMATIC_CHAIN: list[tuple[int, int]] = [
    # Left arm chain
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 14),  # left gripper
    # Right arm chain
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 15),  # right gripper
]


class LeRobotGraphTransformer:
    """Transforms LeRobot observation states into relational graphs.

    Converts flat state vectors (joint angles, velocities) into
    torch_geometric.data.Data objects with:
    - Node features: joint state information
    - Edge indices: kinematic linkages + spatial proximity edges
    - Edge attributes: distance and link type

    Example:
        >>> transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        >>> state = torch.randn(14)  # 14-DOF robot
        >>> graph = transformer.to_graph(state)
        >>> graph.x.shape  # (14, node_feature_dim)
    """

    def __init__(
        self,
        kinematic_chain: Sequence[tuple[int, int]],
        proximity_threshold: float = 0.3,
        include_velocity: bool = True,
        num_joints: int | None = None,
    ) -> None:
        """Initialize the graph transformer.

        Args:
            kinematic_chain: List of (parent_idx, child_idx) tuples defining
                the robot's kinematic structure
            proximity_threshold: Distance threshold for adding spatial proximity edges
            include_velocity: If True, expect state to include velocities
            num_joints: Number of joints. If None, inferred from kinematic_chain
        """
        self.kinematic_chain = list(kinematic_chain)
        self.proximity_threshold = proximity_threshold
        self.include_velocity = include_velocity

        # Infer number of joints from kinematic chain
        if num_joints is not None:
            self.num_joints = num_joints
        else:
            all_indices = set()
            for parent, child in kinematic_chain:
                all_indices.add(parent)
                all_indices.add(child)
            self.num_joints = max(all_indices) + 1 if all_indices else 0

        # Pre-compute kinematic edge index tensor
        self._kinematic_edge_index = self._build_kinematic_edges()

    def _build_kinematic_edges(self) -> Tensor:
        """Build edge index tensor for kinematic links (bidirectional)."""
        if not self.kinematic_chain:
            return torch.empty((2, 0), dtype=torch.long)

        edges = []
        for parent, child in self.kinematic_chain:
            edges.append([parent, child])
            edges.append([child, parent])  # Bidirectional

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _compute_joint_positions(self, state: Tensor) -> Tensor:
        """Estimate joint positions from state vector.

        For simple cases, we use joint angles directly as a proxy for
        relative positions. For more accurate FK, override this method.

        Args:
            state: State tensor of shape (state_dim,)

        Returns:
            Positions tensor of shape (num_joints, 3)
        """
        # Simple heuristic: use cumulative joint angles to approximate positions
        # This is a placeholder - real implementation would use forward kinematics
        num_joints = min(self.num_joints, state.shape[0])

        # Create pseudo-positions based on joint angles
        positions = torch.zeros(self.num_joints, 3)

        # Simple chain approximation: each joint adds displacement based on angle
        link_length = 0.1  # Approximate link length
        cumulative_angle = 0.0

        for i in range(num_joints):
            if i < state.shape[0]:
                cumulative_angle += state[i].item()
            positions[i, 0] = i * link_length * torch.cos(torch.tensor(cumulative_angle))
            positions[i, 1] = i * link_length * torch.sin(torch.tensor(cumulative_angle))
            positions[i, 2] = 0.0  # 2D approximation

        return positions

    def _compute_proximity_edges(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Compute edges based on spatial proximity.

        Args:
            positions: Joint positions of shape (num_joints, 3)

        Returns:
            Tuple of (edge_index, edge_distances)
        """
        num_nodes = positions.shape[0]

        # Compute pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        distances = torch.norm(diff, dim=-1)  # (N, N)

        # Find pairs within threshold (excluding self-loops and kinematic edges)
        mask = (distances < self.proximity_threshold) & (distances > 0)

        # Remove kinematic edges from proximity consideration
        kinematic_set = set()
        for parent, child in self.kinematic_chain:
            kinematic_set.add((parent, child))
            kinematic_set.add((child, parent))

        edge_list = []
        distance_list = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if mask[i, j] and (i, j) not in kinematic_set:
                    edge_list.append([i, j])
                    distance_list.append(distances[i, j].item())

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_distances = torch.tensor(distance_list, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_distances = torch.empty(0, dtype=torch.float)

        return edge_index, edge_distances

    def to_graph(
        self,
        observation_state: Tensor,
        object_positions: Tensor | None = None,
        object_labels: list[str] | None = None,
    ) -> Data:
        """Convert observation state to a relational graph.

        Args:
            observation_state: Robot state tensor of shape (state_dim,)
                Expected format: [joint_angles] or [joint_angles, joint_velocities]
            object_positions: Optional object positions of shape (num_objects, 3)
            object_labels: Optional labels for detected objects

        Returns:
            torch_geometric.data.Data with:
                - x: Node features (num_nodes, feature_dim)
                - edge_index: Edge connectivity (2, num_edges)
                - edge_attr: Edge features (num_edges, edge_feature_dim)
                - node_types: 0=joint, 1=end_effector, 2=object
        """
        state = observation_state.float()
        if state.dim() == 0:
            state = state.unsqueeze(0)

        # Determine state partitioning
        state_dim = state.shape[0]
        if self.include_velocity and state_dim >= 2 * self.num_joints:
            joint_angles = state[: self.num_joints]
            joint_velocities = state[self.num_joints : 2 * self.num_joints]
        else:
            joint_angles = state[: min(self.num_joints, state_dim)]
            joint_velocities = torch.zeros(self.num_joints)

        # Compute joint positions (placeholder FK)
        positions = self._compute_joint_positions(joint_angles)

        # Build node features: [angle, velocity, pos_x, pos_y, pos_z]
        node_features = torch.zeros(self.num_joints, 5)
        num_angles = min(len(joint_angles), self.num_joints)
        num_vels = min(len(joint_velocities), self.num_joints)

        node_features[:num_angles, 0] = joint_angles[:num_angles]
        node_features[:num_vels, 1] = joint_velocities[:num_vels]
        node_features[:, 2:5] = positions

        # Node types: 0=regular joint, 1=end-effector (gripper)
        node_types = torch.zeros(self.num_joints, dtype=torch.long)
        # Mark end-effectors (nodes with no children in kinematic chain)
        children = {child for _, child in self.kinematic_chain}
        parents = {parent for parent, _ in self.kinematic_chain}
        end_effectors = children - parents
        for ee_idx in end_effectors:
            if ee_idx < self.num_joints:
                node_types[ee_idx] = 1

        # Add object nodes if provided
        if object_positions is not None and object_positions.numel() > 0:
            num_objects = object_positions.shape[0]
            object_features = torch.zeros(num_objects, 5)
            object_features[:, 2:5] = object_positions[:, :3]  # Only positions

            node_features = torch.cat([node_features, object_features], dim=0)
            object_types = torch.full((num_objects,), 2, dtype=torch.long)
            node_types = torch.cat([node_types, object_types])
            positions = torch.cat([positions, object_positions[:, :3]], dim=0)

        total_nodes = node_features.shape[0]

        # Build edges: kinematic + proximity
        kinematic_edges = self._kinematic_edge_index

        # Compute proximity edges
        proximity_edges, proximity_distances = self._compute_proximity_edges(positions)

        # Combine edges
        if proximity_edges.numel() > 0:
            edge_index = torch.cat([kinematic_edges, proximity_edges], dim=1)
        else:
            edge_index = kinematic_edges

        # Build edge attributes: [distance, is_kinematic]
        num_kinematic = kinematic_edges.shape[1]
        num_proximity = proximity_edges.shape[1] if proximity_edges.numel() > 0 else 0

        edge_attr = torch.zeros(num_kinematic + num_proximity, 2)

        # Kinematic edges: compute distances, mark as kinematic
        if num_kinematic > 0:
            kin_src = kinematic_edges[0]
            kin_dst = kinematic_edges[1]
            kin_distances = torch.norm(
                positions[kin_src] - positions[kin_dst], dim=-1
            )
            edge_attr[:num_kinematic, 0] = kin_distances
            edge_attr[:num_kinematic, 1] = 1.0  # is_kinematic = True

        # Proximity edges: already have distances, not kinematic
        if num_proximity > 0:
            edge_attr[num_kinematic:, 0] = proximity_distances
            edge_attr[num_kinematic:, 1] = 0.0  # is_kinematic = False

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types,
            num_joints=self.num_joints,
            object_labels=object_labels,
        )

    def batch_to_graphs(self, states: Tensor) -> list[Data]:
        """Convert a batch of states to graphs.

        Args:
            states: Batch of states, shape (batch_size, state_dim)

        Returns:
            List of Data objects
        """
        return [self.to_graph(state) for state in states]

    def to_graph_with_objects(
        self,
        observation_state: Tensor,
        objects_3d: list[Object3D],
        gripper_state: float | None = None,
    ) -> Data:
        """Convert observation state to graph with detected object nodes.

        Convenience wrapper that extracts positions/labels from Object3D.

        Args:
            observation_state: Robot state tensor of shape (state_dim,)
            objects_3d: List of Object3D from detections_to_objects_3d()
            gripper_state: Optional gripper openness (0=closed, 1=open).
                           If None, attempts to extract from observation_state.

        Returns:
            torch_geometric.data.Data with joint + object nodes
        """
        if not objects_3d:
            return self.to_graph(observation_state, None, None)

        # Extract positions and labels
        positions = torch.tensor(
            [obj.position for obj in objects_3d],
            dtype=torch.float32,
        )
        labels = [obj.class_name for obj in objects_3d]

        # Create graph
        graph = self.to_graph(observation_state, positions, labels)

        # Store additional metadata
        graph.object_confidences = torch.tensor(
            [obj.confidence for obj in objects_3d],
            dtype=torch.float32,
        )

        # Store gripper state for predicate computation
        if gripper_state is not None:
            graph.gripper_state = torch.tensor([gripper_state], dtype=torch.float32)
        elif observation_state.numel() > 6:
            # Assume ALOHA: joint 6 is left gripper, joint 13 is right gripper
            # Gripper values are typically normalized, higher = more open
            left_gripper = observation_state[6].item() if observation_state.numel() > 6 else 0.5
            right_gripper = observation_state[13].item() if observation_state.numel() > 13 else 0.5
            graph.gripper_state = torch.tensor([left_gripper, right_gripper], dtype=torch.float32)
        
        # Store global context vector u for GNN conditioning
        # This enables the predicate head to condition on gripper state
        graph.u = self.extract_global_context(observation_state, gripper_state)

        return graph
    
    def extract_global_context(
        self,
        state_vector: Tensor,
        gripper_state: float | None = None,
    ) -> Tensor:
        """Extract global context vector u for conditioning GNN predictions.
        
        The global context captures gripper states which are essential for
        predicting interaction predicates like is_holding. This enables the
        model to learn: is_holding = (gripper_near_object) AND (gripper_closed).
        
        Args:
            state_vector: ALOHA 14-DoF state vector
            gripper_state: Optional single gripper state (if both grippers same)
            
        Returns:
            Global context vector u of shape (1, 2) where:
              u[0] = left gripper openness (0=closed, 1=open)
              u[1] = right gripper openness (0=closed, 1=open)
        """
        # ALOHA gripper range is typically 0.0 (closed) to ~0.08 (open)
        MAX_WIDTH = 0.08
        
        if gripper_state is not None:
            # Single gripper state provided - use for both
            left_gripper = float(gripper_state)
            right_gripper = float(gripper_state)
        elif state_vector.numel() > 6:
            # Extract from state vector
            # Index 6: left gripper, Index 13: right gripper
            raw_left = state_vector[6].item() if state_vector.numel() > 6 else 0.5
            raw_right = state_vector[13].item() if state_vector.numel() > 13 else 0.5
            
            # Normalize to [0, 1] range
            left_gripper = min(1.0, max(0.0, raw_left / MAX_WIDTH))
            right_gripper = min(1.0, max(0.0, raw_right / MAX_WIDTH))
        else:
            # Default to half-open
            left_gripper = 0.5
            right_gripper = 0.5
        
        # Global context vector u: [left_grip, right_grip]
        # Shape: (1, 2) - batch dimension for PyG batching
        u = torch.tensor([[left_gripper, right_gripper]], dtype=torch.float32)
        return u

    def to_graph_from_detections(
        self,
        observation_state: Tensor,
        detections: list[Detection],
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_pose: np.ndarray | None = None,
        gripper_state: float | None = None,
    ) -> Data:
        """Full pipeline: detections + depth → graph with object nodes.

        Args:
            observation_state: Robot state tensor
            detections: List of Detection objects from VisionDetector
            depth_map: Depth map from DepthEstimator
            intrinsics: Camera intrinsic parameters
            camera_pose: Optional camera-to-world transformation
            gripper_state: Optional gripper openness

        Returns:
            torch_geometric.data.Data with joint + object nodes
        """
        from gnn_reasoner.camera import detections_to_objects_3d

        if not detections:
            return self.to_graph(observation_state, None, None)

        # Convert detections to 3D objects
        objects_3d = detections_to_objects_3d(
            detections, depth_map, intrinsics, camera_pose
        )

        return self.to_graph_with_objects(observation_state, objects_3d, gripper_state)

    def get_gripper_positions(self, graph: Data) -> tuple[Tensor | None, Tensor | None]:
        """Extract gripper (end-effector) positions from graph.

        Args:
            graph: PyG Data object

        Returns:
            Tuple of (left_gripper_pos, right_gripper_pos), each (3,) or None
        """
        if not hasattr(graph, "node_types"):
            return None, None

        gripper_mask = graph.node_types == 1
        gripper_indices = torch.where(gripper_mask)[0]

        if len(gripper_indices) == 0:
            return None, None

        positions = graph.x[:, 2:5]  # x, y, z columns

        if len(gripper_indices) >= 2:
            # Assume first is left, second is right (ALOHA convention)
            left_pos = positions[gripper_indices[0]]
            right_pos = positions[gripper_indices[1]]
            return left_pos, right_pos
        else:
            return positions[gripper_indices[0]], None

    def get_object_positions(self, graph: Data) -> Tensor | None:
        """Extract object positions from graph.

        Args:
            graph: PyG Data object

        Returns:
            Tensor of shape (num_objects, 3) or None
        """
        if not hasattr(graph, "node_types"):
            return None

        object_mask = graph.node_types == 2
        if not object_mask.any():
            return None

        return graph.x[object_mask, 2:5]


def compute_heuristic_predicates(
    graph: Data,
    near_threshold: float = 0.2,
    contact_threshold: float = 0.05,
    holding_threshold: float = 0.08,
    velocity_threshold: float = 0.01,
    gripper_closed_threshold: float = 0.3,
    prev_positions: Tensor | None = None,
) -> Tensor:
    """Compute ground-truth predicate labels from graph structure using heuristics.

    This generates supervision signal for training the RelationalGNN without
    manual annotations. Predicates are derived from spatial relationships.

    Args:
        graph: PyG Data object from LeRobotGraphTransformer
        near_threshold: Distance threshold for is_near predicate (meters)
        contact_threshold: Distance threshold for is_contacting (meters)
        holding_threshold: Distance threshold for is_holding (meters)
        velocity_threshold: Velocity threshold for approaching/retracting
        gripper_closed_threshold: Gripper state below this = closed (0-1 scale)
        prev_positions: Previous frame positions for motion-based predicates

    Returns:
        Tensor of shape (num_edges, num_predicates) with binary labels

    Predicate order:
        0: is_near
        1: is_above
        2: is_below
        3: is_left_of
        4: is_right_of
        5: is_holding
        6: is_contacting
        7: is_approaching
        8: is_retracting
    """
    num_predicates = 9
    edge_index = graph.edge_index
    num_edges = edge_index.shape[1]

    if num_edges == 0:
        return torch.zeros(0, num_predicates)

    # Extract positions from node features (columns 2:5 are x, y, z)
    positions = graph.x[:, 2:5]  # (num_nodes, 3)
    node_types = graph.node_types if hasattr(graph, "node_types") else None

    # Get source and target positions for each edge
    src_pos = positions[edge_index[0]]  # (num_edges, 3)
    tgt_pos = positions[edge_index[1]]  # (num_edges, 3)
    diff = tgt_pos - src_pos  # (num_edges, 3)
    distances = torch.norm(diff, dim=-1)  # (num_edges,)

    labels = torch.zeros(num_edges, num_predicates)

    # Spatial predicates
    labels[:, 0] = (distances < near_threshold).float()  # is_near
    labels[:, 1] = (diff[:, 2] > 0.05).float()  # is_above (target above source)
    labels[:, 2] = (diff[:, 2] < -0.05).float()  # is_below
    labels[:, 3] = (diff[:, 0] < -0.05).float()  # is_left_of
    labels[:, 4] = (diff[:, 0] > 0.05).float()  # is_right_of

    # is_holding: gripper (type=1) near object (type=2) AND gripper is closed
    if node_types is not None:
        src_types = node_types[edge_index[0]]
        tgt_types = node_types[edge_index[1]]

        # Identify gripper-object edges (bidirectional)
        is_gripper_to_object = (src_types == 1) & (tgt_types == 2)
        is_object_to_gripper = (src_types == 2) & (tgt_types == 1)
        gripper_object_edge = is_gripper_to_object | is_object_to_gripper

        # Check distance condition
        close_enough = distances < holding_threshold

        # Check gripper state if available
        if hasattr(graph, "gripper_state") and graph.gripper_state is not None:
            gripper_states = graph.gripper_state
            # For each edge, determine which gripper (left=0, right=1) is involved
            # ALOHA: node indices 14=left gripper, 15=right gripper (or similar)
            # Simplified: check if any gripper is closed
            any_gripper_closed = (gripper_states < gripper_closed_threshold).any()

            # More precise: match gripper node to gripper state
            # For now, use simple heuristic
            is_holding = gripper_object_edge & close_enough & any_gripper_closed
        else:
            # No gripper state, use distance only
            is_holding = gripper_object_edge & close_enough

        labels[:, 5] = is_holding.float()

    # is_contacting: very close (any node types)
    labels[:, 6] = (distances < contact_threshold).float()

    # Motion-based predicates (if previous positions available)
    if prev_positions is not None and prev_positions.shape[0] >= edge_index.max() + 1:
        try:
            prev_src = prev_positions[edge_index[0]]
            prev_tgt = prev_positions[edge_index[1]]
            prev_distances = torch.norm(prev_tgt - prev_src, dim=-1)

            velocity = distances - prev_distances  # negative = approaching

            labels[:, 7] = (velocity < -velocity_threshold).float()  # is_approaching
            labels[:, 8] = (velocity > velocity_threshold).float()  # is_retracting
        except IndexError:
            # Previous graph had different number of nodes (e.g., different detections)
            logger.debug("Skipping motion predicates due to node count mismatch")

    return labels


def compute_object_interaction_predicates(
    graph: Data,
    holding_distance: float = 0.08,
    contact_distance: float = 0.05,
) -> dict[str, list[dict]]:
    """Extract human-readable object interaction predicates.

    Unlike compute_heuristic_predicates which returns edge-level tensors,
    this returns structured interaction data for MCP resources.

    Args:
        graph: PyG Data object with object nodes
        holding_distance: Distance threshold for holding
        contact_distance: Distance threshold for contact

    Returns:
        Dictionary with interaction types as keys:
        {
            "holding": [{"gripper": "left", "object": "cup", "confidence": 0.95}],
            "contacting": [...],
            "near": [...]
        }
    """
    interactions: dict[str, list[dict]] = {
        "holding": [],
        "contacting": [],
        "near": [],
    }

    if not hasattr(graph, "node_types"):
        return interactions

    node_types = graph.node_types
    positions = graph.x[:, 2:5]
    object_labels = getattr(graph, "object_labels", None) or []
    object_confidences = getattr(graph, "object_confidences", None)

    # Find gripper and object indices
    gripper_indices = torch.where(node_types == 1)[0]
    object_indices = torch.where(node_types == 2)[0]

    if len(gripper_indices) == 0 or len(object_indices) == 0:
        return interactions

    # Get gripper states if available
    gripper_states = getattr(graph, "gripper_state", None)

    # Compute gripper-object distances
    for g_idx, gripper_idx in enumerate(gripper_indices):
        gripper_pos = positions[gripper_idx]
        gripper_name = "left" if g_idx == 0 else "right"
        gripper_closed = False

        if gripper_states is not None and g_idx < len(gripper_states):
            gripper_closed = gripper_states[g_idx].item() < 0.3

        for o_idx, object_idx in enumerate(object_indices):
            object_pos = positions[object_idx]
            distance = torch.norm(object_pos - gripper_pos).item()

            # Get object info
            object_label_idx = object_idx.item() - graph.num_joints
            if 0 <= object_label_idx < len(object_labels):
                object_name = object_labels[object_label_idx]
            else:
                object_name = f"object_{o_idx}"

            confidence = 1.0
            if object_confidences is not None and o_idx < len(object_confidences):
                confidence = object_confidences[o_idx].item()

            interaction_info = {
                "gripper": gripper_name,
                "object": object_name,
                "distance": round(distance, 4),
                "confidence": round(confidence, 3),
            }

            # Check interaction types
            if distance < holding_distance and gripper_closed:
                interactions["holding"].append(interaction_info)
            elif distance < contact_distance:
                interactions["contacting"].append(interaction_info)
            elif distance < 0.2:  # near threshold
                interactions["near"].append(interaction_info)

    return interactions


def add_predicate_labels(
    graph: Data,
    prev_graph: Data | None = None,
    **kwargs,
) -> Data:
    """Add ground-truth predicate labels to a graph.

    Args:
        graph: Current frame graph
        prev_graph: Previous frame graph (for motion predicates)
        **kwargs: Passed to compute_heuristic_predicates

    Returns:
        Graph with added 'y' attribute containing predicate labels
    """
    prev_positions = None
    if prev_graph is not None and hasattr(prev_graph, "x"):
        prev_positions = prev_graph.x[:, 2:5]

    labels = compute_heuristic_predicates(graph, prev_positions=prev_positions, **kwargs)
    graph.y = labels
    return graph

