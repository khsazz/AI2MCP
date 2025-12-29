"""LeRobot Graph Transformer.

Converts LeRobot observation states into torch_geometric graph structures
suitable for GNN-based relational reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data


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

