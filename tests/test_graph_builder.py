"""Tests for GNN graph builder."""

from __future__ import annotations

import pytest
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_reasoner.graph_builder import (
    GraphNode,
    GraphEdge,
    WorldGraph,
    SemanticGraphBuilder,
)


class TestWorldGraph:
    """Tests for WorldGraph data structure."""

    def test_empty_graph(self) -> None:
        """Test empty graph creation."""
        graph = WorldGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self) -> None:
        """Test adding nodes to graph."""
        graph = WorldGraph()
        node = GraphNode(
            id="robot",
            node_type="robot",
            position=(0.0, 0.0),
        )
        graph.add_node(node)
        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == "robot"

    def test_add_duplicate_node(self) -> None:
        """Test that duplicate nodes are not added."""
        graph = WorldGraph()
        node1 = GraphNode(id="robot", node_type="robot", position=(0.0, 0.0))
        node2 = GraphNode(id="robot", node_type="robot", position=(1.0, 1.0))
        graph.add_node(node1)
        graph.add_node(node2)
        assert len(graph.nodes) == 1

    def test_add_edge(self) -> None:
        """Test adding edges to graph."""
        graph = WorldGraph()
        graph.add_node(GraphNode(id="a", node_type="obstacle", position=(0, 0)))
        graph.add_node(GraphNode(id="b", node_type="obstacle", position=(1, 1)))
        edge = GraphEdge(source="a", target="b", relation="near")
        graph.add_edge(edge)
        assert len(graph.edges) == 1

    def test_to_dict(self) -> None:
        """Test graph serialization."""
        graph = WorldGraph()
        graph.add_node(GraphNode(id="robot", node_type="robot", position=(0.0, 0.0)))
        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 1


class TestSemanticGraphBuilder:
    """Tests for SemanticGraphBuilder."""

    def test_robot_node_always_added(self) -> None:
        """Test that robot node is always present."""
        builder = SemanticGraphBuilder()
        graph = builder.build_graph(robot_pose=(0.0, 0.0, 0.0))
        
        robot_nodes = [n for n in graph.nodes if n.node_type == "robot"]
        assert len(robot_nodes) == 1
        assert robot_nodes[0].id == "robot"

    def test_obstacle_nodes_from_scan(self) -> None:
        """Test adding obstacles from LiDAR data."""
        builder = SemanticGraphBuilder()
        obstacles = [
            {"world_x": 1.0, "world_y": 0.0, "distance": 1.0, "size": 0.2},
            {"world_x": 0.0, "world_y": 1.0, "distance": 1.0, "size": 0.3},
        ]
        graph = builder.build_graph(
            robot_pose=(0.0, 0.0, 0.0),
            scan_obstacles=obstacles,
        )
        
        obstacle_nodes = [n for n in graph.nodes if n.node_type == "obstacle"]
        assert len(obstacle_nodes) == 2

    def test_edge_creation(self) -> None:
        """Test that edges are created between nearby entities."""
        builder = SemanticGraphBuilder(spatial_threshold=2.0)
        obstacles = [
            {"world_x": 0.5, "world_y": 0.0, "distance": 0.5, "size": 0.2},
        ]
        graph = builder.build_graph(
            robot_pose=(0.0, 0.0, 0.0),
            scan_obstacles=obstacles,
        )
        
        # Should have edge from robot to nearby obstacle
        robot_edges = [e for e in graph.edges if e.source == "robot" or e.target == "robot"]
        assert len(robot_edges) >= 1

    def test_blocking_relation(self) -> None:
        """Test that very close obstacles are marked as blocking."""
        builder = SemanticGraphBuilder(blocking_threshold=0.5)
        obstacles = [
            {"world_x": 0.3, "world_y": 0.0, "distance": 0.3, "size": 0.2},
        ]
        graph = builder.build_graph(
            robot_pose=(0.0, 0.0, 0.0),
            scan_obstacles=obstacles,
        )
        
        blocking_edges = [e for e in graph.edges if e.relation == "blocking"]
        assert len(blocking_edges) >= 1

