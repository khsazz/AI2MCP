"""Tests for LeRobot GNN pipeline components.

Tests the full pipeline: DataManager → GraphTransformer → RelationalGNN → MCP
Uses mocked LeRobot data to avoid network dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_lerobot_frame() -> dict:
    """Generate a mock LeRobot frame with ALOHA-style data."""
    return {
        "observation.state": torch.randn(14),  # 14 joints
        "action": torch.randn(14),
        "episode_index": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "observation.images.cam_high": torch.randn(3, 480, 640),
    }


@pytest.fixture
def mock_lerobot_dataset(mock_lerobot_frame):
    """Create a mocked LeRobotDataset."""
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=100)
    mock_dataset.__getitem__ = MagicMock(return_value=mock_lerobot_frame)
    mock_dataset.hf_dataset = {
        "episode_index": [0] * 50 + [1] * 50  # 2 episodes of 50 frames each
    }
    mock_dataset.info = {"action_names": [f"joint_{i}" for i in range(14)]}
    return mock_dataset


@pytest.fixture
def sample_state() -> torch.Tensor:
    """Sample robot state tensor (14 joints + 14 velocities)."""
    return torch.randn(28)


@pytest.fixture
def sample_graph():
    """Sample PyG graph for testing."""
    from gnn_reasoner.lerobot_transformer import LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN

    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    state = torch.randn(14)
    return transformer.to_graph(state)


# ==============================================================================
# DataManager Tests
# ==============================================================================


class TestDataManager:
    """Tests for DataManager class."""

    def test_initialization(self, mock_lerobot_dataset):
        """Test DataManager can be initialized."""
        from gnn_reasoner.data_manager import DataManager

        with patch(
            "gnn_reasoner.data_manager.DataManager._load_dataset",
            return_value=mock_lerobot_dataset,
        ):
            dm = DataManager("lerobot/test_dataset")
            assert dm.repo_id == "lerobot/test_dataset"
            assert dm.streaming is True

    def test_get_frame(self, mock_lerobot_dataset, mock_lerobot_frame):
        """Test frame retrieval."""
        from gnn_reasoner.data_manager import DataManager

        with patch(
            "gnn_reasoner.data_manager.DataManager._load_dataset",
            return_value=mock_lerobot_dataset,
        ):
            dm = DataManager("lerobot/test_dataset")
            frame = dm.get_frame(0)

            assert "observation.state" in frame
            assert "action" in frame

    def test_get_state_returns_tensor(self, mock_lerobot_dataset, mock_lerobot_frame):
        """Test state extraction returns tensor."""
        from gnn_reasoner.data_manager import DataManager

        with patch(
            "gnn_reasoner.data_manager.DataManager._load_dataset",
            return_value=mock_lerobot_dataset,
        ):
            dm = DataManager("lerobot/test_dataset")
            state = dm.get_state(0)

            assert isinstance(state, torch.Tensor)
            assert state.shape[0] == 14

    def test_advance_frame(self, mock_lerobot_dataset):
        """Test frame advancement."""
        from gnn_reasoner.data_manager import DataManager

        with patch(
            "gnn_reasoner.data_manager.DataManager._load_dataset",
            return_value=mock_lerobot_dataset,
        ):
            dm = DataManager("lerobot/test_dataset")
            dm.set_frame_index(0)
            dm.advance_frame()
            assert dm._current_frame_idx == 1

    def test_action_space(self, mock_lerobot_dataset):
        """Test action space retrieval."""
        from gnn_reasoner.data_manager import DataManager

        with patch(
            "gnn_reasoner.data_manager.DataManager._load_dataset",
            return_value=mock_lerobot_dataset,
        ):
            dm = DataManager("lerobot/test_dataset")
            action_space = dm.get_action_space()

            assert action_space.dim == 14
            assert len(action_space.names) == 14


# ==============================================================================
# LeRobotGraphTransformer Tests
# ==============================================================================


class TestLeRobotGraphTransformer:
    """Tests for LeRobotGraphTransformer."""

    def test_initialization(self):
        """Test transformer initialization."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        assert transformer.num_joints == 16
        assert len(transformer.kinematic_chain) == len(ALOHA_KINEMATIC_CHAIN)

    def test_to_graph_output_structure(self, sample_state):
        """Test graph output has correct structure."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        graph = transformer.to_graph(sample_state[:14])

        # Check required attributes
        assert hasattr(graph, "x")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "edge_attr")
        assert hasattr(graph, "node_types")

    def test_node_features_shape(self, sample_state):
        """Test node features have correct shape."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        graph = transformer.to_graph(sample_state[:14])

        # Node features: [angle, velocity, x, y, z]
        assert graph.x.shape == (16, 5)

    def test_kinematic_edges_bidirectional(self):
        """Test kinematic edges are bidirectional."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        state = torch.zeros(14)
        graph = transformer.to_graph(state)

        # Should have 2 * len(kinematic_chain) kinematic edges
        expected_kinematic_edges = 2 * len(ALOHA_KINEMATIC_CHAIN)
        # May have additional proximity edges
        assert graph.edge_index.shape[1] >= expected_kinematic_edges

    def test_edge_attributes_shape(self, sample_state):
        """Test edge attributes have correct shape."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        graph = transformer.to_graph(sample_state[:14])

        # Edge attr: [distance, is_kinematic]
        num_edges = graph.edge_index.shape[1]
        assert graph.edge_attr.shape == (num_edges, 2)

    def test_with_object_positions(self, sample_state):
        """Test graph with additional object nodes."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        object_positions = torch.tensor([[0.5, 0.5, 0.1], [0.3, -0.2, 0.0]])

        graph = transformer.to_graph(
            sample_state[:14],
            object_positions=object_positions,
            object_labels=["cup", "plate"],
        )

        # 16 joints + 2 objects
        assert graph.x.shape[0] == 18
        # Last 2 nodes should be objects (type 2)
        assert graph.node_types[-2:].tolist() == [2, 2]

    def test_batch_to_graphs(self):
        """Test batch conversion."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        states = torch.randn(5, 14)
        graphs = transformer.batch_to_graphs(states)

        assert len(graphs) == 5
        for g in graphs:
            assert g.x.shape[0] == 16


# ==============================================================================
# Heuristic Predicate Tests
# ==============================================================================


class TestHeuristicPredicates:
    """Tests for heuristic predicate computation."""

    def test_compute_predicate_labels_shape(self, sample_graph):
        """Test predicate labels have correct shape."""
        from gnn_reasoner.lerobot_transformer import compute_heuristic_predicates

        labels = compute_heuristic_predicates(sample_graph)
        num_edges = sample_graph.edge_index.shape[1]

        assert labels.shape == (num_edges, 9)

    def test_spatial_predicates_mutual_exclusion(self, sample_graph):
        """Test is_above and is_below are mutually exclusive."""
        from gnn_reasoner.lerobot_transformer import compute_heuristic_predicates

        labels = compute_heuristic_predicates(sample_graph)

        # is_above (1) and is_below (2) should not both be 1
        both_above_below = (labels[:, 1] == 1) & (labels[:, 2] == 1)
        assert not both_above_below.any()

    def test_add_predicate_labels(self, sample_graph):
        """Test adding labels to graph."""
        from gnn_reasoner.lerobot_transformer import add_predicate_labels

        labeled_graph = add_predicate_labels(sample_graph)

        assert hasattr(labeled_graph, "y")
        assert labeled_graph.y.shape[0] == labeled_graph.edge_index.shape[1]


# ==============================================================================
# Global Context Tests (ConditionalPredicateHead)
# ==============================================================================


class TestGlobalContextConditioning:
    """Tests for global context (gripper state) conditioning."""

    def test_extract_global_context_from_state(self):
        """Test extract_global_context returns correct shape."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        state = torch.randn(14)

        u = transformer.extract_global_context(state)

        assert u.shape == (1, 2)  # [left_gripper, right_gripper]

    def test_extract_global_context_with_explicit_gripper(self):
        """Test extract_global_context with explicit gripper_state."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        state = torch.randn(14)

        u = transformer.extract_global_context(state, gripper_state=0.02)

        assert u.shape == (1, 2)
        # Both grippers should have same value when explicit
        assert u[0, 0] == u[0, 1]

    def test_graph_has_u_attribute(self):
        """Test that to_graph_with_objects() sets graph.u."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )
        from gnn_reasoner.camera import Object3D

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        state = torch.randn(14)

        # Create a mock object for to_graph_with_objects
        mock_object = Object3D(
            class_name="cup",
            confidence=0.9,
            position=(0.3, 0.0, 0.1),
            size=(0.05, 0.08),
            bbox=(100, 100, 200, 200),
        )

        graph = transformer.to_graph_with_objects(state, [mock_object], gripper_state=0.02)

        assert hasattr(graph, "u")
        assert graph.u is not None
        assert graph.u.shape == (1, 2)

    def test_closed_gripper_context(self):
        """Test that closed gripper (0.0) produces low u values."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        state = torch.zeros(14)  # All zeros = closed grippers

        u = transformer.extract_global_context(state)

        # Closed gripper should have low values
        assert u[0, 0].item() < 0.5
        assert u[0, 1].item() < 0.5


class TestConditionalPredicateHead:
    """Tests for ConditionalPredicateHead class."""

    def test_head_creation(self):
        """Test head initialization."""
        from gnn_reasoner.model import ConditionalPredicateHead

        head = ConditionalPredicateHead(hidden_dim=64, global_dim=2, num_predicates=9)

        assert head is not None
        assert head.num_predicates == 9
        assert head.global_dim == 2

    def test_head_forward(self):
        """Test forward pass with global context."""
        from gnn_reasoner.model import ConditionalPredicateHead

        head = ConditionalPredicateHead(hidden_dim=64, global_dim=2, num_predicates=9)

        node_embeddings = torch.randn(16, 64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges
        u = torch.tensor([[0.0, 0.0]])  # Closed gripper

        logits = head(node_embeddings, edge_index, u)

        assert logits.shape == (3, 9)

    def test_different_gripper_states_produce_different_outputs(self):
        """Test that open vs closed gripper produces different predicate logits."""
        from gnn_reasoner.model import ConditionalPredicateHead

        head = ConditionalPredicateHead(hidden_dim=64, global_dim=2, num_predicates=9)

        node_embeddings = torch.randn(16, 64)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        u_closed = torch.tensor([[0.0, 0.0]])
        u_open = torch.tensor([[1.0, 1.0]])

        logits_closed = head(node_embeddings, edge_index, u_closed)
        logits_open = head(node_embeddings, edge_index, u_open)

        # Outputs should be different (not necessarily in a specific direction)
        assert not torch.allclose(logits_closed, logits_open)


# ==============================================================================
# RelationalGNN Tests
# ==============================================================================


class TestRelationalGNN:
    """Tests for RelationalGNN model."""

    @pytest.fixture
    def gnn_model(self):
        """Create a RelationalGNN instance."""
        from gnn_reasoner.model import RelationalGNN

        return RelationalGNN(
            node_input_dim=5,
            edge_input_dim=2,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
        )

    @pytest.fixture
    def gnn_model_with_conditioning(self):
        """Create a RelationalGNN with global conditioning."""
        from gnn_reasoner.model import RelationalGNN

        return RelationalGNN(
            node_input_dim=5,
            edge_input_dim=2,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            use_global_conditioning=True,
            global_dim=2,
        )

    def test_forward_output_structure(self, gnn_model, sample_graph):
        """Test forward pass output structure."""
        outputs = gnn_model(sample_graph)

        assert "node_embeddings" in outputs
        assert "predicate_logits" in outputs
        assert "graph_embedding" in outputs

    def test_node_embeddings_shape(self, gnn_model, sample_graph):
        """Test node embeddings shape."""
        outputs = gnn_model(sample_graph)
        num_nodes = sample_graph.x.shape[0]

        assert outputs["node_embeddings"].shape == (num_nodes, 64)

    def test_predicate_logits_shape(self, gnn_model, sample_graph):
        """Test predicate logits shape."""
        outputs = gnn_model(sample_graph)
        num_edges = sample_graph.edge_index.shape[1]

        assert outputs["predicate_logits"].shape == (num_edges, 9)

    def test_graph_embedding_shape(self, gnn_model, sample_graph):
        """Test graph embedding shape."""
        outputs = gnn_model(sample_graph)

        assert outputs["graph_embedding"].shape == (1, 64)

    def test_predict_predicates(self, gnn_model, sample_graph):
        """Test predicate prediction method."""
        predicates = gnn_model.predict_predicates(sample_graph, threshold=0.5)

        assert isinstance(predicates, list)
        if predicates:
            pred = predicates[0]
            assert hasattr(pred, "predicate_name")
            assert hasattr(pred, "source_node")
            assert hasattr(pred, "target_node")
            assert hasattr(pred, "probability")
            assert hasattr(pred, "active")

    def test_to_world_context(self, gnn_model, sample_graph):
        """Test world context generation."""
        context = gnn_model.to_world_context(sample_graph)

        assert "num_nodes" in context
        assert "num_edges" in context
        assert "spatial_predicates" in context
        assert "interaction_predicates" in context
        assert "graph_embedding" in context

    def test_graph_to_json(self, gnn_model, sample_graph):
        """Test JSON serialization."""
        import json

        json_data = gnn_model.graph_to_json(sample_graph)

        # Should be JSON-serializable
        json_str = json.dumps(json_data)
        assert json_str is not None

        assert "nodes" in json_data
        assert "edges" in json_data

    def test_forward_with_global_context(self, gnn_model_with_conditioning):
        """Test forward pass with explicit global context u."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )
        from gnn_reasoner.camera import Object3D

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        state = torch.randn(14)
        
        # Use to_graph_with_objects which sets graph.u
        mock_obj = Object3D(
            class_name="cup",
            confidence=0.9,
            position=(0.3, 0.0, 0.1),
            size=(0.05, 0.08),
            bbox=(100, 100, 200, 200),
        )
        graph = transformer.to_graph_with_objects(state, [mock_obj], gripper_state=0.02)

        outputs = gnn_model_with_conditioning(graph)

        assert "predicate_logits" in outputs
        assert outputs["predicate_logits"].shape[0] == graph.edge_index.shape[1]

    def test_conditioning_affects_is_holding(self, gnn_model_with_conditioning):
        """Test that gripper state affects is_holding predictions."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
        )
        from gnn_reasoner.camera import Object3D

        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        state = torch.randn(14)
        
        mock_obj = Object3D(
            class_name="cup",
            confidence=0.9,
            position=(0.3, 0.0, 0.1),
            size=(0.05, 0.08),
            bbox=(100, 100, 200, 200),
        )

        # Create two graphs: one with closed gripper, one with open
        graph_closed = transformer.to_graph_with_objects(state, [mock_obj], gripper_state=0.0)
        graph_open = transformer.to_graph_with_objects(state, [mock_obj], gripper_state=0.08)

        with torch.no_grad():
            out_closed = gnn_model_with_conditioning(graph_closed)
            out_open = gnn_model_with_conditioning(graph_open)

        # Outputs should be different
        assert not torch.allclose(
            out_closed["predicate_logits"],
            out_open["predicate_logits"],
        )


# ==============================================================================
# BenchmarkLogger Tests
# ==============================================================================


class TestBenchmarkLogger:
    """Tests for BenchmarkLogger."""

    def test_log_inference_latency(self):
        """Test logging inference latency."""
        from gnn_reasoner.benchmark import BenchmarkLogger

        logger = BenchmarkLogger("test")
        logger.log_inference_latency(15.0)
        logger.log_inference_latency(20.0)

        metrics = logger.export_metrics()
        assert metrics["timing"]["inference_latency"]["count"] == 2
        assert metrics["timing"]["inference_latency"]["mean_ms"] == 17.5

    def test_log_protocol_overhead(self):
        """Test logging protocol overhead."""
        from gnn_reasoner.benchmark import BenchmarkLogger

        logger = BenchmarkLogger("test")
        logger.log_protocol_overhead(serialization_ms=2.0, deserialization_ms=1.5)

        metrics = logger.export_metrics()
        assert metrics["timing"]["serialization_time"]["count"] == 1
        assert metrics["timing"]["deserialization_time"]["count"] == 1

    def test_pass_at_k_scores(self):
        """Test pass@k calculation."""
        from gnn_reasoner.benchmark import BenchmarkLogger

        logger = BenchmarkLogger("test")

        # Log predictions that should pass@1
        logger.log_prediction([0.1, 0.2, 0.3], [0.1])  # Exact match
        logger.log_prediction([0.5, 0.1, 0.2], [0.1])  # Match at position 2

        scores = logger.get_pass_at_k_scores([1, 3])

        assert scores["pass@1"] == 0.5  # Only first prediction passes
        assert scores["pass@3"] == 1.0  # Both pass within top 3

    def test_timing_decorator(self):
        """Test timing decorator."""
        from gnn_reasoner.benchmark import BenchmarkLogger
        import time

        logger = BenchmarkLogger("test")

        @logger.time_function("test_op")
        def slow_operation():
            time.sleep(0.01)
            return 42

        result = slow_operation()
        assert result == 42

        metrics = logger.export_metrics()
        assert "test_op" in metrics["custom_metrics"]
        assert metrics["custom_metrics"]["test_op"]["count"] == 1

    def test_summary_output(self):
        """Test summary string generation."""
        from gnn_reasoner.benchmark import BenchmarkLogger

        logger = BenchmarkLogger("test")
        logger.log_inference_latency(15.0)
        summary = logger.summary()

        assert "Benchmark: test" in summary
        assert "inference_latency" in summary

    def test_reset(self):
        """Test reset functionality."""
        from gnn_reasoner.benchmark import BenchmarkLogger

        logger = BenchmarkLogger("test")
        logger.log_inference_latency(15.0)
        logger.reset()

        metrics = logger.export_metrics()
        assert metrics["timing"]["inference_latency"]["count"] == 0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_without_lerobot(self):
        """Test full pipeline with synthetic data (no LeRobot dependency)."""
        from gnn_reasoner.lerobot_transformer import (
            LeRobotGraphTransformer,
            ALOHA_KINEMATIC_CHAIN,
            add_predicate_labels,
        )
        from gnn_reasoner.model import RelationalGNN
        from gnn_reasoner.benchmark import BenchmarkLogger
        import time

        # Initialize components
        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        gnn = RelationalGNN(hidden_dim=64, num_layers=2)
        benchmark = BenchmarkLogger("integration_test")

        # Simulate 10 frames
        prev_graph = None
        for i in range(10):
            state = torch.randn(14)

            # Transform to graph
            start = time.perf_counter()
            graph = transformer.to_graph(state)
            benchmark.log_graph_construction_time((time.perf_counter() - start) * 1000)

            # Add ground truth labels
            graph = add_predicate_labels(graph, prev_graph)

            # GNN inference
            start = time.perf_counter()
            context = gnn.to_world_context(graph)
            benchmark.log_inference_latency((time.perf_counter() - start) * 1000)

            prev_graph = graph

            # Verify output structure
            assert "spatial_predicates" in context
            assert "interaction_predicates" in context

        # Check benchmarks
        metrics = benchmark.export_metrics()
        assert metrics["timing"]["inference_latency"]["count"] == 10
        assert metrics["timing"]["graph_construction_time"]["count"] == 10

    def test_json_serialization_roundtrip(self, sample_graph):
        """Test that graph can be serialized and predicates extracted."""
        import json
        from gnn_reasoner.model import RelationalGNN

        gnn = RelationalGNN(hidden_dim=64, num_layers=2)
        context = gnn.to_world_context(sample_graph)

        # Should be JSON serializable
        json_str = json.dumps(context)
        parsed = json.loads(json_str)

        assert parsed["num_nodes"] == context["num_nodes"]
        assert len(parsed["spatial_predicates"]) == len(context["spatial_predicates"])

