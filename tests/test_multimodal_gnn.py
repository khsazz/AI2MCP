"""Tests for MultiModalGNN (Option C).

Tests the vision-enhanced GNN architecture including:
- VisionEncoder with DINOv2
- CrossAttentionFusion layer
- Full MultiModalGNN forward pass
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from gnn_reasoner import (
    LeRobotGraphTransformer,
    ALOHA_KINEMATIC_CHAIN,
    MockVisionDetector,
    MockDepthEstimator,
    CameraIntrinsics,
    detections_to_objects_3d,
)
from gnn_reasoner.model import (
    MultiModalGNN,
    VisionEncoder,
    CrossAttentionFusion,
    MockVisionEncoder,
)


class TestVisionEncoder:
    """Tests for VisionEncoder."""

    def test_mock_vision_encoder(self):
        encoder = MockVisionEncoder(hidden_dim=128)
        image = torch.randn(1, 3, 480, 640)
        bboxes = [(100, 100, 200, 200), (300, 200, 400, 350)]

        features = encoder(image, bboxes)

        assert features.shape == (2, 128)

    def test_mock_vision_encoder_empty_bboxes(self):
        encoder = MockVisionEncoder(hidden_dim=128)
        image = torch.randn(1, 3, 480, 640)
        bboxes = []

        features = encoder(image, bboxes)

        # Should return at least one feature vector
        assert features.shape[0] >= 1
        assert features.shape[1] == 128


class TestCrossAttentionFusion:
    """Tests for CrossAttentionFusion layer."""

    def test_fusion_creation(self):
        fusion = CrossAttentionFusion(hidden_dim=128, num_heads=4)
        assert fusion is not None

    def test_fusion_forward(self):
        fusion = CrossAttentionFusion(hidden_dim=128, num_heads=4)

        kinematic = torch.randn(16, 128)  # 16 kinematic nodes
        vision = torch.randn(3, 128)  # 3 object nodes

        fused_kin, fused_vis = fusion(kinematic, vision)

        assert fused_kin.shape == (16, 128)
        assert fused_vis.shape == (3, 128)

    def test_fusion_single_object(self):
        fusion = CrossAttentionFusion(hidden_dim=128, num_heads=4)

        kinematic = torch.randn(16, 128)
        vision = torch.randn(1, 128)  # Single object

        fused_kin, fused_vis = fusion(kinematic, vision)

        assert fused_kin.shape == (16, 128)
        assert fused_vis.shape == (1, 128)


class TestMultiModalGNN:
    """Tests for MultiModalGNN."""

    @pytest.fixture
    def model(self):
        """Create MultiModalGNN with mock vision encoder for fast testing."""
        model = MultiModalGNN(
            hidden_dim=64,
            num_predicates=9,
            vision_model="dinov2_vits14",
            freeze_vision=True,
        )
        # Replace vision encoder with mock for testing
        model.vision_encoder = MockVisionEncoder(hidden_dim=64)
        return model

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph with objects."""
        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        detector = MockVisionDetector()
        depth_estimator = MockDepthEstimator()
        intrinsics = CameraIntrinsics.default_aloha()

        state = torch.randn(14)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image, prompts=["cup", "mug"])
        depth_map = depth_estimator.estimate(image)
        objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)

        graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state=0.2)
        bboxes = [det.bbox for det in detections]
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        return graph, image_tensor.unsqueeze(0), bboxes

    def test_model_creation(self, model):
        assert model is not None
        assert model.hidden_dim == 64
        assert model.num_predicates == 9

    def test_parameter_count(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_forward_without_vision(self, model, sample_graph):
        graph, image, bboxes = sample_graph

        with torch.no_grad():
            output = model(graph)

        assert "node_embeddings" in output
        assert "predicate_logits" in output
        assert "graph_embedding" in output

    def test_forward_with_vision(self, model, sample_graph):
        graph, image, bboxes = sample_graph

        with torch.no_grad():
            output = model(graph, image, bboxes)

        assert output["node_embeddings"].shape[0] == graph.x.shape[0]
        assert output["predicate_logits"].shape[0] == graph.edge_index.shape[1]
        assert output["graph_embedding"].shape == (1, 64)

    def test_predict_predicates(self, model, sample_graph):
        graph, image, bboxes = sample_graph

        predicates = model.predict_predicates(graph, image, bboxes, threshold=0.5)

        assert len(predicates) > 0
        assert all(hasattr(p, "predicate_name") for p in predicates)
        assert all(hasattr(p, "probability") for p in predicates)

    def test_get_active_predicates(self, model, sample_graph):
        graph, image, bboxes = sample_graph

        active = model.get_active_predicates(graph, image, bboxes, threshold=0.5)

        assert all(p.active for p in active)

    def test_to_world_context(self, model, sample_graph):
        graph, image, bboxes = sample_graph

        context = model.to_world_context(graph, image, bboxes, threshold=0.5)

        assert "num_nodes" in context
        assert "num_edges" in context
        assert "graph_embedding" in context
        assert "spatial_predicates" in context
        assert "interaction_predicates" in context


class TestMultiModalIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_cpu(self):
        """Test full pipeline on CPU."""
        device = "cpu"

        # Create model
        model = MultiModalGNN(
            hidden_dim=64,
            num_predicates=9,
            vision_model="dinov2_vits14",
            freeze_vision=True,
        )
        # Use mock for testing
        model.vision_encoder = MockVisionEncoder(hidden_dim=64)
        model = model.to(device)
        model.eval()

        # Create data
        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        detector = MockVisionDetector()
        depth_estimator = MockDepthEstimator()
        intrinsics = CameraIntrinsics.default_aloha()

        state = torch.randn(14)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image, prompts=["cup"])
        depth_map = depth_estimator.estimate(image)
        objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)

        graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state=0.2)
        graph = graph.to(device)

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        bboxes = [det.bbox for det in detections]

        # Forward
        with torch.no_grad():
            output = model(graph, image_tensor, bboxes)

        assert output["node_embeddings"].device.type == device
        assert output["predicate_logits"].device.type == device

    def test_training_step(self):
        """Test a single training step."""
        device = "cpu"

        model = MultiModalGNN(hidden_dim=64, num_predicates=9)
        model.vision_encoder = MockVisionEncoder(hidden_dim=64)
        model = model.to(device)
        model.train()

        # Create sample data with labels
        transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
        detector = MockVisionDetector()
        depth_estimator = MockDepthEstimator()
        intrinsics = CameraIntrinsics.default_aloha()

        state = torch.randn(14)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image, prompts=["cup"])
        depth_map = depth_estimator.estimate(image)
        objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)

        graph = transformer.to_graph_with_objects(state, objects_3d)
        graph = graph.to(device)

        # Add fake labels
        num_edges = graph.edge_index.shape[1]
        graph.y = torch.randint(0, 2, (num_edges, 9)).float()

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        bboxes = [det.bbox for det in detections]

        # Training step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        output = model(graph, image_tensor, bboxes)
        loss = criterion(output["predicate_logits"], graph.y)

        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
        assert loss.item() >= 0

