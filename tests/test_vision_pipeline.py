"""Tests for Phase 1 vision pipeline components.

Tests detector, depth estimator, and camera projection utilities
using mock implementations to avoid GPU/model dependencies.
"""

import numpy as np
import pytest

from gnn_reasoner.camera import (
    CameraIntrinsics,
    Object3D,
    bbox_to_3d,
    detections_to_objects_3d,
    estimate_object_size,
    pixel_to_camera,
    pixel_to_world,
)
from gnn_reasoner.depth import MockDepthEstimator
from gnn_reasoner.detector import Detection, MockVisionDetector


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        det = Detection(
            class_name="cup",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
        )
        assert det.class_name == "cup"
        assert det.confidence == 0.95
        assert det.bbox == (100, 100, 200, 200)

    def test_detection_center(self):
        det = Detection(
            class_name="cup",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
        )
        assert det.center == (150, 150)

    def test_detection_area(self):
        det = Detection(
            class_name="cup",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
        )
        assert det.area == 10000  # 100 * 100

    def test_detection_to_dict(self):
        det = Detection(
            class_name="cup",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
        )
        d = det.to_dict()
        assert d["class_name"] == "cup"
        assert d["confidence"] == 0.95
        assert d["bbox"] == [100, 100, 200, 200]
        assert d["center"] == [150, 150]
        assert d["area"] == 10000


class TestMockVisionDetector:
    """Tests for MockVisionDetector."""

    def test_mock_detector_init(self):
        detector = MockVisionDetector()
        assert detector.is_open_vocabulary

    def test_mock_detector_detect(self):
        detector = MockVisionDetector()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        prompts = ["cup", "mug", "bottle"]

        detections = detector.detect(image, prompts)

        assert len(detections) >= 1
        assert len(detections) <= 3
        for det in detections:
            assert isinstance(det, Detection)
            assert det.class_name in prompts
            assert 0.5 <= det.confidence <= 0.95

    def test_mock_detector_bbox_bounds(self):
        detector = MockVisionDetector()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image, ["object"])

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            assert 0 <= x1 < x2 <= 640
            assert 0 <= y1 < y2 <= 480


class TestMockDepthEstimator:
    """Tests for MockDepthEstimator."""

    def test_mock_depth_init(self):
        estimator = MockDepthEstimator()
        assert not estimator.is_metric

    def test_mock_depth_estimate(self):
        estimator = MockDepthEstimator()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        depth_map = estimator.estimate(image)

        assert depth_map.shape == (480, 640)
        assert depth_map.dtype == np.float32
        assert depth_map.min() >= 0

    def test_mock_depth_output_size(self):
        estimator = MockDepthEstimator()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        depth_map = estimator.estimate(image, output_size=(240, 320))

        assert depth_map.shape == (240, 320)

    def test_mock_depth_get_depth_at_point(self):
        estimator = MockDepthEstimator()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_map = estimator.estimate(image)

        depth = estimator.get_depth_at_point(depth_map, (320, 240))

        assert isinstance(depth, float)
        assert depth > 0


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics."""

    def test_intrinsics_creation(self):
        intrinsics = CameraIntrinsics(
            fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480
        )
        assert intrinsics.fx == 600.0
        assert intrinsics.width == 640

    def test_intrinsics_from_matrix(self):
        K = np.array([
            [600.0, 0, 320.0],
            [0, 600.0, 240.0],
            [0, 0, 1],
        ])
        intrinsics = CameraIntrinsics.from_matrix(K, 640, 480)
        assert intrinsics.fx == 600.0
        assert intrinsics.cx == 320.0

    def test_intrinsics_to_matrix(self):
        intrinsics = CameraIntrinsics(
            fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480
        )
        K = intrinsics.to_matrix()
        assert K.shape == (3, 3)
        assert K[0, 0] == 600.0
        assert K[0, 2] == 320.0
        assert K[2, 2] == 1.0

    def test_intrinsics_from_fov(self):
        intrinsics = CameraIntrinsics.from_fov(60.0, 640, 480)
        assert intrinsics.width == 640
        assert intrinsics.cx == 320.0
        # 60° FOV => fx ≈ 554
        assert 500 < intrinsics.fx < 600

    def test_default_aloha(self):
        intrinsics = CameraIntrinsics.default_aloha()
        assert intrinsics.width == 640
        assert intrinsics.height == 480

    def test_default_realsense(self):
        intrinsics = CameraIntrinsics.default_realsense()
        assert intrinsics.width == 640
        assert intrinsics.fx == 617.0


class TestCameraProjection:
    """Tests for camera projection utilities."""

    @pytest.fixture
    def intrinsics(self):
        return CameraIntrinsics(
            fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480
        )

    def test_pixel_to_camera_center(self, intrinsics):
        # Principal point at 1m depth should be (0, 0, 1)
        X, Y, Z = pixel_to_camera(320, 240, 1.0, intrinsics)
        assert abs(X) < 1e-6
        assert abs(Y) < 1e-6
        assert Z == 1.0

    def test_pixel_to_camera_offset(self, intrinsics):
        # Offset from center
        X, Y, Z = pixel_to_camera(320 + 60, 240, 1.0, intrinsics)
        # 60 pixels at fx=600, depth=1 => X = 0.1
        assert abs(X - 0.1) < 1e-6
        assert abs(Y) < 1e-6

    def test_pixel_to_world_default_pose(self, intrinsics):
        # Default camera pose: Z-forward maps to X-forward
        x, y, z = pixel_to_world(320, 240, 1.0, intrinsics, camera_pose=None)
        # Camera (0, 0, 1) -> World (1, 0, 0) with default transform
        assert abs(x - 1.0) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z) < 1e-6

    def test_bbox_to_3d(self, intrinsics):
        depth_map = np.ones((480, 640), dtype=np.float32) * 0.5

        x, y, z = bbox_to_3d(
            (100, 100, 200, 200),
            depth_map,
            intrinsics,
        )

        assert isinstance(x, (float, np.floating))
        assert isinstance(y, (float, np.floating))
        assert isinstance(z, (float, np.floating))

    def test_bbox_to_3d_clamping(self, intrinsics):
        """Test that bbox is clamped to image bounds."""
        depth_map = np.ones((480, 640), dtype=np.float32) * 0.5

        # Bbox extends beyond image
        x, y, z = bbox_to_3d(
            (-50, -50, 100, 100),
            depth_map,
            intrinsics,
        )

        # Should not raise, returns valid coordinates
        assert isinstance(x, (float, np.floating))

    def test_estimate_object_size(self, intrinsics):
        bbox = (200, 200, 300, 350)  # 100x150 pixels
        depth = 1.0

        width, height = estimate_object_size(bbox, depth, intrinsics)

        # 100 pixels at fx=600, depth=1 => ~0.167m
        assert abs(width - 100 / 600) < 1e-6
        assert abs(height - 150 / 600) < 1e-6


class TestDetectionsTo3D:
    """Tests for converting detections to 3D objects."""

    def test_detections_to_objects_3d(self):
        intrinsics = CameraIntrinsics.default_aloha()
        depth_map = np.ones((480, 640), dtype=np.float32) * 0.5

        detections = [
            Detection("cup", 0.95, (200, 200, 300, 300)),
            Detection("mug", 0.85, (400, 300, 500, 400)),
        ]

        objects = detections_to_objects_3d(detections, depth_map, intrinsics)

        assert len(objects) == 2
        assert all(isinstance(obj, Object3D) for obj in objects)
        assert objects[0].class_name == "cup"
        assert objects[1].class_name == "mug"

    def test_object_3d_to_dict(self):
        obj = Object3D(
            class_name="cup",
            confidence=0.95,
            position=(0.5, 0.1, 0.3),
            size=(0.08, 0.12),
            bbox=(200, 200, 300, 300),
        )

        d = obj.to_dict()

        assert d["class_name"] == "cup"
        assert d["position"] == [0.5, 0.1, 0.3]
        assert d["size"] == [0.08, 0.12]


class TestIntegration:
    """Integration tests for the full vision pipeline."""

    def test_full_pipeline_mock(self):
        """Test detector -> depth -> 3D projection pipeline."""
        # Setup
        detector = MockVisionDetector()
        depth_estimator = MockDepthEstimator()
        intrinsics = CameraIntrinsics.default_aloha()

        # Generate synthetic image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run pipeline
        detections = detector.detect(image, prompts=["cup", "mug"])
        depth_map = depth_estimator.estimate(image)
        objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)

        # Verify
        assert len(objects_3d) >= 1
        for obj in objects_3d:
            assert obj.class_name in ["cup", "mug"]
            assert len(obj.position) == 3
            assert obj.size is not None
            assert len(obj.size) == 2

