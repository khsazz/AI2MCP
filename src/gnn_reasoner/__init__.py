"""GNN-based Semantic Scene Reasoner.

This package provides graph neural network models for processing
robot perception data into semantic world graphs.
"""

from gnn_reasoner.data_manager import DataManager, Episode, ActionSpace
from gnn_reasoner.lerobot_transformer import (
    LeRobotGraphTransformer,
    ALOHA_KINEMATIC_CHAIN,
    compute_heuristic_predicates,
    compute_object_interaction_predicates,
    add_predicate_labels,
)
from gnn_reasoner.benchmark import BenchmarkLogger, get_benchmark_logger
from gnn_reasoner.detector import VisionDetector, MockVisionDetector, Detection
from gnn_reasoner.depth import DepthEstimator, MockDepthEstimator
from gnn_reasoner.camera import (
    CameraIntrinsics,
    pixel_to_world,
    bbox_to_3d,
    detections_to_objects_3d,
    Object3D,
)

__version__ = "0.1.0"

__all__ = [
    # Data management
    "DataManager",
    "Episode",
    "ActionSpace",
    # Graph transformation
    "LeRobotGraphTransformer",
    "ALOHA_KINEMATIC_CHAIN",
    "compute_heuristic_predicates",
    "compute_object_interaction_predicates",
    "add_predicate_labels",
    # Benchmarking
    "BenchmarkLogger",
    "get_benchmark_logger",
    # Vision (Phase 1)
    "VisionDetector",
    "MockVisionDetector",
    "Detection",
    "DepthEstimator",
    "MockDepthEstimator",
    "CameraIntrinsics",
    "pixel_to_world",
    "bbox_to_3d",
    "detections_to_objects_3d",
    "Object3D",
]

