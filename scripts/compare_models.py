#!/usr/bin/env python3
"""Comparison benchmark: Option A (RelationalGNN) vs Option C (MultiModalGNN).

Evaluates both models on:
- Predicate detection accuracy (per-class and macro)
- Inference latency (mean, P50, P95, P99)
- Memory usage
- Generalization to held-out data

Usage:
    python scripts/compare_models.py --synthetic --frames 500
    python scripts/compare_models.py --repo lerobot/aloha_static_coffee --frames 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from gnn_reasoner import (
    DataManager,
    LeRobotGraphTransformer,
    ALOHA_KINEMATIC_CHAIN,
    add_predicate_labels,
    MockVisionDetector,
    MockDepthEstimator,
    CameraIntrinsics,
    detections_to_objects_3d,
)
from gnn_reasoner.model import RelationalGNN, MultiModalGNN, MockVisionEncoder
from gnn_reasoner.model.relational_gnn import ALL_PREDICATES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single model benchmark."""

    model_name: str
    total_samples: int

    # Accuracy metrics
    accuracy_per_predicate: dict[str, float] = field(default_factory=dict)
    precision_per_predicate: dict[str, float] = field(default_factory=dict)
    recall_per_predicate: dict[str, float] = field(default_factory=dict)
    f1_per_predicate: dict[str, float] = field(default_factory=dict)
    macro_f1: float = 0.0
    micro_accuracy: float = 0.0

    # Latency metrics (ms)
    latency_mean: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Memory metrics (MB)
    peak_memory_mb: float = 0.0
    model_size_mb: float = 0.0

    # Timing breakdown (ms)
    detection_time: float = 0.0
    depth_time: float = 0.0
    graph_time: float = 0.0
    inference_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "accuracy": {
                "per_predicate": self.accuracy_per_predicate,
                "precision_per_predicate": self.precision_per_predicate,
                "recall_per_predicate": self.recall_per_predicate,
                "f1_per_predicate": self.f1_per_predicate,
                "macro_f1": self.macro_f1,
                "micro_accuracy": self.micro_accuracy,
            },
            "latency_ms": {
                "mean": self.latency_mean,
                "p50": self.latency_p50,
                "p95": self.latency_p95,
                "p99": self.latency_p99,
            },
            "memory_mb": {
                "peak": self.peak_memory_mb,
                "model_size": self.model_size_mb,
            },
            "timing_breakdown_ms": {
                "detection": self.detection_time,
                "depth": self.depth_time,
                "graph": self.graph_time,
                "inference": self.inference_time,
            },
        }


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics.

    Args:
        predictions: Logits (N, num_predicates)
        targets: Binary labels (N, num_predicates)
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary with per-class and aggregate metrics
    """
    pred_binary = (torch.sigmoid(predictions) > threshold).float()

    num_predicates = predictions.shape[1]
    metrics = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
    }

    total_correct = 0
    total_samples = 0
    f1_scores = []

    for i, pred_name in enumerate(ALL_PREDICATES):
        p = pred_binary[:, i]
        t = targets[:, i]

        # Accuracy
        correct = (p == t).sum().item()
        total = t.numel()
        acc = correct / total if total > 0 else 0

        # Precision, Recall, F1
        tp = ((p == 1) & (t == 1)).sum().item()
        fp = ((p == 1) & (t == 0)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics["accuracy"][pred_name] = round(acc, 4)
        metrics["precision"][pred_name] = round(precision, 4)
        metrics["recall"][pred_name] = round(recall, 4)
        metrics["f1"][pred_name] = round(f1, 4)

        total_correct += correct
        total_samples += total
        f1_scores.append(f1)

    metrics["macro_f1"] = round(np.mean(f1_scores), 4)
    metrics["micro_accuracy"] = round(total_correct / total_samples, 4) if total_samples > 0 else 0

    return metrics


def benchmark_relational_gnn(
    data_list: list[dict],
    model: RelationalGNN,
    device: str,
    use_vision: bool = True,
) -> BenchmarkResult:
    """Benchmark RelationalGNN (Option A).

    Option A uses geometric fusion: detection → depth → 3D → graph → GNN
    """
    model.eval()
    result = BenchmarkResult(
        model_name="RelationalGNN (Option A)",
        total_samples=len(data_list),
    )

    # Model size
    result.model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    all_predictions = []
    all_targets = []
    latencies = []
    detection_times = []
    depth_times = []
    graph_times = []
    inference_times = []

    # Warmup
    for _ in range(5):
        sample = data_list[0]
        graph = sample["graph"].to(device)
        with torch.no_grad():
            _ = model(graph)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for sample in tqdm(data_list, desc="Benchmarking RelationalGNN"):
        total_start = time.perf_counter()

        # Detection and depth already done in data preparation
        t0 = time.perf_counter()
        detection_times.append(sample.get("detection_time", 0))
        depth_times.append(sample.get("depth_time", 0))

        # Graph is pre-built
        t0 = time.perf_counter()
        graph = sample["graph"].to(device)
        graph_times.append((time.perf_counter() - t0) * 1000)

        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(graph)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.perf_counter() - t0) * 1000)

        latencies.append((time.perf_counter() - total_start) * 1000)

        # Collect predictions and targets
        if hasattr(graph, "y") and graph.y is not None:
            all_predictions.append(output["predicate_logits"].cpu())
            all_targets.append(graph.y.cpu())

    # Memory
    if device == "cuda":
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    # Latency stats
    latencies = np.array(latencies)
    result.latency_mean = round(latencies.mean(), 2)
    result.latency_p50 = round(np.percentile(latencies, 50), 2)
    result.latency_p95 = round(np.percentile(latencies, 95), 2)
    result.latency_p99 = round(np.percentile(latencies, 99), 2)

    result.detection_time = round(np.mean(detection_times), 2) if detection_times else 0
    result.depth_time = round(np.mean(depth_times), 2) if depth_times else 0
    result.graph_time = round(np.mean(graph_times), 2)
    result.inference_time = round(np.mean(inference_times), 2)

    # Accuracy metrics
    if all_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(predictions, targets)

        result.accuracy_per_predicate = metrics["accuracy"]
        result.precision_per_predicate = metrics["precision"]
        result.recall_per_predicate = metrics["recall"]
        result.f1_per_predicate = metrics["f1"]
        result.macro_f1 = metrics["macro_f1"]
        result.micro_accuracy = metrics["micro_accuracy"]

    return result


def benchmark_multimodal_gnn(
    data_list: list[dict],
    model: MultiModalGNN,
    device: str,
) -> BenchmarkResult:
    """Benchmark MultiModalGNN (Option C).

    Option C uses learned fusion: detection → DINOv2 → cross-attention → GNN
    """
    model.eval()
    result = BenchmarkResult(
        model_name="MultiModalGNN (Option C)",
        total_samples=len(data_list),
    )

    # Model size
    result.model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    all_predictions = []
    all_targets = []
    latencies = []
    detection_times = []
    vision_times = []
    graph_times = []
    inference_times = []

    # Warmup
    for _ in range(3):
        sample = data_list[0]
        graph = sample["graph"].to(device)
        image = sample["image"].unsqueeze(0).to(device)
        bboxes = sample["bboxes"]
        with torch.no_grad():
            _ = model(graph, image, bboxes)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for sample in tqdm(data_list, desc="Benchmarking MultiModalGNN"):
        total_start = time.perf_counter()

        # Detection time (from data prep)
        detection_times.append(sample.get("detection_time", 0))

        # Graph preparation
        t0 = time.perf_counter()
        graph = sample["graph"].to(device)
        image = sample["image"].unsqueeze(0).to(device)
        bboxes = sample["bboxes"]
        graph_times.append((time.perf_counter() - t0) * 1000)

        # Inference (includes vision encoding)
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(graph, image, bboxes)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.perf_counter() - t0) * 1000)

        latencies.append((time.perf_counter() - total_start) * 1000)

        # Collect predictions and targets
        if hasattr(graph, "y") and graph.y is not None:
            all_predictions.append(output["predicate_logits"].cpu())
            all_targets.append(graph.y.cpu())

    # Memory
    if device == "cuda":
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    # Latency stats
    latencies = np.array(latencies)
    result.latency_mean = round(latencies.mean(), 2)
    result.latency_p50 = round(np.percentile(latencies, 50), 2)
    result.latency_p95 = round(np.percentile(latencies, 95), 2)
    result.latency_p99 = round(np.percentile(latencies, 99), 2)

    result.detection_time = round(np.mean(detection_times), 2) if detection_times else 0
    result.depth_time = 0  # MultiModal doesn't use depth estimation
    result.graph_time = round(np.mean(graph_times), 2)
    result.inference_time = round(np.mean(inference_times), 2)

    # Accuracy metrics
    if all_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(predictions, targets)

        result.accuracy_per_predicate = metrics["accuracy"]
        result.precision_per_predicate = metrics["precision"]
        result.recall_per_predicate = metrics["recall"]
        result.f1_per_predicate = metrics["f1"]
        result.macro_f1 = metrics["macro_f1"]
        result.micro_accuracy = metrics["micro_accuracy"]

    return result


def create_evaluation_data(
    synthetic: bool = True,
    repo_id: str = "lerobot/aloha_static_coffee",
    num_frames: int = 500,
    holding_ratio: float = 0.30,  # 30% holding frames for evaluation (matches training)
    contacting_ratio: float = 0.10,  # 10% contacting frames
) -> list[dict]:
    """Create evaluation data with realistic interaction scenarios.
    
    Args:
        synthetic: Use synthetic data if True, else load from LeRobot
        repo_id: LeRobot dataset ID (if not synthetic)
        num_frames: Number of evaluation samples
        holding_ratio: Fraction of holding scenarios
        contacting_ratio: Fraction of contacting scenarios
    """
    from gnn_reasoner.camera import Object3D
    
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    detector = MockVisionDetector()
    depth_estimator = MockDepthEstimator()
    intrinsics = CameraIntrinsics.default_aloha()

    data_list = []
    prev_graph = None
    
    # Calculate scenario counts
    num_holding = int(num_frames * holding_ratio)
    num_contacting = int(num_frames * contacting_ratio)

    if synthetic:
        logger.info(f"Creating {num_frames} synthetic samples...")
        logger.info(f"  Holding: {num_holding}, Contacting: {num_contacting}")
        
        for i in tqdm(range(num_frames), desc="Generating data"):
            # Determine scenario type (same distribution as training)
            if i < num_holding:
                scenario = "holding"
            elif i < num_holding + num_contacting:
                scenario = "contacting"
            else:
                scenario = "normal"
            
            state = torch.randn(14) * 0.5
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            detection_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            depth_map = depth_estimator.estimate(image)
            depth_time = (time.perf_counter() - t0) * 1000
            
            # Build base graph to get gripper positions
            base_graph = transformer.to_graph(state)
            left_gripper_pos, right_gripper_pos = transformer.get_gripper_positions(base_graph)
            
            if scenario == "holding":
                # Object at gripper, gripper closed
                gripper_state = np.random.uniform(0.0, 0.25)
                if left_gripper_pos is not None and np.random.random() < 0.5:
                    gripper_pos = left_gripper_pos.numpy()
                elif right_gripper_pos is not None:
                    gripper_pos = right_gripper_pos.numpy()
                else:
                    gripper_pos = np.array([0.0, 0.0, 0.0])
                
                offset = np.random.uniform(-0.03, 0.03, size=3)
                object_pos = tuple(gripper_pos + offset)
                
                objects_3d = [Object3D(
                    class_name="held_cup",
                    confidence=np.random.uniform(0.9, 0.99),
                    position=object_pos,
                    size=(0.05, 0.08),
                    bbox=(300, 200, 360, 280),
                )]
                bboxes = [objects_3d[0].bbox]
                
            elif scenario == "contacting":
                # Object close, gripper open
                gripper_state = np.random.uniform(0.5, 1.0)
                if left_gripper_pos is not None and np.random.random() < 0.5:
                    gripper_pos = left_gripper_pos.numpy()
                elif right_gripper_pos is not None:
                    gripper_pos = right_gripper_pos.numpy()
                else:
                    gripper_pos = np.array([0.0, 0.0, 0.0])
                
                offset = np.random.uniform(-0.04, 0.04, size=3)
                object_pos = tuple(gripper_pos + offset)
                
                objects_3d = [Object3D(
                    class_name="contact_mug",
                    confidence=np.random.uniform(0.85, 0.95),
                    position=object_pos,
                    size=(0.06, 0.09),
                    bbox=(310, 210, 370, 290),
                )]
                bboxes = [objects_3d[0].bbox]
                
            else:
                # Normal: random objects
                gripper_state = np.random.uniform(0, 1)
                detections = detector.detect(image, prompts=["cup", "mug", "bottle"])
                objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)
                bboxes = [det.bbox for det in detections]

            graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state)
            graph = add_predicate_labels(graph, prev_graph)

            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            data_list.append({
                "graph": graph,
                "image": image_tensor,
                "bboxes": bboxes,
                "detection_time": detection_time,
                "depth_time": depth_time,
            })

            prev_graph = graph
    else:
        logger.info(f"Loading {num_frames} frames from {repo_id}...")
        dm = DataManager(repo_id)
        num_frames = min(len(dm), num_frames)

        for idx in tqdm(range(num_frames), desc="Loading data"):
            frame = dm.get_frame(idx)
            state = frame.get("observation.state")
            if state is None:
                continue

            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)

            image = dm.get_image(idx)
            if image is None:
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            detections = detector.detect(image, prompts=["cup", "mug", "coffee"])
            detection_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            depth_map = depth_estimator.estimate(image)
            depth_time = (time.perf_counter() - t0) * 1000

            objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)
            gripper_state = state[6].item() if len(state) > 6 else 0.5
            graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state)
            graph = add_predicate_labels(graph, prev_graph)

            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            data_list.append({
                "graph": graph,
                "image": image_tensor,
                "bboxes": [det.bbox for det in detections],
                "detection_time": detection_time,
                "depth_time": depth_time,
            })

            prev_graph = graph

    return data_list


def print_comparison(result_a: BenchmarkResult, result_c: BenchmarkResult):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Option A':<18} {'Option C':<18}")
    print("-" * 70)

    # Overall metrics
    print(f"{'Micro Accuracy':<30} {result_a.micro_accuracy:<18.4f} {result_c.micro_accuracy:<18.4f}")
    print(f"{'Macro F1':<30} {result_a.macro_f1:<18.4f} {result_c.macro_f1:<18.4f}")

    print(f"\n{'Latency (ms)':<30}")
    print(f"{'  Mean':<30} {result_a.latency_mean:<18.2f} {result_c.latency_mean:<18.2f}")
    print(f"{'  P50':<30} {result_a.latency_p50:<18.2f} {result_c.latency_p50:<18.2f}")
    print(f"{'  P95':<30} {result_a.latency_p95:<18.2f} {result_c.latency_p95:<18.2f}")
    print(f"{'  P99':<30} {result_a.latency_p99:<18.2f} {result_c.latency_p99:<18.2f}")

    print(f"\n{'Memory (MB)':<30}")
    print(f"{'  Model Size':<30} {result_a.model_size_mb:<18.2f} {result_c.model_size_mb:<18.2f}")
    print(f"{'  Peak Usage':<30} {result_a.peak_memory_mb:<18.2f} {result_c.peak_memory_mb:<18.2f}")

    print(f"\n{'Timing Breakdown (ms)':<30}")
    print(f"{'  Detection':<30} {result_a.detection_time:<18.2f} {result_c.detection_time:<18.2f}")
    print(f"{'  Depth/Vision':<30} {result_a.depth_time:<18.2f} {result_c.inference_time - result_a.inference_time:<18.2f}")
    print(f"{'  Graph Build':<30} {result_a.graph_time:<18.2f} {result_c.graph_time:<18.2f}")
    print(f"{'  GNN Inference':<30} {result_a.inference_time:<18.2f} {result_c.inference_time:<18.2f}")

    print(f"\n{'Per-Predicate F1':<30}")
    for pred in ALL_PREDICATES:
        f1_a = result_a.f1_per_predicate.get(pred, 0)
        f1_c = result_c.f1_per_predicate.get(pred, 0)
        better = "←" if f1_a > f1_c else ("→" if f1_c > f1_a else "=")
        print(f"  {pred:<28} {f1_a:<18.4f} {f1_c:<18.4f} {better}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare Option A vs Option C")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--repo", type=str, default="lerobot/aloha_static_coffee")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--output", type=str, default="experiments/comparison")
    parser.add_argument("--model-a", type=str, default=None, help="Path to RelationalGNN checkpoint")
    parser.add_argument("--model-c", type=str, default=None, help="Path to MultiModalGNN checkpoint")
    parser.add_argument("--use-mock-vision", action="store_true", help="Use mock vision encoder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluation data
    data_list = create_evaluation_data(
        synthetic=args.synthetic,
        repo_id=args.repo,
        num_frames=args.frames,
    )

    # Load models
    logger.info("Loading RelationalGNN (Option A)...")
    # Use global conditioning (GFC) for interaction predicates like is_holding
    model_a = RelationalGNN(
        hidden_dim=128, 
        num_predicates=9,
        use_global_conditioning=True,
        global_dim=2,
    )
    if args.model_a and Path(args.model_a).exists():
        checkpoint = torch.load(args.model_a, map_location=device, weights_only=False)
        model_a.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded weights from {args.model_a}")
    model_a = model_a.to(device)

    logger.info("Loading MultiModalGNN (Option C)...")
    model_c = MultiModalGNN(hidden_dim=128, num_predicates=9, vision_model="dinov2_vits14")
    if args.model_c and Path(args.model_c).exists():
        checkpoint = torch.load(args.model_c, map_location=device, weights_only=False)
        # Use strict=False to handle mock vs real vision encoder mismatch
        model_c.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded weights from {args.model_c} (strict=False)")
    if args.use_mock_vision:
        # Replace vision encoder with mock AFTER loading weights
        model_c.vision_encoder = MockVisionEncoder(hidden_dim=128)
        logger.info("Using MockVisionEncoder for inference")
    model_c = model_c.to(device)

    # Run benchmarks
    logger.info("\nBenchmarking Option A (RelationalGNN)...")
    result_a = benchmark_relational_gnn(data_list, model_a, device)

    logger.info("\nBenchmarking Option C (MultiModalGNN)...")
    result_c = benchmark_multimodal_gnn(data_list, model_c, device)

    # Print comparison
    print_comparison(result_a, result_c)

    # Save results
    results = {
        "option_a": result_a.to_dict(),
        "option_c": result_c.to_dict(),
        "config": {
            "synthetic": args.synthetic,
            "repo": args.repo,
            "frames": args.frames,
            "device": device,
        },
    }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()

