#!/usr/bin/env python3
"""Demo script for LeRobot GNN-MCP pipeline.

Demonstrates the full pipeline:
1. Load LeRobot dataset (or use synthetic data)
2. Transform states to graphs
3. Run GNN inference for predicate prediction
4. Serialize as MCP-compatible JSON
5. Report benchmark metrics

Usage:
    # With real LeRobot data:
    python scripts/demo_lerobot_pipeline.py --repo lerobot/aloha_static_coffee

    # With synthetic data (no network required):
    python scripts/demo_lerobot_pipeline.py --synthetic

    # Run benchmark on N frames:
    python scripts/demo_lerobot_pipeline.py --synthetic --frames 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def create_synthetic_data_manager(num_frames: int = 100, num_episodes: int = 2):
    """Create a mock DataManager with synthetic data."""

    class SyntheticDataManager:
        """Mock DataManager for testing without LeRobot dependency."""

        def __init__(self, num_frames: int, num_episodes: int):
            self.repo_id = "synthetic/aloha_demo"
            self._num_frames = num_frames
            self._num_episodes = num_episodes
            self._current_frame_idx = 0
            self._frames_per_episode = num_frames // num_episodes

            # Pre-generate consistent synthetic data
            torch.manual_seed(42)
            self._states = torch.randn(num_frames, 14) * 0.5  # 14 joints
            self._actions = torch.randn(num_frames, 14) * 0.1

        def __len__(self) -> int:
            return self._num_frames

        def get_frame(self, idx: int) -> dict:
            self._current_frame_idx = idx
            return {
                "observation.state": self._states[idx],
                "action": self._actions[idx],
                "episode_index": idx // self._frames_per_episode,
                "frame_index": idx,
            }

        def get_current_frame(self) -> dict:
            return self.get_frame(self._current_frame_idx)

        def get_state(self, idx: int) -> torch.Tensor:
            return self._states[idx]

        def get_action(self, idx: int) -> torch.Tensor:
            return self._actions[idx]

        def advance_frame(self) -> dict:
            self._current_frame_idx = min(self._current_frame_idx + 1, len(self) - 1)
            return self.get_current_frame()

        def set_frame_index(self, idx: int) -> None:
            self._current_frame_idx = max(0, min(idx, len(self) - 1))

        @property
        def state_dim(self) -> int:
            return 14

        @property
        def action_dim(self) -> int:
            return 14

        @property
        def num_episodes(self) -> int:
            return self._num_episodes

    return SyntheticDataManager(num_frames, num_episodes)


def run_pipeline(
    data_manager,
    num_frames: int,
    verbose: bool = False,
) -> dict:
    """Run the full GNN inference pipeline.

    Args:
        data_manager: DataManager or SyntheticDataManager instance
        num_frames: Number of frames to process
        verbose: Print per-frame output

    Returns:
        Benchmark metrics dictionary
    """
    from gnn_reasoner.lerobot_transformer import (
        LeRobotGraphTransformer,
        ALOHA_KINEMATIC_CHAIN,
        add_predicate_labels,
    )
    from gnn_reasoner.model import RelationalGNN
    from gnn_reasoner.benchmark import BenchmarkLogger

    print(f"\n{'='*60}")
    print("LeRobot GNN-MCP Pipeline Demo")
    print(f"{'='*60}")
    print(f"Dataset: {data_manager.repo_id}")
    print(f"Frames to process: {num_frames}")
    print(f"{'='*60}\n")

    # Initialize components
    print("[1/4] Initializing components...")
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    gnn = RelationalGNN(hidden_dim=128, num_layers=3)
    benchmark = BenchmarkLogger("lerobot_pipeline")

    print(f"  - Graph Transformer: {transformer.num_joints} joints")
    print(f"  - RelationalGNN: {sum(p.numel() for p in gnn.parameters())} parameters")
    print(f"  - Predicate classes: {gnn.num_predicates}")

    # Process frames
    print(f"\n[2/4] Processing {num_frames} frames...")
    prev_graph = None
    all_contexts = []

    for i in range(min(num_frames, len(data_manager))):
        frame_start = time.perf_counter()

        # Get state
        state = data_manager.get_state(i)

        # Transform to graph
        graph_start = time.perf_counter()
        graph = transformer.to_graph(state)
        graph = add_predicate_labels(graph, prev_graph)
        benchmark.log_graph_construction_time((time.perf_counter() - graph_start) * 1000)

        # GNN inference
        inference_start = time.perf_counter()
        with torch.no_grad():
            context = gnn.to_world_context(graph, threshold=0.5)
        benchmark.log_inference_latency((time.perf_counter() - inference_start) * 1000)

        # Serialize to JSON (simulating MCP response)
        serialize_start = time.perf_counter()
        json_str = json.dumps(context)
        benchmark.log_protocol_overhead(
            serialization_ms=(time.perf_counter() - serialize_start) * 1000
        )

        # Total request time
        benchmark.log_total_request_time((time.perf_counter() - frame_start) * 1000)

        all_contexts.append(context)
        prev_graph = graph

        if verbose and i < 5:  # Print first 5 frames
            print(f"\n  Frame {i}:")
            print(f"    Nodes: {context['num_nodes']}, Edges: {context['num_edges']}")
            print(f"    Spatial predicates: {len(context['spatial_predicates'])}")
            print(f"    Interaction predicates: {len(context['interaction_predicates'])}")

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")

    # Compute aggregate stats
    print(f"\n[3/4] Computing metrics...")

    # Sample predicate statistics
    total_spatial = sum(len(c["spatial_predicates"]) for c in all_contexts)
    total_interaction = sum(len(c["interaction_predicates"]) for c in all_contexts)

    print(f"  - Total spatial predicates detected: {total_spatial}")
    print(f"  - Total interaction predicates detected: {total_interaction}")
    print(f"  - Avg spatial per frame: {total_spatial / len(all_contexts):.1f}")
    print(f"  - Avg interaction per frame: {total_interaction / len(all_contexts):.1f}")

    # Export metrics
    print(f"\n[4/4] Benchmark Results:")
    metrics = benchmark.export_metrics()

    print(benchmark.summary())

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Demo LeRobot GNN-MCP pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="lerobot/aloha_static_coffee",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real LeRobot dataset",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames to process",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-frame details",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save benchmark results to JSON file",
    )
    args = parser.parse_args()

    # Create data manager
    if args.synthetic:
        print("Using synthetic data (no network required)")
        data_manager = create_synthetic_data_manager(
            num_frames=args.frames,
            num_episodes=2,
        )
    else:
        print(f"Loading LeRobot dataset: {args.repo}")
        try:
            from gnn_reasoner import DataManager

            data_manager = DataManager(args.repo, streaming=True)
            print(f"Dataset loaded: {len(data_manager)} frames")
        except ImportError as e:
            print(f"Error: LeRobot not installed. Install with: pip install lerobot")
            print(f"Or use --synthetic flag for testing without LeRobot.")
            print(f"Details: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data...")
            data_manager = create_synthetic_data_manager(num_frames=args.frames)

    # Run pipeline
    metrics = run_pipeline(
        data_manager=data_manager,
        num_frames=args.frames,
        verbose=args.verbose,
    )

    # Save results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Demo Complete")
    print(f"{'='*60}")

    timing = metrics["timing"]
    print(f"Inference latency:     {timing['inference_latency']['mean_ms']:.2f}ms (p95: {timing['inference_latency']['p95_ms']:.2f}ms)")
    print(f"Graph construction:    {timing['graph_construction_time']['mean_ms']:.2f}ms")
    print(f"Serialization:         {timing['serialization_time']['mean_ms']:.2f}ms")
    print(f"Total per frame:       {timing['total_request_time']['mean_ms']:.2f}ms")

    if metrics.get("protocol_overhead_percent"):
        print(f"Protocol overhead:     {metrics['protocol_overhead_percent']:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())

