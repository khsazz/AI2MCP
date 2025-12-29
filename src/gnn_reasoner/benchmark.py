"""Benchmark Logger for GNN and MCP performance metrics.

Provides utilities for measuring and logging:
- Inference latency (GNN forward pass)
- Protocol overhead (MCP serialization/deserialization)
- Pass@k accuracy metrics for action prediction
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class TimingStats:
    """Statistics for a timing metric."""

    name: str
    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if len(self.samples) > 1 else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.samples)) if self.samples else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.samples)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        """95th percentile."""
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0

    @property
    def p99(self) -> float:
        """99th percentile."""
        return float(np.percentile(self.samples, 99)) if self.samples else 0.0

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "std_ms": round(self.std, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "median_ms": round(self.median, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
        }


class BenchmarkLogger:
    """Logger for benchmarking GNN inference and MCP protocol performance.

    Tracks:
    - inference_latency: Time for GNN forward pass
    - serialization_time: Time to serialize results to JSON
    - deserialization_time: Time to deserialize input from JSON
    - total_request_time: End-to-end request handling time
    - pass_at_k: Action prediction accuracy metrics

    Example:
        >>> logger = BenchmarkLogger()
        >>> logger.log_inference_latency(15.3)
        >>> logger.log_protocol_overhead(serialization_ms=2.1, deserialization_ms=1.5)
        >>> metrics = logger.export_metrics()
    """

    def __init__(self, name: str = "gnn_benchmark"):
        """Initialize benchmark logger.

        Args:
            name: Name identifier for this benchmark session
        """
        self.name = name
        self._start_time = time.time()

        # Timing metrics
        self._inference_latency = TimingStats("inference_latency")
        self._serialization_time = TimingStats("serialization_time")
        self._deserialization_time = TimingStats("deserialization_time")
        self._total_request_time = TimingStats("total_request_time")
        self._graph_construction_time = TimingStats("graph_construction_time")

        # Pass@k tracking
        self._predictions: list[tuple[list[Any], list[Any]]] = []  # (predicted, expert)

        # Custom metrics
        self._custom_metrics: dict[str, TimingStats] = {}

    def log_inference_latency(self, ms: float) -> None:
        """Log GNN inference latency in milliseconds."""
        self._inference_latency.samples.append(ms)

    def log_protocol_overhead(
        self,
        serialization_ms: float | None = None,
        deserialization_ms: float | None = None,
    ) -> None:
        """Log MCP protocol serialization/deserialization overhead.

        Args:
            serialization_ms: Time to serialize response to JSON
            deserialization_ms: Time to deserialize request from JSON
        """
        if serialization_ms is not None:
            self._serialization_time.samples.append(serialization_ms)
        if deserialization_ms is not None:
            self._deserialization_time.samples.append(deserialization_ms)

    def log_total_request_time(self, ms: float) -> None:
        """Log total end-to-end request handling time."""
        self._total_request_time.samples.append(ms)

    def log_graph_construction_time(self, ms: float) -> None:
        """Log time to construct graph from state."""
        self._graph_construction_time.samples.append(ms)

    def log_custom_metric(self, name: str, value: float) -> None:
        """Log a custom timing metric.

        Args:
            name: Metric name
            value: Value in milliseconds
        """
        if name not in self._custom_metrics:
            self._custom_metrics[name] = TimingStats(name)
        self._custom_metrics[name].samples.append(value)

    def log_prediction(
        self,
        predicted: list[Any],
        expert: list[Any],
    ) -> None:
        """Log a prediction for pass@k calculation.

        Args:
            predicted: List of predicted actions (ranked by confidence)
            expert: List of expert/ground-truth actions
        """
        self._predictions.append((predicted, expert))

    def compute_pass_at_k(
        self,
        predicted: list[Any],
        expert: list[Any],
        k: int,
    ) -> float:
        """Compute pass@k metric for a single prediction.

        Pass@k = 1 if any of the top-k predictions matches expert action.

        Args:
            predicted: Ranked list of predicted actions
            expert: Ground-truth actions
            k: Number of top predictions to consider

        Returns:
            1.0 if pass, 0.0 otherwise
        """
        top_k = predicted[:k]

        # Check if any top-k prediction matches any expert action
        for pred in top_k:
            if self._actions_match(pred, expert):
                return 1.0
        return 0.0

    def _actions_match(self, pred: Any, experts: list[Any], tolerance: float = 0.1) -> bool:
        """Check if a prediction matches any expert action.

        Args:
            pred: Predicted action
            experts: List of expert actions
            tolerance: Tolerance for numerical comparison

        Returns:
            True if match found
        """
        for expert in experts:
            if self._single_action_match(pred, expert, tolerance):
                return True
        return False

    def _single_action_match(
        self, pred: Any, expert: Any, tolerance: float = 0.1
    ) -> bool:
        """Check if two actions match within tolerance."""
        if isinstance(pred, (list, tuple)) and isinstance(expert, (list, tuple)):
            if len(pred) != len(expert):
                return False
            for p, e in zip(pred, expert):
                if abs(p - e) > tolerance:
                    return False
            return True
        elif isinstance(pred, (int, float)) and isinstance(expert, (int, float)):
            return abs(pred - expert) <= tolerance
        else:
            return pred == expert

    def get_pass_at_k_scores(self, k_values: list[int] | None = None) -> dict[str, float]:
        """Compute pass@k scores for all logged predictions.

        Args:
            k_values: List of k values to compute. Defaults to [1, 3, 5, 10].

        Returns:
            Dictionary mapping "pass@k" to score
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        if not self._predictions:
            return {f"pass@{k}": 0.0 for k in k_values}

        scores = {}
        for k in k_values:
            total = 0.0
            for predicted, expert in self._predictions:
                total += self.compute_pass_at_k(predicted, expert, k)
            scores[f"pass@{k}"] = round(total / len(self._predictions), 4)

        return scores

    def time_function(self, metric_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to time a function and log to a metric.

        Args:
            metric_name: Name of the metric to log to

        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                self.log_custom_metric(metric_name, elapsed_ms)
                return result
            return wrapper
        return decorator

    def export_metrics(self) -> dict:
        """Export all metrics as a dictionary.

        Returns:
            Dictionary with all timing stats and pass@k scores
        """
        elapsed_time = time.time() - self._start_time

        metrics = {
            "name": self.name,
            "session_duration_seconds": round(elapsed_time, 2),
            "timing": {
                "inference_latency": self._inference_latency.to_dict(),
                "serialization_time": self._serialization_time.to_dict(),
                "deserialization_time": self._deserialization_time.to_dict(),
                "total_request_time": self._total_request_time.to_dict(),
                "graph_construction_time": self._graph_construction_time.to_dict(),
            },
            "custom_metrics": {
                name: stats.to_dict() for name, stats in self._custom_metrics.items()
            },
            "accuracy": self.get_pass_at_k_scores(),
            "total_predictions": len(self._predictions),
        }

        # Compute protocol overhead as percentage
        if (
            self._total_request_time.count > 0
            and self._inference_latency.count > 0
        ):
            total_mean = self._total_request_time.mean
            inference_mean = self._inference_latency.mean
            if total_mean > 0:
                overhead_pct = ((total_mean - inference_mean) / total_mean) * 100
                metrics["protocol_overhead_percent"] = round(overhead_pct, 2)

        return metrics

    def save_metrics(self, path: str | Path) -> None:
        """Save metrics to a JSON file.

        Args:
            path: File path to save metrics
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.export_metrics(), f, indent=2)

    def reset(self) -> None:
        """Reset all metrics."""
        self._start_time = time.time()
        self._inference_latency = TimingStats("inference_latency")
        self._serialization_time = TimingStats("serialization_time")
        self._deserialization_time = TimingStats("deserialization_time")
        self._total_request_time = TimingStats("total_request_time")
        self._graph_construction_time = TimingStats("graph_construction_time")
        self._predictions = []
        self._custom_metrics = {}

    def summary(self) -> str:
        """Generate a human-readable summary of metrics.

        Returns:
            Formatted string summary
        """
        metrics = self.export_metrics()

        lines = [
            f"=== Benchmark: {self.name} ===",
            f"Duration: {metrics['session_duration_seconds']}s",
            "",
            "Timing Metrics (ms):",
        ]

        for name, stats in metrics["timing"].items():
            if stats["count"] > 0:
                lines.append(
                    f"  {name}: mean={stats['mean_ms']:.2f}, "
                    f"p95={stats['p95_ms']:.2f}, n={stats['count']}"
                )

        if metrics.get("protocol_overhead_percent"):
            lines.append(f"\nProtocol Overhead: {metrics['protocol_overhead_percent']}%")

        if metrics["total_predictions"] > 0:
            lines.append("\nAccuracy Metrics:")
            for k, score in metrics["accuracy"].items():
                lines.append(f"  {k}: {score:.4f}")

        return "\n".join(lines)


# Global benchmark logger instance
_global_logger: BenchmarkLogger | None = None


def get_benchmark_logger() -> BenchmarkLogger:
    """Get or create the global benchmark logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = BenchmarkLogger()
    return _global_logger


def reset_benchmark_logger() -> None:
    """Reset the global benchmark logger."""
    global _global_logger
    _global_logger = None

