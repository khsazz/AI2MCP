#!/usr/bin/env python3
"""Generate thesis figures comparing Option A vs Option C.

Creates publication-ready figures:
1. Accuracy comparison (bar chart)
2. Latency breakdown (stacked bar)
3. Per-predicate F1 heatmap
4. Memory usage comparison
5. Radar chart of overall performance

Usage:
    python scripts/generate_comparison_figures.py
    python scripts/generate_comparison_figures.py --results experiments/comparison/comparison_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration for thesis figures
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color scheme
COLORS = {
    "option_a": "#2E86AB",  # Blue
    "option_c": "#A23B72",  # Magenta
    "accent": "#F18F01",    # Orange
    "neutral": "#C73E1D",   # Red
}


def load_results(results_path: Path) -> dict:
    """Load comparison results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def save_figure(fig, output_dir: Path, name: str, formats: list[str]) -> None:
    """Save figure in specified formats."""
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"Saved: {path}")


def plot_accuracy_comparison(results: dict, output_dir: Path, formats: list[str]):
    """Create accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Micro Accuracy", "Macro F1"]
    option_a = [
        results["option_a"]["accuracy"]["micro_accuracy"],
        results["option_a"]["accuracy"]["macro_f1"],
    ]
    option_c = [
        results["option_c"]["accuracy"]["micro_accuracy"],
        results["option_c"]["accuracy"]["macro_f1"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, option_a, width, label="Option A (Geometric)", color=COLORS["option_a"])
    bars2 = ax.bar(x + width/2, option_c, width, label="Option C (Multi-Modal)", color=COLORS["option_c"])

    ax.set_ylabel("Score")
    ax.set_title("Predicate Detection Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, "accuracy_comparison", formats)
    plt.close()


def plot_latency_breakdown(results: dict, output_dir: Path, formats: list[str]):
    """Create latency breakdown stacked bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["Option A\n(Geometric)", "Option C\n(Multi-Modal)"]

    # Timing components
    detection = [
        results["option_a"]["timing_breakdown_ms"]["detection"],
        results["option_c"]["timing_breakdown_ms"]["detection"],
    ]
    depth = [
        results["option_a"]["timing_breakdown_ms"]["depth"],
        0,  # Option C doesn't use depth
    ]
    graph = [
        results["option_a"]["timing_breakdown_ms"]["graph"],
        results["option_c"]["timing_breakdown_ms"]["graph"],
    ]
    inference = [
        results["option_a"]["timing_breakdown_ms"]["inference"],
        results["option_c"]["timing_breakdown_ms"]["inference"],
    ]

    x = np.arange(len(models))
    width = 0.5

    ax.bar(x, detection, width, label="Detection", color="#4ECDC4")
    ax.bar(x, depth, width, bottom=detection, label="Depth Est.", color="#45B7D1")
    ax.bar(x, graph, width, bottom=np.array(detection) + np.array(depth), label="Graph Build", color="#96CEB4")
    ax.bar(x, inference, width, bottom=np.array(detection) + np.array(depth) + np.array(graph), label="GNN Inference", color="#FFEAA7")

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper right")

    # Add total labels
    totals = [
        results["option_a"]["latency_ms"]["mean"],
        results["option_c"]["latency_ms"]["mean"],
    ]
    for i, total in enumerate(totals):
        ax.annotate(f'{total:.1f}ms',
                    xy=(i, total),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_dir, "latency_breakdown", formats)
    plt.close()


def plot_per_predicate_f1(results: dict, output_dir: Path, formats: list[str]):
    """Create per-predicate F1 comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    predicates = list(results["option_a"]["accuracy"]["f1_per_predicate"].keys())
    f1_a = [results["option_a"]["accuracy"]["f1_per_predicate"][p] for p in predicates]
    f1_c = [results["option_c"]["accuracy"]["f1_per_predicate"][p] for p in predicates]

    x = np.arange(len(predicates))
    width = 0.35

    bars1 = ax.bar(x - width/2, f1_a, width, label="Option A (Geometric)", color=COLORS["option_a"])
    bars2 = ax.bar(x + width/2, f1_c, width, label="Option C (Multi-Modal)", color=COLORS["option_c"])

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Predicate F1 Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("is_", "") for p in predicates], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    save_figure(fig, output_dir, "per_predicate_f1", formats)
    plt.close()


def plot_memory_comparison(results: dict, output_dir: Path, formats: list[str]):
    """Create memory usage comparison chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Model Size", "Peak Memory"]
    option_a = [
        results["option_a"]["memory_mb"]["model_size"],
        results["option_a"]["memory_mb"]["peak"],
    ]
    option_c = [
        results["option_c"]["memory_mb"]["model_size"],
        results["option_c"]["memory_mb"]["peak"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, option_a, width, label="Option A (Geometric)", color=COLORS["option_a"])
    bars2 = ax.bar(x + width/2, option_c, width, label="Option C (Multi-Modal)", color=COLORS["option_c"])

    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, "memory_comparison", formats)
    plt.close()


def plot_radar_chart(results: dict, output_dir: Path, formats: list[str]):
    """Create radar chart comparing overall performance."""
    categories = [
        "Accuracy",
        "Speed\n(1/latency)",
        "Memory\nEfficiency",
        "Spatial\nPredicates",
        "Interaction\nPredicates",
    ]

    # Normalize metrics to 0-1 scale
    max_latency = max(
        results["option_a"]["latency_ms"]["mean"],
        results["option_c"]["latency_ms"]["mean"],
    )
    max_memory = max(
        results["option_a"]["memory_mb"]["peak"],
        results["option_c"]["memory_mb"]["peak"],
    )

    # Compute spatial and interaction F1 averages
    spatial_preds = ["is_near", "is_above", "is_below", "is_left_of", "is_right_of"]
    interaction_preds = ["is_holding", "is_contacting", "is_approaching", "is_retracting"]

    spatial_a = np.mean([results["option_a"]["accuracy"]["f1_per_predicate"].get(p, 0) for p in spatial_preds])
    spatial_c = np.mean([results["option_c"]["accuracy"]["f1_per_predicate"].get(p, 0) for p in spatial_preds])
    interaction_a = np.mean([results["option_a"]["accuracy"]["f1_per_predicate"].get(p, 0) for p in interaction_preds])
    interaction_c = np.mean([results["option_c"]["accuracy"]["f1_per_predicate"].get(p, 0) for p in interaction_preds])

    values_a = [
        results["option_a"]["accuracy"]["micro_accuracy"],
        1 - (results["option_a"]["latency_ms"]["mean"] / max_latency),
        1 - (results["option_a"]["memory_mb"]["peak"] / max_memory),
        spatial_a,
        interaction_a,
    ]

    values_c = [
        results["option_c"]["accuracy"]["micro_accuracy"],
        1 - (results["option_c"]["latency_ms"]["mean"] / max_latency),
        1 - (results["option_c"]["memory_mb"]["peak"] / max_memory),
        spatial_c,
        interaction_c,
    ]

    # Close the radar chart
    values_a += values_a[:1]
    values_c += values_c[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, values_a, 'o-', linewidth=2, label="Option A (Geometric)", color=COLORS["option_a"])
    ax.fill(angles, values_a, alpha=0.25, color=COLORS["option_a"])

    ax.plot(angles, values_c, 's-', linewidth=2, label="Option C (Multi-Modal)", color=COLORS["option_c"])
    ax.fill(angles, values_c, alpha=0.25, color=COLORS["option_c"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Overall Performance Comparison", y=1.08)

    plt.tight_layout()
    save_figure(fig, output_dir, "radar_comparison", formats)
    plt.close()


def plot_latency_distribution(results: dict, output_dir: Path, formats: list[str]):
    """Create latency percentile comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    percentiles = ["Mean", "P50", "P95", "P99"]
    latency_a = [
        results["option_a"]["latency_ms"]["mean"],
        results["option_a"]["latency_ms"]["p50"],
        results["option_a"]["latency_ms"]["p95"],
        results["option_a"]["latency_ms"]["p99"],
    ]
    latency_c = [
        results["option_c"]["latency_ms"]["mean"],
        results["option_c"]["latency_ms"]["p50"],
        results["option_c"]["latency_ms"]["p95"],
        results["option_c"]["latency_ms"]["p99"],
    ]

    x = np.arange(len(percentiles))
    width = 0.35

    ax.bar(x - width/2, latency_a, width, label="Option A (Geometric)", color=COLORS["option_a"])
    ax.bar(x + width/2, latency_c, width, label="Option C (Multi-Modal)", color=COLORS["option_c"])

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, output_dir, "latency_distribution", formats)
    plt.close()


def generate_latex_table(results: dict, output_dir: Path):
    """Generate LaTeX table for thesis."""
    table = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of Option A (Geometric Fusion) vs Option C (Multi-Modal Fusion)}
\label{tab:model_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Option A} & \textbf{Option C} \\
\midrule
\multicolumn{3}{l}{\textit{Accuracy}} \\
Micro Accuracy & %.4f & %.4f \\
Macro F1 & %.4f & %.4f \\
\midrule
\multicolumn{3}{l}{\textit{Latency (ms)}} \\
Mean & %.2f & %.2f \\
P95 & %.2f & %.2f \\
\midrule
\multicolumn{3}{l}{\textit{Memory (MB)}} \\
Model Size & %.2f & %.2f \\
Peak Usage & %.2f & %.2f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        results["option_a"]["accuracy"]["micro_accuracy"],
        results["option_c"]["accuracy"]["micro_accuracy"],
        results["option_a"]["accuracy"]["macro_f1"],
        results["option_c"]["accuracy"]["macro_f1"],
        results["option_a"]["latency_ms"]["mean"],
        results["option_c"]["latency_ms"]["mean"],
        results["option_a"]["latency_ms"]["p95"],
        results["option_c"]["latency_ms"]["p95"],
        results["option_a"]["memory_mb"]["model_size"],
        results["option_c"]["memory_mb"]["model_size"],
        results["option_a"]["memory_mb"]["peak"],
        results["option_c"]["memory_mb"]["peak"],
    )

    with open(output_dir / "comparison_table.tex", "w") as f:
        f.write(table)

    print(f"Saved: comparison_table.tex")


def parse_formats(format_arg: str) -> list[str]:
    """Parse format argument into list of formats."""
    if format_arg == "both":
        return ["png", "pdf"]
    return [format_arg]


def main():
    parser = argparse.ArgumentParser(description="Generate comparison figures")
    parser.add_argument(
        "--results",
        type=str,
        default="experiments/comparison_final_real/comparison_results.json",
        help="Path to comparison results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["png", "pdf", "both"],
        default="png",
        help="Output format: png (default), pdf, or both",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.format)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run compare_models.py first to generate results.")
        return

    results = load_results(results_path)
    print(f"Loaded results from {results_path}")
    print(f"Output format(s): {', '.join(formats)}")

    # Generate all figures
    plot_accuracy_comparison(results, output_dir, formats)
    plot_latency_breakdown(results, output_dir, formats)
    plot_per_predicate_f1(results, output_dir, formats)
    plot_memory_comparison(results, output_dir, formats)
    plot_radar_chart(results, output_dir, formats)
    plot_latency_distribution(results, output_dir, formats)
    generate_latex_table(results, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()

