#!/usr/bin/env python3
"""Generate thesis figures from training and benchmark data.

Creates publication-ready figures for:
1. Training loss curves
2. Validation accuracy over epochs
3. Inference latency distribution
4. Protocol overhead breakdown
5. Predicate prediction accuracy by type

Usage:
    python scripts/generate_thesis_figures.py
    python scripts/generate_thesis_figures.py --output figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#3B3B3B',
}


def load_training_history(path: Path) -> dict | None:
    """Load training history from JSON file."""
    if not path.exists():
        print(f"Warning: Training history not found at {path}")
        return None
    
    with open(path) as f:
        return json.load(f)


def load_benchmark_results(path: Path) -> dict | None:
    """Load benchmark results from JSON file."""
    if not path.exists():
        print(f"Warning: Benchmark results not found at {path}")
        return None
    
    with open(path) as f:
        return json.load(f)


def plot_training_curves(history: dict, output_dir: Path) -> None:
    """Generate training loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Handle nested structure
    data = history.get('history', history)
    epochs = range(1, len(data['train_loss']) + 1)
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, data['train_loss'], 
             color=COLORS['primary'], linewidth=2, label='Training Loss')
    ax1.plot(epochs, data['val_loss'], 
             color=COLORS['secondary'], linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.set_title('(a) Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.set_xlim(1, len(epochs))
    ax1.set_ylim(0, max(data['train_loss']) * 1.1)
    
    # Find best epoch
    best_epoch = data['val_loss'].index(min(data['val_loss'])) + 1
    ax1.axvline(x=best_epoch, color=COLORS['tertiary'], linestyle='--', 
                alpha=0.7, label=f'Best (epoch {best_epoch})')
    
    # Accuracy curve
    ax2 = axes[1]
    ax2.plot(epochs, [acc * 100 for acc in data['val_accuracy']], 
             color=COLORS['success'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('(b) Predicate Prediction Accuracy')
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(80, 100)
    
    # Add final accuracy annotation
    final_acc = data['val_accuracy'][-1] * 100
    ax2.annotate(f'{final_acc:.1f}%', 
                 xy=(len(epochs), final_acc),
                 xytext=(-30, -20), textcoords='offset points',
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['neutral']))
    
    plt.tight_layout()
    
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'training_curves.pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_learning_rate_schedule(history: dict, output_dir: Path) -> None:
    """Plot learning rate schedule over training."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    data = history.get('history', history)
    epochs = range(1, len(data['learning_rate']) + 1)
    
    ax.plot(epochs, data['learning_rate'], 
            color=COLORS['primary'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Cosine Annealing Learning Rate Schedule')
    ax.set_xlim(1, len(epochs))
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_path = output_dir / 'learning_rate.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_inference_latency(benchmark: dict, output_dir: Path) -> None:
    """Plot inference latency distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    timing = benchmark.get('timing', {})
    
    # Bar chart of different timing components
    ax1 = axes[0]
    components = ['inference_latency', 'graph_construction_time', 'serialization_time']
    labels = ['GNN Inference', 'Graph Construction', 'MCP Serialization']
    means = []
    p95s = []
    
    for comp in components:
        if comp in timing:
            means.append(timing[comp].get('mean_ms', 0))
            p95s.append(timing[comp].get('p95_ms', 0))
        else:
            means.append(0)
            p95s.append(0)
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, means, width, label='Mean', color=COLORS['primary'])
    bars2 = ax1.bar(x + width/2, p95s, width, label='P95', color=COLORS['secondary'])
    
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('(a) Latency Breakdown by Component')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    # Pie chart of overhead
    ax2 = axes[1]
    if means[0] > 0:  # If we have inference latency
        total = sum(means)
        sizes = means
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
        explode = (0.05, 0, 0)
        
        wedges, texts, autotexts = ax2.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10}
        )
        ax2.set_title('(b) Time Distribution')
    
    plt.tight_layout()
    
    output_path = output_dir / 'inference_latency.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'inference_latency.pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_predicate_distribution(output_dir: Path) -> None:
    """Plot predicate type distribution (synthetic data for illustration)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Based on demo output
    predicates = ['is_near', 'is_left_of', 'is_right_of', 'is_above', 'is_below']
    counts = [4658, 2586, 2591, 0, 0]  # From demo run
    
    colors = [COLORS['primary'] if c > 0 else '#cccccc' for c in counts]
    
    bars = ax.barh(predicates, counts, color=colors)
    ax.set_xlabel('Total Detections (100 frames)')
    ax.set_title('Spatial Predicate Distribution on ALOHA Dataset')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(count + 50, bar.get_y() + bar.get_height()/2,
                    f'{count}', va='center', fontsize=10)
    
    ax.set_xlim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    
    output_path = output_dir / 'predicate_distribution.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_mcp_architecture(output_dir: Path) -> None:
    """Generate MCP architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simplified block diagram
    boxes = [
        # (x, y, width, height, label, color)
        (0.1, 0.7, 0.2, 0.15, 'AI Model\n(LLM/VLM)', COLORS['primary']),
        (0.4, 0.7, 0.2, 0.15, 'MCP Client', COLORS['secondary']),
        (0.4, 0.4, 0.2, 0.15, 'MCP Server\n(SSE Transport)', COLORS['secondary']),
        (0.4, 0.1, 0.2, 0.15, 'ROS 2 Bridge', COLORS['tertiary']),
        (0.75, 0.55, 0.15, 0.12, 'GNN\nReasoner', COLORS['success']),
        (0.75, 0.25, 0.15, 0.12, 'LeRobot\nDataset', COLORS['neutral']),
        (0.1, 0.1, 0.2, 0.15, 'Robot\n(Gazebo/Real)', '#666666'),
    ]
    
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, 
                              edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    
    # Arrows
    arrows = [
        # (start_x, start_y, end_x, end_y, label)
        (0.3, 0.775, 0.4, 0.775, 'JSON-RPC'),
        (0.5, 0.7, 0.5, 0.55, ''),
        (0.5, 0.4, 0.5, 0.25, ''),
        (0.4, 0.175, 0.3, 0.175, 'ROS 2'),
        (0.6, 0.475, 0.75, 0.6, 'Predicates'),
        (0.6, 0.45, 0.75, 0.35, 'States'),
    ]
    
    for x1, y1, x2, y2, label in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.03, label, ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('MCP-ROS2 Bridge Architecture', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / 'architecture.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'architecture.pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison_table(output_dir: Path) -> None:
    """Generate comparison table as figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    data = [
        ['Metric', 'Synthetic', 'Real ALOHA', 'Improvement'],
        ['Final Accuracy', '95.9%', '99.4%', '+3.5%'],
        ['Best Val Loss', '0.1086', '0.0232', '4.7×'],
        ['Training Time', '21s', '205s', '~10×'],
        ['Convergence', '~30 epochs', '~20 epochs', 'Faster'],
        ['Inference Latency', '7.7ms', '14.4ms', '+87%*'],
    ]
    
    table = ax.table(
        cellText=data,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.2, 0.2],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor(COLORS['primary'])
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Training Results: Synthetic vs Real Data', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add footnote
    ax.text(0.5, -0.1, '* Includes data loading overhead from dataset',
            ha='center', fontsize=9, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    
    output_path = output_dir / 'comparison_table.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--training-history",
        type=Path,
        default=Path("experiments/aloha_training/training_history.json"),
        help="Path to training history JSON",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("experiments/aloha_training/benchmark.json"),
        help="Path to benchmark results JSON",
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in: {args.output}")
    print("=" * 50)
    
    # Load data
    history = load_training_history(args.training_history)
    benchmark = load_benchmark_results(args.benchmark)
    
    # Generate figures
    if history:
        print("\nGenerating training figures...")
        plot_training_curves(history, args.output)
        plot_learning_rate_schedule(history, args.output)
    
    if benchmark:
        print("\nGenerating benchmark figures...")
        plot_inference_latency(benchmark, args.output)
    
    print("\nGenerating additional figures...")
    plot_predicate_distribution(args.output)
    plot_mcp_architecture(args.output)
    plot_comparison_table(args.output)
    
    print("\n" + "=" * 50)
    print(f"All figures saved to: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()

