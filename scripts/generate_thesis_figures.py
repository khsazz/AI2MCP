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
    """Plot predicate type distribution from latest comparison results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Load from latest comparison results (2026-01-08)
    comparison_path = Path("experiments/comparison_final_real/comparison_results.json")
    if comparison_path.exists():
        import json
        with open(comparison_path) as f:
            results = json.load(f)
        # Use per-predicate accuracy as proxy for detection quality
        predicates = list(results["option_a"]["accuracy"]["per_predicate"].keys())
        # Show F1 scores (most informative)
        f1_scores = list(results["option_a"]["accuracy"]["f1_per_predicate"].values())
        
        colors = [COLORS['primary'] if f1 > 0 else '#cccccc' for f1 in f1_scores]
        bars = ax.barh(predicates, f1_scores, color=colors)
        ax.set_xlabel('F1 Score')
        ax.set_title('Per-Predicate F1 Score (RelationalGNN, 500 frames)')
        
        # Add value labels
        for bar, f1 in zip(bars, f1_scores):
            if f1 > 0:
                ax.text(f1 + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{f1:.3f}', va='center', fontsize=10)
        
        ax.set_xlim(0, 1.1)
    else:
        # Fallback: Updated estimates from CONTEXT_DUMP.txt
        predicates = ['is_near', 'is_right_of', 'is_left_of', 'is_above', 'is_below',
                      'is_holding', 'is_contacting', 'is_approaching', 'is_retracting']
        f1_scores = [0.954, 0.968, 0.969, 0.0, 0.0, 0.0, 0.0, 0.182, 0.152]
        
        colors = [COLORS['primary'] if f1 > 0 else '#cccccc' for f1 in f1_scores]
        bars = ax.barh(predicates, f1_scores, color=colors)
        ax.set_xlabel('F1 Score')
        ax.set_title('Per-Predicate F1 Score (RelationalGNN)')
        ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    
    output_path = output_dir / 'predicate_distribution.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_pass_at_k(benchmark: dict, output_dir: Path) -> None:
    """Plot pass@k accuracy metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    accuracy = benchmark.get('accuracy', {})
    if not accuracy:
        print("Warning: No accuracy data in benchmark")
        return
    
    k_values = ['pass@1', 'pass@3', 'pass@5', 'pass@10']
    scores = [accuracy.get(k, 0) * 100 for k in k_values]
    k_labels = ['k=1', 'k=3', 'k=5', 'k=10']
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['success']]
    
    bars = ax.bar(k_labels, scores, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Top-k Predictions')
    ax.set_title('Pass@k Predicate Prediction Accuracy')
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.annotate(f'{score:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add horizontal line at 90%
    ax.axhline(y=90, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    output_path = output_dir / 'pass_at_k.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'pass_at_k.pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_classification_metrics(benchmark: dict, output_dir: Path) -> None:
    """Plot predicate classification metrics (accuracy, precision, recall, F1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    custom = benchmark.get('custom_metrics', {})
    if not custom:
        print("Warning: No custom metrics in benchmark")
        return
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        custom.get('predicate_accuracy', {}).get('mean_ms', 0),
        custom.get('predicate_precision', {}).get('mean_ms', 0),
        custom.get('predicate_recall', {}).get('mean_ms', 0),
        custom.get('predicate_f1', {}).get('mean_ms', 0),
    ]
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['success']]
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Predicate Classification Performance')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'classification_metrics.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'classification_metrics.pdf')
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


def plot_predicate_definitions(output_dir: Path) -> None:
    """Generate visual predicate definitions figure with geometric conditions."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Predicate Definitions: Geometric Conditions', 
            ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
    
    # Two columns: Spatial (left) and Interaction (right)
    # Spatial predicates
    spatial_title = plt.Rectangle((0.02, 0.78), 0.45, 0.08, 
                                   facecolor=COLORS['primary'], edgecolor='black', linewidth=2)
    ax.add_patch(spatial_title)
    ax.text(0.245, 0.82, 'SPATIAL PREDICATES', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    spatial_predicates = [
        ('is_near', 'd(i,j) < 0.2m', 'O---O'),
        ('is_above', 'dz > 0.1m', 'O (top)'),
        ('is_below', 'dz < -0.1m', 'O (bottom)'),
        ('is_left_of', 'dx < -0.05m', '<--O  O'),
        ('is_right_of', 'dx > 0.05m', 'O  O-->'),
    ]
    
    for i, (name, condition, icon) in enumerate(spatial_predicates):
        y = 0.68 - i * 0.12
        # Name box
        name_box = plt.Rectangle((0.02, y-0.04), 0.15, 0.08, 
                                  facecolor='#E3F2FD', edgecolor=COLORS['primary'], linewidth=1)
        ax.add_patch(name_box)
        ax.text(0.095, y, f'is_{name.split("_")[1]}' if '_' in name else name, 
                ha='center', va='center', fontsize=10, fontweight='bold', 
                family='monospace')
        
        # Condition
        ax.text(0.28, y, condition, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor=COLORS['tertiary']))
        
        # Icon/diagram
        ax.text(0.42, y, icon, ha='center', va='center', fontsize=9, family='monospace')
    
    # Interaction predicates
    interact_title = plt.Rectangle((0.52, 0.78), 0.45, 0.08, 
                                    facecolor=COLORS['secondary'], edgecolor='black', linewidth=2)
    ax.add_patch(interact_title)
    ax.text(0.745, 0.82, 'INTERACTION PREDICATES', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    
    interaction_predicates = [
        ('is_holding', 'gripper_closed & d<0.05m', '[G]--O'),
        ('is_contacting', 'd(i,j) < 0.05m', 'O--O'),
        ('is_approaching', 'v.d < 0 (toward)', 'O-->O'),
        ('is_retracting', 'v.d > 0 (away)', 'O<--O'),
    ]
    
    for i, (name, condition, icon) in enumerate(interaction_predicates):
        y = 0.68 - i * 0.12
        # Name box
        name_box = plt.Rectangle((0.52, y-0.04), 0.15, 0.08, 
                                  facecolor='#FCE4EC', edgecolor=COLORS['secondary'], linewidth=1)
        ax.add_patch(name_box)
        ax.text(0.595, y, name.replace('is_', ''), 
                ha='center', va='center', fontsize=10, fontweight='bold',
                family='monospace')
        
        # Condition
        ax.text(0.78, y, condition, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor=COLORS['tertiary']))
        
        # Icon/diagram
        ax.text(0.93, y, icon, ha='center', va='center', fontsize=9, family='monospace')
    
    # Legend/notes at bottom
    notes_box = plt.Rectangle((0.1, 0.02), 0.8, 0.12, 
                               facecolor='#F5F5F5', edgecolor='gray', linewidth=1)
    ax.add_patch(notes_box)
    ax.text(0.5, 0.11, 'Notation: d = Euclidean distance, dx/dz = coordinate difference,',
            ha='center', fontsize=10)
    ax.text(0.5, 0.05, 'v = velocity vector, O = node (joint), [G] = gripper',
            ha='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / 'predicate_definitions.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'predicate_definitions.pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gnn_dataflow(output_dir: Path) -> None:
    """Generate GNN data flow diagram showing 14-DoF to graph transformation."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Input: 14-DoF vector
    input_box = plt.Rectangle((0.02, 0.4), 0.15, 0.25, 
                               facecolor=COLORS['primary'], edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(input_box)
    ax.text(0.095, 0.52, "14-DoF\nState Vector", ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(0.095, 0.35, "θ₁...θ₇ (left)\nθ₁...θ₇ (right)", ha='center', 
            fontsize=9, color='gray')
    
    # Arrow
    ax.annotate('', xy=(0.22, 0.52), xytext=(0.17, 0.52),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.195, 0.58, "FK", ha='center', fontsize=9, style='italic')
    
    # Forward Kinematics
    fk_box = plt.Rectangle((0.22, 0.4), 0.15, 0.25, 
                            facecolor=COLORS['secondary'], edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(fk_box)
    ax.text(0.295, 0.52, "Forward\nKinematics", ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # Arrow
    ax.annotate('', xy=(0.42, 0.52), xytext=(0.37, 0.52),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Node Features
    node_box = plt.Rectangle((0.42, 0.35), 0.2, 0.35, 
                              facecolor='#E8E8E8', edgecolor='black', linewidth=2)
    ax.add_patch(node_box)
    ax.text(0.52, 0.6, "16 Nodes", ha='center', fontsize=11, fontweight='bold')
    ax.text(0.52, 0.52, "• 7 joints (left)", ha='center', fontsize=9)
    ax.text(0.52, 0.46, "• 7 joints (right)", ha='center', fontsize=9)
    ax.text(0.52, 0.40, "• 2 end-effectors", ha='center', fontsize=9)
    
    # Arrow
    ax.annotate('', xy=(0.67, 0.52), xytext=(0.62, 0.52),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Graph Construction
    graph_box = plt.Rectangle((0.67, 0.35), 0.15, 0.35, 
                               facecolor=COLORS['tertiary'], edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(graph_box)
    ax.text(0.745, 0.58, "Graph", ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(0.745, 0.48, "G=(V,E)", ha='center', fontsize=10, color='white')
    ax.text(0.745, 0.40, "~240 edges", ha='center', fontsize=9, color='white')
    
    # Arrow
    ax.annotate('', xy=(0.87, 0.52), xytext=(0.82, 0.52),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # GNN Output
    out_box = plt.Rectangle((0.87, 0.4), 0.11, 0.25, 
                             facecolor=COLORS['success'], edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(out_box)
    ax.text(0.925, 0.52, "9\nPredicates", ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # Node feature detail box (bottom)
    detail_box = plt.Rectangle((0.25, 0.08), 0.5, 0.18, 
                                facecolor='#F5F5F5', edgecolor='gray', linewidth=1)
    ax.add_patch(detail_box)
    ax.text(0.5, 0.21, "Node Features: xᵢ = [x, y, z, θ, type]", 
            ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.13, "Edge Features: eᵢⱼ = [distance, Δx, Δy, Δz]", 
            ha='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.set_title('RelationalGNN Data Flow: From 14-DoF to Spatial Predicates', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    output_path = output_dir / 'gnn_dataflow.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'gnn_dataflow.pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison_table(output_dir: Path) -> None:
    """Generate comparison table as figure."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    # Load from latest comparison results (2026-01-08)
    comparison_path = Path("experiments/comparison_final_real/comparison_results.json")
    if comparison_path.exists():
        import json
        with open(comparison_path) as f:
            results = json.load(f)
        a = results["option_a"]
        c = results["option_c"]
        
        data = [
            ['Metric', 'RelationalGNN (A)', 'MultiModalGNN (C)', 'Winner'],
            ['Micro Accuracy', f'{a["accuracy"]["micro_accuracy"]*100:.1f}%', 
             f'{c["accuracy"]["micro_accuracy"]*100:.1f}%', 'A ✓'],
            ['Macro F1', f'{a["accuracy"]["macro_f1"]:.3f}', 
             f'{c["accuracy"]["macro_f1"]:.3f}', 'A ✓'],
            ['is_near F1', f'{a["accuracy"]["f1_per_predicate"]["is_near"]:.3f}', 
             f'{c["accuracy"]["f1_per_predicate"]["is_near"]:.3f}', 'A ✓'],
            ['Latency (mean)', f'{a["latency_ms"]["mean"]:.2f}ms', 
             f'{c["latency_ms"]["mean"]:.2f}ms', 'A (16×)'],
            ['Memory (peak)', f'{a["memory_mb"]["peak"]:.1f}MB', 
             f'{c["memory_mb"]["peak"]:.1f}MB', 'A (7×)'],
            ['Model Size', f'{a["memory_mb"]["model_size"]:.2f}MB', 
             f'{c["memory_mb"]["model_size"]:.2f}MB', 'A (2.6×)'],
            ['Training (55k)', '29 min', '31 min', '—'],
        ]
    else:
        # Fallback with updated values from CONTEXT_DUMP.txt
        data = [
            ['Metric', 'RelationalGNN (A)', 'MultiModalGNN (C)', 'Winner'],
            ['Micro Accuracy', '97.03%', '96.51%', 'A ✓'],
            ['Macro F1', '0.358', '0.348', 'A ✓'],
            ['is_near F1', '0.954', '0.920', 'A ✓'],
            ['Latency (mean)', '1.52ms', '24.29ms', 'A (16×)'],
            ['Memory (peak)', '19.4MB', '141.8MB', 'A (7×)'],
            ['Model Size', '0.81MB', '2.14MB', 'A (2.6×)'],
            ['Training (55k)', '29 min', '31 min', '—'],
        ]
    
    table = ax.table(
        cellText=data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.22, 0.22, 0.25],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.7)
    
    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor(COLORS['primary'])
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Complete System Performance Summary', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add footnote
    ax.text(0.5, -0.08, '* On synthetic test data (200 frames). Training accuracy measured on validation set.',
            ha='center', fontsize=9, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    
    output_path = output_dir / 'comparison_table.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'comparison_table.pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_before_after_comparison(output_dir: Path) -> None:
    """Plot Option A vs Option C comparison (fair comparison, 55k vs 55k)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Load from latest comparison results
    comparison_path = Path("experiments/comparison_final_real/comparison_results.json")
    if comparison_path.exists():
        import json
        with open(comparison_path) as f:
            results = json.load(f)
        a_acc = results["option_a"]["accuracy"]["micro_accuracy"] * 100
        c_acc = results["option_c"]["accuracy"]["micro_accuracy"] * 100
        a_f1 = results["option_a"]["accuracy"]["macro_f1"]
        c_f1 = results["option_c"]["accuracy"]["macro_f1"]
        a_lat = results["option_a"]["latency_ms"]["mean"]
        c_lat = results["option_c"]["latency_ms"]["mean"]
    else:
        a_acc, c_acc = 97.03, 96.51
        a_f1, c_f1 = 0.358, 0.348
        a_lat, c_lat = 1.52, 24.29
    
    # Accuracy comparison
    ax1 = axes[0]
    metrics = ['Micro\nAccuracy (%)', 'Macro F1\n(×100)']
    option_a = [a_acc, a_f1 * 100]
    option_c = [c_acc, c_f1 * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, option_a, width, label='RelationalGNN (A)', 
                     color=COLORS['primary'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, option_c, width, label='MultiModalGNN (C)', 
                     color=COLORS['secondary'], edgecolor='black')
    
    ax1.set_ylabel('Score')
    ax1.set_title('(a) Accuracy Comparison (55k frames)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 110)
    
    # Add value labels
    for bar, val in zip(bars1, option_a):
        ax1.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, option_c):
        ax1.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Latency comparison
    ax2 = axes[1]
    models = ['RelationalGNN\n(A)', 'MultiModalGNN\n(C)']
    latencies = [a_lat, c_lat]
    colors = [COLORS['primary'], COLORS['secondary']]
    
    bars = ax2.bar(models, latencies, color=colors, edgecolor='black')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('(b) Inference Latency')
    
    # Add speedup annotation
    speedup = c_lat / a_lat
    ax2.annotate(f'{speedup:.0f}× faster',
                 xy=(0, a_lat), xytext=(0.5, c_lat * 0.6),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=12, fontweight='bold', ha='center')
    
    # Add value labels
    for bar, val in zip(bars, latencies):
        ax2.annotate(f'{val:.1f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'before_after_comparison.png'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'before_after_comparison.pdf')
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
        default=Path("experiments/remote_training/relational_gnn/training_history.json"),
        help="Path to training history JSON",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("experiments/benchmark_with_trained_model.json"),
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
        plot_pass_at_k(benchmark, args.output)
        plot_classification_metrics(benchmark, args.output)
    
    print("\nGenerating additional figures...")
    plot_predicate_distribution(args.output)
    plot_predicate_definitions(args.output)
    plot_mcp_architecture(args.output)
    plot_gnn_dataflow(args.output)
    plot_comparison_table(args.output)
    plot_before_after_comparison(args.output)
    
    print("\n" + "=" * 50)
    print(f"All figures saved to: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()

