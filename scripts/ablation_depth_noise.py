#!/usr/bin/env python3
"""
Depth Noise Ablation Study

Evaluates model performance under varying depth estimation noise levels.
Simulates the difference between ground-truth depth (simulation) and 
estimated depth (monocular depth estimation like ZoeDepth).

Usage:
    python scripts/ablation_depth_noise.py --model-a experiments/aloha_training/best_model.pt \
        --model-c experiments/multimodal_aloha/best_model.pt --frames 200
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_depth_noise(depth_map: np.ndarray, sigma: float, noise_type: str = "gaussian") -> np.ndarray:
    """Add noise to depth map to simulate estimation error.
    
    Args:
        depth_map: Ground truth depth map
        sigma: Noise standard deviation (in meters)
        noise_type: 'gaussian', 'uniform', or 'structured'
    
    Returns:
        Noisy depth map
    """
    if sigma == 0:
        return depth_map.copy()
    
    if noise_type == "gaussian":
        noise = np.random.normal(0, sigma, depth_map.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(-sigma * 1.73, sigma * 1.73, depth_map.shape)  # Same std
    elif noise_type == "structured":
        # Simulate structured errors (e.g., depth discontinuities)
        noise = np.random.normal(0, sigma, depth_map.shape)
        # Add some smooth spatial correlation
        try:
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=3)
        except ImportError:
            pass  # Fall back to unfiltered noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    noisy_depth = depth_map + noise
    noisy_depth = np.clip(noisy_depth, 0.1, 10.0)  # Clamp to valid range
    return noisy_depth


def run_ablation(
    model_a_path: str,
    model_c_path: str,
    num_frames: int = 200,
    noise_levels: List[float] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
    device: str = "cuda",
) -> Dict:
    """Run depth noise ablation study."""
    
    from gnn_reasoner import DataManager, LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN
    from gnn_reasoner.lerobot_transformer import compute_heuristic_predicates
    from gnn_reasoner.model import RelationalGNN, MultiModalGNN
    from gnn_reasoner.depth import MockDepthEstimator
    from gnn_reasoner.detector import MockVisionDetector
    from gnn_reasoner.camera import CameraIntrinsics, detections_to_objects_3d
    
    # Load data
    logger.info(f"Loading {num_frames} frames from ALOHA...")
    dm = DataManager("lerobot/aloha_static_coffee")
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    
    # Load models
    logger.info("Loading models...")
    
    # Option A: RelationalGNN
    model_a = RelationalGNN(
        node_input_dim=5,
        edge_input_dim=2,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        num_predicates=9,
    ).to(device)
    
    checkpoint_a = torch.load(model_a_path, map_location=device, weights_only=False)
    model_a.load_state_dict(checkpoint_a["model_state_dict"])
    model_a.eval()
    
    # Option C: MultiModalGNN
    model_c = MultiModalGNN(
        node_input_dim=5,
        edge_input_dim=2,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        num_predicates=9,
        vision_model="dinov2_vits14",
        freeze_vision=True,
    ).to(device)
    
    checkpoint_c = torch.load(model_c_path, map_location=device, weights_only=False)
    # Handle frozen backbone loading
    model_c.load_state_dict(checkpoint_c["model_state_dict"], strict=False)
    model_c.eval()
    
    # Setup vision pipeline
    detector = MockVisionDetector()
    depth_estimator = MockDepthEstimator()
    intrinsics = CameraIntrinsics.default_aloha()
    
    results = {
        "noise_levels": noise_levels,
        "option_a": {"accuracy": [], "f1_is_near": []},
        "option_c": {"accuracy": [], "f1_is_near": []},
    }
    
    # Prepare base data (compute once)
    logger.info("Preparing base data...")
    base_data = []
    indices = np.linspace(0, len(dm) - 1, num_frames, dtype=int)
    
    for idx in tqdm(indices, desc="Loading frames"):
        state = dm.get_state(idx)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # Get image and run detection
        try:
            image = dm.get_image(idx, "cam_high")
        except:
            image = None
        
        if image is not None:
            detections = detector.detect(image)
            base_depth = depth_estimator.estimate(image)
        else:
            detections = []
            base_depth = np.ones((480, 640)) * 0.5  # Default depth
        
        # Get gripper state
        gripper_state = float(state[-1]) if len(state) > 0 else 0.5
        
        base_data.append({
            "state": state,
            "detections": detections,
            "base_depth": base_depth,
            "gripper_state": gripper_state,
            "image": image,
        })
    
    # Run ablation for each noise level
    for sigma in noise_levels:
        logger.info(f"\nEvaluating noise level σ = {sigma:.3f}m...")
        
        correct_a = 0
        correct_c = 0
        total_preds = 0
        
        tp_near_a, fp_near_a, fn_near_a = 0, 0, 0
        tp_near_c, fp_near_c, fn_near_c = 0, 0, 0
        
        for data in tqdm(base_data, desc=f"σ={sigma:.2f}"):
            # Build GT graph with base depth (ground truth positions)
            gt_objects = detections_to_objects_3d(data["detections"], data["base_depth"], intrinsics)
            gt_graph = transformer.to_graph_with_objects(
                data["state"],
                gt_objects,
                gripper_state=data["gripper_state"],
            )
            gt_predicates = compute_heuristic_predicates(gt_graph)
            
            # Create noisy graph by adding noise to node positions
            graph = gt_graph.clone()
            if sigma > 0:
                # Add Gaussian noise to position features (x, y, z = first 3 dims)
                noise = torch.randn_like(graph.x[:, :3]) * sigma
                graph.x = graph.x.clone()
                graph.x[:, :3] = graph.x[:, :3] + noise
            graph = graph.to(device)
            
            # Option A inference
            with torch.no_grad():
                out_a = model_a(graph)
                preds_a = (torch.sigmoid(out_a["predicate_logits"]) > 0.5).cpu().numpy()
            
            # Option C inference
            with torch.no_grad():
                out_c = model_c(graph)
                preds_c = (torch.sigmoid(out_c["predicate_logits"]) > 0.5).cpu().numpy()
            
            gt = gt_predicates.cpu().numpy()
            
            # Compute accuracy
            correct_a += (preds_a == gt).sum()
            correct_c += (preds_c == gt).sum()
            total_preds += gt.size
            
            # Compute F1 for is_near (predicate 0)
            if gt.shape[1] > 0:
                gt_near = gt[:, 0]
                pred_near_a = preds_a[:, 0]
                pred_near_c = preds_c[:, 0]
                
                tp_near_a += ((pred_near_a == 1) & (gt_near == 1)).sum()
                fp_near_a += ((pred_near_a == 1) & (gt_near == 0)).sum()
                fn_near_a += ((pred_near_a == 0) & (gt_near == 1)).sum()
                
                tp_near_c += ((pred_near_c == 1) & (gt_near == 1)).sum()
                fp_near_c += ((pred_near_c == 1) & (gt_near == 0)).sum()
                fn_near_c += ((pred_near_c == 0) & (gt_near == 1)).sum()
        
        # Compute metrics
        acc_a = correct_a / total_preds if total_preds > 0 else 0
        acc_c = correct_c / total_preds if total_preds > 0 else 0
        
        # F1 for is_near
        prec_a = tp_near_a / (tp_near_a + fp_near_a) if (tp_near_a + fp_near_a) > 0 else 0
        rec_a = tp_near_a / (tp_near_a + fn_near_a) if (tp_near_a + fn_near_a) > 0 else 0
        f1_near_a = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0
        
        prec_c = tp_near_c / (tp_near_c + fp_near_c) if (tp_near_c + fp_near_c) > 0 else 0
        rec_c = tp_near_c / (tp_near_c + fn_near_c) if (tp_near_c + fn_near_c) > 0 else 0
        f1_near_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0
        
        results["option_a"]["accuracy"].append(float(acc_a))
        results["option_a"]["f1_is_near"].append(float(f1_near_a))
        results["option_c"]["accuracy"].append(float(acc_c))
        results["option_c"]["f1_is_near"].append(float(f1_near_c))
        
        logger.info(f"  Option A: acc={acc_a:.4f}, F1(is_near)={f1_near_a:.4f}")
        logger.info(f"  Option C: acc={acc_c:.4f}, F1(is_near)={f1_near_c:.4f}")
    
    return results


def generate_figures(results: Dict, output_dir: Path):
    """Generate ablation study figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    noise_levels = results["noise_levels"]
    noise_levels_cm = [s * 100 for s in noise_levels]  # Convert to cm
    
    # Figure 1: Accuracy vs Depth Noise
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels_cm, results["option_a"]["accuracy"], 
            'o-', color='#2ecc71', linewidth=2, markersize=8, label='Option A (Geometric)')
    ax.plot(noise_levels_cm, results["option_c"]["accuracy"], 
            's-', color='#3498db', linewidth=2, markersize=8, label='Option C (MultiModal)')
    ax.set_xlabel('Depth Noise σ (cm)', fontsize=12)
    ax.set_ylabel('Micro Accuracy', fontsize=12)
    ax.set_title('Accuracy Degradation with Depth Noise', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_accuracy_vs_noise.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'ablation_accuracy_vs_noise.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Figure 2: F1(is_near) vs Depth Noise
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels_cm, results["option_a"]["f1_is_near"], 
            'o-', color='#2ecc71', linewidth=2, markersize=8, label='Option A (Geometric)')
    ax.plot(noise_levels_cm, results["option_c"]["f1_is_near"], 
            's-', color='#3498db', linewidth=2, markersize=8, label='Option C (MultiModal)')
    ax.set_xlabel('Depth Noise σ (cm)', fontsize=12)
    ax.set_ylabel('F1 Score (is_near)', fontsize=12)
    ax.set_title('Proximity Detection Degradation with Depth Noise', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_f1_near_vs_noise.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'ablation_f1_near_vs_noise.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Figure 3: Relative degradation
    fig, ax = plt.subplots(figsize=(8, 5))
    
    base_acc_a = results["option_a"]["accuracy"][0]
    base_acc_c = results["option_c"]["accuracy"][0]
    
    rel_deg_a = [(base_acc_a - a) / base_acc_a * 100 for a in results["option_a"]["accuracy"]]
    rel_deg_c = [(base_acc_c - a) / base_acc_c * 100 for a in results["option_c"]["accuracy"]]
    
    ax.plot(noise_levels_cm, rel_deg_a, 
            'o-', color='#2ecc71', linewidth=2, markersize=8, label='Option A (Geometric)')
    ax.plot(noise_levels_cm, rel_deg_c, 
            's-', color='#3498db', linewidth=2, markersize=8, label='Option C (MultiModal)')
    ax.set_xlabel('Depth Noise σ (cm)', fontsize=12)
    ax.set_ylabel('Accuracy Degradation (%)', fontsize=12)
    ax.set_title('Relative Performance Loss vs Depth Noise', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_relative_degradation.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'ablation_relative_degradation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved figures to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Depth noise ablation study")
    parser.add_argument("--model-a", type=str, default="experiments/aloha_training/best_model.pt",
                        help="Path to RelationalGNN checkpoint")
    parser.add_argument("--model-c", type=str, default="experiments/multimodal_aloha/best_model.pt",
                        help="Path to MultiModalGNN checkpoint")
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of frames to evaluate")
    parser.add_argument("--output", type=str, default="experiments/ablation_depth",
                        help="Output directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Run ablation
    results = run_ablation(
        args.model_a,
        args.model_c,
        num_frames=args.frames,
        noise_levels=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
        device=device,
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate figures
    generate_figures(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEPTH NOISE ABLATION RESULTS")
    print("=" * 60)
    print(f"\n{'σ (cm)':<10} {'Acc A':<12} {'Acc C':<12} {'F1(near) A':<12} {'F1(near) C':<12}")
    print("-" * 60)
    for i, sigma in enumerate(results["noise_levels"]):
        print(f"{sigma*100:<10.1f} "
              f"{results['option_a']['accuracy'][i]:<12.4f} "
              f"{results['option_c']['accuracy'][i]:<12.4f} "
              f"{results['option_a']['f1_is_near'][i]:<12.4f} "
              f"{results['option_c']['f1_is_near'][i]:<12.4f}")
    print("=" * 60)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

