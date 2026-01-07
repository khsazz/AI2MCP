#!/usr/bin/env python3
"""Training script for RelationalGNN predicate prediction.

Supports multiple GPU configurations:
- RTX 500 Ada (4GB): Conservative settings, gradient accumulation
- RTX 4080 (16GB): Large batches, faster training

Usage:
    # Auto-detect GPU and use appropriate settings
    python scripts/train_relational_gnn.py --synthetic --epochs 100

    # Force specific profile
    python scripts/train_relational_gnn.py --profile rtx500 --synthetic --epochs 50
    python scripts/train_relational_gnn.py --profile rtx4080 --repo lerobot/aloha_static_coffee

    # Custom settings
    python scripts/train_relational_gnn.py --batch-size 32 --lr 1e-4 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch


# =============================================================================
# GPU Profiles
# =============================================================================

@dataclass
class GPUProfile:
    """Training configuration for a specific GPU."""
    name: str
    batch_size: int
    accumulation_steps: int
    hidden_dim: int
    num_layers: int
    learning_rate: float
    weight_decay: float
    use_amp: bool  # Automatic Mixed Precision
    num_workers: int
    pin_memory: bool


# Pre-defined profiles
GPU_PROFILES = {
    "rtx500": GPUProfile(
        name="RTX 500 Ada (4GB)",
        batch_size=16,
        accumulation_steps=4,  # Effective batch = 64
        hidden_dim=128,
        num_layers=3,
        learning_rate=3e-4,
        weight_decay=1e-4,
        use_amp=True,  # Critical for 4GB
        num_workers=4,
        pin_memory=True,
    ),
    "rtx3070": GPUProfile(
        name="RTX 3070 (8GB)",
        batch_size=32,
        accumulation_steps=2,  # Effective batch = 64
        hidden_dim=128,
        num_layers=3,
        learning_rate=3e-4,
        weight_decay=1e-4,
        use_amp=True,
        num_workers=4,
        pin_memory=True,
    ),
    "rtx4080": GPUProfile(
        name="RTX 4080 (16GB)",
        batch_size=64,
        accumulation_steps=1,
        hidden_dim=128,
        num_layers=3,
        learning_rate=3e-4,
        weight_decay=1e-4,
        use_amp=True,  # Still beneficial for speed
        num_workers=8,
        pin_memory=True,
    ),
    "cpu": GPUProfile(
        name="CPU Only",
        batch_size=8,
        accumulation_steps=8,
        hidden_dim=64,
        num_layers=2,
        learning_rate=3e-4,
        weight_decay=1e-4,
        use_amp=False,
        num_workers=4,
        pin_memory=False,
    ),
}


def detect_gpu_profile() -> str:
    """Auto-detect GPU and return appropriate profile name."""
    if not torch.cuda.is_available():
        print("No CUDA GPU detected, using CPU profile")
        return "cpu"
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"Detected GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB)")
    
    # Select profile based on VRAM
    if vram_gb >= 12:
        return "rtx4080"
    elif vram_gb >= 6:
        return "rtx3070"
    else:
        return "rtx500"


# =============================================================================
# Class Imbalance Handling
# =============================================================================

# Predicate indices
HOLDING_IDX = 5
CONTACTING_IDX = 6
APPROACHING_IDX = 7
RETRACTING_IDX = 8


class WeightedFocalLoss(nn.Module):
    """Focal Loss with per-class positive weights.
    
    Addresses class imbalance where interaction predicates are rare.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(inputs.device)
            weight_matrix = pos_weight.unsqueeze(0).expand_as(targets)
            class_weight = torch.where(targets > 0.5, weight_matrix, torch.ones_like(weight_matrix))
            focal_loss = focal_loss * class_weight
        
        return focal_loss.mean()


def compute_class_weights(dataset: list, num_predicates: int = 9) -> torch.Tensor:
    """Compute per-class weights from dataset label distribution."""
    pos_counts = torch.zeros(num_predicates)
    neg_counts = torch.zeros(num_predicates)
    
    for graph in dataset:
        labels = graph.y
        if labels is None or labels.numel() == 0:
            continue
        pos_counts += (labels > 0.5).float().sum(dim=0).cpu()
        neg_counts += (labels <= 0.5).float().sum(dim=0).cpu()
    
    total = pos_counts + neg_counts
    pos_ratio = pos_counts / (total + 1e-8)
    neg_ratio = neg_counts / (total + 1e-8)
    weights = (neg_ratio / (pos_ratio + 1e-8)).clamp(1.0, 50.0)
    
    # Minimum weights for critical predicates
    weights[HOLDING_IDX] = max(weights[HOLDING_IDX].item(), 20.0)
    weights[CONTACTING_IDX] = max(weights[CONTACTING_IDX].item(), 10.0)
    weights[APPROACHING_IDX] = max(weights[APPROACHING_IDX].item(), 5.0)
    weights[RETRACTING_IDX] = max(weights[RETRACTING_IDX].item(), 5.0)
    
    return weights


# =============================================================================
# Data Loading
# =============================================================================

def create_synthetic_dataset(
    num_samples: int = 1000,
    num_joints: int = 16,
    holding_ratio: float = 0.30,  # 30% holding frames (increased for learning)
    contacting_ratio: float = 0.10,  # 10% contacting frames
) -> list[Data]:
    """Create synthetic graph dataset for training with interaction scenarios.
    
    Generates holding/contacting scenarios by placing objects near grippers.
    
    Args:
        num_samples: Total samples to generate
        num_joints: Number of robot joints (ignored, uses ALOHA)
        holding_ratio: Fraction of samples with is_holding=True
        contacting_ratio: Fraction of samples with is_contacting=True
    """
    from gnn_reasoner.lerobot_transformer import (
        LeRobotGraphTransformer,
        ALOHA_KINEMATIC_CHAIN,
        add_predicate_labels,
    )
    from gnn_reasoner.camera import Object3D
    import numpy as np
    
    num_holding = int(num_samples * holding_ratio)
    num_contacting = int(num_samples * contacting_ratio)
    num_normal = num_samples - num_holding - num_contacting
    
    print(f"Generating {num_samples} synthetic training samples...")
    print(f"  Holding: {num_holding}, Contacting: {num_contacting}, Normal: {num_normal}")
    
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = []
    prev_graph = None
    
    for i in range(num_samples):
        # Determine scenario type
        if i < num_holding:
            scenario = "holding"
        elif i < num_holding + num_contacting:
            scenario = "contacting"
        else:
            scenario = "normal"
        
        # Generate random state with smooth transitions
        if i == 0:
            state = torch.randn(14) * 0.5
        else:
            state = dataset[-1].x[:14, 0] + torch.randn(14) * 0.1
        
        # First create base graph to get gripper positions
        base_graph = transformer.to_graph(state)
        left_gripper_pos, right_gripper_pos = transformer.get_gripper_positions(base_graph)
        
        if scenario == "holding":
            # HOLDING: Object at gripper, gripper closed
            gripper_state = float(np.random.uniform(0.0, 0.25))
            
            if left_gripper_pos is not None and np.random.random() < 0.5:
                gripper_pos = left_gripper_pos.numpy()
            elif right_gripper_pos is not None:
                gripper_pos = right_gripper_pos.numpy()
            else:
                gripper_pos = np.array([0.0, 0.0, 0.0])
            
            # Object very close to gripper (within 0.08m holding threshold)
            offset = np.random.uniform(-0.03, 0.03, size=3)
            object_pos = tuple(gripper_pos + offset)
            
            objects_3d = [Object3D(
                class_name="held_cup",
                confidence=np.random.uniform(0.9, 0.99),
                position=object_pos,
                size=(0.05, 0.08),
                bbox=(300, 200, 360, 280),
            )]
            
        elif scenario == "contacting":
            # CONTACTING: Object very close, gripper open
            gripper_state = float(np.random.uniform(0.5, 1.0))
            
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
            
        else:
            # NORMAL: Random object positions, random gripper state
            gripper_state = float(np.random.uniform(0, 1))
            
            # Add 1-2 random objects far from grippers
            num_objs = np.random.randint(1, 3)
            objects_3d = []
            for j in range(num_objs):
                objects_3d.append(Object3D(
                    class_name=f"object_{j}",
                    confidence=np.random.uniform(0.6, 0.95),
                    position=(
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-0.2, 0.2),
                    ),
                    size=(0.05, 0.08),
                    bbox=(np.random.randint(100, 500), np.random.randint(100, 400),
                          np.random.randint(150, 550), np.random.randint(150, 450)),
                ))
        
        # Create graph with objects
        graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state)
        graph = add_predicate_labels(graph, prev_graph)
        dataset.append(graph)
        prev_graph = graph
        
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    return dataset


def create_lerobot_dataset(
    repo_id: str,
    max_samples: int | None = None,
) -> list[Data]:
    """Create dataset from LeRobot trajectories."""
    from gnn_reasoner import DataManager
    from gnn_reasoner.lerobot_transformer import (
        LeRobotGraphTransformer,
        ALOHA_KINEMATIC_CHAIN,
        add_predicate_labels,
    )
    
    print(f"Loading LeRobot dataset: {repo_id}")
    dm = DataManager(repo_id, streaming=True)
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    
    num_samples = min(len(dm), max_samples) if max_samples else len(dm)
    print(f"Processing {num_samples} frames...")
    
    dataset = []
    prev_graph = None
    
    for i in range(num_samples):
        state = dm.get_state(i)
        graph = transformer.to_graph(state)
        graph = add_predicate_labels(graph, prev_graph)
        dataset.append(graph)
        prev_graph = graph
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{num_samples} frames")
    
    return dataset


def create_dataloader(
    dataset: list[Data],
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Batch]:
    """Simple batched iterator over graph dataset."""
    from torch_geometric.data import Batch
    
    indices = list(range(len(dataset)))
    if shuffle:
        import random
        random.shuffle(indices)
    
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_graphs = [dataset[i] for i in batch_indices]
        yield Batch.from_data_list(batch_graphs)


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataset: list[Data],
    optimizer: torch.optim.Optimizer,
    profile: GPUProfile,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    criterion: nn.Module | None = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Default to BCE if no criterion provided
    if criterion is None:
        criterion = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(create_dataloader(
        dataset, profile.batch_size, shuffle=True
    )):
        batch = batch.to(device)
        
        # Forward pass with optional AMP
        with torch.amp.autocast('cuda', enabled=profile.use_amp):
            outputs = model(batch)
            
            # Use provided criterion (WeightedFocalLoss or BCE)
            loss = criterion(
                outputs["predicate_logits"],
                batch.y,
            )
            loss = loss / profile.accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % profile.accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * profile.accumulation_steps
        num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "num_batches": num_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: list[Data],
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    num_batches = 0
    
    for batch in create_dataloader(dataset, batch_size, shuffle=False):
        batch = batch.to(device)
        outputs = model(batch)
        
        # Loss
        loss = F.binary_cross_entropy_with_logits(
            outputs["predicate_logits"],
            batch.y,
        )
        total_loss += loss.item()
        
        # Accuracy (per-predicate)
        preds = (torch.sigmoid(outputs["predicate_logits"]) > 0.5).float()
        total_correct += (preds == batch.y).sum().item()
        total_predictions += batch.y.numel()
        
        num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "accuracy": total_correct / total_predictions if total_predictions > 0 else 0.0,
    }


def train(
    profile: GPUProfile,
    epochs: int,
    dataset: list[Data],
    val_split: float = 0.1,
    save_path: Path | None = None,
    log_interval: int = 10,
) -> dict:
    """Full training loop."""
    from gnn_reasoner.model import RelationalGNN
    
    print(f"\n{'='*60}")
    print(f"Training RelationalGNN")
    print(f"{'='*60}")
    print(f"GPU Profile: {profile.name}")
    print(f"Batch Size: {profile.batch_size} (effective: {profile.batch_size * profile.accumulation_steps})")
    print(f"Hidden Dim: {profile.hidden_dim}")
    print(f"Num Layers: {profile.num_layers}")
    print(f"Learning Rate: {profile.learning_rate}")
    print(f"Mixed Precision: {profile.use_amp}")
    print(f"Epochs: {epochs}")
    print(f"Dataset Size: {len(dataset)}")
    print(f"{'='*60}\n")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_data = dataset[val_size:]
    val_data = dataset[:val_size]
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Compute class weights for handling imbalance
    pos_weight = compute_class_weights(train_data, num_predicates=9)
    print(f"Class weights: {[f'{w:.1f}' for w in pos_weight.tolist()]}")
    
    # Create weighted focal loss criterion
    criterion = WeightedFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    print("Using WeightedFocalLoss (alpha=0.25, gamma=2.0)")
    
    # Model with Global Feature Conditioning (GFC) for is_holding
    model = RelationalGNN(
        hidden_dim=profile.hidden_dim,
        num_layers=profile.num_layers,
        use_global_conditioning=True,
        global_dim=2,  # [left_gripper, right_gripper]
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=profile.learning_rate,
        weight_decay=profile.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if profile.use_amp and device.type == "cuda" else None
    
    # Create output directory early
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    best_val_loss = float("inf")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_data, optimizer, profile, device, scaler, criterion
        )
        
        # Validate
        val_metrics = evaluate(model, val_data, profile.batch_size, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(current_lr)
        
        # Save best model
        if val_metrics["loss"] < best_val_loss and save_path:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "profile": profile.name,
            }, save_path / "best_model.pt")
        
        # Logging
        epoch_time = time.time() - epoch_start
        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
    
    total_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = evaluate(model, val_data, profile.batch_size, device)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time:.1f}s ({total_time/epochs:.2f}s/epoch)")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Final Val Accuracy: {final_metrics['accuracy']:.4f}")
    
    # Save final model and history
    if save_path:
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": final_metrics["loss"],
            "profile": profile.name,
        }, save_path / "final_model.pt")
        
        with open(save_path / "training_history.json", "w") as f:
            json.dump({
                "profile": profile.name,
                "epochs": epochs,
                "total_time_seconds": total_time,
                "best_val_loss": best_val_loss,
                "final_val_accuracy": final_metrics["accuracy"],
                "history": history,
            }, f, indent=2)
        
        print(f"\nModels saved to: {save_path}")
    
    return {
        "total_time": total_time,
        "best_val_loss": best_val_loss,
        "final_accuracy": final_metrics["accuracy"],
        "history": history,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train RelationalGNN for predicate prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for training",
    )
    data_group.add_argument(
        "--repo",
        type=str,
        help="LeRobot dataset repo ID (e.g., lerobot/aloha_static_coffee)",
    )
    
    # GPU profile
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(GPU_PROFILES.keys()),
        default=None,
        help="GPU profile to use (auto-detected if not specified)",
    )
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames from dataset")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    
    # Override profile settings
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dimension")
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/training"),
        help="Output directory for models and logs",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N epochs")
    
    args = parser.parse_args()
    
    # Select GPU profile
    if args.profile:
        profile_name = args.profile
    else:
        profile_name = detect_gpu_profile()
    
    profile = GPU_PROFILES[profile_name]
    
    # Apply overrides
    if args.batch_size:
        profile = GPUProfile(**{**profile.__dict__, "batch_size": args.batch_size})
    if args.lr:
        profile = GPUProfile(**{**profile.__dict__, "learning_rate": args.lr})
    if args.hidden_dim:
        profile = GPUProfile(**{**profile.__dict__, "hidden_dim": args.hidden_dim})
    
    # Create dataset
    if args.synthetic:
        dataset = create_synthetic_dataset(num_samples=args.samples)
    else:
        dataset = create_lerobot_dataset(args.repo, max_samples=args.max_frames)
    
    # Train
    results = train(
        profile=profile,
        epochs=args.epochs,
        dataset=dataset,
        val_split=args.val_split,
        save_path=args.output,
        log_interval=args.log_interval,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

