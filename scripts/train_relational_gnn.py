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
    
    if "4080" in gpu_name or "4090" in gpu_name or vram_gb >= 12:
        return "rtx4080"
    elif "500" in gpu_name or "4060" in gpu_name or vram_gb < 6:
        return "rtx500"
    else:
        # Default to conservative profile
        return "rtx500"


# =============================================================================
# Data Loading
# =============================================================================

def create_synthetic_dataset(
    num_samples: int = 1000,
    num_joints: int = 16,
) -> list[Data]:
    """Create synthetic graph dataset for training."""
    from gnn_reasoner.lerobot_transformer import (
        LeRobotGraphTransformer,
        ALOHA_KINEMATIC_CHAIN,
        add_predicate_labels,
    )
    
    print(f"Generating {num_samples} synthetic training samples...")
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    
    torch.manual_seed(42)
    dataset = []
    prev_graph = None
    
    for i in range(num_samples):
        # Generate random state with smooth transitions
        if i == 0:
            state = torch.randn(14) * 0.5
        else:
            # Smooth transition from previous state
            state = dataset[-1].x[:14, 0] + torch.randn(14) * 0.1
        
        graph = transformer.to_graph(state)
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
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(create_dataloader(
        dataset, profile.batch_size, shuffle=True
    )):
        batch = batch.to(device)
        
        # Forward pass with optional AMP
        with torch.amp.autocast('cuda', enabled=profile.use_amp):
            outputs = model(batch)
            
            # BCE loss for multi-label predicate classification
            loss = F.binary_cross_entropy_with_logits(
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
    
    # Model
    model = RelationalGNN(
        hidden_dim=profile.hidden_dim,
        num_layers=profile.num_layers,
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
            model, train_data, optimizer, profile, device, scaler
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

