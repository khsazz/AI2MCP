#!/usr/bin/env python3
"""Training script for Forward Dynamics Model (Phase 10.3).

Trains the forward dynamics model for pre-execution simulation.
This enables LLM plan verification before physical robot execution.

Usage:
    # Quick test on 1k frames (RTX 500)
    python scripts/train_forward_model.py --max-frames 1000 --epochs 20

    # Full training (RTX 3070 recommended)
    python scripts/train_forward_model.py --repo lerobot/aloha_static_coffee --epochs 100

    # With pre-trained GNN encoder
    python scripts/train_forward_model.py --gnn-checkpoint experiments/aloha_training/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_reasoner.data_manager import DataManager
from gnn_reasoner.lerobot_transformer import LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN
from gnn_reasoner.model.forward_dynamics import (
    ForwardDynamicsModel,
    ForwardDynamicsLoss,
)
from gnn_reasoner.model.relational_gnn import RelationalGNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    repo_id: str = "lerobot/aloha_static_coffee"
    max_frames: int | None = None
    val_split: float = 0.1
    sequence_length: int = 2  # (state_t, action_t, state_{t+1})

    # Model
    hidden_dim: int = 128
    action_dim: int = 14
    num_nodes: int = 16
    dropout: float = 0.1
    freeze_encoder: bool = True

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    accumulation_steps: int = 1
    use_amp: bool = True

    # Loss weights
    delta_weight: float = 1.0
    uncertainty_weight: float = 0.1
    feasibility_weight: float = 0.5
    consistency_weight: float = 0.1

    # Checkpoints
    output_dir: str = "experiments/forward_dynamics"
    gnn_checkpoint: str | None = None

    # Hardware
    device: str = "auto"


class TransitionDataset(Dataset):
    """Dataset of state transitions for forward dynamics training.
    
    Each sample is a (state_t, action_t, state_{t+1}) tuple.
    """

    def __init__(
        self,
        data_manager: DataManager,
        graph_transformer: LeRobotGraphTransformer,
        indices: list[int],
    ):
        self.data_manager = data_manager
        self.graph_transformer = graph_transformer
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        frame_idx = self.indices[idx]

        # Get current frame
        current_frame = self.data_manager.get_frame(frame_idx)
        current_state = current_frame.get("observation.state")
        action = current_frame.get("action")

        # Get next frame
        next_frame = self.data_manager.get_frame(frame_idx + 1)
        next_state = next_frame.get("observation.state")

        # Convert to tensors
        if not isinstance(current_state, torch.Tensor):
            current_state = torch.tensor(current_state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32)

        # Build graphs
        current_graph = self.graph_transformer.to_graph(current_state)
        next_graph = self.graph_transformer.to_graph(next_state)

        # Compute ground truth delta
        current_positions = current_graph.x[:, 2:5]
        next_positions = next_graph.x[:, 2:5]
        delta = next_positions - current_positions

        return {
            "current_graph": current_graph,
            "action": action,
            "target_delta": delta,
            "frame_idx": frame_idx,
        }


def collate_transitions(batch: list[dict]) -> dict:
    """Collate function for transition dataset."""
    from torch_geometric.data import Batch

    current_graphs = [item["current_graph"] for item in batch]
    actions = torch.stack([item["action"] for item in batch])
    target_deltas = torch.stack([item["target_delta"] for item in batch])
    frame_indices = [item["frame_idx"] for item in batch]

    return {
        "current_graph": Batch.from_data_list(current_graphs),
        "action": actions,
        "target_delta": target_deltas,
        "frame_idx": frame_indices,
    }


def load_gnn_encoder(checkpoint_path: str, device: torch.device) -> RelationalGNN:
    """Load pre-trained GNN encoder."""
    logger.info(f"Loading GNN encoder from {checkpoint_path}")

    model = RelationalGNN(
        node_input_dim=5,
        edge_input_dim=2,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        num_predicates=9,
        use_global_conditioning=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded GNN encoder from epoch {checkpoint.get('epoch', '?')}")
    return model


def train_epoch(
    model: ForwardDynamicsModel,
    train_loader: DataLoader,
    criterion: ForwardDynamicsLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    # Keep encoder frozen if specified
    if config.freeze_encoder and model._use_external_encoder:
        model.gnn_encoder.eval()

    total_loss = 0.0
    loss_components = {"delta": 0.0, "uncertainty": 0.0, "feasibility": 0.0}
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        current_graph = batch["current_graph"].to(device)
        actions = batch["action"].to(device)
        target_deltas = batch["target_delta"].to(device)

        with autocast(enabled=config.use_amp):
            # Forward pass
            predictions = model(current_graph, actions)

            # Compute loss
            losses = criterion(predictions, target_deltas)
            loss = losses["total"] / config.accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Accumulation step
        if (batch_idx + 1) % config.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses["total"].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()},
    }


@torch.no_grad()
def validate(
    model: ForwardDynamicsModel,
    val_loader: DataLoader,
    criterion: ForwardDynamicsLoss,
    device: torch.device,
) -> dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_delta_error = 0.0
    total_confidence = 0.0
    num_batches = 0

    for batch in val_loader:
        current_graph = batch["current_graph"].to(device)
        actions = batch["action"].to(device)
        target_deltas = batch["target_delta"].to(device)

        # Forward pass
        predictions = model(current_graph, actions)

        # Compute loss
        losses = criterion(predictions, target_deltas)

        # Metrics
        pred_delta = predictions["delta"]
        delta_error = (pred_delta - target_deltas).abs().mean()
        confidence = predictions["confidence"].mean()

        total_loss += losses["total"].item()
        total_delta_error += delta_error.item()
        total_confidence += confidence.item()
        num_batches += 1

    return {
        "val_loss": total_loss / num_batches,
        "val_delta_error": total_delta_error / num_batches,
        "val_confidence": total_confidence / num_batches,
    }


def train(config: TrainingConfig) -> dict:
    """Main training function."""
    # Setup device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading dataset: {config.repo_id}")
    data_manager = DataManager(repo_id=config.repo_id)

    # Limit frames if specified
    total_frames = len(data_manager)
    if config.max_frames is not None:
        total_frames = min(total_frames, config.max_frames)
    logger.info(f"Using {total_frames} frames")

    # Create transition indices (need pairs, so exclude last frame)
    all_indices = list(range(total_frames - 1))

    # Train/val split
    val_size = int(len(all_indices) * config.val_split)
    train_indices = all_indices[:-val_size]
    val_indices = all_indices[-val_size:]

    logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Create graph transformer
    graph_transformer = LeRobotGraphTransformer(
        kinematic_chain=ALOHA_KINEMATIC_CHAIN,
        num_joints=16,
    )

    # Create datasets
    train_dataset = TransitionDataset(data_manager, graph_transformer, train_indices)
    val_dataset = TransitionDataset(data_manager, graph_transformer, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_transitions,
        num_workers=0,  # Keep 0 for LeRobot compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_transitions,
        num_workers=0,
    )

    # Load GNN encoder if checkpoint provided
    gnn_encoder = None
    if config.gnn_checkpoint:
        gnn_encoder = load_gnn_encoder(config.gnn_checkpoint, device)

    # Create model
    model = ForwardDynamicsModel(
        gnn_encoder=gnn_encoder,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        num_nodes=config.num_nodes,
        freeze_encoder=config.freeze_encoder,
        dropout=config.dropout,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create loss and optimizer
    criterion = ForwardDynamicsLoss(
        delta_weight=config.delta_weight,
        uncertainty_weight=config.uncertainty_weight,
        feasibility_weight=config.feasibility_weight,
        consistency_weight=config.consistency_weight,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=config.use_amp)

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_delta_error": [],
        "val_confidence": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    logger.info("Starting training...")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_delta_error"].append(val_metrics["val_delta_error"])
        history["val_confidence"].append(val_metrics["val_confidence"])
        history["learning_rate"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Log progress
        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Delta Err: {val_metrics['val_delta_error']:.4f} | "
            f"Conf: {val_metrics['val_confidence']:.3f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_delta_error": val_metrics["val_delta_error"],
                    "config": asdict(config),
                },
                output_dir / "best_model.pt",
            )
            logger.info(f"  → Saved best model (loss: {best_val_loss:.4f})")

    # Save final model
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": history["val_loss"][-1],
            "config": asdict(config),
        },
        output_dir / "final_model.pt",
    )

    total_time = time.time() - start_time

    # Save training history
    training_summary = {
        "config": asdict(config),
        "total_time_seconds": total_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": history["val_loss"][-1],
        "final_delta_error": history["val_delta_error"][-1],
        "final_confidence": history["val_confidence"][-1],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": history,
    }

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")
    logger.info(f"Checkpoints saved to: {output_dir}")

    return training_summary


def main():
    parser = argparse.ArgumentParser(
        description="Train Forward Dynamics Model for Pre-Execution Simulation"
    )

    # Data arguments
    parser.add_argument(
        "--repo",
        type=str,
        default="lerobot/aloha_static_coffee",
        help="LeRobot dataset repository ID",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to use (for quick testing)",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained GNN checkpoint",
    )
    parser.add_argument(
        "--no-freeze-encoder",
        action="store_true",
        help="Don't freeze the GNN encoder",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/forward_dynamics",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Build config
    config = TrainingConfig(
        repo_id=args.repo,
        max_frames=args.max_frames,
        hidden_dim=args.hidden_dim,
        gnn_checkpoint=args.gnn_checkpoint,
        freeze_encoder=not args.no_freeze_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        accumulation_steps=args.accumulation_steps,
        output_dir=args.output,
    )

    # Run training
    train(config)


if __name__ == "__main__":
    main()

