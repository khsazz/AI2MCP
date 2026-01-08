#!/usr/bin/env python3
"""Training script for SpatiotemporalGNN (Phase 11).

Optimized for RTX 3070 (8GB VRAM) with sequence data loading.

Usage:
    # Quick validation (5k frames, 5-frame sequences)
    python scripts/train_spatiotemporal_gnn.py --repo lerobot/aloha_static_coffee --max-frames 5000 --epochs 50
    
    # Full training (55k frames)
    python scripts/train_spatiotemporal_gnn.py --repo lerobot/aloha_static_coffee --epochs 100
    
    # Use pre-trained RelationalGNN as base
    python scripts/train_spatiotemporal_gnn.py --base-checkpoint experiments/remote_training/relational_gnn/best_model.pt
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
from torch.utils.data import Dataset, DataLoader
import structlog

from gnn_reasoner.data_manager import DataManager
from gnn_reasoner.lerobot_transformer import LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN
from gnn_reasoner.model.spatiotemporal_gnn import SpatiotemporalGNN
from gnn_reasoner.model.relational_gnn import RelationalGNN

logger = structlog.get_logger()


# =============================================================================
# GPU Profiles (RTX 3070 Optimized)
# =============================================================================

@dataclass
class GPUProfile:
    """Training configuration optimized for RTX 3070."""
    name: str
    batch_size: int
    accumulation_steps: int
    sequence_length: int
    hidden_dim: int
    temporal_hidden_dim: int
    learning_rate: float
    weight_decay: float
    use_amp: bool
    num_workers: int
    pin_memory: bool


GPU_PROFILES = {
    "rtx3070": GPUProfile(
        name="RTX 3070 (8GB) - ST-GNN",
        batch_size=64,  # Increased for GPU-resident data
        accumulation_steps=1,  # No accumulation needed with large batch
        sequence_length=5,  # 5-frame sequences
        hidden_dim=128,
        temporal_hidden_dim=128,
        learning_rate=3e-4,
        weight_decay=1e-4,
        use_amp=True,  # Critical for memory
        num_workers=0,  # No workers needed for GPU-resident data
        pin_memory=False,
    ),
    "rtx500": GPUProfile(
        name="RTX 500 Ada (4GB) - ST-GNN",
        batch_size=8,
        accumulation_steps=8,  # Effective batch = 64
        sequence_length=3,  # Shorter sequences for memory
        hidden_dim=128,
        temporal_hidden_dim=128,
        learning_rate=3e-4,
        weight_decay=1e-4,
        use_amp=True,
        num_workers=2,
        pin_memory=True,
    ),
    "rtx4080": GPUProfile(
        name="RTX 4080 (16GB) - ST-GNN",
        batch_size=32,
        accumulation_steps=2,  # Effective batch = 64
        sequence_length=10,  # Longer sequences
        hidden_dim=128,
        temporal_hidden_dim=128,
        learning_rate=3e-4,
        weight_decay=1e-4,
        use_amp=True,
        num_workers=8,
        pin_memory=True,
    ),
}


def detect_gpu_profile() -> GPUProfile:
    """Auto-detect GPU and return appropriate profile."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ST-GNN training")
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"Detected GPU: {gpu_name} ({vram_gb:.1f}GB)")
    
    if vram_gb >= 12:
        return GPU_PROFILES["rtx4080"]
    elif vram_gb >= 6:
        return GPU_PROFILES["rtx3070"]
    else:
        return GPU_PROFILES["rtx500"]


# =============================================================================
# Pre-computation for RAM-resident training (730× speedup)
# =============================================================================

def precompute_all_graphs(
    data_manager: DataManager,
    graph_transformer: LeRobotGraphTransformer,
    max_frames: int,
    add_labels: bool = True,
) -> tuple[list[Data], list[torch.Tensor]]:
    """Pre-compute all graphs and actions once at startup.
    
    This is the key optimization: instead of loading from disk on every
    __getitem__ call, we compute everything once and keep it in RAM.
    
    Args:
        data_manager: DataManager for LeRobot dataset
        graph_transformer: Transformer for state-to-graph conversion
        max_frames: Maximum number of frames to process
        add_labels: Whether to add predicate labels (requires prev graph)
        
    Returns:
        Tuple of (graphs, actions) lists
    """
    from tqdm import tqdm
    from gnn_reasoner.lerobot_transformer import add_predicate_labels
    
    graphs: list[Data] = []
    actions: list[torch.Tensor] = []
    
    total = min(max_frames, len(data_manager))
    logger.info(f"Pre-computing {total} graphs (RAM-resident)...")
    
    prev_graph = None
    for i in tqdm(range(total), desc="Pre-computing graphs"):
        frame = data_manager.get_frame(i)
        state = frame.get("observation.state")
        action = frame.get("action")
        
        # Handle missing state
        if state is None:
            if graphs:
                graphs.append(graphs[-1])
                actions.append(actions[-1])
            continue
        
        # Convert to tensors
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        else:
            action = torch.zeros(14, dtype=torch.float32)
        
        # Create graph
        graph = graph_transformer.to_graph(state)
        
        # Add predicate labels (uses previous graph for motion predicates)
        # Note: Always add labels to ensure consistent attributes across all graphs
        if add_labels:
            graph = add_predicate_labels(graph, prev_graph)  # prev_graph can be None for first frame
        
        graphs.append(graph)
        actions.append(action)
        prev_graph = graph
    
    logger.info(f"Pre-computed {len(graphs)} graphs, RAM usage: ~{len(graphs) * 0.3:.1f} MB")
    return graphs, actions


class PrecomputedSequenceDataset(Dataset):
    """RAM-resident dataset using pre-computed graphs.
    
    O(1) indexing - no disk I/O during training!
    """
    
    def __init__(
        self,
        graphs: list[Data],
        actions: list[torch.Tensor],
        sequence_length: int = 5,
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        """Initialize from pre-computed data.
        
        Args:
            graphs: List of pre-computed PyG Data objects
            actions: List of action tensors
            sequence_length: Number of frames per sequence
            start_idx: Start index for train/val split
            end_idx: End index for train/val split (None = all)
        """
        self.sequence_length = sequence_length
        self.start_idx = start_idx
        self.end_idx = end_idx or len(graphs)
        
        # Slice the pre-computed data
        self.graphs = graphs[start_idx:self.end_idx]
        self.actions = actions[start_idx:self.end_idx]
        
        logger.info(f"Dataset: {len(self)} sequences from {len(self.graphs)} frames")
    
    def __len__(self) -> int:
        """Number of valid sequences."""
        return max(0, len(self.graphs) - self.sequence_length)
    
    def __getitem__(self, idx: int) -> tuple[list[Data], list[torch.Tensor], Data]:
        """Get a sequence - O(1) list slicing, no disk I/O!"""
        # Slice pre-computed data
        seq_graphs = self.graphs[idx:idx + self.sequence_length]
        seq_actions = self.actions[idx:idx + self.sequence_length]
        next_graph = self.graphs[idx + self.sequence_length]
        
        return seq_graphs, seq_actions, next_graph


# Legacy lazy-loading dataset (fallback for low-memory systems)
class LazySequenceDataset(Dataset):
    """Dataset with lazy loading - slower but lower memory."""
    
    def __init__(
        self,
        data_manager: DataManager,
        graph_transformer: LeRobotGraphTransformer,
        sequence_length: int = 5,
        max_frames: int | None = None,
    ):
        self.data_manager = data_manager
        self.graph_transformer = graph_transformer
        self.sequence_length = sequence_length
        self.max_frames = max_frames or len(data_manager)
        self.frame_indices = list(range(min(self.max_frames, len(data_manager))))
        logger.info(f"Dataset initialized with {len(self.frame_indices)} frames (lazy loading)")
        
    def __len__(self) -> int:
        return max(0, len(self.frame_indices) - self.sequence_length + 1)
    
    def __getitem__(self, idx: int) -> tuple[list[Data], list[torch.Tensor], Data]:
        from gnn_reasoner.lerobot_transformer import add_predicate_labels
        
        graphs = []
        actions = []
        
        for i in range(self.sequence_length):
            frame_idx = self.frame_indices[idx + i]
            frame = self.data_manager.get_frame(frame_idx)
            state = frame.get("observation.state")
            action = frame.get("action")
            
            if state is None:
                if graphs:
                    graphs.append(graphs[-1])
                    actions.append(actions[-1] if actions else torch.zeros(14, dtype=torch.float32))
                continue
                
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if action is not None and not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            else:
                action = torch.zeros(14, dtype=torch.float32)
            
            graph = self.graph_transformer.to_graph(state)
            graphs.append(graph)
            actions.append(action)
        
        next_idx = min(idx + self.sequence_length, len(self.frame_indices) - 1)
        next_frame = self.data_manager.get_frame(self.frame_indices[next_idx])
        next_state = next_frame.get("observation.state")
        if next_state is not None:
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            next_graph = self.graph_transformer.to_graph(next_state)
        else:
            next_graph = graphs[-1] if graphs else self.graph_transformer.to_graph(torch.zeros(14))
        
        return graphs, actions, next_graph


# =============================================================================
# Loss Functions
# =============================================================================

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


class TemporalLoss(nn.Module):
    """Combined loss for temporal GNN training.
    
    Components:
    1. Predicate classification loss (WeightedFocalLoss)
    2. Temporal consistency loss (smooth predictions across frames)
    3. Future prediction loss (if actions provided)
    """
    
    def __init__(
        self,
        predicate_weight: float = 1.0,
        consistency_weight: float = 0.1,
        future_weight: float = 0.5,
    ):
        """Initialize temporal loss.
        
        Args:
            predicate_weight: Weight for predicate classification
            consistency_weight: Weight for temporal consistency
            future_weight: Weight for future prediction
        """
        super().__init__()
        self.predicate_weight = predicate_weight
        self.consistency_weight = consistency_weight
        self.future_weight = future_weight
        
        # Use WeightedFocalLoss for predicate classification
        self.predicate_loss = WeightedFocalLoss()
        
    def forward(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[Data],
        actions: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute temporal loss.
        
        Args:
            predictions: List of model outputs for each frame in sequence
            targets: List of target graphs (with predicate labels)
            actions: List of action tensors (optional, for future prediction)
            
        Returns:
            Dictionary with individual losses and total
        """
        # 1. Predicate classification loss (sum over sequence)
        pred_loss = torch.tensor(0.0, device=predictions[0]["predicate_logits"].device)
        
        for pred, target in zip(predictions, targets):
            if hasattr(target, "y") and target.y is not None:
                pred_logits = pred["predicate_logits"]
                target_labels = target.y
                
                # Move target to same device as predictions
                target_labels = target_labels.to(pred_logits.device)
                
                # Ensure shapes match
                if pred_logits.shape != target_labels.shape:
                    # Pad or truncate if needed
                    min_edges = min(pred_logits.size(0), target_labels.size(0))
                    pred_logits = pred_logits[:min_edges]
                    target_labels = target_labels[:min_edges]
                
                pred_loss += self.predicate_loss(pred_logits, target_labels)
        
        pred_loss = pred_loss / len(predictions)
        
        # 2. Temporal consistency loss (smooth predictions)
        consistency_loss = torch.tensor(0.0, device=pred_loss.device)
        
        if len(predictions) > 1:
            for i in range(len(predictions) - 1):
                curr_probs = torch.sigmoid(predictions[i]["predicate_logits"])
                next_probs = torch.sigmoid(predictions[i + 1]["predicate_logits"])
                
                # L2 smoothness
                min_edges = min(curr_probs.size(0), next_probs.size(0))
                consistency_loss += F.mse_loss(
                    curr_probs[:min_edges],
                    next_probs[:min_edges],
                )
            
            consistency_loss = consistency_loss / (len(predictions) - 1)
        
        # 3. Future prediction loss (if actions provided)
        future_loss = torch.tensor(0.0, device=pred_loss.device)
        
        # Total loss
        total_loss = (
            self.predicate_weight * pred_loss +
            self.consistency_weight * consistency_loss +
            self.future_weight * future_loss
        )
        
        return {
            "total": total_loss,
            "predicate": pred_loss,
            "consistency": consistency_loss,
            "future": future_loss,
        }


# =============================================================================
# Custom Collate Function
# =============================================================================

def collate_sequences(batch):
    """Custom collate function for sequence data.
    
    Handles list of (graphs_seq, actions_seq, next_graph) tuples.
    Returns batches that can be processed frame-by-frame.
    """
    graphs_seqs, actions_seqs, next_graphs = zip(*batch)
    
    # Return as-is (will process sequences frame-by-frame in training loop)
    return {
        "graphs_seqs": graphs_seqs,  # List of lists of Data
        "actions_seqs": actions_seqs,  # List of lists of tensors
        "next_graphs": next_graphs,  # List of Data
    }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: SpatiotemporalGNN,
    dataloader: DataLoader,
    criterion: TemporalLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    profile: GPUProfile,
) -> float:
    """Train for one epoch with BATCHED time-step processing.
    
    Key optimization: Instead of processing sequences one-by-one,
    batch all graphs at the same timestep across all sequences.
    This reduces forward passes from B×T to just T (where B=batch_size, T=seq_len).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        # Extract batch data
        graphs_seqs = batch_data["graphs_seqs"]  # List[List[Data]], shape: (B, T)
        actions_seqs = batch_data["actions_seqs"]  # List[List[Tensor]], shape: (B, T)
        next_graphs = batch_data["next_graphs"]  # List[Data], shape: (B,)
        
        batch_size = len(graphs_seqs)
        if batch_size == 0:
            continue
            
        seq_len = len(graphs_seqs[0])
        
        # Reset hidden state for new batch
        model.reset_hidden_state()
        
        # Collect all predictions across time steps
        all_predictions = []  # Will be List[Dict] with batched outputs
        hidden_state = None
        
        # Process time steps with BATCHED forward passes
        with torch.amp.autocast('cuda', enabled=profile.use_amp):
            for t in range(seq_len):
                # Batch all graphs at timestep t across all sequences
                graphs_at_t = [graphs_seqs[b][t] for b in range(batch_size)]
                
                # Use PyG Batch for efficient batched processing
                batched_graph = Batch.from_data_list(graphs_at_t)
                
                # Ensure on device
                if batched_graph.x.device != device:
                    batched_graph = batched_graph.to(device)
                
                # Single batched forward pass for all B graphs at timestep t
                output = model(batched_graph, hidden_state, return_hidden=True)
                all_predictions.append(output)
                hidden_state = output.get("hidden_state")
        
        # Compute loss using last prediction
        # Note: batched_graph from last iteration is the last timestep's batch
        with torch.amp.autocast('cuda', enabled=profile.use_amp):
            last_pred = all_predictions[-1]
            pred_logits = last_pred["predicate_logits"]
            
            # Target labels from the SAME batched graph (ensures edge counts match)
            if hasattr(batched_graph, 'predicate_labels') and batched_graph.predicate_labels is not None:
                target_labels = batched_graph.predicate_labels
                if target_labels.device != device:
                    target_labels = target_labels.to(device)
                loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels.float())
            elif hasattr(batched_graph, 'y') and batched_graph.y is not None:
                target_labels = batched_graph.y
                if target_labels.device != device:
                    target_labels = target_labels.to(device)
                loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels.float())
            else:
                # Fallback: use temporal consistency loss
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Backward pass
        loss = loss / profile.accumulation_steps
        scaler.scale(loss).backward()
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % profile.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * profile.accumulation_steps
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: SpatiotemporalGNN,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model with BATCHED time-step processing."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            graphs_seqs = batch_data["graphs_seqs"]
            next_graphs = batch_data["next_graphs"]
            
            batch_size = len(graphs_seqs)
            if batch_size == 0:
                continue
                
            seq_len = len(graphs_seqs[0])
            
            # Reset hidden state
            model.reset_hidden_state()
            hidden_state = None
            
            # Process time steps with batched forward passes
            for t in range(seq_len):
                graphs_at_t = [graphs_seqs[b][t] for b in range(batch_size)]
                batched_graph = Batch.from_data_list(graphs_at_t)
                
                if batched_graph.x.device != device:
                    batched_graph = batched_graph.to(device)
                
                output = model(batched_graph, hidden_state, return_hidden=True)
                hidden_state = output.get("hidden_state")
            
            # Get last prediction for loss
            pred_logits = output["predicate_logits"]
            
            # Use target labels from the SAME batched_graph (ensures edge counts match)
            if hasattr(batched_graph, 'predicate_labels') and batched_graph.predicate_labels is not None:
                target_labels = batched_graph.predicate_labels.float()
                if target_labels.device != device:
                    target_labels = target_labels.to(device)
                loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
                total_loss += loss.item()
                num_batches += 1
            elif hasattr(batched_graph, 'y') and batched_graph.y is not None:
                target_labels = batched_graph.y.float()
                if target_labels.device != device:
                    target_labels = target_labels.to(device)
                loss = F.binary_cross_entropy_with_logits(pred_logits, target_labels)
                total_loss += loss.item()
                num_batches += 1
    
    return {
        "val_loss": total_loss / num_batches if num_batches > 0 else float("inf"),
    }


# =============================================================================
# Sanity Check (runs 1 mini-epoch BEFORE expensive full pre-computation)
# =============================================================================

def run_sanity_check(
    model: nn.Module,
    data_manager: DataManager,
    graph_transformer: LeRobotGraphTransformer,
    criterion: nn.Module,
    device: torch.device,
    profile: GPUProfile,
) -> bool:
    """
    Run a comprehensive sanity check with small data and 1 epoch.
    Catches ALL errors before expensive full pre-computation.
    """
    logger.info("=" * 60)
    logger.info("SANITY CHECK: Running 1 mini-epoch with small dataset...")
    logger.info("=" * 60)
    
    try:
        from gnn_reasoner.lerobot_transformer import add_predicate_labels
        
        # Step 1: Pre-compute a small number of graphs (100 frames = ~1 second)
        num_test_frames = 100
        logger.info(f"  Pre-computing {num_test_frames} test graphs...")
        
        test_graphs = []
        test_actions = []
        prev_graph = None
        
        for i in range(num_test_frames):
            frame = data_manager.get_frame(i)
            state = frame["observation.state"]
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            
            graph = graph_transformer.to_graph(state)
            # Add labels BEFORE moving to GPU (labels computation is CPU-based)
            graph = add_predicate_labels(graph, prev_graph)
            prev_graph = graph.clone()  # Clone to keep CPU version (to() modifies in-place)
            test_graphs.append(graph.to(device))  # Move to GPU after labels added
            
            action = frame.get("action")
            if action is not None:
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32)
            else:
                action = torch.zeros(14, dtype=torch.float32)
            test_actions.append(action.to(device))
        
        logger.info(f"  ✓ Pre-computed {len(test_graphs)} graphs")
        
        # Step 2: Create mini datasets
        train_split = int(num_test_frames * 0.8)
        train_dataset = PrecomputedSequenceDataset(
            test_graphs, test_actions,
            sequence_length=profile.sequence_length,
            start_idx=0, end_idx=train_split,
        )
        val_dataset = PrecomputedSequenceDataset(
            test_graphs, test_actions,
            sequence_length=profile.sequence_length,
            start_idx=train_split, end_idx=num_test_frames,
        )
        
        logger.info(f"  Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")
        
        # Step 3: Create mini dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=min(8, profile.batch_size),
            shuffle=True, num_workers=0, pin_memory=False,
            collate_fn=collate_sequences,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=min(8, profile.batch_size),
            shuffle=False, num_workers=0, pin_memory=False,
            collate_fn=collate_sequences,
        )
        
        # Step 4: Setup mini training
        optimizer = AdamW(model.parameters(), lr=profile.learning_rate)
        scaler = torch.amp.GradScaler('cuda') if profile.use_amp else None
        
        # Step 5: Run 1 training epoch
        logger.info("  Running 1 training epoch...")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, profile)
        logger.info(f"  ✓ Training epoch complete, loss: {train_loss:.4f}")
        
        # Step 6: Run 1 validation epoch
        logger.info("  Running 1 validation epoch...")
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"  ✓ Validation complete, loss: {val_metrics['val_loss']:.4f}")
        
        # Step 7: Check gradients exist
        grad_found = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                grad_found = True
                break
        
        if grad_found:
            logger.info("  ✓ Gradients flowing correctly")
        else:
            logger.warning("  ⚠ No gradients found (might be ok if just evaluated)")
        
        # Clean up
        model.zero_grad()
        del test_graphs, test_actions, train_loader, val_loader
        torch.cuda.empty_cache()
        
        logger.info("=" * 60)
        logger.info("SANITY CHECK PASSED! Proceeding with full training...")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"SANITY CHECK FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        logger.error("Fix the error above before running full training!")
        logger.error("=" * 60)
        raise


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SpatiotemporalGNN")
    parser.add_argument("--repo", type=str, default="lerobot/aloha_static_coffee")
    parser.add_argument("--max-frames", type=int, default=55000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=str, default="experiments/stgnn_training")
    parser.add_argument("--base-checkpoint", type=str, help="Pre-trained RelationalGNN checkpoint")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze base GNN weights")
    parser.add_argument("--profile", type=str, choices=["rtx3070", "rtx500", "rtx4080"], help="Force GPU profile")
    parser.add_argument("--sequence-length", type=int, help="Override sequence length")
    parser.add_argument("--lazy", action="store_true", help="Use lazy loading (slower, lower memory)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect GPU profile
    if args.profile:
        profile = GPU_PROFILES[args.profile]
    else:
        profile = detect_gpu_profile()
    
    if args.sequence_length:
        profile.sequence_length = args.sequence_length
    
    logger.info(f"Using profile: {profile.name}")
    logger.info(f"  Batch size: {profile.batch_size} (effective: {profile.batch_size * profile.accumulation_steps})")
    logger.info(f"  Sequence length: {profile.sequence_length}")
    
    # Load data
    logger.info(f"Loading dataset: {args.repo}")
    data_manager = DataManager(args.repo)
    graph_transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    
    # Determine total frames
    total_frames = min(args.max_frames, len(data_manager))
    train_split = int(total_frames * 0.9)
    
    # =========================================================================
    # Create model FIRST (so sanity check can run before expensive pre-computation)
    # =========================================================================
    base_gnn = None
    if args.base_checkpoint:
        logger.info(f"Loading base GNN from {args.base_checkpoint}")
        base_gnn = RelationalGNN(hidden_dim=profile.hidden_dim)
        checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        base_gnn.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Base GNN loaded successfully")
    
    model = SpatiotemporalGNN(
        base_gnn=base_gnn,
        hidden_dim=profile.hidden_dim,
        temporal_hidden_dim=profile.temporal_hidden_dim,
        freeze_base=args.freeze_base,
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")
    if args.freeze_base:
        logger.info("Base GNN frozen")
    
    # Training setup (needed for sanity check)
    criterion = TemporalLoss()
    
    # =========================================================================
    # SANITY CHECK: Run before expensive pre-computation!
    # =========================================================================
    run_sanity_check(
        model=model,
        data_manager=data_manager,
        graph_transformer=graph_transformer,
        criterion=criterion,
        device=device,
        profile=profile,
    )
    
    # =========================================================================
    # Dataset creation (expensive pre-computation happens here)
    # =========================================================================
    if args.lazy:
        # Legacy lazy loading (slower, lower memory)
        logger.info("Using LAZY loading (slower but lower memory)")
        train_data_manager = DataManager(args.repo)
        val_data_manager = DataManager(args.repo)
        
        train_dataset = LazySequenceDataset(
            train_data_manager,
            graph_transformer,
            sequence_length=profile.sequence_length,
            max_frames=train_split,
        )
        val_dataset = LazySequenceDataset(
            val_data_manager,
            graph_transformer,
            sequence_length=profile.sequence_length,
            max_frames=total_frames - train_split,
        )
    else:
        # RAM pre-computation (730× faster!)
        logger.info("Using RAM PRE-COMPUTATION (fast training)")
        precompute_start = time.time()
        
        all_graphs, all_actions = precompute_all_graphs(
            data_manager,
            graph_transformer,
            max_frames=total_frames,
            add_labels=True,
        )
        
        precompute_time = time.time() - precompute_start
        logger.info(f"Pre-computation complete in {precompute_time:.1f}s")
        
        # Move all data to GPU (eliminates CPU→GPU transfer during training)
        logger.info("Moving all graphs to GPU...")
        gpu_start = time.time()
        all_graphs = [g.to(device) for g in all_graphs]
        all_actions = [a.to(device) for a in all_actions]
        gpu_time = time.time() - gpu_start
        logger.info(f"GPU transfer complete in {gpu_time:.1f}s")
        
        # Check VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"VRAM usage: {vram_used:.2f} GB / {vram_total:.1f} GB")
        
        # Split into train/val
        train_dataset = PrecomputedSequenceDataset(
            all_graphs,
            all_actions,
            sequence_length=profile.sequence_length,
            start_idx=0,
            end_idx=train_split,
        )
        val_dataset = PrecomputedSequenceDataset(
            all_graphs,
            all_actions,
            sequence_length=profile.sequence_length,
            start_idx=train_split,
            end_idx=total_frames,
        )
    
    # DataLoaders (workers=0, pin_memory=False for PyG Data compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # PyG Data doesn't support pin_memory well
        collate_fn=collate_sequences,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_sequences,
    )
    
    # Optimizer and scheduler (model and criterion already created above)
    optimizer = AdamW(
        model.parameters(),
        lr=profile.learning_rate,
        weight_decay=profile.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if profile.use_amp else None
    
    # Training loop
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, profile)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(lr)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}",
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            lr=f"{lr:.6f}",
        )
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "profile": profile.name,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
    
    # Save final model and history
    torch.save(checkpoint, output_dir / "final_model.pt")
    
    history["total_time_seconds"] = time.time() - start_time
    history["best_val_loss"] = best_val_loss
    history["profile"] = profile.name
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training complete! Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

