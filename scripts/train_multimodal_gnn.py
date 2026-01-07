#!/usr/bin/env python3
"""Training script for MultiModalGNN.

Trains the vision-enhanced GNN model on LeRobot data with image features.

Usage:
    python scripts/train_multimodal_gnn.py --repo lerobot/aloha_static_coffee --epochs 100
    python scripts/train_multimodal_gnn.py --synthetic --epochs 50  # Quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gnn_reasoner import (
    DataManager,
    LeRobotGraphTransformer,
    ALOHA_KINEMATIC_CHAIN,
    add_predicate_labels,
    MockVisionDetector,
    MockDepthEstimator,
    CameraIntrinsics,
    detections_to_objects_3d,
)
from gnn_reasoner.model import MultiModalGNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# GPU profiles for different hardware
GPU_PROFILES = {
    "RTX 500 Ada (4GB)": {
        "batch_size": 8,
        "accumulation_steps": 8,
        "use_amp": True,
    },
    "RTX 3070 (8GB)": {
        "batch_size": 16,
        "accumulation_steps": 4,
        "use_amp": True,
    },
    "RTX 4080 (16GB)": {
        "batch_size": 32,
        "accumulation_steps": 2,
        "use_amp": True,
    },
    "CPU Fallback": {
        "batch_size": 4,
        "accumulation_steps": 16,
        "use_amp": False,
    },
}

# Predicate indices for reference
# ALL_PREDICATES = [
#   "is_near",        # 0 - spatial (common)
#   "is_above",       # 1 - spatial (common)
#   "is_below",       # 2 - spatial (common)
#   "is_left_of",     # 3 - spatial (common)
#   "is_right_of",    # 4 - spatial (common)
#   "is_holding",     # 5 - interaction (RARE - critical)
#   "is_contacting",  # 6 - interaction (rare)
#   "is_approaching", # 7 - interaction (uncommon)
#   "is_retracting",  # 8 - interaction (uncommon)
# ]
HOLDING_IDX = 5
CONTACTING_IDX = 6
APPROACHING_IDX = 7
RETRACTING_IDX = 8


class WeightedFocalLoss(nn.Module):
    """Focal Loss with per-class positive weights.
    
    Combines:
    - Focal Loss: Down-weights easy examples, focuses on hard ones
    - Per-class weights: Higher penalty for missing rare positives
    
    This addresses the class imbalance problem where interaction predicates
    (especially is_holding) are extremely rare in the dataset.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """Initialize WeightedFocalLoss.
        
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
            gamma: Focusing parameter (default: 2.0, higher = more focus on hard examples)
            pos_weight: Per-class positive weights tensor of shape [num_predicates]
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted focal loss.
        
        Args:
            inputs: Logits of shape [N, num_predicates]
            targets: Binary labels of shape [N, num_predicates]
            
        Returns:
            Loss scalar (if reduction='mean' or 'sum') or tensor (if 'none')
        """
        # Compute base BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute probabilities
        probs = torch.sigmoid(inputs)
        
        # pt = p if y=1 else 1-p
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply per-class positive weights for rare predicates
        if self.pos_weight is not None:
            # Only apply to positive samples
            pos_weight = self.pos_weight.to(inputs.device)
            # Expand weight to match batch: [num_predicates] -> [1, num_predicates]
            weight_matrix = pos_weight.unsqueeze(0).expand_as(targets)
            # Only boost loss for positive labels
            class_weight = torch.where(targets > 0.5, weight_matrix, torch.ones_like(weight_matrix))
            focal_loss = focal_loss * class_weight
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(data_list: list[dict], num_predicates: int = 9) -> torch.Tensor:
    """Compute per-class positive weights based on dataset statistics.
    
    Analyzes the dataset to find the ratio of positive to negative samples
    for each predicate, then computes inverse frequency weights.
    
    Args:
        data_list: List of data dicts with 'graph' containing .y labels
        num_predicates: Number of predicates
        
    Returns:
        pos_weight tensor of shape [num_predicates]
    """
    pos_counts = torch.zeros(num_predicates)
    neg_counts = torch.zeros(num_predicates)
    
    for item in data_list:
        labels = item["graph"].y
        if labels is None or labels.numel() == 0:
            continue
        
        # labels shape: [num_edges, num_predicates]
        pos_counts += (labels > 0.5).float().sum(dim=0).cpu()
        neg_counts += (labels <= 0.5).float().sum(dim=0).cpu()
    
    # Compute inverse frequency ratio with smoothing to avoid division by zero
    total = pos_counts + neg_counts
    pos_ratio = pos_counts / (total + 1e-8)
    neg_ratio = neg_counts / (total + 1e-8)
    
    # Weight = neg/pos ratio (higher weight for rarer positives)
    # Clamp to reasonable range [1.0, 50.0]
    weights = (neg_ratio / (pos_ratio + 1e-8)).clamp(1.0, 50.0)
    
    # Apply minimum weights for critical interaction predicates
    weights[HOLDING_IDX] = max(weights[HOLDING_IDX].item(), 20.0)
    weights[CONTACTING_IDX] = max(weights[CONTACTING_IDX].item(), 10.0)
    weights[APPROACHING_IDX] = max(weights[APPROACHING_IDX].item(), 5.0)
    weights[RETRACTING_IDX] = max(weights[RETRACTING_IDX].item(), 5.0)
    
    return weights


def log_class_statistics(data_list: list[dict], num_predicates: int = 9):
    """Log class distribution statistics for debugging."""
    predicate_names = [
        "is_near", "is_above", "is_below", "is_left_of", "is_right_of",
        "is_holding", "is_contacting", "is_approaching", "is_retracting"
    ]
    
    pos_counts = torch.zeros(num_predicates)
    total_edges = 0
    
    for item in data_list:
        labels = item["graph"].y
        if labels is None or labels.numel() == 0:
            continue
        pos_counts += (labels > 0.5).float().sum(dim=0).cpu()
        total_edges += labels.size(0)
    
    logger.info("=" * 60)
    logger.info("CLASS DISTRIBUTION STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total edges: {total_edges:,}")
    logger.info("-" * 60)
    
    for i, name in enumerate(predicate_names):
        count = int(pos_counts[i].item())
        ratio = count / total_edges if total_edges > 0 else 0
        marker = " ⚠️ RARE" if ratio < 0.01 else ""
        logger.info(f"  {name:15s}: {count:6d} positives ({ratio*100:5.2f}%){marker}")
    
    logger.info("=" * 60)


def detect_gpu_profile() -> tuple[str, dict]:
    """Detect GPU and select appropriate training profile."""
    if not torch.cuda.is_available():
        return "CPU Fallback", GPU_PROFILES["CPU Fallback"]

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"Detected GPU: {gpu_name} ({vram_gb:.1f}GB)")

    # Select profile based on VRAM
    if vram_gb >= 12:
        return "RTX 4080 (16GB)", GPU_PROFILES["RTX 4080 (16GB)"]
    elif vram_gb >= 6:
        return "RTX 3070 (8GB)", GPU_PROFILES["RTX 3070 (8GB)"]
    else:
        return "RTX 500 Ada (4GB)", GPU_PROFILES["RTX 500 Ada (4GB)"]


def create_synthetic_data(
    num_samples: int = 1000,
    holding_ratio: float = 0.10,  # 10% holding frames
    contacting_ratio: float = 0.05,  # 5% contacting frames
) -> list[dict]:
    """Create synthetic data with realistic holding/contacting scenarios.
    
    Args:
        num_samples: Total number of samples to generate
        holding_ratio: Fraction of samples with is_holding=True
        contacting_ratio: Fraction of samples with is_contacting=True
        
    Returns:
        List of data dicts with graph, image, and bboxes
    """
    from gnn_reasoner.camera import Object3D
    
    data_list = []
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    detector = MockVisionDetector()
    depth_estimator = MockDepthEstimator()
    intrinsics = CameraIntrinsics.default_aloha()
    
    # Track scenario distribution
    num_holding = int(num_samples * holding_ratio)
    num_contacting = int(num_samples * contacting_ratio)
    num_normal = num_samples - num_holding - num_contacting
    
    logger.info(f"Generating {num_samples} samples: {num_holding} holding, "
                f"{num_contacting} contacting, {num_normal} normal")

    for i in range(num_samples):
        # Determine scenario type
        if i < num_holding:
            scenario = "holding"
        elif i < num_holding + num_contacting:
            scenario = "contacting"
        else:
            scenario = "normal"
        
        # Random state and image
        state = torch.randn(14) * 0.5
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Build a base graph to get gripper positions
        base_graph = transformer.to_graph(state)
        left_gripper_pos, right_gripper_pos = transformer.get_gripper_positions(base_graph)
        
        if scenario == "holding":
            # HOLDING: Object AT gripper position, gripper CLOSED
            gripper_state = np.random.uniform(0.0, 0.25)  # Closed (< 0.3 threshold)
            
            # Choose which gripper is holding
            if left_gripper_pos is not None and np.random.random() < 0.5:
                gripper_pos = left_gripper_pos.numpy()
            elif right_gripper_pos is not None:
                gripper_pos = right_gripper_pos.numpy()
            else:
                gripper_pos = np.array([0.0, 0.0, 0.0])
            
            # Place object VERY close to gripper (within holding threshold 0.08m)
            offset = np.random.uniform(-0.03, 0.03, size=3)
            object_pos = tuple(gripper_pos + offset)
            
            # Create object at gripper position
            held_object = Object3D(
                class_name="held_cup",
                confidence=np.random.uniform(0.9, 0.99),
                position=object_pos,
                size=(0.05, 0.08),
                bbox=(300, 200, 360, 280),  # Dummy bbox
            )
            
            # Also add some random detections
            random_detections = detector.detect(image, prompts=["mug", "bottle"])
            depth_map = depth_estimator.estimate(image)
            random_objects = detections_to_objects_3d(random_detections, depth_map, intrinsics)
            
            objects_3d = [held_object] + random_objects
            
        elif scenario == "contacting":
            # CONTACTING: Object VERY close to gripper, but gripper OPEN
            gripper_state = np.random.uniform(0.5, 1.0)  # Open
            
            if left_gripper_pos is not None and np.random.random() < 0.5:
                gripper_pos = left_gripper_pos.numpy()
            elif right_gripper_pos is not None:
                gripper_pos = right_gripper_pos.numpy()
            else:
                gripper_pos = np.array([0.0, 0.0, 0.0])
            
            # Place object close but not inside gripper (contact threshold 0.05m)
            offset = np.random.uniform(-0.04, 0.04, size=3)
            object_pos = tuple(gripper_pos + offset)
            
            contact_object = Object3D(
                class_name="contact_mug",
                confidence=np.random.uniform(0.85, 0.95),
                position=object_pos,
                size=(0.06, 0.09),
                bbox=(310, 210, 370, 290),
            )
            
            random_detections = detector.detect(image, prompts=["cup", "bottle"])
            depth_map = depth_estimator.estimate(image)
            random_objects = detections_to_objects_3d(random_detections, depth_map, intrinsics)
            
            objects_3d = [contact_object] + random_objects
            
        else:
            # NORMAL: Random objects, random gripper state
            gripper_state = np.random.uniform(0, 1)
            
            detections = detector.detect(image, prompts=["cup", "mug", "bottle"])
            depth_map = depth_estimator.estimate(image)
            objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)

        # Create graph with objects
        graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state)

        # Add labels
        prev_graph = data_list[-1]["graph"] if data_list else None
        graph = add_predicate_labels(graph, prev_graph)

        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        data_list.append({
            "graph": graph,
            "image": image_tensor,
            "bboxes": [obj.bbox for obj in objects_3d] if objects_3d else [],
        })

    return data_list


def create_lerobot_data(
    repo_id: str,
    max_frames: int = 5000,
) -> list[dict]:
    """Create training data from LeRobot dataset."""
    logger.info(f"Loading dataset: {repo_id}")
    dm = DataManager(repo_id)
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    detector = MockVisionDetector()  # Use mock for now
    depth_estimator = MockDepthEstimator()
    intrinsics = CameraIntrinsics.default_aloha()

    data_list = []
    prev_graph = None
    num_frames = min(len(dm), max_frames)

    for idx in tqdm(range(num_frames), desc="Processing frames"):
        frame = dm.get_frame(idx)
        state = frame.get("observation.state")

        if state is None:
            continue

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        # Get image if available
        image = dm.get_image(idx)
        if image is None:
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Get detections (mock for now, will use real detector later)
        detections = detector.detect(image, prompts=["cup", "mug", "coffee"])
        depth_map = depth_estimator.estimate(image)
        objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)

        # Infer gripper state from state vector
        gripper_state = state[6].item() if len(state) > 6 else 0.5

        # Create graph
        graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state)
        graph = add_predicate_labels(graph, prev_graph)

        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        data_list.append({
            "graph": graph,
            "image": image_tensor,
            "bboxes": [det.bbox for det in detections],
        })

        prev_graph = graph

    return data_list


class MultiModalDataset(torch.utils.data.Dataset):
    """Dataset wrapper for MultiModal training data."""

    def __init__(self, data_list: list[dict]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_multimodal(batch: list[dict]) -> dict:
    """Custom collate function for MultiModal data."""
    from torch_geometric.data import Batch

    graphs = [item["graph"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    bboxes = [item["bboxes"] for item in batch]

    return {
        "graph_batch": Batch.from_data_list(graphs),
        "images": images,
        "bboxes": bboxes,
    }


def train(
    model: MultiModalGNN,
    train_loader,
    val_loader,
    epochs: int,
    device: str,
    output_dir: Path,
    profile: dict,
    use_vision: bool = True,
    pos_weight: torch.Tensor | None = None,
):
    """Train the MultiModalGNN.
    
    Args:
        model: The MultiModalGNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        output_dir: Directory to save checkpoints
        profile: GPU profile with batch_size, accumulation_steps, use_amp
        use_vision: Whether to use vision features
        pos_weight: Per-class positive weights for handling class imbalance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Use Weighted Focal Loss to handle class imbalance
    # This is critical for detecting rare predicates like is_holding
    criterion = WeightedFocalLoss(
        alpha=0.25,           # Baseline weight for positives
        gamma=2.0,            # Focus on hard examples
        pos_weight=pos_weight # Per-class weights (computed from data)
    )
    logger.info(f"Using WeightedFocalLoss (alpha=0.25, gamma=2.0)")
    if pos_weight is not None:
        logger.info(f"Per-class weights: {pos_weight.tolist()}")
    
    scaler = GradScaler(enabled=profile["use_amp"])

    accumulation_steps = profile["accumulation_steps"]
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            graph_batch = batch["graph_batch"].to(device)

            # Get image and bboxes for first item in batch
            # (simplified - full implementation would handle batch properly)
            if use_vision:
                image = batch["images"][0:1].to(device)
                bboxes = batch["bboxes"][0] if batch["bboxes"] else []
            else:
                image = None
                bboxes = None

            with autocast(enabled=profile["use_amp"]):
                outputs = model(graph_batch, image, bboxes)
                pred = outputs["predicate_logits"]
                target = graph_batch.y

                if target is None or target.numel() == 0:
                    continue

                loss = criterion(pred, target) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                graph_batch = batch["graph_batch"].to(device)

                if use_vision:
                    image = batch["images"][0:1].to(device)
                    bboxes = batch["bboxes"][0] if batch["bboxes"] else []
                else:
                    image = None
                    bboxes = None

                outputs = model(graph_batch, image, bboxes)
                pred = outputs["predicate_logits"]
                target = graph_batch.y

                if target is None or target.numel() == 0:
                    continue

                val_loss += criterion(pred, target).item()

                pred_binary = (torch.sigmoid(pred) > 0.5).float()
                correct += (pred_binary == target).sum().item()
                total += target.numel()

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(current_lr)

        logger.info(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                output_dir / "best_model.pt",
            )
            logger.info(f"  → New best model saved (val_loss={val_loss:.4f})")

    total_time = time.time() - start_time

    # Save final model and history
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        },
        output_dir / "final_model.pt",
    )

    history_data = {
        "total_time_seconds": total_time,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "final_val_accuracy": val_acc,
        "history": history,
    }

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history_data, f, indent=2)

    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train MultiModalGNN")
    parser.add_argument("--repo", type=str, default="lerobot/aloha_static_coffee")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames (None=full dataset)")
    parser.add_argument("--output", type=str, default="experiments/multimodal_training")
    parser.add_argument("--vision-model", type=str, default="dinov2_vits14")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision features")
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile_name, profile = detect_gpu_profile()
    logger.info(f"Using profile: {profile_name}")

    output_dir = Path(args.output)

    # Create data
    if args.synthetic:
        logger.info("Creating synthetic data...")
        data_list = create_synthetic_data(num_samples=1000)
    else:
        data_list = create_lerobot_data(args.repo, args.max_frames)

    # Split data
    np.random.shuffle(data_list)
    split_idx = int(0.9 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Analyze class distribution and compute weights
    log_class_statistics(train_data, num_predicates=9)
    pos_weight = compute_class_weights(train_data, num_predicates=9)
    logger.info(f"Computed class weights: {pos_weight.tolist()}")

    # Create data loaders
    train_dataset = MultiModalDataset(train_data)
    val_dataset = MultiModalDataset(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=profile["batch_size"],
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=profile["batch_size"],
        shuffle=False,
        collate_fn=collate_multimodal,
        num_workers=0,
    )

    # Create model
    model = MultiModalGNN(
        hidden_dim=args.hidden_dim,
        num_predicates=9,
        vision_model=args.vision_model,
        freeze_vision=True,
    )
    model = model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train with class-balanced loss
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        output_dir=output_dir,
        profile=profile,
        use_vision=not args.no_vision,
        pos_weight=pos_weight,
    )


if __name__ == "__main__":
    main()

