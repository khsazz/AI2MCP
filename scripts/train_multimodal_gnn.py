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


# GPU profiles (same as train_relational_gnn.py)
GPU_PROFILES = {
    "RTX 500 Ada (4GB)": {
        "batch_size": 8,
        "accumulation_steps": 8,
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


def detect_gpu_profile() -> tuple[str, dict]:
    """Detect GPU and select appropriate training profile."""
    if not torch.cuda.is_available():
        return "CPU Fallback", GPU_PROFILES["CPU Fallback"]

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"Detected GPU: {gpu_name} ({vram_gb:.1f}GB)")

    if vram_gb >= 12:
        return "RTX 4080 (16GB)", GPU_PROFILES["RTX 4080 (16GB)"]
    else:
        return "RTX 500 Ada (4GB)", GPU_PROFILES["RTX 500 Ada (4GB)"]


def create_synthetic_data(num_samples: int = 1000) -> list[dict]:
    """Create synthetic data for quick testing."""
    data_list = []
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    detector = MockVisionDetector()
    depth_estimator = MockDepthEstimator()
    intrinsics = CameraIntrinsics.default_aloha()

    for i in range(num_samples):
        # Random state and image
        state = torch.randn(14)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Get mock detections
        detections = detector.detect(image, prompts=["cup", "mug", "bottle"])
        depth_map = depth_estimator.estimate(image)
        objects_3d = detections_to_objects_3d(detections, depth_map, intrinsics)

        # Create graph
        gripper_state = np.random.uniform(0, 1)
        graph = transformer.to_graph_with_objects(state, objects_3d, gripper_state)

        # Add labels
        prev_graph = data_list[-1]["graph"] if data_list else None
        graph = add_predicate_labels(graph, prev_graph)

        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        data_list.append({
            "graph": graph,
            "image": image_tensor,
            "bboxes": [det.bbox for det in detections],
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
):
    """Train the MultiModalGNN."""
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
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
    parser.add_argument("--max-frames", type=int, default=5000)
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

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        output_dir=output_dir,
        profile=profile,
        use_vision=not args.no_vision,
    )


if __name__ == "__main__":
    main()

