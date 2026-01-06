"""Multi-Modal GNN with Vision Encoder.

Combines DINOv2 visual features with kinematic graph features
for enhanced object-centric relational reasoning.

Option C: Learned fusion approach with frozen vision encoder.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

    class Data:  # type: ignore
        pass


from gnn_reasoner.model.relational_gnn import (
    NodeEncoder,
    EdgeEncoder,
    PredicateHead,
    ALL_PREDICATES,
    SPATIAL_PREDICATES,
    INTERACTION_PREDICATES,
    PredicateOutput,
)

if TYPE_CHECKING:
    from torch import Tensor
    import numpy as np

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """Frozen DINOv2 vision encoder with RoI feature extraction.

    Extracts visual features for detected objects using:
    1. Full image encoding with DINOv2
    2. RoI pooling to extract patch features per detection
    3. Projection to GNN hidden dimension

    Example:
        >>> encoder = VisionEncoder(model_name="dinov2_vits14")
        >>> image = torch.randn(1, 3, 518, 518)  # DINOv2 expects 518x518
        >>> bboxes = [(100, 100, 200, 200), (300, 200, 400, 350)]
        >>> features = encoder(image, bboxes)  # (2, hidden_dim)
    """

    # DINOv2 model configurations
    MODEL_CONFIGS = {
        "dinov2_vits14": {"embed_dim": 384, "patch_size": 14},
        "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14},
        "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14},
    }

    def __init__(
        self,
        model_name: Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"] = "dinov2_vits14",
        hidden_dim: int = 128,
        freeze: bool = True,
        cache_dir: str | None = None,
    ):
        """Initialize the vision encoder.

        Args:
            model_name: DINOv2 model variant
            hidden_dim: Output dimension for GNN compatibility
            freeze: If True, freeze the DINOv2 backbone
            cache_dir: Directory for model cache
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze = freeze

        config = self.MODEL_CONFIGS[model_name]
        self.embed_dim = config["embed_dim"]
        self.patch_size = config["patch_size"]

        self._backbone = None
        self._is_loaded = False

        # Projection from DINOv2 embedding to hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable token for "no object detected" case
        self.no_object_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.no_object_token, std=0.02)

    def _load_backbone(self) -> None:
        """Lazy-load the DINOv2 backbone."""
        if self._is_loaded:
            return

        try:
            logger.info(f"Loading DINOv2 backbone: {self.model_name}")

            # Load via torch hub
            self._backbone = torch.hub.load(
                "facebookresearch/dinov2",
                self.model_name,
                pretrained=True,
                trust_repo=True,
            )

            if self.freeze:
                for param in self._backbone.parameters():
                    param.requires_grad = False
                self._backbone.eval()

            logger.info(f"DINOv2 loaded successfully (frozen={self.freeze})")
            self._is_loaded = True

        except Exception as e:
            logger.warning(f"Failed to load DINOv2: {e}. Using random features.")
            self._backbone = None
            self._is_loaded = True

    @property
    def backbone(self) -> nn.Module | None:
        """Get the backbone, loading if needed."""
        self._load_backbone()
        # Ensure backbone is on same device as projection layer
        if self._backbone is not None:
            target_device = self.projection[0].weight.device
            if next(self._backbone.parameters()).device != target_device:
                self._backbone = self._backbone.to(target_device)
        return self._backbone

    def _resize_for_patches(self, image: Tensor) -> Tensor:
        """Resize image to be compatible with patch size.

        DINOv2 requires input dimensions to be multiples of patch_size.

        Args:
            image: Input image (B, 3, H, W)

        Returns:
            Resized image with dimensions divisible by patch_size
        """
        B, C, H, W = image.shape

        # Calculate new dimensions (round up to nearest multiple)
        new_H = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_W = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size

        if new_H != H or new_W != W:
            image = F.interpolate(image, size=(new_H, new_W), mode="bilinear", align_corners=False)

        return image

    def extract_patch_tokens(self, image: Tensor) -> Tensor:
        """Extract patch tokens from image.

        Args:
            image: Input image (B, 3, H, W), normalized

        Returns:
            Patch tokens (B, num_patches, embed_dim)
        """
        if self.backbone is None:
            # Return random features if backbone unavailable
            B = image.size(0)
            H, W = image.shape[2:]
            num_patches = (H // self.patch_size) * (W // self.patch_size)
            return torch.randn(B, num_patches, self.embed_dim, device=image.device)

        # Resize to be compatible with patch size
        image = self._resize_for_patches(image)

        with torch.set_grad_enabled(not self.freeze):
            # DINOv2 forward returns dict with patch tokens
            output = self.backbone.forward_features(image)

            # Get patch tokens (excluding CLS token)
            if isinstance(output, dict):
                patch_tokens = output.get("x_norm_patchtokens", output.get("patch_tokens"))
            else:
                # Some versions return tensor directly
                patch_tokens = output[:, 1:]  # Skip CLS token

        return patch_tokens

    def roi_pool_patches(
        self,
        patch_tokens: Tensor,
        bboxes: list[tuple[int, int, int, int]],
        image_size: tuple[int, int],
    ) -> Tensor:
        """Pool patch features within bounding boxes.

        Args:
            patch_tokens: (B, num_patches, embed_dim)
            bboxes: List of (x1, y1, x2, y2) bounding boxes in pixel coords
            image_size: (H, W) of original image

        Returns:
            Pooled features (num_bboxes, embed_dim)
        """
        if len(bboxes) == 0:
            return self.no_object_token.expand(1, -1)

        H, W = image_size
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size

        device = patch_tokens.device
        pooled_features = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # Convert pixel coords to patch indices
            patch_x1 = int(x1 / W * num_patches_w)
            patch_x2 = int(x2 / W * num_patches_w)
            patch_y1 = int(y1 / H * num_patches_h)
            patch_y2 = int(y2 / H * num_patches_h)

            # Clamp to valid range
            patch_x1 = max(0, min(patch_x1, num_patches_w - 1))
            patch_x2 = max(patch_x1 + 1, min(patch_x2, num_patches_w))
            patch_y1 = max(0, min(patch_y1, num_patches_h - 1))
            patch_y2 = max(patch_y1 + 1, min(patch_y2, num_patches_h))

            # Get patch indices (row-major order)
            indices = []
            for py in range(patch_y1, patch_y2):
                for px in range(patch_x1, patch_x2):
                    idx = py * num_patches_w + px
                    indices.append(idx)

            if not indices:
                # Fallback to center patch
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                idx = (cy // self.patch_size) * num_patches_w + (cx // self.patch_size)
                indices = [min(idx, patch_tokens.size(1) - 1)]

            # Pool patches (mean pooling)
            indices_tensor = torch.tensor(indices, device=device)
            roi_tokens = patch_tokens[0, indices_tensor]  # Assume batch=1
            pooled = roi_tokens.mean(dim=0)
            pooled_features.append(pooled)

        return torch.stack(pooled_features, dim=0)

    def forward(
        self,
        image: Tensor,
        bboxes: list[tuple[int, int, int, int]],
    ) -> Tensor:
        """Extract visual features for detected objects.

        Args:
            image: Input image (1, 3, H, W), normalized to DINOv2 format
            bboxes: List of (x1, y1, x2, y2) bounding boxes

        Returns:
            Visual features (num_objects, hidden_dim)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        orig_H, orig_W = image.shape[2:]

        # Extract patch tokens (image may be resized internally)
        patch_tokens = self.extract_patch_tokens(image)

        # Get the actual size used for patches (after potential resize)
        resized_H = ((orig_H + self.patch_size - 1) // self.patch_size) * self.patch_size
        resized_W = ((orig_W + self.patch_size - 1) // self.patch_size) * self.patch_size

        # Scale bboxes to resized dimensions
        scale_y = resized_H / orig_H
        scale_x = resized_W / orig_W
        scaled_bboxes = [
            (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
            for x1, y1, x2, y2 in bboxes
        ]

        # Pool features for each bbox
        pooled = self.roi_pool_patches(patch_tokens, scaled_bboxes, (resized_H, resized_W))

        # Project to hidden dimension
        return self.projection(pooled)


class CrossAttentionFusion(nn.Module):
    """Cross-attention layer to fuse vision and kinematic features.

    Allows object nodes to attend to kinematic nodes and vice versa.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Cross-attention: vision queries kinematic
        self.vision_to_kinematic = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention: kinematic queries vision
        self.kinematic_to_vision = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN for fusion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        kinematic_features: Tensor,
        vision_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fuse kinematic and vision features.

        Args:
            kinematic_features: (num_kinematic, hidden_dim)
            vision_features: (num_objects, hidden_dim)

        Returns:
            Tuple of (fused_kinematic, fused_vision)
        """
        # Add batch dimension
        kinematic = kinematic_features.unsqueeze(0)  # (1, num_kin, hidden)
        vision = vision_features.unsqueeze(0)  # (1, num_obj, hidden)

        # Kinematic attends to vision
        kinematic_attended, _ = self.kinematic_to_vision(
            kinematic, vision, vision
        )
        kinematic = self.norm1(kinematic + kinematic_attended)

        # Vision attends to kinematic
        vision_attended, _ = self.vision_to_kinematic(
            vision, kinematic, kinematic
        )
        vision = self.norm2(vision + vision_attended)

        # Apply FFN to both
        kinematic = self.norm3(kinematic + self.ffn(kinematic))
        vision = self.norm3(vision + self.ffn(vision))

        return kinematic.squeeze(0), vision.squeeze(0)


class MultiModalGNN(nn.Module):
    """Multi-Modal GNN combining vision and kinematic features.

    Architecture:
        1. Kinematic features → NodeEncoder → kinematic embeddings
        2. Image → DINOv2 → RoI Pool → vision embeddings
        3. Cross-attention fusion between kinematic and vision
        4. Concatenate and process through GNN layers
        5. Predicate prediction on edges

    This is Option C: Learned fusion with frozen vision encoder.
    """

    def __init__(
        self,
        node_input_dim: int = 5,
        edge_input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_predicates: int = len(ALL_PREDICATES),
        vision_model: str = "dinov2_vits14",
        freeze_vision: bool = True,
    ):
        """Initialize MultiModalGNN.

        Args:
            node_input_dim: Input dimension for kinematic node features
            edge_input_dim: Input dimension for edge features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN message passing layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_predicates: Number of predicates to predict
            vision_model: DINOv2 model variant
            freeze_vision: If True, freeze the vision encoder
        """
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for MultiModalGNN")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_predicates = num_predicates

        # Kinematic encoder (same as RelationalGNN)
        self.node_encoder = NodeEncoder(node_input_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(edge_input_dim, hidden_dim // 4)

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_model,
            hidden_dim=hidden_dim,
            freeze=freeze_vision,
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim // 4,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Predicate prediction head
        self.predicate_head = PredicateHead(hidden_dim, num_predicates)

        # Graph-level output
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        data: Data,
        image: Tensor | None = None,
        bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass through the multi-modal GNN.

        Args:
            data: PyG Data object with kinematic graph
            image: Optional image tensor (1, 3, H, W) for vision features
            bboxes: Optional bounding boxes for detected objects

        Returns:
            Dictionary with:
                - node_embeddings: (num_nodes, hidden_dim)
                - predicate_logits: (num_edges, num_predicates)
                - graph_embedding: (batch_size, hidden_dim)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        node_types = data.node_types if hasattr(data, "node_types") else None
        batch = (
            data.batch
            if hasattr(data, "batch")
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        # Encode kinematic features
        kinematic_embed = self.node_encoder(x, node_types)

        # Get vision features if available
        if image is not None and bboxes:
            vision_embed = self.vision_encoder(image, bboxes)

            # Separate kinematic and object nodes
            if node_types is not None:
                kinematic_mask = node_types != 2
                object_mask = node_types == 2
                num_kinematic = kinematic_mask.sum().item()
                num_objects = object_mask.sum().item()

                if num_objects > 0 and vision_embed.size(0) == num_objects:
                    # Apply cross-attention fusion
                    kinematic_features = kinematic_embed[kinematic_mask]
                    fused_kinematic, fused_vision = self.fusion(
                        kinematic_features, vision_embed
                    )

                    # Replace embeddings with fused versions
                    kinematic_embed[kinematic_mask] = fused_kinematic
                    kinematic_embed[object_mask] = fused_vision

        x = kinematic_embed

        # Encode edges
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # GNN message passing
        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x + x_residual)
            x = F.relu(x)
            x = self.dropout(x)

        # Predicate prediction
        predicate_logits = self.predicate_head(x, edge_index)

        # Graph embedding
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_embed = self.graph_head(torch.cat([mean_pool, max_pool], dim=-1))

        return {
            "node_embeddings": x,
            "predicate_logits": predicate_logits,
            "graph_embedding": graph_embed,
        }

    def predict_predicates(
        self,
        data: Data,
        image: Tensor | None = None,
        bboxes: list[tuple[int, int, int, int]] | None = None,
        threshold: float = 0.5,
    ) -> list[PredicateOutput]:
        """Predict active predicates for a graph.

        Args:
            data: PyG Data object
            image: Optional image for vision features
            bboxes: Optional bounding boxes
            threshold: Probability threshold for predicate activation

        Returns:
            List of PredicateOutput for all edges
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data, image, bboxes)
            probs = torch.sigmoid(outputs["predicate_logits"])

            results = []
            edge_index = data.edge_index

            for edge_idx in range(probs.size(0)):
                src = edge_index[0, edge_idx].item()
                tgt = edge_index[1, edge_idx].item()

                for pred_idx, pred_name in enumerate(ALL_PREDICATES):
                    prob = probs[edge_idx, pred_idx].item()
                    results.append(
                        PredicateOutput(
                            predicate_name=pred_name,
                            source_node=src,
                            target_node=tgt,
                            probability=prob,
                            active=prob > threshold,
                        )
                    )

            return results

    def get_active_predicates(
        self,
        data: Data,
        image: Tensor | None = None,
        bboxes: list[tuple[int, int, int, int]] | None = None,
        threshold: float = 0.5,
    ) -> list[PredicateOutput]:
        """Get only active predicates above threshold."""
        all_preds = self.predict_predicates(data, image, bboxes, threshold)
        return [p for p in all_preds if p.active]

    def to_world_context(
        self,
        data: Data,
        image: Tensor | None = None,
        bboxes: list[tuple[int, int, int, int]] | None = None,
        threshold: float = 0.5,
    ) -> dict:
        """Convert predictions to structured world context for MCP."""
        outputs = self.forward(data, image, bboxes)
        active_predicates = self.get_active_predicates(data, image, bboxes, threshold)

        spatial = [p for p in active_predicates if p.predicate_name in SPATIAL_PREDICATES]
        interaction = [
            p for p in active_predicates if p.predicate_name in INTERACTION_PREDICATES
        ]

        return {
            "num_nodes": data.x.size(0),
            "num_edges": data.edge_index.size(1),
            "graph_embedding": outputs["graph_embedding"].tolist(),
            "spatial_predicates": [
                {
                    "predicate": p.predicate_name,
                    "source": p.source_node,
                    "target": p.target_node,
                    "confidence": round(p.probability, 3),
                }
                for p in spatial
            ],
            "interaction_predicates": [
                {
                    "predicate": p.predicate_name,
                    "source": p.source_node,
                    "target": p.target_node,
                    "confidence": round(p.probability, 3),
                }
                for p in interaction
            ],
        }


class MockVisionEncoder(nn.Module):
    """Mock vision encoder for testing without DINOv2 weights."""

    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(384, hidden_dim)

    def forward(
        self,
        image: Tensor,
        bboxes: list[tuple[int, int, int, int]],
    ) -> Tensor:
        """Return random features for testing."""
        num_objects = max(1, len(bboxes))
        return torch.randn(num_objects, self.hidden_dim, device=image.device)

