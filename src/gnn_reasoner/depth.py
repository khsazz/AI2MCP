"""Monocular Depth Estimation using ZoeDepth.

Provides metric depth estimation from single RGB images for
3D object localization without dedicated depth sensors.

Supports:
- ZoeDepth (state-of-art, metric depth)
- MiDaS (relative depth, faster)
- Depth Anything (good balance)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Monocular depth estimation.

    Provides metric or relative depth maps from single RGB images.

    Example:
        >>> estimator = DepthEstimator(model="zoedepth", device="cuda")
        >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> depth_map = estimator.estimate(image)
        >>> print(f"Depth range: {depth_map.min():.2f}m - {depth_map.max():.2f}m")
    """

    def __init__(
        self,
        model: Literal["zoedepth", "midas", "depth_anything"] = "zoedepth",
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the depth estimator.

        Args:
            model: Depth estimation model to use
            device: Device for inference ("cuda", "cpu", or None for auto)
            cache_dir: Directory for model weights cache
        """
        self.model_name = model
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ai2mcp"

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._transform = None
        self._is_loaded = False

        logger.info(f"DepthEstimator initialized: model={model}, device={self.device}")

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._is_loaded:
            return

        if self.model_name == "zoedepth":
            self._load_zoedepth()
        elif self.model_name == "midas":
            self._load_midas()
        elif self.model_name == "depth_anything":
            self._load_depth_anything()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self._is_loaded = True

    def _load_zoedepth(self) -> None:
        """Load ZoeDepth model."""
        try:
            # ZoeDepth via torch hub
            logger.info("Loading ZoeDepth model...")

            # Try the official ZoeDepth repo
            self._model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_NK",  # NYU + KITTI pretrained (metric depth)
                pretrained=True,
                trust_repo=True,
            )
            self._model = self._model.to(self.device)
            self._model.eval()

            logger.info("ZoeDepth loaded successfully")

        except Exception as e:
            logger.warning(f"ZoeDepth not available: {e}. Falling back to MiDaS.")
            self.model_name = "midas"
            self._load_midas()

    def _load_midas(self) -> None:
        """Load MiDaS model."""
        try:
            logger.info("Loading MiDaS model...")

            # MiDaS via torch hub
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",  # Smaller, faster model
                pretrained=True,
                trust_repo=True,
            )
            self._model = self._model.to(self.device)
            self._model.eval()

            # Get MiDaS transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = midas_transforms.small_transform

            logger.info("MiDaS loaded successfully")
            logger.warning(
                "MiDaS provides relative depth only. "
                "Absolute distances require scale calibration."
            )

        except Exception as e:
            logger.warning(f"MiDaS not available: {e}. Falling back to Depth Anything.")
            self.model_name = "depth_anything"
            self._load_depth_anything()

    def _load_depth_anything(self) -> None:
        """Load Depth Anything model."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = "depth-anything/Depth-Anything-V2-Small-hf"

            logger.info(f"Loading Depth Anything from {model_id}...")

            self._transform = AutoImageProcessor.from_pretrained(
                model_id, cache_dir=self.cache_dir
            )
            self._model = AutoModelForDepthEstimation.from_pretrained(
                model_id, cache_dir=self.cache_dir
            ).to(self.device)
            self._model.eval()

            logger.info("Depth Anything loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                f"No depth model available. Install one of: "
                f"zoedepth, midas, or transformers (depth_anything). "
                f"Error: {e}"
            )

    def estimate(
        self,
        image: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Estimate depth from a single RGB image.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), dtype uint8
            output_size: Optional (H, W) to resize output depth map.
                         If None, matches input image size.

        Returns:
            Depth map as numpy array, shape (H, W), values in meters
            (for ZoeDepth) or relative units (for MiDaS)
        """
        self._ensure_loaded()

        h, w = image.shape[:2]
        if output_size is None:
            output_size = (h, w)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        if self.model_name == "zoedepth":
            depth = self._estimate_zoedepth(image)
        elif self.model_name == "midas":
            depth = self._estimate_midas(image)
        elif self.model_name == "depth_anything":
            depth = self._estimate_depth_anything(image)
        else:
            raise RuntimeError(f"Model {self.model_name} not properly loaded")

        # Resize to output size if needed
        if depth.shape != output_size:
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
            depth_tensor = F.interpolate(
                depth_tensor,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            depth = depth_tensor.squeeze().numpy()

        return depth

    def _estimate_zoedepth(self, image: np.ndarray) -> np.ndarray:
        """Run ZoeDepth estimation."""
        from PIL import Image

        pil_image = Image.fromarray(image)

        with torch.no_grad():
            # ZoeDepth expects PIL image and returns metric depth
            depth = self._model.infer_pil(pil_image, output_type="numpy")

        return depth

    def _estimate_midas(self, image: np.ndarray) -> np.ndarray:
        """Run MiDaS estimation."""
        # Apply MiDaS transform
        input_batch = self._transform(image).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_batch)

            # MiDaS outputs inverse depth, convert to depth
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert inverse depth to depth (relative scale)
        depth = prediction.cpu().numpy()
        # Invert and normalize to reasonable range
        depth = 1.0 / (depth + 1e-6)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = depth * 2.0  # Scale to ~0-2m range (heuristic)

        return depth

    def _estimate_depth_anything(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Anything estimation."""
        from PIL import Image

        pil_image = Image.fromarray(image)

        # Process image
        inputs = self._transform(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        depth = prediction.squeeze().cpu().numpy()

        # Depth Anything outputs relative depth, normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = depth * 2.0  # Scale to ~0-2m range (heuristic)

        return depth

    def estimate_batch(
        self,
        images: list[np.ndarray],
        output_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        """Estimate depth for multiple images.

        Args:
            images: List of RGB images
            output_size: Optional output size for all depth maps

        Returns:
            List of depth maps, one per image
        """
        # For now, process sequentially
        # TODO: Implement true batch processing for efficiency
        return [self.estimate(img, output_size) for img in images]

    @property
    def is_metric(self) -> bool:
        """Whether the model provides metric (absolute) depth."""
        return self.model_name == "zoedepth"

    def get_depth_at_point(
        self,
        depth_map: np.ndarray,
        point: tuple[int, int],
        kernel_size: int = 5,
    ) -> float:
        """Get depth at a specific pixel location with local averaging.

        Args:
            depth_map: Depth map from estimate()
            point: (u, v) pixel coordinates
            kernel_size: Size of averaging kernel (reduces noise)

        Returns:
            Depth value in meters (or relative units)
        """
        u, v = point
        h, w = depth_map.shape

        # Compute kernel bounds
        half_k = kernel_size // 2
        u_min, u_max = max(0, u - half_k), min(w, u + half_k + 1)
        v_min, v_max = max(0, v - half_k), min(h, v + half_k + 1)

        # Extract region and compute median (robust to outliers)
        region = depth_map[v_min:v_max, u_min:u_max]
        return float(np.median(region))


class MockDepthEstimator:
    """Mock depth estimator for testing without GPU/model weights."""

    def __init__(self, **kwargs) -> None:
        logger.info("Using MockDepthEstimator (synthetic depth)")

    def estimate(
        self,
        image: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return synthetic depth map for testing."""
        h, w = image.shape[:2]
        if output_size:
            h, w = output_size

        # Generate smooth synthetic depth (center closer, edges farther)
        y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        distance_from_center = np.sqrt(x**2 + y**2)

        # Depth increases with distance from center
        depth = 0.5 + distance_from_center * 0.5

        # Add some noise
        depth += np.random.normal(0, 0.02, depth.shape)

        return depth.astype(np.float32)

    @property
    def is_metric(self) -> bool:
        return False

    def get_depth_at_point(
        self,
        depth_map: np.ndarray,
        point: tuple[int, int],
        kernel_size: int = 5,
    ) -> float:
        u, v = point
        return float(depth_map[v, u])

