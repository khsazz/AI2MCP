"""Open-Vocabulary Object Detection using DETIC.

Provides object detection with arbitrary text prompts for dynamic
object node creation in the world graph.

Supports:
- DETIC (Detectron2-based, 21k vocabulary + custom prompts)
- GroundingDINO (transformer-based, open-vocab)
- YOLOv8 (fast, but fixed vocabulary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    mask: np.ndarray | None = None  # Optional instance segmentation mask
    center: tuple[int, int] = field(init=False)  # Bbox center (u, v)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": list(self.bbox),
            "center": list(self.center),
            "area": self.area,
        }


class VisionDetector:
    """Open-vocabulary object detector.

    Wraps DETIC, GroundingDINO, or YOLOv8 with a unified interface.

    Example:
        >>> detector = VisionDetector(model="detic", device="cuda")
        >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> detections = detector.detect(image, prompts=["cup", "mug", "bottle"])
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f} at {det.bbox}")
    """

    # Default prompts for robotic manipulation tasks
    DEFAULT_PROMPTS: list[str] = [
        "cup",
        "mug",
        "bottle",
        "bowl",
        "plate",
        "spoon",
        "fork",
        "knife",
        "box",
        "container",
        "drawer",
        "button",
        "handle",
        "lid",
        "object",
    ]

    def __init__(
        self,
        model: Literal["detic", "grounding_dino", "yolov8"] = "detic",
        device: str | None = None,
        confidence_threshold: float = 0.3,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            model: Detection model to use
            device: Device for inference ("cuda", "cpu", or None for auto)
            confidence_threshold: Minimum confidence for detections
            cache_dir: Directory for model weights cache
        """
        self.model_name = model
        self.confidence_threshold = confidence_threshold
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ai2mcp"

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._processor = None
        self._is_loaded = False

        logger.info(f"VisionDetector initialized: model={model}, device={self.device}")

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._is_loaded:
            return

        if self.model_name == "detic":
            self._load_detic()
        elif self.model_name == "grounding_dino":
            self._load_grounding_dino()
        elif self.model_name == "yolov8":
            self._load_yolov8()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self._is_loaded = True

    def _load_detic(self) -> None:
        """Load DETIC model via Detectron2."""
        try:
            # DETIC requires detectron2 and custom installation
            # For now, we'll use a fallback to GroundingDINO if DETIC unavailable
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor

            logger.info("Loading DETIC model...")

            # DETIC config setup would go here
            # This is a placeholder - actual DETIC setup requires more config
            cfg = get_cfg()
            # Add DETIC-specific config
            # cfg.merge_from_file("path/to/detic_config.yaml")
            # self._model = DefaultPredictor(cfg)

            raise NotImplementedError(
                "DETIC requires manual setup. Falling back to GroundingDINO."
            )

        except (ImportError, NotImplementedError) as e:
            logger.warning(f"DETIC not available: {e}. Falling back to GroundingDINO.")
            self.model_name = "grounding_dino"
            self._load_grounding_dino()

    def _load_grounding_dino(self) -> None:
        """Load GroundingDINO model via transformers."""
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            model_id = "IDEA-Research/grounding-dino-tiny"

            logger.info(f"Loading GroundingDINO from {model_id}...")

            self._processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=self.cache_dir
            )
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, cache_dir=self.cache_dir
            ).to(self.device)
            self._model.eval()

            logger.info("GroundingDINO loaded successfully")

        except ImportError as e:
            logger.warning(f"GroundingDINO not available: {e}. Falling back to YOLOv8.")
            self.model_name = "yolov8"
            self._load_yolov8()

    def _load_yolov8(self) -> None:
        """Load YOLOv8 model via ultralytics."""
        try:
            from ultralytics import YOLO

            logger.info("Loading YOLOv8...")

            # Use YOLOv8n (nano) for speed, or YOLOv8s/m for accuracy
            self._model = YOLO("yolov8n.pt")

            logger.info("YOLOv8 loaded successfully")
            logger.warning(
                "YOLOv8 uses fixed COCO vocabulary (80 classes). "
                "Custom prompts will be mapped to nearest COCO class."
            )

        except ImportError as e:
            raise RuntimeError(
                f"No detection model available. Install one of: "
                f"detectron2 (DETIC), transformers (GroundingDINO), or ultralytics (YOLOv8). "
                f"Error: {e}"
            )

    def detect(
        self,
        image: np.ndarray,
        prompts: list[str] | None = None,
    ) -> list[Detection]:
        """Detect objects in an image.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), dtype uint8
            prompts: Text prompts for open-vocabulary detection.
                     Ignored for YOLOv8 (fixed vocabulary).

        Returns:
            List of Detection objects sorted by confidence (descending)
        """
        self._ensure_loaded()

        if prompts is None:
            prompts = self.DEFAULT_PROMPTS

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        if self.model_name == "grounding_dino":
            return self._detect_grounding_dino(image, prompts)
        elif self.model_name == "yolov8":
            return self._detect_yolov8(image)
        else:
            raise RuntimeError(f"Model {self.model_name} not properly loaded")

    def _detect_grounding_dino(
        self,
        image: np.ndarray,
        prompts: list[str],
    ) -> list[Detection]:
        """Run GroundingDINO detection."""
        from PIL import Image

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Prepare text prompt (GroundingDINO expects "." separated classes)
        text_prompt = ". ".join(prompts) + "."

        # Process inputs
        inputs = self._processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process - API changed in newer transformers versions
        # Try new API first, fall back to old API
        try:
            # New API (transformers >= 4.40): threshold instead of box_threshold/text_threshold
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]],  # (H, W)
            )[0]
        except TypeError:
            # Old API (transformers < 4.40)
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]],  # (H, W)
            )[0]

        detections = []
        for box, score, label in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["labels"],
        ):
            x1, y1, x2, y2 = box.astype(int)
            detections.append(
                Detection(
                    class_name=label,
                    confidence=float(score),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _detect_yolov8(self, image: np.ndarray) -> list[Detection]:
        """Run YOLOv8 detection."""
        # Run inference
        results = self._model(image, verbose=False)[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_batch(
        self,
        images: list[np.ndarray],
        prompts: list[str] | None = None,
    ) -> list[list[Detection]]:
        """Detect objects in multiple images.

        Args:
            images: List of RGB images
            prompts: Text prompts for detection

        Returns:
            List of detection lists, one per image
        """
        # For now, process sequentially
        # TODO: Implement true batch processing for efficiency
        return [self.detect(img, prompts) for img in images]

    @property
    def is_open_vocabulary(self) -> bool:
        """Whether the model supports custom text prompts."""
        return self.model_name in ("detic", "grounding_dino")


class MockVisionDetector:
    """Mock detector for testing without GPU/model weights.
    
    Supports generating detections at specific image locations to simulate
    objects near the gripper for training is_holding predicates.
    """

    def __init__(self, **kwargs) -> None:
        logger.info("Using MockVisionDetector (no real detection)")

    def detect(
        self,
        image: np.ndarray,
        prompts: list[str] | None = None,
        gripper_pixel: tuple[int, int] | None = None,
        spawn_at_gripper: bool = False,
    ) -> list[Detection]:
        """Return synthetic detections for testing.
        
        Args:
            image: Input image (H, W, 3)
            prompts: Object class names to use
            gripper_pixel: Optional (u, v) pixel location of gripper
            spawn_at_gripper: If True and gripper_pixel provided, spawn 
                             an object detection at the gripper location
                             (for simulating holding scenarios)
        
        Returns:
            List of Detection objects
        """
        h, w = image.shape[:2]
        detections = []
        
        # Spawn object at gripper location (for holding scenarios)
        if spawn_at_gripper and gripper_pixel is not None:
            cx, cy = gripper_pixel
            # Small bbox at gripper location (held object)
            bw, bh = np.random.randint(40, 80), np.random.randint(40, 80)
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)
            
            class_name = prompts[0] if prompts else "held_object"
            
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=np.random.uniform(0.85, 0.98),
                    bbox=(x1, y1, x2, y2),
                )
            )

        # Generate 1-3 additional random detections
        num_detections = np.random.randint(1, 4)

        for i in range(num_detections):
            # Random bounding box
            cx, cy = np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4)
            bw, bh = np.random.randint(50, 150), np.random.randint(50, 150)
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)

            class_name = prompts[i % len(prompts)] if prompts else f"object_{i}"

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=np.random.uniform(0.5, 0.95),
                    bbox=(x1, y1, x2, y2),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
    
    def detect_at_position(
        self,
        image: np.ndarray,
        position_3d: np.ndarray,
        class_name: str = "held_object",
        size_px: int = 60,
    ) -> Detection:
        """Create a detection at a specific 3D position.
        
        This is useful for simulating objects exactly at gripper position
        for training is_holding predicates.
        
        Args:
            image: Input image for getting dimensions
            position_3d: 3D world position (x, y, z)
            class_name: Object class name
            size_px: Bounding box size in pixels
            
        Returns:
            Single Detection at the specified position
        """
        h, w = image.shape[:2]
        
        # Simple orthographic projection (assumes camera at origin looking at +z)
        # Scale factor to convert meters to pixels (rough approximation)
        scale = 500  # pixels per meter at 1m distance
        
        # Project to image plane
        cx = int(w / 2 + position_3d[0] * scale)
        cy = int(h / 2 - position_3d[1] * scale)  # y is flipped in image coords
        
        # Clamp to image bounds
        cx = np.clip(cx, size_px // 2, w - size_px // 2)
        cy = np.clip(cy, size_px // 2, h - size_px // 2)
        
        x1, y1 = cx - size_px // 2, cy - size_px // 2
        x2, y2 = cx + size_px // 2, cy + size_px // 2
        
        return Detection(
            class_name=class_name,
            confidence=np.random.uniform(0.9, 0.99),
            bbox=(x1, y1, x2, y2),
        )

    @property
    def is_open_vocabulary(self) -> bool:
        return True
