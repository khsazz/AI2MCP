"""Object detection wrapper for semantic perception.

Uses YOLOv8 for real-time object detection from camera images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger()


@dataclass
class Detection:
    """Single object detection result."""
    
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]
    class_id: int


@dataclass
class DetectionResult:
    """Collection of detections from a single image."""
    
    detections: list[Detection]
    image_width: int
    image_height: int
    inference_time_ms: float


class ObjectDetector:
    """YOLO-based object detector for semantic perception."""

    # Classes relevant for indoor robotics
    INDOOR_CLASSES = {
        0: "person",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        73: "book",
        74: "clock",
        75: "vase",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ):
        """Initialize detector with YOLO model.
        
        Args:
            model_path: Path to YOLO weights or model name (e.g., "yolov8n.pt")
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference ("auto", "cpu", "cuda", "mps")
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model: YOLO | None = None

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                logger.info("YOLO model loaded", model=model_path)
            except Exception as e:
                logger.warning("Failed to load YOLO model", error=str(e))
        else:
            logger.warning("Ultralytics not available - detection disabled")

    def detect(self, image: NDArray[np.uint8]) -> DetectionResult:
        """Run object detection on image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            DetectionResult with all detections
        """
        if self.model is None:
            return DetectionResult(
                detections=[],
                image_width=image.shape[1] if len(image.shape) > 1 else 0,
                image_height=image.shape[0] if len(image.shape) > 0 else 0,
                inference_time_ms=0.0,
            )

        import time
        start = time.perf_counter()

        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device if self.device != "auto" else None,
        )

        inference_time = (time.perf_counter() - start) * 1000

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # Get label from COCO classes
                label = self.model.names.get(cls_id, f"class_{cls_id}")

                detection = Detection(
                    label=label,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    class_id=cls_id,
                )
                detections.append(detection)

        return DetectionResult(
            detections=detections,
            image_width=image.shape[1],
            image_height=image.shape[0],
            inference_time_ms=inference_time,
        )

    def detect_indoor_objects(self, image: NDArray[np.uint8]) -> DetectionResult:
        """Detect only indoor-relevant objects.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Filtered DetectionResult with indoor objects only
        """
        result = self.detect(image)
        indoor_detections = [
            d for d in result.detections
            if d.class_id in self.INDOOR_CLASSES
        ]
        return DetectionResult(
            detections=indoor_detections,
            image_width=result.image_width,
            image_height=result.image_height,
            inference_time_ms=result.inference_time_ms,
        )

    def to_dict(self, result: DetectionResult) -> dict:
        """Convert detection result to JSON-serializable dict."""
        return {
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "bbox": list(d.bbox),
                    "center": list(d.center),
                }
                for d in result.detections
            ],
            "image_size": [result.image_width, result.image_height],
            "inference_time_ms": round(result.inference_time_ms, 2),
            "count": len(result.detections),
        }

