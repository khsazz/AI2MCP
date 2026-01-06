"""Camera utilities for 2D to 3D projection.

Provides functions to project pixel coordinates to 3D world coordinates
using depth maps and camera intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters.

    Follows the pinhole camera model:
        u = fx * X/Z + cx
        v = fy * Y/Z + cy

    Attributes:
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x (pixels)
        cy: Principal point y (pixels)
        width: Image width (pixels)
        height: Image height (pixels)
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> CameraIntrinsics:
        """Create from 3x3 intrinsic matrix.

        Args:
            K: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            width: Image width
            height: Image height
        """
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
        )

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    @classmethod
    def from_fov(
        cls,
        fov_horizontal: float,
        width: int,
        height: int,
    ) -> CameraIntrinsics:
        """Create from horizontal field of view.

        Args:
            fov_horizontal: Horizontal FOV in degrees
            width: Image width
            height: Image height
        """
        fov_rad = np.radians(fov_horizontal)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2

        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    @classmethod
    def default_aloha(cls) -> CameraIntrinsics:
        """Default intrinsics for ALOHA robot cameras.

        Based on typical webcam parameters. Should be calibrated for
        accurate results.
        """
        # ALOHA typically uses 640x480 USB cameras with ~60° FOV
        return cls.from_fov(fov_horizontal=60.0, width=640, height=480)

    @classmethod
    def default_realsense(cls) -> CameraIntrinsics:
        """Default intrinsics for Intel RealSense D435.

        Approximate values for color stream at 640x480.
        """
        return cls(
            fx=617.0,
            fy=617.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )


def pixel_to_camera(
    u: float,
    v: float,
    depth: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Project a pixel to 3D camera coordinates.

    Args:
        u: Pixel x coordinate
        v: Pixel y coordinate
        depth: Depth at the pixel (meters)
        intrinsics: Camera intrinsic parameters

    Returns:
        (X, Y, Z) in camera coordinate frame (meters)
        X: right, Y: down, Z: forward
    """
    X = (u - intrinsics.cx) * depth / intrinsics.fx
    Y = (v - intrinsics.cy) * depth / intrinsics.fy
    Z = depth

    return (X, Y, Z)


def camera_to_world(
    point_camera: tuple[float, float, float],
    camera_pose: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Transform from camera frame to world frame.

    Args:
        point_camera: (X, Y, Z) in camera frame
        camera_pose: 4x4 transformation matrix (camera to world)
                     If None, assumes camera at origin aligned with world

    Returns:
        (x, y, z) in world frame
    """
    if camera_pose is None:
        # Default: camera Z-forward to world X-forward, camera Y-down to world Z-down
        # This is a common robotics convention
        X, Y, Z = point_camera
        return (Z, -X, -Y)  # (x=forward, y=left, z=up)

    # Apply transformation
    point_h = np.array([*point_camera, 1.0])
    point_world = camera_pose @ point_h
    return tuple(point_world[:3])


def pixel_to_world(
    u: float,
    v: float,
    depth: float,
    intrinsics: CameraIntrinsics,
    camera_pose: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Project a pixel directly to world coordinates.

    Args:
        u: Pixel x coordinate
        v: Pixel y coordinate
        depth: Depth at the pixel (meters)
        intrinsics: Camera intrinsic parameters
        camera_pose: 4x4 camera-to-world transformation

    Returns:
        (x, y, z) in world frame (meters)
    """
    point_camera = pixel_to_camera(u, v, depth, intrinsics)
    return camera_to_world(point_camera, camera_pose)


def bbox_to_3d(
    bbox: tuple[int, int, int, int],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose: np.ndarray | None = None,
    depth_percentile: float = 50.0,
) -> tuple[float, float, float]:
    """Project a bounding box center to 3D world coordinates.

    Uses the median depth within the bbox region for robustness.

    Args:
        bbox: (x1, y1, x2, y2) bounding box in pixels
        depth_map: Depth image, shape (H, W)
        intrinsics: Camera intrinsic parameters
        camera_pose: 4x4 camera-to-world transformation
        depth_percentile: Percentile of depth values to use (50 = median)

    Returns:
        (x, y, z) in world frame (meters)
    """
    x1, y1, x2, y2 = bbox

    # Clamp to image bounds
    x1 = max(0, min(x1, intrinsics.width - 1))
    x2 = max(0, min(x2, intrinsics.width))
    y1 = max(0, min(y1, intrinsics.height - 1))
    y2 = max(0, min(y2, intrinsics.height))

    # Extract depth region
    depth_region = depth_map[y1:y2, x1:x2]

    if depth_region.size == 0:
        # Fallback to center pixel
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2
        depth = depth_map[v, u]
    else:
        # Use percentile for robustness
        depth = np.percentile(depth_region, depth_percentile)

    # Project center of bbox
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2

    return pixel_to_world(u, v, depth, intrinsics, camera_pose)


def project_detections_to_3d(
    detections: Sequence,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose: np.ndarray | None = None,
) -> list[dict]:
    """Project multiple detections to 3D world coordinates.

    Args:
        detections: List of Detection objects (from detector.py)
        depth_map: Depth image, shape (H, W)
        intrinsics: Camera intrinsic parameters
        camera_pose: 4x4 camera-to-world transformation

    Returns:
        List of dicts with detection info plus 3D position:
        [{"class_name": str, "confidence": float, "position": (x,y,z), ...}, ...]
    """
    results = []

    for det in detections:
        x, y, z = bbox_to_3d(det.bbox, depth_map, intrinsics, camera_pose)

        result = det.to_dict() if hasattr(det, "to_dict") else {}
        result.update({
            "class_name": det.class_name,
            "confidence": det.confidence,
            "bbox": det.bbox,
            "position": (x, y, z),
        })
        results.append(result)

    return results


def estimate_object_size(
    bbox: tuple[int, int, int, int],
    depth: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Estimate object size in meters from bounding box and depth.

    Args:
        bbox: (x1, y1, x2, y2) bounding box in pixels
        depth: Depth to the object (meters)
        intrinsics: Camera intrinsic parameters

    Returns:
        (width, height) in meters
    """
    x1, y1, x2, y2 = bbox
    width_px = x2 - x1
    height_px = y2 - y1

    # Project to metric dimensions
    width_m = width_px * depth / intrinsics.fx
    height_m = height_px * depth / intrinsics.fy

    return (width_m, height_m)


@dataclass
class Object3D:
    """A detected object with 3D position and size."""

    class_name: str
    confidence: float
    position: tuple[float, float, float]  # (x, y, z) in world frame
    size: tuple[float, float] | None  # (width, height) in meters
    bbox: tuple[int, int, int, int]  # Original 2D bbox

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "position": list(self.position),
            "size": list(self.size) if self.size else None,
            "bbox": list(self.bbox),
        }


def detections_to_objects_3d(
    detections: Sequence,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pose: np.ndarray | None = None,
) -> list[Object3D]:
    """Convert detections to Object3D with position and size.

    Args:
        detections: List of Detection objects
        depth_map: Depth image, shape (H, W)
        intrinsics: Camera intrinsic parameters
        camera_pose: 4x4 camera-to-world transformation

    Returns:
        List of Object3D instances
    """
    objects = []

    for det in detections:
        x, y, z = bbox_to_3d(det.bbox, depth_map, intrinsics, camera_pose)

        # Get depth at bbox center for size estimation
        u, v = det.center
        h, w = depth_map.shape
        u = min(max(0, u), w - 1)
        v = min(max(0, v), h - 1)
        center_depth = depth_map[v, u]

        size = estimate_object_size(det.bbox, center_depth, intrinsics)

        objects.append(Object3D(
            class_name=det.class_name,
            confidence=det.confidence,
            position=(x, y, z),
            size=size,
            bbox=det.bbox,
        ))

    return objects

