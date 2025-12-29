"""Pytest configuration and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_robot_pose() -> tuple[float, float, float]:
    """Sample robot pose (x, y, theta)."""
    return (0.0, 0.0, 0.0)


@pytest.fixture
def sample_scan_ranges() -> list[float]:
    """Sample LiDAR scan data (360 readings)."""
    import math
    # Simulate circular room with radius 3m
    ranges = []
    for i in range(360):
        angle = math.radians(i)
        # Add some variation
        base_range = 3.0
        if 45 <= i <= 135:  # Obstacle on left
            base_range = 1.5
        if 180 <= i <= 200:  # Obstacle behind
            base_range = 0.8
        ranges.append(base_range)
    return ranges


@pytest.fixture
def sample_obstacles() -> list[dict]:
    """Sample clustered obstacles."""
    return [
        {"world_x": 1.0, "world_y": 0.5, "distance": 1.1, "size": 0.3},
        {"world_x": -0.5, "world_y": 1.0, "distance": 1.1, "size": 0.2},
        {"world_x": 0.0, "world_y": -0.8, "distance": 0.8, "size": 0.4},
    ]

