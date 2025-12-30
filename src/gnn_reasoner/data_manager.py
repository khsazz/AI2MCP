"""Data Manager for LeRobot dataset ingestion.

Provides high-level API for accessing imitation learning trajectories
from Hugging Face LeRobot datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class ActionSpace:
    """Describes the action space of the robot."""

    dim: int
    low: np.ndarray
    high: np.ndarray
    names: list[str]


@dataclass
class Episode:
    """A single demonstration episode."""

    episode_id: int
    states: Tensor  # (T, state_dim)
    actions: Tensor  # (T, action_dim)
    images: dict[str, Tensor] | None  # camera_name -> (T, C, H, W)
    length: int


class DataManager:
    """Manages LeRobot dataset loading and trajectory access.

    Supports both full dataset loading and streaming mode for large-scale
    datasets to minimize local disk footprint.

    Example:
        >>> dm = DataManager("lerobot/aloha_static_towel")
        >>> frame = dm.get_frame(0)
        >>> state = frame["observation.state"]
    """

    def __init__(self, repo_id: str, streaming: bool = True) -> None:
        """Initialize the data manager.

        Args:
            repo_id: Hugging Face repository ID (e.g., "lerobot/aloha_static_towel")
            streaming: If True, use streaming mode to minimize disk usage
        """
        self.repo_id = repo_id
        self.streaming = streaming
        self._dataset: LeRobotDataset | None = None
        self._episode_boundaries: list[tuple[int, int]] | None = None
        self._current_frame_idx: int = 0

    @property
    def dataset(self) -> LeRobotDataset:
        """Lazy-load the dataset on first access."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def _load_dataset(self) -> LeRobotDataset:
        """Load the LeRobot dataset."""
        try:
            # New LeRobot structure (v0.2+)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            # Fallback to old structure (v0.1.x)
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        # LeRobotDataset handles streaming internally via HF datasets
        dataset = LeRobotDataset(self.repo_id)
        return dataset

    def _compute_episode_boundaries(self) -> list[tuple[int, int]]:
        """Compute start/end indices for each episode."""
        if self._episode_boundaries is not None:
            return self._episode_boundaries

        boundaries = []
        episode_ids = self.dataset.hf_dataset["episode_index"]

        if len(episode_ids) == 0:
            return boundaries

        current_episode = episode_ids[0]
        start_idx = 0

        for i, ep_id in enumerate(episode_ids):
            if ep_id != current_episode:
                boundaries.append((start_idx, i))
                start_idx = i
                current_episode = ep_id

        # Add final episode
        boundaries.append((start_idx, len(episode_ids)))
        self._episode_boundaries = boundaries
        return boundaries

    def __len__(self) -> int:
        """Return total number of frames in the dataset."""
        return len(self.dataset)

    def get_frame(self, idx: int) -> dict:
        """Get a single frame from the dataset.

        Args:
            idx: Frame index

        Returns:
            Dictionary containing:
                - observation.state: Joint positions/velocities
                - observation.images.*: Camera images (if available)
                - action: Action taken at this timestep
        """
        self._current_frame_idx = idx
        return self.dataset[idx]

    def get_current_frame(self) -> dict:
        """Get the current frame (for MCP resource exposure)."""
        return self.get_frame(self._current_frame_idx)

    def advance_frame(self) -> dict:
        """Advance to the next frame and return it."""
        self._current_frame_idx = min(self._current_frame_idx + 1, len(self) - 1)
        return self.get_current_frame()

    def set_frame_index(self, idx: int) -> None:
        """Set the current frame index."""
        self._current_frame_idx = max(0, min(idx, len(self) - 1))

    def get_state(self, idx: int) -> Tensor:
        """Get observation state for a given frame.

        Args:
            idx: Frame index

        Returns:
            State tensor of shape (state_dim,)
        """
        frame = self.get_frame(idx)
        state = frame.get("observation.state")
        if state is None:
            raise KeyError(f"No 'observation.state' found in frame {idx}")
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        return state

    def get_action(self, idx: int) -> Tensor:
        """Get action for a given frame.

        Args:
            idx: Frame index

        Returns:
            Action tensor of shape (action_dim,)
        """
        frame = self.get_frame(idx)
        action = frame.get("action")
        if action is None:
            raise KeyError(f"No 'action' found in frame {idx}")
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        return action

    def get_images(self, idx: int) -> dict[str, Tensor]:
        """Get all camera images for a given frame.

        Args:
            idx: Frame index

        Returns:
            Dictionary mapping camera names to image tensors
        """
        frame = self.get_frame(idx)
        images = {}
        for key, value in frame.items():
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                images[camera_name] = value
        return images

    def iter_episodes(self) -> Iterator[Episode]:
        """Iterate over all episodes in the dataset.

        Yields:
            Episode objects containing full trajectory data
        """
        boundaries = self._compute_episode_boundaries()

        for episode_id, (start, end) in enumerate(boundaries):
            states = []
            actions = []
            images_by_camera: dict[str, list[Tensor]] = {}

            for idx in range(start, end):
                frame = self.get_frame(idx)

                # Collect state
                state = frame.get("observation.state")
                if state is not None:
                    if isinstance(state, np.ndarray):
                        state = torch.from_numpy(state)
                    states.append(state)

                # Collect action
                action = frame.get("action")
                if action is not None:
                    if isinstance(action, np.ndarray):
                        action = torch.from_numpy(action)
                    actions.append(action)

                # Collect images
                for key, value in frame.items():
                    if key.startswith("observation.images."):
                        camera = key.replace("observation.images.", "")
                        if camera not in images_by_camera:
                            images_by_camera[camera] = []
                        if isinstance(value, np.ndarray):
                            value = torch.from_numpy(value)
                        images_by_camera[camera].append(value)

            # Stack tensors
            states_tensor = torch.stack(states) if states else torch.empty(0)
            actions_tensor = torch.stack(actions) if actions else torch.empty(0)

            images_tensor = None
            if images_by_camera:
                images_tensor = {
                    cam: torch.stack(imgs) for cam, imgs in images_by_camera.items()
                }

            yield Episode(
                episode_id=episode_id,
                states=states_tensor,
                actions=actions_tensor,
                images=images_tensor,
                length=end - start,
            )

    def get_action_space(self) -> ActionSpace:
        """Get the action space specification.

        Returns:
            ActionSpace describing the robot's action dimensions
        """
        # Get a sample action to determine dimensions
        sample_action = self.get_action(0)
        action_dim = sample_action.shape[-1]

        # LeRobot datasets typically have normalized actions in [-1, 1]
        # but we can try to infer from metadata if available
        try:
            info = self.dataset.info
            action_names = info.get("action_names", [f"action_{i}" for i in range(action_dim)])
        except (AttributeError, KeyError):
            action_names = [f"action_{i}" for i in range(action_dim)]

        return ActionSpace(
            dim=action_dim,
            low=np.full(action_dim, -1.0),
            high=np.full(action_dim, 1.0),
            names=list(action_names),
        )

    @property
    def state_dim(self) -> int:
        """Get the dimension of the observation state."""
        sample_state = self.get_state(0)
        return sample_state.shape[-1]

    @property
    def action_dim(self) -> int:
        """Get the dimension of actions."""
        return self.get_action_space().dim

    @property
    def num_episodes(self) -> int:
        """Get the number of episodes in the dataset."""
        return len(self._compute_episode_boundaries())

