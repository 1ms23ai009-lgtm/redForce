"""Experience replay buffer for PPO training."""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class Experience:
    """Single experience tuple for RL training."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class ReplayBuffer:
    """Experience replay buffer for PPO training.

    Stores trajectories and provides batched sampling for training.
    """

    def __init__(self, capacity: int = 10000):
        self._buffer: deque[Experience] = deque(maxlen=capacity)
        self._episode_buffer: list[Experience] = []

    def add(self, experience: Experience) -> None:
        """Add a single experience to the buffer."""
        self._buffer.append(experience)
        self._episode_buffer.append(experience)

    def end_episode(self) -> list[Experience]:
        """Mark end of episode and return the episode's experiences."""
        episode = list(self._episode_buffer)
        self._episode_buffer = []
        return episode

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a random batch of experiences."""
        batch_size = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), batch_size)

    def get_all(self) -> list[Experience]:
        """Get all experiences in the buffer."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._episode_buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def to_tensors(
        self, experiences: Optional[list[Experience]] = None, device: str = "cpu"
    ) -> dict:
        """Convert experiences to PyTorch tensors for training.

        Args:
            experiences: List of experiences (defaults to all in buffer)
            device: torch device

        Returns:
            Dict with states, actions, rewards, next_states, dones, log_probs, values
        """
        if experiences is None:
            experiences = self.get_all()

        if not experiences:
            return {}

        return {
            "states": torch.FloatTensor(
                np.array([e.state for e in experiences])
            ).to(device),
            "actions": torch.LongTensor(
                [e.action for e in experiences]
            ).to(device),
            "rewards": torch.FloatTensor(
                [e.reward for e in experiences]
            ).to(device),
            "next_states": torch.FloatTensor(
                np.array([e.next_state for e in experiences])
            ).to(device),
            "dones": torch.FloatTensor(
                [float(e.done) for e in experiences]
            ).to(device),
            "log_probs": torch.FloatTensor(
                [e.log_prob for e in experiences]
            ).to(device),
            "values": torch.FloatTensor(
                [e.value for e in experiences]
            ).to(device),
        }


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: List of rewards
        values: List of state values
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        (advantages, returns) - both as lists of floats
    """
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        if dones[t]:
            next_value = 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * (0.0 if dones[t] else 1.0) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
