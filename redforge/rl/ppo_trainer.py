"""PPO training loop for the high-level policy."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from redforge.rl.high_level_policy import PolicyNetwork, STATE_DIM, ACTION_DIM
from redforge.rl.replay_buffer import ReplayBuffer, Experience, compute_gae

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Proximal Policy Optimization trainer for the high-level policy."""

    def __init__(
        self,
        policy: PolicyNetwork,
        clip_epsilon: float = 0.2,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coefficient: float = 0.01,
        value_loss_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cpu",
    ):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coefficient
        self.value_loss_coeff = value_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer()

        # Training metrics
        self._total_updates = 0
        self._episode_rewards: list[float] = []

    def collect_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Add experience to the buffer."""
        self.buffer.add(
            Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
            )
        )

    def train_step(self) -> dict:
        """Perform a PPO training step using collected experiences.

        Returns:
            Dict with training metrics
        """
        experiences = self.buffer.get_all()
        if len(experiences) < self.batch_size:
            return {"status": "insufficient_data", "buffer_size": len(experiences)}

        # Compute GAE
        rewards = [e.reward for e in experiences]
        values = [e.value for e in experiences]
        dones = [e.done for e in experiences]
        advantages, returns = compute_gae(
            rewards, values, dones, self.gamma, self.gae_lambda
        )

        # Convert to tensors
        states = torch.FloatTensor(
            np.array([e.state for e in experiences])
        ).to(self.device)
        actions = torch.LongTensor(
            [e.action for e in experiences]
        ).to(self.device)
        old_log_probs = torch.FloatTensor(
            [e.log_prob for e in experiences]
        ).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.n_epochs):
            # Create random mini-batches
            indices = np.random.permutation(len(experiences))

            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # Evaluate current policy
                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * batch_advantages, clipped_ratio * batch_advantages
                ).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        self._total_updates += 1

        # Clear buffer after training
        self.buffer.clear()

        metrics = {
            "status": "trained",
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
            "total_updates": self._total_updates,
            "experiences_used": len(experiences),
        }

        logger.info(
            f"PPO update {self._total_updates}: "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}"
        )

        return metrics

    def train_episodes(
        self,
        env_step_fn,
        n_episodes: int = 100,
        max_steps_per_episode: int = 100,
        save_path: Optional[str] = None,
        save_every: int = 10,
    ) -> list[dict]:
        """Train over multiple episodes.

        Args:
            env_step_fn: Callable(action) -> (next_state, reward, done, info)
                The environment step function
            n_episodes: Number of episodes to train
            max_steps_per_episode: Max steps per episode
            save_path: Optional path to save checkpoints
            save_every: Save checkpoint every N episodes

        Returns:
            List of per-episode metrics
        """
        all_metrics = []

        for episode in range(n_episodes):
            # Reset environment (caller provides initial state)
            state = np.zeros(STATE_DIM, dtype=np.float32)
            episode_reward = 0.0

            for step in range(max_steps_per_episode):
                state_tensor = torch.FloatTensor(state).to(self.device)
                action, log_prob, value = self.policy.get_action(state_tensor)

                next_state, reward, done, info = env_step_fn(action)
                episode_reward += reward

                self.collect_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )

                state = next_state
                if done:
                    break

            self._episode_rewards.append(episode_reward)

            # Train after each episode
            if len(self.buffer) >= self.batch_size:
                metrics = self.train_step()
                metrics["episode"] = episode
                metrics["episode_reward"] = episode_reward
                all_metrics.append(metrics)

            # Save checkpoint
            if save_path and (episode + 1) % save_every == 0:
                torch.save(self.policy.state_dict(), save_path)
                logger.info(f"Checkpoint saved at episode {episode + 1}")

        # Final save
        if save_path:
            torch.save(self.policy.state_dict(), save_path)

        return all_metrics

    @property
    def episode_rewards(self) -> list[float]:
        return list(self._episode_rewards)
