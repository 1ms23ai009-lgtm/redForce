"""High-level RL policy - the strategic brain of REDFORGE.

Selects which attack strategy to pursue next based on a 67-dimensional
state vector. Uses a PPO-trained policy network with separate actor and
critic heads.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from redforge.config import ACTION_TO_STRATEGY, STRATEGY_CATEGORIES
from redforge.core.state import AttackStateObject, OWASP_LLM_CATEGORIES
from redforge.graph.nash import encode_nash_signal

logger = logging.getLogger(__name__)

# State vector dimensions
STATE_DIM = 67
ACTION_DIM = 8

# Per-category encoding: (attempts, successes, partial_bypasses, last_5_rewards)
# 7 categories * 4 floats + last_5 (5 floats per category but we pack into 4) = wouldn't add up
# Spec says 32 floats for attack history, so 7 categories * ~4.57 =>
# Actually: 7 categories x (attempts, successes, partial_bypasses, last_5_mean_reward) = 7*4 = 28
# plus 4 more for aggregate stats = 32
ATTACK_HISTORY_DIM = 32
COVERAGE_DIM = 20  # 10 categories x (tested, finding_count)
TOPOLOGY_DIM = 5   # has_tools, has_memory, has_multi_agent, num_agents, num_tools
BUDGET_DIM = 1     # normalized remaining/total
NASH_DIM = 9       # top 3 paths x (probability, effort_score, path_length_norm)


class PolicyNetwork(nn.Module):
    """PPO actor-critic policy network."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()

        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Linear(128, action_dim)

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value.

        Args:
            state: (batch, state_dim) tensor

        Returns:
            (action_logits, state_value)
        """
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Select an action given a state.

        Args:
            state: (state_dim,) tensor
            deterministic: If True, select argmax instead of sampling

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO training.

        Args:
            states: (batch, state_dim)
            actions: (batch,) action indices

        Returns:
            (log_probs, values, entropy)
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class HighLevelPolicy:
    """High-level RL policy for strategy selection."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.network = PolicyNetwork().to(device)

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"Loaded policy from {model_path}")

    def select_action(
        self,
        aso: AttackStateObject,
        deterministic: bool = False,
    ) -> tuple[str, float, float]:
        """Select the next attack strategy.

        Args:
            aso: Current attack state object
            deterministic: If True, always select the best action

        Returns:
            (strategy_name, log_prob, value)
        """
        state_vec = self.encode_state(aso)
        state_tensor = torch.FloatTensor(state_vec).to(self.device)

        action_idx, log_prob, value = self.network.get_action(
            state_tensor, deterministic=deterministic
        )

        strategy = ACTION_TO_STRATEGY[action_idx]

        # Check if strategy has been tried too many times without reward
        if self.strategy_already_tried(aso, strategy):
            # Force exploration: mask this action and re-select
            action_idx, log_prob, value = self._force_explore(
                state_tensor, action_idx, aso
            )
            strategy = ACTION_TO_STRATEGY[action_idx]

        return strategy, log_prob, value

    def encode_state(self, aso: AttackStateObject) -> np.ndarray:
        """Encode the ASO into a 67-dimensional state vector.

        Layout:
        [0:32]  - Attack history vector (32 floats)
        [32:52] - Coverage state vector (20 floats)
        [52:57] - Topology discovery vector (5 floats)
        [57:58] - Budget remaining (1 float)
        [58:67] - Nash equilibrium signal (9 floats)
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)

        # Attack history encoding (32 floats)
        history = aso.get("attack_history", [])
        categories = [
            "jailbreak", "prompt_injection_direct", "prompt_injection_indirect",
            "cross_agent_injection", "data_exfiltration", "tool_abuse",
            "memory_poisoning",
        ]
        for cat_idx, cat in enumerate(categories):
            cat_entries = [e for e in history if e.get("strategy_category") == cat]
            base = cat_idx * 4
            state[base] = min(len(cat_entries) / 10.0, 1.0)  # attempts (norm)
            successes = sum(
                1 for e in cat_entries
                if isinstance(e.get("judge_verdict"), dict)
                and e["judge_verdict"].get("verdict") == "CONFIRMED"
            )
            state[base + 1] = min(successes / 5.0, 1.0)  # successes (norm)
            partials = sum(
                1 for e in cat_entries
                if isinstance(e.get("judge_verdict"), dict)
                and e["judge_verdict"].get("verdict") == "PARTIAL"
            )
            state[base + 2] = min(partials / 5.0, 1.0)  # partials (norm)
            # Last 5 rewards mean
            last5 = cat_entries[-5:]
            if last5:
                rewards = []
                for e in last5:
                    v = e.get("judge_verdict", {})
                    if isinstance(v, dict):
                        if v.get("verdict") == "CONFIRMED":
                            rewards.append(1.0)
                        elif v.get("verdict") == "PARTIAL":
                            rewards.append(0.3)
                        else:
                            rewards.append(-0.1)
                state[base + 3] = np.mean(rewards) if rewards else 0.0

        # Aggregate attack stats (fills remaining 4 floats of the 32)
        state[28] = min(len(history) / 100.0, 1.0)  # total attempts
        state[29] = min(len(aso.get("confirmed_exploits", [])) / 10.0, 1.0)
        state[30] = min(len(aso.get("partial_bypass_registry", [])) / 10.0, 1.0)
        state[31] = min(len(aso.get("active_hypotheses", [])) / 10.0, 1.0)

        # Coverage state (20 floats: 10 categories x 2)
        coverage = aso.get("coverage_state", {})
        for cat_idx, cat in enumerate(OWASP_LLM_CATEGORIES):
            base = 32 + cat_idx * 2
            cat_state = coverage.get(cat, {})
            state[base] = 1.0 if cat_state.get("tested", False) else 0.0
            state[base + 1] = min(cat_state.get("finding_count", 0) / 5.0, 1.0)

        # Topology discovery (5 floats)
        topology = aso.get("target_topology", {})
        state[52] = 1.0 if topology.get("tools") else 0.0
        state[53] = 1.0 if topology.get("memory_systems") else 0.0
        state[54] = 1.0 if len(topology.get("agents", [])) > 1 else 0.0
        state[55] = min(len(topology.get("agents", [])) / 5.0, 1.0)
        state[56] = min(len(topology.get("tools", [])) / 10.0, 1.0)

        # Budget remaining (1 float)
        total_budget = aso.get("target_config", {}).get(
            "authorization_scope", {}
        ).get("query_budget", 100)
        remaining = aso.get("query_budget_remaining", 0)
        state[57] = remaining / max(total_budget, 1)

        # Nash equilibrium signal (9 floats)
        nash = aso.get("nash_equilibrium")
        nash_signal = encode_nash_signal(nash)
        state[58:67] = nash_signal

        return state

    def strategy_already_tried(
        self, aso: AttackStateObject, strategy: str
    ) -> bool:
        """Check if strategy has been tried > 3 times without any reward signal.

        Returns True if we should force exploration of alternatives.
        """
        history = aso.get("attack_history", [])
        cat_entries = [
            e for e in history if e.get("strategy_category") == strategy
        ]

        if len(cat_entries) <= 3:
            return False

        # Check last 3 entries for any positive signal
        last3 = cat_entries[-3:]
        for e in last3:
            verdict = e.get("judge_verdict", {})
            if isinstance(verdict, dict):
                if verdict.get("verdict") in ("CONFIRMED", "PARTIAL"):
                    return False

        return True

    def _force_explore(
        self,
        state_tensor: torch.Tensor,
        excluded_action: int,
        aso: AttackStateObject,
    ) -> tuple[int, float, float]:
        """Force exploration by masking the excluded action."""
        with torch.no_grad():
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            logits, value = self.network.forward(state_tensor)

            # Mask exhausted strategies
            mask = torch.zeros(ACTION_DIM)
            mask[excluded_action] = float("-inf")

            # Also mask other exhausted strategies
            for i in range(ACTION_DIM):
                strategy = ACTION_TO_STRATEGY[i]
                if self.strategy_already_tried(aso, strategy):
                    mask[i] = float("-inf")

            masked_logits = logits + mask
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def save(self, path: str) -> None:
        """Save the policy network."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.network.state_dict(), path)
        logger.info(f"Policy saved to {path}")

    def load(self, path: str) -> None:
        """Load the policy network."""
        self.network.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.network.eval()
