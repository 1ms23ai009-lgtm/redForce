"""MDP definitions and reward computation for the REDFORGE system.

Defines the two-level Markov Decision Process:
- High-level: strategy selection (which attack category)
- Low-level: prompt generation and refinement
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from redforge.config import ACTION_TO_STRATEGY, STRATEGY_TO_ACTION
from redforge.core.state import AttackStateObject, OWASP_LLM_CATEGORIES
from redforge.rl.reward import compute_reward


@dataclass
class MDPTransition:
    """A single MDP transition record."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict


def compute_coverage_percentage(aso: AttackStateObject) -> float:
    """Compute the percentage of OWASP categories tested."""
    coverage = aso.get("coverage_state", {})
    if not coverage:
        return 0.0
    tested = sum(1 for v in coverage.values() if v.get("tested", False))
    return tested / len(OWASP_LLM_CATEGORIES) * 100


def update_coverage_state(
    aso: AttackStateObject,
    owasp_category: str,
    severity: Optional[str] = None,
) -> None:
    """Update the coverage state for a given OWASP category."""
    coverage = aso.get("coverage_state", {})
    if owasp_category not in coverage:
        return

    state = coverage[owasp_category]
    state["tested"] = True

    if severity:
        state["finding_count"] = state.get("finding_count", 0) + 1
        current_severity = state.get("highest_severity", "none")
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        if severity_order.get(severity, 0) > severity_order.get(current_severity, 0):
            state["highest_severity"] = severity


def check_engagement_done(aso: AttackStateObject) -> bool:
    """Check if the engagement should end."""
    if aso.get("kill_switch_triggered", False):
        return True
    if aso.get("query_budget_remaining", 0) <= 0:
        return True
    return False


def get_strategy_for_action(action: int) -> str:
    """Map action index to strategy name."""
    return ACTION_TO_STRATEGY.get(action, "jailbreak")


def get_action_for_strategy(strategy: str) -> int:
    """Map strategy name to action index."""
    return STRATEGY_TO_ACTION.get(strategy, 0)
