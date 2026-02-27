"""Nash equilibrium computation for the CTR game-theoretic model.

Implements the attacker-defender game from the G-CTR paper:
- Attacker tries to reach vulnerable nodes undetected
- Defender inspects nodes to catch the attacker
- Uses Poisson distribution for progress probabilities
"""

import math
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson

from redforge.config import LAMBDA_ATTACKER, LAMBDA_DEFENDER
from redforge.graph.attack_graph import get_attack_paths, get_vulnerable_nodes


def _poisson_reach_probability(
    path_length: int,
    lambda_attacker: float = LAMBDA_ATTACKER,
    lambda_defender: float = LAMBDA_DEFENDER,
) -> float:
    """Compute probability of attacker reaching target before defender catches them.

    Uses Poisson distribution: attacker advances lambda_attacker edges per
    defender inspection round.
    """
    if path_length <= 0:
        return 1.0

    # Probability attacker completes the path in k rounds
    # before defender inspects the right node
    prob_success = 0.0
    for k in range(1, path_length + 5):  # check up to path_length + buffer
        # P(attacker advances >= path_length edges in k rounds)
        p_attacker_done = 1.0 - poisson.cdf(path_length - 1, k * lambda_attacker)

        # P(defender has NOT caught attacker in k rounds)
        # Defender inspects 1 node per round; attacker is on path_length nodes
        p_defender_miss_per_round = 1.0 - (lambda_defender / max(path_length, 1))
        p_defender_miss_per_round = max(0.0, min(1.0, p_defender_miss_per_round))
        p_not_caught = p_defender_miss_per_round ** k

        prob_success += p_attacker_done * p_not_caught * (
            poisson.pmf(k, lambda_attacker) if k > 1 else 1.0
        )

    return max(0.0, min(1.0, prob_success))


def _path_effort_score(graph: nx.DiGraph, path: list[str]) -> float:
    """Compute average effort score along a path."""
    if len(path) < 2:
        return 0.0
    scores = []
    for i in range(len(path) - 1):
        if graph.has_edge(path[i], path[i + 1]):
            scores.append(graph.edges[path[i], path[i + 1]].get("effort_score", 0.5))
    return sum(scores) / len(scores) if scores else 0.5


def compute_nash_equilibrium(graph: nx.DiGraph) -> Optional[dict]:
    """Compute Nash equilibrium for the attacker-defender game.

    Args:
        graph: The attack graph (NetworkX DiGraph)

    Returns:
        Dict with defender_strategy, attacker_strategy, equilibrium_value,
        or None if computation is not possible.
    """
    paths = get_attack_paths(graph)
    nodes = list(graph.nodes())

    if not paths or not nodes:
        return None

    num_nodes = len(nodes)
    num_paths = len(paths)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build payoff matrix: A[i][j] = attacker payoff when
    # attacker uses path i, defender inspects node j
    payoff_matrix = np.zeros((num_paths, num_nodes))

    for i, path in enumerate(paths):
        path_len = len(path) - 1
        base_prob = _poisson_reach_probability(path_len)
        effort = _path_effort_score(graph, path)

        for j, node in enumerate(nodes):
            if node in path:
                # Defender inspects a node on the attack path
                # Attacker success reduced proportionally
                position_in_path = path.index(node)
                remaining = path_len - position_in_path
                detection_prob = 1.0 / max(remaining + 1, 1)
                payoff_matrix[i][j] = base_prob * (1.0 - detection_prob) * effort
            else:
                # Defender inspects a node NOT on the path - no effect
                payoff_matrix[i][j] = base_prob * effort

    # Solve for defender's mixed strategy using linear programming
    # Defender minimizes attacker's maximum expected payoff
    # min v s.t. A @ y <= v*1, sum(y)=1, y>=0
    try:
        # Defender LP: minimize v
        # Variables: [y_0, ..., y_{n-1}, v]
        c = np.zeros(num_nodes + 1)
        c[-1] = 1.0  # minimize v

        # Constraints: for each attacker path i, sum_j A[i][j]*y[j] <= v
        # => sum_j A[i][j]*y[j] - v <= 0
        A_ub = np.zeros((num_paths, num_nodes + 1))
        b_ub = np.zeros(num_paths)
        for i in range(num_paths):
            A_ub[i, :num_nodes] = -payoff_matrix[i]  # negate for <= form
            A_ub[i, -1] = 1.0  # +v
            # -A[i]@y + v >= 0 => A[i]@y - v <= 0

        # Fix: correct formulation
        # We want: for each path i, payoff_matrix[i] @ y <= v
        # => payoff_matrix[i] @ y - v <= 0
        A_ub_correct = np.zeros((num_paths, num_nodes + 1))
        for i in range(num_paths):
            A_ub_correct[i, :num_nodes] = payoff_matrix[i]
            A_ub_correct[i, -1] = -1.0
        b_ub_correct = np.zeros(num_paths)

        # Equality constraint: sum(y) = 1
        A_eq = np.zeros((1, num_nodes + 1))
        A_eq[0, :num_nodes] = 1.0
        b_eq = np.array([1.0])

        # Bounds: y_j >= 0, v unrestricted (but practically >= 0)
        bounds = [(0, None)] * num_nodes + [(None, None)]

        result = linprog(
            c, A_ub=A_ub_correct, b_ub=b_ub_correct,
            A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )

        if not result.success:
            return _fallback_equilibrium(graph, paths, nodes)

        defender_probs = result.x[:num_nodes]
        equilibrium_value = result.x[-1]

    except Exception:
        return _fallback_equilibrium(graph, paths, nodes)

    # Compute attacker's best response given defender strategy
    attacker_payoffs = []
    for i, path in enumerate(paths):
        expected_payoff = sum(
            payoff_matrix[i][j] * defender_probs[j]
            for j in range(num_nodes)
        )
        attacker_payoffs.append(expected_payoff)

    # Normalize attacker probabilities (softmax-style)
    max_payoff = max(attacker_payoffs) if attacker_payoffs else 0
    if max_payoff > 0:
        exp_payoffs = [math.exp(min(p / max_payoff * 5, 20)) for p in attacker_payoffs]
        total_exp = sum(exp_payoffs)
        attacker_probs = [e / total_exp for e in exp_payoffs]
    else:
        attacker_probs = [1.0 / num_paths] * num_paths

    # Build result
    defender_strategy = {
        nodes[j]: float(defender_probs[j])
        for j in range(num_nodes)
        if defender_probs[j] > 0.01
    }

    attacker_strategy = [
        {
            "path": path,
            "path_names": [graph.nodes[n].get("name", n) for n in path],
            "probability": float(attacker_probs[i]),
            "effort_score": _path_effort_score(graph, path),
        }
        for i, path in enumerate(paths)
        if attacker_probs[i] > 0.01
    ]

    # Sort by probability descending
    attacker_strategy.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "defender_strategy": defender_strategy,
        "attacker_strategy": attacker_strategy,
        "equilibrium_value": float(equilibrium_value),
        "computed_at_turn": 0,  # caller should update
    }


def _fallback_equilibrium(
    graph: nx.DiGraph, paths: list, nodes: list
) -> dict:
    """Fallback equilibrium when LP solver fails - uniform distribution."""
    num_nodes = len(nodes)
    num_paths = len(paths)

    defender_strategy = {n: 1.0 / num_nodes for n in nodes}
    attacker_strategy = [
        {
            "path": path,
            "path_names": [graph.nodes[n].get("name", n) for n in path],
            "probability": 1.0 / num_paths,
            "effort_score": _path_effort_score(graph, path),
        }
        for path in paths
    ]

    return {
        "defender_strategy": defender_strategy,
        "attacker_strategy": attacker_strategy,
        "equilibrium_value": 0.5,
        "computed_at_turn": 0,
    }


def get_recommended_paths(
    nash_result: dict,
    top_k: int = 3,
) -> list[dict]:
    """Get top-k recommended attack paths from Nash result."""
    if not nash_result or "attacker_strategy" not in nash_result:
        return []
    return nash_result["attacker_strategy"][:top_k]


def encode_nash_signal(nash_result: Optional[dict]) -> list[float]:
    """Encode Nash equilibrium result into a fixed-size vector for RL state.

    Returns 9 floats: top 3 paths x (probability, effort_score, path_length_normalized)
    """
    signal = [0.0] * 9
    if not nash_result or "attacker_strategy" not in nash_result:
        return signal

    paths = nash_result["attacker_strategy"][:3]
    for i, path_info in enumerate(paths):
        base = i * 3
        signal[base] = path_info.get("probability", 0.0)
        signal[base + 1] = path_info.get("effort_score", 0.0)
        path_len = len(path_info.get("path", []))
        signal[base + 2] = min(path_len / 10.0, 1.0)  # normalize

    return signal
