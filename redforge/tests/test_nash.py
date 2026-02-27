"""Tests for Nash equilibrium computation."""

import pytest
import networkx as nx

from redforge.graph.attack_graph import (
    create_attack_graph,
    add_node,
    add_edge,
    mark_node_vulnerable,
)
from redforge.graph.nash import (
    compute_nash_equilibrium,
    get_recommended_paths,
    encode_nash_signal,
    _poisson_reach_probability,
)


class TestNashEquilibrium:
    def _build_simple_graph(self):
        """Build a simple attack graph with one path to a vulnerable node."""
        g = create_attack_graph()
        entry = add_node(g, "entry", "user_input")
        guard = add_node(g, "guard", "guardrail")
        vuln = add_node(g, "vuln", "tool", vulnerability=True, vulnerability_severity="high")
        add_edge(g, entry, guard, "information_flow", effort_score=0.7)
        add_edge(g, guard, vuln, "exploitable_transition", effort_score=0.8)
        mark_node_vulnerable(g, vuln, "high")
        return g

    def test_compute_nash_empty_graph(self):
        g = create_attack_graph()
        result = compute_nash_equilibrium(g)
        assert result is None

    def test_compute_nash_no_vulnerable(self):
        g = create_attack_graph()
        add_node(g, "a", "user_input")
        add_node(g, "b", "guardrail")
        result = compute_nash_equilibrium(g)
        assert result is None

    def test_compute_nash_simple(self):
        g = self._build_simple_graph()
        result = compute_nash_equilibrium(g)
        assert result is not None
        assert "defender_strategy" in result
        assert "attacker_strategy" in result
        assert "equilibrium_value" in result
        assert isinstance(result["equilibrium_value"], float)

    def test_defender_strategy_sums_to_one(self):
        g = self._build_simple_graph()
        result = compute_nash_equilibrium(g)
        if result and result["defender_strategy"]:
            total = sum(result["defender_strategy"].values())
            assert abs(total - 1.0) < 0.1  # Allow some tolerance

    def test_attacker_strategy_has_paths(self):
        g = self._build_simple_graph()
        result = compute_nash_equilibrium(g)
        assert len(result["attacker_strategy"]) > 0
        for path_info in result["attacker_strategy"]:
            assert "path" in path_info
            assert "probability" in path_info
            assert path_info["probability"] >= 0

    def test_get_recommended_paths(self):
        g = self._build_simple_graph()
        result = compute_nash_equilibrium(g)
        paths = get_recommended_paths(result, top_k=3)
        assert len(paths) > 0
        assert len(paths) <= 3

    def test_get_recommended_paths_none(self):
        paths = get_recommended_paths(None)
        assert paths == []

    def test_encode_nash_signal(self):
        g = self._build_simple_graph()
        result = compute_nash_equilibrium(g)
        signal = encode_nash_signal(result)
        assert len(signal) == 9
        assert all(isinstance(v, float) for v in signal)

    def test_encode_nash_signal_none(self):
        signal = encode_nash_signal(None)
        assert signal == [0.0] * 9

    def test_poisson_reach_probability_zero_length(self):
        prob = _poisson_reach_probability(0)
        assert prob == 1.0

    def test_poisson_reach_probability_bounds(self):
        prob = _poisson_reach_probability(5)
        assert 0.0 <= prob <= 1.0


class TestNashMultiPath:
    def test_multi_path_graph(self):
        """Test with multiple attack paths."""
        g = create_attack_graph()
        entry = add_node(g, "entry", "user_input")
        guard1 = add_node(g, "guard1", "guardrail")
        guard2 = add_node(g, "guard2", "guardrail")
        vuln1 = add_node(g, "vuln1", "tool", vulnerability=True, vulnerability_severity="high")
        vuln2 = add_node(g, "vuln2", "agent", vulnerability=True, vulnerability_severity="medium")

        add_edge(g, entry, guard1, "information_flow", effort_score=0.5)
        add_edge(g, entry, guard2, "information_flow", effort_score=0.6)
        add_edge(g, guard1, vuln1, "exploitable_transition", effort_score=0.8)
        add_edge(g, guard2, vuln2, "exploitable_transition", effort_score=0.3)
        mark_node_vulnerable(g, vuln1, "high")
        mark_node_vulnerable(g, vuln2, "medium")

        result = compute_nash_equilibrium(g)
        assert result is not None
        assert len(result["attacker_strategy"]) >= 1
