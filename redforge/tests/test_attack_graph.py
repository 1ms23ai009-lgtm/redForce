"""Tests for the attack graph engine."""

import pytest
import networkx as nx

from redforge.graph.attack_graph import (
    create_attack_graph,
    add_node,
    add_edge,
    mark_node_vulnerable,
    increment_bypass_attempt,
    confirm_edge,
    get_vulnerable_nodes,
    get_attack_paths,
    serialize_graph,
    deserialize_graph,
    remove_cycles,
    add_artificial_leaves,
    connect_components,
    prune_nonvulnerable_leaves,
    get_graph_summary,
    NODE_TYPES,
    EDGE_TYPES,
)
from redforge.graph.effort_scorer import (
    compute_phi_msg,
    compute_phi_tok,
    compute_phi_cost,
    compute_effort_score,
)


class TestAttackGraph:
    def test_create_empty_graph(self):
        g = create_attack_graph()
        assert isinstance(g, nx.DiGraph)
        assert g.number_of_nodes() == 0

    def test_add_node(self):
        g = create_attack_graph()
        node_id = add_node(g, "test_node", "guardrail")
        assert node_id in g
        assert g.nodes[node_id]["name"] == "test_node"
        assert g.nodes[node_id]["node_type"] == "guardrail"
        assert g.nodes[node_id]["vulnerability"] is False

    def test_add_node_invalid_type(self):
        g = create_attack_graph()
        with pytest.raises(ValueError):
            add_node(g, "test", "invalid_type")

    def test_add_edge(self):
        g = create_attack_graph()
        n1 = add_node(g, "source", "user_input")
        n2 = add_node(g, "target", "guardrail")
        edge_id = add_edge(g, n1, n2, "exploitable_transition", effort_score=0.7)
        assert g.has_edge(n1, n2)
        assert g.edges[n1, n2]["effort_score"] == 0.7

    def test_add_edge_invalid_type(self):
        g = create_attack_graph()
        n1 = add_node(g, "a", "user_input")
        n2 = add_node(g, "b", "guardrail")
        with pytest.raises(ValueError):
            add_edge(g, n1, n2, "invalid_edge_type")

    def test_mark_vulnerable(self):
        g = create_attack_graph()
        n = add_node(g, "test", "guardrail")
        mark_node_vulnerable(g, n, "high")
        assert g.nodes[n]["vulnerability"] is True
        assert g.nodes[n]["vulnerability_severity"] == "high"

    def test_increment_bypass(self):
        g = create_attack_graph()
        n = add_node(g, "test", "guardrail")
        increment_bypass_attempt(g, n, success=False)
        assert g.nodes[n]["bypass_attempts"] == 1
        assert g.nodes[n]["bypass_successes"] == 0
        increment_bypass_attempt(g, n, success=True)
        assert g.nodes[n]["bypass_attempts"] == 2
        assert g.nodes[n]["bypass_successes"] == 1

    def test_get_vulnerable_nodes(self):
        g = create_attack_graph()
        n1 = add_node(g, "a", "guardrail")
        n2 = add_node(g, "b", "guardrail", vulnerability=True)
        assert get_vulnerable_nodes(g) == [n2]

    def test_get_attack_paths(self):
        g = create_attack_graph()
        entry = add_node(g, "entry", "user_input")
        guard = add_node(g, "guard", "guardrail")
        vuln = add_node(g, "vuln", "tool", vulnerability=True, vulnerability_severity="high")
        add_edge(g, entry, guard, "information_flow")
        add_edge(g, guard, vuln, "exploitable_transition")
        mark_node_vulnerable(g, vuln, "high")
        paths = get_attack_paths(g)
        assert len(paths) >= 1
        assert paths[0] == [entry, guard, vuln]

    def test_serialize_deserialize(self):
        g = create_attack_graph()
        n1 = add_node(g, "a", "user_input")
        n2 = add_node(g, "b", "guardrail")
        add_edge(g, n1, n2, "information_flow")

        data = serialize_graph(g)
        g2 = deserialize_graph(data)
        assert g2.number_of_nodes() == 2
        assert g2.number_of_edges() == 1

    def test_deserialize_empty(self):
        g = deserialize_graph({})
        assert g.number_of_nodes() == 0

    def test_remove_cycles(self):
        g = create_attack_graph()
        n1 = add_node(g, "a", "user_input")
        n2 = add_node(g, "b", "guardrail")
        add_edge(g, n1, n2, "information_flow")
        add_edge(g, n2, n1, "information_flow")
        g = remove_cycles(g)
        assert nx.is_directed_acyclic_graph(g)

    def test_connect_components(self):
        g = create_attack_graph()
        n1 = add_node(g, "a", "user_input")
        n2 = add_node(g, "b", "guardrail")
        # Two disconnected nodes
        connect_components(g)
        components = list(nx.weakly_connected_components(g))
        assert len(components) == 1

    def test_graph_summary(self):
        g = create_attack_graph()
        add_node(g, "a", "user_input")
        add_node(g, "b", "guardrail", vulnerability=True, vulnerability_severity="high")
        summary = get_graph_summary(g)
        assert summary["total_nodes"] == 2
        assert summary["vulnerable_nodes"] == 1


class TestEffortScorer:
    def test_phi_msg_single_turn(self):
        assert compute_phi_msg(1, 1) == 1.0

    def test_phi_msg_normal(self):
        score = compute_phi_msg(3, 10)
        assert 0 <= score <= 1

    def test_phi_tok_zero_total(self):
        assert compute_phi_tok(0, 0) == 1.0

    def test_phi_cost_zero_total(self):
        assert compute_phi_cost(0, 0) == 1.0

    def test_effort_score_bounds(self):
        score = compute_effort_score(
            turns_to_exploit=5, total_turns=20,
            tokens_in_chain=1000, total_tokens=5000,
            cost_of_path=0.1, total_cost=1.0,
        )
        assert 0.0 <= score <= 1.0

    def test_effort_score_easy_exploit(self):
        easy = compute_effort_score(1, 100, 10, 10000, 0.001, 10.0)
        hard = compute_effort_score(50, 100, 5000, 10000, 5.0, 10.0)
        assert easy > hard
