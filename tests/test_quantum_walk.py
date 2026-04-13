"""Tests for QuantumWalk."""

import pytest
import networkx as nx
import numpy as np

from quantum_researcher.quantum_walk import QuantumWalk, WalkResult


class TestQuantumWalk:
    """Test quantum walk on graphs."""

    def test_empty_graph(self):
        G = nx.Graph()
        walker = QuantumWalk(graph=G)
        result = walker.walk(start="nonexistent")
        assert result.visited_nodes == []
        assert result.method == "empty"

    def test_single_node(self):
        G = nx.Graph()
        G.add_node("A")
        walker = QuantumWalk(graph=G)
        result = walker.walk(start="A")
        # Single node: walker stays at A
        assert "A" in result.visit_probabilities

    def test_two_nodes(self):
        G = nx.Graph()
        G.add_edge("A", "B")
        walker = QuantumWalk(graph=G, walk_time=1.0, n_steps=5)
        result = walker.walk(start="A")

        assert "A" in result.visit_probabilities
        assert "B" in result.visit_probabilities
        assert result.method == "classical"

    def test_path_graph(self):
        """Walk on a path should spread outward from start."""
        G = nx.path_graph(10)
        walker = QuantumWalk(graph=G, walk_time=2.0, n_steps=10, threshold=0.001)
        result = walker.walk(start=0)

        # Should visit multiple nodes
        assert len(result.visited_nodes) > 1
        assert result.elapsed_ms > 0

    def test_karate_club(self):
        """Walk on Zachary's karate club — a real-world social network."""
        G = nx.karate_club_graph()
        walker = QuantumWalk(graph=G, walk_time=3.0, n_steps=20)
        result = walker.walk(start=0)

        # Hub nodes should have higher probability
        assert len(result.visited_nodes) > 5
        assert result.n_steps == 20

    def test_statevector_returned(self):
        """Walk should return statevector probabilities."""
        G = nx.cycle_graph(4)
        walker = QuantumWalk(graph=G, walk_time=1.0, n_steps=5)
        result = walker.walk(start=0)

        assert result.statevector is not None
        assert len(result.statevector) == 4
        # Probabilities should sum to ~1
        assert abs(result.statevector.sum() - 1.0) < 0.1

    def test_target_path_finding(self):
        """Should find paths when target is specified."""
        G = nx.grid_2d_graph(4, 4)
        walker = QuantumWalk(graph=G, walk_time=2.0, n_steps=10)
        result = walker.walk(start=(0, 0), target=(3, 3))

        assert len(result.paths_found) > 0
        # First path should start at source and end at target
        assert result.paths_found[0][0] == (0, 0)
        assert result.paths_found[0][-1] == (3, 3)

    def test_disconnected_target(self):
        """Should return empty paths for unreachable target."""
        G = nx.Graph()
        G.add_edge("A", "B")
        G.add_edge("C", "D")  # disconnected
        walker = QuantumWalk(graph=G)
        result = walker.walk(start="A", target="C")
        assert result.paths_found == []

    def test_find_connections(self):
        """find_connections should return connected nodes with probabilities."""
        G = nx.star_graph(5)  # center = 0, spokes = 1-5
        walker = QuantumWalk(graph=G, walk_time=2.0, n_steps=10, threshold=0.01)
        connections = walker.find_connections(0)

        # All spoke nodes should be found
        connected_nodes = [node for node, _ in connections]
        assert len(connected_nodes) >= 3  # at least some spokes
        # Probabilities should be positive
        for _, prob in connections:
            assert prob > 0

    def test_from_edges_factory(self):
        """from_edges should create a QuantumWalk from edge list."""
        edges = [("A", "B", None), ("B", "C", None), ("C", "D", None)]
        walker = QuantumWalk.from_edges(edges, walk_time=1.0)

        assert walker.graph.number_of_nodes() == 4
        assert walker.graph.number_of_edges() == 3

        result = walker.walk(start="A")
        assert len(result.visited_nodes) > 0

    def test_large_graph_subsampling(self):
        """Graphs larger than max_nodes should be subsampled."""
        G = nx.barabasi_albert_graph(200, 3)
        walker = QuantumWalk(graph=G, max_nodes=50, walk_time=1.0, n_steps=5)
        result = walker.walk(start=0)

        # Should still work, just on a subgraph
        assert result.method == "classical"
        assert result.elapsed_ms > 0

    def test_walk_time_affects_spread(self):
        """Longer walk time should spread probability further."""
        G = nx.path_graph(20)

        short_walk = QuantumWalk(graph=G, walk_time=0.5, n_steps=5, threshold=0.01)
        long_walk = QuantumWalk(graph=G, walk_time=5.0, n_steps=20, threshold=0.01)

        short_result = short_walk.walk(start=0)
        long_result = long_walk.walk(start=0)

        # Longer walk should visit more nodes (generally)
        assert len(long_result.visited_nodes) >= len(short_result.visited_nodes)
