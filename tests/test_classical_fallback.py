"""Tests for classical fallback algorithms."""

import pytest
import networkx as nx

from quantum_researcher.classical_fallback import (
    classical_search,
    classical_random_walk,
    classical_path_finder,
    bfs_similarity_search,
)


class TestClassicalSearch:
    """Test classical linear search."""

    def test_finds_matches(self):
        items = list(range(20))
        result = classical_search(lambda x: x % 5 == 0, items)
        assert result.found_items == [0, 5, 10, 15]
        assert result.n_items == 20
        assert result.method == "classical_linear"

    def test_no_matches(self):
        result = classical_search(lambda x: False, [1, 2, 3])
        assert result.found_items == []

    def test_all_match(self):
        result = classical_search(lambda x: True, [1, 2, 3])
        assert len(result.found_items) == 3

    def test_elapsed_tracked(self):
        result = classical_search(lambda x: True, list(range(1000)))
        assert result.elapsed_ms >= 0


class TestClassicalRandomWalk:
    """Test classical random walk."""

    def test_basic_walk(self):
        G = nx.path_graph(10)
        result = classical_random_walk(G, start=0, n_steps=50, n_walks=20)
        assert len(result.visited_nodes) > 0
        assert result.method == "classical_random_walk"

    def test_empty_graph(self):
        G = nx.Graph()
        result = classical_random_walk(G, start="nonexistent")
        assert result.visited_nodes == []

    def test_visit_counts(self):
        G = nx.complete_graph(5)
        result = classical_random_walk(G, start=0, n_steps=100, n_walks=50)
        # All nodes should be visited on a complete graph
        assert len(result.visit_counts) >= 3

    def test_isolated_node(self):
        G = nx.Graph()
        G.add_node("lonely")
        result = classical_random_walk(G, start="lonely")
        assert result.visited_nodes == []  # can't go anywhere


class TestClassicalPathFinder:
    """Test classical path finding."""

    def test_finds_paths(self):
        G = nx.grid_2d_graph(3, 3)
        paths = classical_path_finder(G, (0, 0), (2, 2))
        assert len(paths) > 0
        assert paths[0][0] == (0, 0)
        assert paths[0][-1] == (2, 2)

    def test_no_path(self):
        G = nx.Graph()
        G.add_edge("A", "B")
        G.add_edge("C", "D")
        paths = classical_path_finder(G, "A", "C")
        assert paths == []

    def test_sorted_by_length(self):
        G = nx.complete_graph(5)
        paths = classical_path_finder(G, 0, 4)
        # Should be sorted by length
        for i in range(len(paths) - 1):
            assert len(paths[i]) <= len(paths[i + 1])

    def test_max_paths(self):
        G = nx.complete_graph(6)
        paths = classical_path_finder(G, 0, 5, max_paths=3)
        assert len(paths) <= 3


class TestBFSSimilaritySearch:
    """Test BFS-based similarity search."""

    def test_basic_search(self):
        G = nx.star_graph(5)
        results = bfs_similarity_search(G, start=0)
        assert len(results) > 0
        # All nodes should be found
        nodes = [n for n, _ in results]
        for i in range(1, 6):
            assert i in nodes

    def test_distance_scoring(self):
        G = nx.path_graph(10)
        results = bfs_similarity_search(G, start=0, max_depth=5)
        # Closer nodes should have higher scores
        if len(results) >= 2:
            assert results[0][1] >= results[-1][1]

    def test_nonexistent_start(self):
        G = nx.Graph()
        results = bfs_similarity_search(G, start="nope")
        assert results == []

    def test_max_results(self):
        G = nx.complete_graph(20)
        results = bfs_similarity_search(G, start=0, max_results=5)
        assert len(results) <= 5

    def test_max_depth(self):
        G = nx.path_graph(20)
        results = bfs_similarity_search(G, start=0, max_depth=2)
        # Should only find nodes within 2 hops
        max_node = max(n for n, _ in results) if results else 0
        assert max_node <= 2
