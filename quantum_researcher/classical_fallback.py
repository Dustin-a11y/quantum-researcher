"""Classical Fallback — CPU-only search when no QPU available.

Provides the same interfaces as quantum modules but using classical
algorithms. Users get correct results (just without speedup) when
Qiskit isn't installed or no quantum hardware is accessible.

This ensures quantum-researcher works EVERYWHERE — laptop, server,
Raspberry Pi — and gracefully upgrades when quantum resources appear.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np


@dataclass
class ClassicalSearchResult:
    """Result from classical search."""

    found_indices: list[int]
    found_items: list[Any]
    n_items: int
    elapsed_ms: float
    method: str = "classical_linear"


@dataclass
class ClassicalWalkResult:
    """Result from classical random walk."""

    visited_nodes: list[Any]
    visit_counts: dict[Any, int]
    n_steps: int
    elapsed_ms: float
    method: str = "classical_random_walk"


def classical_search(
    predicate: Callable[[Any], bool],
    items: Sequence[Any],
) -> ClassicalSearchResult:
    """Linear search through items.

    O(N) — the baseline that Grover improves to O(√N).
    """
    t0 = time.monotonic()
    found_indices = [i for i, item in enumerate(items) if predicate(item)]
    found_items = [items[i] for i in found_indices]
    elapsed = (time.monotonic() - t0) * 1000

    return ClassicalSearchResult(
        found_indices=found_indices,
        found_items=found_items,
        n_items=len(items),
        elapsed_ms=elapsed,
    )


def classical_random_walk(
    graph: nx.Graph,
    start: Any,
    n_steps: int = 100,
    n_walks: int = 50,
) -> ClassicalWalkResult:
    """Classical random walk on graph.

    O(N) exploration vs quantum walk's O(√N) for certain topologies.
    Runs multiple random walks and counts visit frequency.
    """
    t0 = time.monotonic()
    visit_counts: dict[Any, int] = {}
    nodes = list(graph.nodes())

    if not nodes or start not in nodes:
        return ClassicalWalkResult(
            visited_nodes=[], visit_counts={},
            n_steps=0, elapsed_ms=0,
        )

    for _ in range(n_walks):
        current = start
        for _ in range(n_steps):
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                break
            current = neighbors[np.random.randint(len(neighbors))]
            visit_counts[current] = visit_counts.get(current, 0) + 1

    visited = sorted(visit_counts.keys(), key=lambda n: visit_counts[n], reverse=True)
    elapsed = (time.monotonic() - t0) * 1000

    return ClassicalWalkResult(
        visited_nodes=visited,
        visit_counts=visit_counts,
        n_steps=n_steps * n_walks,
        elapsed_ms=elapsed,
    )


def classical_path_finder(
    graph: nx.Graph,
    source: Any,
    target: Any,
    max_paths: int = 10,
    cutoff: int = 8,
) -> list[list[Any]]:
    """Find paths between nodes using classical shortest-path algorithms.

    Uses Dijkstra / all_simple_paths as baseline.
    Quantum walk finds these paths with interference-guided exploration.
    """
    try:
        paths = list(nx.all_simple_paths(graph, source, target, cutoff=cutoff))
        # Sort by length (shortest first)
        paths.sort(key=len)
        return paths[:max_paths]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def bfs_similarity_search(
    graph: nx.Graph,
    start: Any,
    max_depth: int = 3,
    max_results: int = 20,
) -> list[tuple[Any, float]]:
    """BFS-based similarity search on knowledge graph.

    Classical equivalent of quantum walk's find_connections().
    Returns nodes with distance-based similarity scores.
    """
    if start not in graph:
        return []

    visited = {start: 0}
    queue = [(start, 0)]
    results = []

    while queue:
        node, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
                # Score: inverse distance (closer = higher score)
                score = 1.0 / (depth + 2)
                results.append((neighbor, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_results]
