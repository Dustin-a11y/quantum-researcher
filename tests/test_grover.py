"""Tests for GroverSearch."""

import pytest
import numpy as np

from quantum_researcher.oracle import QuantumOracle
from quantum_researcher.grover_search import GroverSearch


class TestGroverSearch:
    """Test Grover search algorithm."""

    def test_empty_items(self):
        oracle = QuantumOracle(predicate=lambda x: True, items=[])
        grover = GroverSearch(oracle=oracle)
        result = grover.search()
        assert result.n_items == 0
        assert result.found_items == []
        assert result.method == "empty"

    def test_no_matches(self):
        items = list(range(8))
        oracle = QuantumOracle(
            predicate=lambda x: x > 100, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()
        assert result.found_items == []
        assert result.method == "no_match"

    def test_single_match_classical(self):
        """Grover should find the single marked item."""
        items = list(range(8))
        oracle = QuantumOracle(
            predicate=lambda x: x == 5, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()

        assert 5 in result.found_items
        assert result.method == "classical_simulation"
        assert result.n_iterations > 0
        assert result.speedup > 1.0

    def test_multiple_matches_classical(self):
        """Grover should find all marked items."""
        items = list(range(64))
        marked = {0, 16, 32, 48}
        oracle = QuantumOracle(
            predicate=lambda x: x in marked, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle, confidence_threshold=0.02)
        result = grover.search()

        # All marked items should be found
        for m in marked:
            assert m in result.found_items

        assert result.method == "classical_simulation"

    def test_speedup_positive(self):
        """Speedup should be > 1 (Grover is faster than linear)."""
        items = list(range(64))
        oracle = QuantumOracle(
            predicate=lambda x: x == 42, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()
        assert result.speedup > 1.0

    def test_iteration_count(self):
        """Optimal iterations should be ~π/4 * √N for 1 marked item."""
        import math
        items = list(range(64))
        oracle = QuantumOracle(
            predicate=lambda x: x == 0, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()

        expected = round(math.pi / 4 * math.sqrt(64))
        assert result.n_iterations == expected

    def test_keyword_search(self):
        """Integration test: keyword oracle + Grover search."""
        # Use 16 items so marked ratio (4/16=25%) doesn't cause
        # Grover over-rotation. Pad with non-matching filler.
        items = [
            {"id": 0, "text": "quantum computing breakthrough"},
            {"id": 1, "text": "classical music concert"},
            {"id": 2, "text": "quantum mechanics fundamentals"},
            {"id": 3, "text": "cooking recipes for dinner"},
            {"id": 4, "text": "quantum entanglement paper"},
            {"id": 5, "text": "sports news today"},
            {"id": 6, "text": "weather forecast"},
            {"id": 7, "text": "quantum error correction"},
            {"id": 8, "text": "history of ancient rome"},
            {"id": 9, "text": "gardening tips and tricks"},
            {"id": 10, "text": "travel destinations guide"},
            {"id": 11, "text": "home improvement ideas"},
            {"id": 12, "text": "fitness workout routine"},
            {"id": 13, "text": "photography basics tutorial"},
            {"id": 14, "text": "financial planning advice"},
            {"id": 15, "text": "movie reviews and ratings"},
        ]
        oracle = QuantumOracle.from_keywords(
            keywords=["quantum"], items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle, confidence_threshold=0.05)
        result = grover.search()

        found_ids = [item["id"] for item in result.found_items]
        assert 0 in found_ids
        assert 2 in found_ids
        assert 4 in found_ids
        assert 7 in found_ids

    def test_probabilities_returned(self):
        """Search should return probability distribution."""
        items = list(range(8))
        oracle = QuantumOracle(
            predicate=lambda x: x == 3, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()

        assert result.probabilities is not None
        assert len(result.probabilities) == 8
        # Marked item should have highest probability
        assert result.probabilities[3] == max(result.probabilities)

    def test_max_iterations_override(self):
        """Should respect max_iterations override."""
        items = list(range(16))
        oracle = QuantumOracle(
            predicate=lambda x: x == 0, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle, max_iterations=1)
        result = grover.search()
        assert result.n_iterations == 1

    def test_elapsed_ms_tracked(self):
        """Should track execution time."""
        items = list(range(32))
        oracle = QuantumOracle(
            predicate=lambda x: x == 15, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()
        assert result.elapsed_ms > 0

    def test_large_search_space(self):
        """Grover should work with larger search spaces."""
        items = list(range(256))
        oracle = QuantumOracle(
            predicate=lambda x: x == 200, items=items, use_quantum=False
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()

        assert 200 in result.found_items
        assert result.speedup > 5  # √256 = 16, classical = 128, so ~8x
