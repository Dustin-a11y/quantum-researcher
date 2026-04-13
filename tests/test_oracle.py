"""Tests for QuantumOracle."""

import numpy as np
import pytest

from quantum_researcher.oracle import QuantumOracle, OracleResult


class TestQuantumOracle:
    """Test oracle encoding."""

    def test_empty_items(self):
        oracle = QuantumOracle(predicate=lambda x: True, items=[])
        result = oracle.encode()
        assert result.n_items == 0
        assert result.n_marked == 0
        assert result.marked_indices == []

    def test_single_match(self):
        items = [1, 2, 3, 4, 5, 6, 7, 8]
        oracle = QuantumOracle(
            predicate=lambda x: x == 5,
            items=items,
            use_quantum=False,
        )
        result = oracle.encode()
        assert result.n_items == 8
        assert result.n_marked == 1
        assert result.marked_indices == [4]
        assert result.bitmap[4] is np.True_

    def test_multiple_matches(self):
        items = list(range(16))
        oracle = QuantumOracle(
            predicate=lambda x: x % 4 == 0,
            items=items,
            use_quantum=False,
        )
        result = oracle.encode()
        assert result.n_marked == 4
        assert result.marked_indices == [0, 4, 8, 12]

    def test_no_matches(self):
        items = [1, 2, 3, 4]
        oracle = QuantumOracle(
            predicate=lambda x: x > 100,
            items=items,
            use_quantum=False,
        )
        result = oracle.encode()
        assert result.n_marked == 0
        assert result.marked_indices == []

    def test_all_match(self):
        items = [1, 2, 3, 4]
        oracle = QuantumOracle(
            predicate=lambda x: True,
            items=items,
            use_quantum=False,
        )
        result = oracle.encode()
        assert result.n_marked == 4

    def test_n_qubits_calculation(self):
        # 8 items → 3 qubits (2^3 = 8)
        oracle = QuantumOracle(predicate=lambda x: False, items=list(range(8)))
        result = oracle.encode()
        assert result.n_qubits == 3

        # 5 items → 3 qubits (ceil(log2(5)) = 3)
        oracle = QuantumOracle(predicate=lambda x: False, items=list(range(5)))
        result = oracle.encode()
        assert result.n_qubits == 3

        # 1 item → 1 qubit
        oracle = QuantumOracle(predicate=lambda x: False, items=[1])
        result = oracle.encode()
        assert result.n_qubits == 1

    def test_bitmap_shape(self):
        items = list(range(10))
        oracle = QuantumOracle(
            predicate=lambda x: x < 3,
            items=items,
            use_quantum=False,
        )
        result = oracle.encode()
        assert result.bitmap.shape == (10,)
        assert result.bitmap[:3].all()
        assert not result.bitmap[3:].any()


class TestFromKeywords:
    """Test keyword-based oracle factory."""

    def test_single_keyword(self):
        items = [
            {"text": "quantum computing is cool"},
            {"text": "classical computing is old"},
            {"text": "quantum mechanics rocks"},
            {"text": "nothing related here"},
        ]
        oracle = QuantumOracle.from_keywords(
            keywords=["quantum"], items=items, use_quantum=False
        )
        result = oracle.encode()
        assert result.n_marked == 2
        assert result.marked_indices == [0, 2]

    def test_multiple_keywords(self):
        items = [
            {"text": "quantum computing"},
            {"text": "machine learning"},
            {"text": "deep learning networks"},
        ]
        oracle = QuantumOracle.from_keywords(
            keywords=["quantum", "learning"], items=items, use_quantum=False
        )
        result = oracle.encode()
        assert result.n_marked == 3  # all match at least one keyword

    def test_case_insensitive(self):
        items = [{"text": "QUANTUM COMPUTING"}]
        oracle = QuantumOracle.from_keywords(
            keywords=["quantum"], items=items, use_quantum=False
        )
        result = oracle.encode()
        assert result.n_marked == 1

    def test_custom_text_key(self):
        items = [{"content": "quantum stuff"}]
        oracle = QuantumOracle.from_keywords(
            keywords=["quantum"], items=items, text_key="content", use_quantum=False
        )
        result = oracle.encode()
        assert result.n_marked == 1


class TestFromEmbeddingSimilarity:
    """Test embedding-based oracle factory."""

    def test_similarity_threshold(self):
        query = np.array([1.0, 0.0, 0.0])
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # cosine = 1.0
            [0.0, 1.0, 0.0],  # cosine = 0.0
            [0.7, 0.7, 0.0],  # cosine ≈ 0.707
            [0.9, 0.1, 0.0],  # cosine ≈ 0.994
        ])
        items = ["exact", "orthogonal", "partial", "close"]
        oracle = QuantumOracle.from_embedding_similarity(
            query_embedding=query,
            item_embeddings=embeddings,
            threshold=0.7,
            items=items,
            use_quantum=False,
        )
        result = oracle.encode()
        # "exact" (1.0), "partial" (~0.707), "close" (~0.994) should match
        assert result.n_marked == 3
        assert 1 not in result.marked_indices  # "orthogonal" shouldn't match

    def test_high_threshold(self):
        query = np.array([1.0, 0.0])
        embeddings = np.array([
            [1.0, 0.0],
            [0.5, 0.866],   # cosine ~0.5 with [1,0]
            [0.5, 0.5],
        ])
        oracle = QuantumOracle.from_embedding_similarity(
            query_embedding=query,
            item_embeddings=embeddings,
            threshold=0.99,
            use_quantum=False,
        )
        result = oracle.encode()
        assert result.n_marked == 1  # only exact match
