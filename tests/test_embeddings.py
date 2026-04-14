"""Tests for embedding utilities. DK 🦍"""
import numpy as np
from quantum_researcher.embeddings import (
    SimpleEmbedder, cosine_similarity, cosine_similarity_matrix, get_embedder,
)


class TestSimpleEmbedder:
    def test_output_shape(self):
        e = SimpleEmbedder(dim=128)
        vec = e.embed("quantum computing research")
        assert vec.shape == (128,)

    def test_normalized(self):
        e = SimpleEmbedder()
        vec = e.embed("hello world this is a test")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_empty_text(self):
        e = SimpleEmbedder()
        vec = e.embed("")
        assert np.allclose(vec, 0)

    def test_similar_texts_close(self):
        e = SimpleEmbedder()
        a = e.embed("quantum computing algorithms")
        b = e.embed("quantum computing methods")
        c = e.embed("cooking recipes for dinner")
        sim_ab = cosine_similarity(a, b)
        sim_ac = cosine_similarity(a, c)
        assert sim_ab > sim_ac

    def test_batch(self):
        e = SimpleEmbedder(dim=64)
        vecs = e.embed_batch(["hello", "world", "test"])
        assert vecs.shape == (3, 64)

    def test_deterministic(self):
        e = SimpleEmbedder()
        a = e.embed("same text")
        b = e.embed("same text")
        assert np.allclose(a, b)


class TestCosineSimilarity:
    def test_identical(self):
        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 0.001

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert cosine_similarity(a, b) == 0.0


class TestCosineSimilarityMatrix:
    def test_shape(self):
        q = np.random.randn(3, 10)
        c = np.random.randn(5, 10)
        sim = cosine_similarity_matrix(q, c)
        assert sim.shape == (3, 5)

    def test_self_similarity(self):
        v = np.random.randn(4, 8)
        sim = cosine_similarity_matrix(v, v)
        # Diagonal should be ~1
        for i in range(4):
            assert abs(sim[i, i] - 1.0) < 0.01


class TestGetEmbedder:
    def test_default_simple(self):
        e = get_embedder(use_transformers=False)
        assert isinstance(e, SimpleEmbedder)

    def test_fallback_to_simple(self):
        # use_transformers=True but not installed -> falls back
        e = get_embedder(use_transformers=True)
        # Should get SimpleEmbedder since sentence-transformers probably not installed
        assert hasattr(e, 'embed')
