"""
Embedding utilities for quantum-researcher.
Lightweight — uses numpy only, no ML dependencies required.
Optional: sentence-transformers for real embeddings.

DK 🦍
"""

import hashlib
import numpy as np
from typing import List, Optional


class SimpleEmbedder:
    """Bag-of-words + TF-IDF style embeddings. No ML deps needed."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._vocab: dict = {}
        self._idf: Optional[np.ndarray] = None

    def _hash_token(self, token: str) -> int:
        """Deterministic hash to dimension index."""
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return h % self.dim

    def embed(self, text: str) -> np.ndarray:
        """Convert text to a fixed-dimension vector."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for token in tokens:
            # Strip punctuation
            token = ''.join(c for c in token if c.isalnum())
            if len(token) < 2:
                continue
            idx = self._hash_token(token)
            vec[idx] += 1.0
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(t) for t in texts])


class TransformerEmbedder:
    """sentence-transformers based embeddings (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between query vectors and corpus vectors.
    Returns shape (n_queries, n_corpus).
    """
    # Normalize
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    c_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return q_norm @ c_norm.T


def get_embedder(use_transformers: bool = False, model_name: str = "all-MiniLM-L6-v2"):
    """Factory: get the best available embedder."""
    if use_transformers:
        try:
            return TransformerEmbedder(model_name)
        except ImportError:
            pass
    return SimpleEmbedder()
