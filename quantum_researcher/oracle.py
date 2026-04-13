"""Quantum Oracle — encode search queries as quantum oracles.

The oracle marks target states (matching documents/nodes) in a superposition,
enabling Grover's algorithm to amplify their probability amplitude.

Two modes:
  1. Qiskit circuit (real QPU / simulator) — uses phase kickback
  2. Classical bitmap (fallback) — boolean mask over items
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate

    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


@dataclass
class OracleResult:
    """Result of oracle encoding."""

    n_qubits: int
    n_items: int
    n_marked: int
    marked_indices: list[int]
    circuit: Any | None = None  # QuantumCircuit when qiskit available
    bitmap: np.ndarray | None = None  # classical fallback


@dataclass
class QuantumOracle:
    """Encode a search predicate as a quantum oracle.

    The oracle takes a predicate function and a list of items.
    It identifies which items match (marked states) and builds
    either a Qiskit circuit or classical bitmap for downstream use.

    Parameters
    ----------
    predicate : callable
        Function that returns True for items matching the search query.
    items : sequence
        The search space (documents, nodes, records, etc.).
    use_quantum : bool
        If True and Qiskit available, build a quantum circuit.
    """

    predicate: Callable[[Any], bool]
    items: Sequence[Any] = field(default_factory=list)
    use_quantum: bool = True

    def encode(self) -> OracleResult:
        """Encode the predicate over items into an oracle."""
        n_items = len(self.items)
        if n_items == 0:
            return OracleResult(
                n_qubits=0, n_items=0, n_marked=0,
                marked_indices=[], bitmap=np.array([], dtype=bool),
            )

        # Find marked items
        marked = [i for i, item in enumerate(self.items) if self.predicate(item)]
        n_qubits = max(1, math.ceil(math.log2(n_items))) if n_items > 1 else 1

        # Build bitmap (always — used by classical fallback)
        bitmap = np.zeros(n_items, dtype=bool)
        for idx in marked:
            bitmap[idx] = True

        # Build quantum circuit if possible
        circuit = None
        if self.use_quantum and _HAS_QISKIT and n_items > 1:
            circuit = self._build_circuit(n_qubits, marked)

        return OracleResult(
            n_qubits=n_qubits,
            n_items=n_items,
            n_marked=len(marked),
            marked_indices=marked,
            circuit=circuit,
            bitmap=bitmap,
        )

    def _build_circuit(self, n_qubits: int, marked: list[int]) -> Any:
        """Build Grover oracle circuit using phase kickback.

        For each marked state |x⟩, flip its phase: |x⟩ → -|x⟩
        This is done by applying X gates to put qubits in the right
        basis state, then a multi-controlled Z, then undoing the X gates.
        """
        qc = QuantumCircuit(n_qubits, name="oracle")

        for idx in marked:
            # Convert index to binary and flip non-1 qubits
            bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian
            flip_qubits = [i for i, b in enumerate(bits) if b == "0"]

            # X gates to transform |marked⟩ → |11...1⟩
            for q in flip_qubits:
                qc.x(q)

            # Multi-controlled Z = H on target, MCX, H on target
            if n_qubits == 1:
                qc.z(0)
            elif n_qubits == 2:
                qc.cz(0, 1)
            else:
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)

            # Undo X gates
            for q in flip_qubits:
                qc.x(q)

        return qc

    @staticmethod
    def from_keywords(
        keywords: list[str],
        items: Sequence[dict[str, Any]],
        text_key: str = "text",
        use_quantum: bool = True,
    ) -> "QuantumOracle":
        """Create oracle from keyword matching.

        Parameters
        ----------
        keywords : list of str
            Search terms (any match = marked).
        items : sequence of dicts
            Each item should have a text field to search.
        text_key : str
            Key in each item dict containing searchable text.
        """
        kw_lower = [k.lower() for k in keywords]

        def predicate(item: dict) -> bool:
            text = str(item.get(text_key, "")).lower()
            return any(kw in text for kw in kw_lower)

        return QuantumOracle(predicate=predicate, items=items, use_quantum=use_quantum)

    @staticmethod
    def from_embedding_similarity(
        query_embedding: np.ndarray,
        item_embeddings: np.ndarray,
        threshold: float = 0.7,
        items: Sequence[Any] | None = None,
        use_quantum: bool = True,
    ) -> "QuantumOracle":
        """Create oracle from embedding cosine similarity.

        Parameters
        ----------
        query_embedding : ndarray
            Query vector (1D).
        item_embeddings : ndarray
            Matrix of item vectors (n_items × dim).
        threshold : float
            Cosine similarity threshold to mark as match.
        items : sequence, optional
            Original items. If None, uses indices.
        """
        # Normalize
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        i_norms = item_embeddings / (
            np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-10
        )
        similarities = i_norms @ q_norm

        actual_items = items if items is not None else list(range(len(item_embeddings)))

        def predicate(item: Any) -> bool:
            idx = actual_items.index(item) if items is not None else item
            return float(similarities[idx]) >= threshold

        return QuantumOracle(
            predicate=predicate, items=list(actual_items), use_quantum=use_quantum
        )
