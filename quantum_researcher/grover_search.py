"""Grover Search — quantum amplitude amplification for √N speedup.

Given an oracle that marks target states, Grover's algorithm amplifies
their probability amplitude so they're found with high probability
in O(√N) queries instead of O(N).

Two modes:
  1. Qiskit simulation/QPU — full quantum Grover circuit
  2. Classical simulation — simulates the amplitude amplification math
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from quantum_researcher.oracle import OracleResult, QuantumOracle

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


@dataclass
class SearchResult:
    """Result from Grover search."""

    found_indices: list[int]
    found_items: list[Any]
    n_iterations: int
    n_items: int
    speedup: float  # √N / actual_queries ratio
    elapsed_ms: float
    method: str  # "quantum" | "classical_simulation" | "classical_fallback"
    probabilities: np.ndarray | None = None


@dataclass
class GroverSearch:
    """Grover's search algorithm with quantum and classical modes.

    Parameters
    ----------
    oracle : QuantumOracle
        Oracle with predicate and items defined.
    max_iterations : int, optional
        Override automatic iteration count. Default: optimal √N.
    shots : int
        Number of measurement shots (quantum mode).
    confidence_threshold : float
        Minimum probability to consider an item "found".
    """

    oracle: QuantumOracle
    max_iterations: int | None = None
    shots: int = 1024
    confidence_threshold: float = 0.1

    def search(self) -> SearchResult:
        """Execute Grover search.

        Returns the found items with their probabilities.
        Automatically chooses quantum or classical mode.
        """
        t0 = time.monotonic()
        oracle_result = self.oracle.encode()

        if oracle_result.n_items == 0:
            return SearchResult(
                found_indices=[], found_items=[], n_iterations=0,
                n_items=0, speedup=0, elapsed_ms=0, method="empty",
            )

        if oracle_result.n_marked == 0:
            elapsed = (time.monotonic() - t0) * 1000
            return SearchResult(
                found_indices=[], found_items=[], n_iterations=0,
                n_items=oracle_result.n_items, speedup=0,
                elapsed_ms=elapsed, method="no_match",
            )

        # Optimal iterations: π/4 * √(N/M) where M = marked items
        n = oracle_result.n_items
        m = oracle_result.n_marked
        optimal_iters = max(1, round(math.pi / 4 * math.sqrt(n / m)))
        n_iters = self.max_iterations or optimal_iters

        if oracle_result.circuit is not None and _HAS_QISKIT:
            result = self._quantum_search(oracle_result, n_iters)
        else:
            result = self._classical_simulation(oracle_result, n_iters)

        result.elapsed_ms = (time.monotonic() - t0) * 1000

        # Calculate speedup: classical needs N/2 avg, Grover needs √N
        classical_queries = n / 2
        result.speedup = classical_queries / max(n_iters, 1)

        return result

    def _quantum_search(self, oracle_result: OracleResult, n_iters: int) -> SearchResult:
        """Run Grover search using Qiskit circuit."""
        n_qubits = oracle_result.n_qubits

        # Build full Grover circuit
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Initial superposition
        qc.h(range(n_qubits))

        # Grover iterations
        for _ in range(n_iters):
            # Oracle
            qc.compose(oracle_result.circuit, inplace=True)
            # Diffusion operator
            qc.compose(self._diffusion_operator(n_qubits), inplace=True)

        # Measure
        qc.measure(range(n_qubits), range(n_qubits))

        # Run
        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=self.shots)
        result = job.result()
        counts = result[0].data.meas.get_counts()

        # Parse results
        n_items = oracle_result.n_items
        probabilities = np.zeros(n_items)

        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            if idx < n_items:
                probabilities[idx] = count / self.shots

        # Find items above threshold
        found_indices = [
            i for i in range(n_items)
            if probabilities[i] >= self.confidence_threshold
        ]

        # Sort by probability (highest first)
        found_indices.sort(key=lambda i: probabilities[i], reverse=True)

        found_items = [self.oracle.items[i] for i in found_indices]

        return SearchResult(
            found_indices=found_indices,
            found_items=found_items,
            n_iterations=n_iters,
            n_items=n_items,
            speedup=0,  # calculated by caller
            elapsed_ms=0,
            method="quantum",
            probabilities=probabilities,
        )

    def _classical_simulation(
        self, oracle_result: OracleResult, n_iters: int
    ) -> SearchResult:
        """Simulate Grover's algorithm classically using state vector math.

        This gives identical results to quantum execution but runs on CPU.
        Still demonstrates the √N query advantage in iteration count.
        """
        n = oracle_result.n_items
        n_states = 2 ** oracle_result.n_qubits

        # Initialize uniform superposition
        amplitudes = np.full(n_states, 1.0 / math.sqrt(n_states))

        # Oracle bitmap (extended to power of 2)
        oracle_mask = np.zeros(n_states, dtype=bool)
        for idx in oracle_result.marked_indices:
            oracle_mask[idx] = True

        # Grover iterations
        for _ in range(n_iters):
            # Oracle: flip phase of marked states
            amplitudes[oracle_mask] *= -1

            # Diffusion: 2|ψ⟩⟨ψ| - I
            mean_amp = np.mean(amplitudes)
            amplitudes = 2 * mean_amp - amplitudes

        # Probabilities
        probs_full = np.abs(amplitudes) ** 2
        probabilities = probs_full[:n]

        # Normalize probabilities to account for unused states
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total

        found_indices = [
            i for i in range(n) if probabilities[i] >= self.confidence_threshold
        ]
        found_indices.sort(key=lambda i: probabilities[i], reverse=True)
        found_items = [self.oracle.items[i] for i in found_indices]

        return SearchResult(
            found_indices=found_indices,
            found_items=found_items,
            n_iterations=n_iters,
            n_items=n,
            speedup=0,
            elapsed_ms=0,
            method="classical_simulation",
            probabilities=probabilities,
        )

    @staticmethod
    def _diffusion_operator(n_qubits: int) -> Any:
        """Build Grover diffusion operator: 2|s⟩⟨s| - I.

        Reflects amplitudes about the mean, amplifying marked states.
        """
        qc = QuantumCircuit(n_qubits, name="diffusion")

        qc.h(range(n_qubits))
        qc.x(range(n_qubits))

        # Multi-controlled Z
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

        return qc
