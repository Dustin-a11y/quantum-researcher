"""Quantum Researcher — quantum-enhanced knowledge discovery.

Grover search for √N speedup, quantum walks for knowledge graph traversal,
and oracle encoding for query→quantum translation.

MIT License · Coinkong
"""

__version__ = "0.1.0"

from quantum_researcher.oracle import QuantumOracle
from quantum_researcher.grover_search import GroverSearch
from quantum_researcher.quantum_walk import QuantumWalk
from quantum_researcher.researcher import QuantumResearcher

__all__ = [
    "QuantumOracle",
    "GroverSearch",
    "QuantumWalk",
    "QuantumResearcher",
]
