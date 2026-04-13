# Quantum Researcher 🔬

**Quantum-enhanced research agent — find answers faster with quantum algorithms.**

Grover search for √N speedup on document filtering. Quantum walks for discovering hidden connections in knowledge graphs. Oracle encoding to translate any search query into a quantum predicate.

Works standalone or plugs into [Quantum Memory Graph](https://github.com/Dustin-a11y/quantum-memory-graph) for persistent agent memory.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What It Does

| Module | Algorithm | Speedup | Use Case |
|--------|-----------|---------|----------|
| `grover_search` | Grover's Algorithm | O(√N) vs O(N) | Finding relevant documents |
| `quantum_walk` | Continuous-Time QW | Exponential on some graphs | Discovering hidden connections |
| `oracle` | Phase Kickback | — | Encoding search queries as quantum circuits |
| `classical_fallback` | BFS / Linear | Baseline | CPU fallback when no QPU |

## Quick Start

```bash
pip install quantum-researcher
```

### Grover Search — Find Needles in Haystacks

```python
from quantum_researcher import QuantumOracle, GroverSearch

# Your documents
docs = [
    {"text": "quantum computing breakthrough"},
    {"text": "classical music concert"},
    {"text": "quantum error correction paper"},
    {"text": "cooking recipes"},
    # ... hundreds more
]

# Create oracle + search
oracle = QuantumOracle.from_keywords(keywords=["quantum"], items=docs)
result = GroverSearch(oracle=oracle).search()

print(f"Found {len(result.found_items)} docs in {result.n_iterations} iterations")
print(f"Classical would need ~{result.n_items // 2} — that's {result.speedup:.1f}x faster")
```

### Quantum Walk — Discover Hidden Connections

```python
import networkx as nx
from quantum_researcher import QuantumWalk

# Your knowledge graph
G = nx.Graph()
G.add_edges_from([
    ("quantum", "computing"), ("computing", "algorithms"),
    ("algorithms", "optimization"), ("optimization", "finance"),
    ("quantum", "physics"), ("physics", "energy"),
])

walker = QuantumWalk(graph=G, walk_time=3.0)
connections = walker.find_connections("quantum")

for node, probability in connections:
    print(f"  {node}: {probability:.4f}")
# Quantum walker finds non-obvious paths classical walk might miss
```

### Full Research Pipeline

```python
from quantum_researcher import QuantumResearcher

class MySource:
    async def fetch(self, query, limit=50):
        # Your data source — API, database, files, etc.
        return [{"id": "1", "text": "...", "source": "my_api"}]

researcher = QuantumResearcher(sources=[MySource()])
result = await researcher.research("quantum error correction")

for finding in result.findings:
    print(f"[{finding.relevance:.2f}] {finding.content[:100]}")
    if finding.connections:
        print(f"  Connected to: {finding.connections[:3]}")
```

### REST API

```bash
quantum-researcher serve --port 8505
```

```bash
# Grover search
curl -X POST http://localhost:8505/search \
  -H "Content-Type: application/json" \
  -d '{"keywords": ["quantum"], "items": [{"text": "quantum paper"}, {"text": "cooking"}]}'

# Quantum walk
curl -X POST http://localhost:8505/walk \
  -H "Content-Type: application/json" \
  -d '{"edges": [["A","B"],["B","C"],["A","C"]], "start": "A"}'
```

## Installation

```bash
# Basic (classical fallback only)
pip install quantum-researcher

# With Qiskit (real quantum circuits)
pip install quantum-researcher[quantum]

# With QMG integration
pip install quantum-researcher[qmg]

# Everything
pip install quantum-researcher[all]
```

## How It Works

### Grover's Algorithm

Classical search checks items one by one: O(N). Grover's algorithm puts all items in superposition, marks matches with an oracle, and amplifies their probability with O(√N) iterations.

For 1 million documents: classical needs ~500,000 checks. Grover needs ~785.

### Quantum Walks

Classical random walks explore graphs by randomly choosing neighbors. Quantum walks use superposition and interference — the walker exists at multiple nodes simultaneously, and interference guides it toward relevant structures.

On certain graph topologies, quantum walks achieve exponential speedup over classical random walks.

### Oracle Encoding

The oracle translates your search predicate (keywords, embeddings, custom logic) into a quantum circuit that flips the phase of matching states. This is the "quantum function" that Grover searches over.

## Optional: QMG Integration

```python
from quantum_researcher import QuantumResearcher
from quantum_memory_graph import QuantumMemoryGraph

# QMG as a knowledge source
class QMGSource:
    def __init__(self, qmg):
        self.qmg = qmg

    async def fetch(self, query, limit=50):
        results = self.qmg.search(query, limit=limit)
        return [{"id": r.id, "text": r.text, "source": "qmg"} for r in results]

qmg = QuantumMemoryGraph(...)
researcher = QuantumResearcher(sources=[QMGSource(qmg)])
```

## Benchmarks

Run the demo:

```bash
quantum-researcher demo
```

## Development

```bash
git clone https://github.com/Dustin-a11y/quantum-researcher
cd quantum-researcher
pip install -e ".[dev]"
pytest
```

## License

MIT · Coinkong
