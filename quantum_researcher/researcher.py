"""Quantum Researcher — the main agent loop.

Orchestrates quantum search, walks, and oracles to answer research
queries. Takes a question, builds a knowledge graph from sources,
runs quantum-enhanced search, and returns synthesized findings.

Pipeline:
  1. Parse query → extract search terms
  2. Gather sources (web, files, APIs, QMG)
  3. Build knowledge graph from source content
  4. Oracle: encode query as quantum predicate
  5. Grover: find relevant nodes in √N
  6. Quantum Walk: discover hidden connections
  7. Synthesize: combine findings into answer
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import networkx as nx
import numpy as np

from quantum_researcher.oracle import QuantumOracle
from quantum_researcher.grover_search import GroverSearch, SearchResult
from quantum_researcher.quantum_walk import QuantumWalk, WalkResult
from quantum_researcher import classical_fallback as cf


class KnowledgeSource(Protocol):
    """Protocol for pluggable knowledge sources."""

    async def fetch(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch documents matching query.

        Returns list of dicts with at least 'id', 'text', and optionally
        'embedding', 'metadata', 'source'.
        """
        ...


@dataclass
class ResearchFinding:
    """A single finding from the research process."""

    content: str
    source: str
    relevance: float  # 0-1, from Grover probability
    connections: list[str] = field(default_factory=list)  # related findings
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Complete research result."""

    query: str
    findings: list[ResearchFinding]
    knowledge_graph: nx.Graph
    search_result: SearchResult | None = None
    walk_result: WalkResult | None = None
    n_sources: int = 0
    elapsed_ms: float = 0
    quantum_speedup: float = 0


@dataclass
class QuantumResearcher:
    """Main quantum research agent.

    Parameters
    ----------
    sources : list of KnowledgeSource
        Pluggable knowledge sources to search.
    walk_time : float
        Quantum walk evolution time.
    grover_shots : int
        Measurement shots for Grover search.
    use_quantum : bool
        Enable quantum circuits (requires Qiskit).
    max_graph_nodes : int
        Maximum knowledge graph size.
    """

    sources: list[KnowledgeSource] = field(default_factory=list)
    walk_time: float = 2.0
    grover_shots: int = 1024
    use_quantum: bool = True
    max_graph_nodes: int = 512

    async def research(self, query: str, limit: int = 50) -> ResearchResult:
        """Execute a full research pipeline.

        Parameters
        ----------
        query : str
            The research question.
        limit : int
            Max documents to fetch per source.

        Returns
        -------
        ResearchResult with findings, graph, and metrics.
        """
        t0 = time.monotonic()

        # 1. Gather documents from all sources
        all_docs = await self._gather_sources(query, limit)

        if not all_docs:
            return ResearchResult(
                query=query, findings=[], knowledge_graph=nx.Graph(),
                n_sources=0, elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        # 2. Build knowledge graph
        kg = self._build_knowledge_graph(all_docs)

        # 3. Grover search — find most relevant documents
        keywords = self._extract_keywords(query)
        oracle = QuantumOracle.from_keywords(
            keywords=keywords,
            items=all_docs,
            text_key="text",
            use_quantum=self.use_quantum,
        )
        grover = GroverSearch(
            oracle=oracle,
            shots=self.grover_shots,
        )
        search_result = grover.search()

        # 4. Quantum walk — find hidden connections from top results
        walk_result = None
        if search_result.found_items and kg.number_of_nodes() > 1:
            walker = QuantumWalk(
                graph=kg,
                walk_time=self.walk_time,
                max_nodes=self.max_graph_nodes,
            )
            # Walk from the most relevant document
            top_doc_id = search_result.found_items[0].get("id", 0)
            if top_doc_id in kg.nodes:
                walk_result = walker.walk(top_doc_id)

        # 5. Synthesize findings
        findings = self._synthesize(
            all_docs, search_result, walk_result, kg
        )

        elapsed = (time.monotonic() - t0) * 1000

        return ResearchResult(
            query=query,
            findings=findings,
            knowledge_graph=kg,
            search_result=search_result,
            walk_result=walk_result,
            n_sources=len(all_docs),
            elapsed_ms=elapsed,
            quantum_speedup=search_result.speedup,
        )

    async def _gather_sources(
        self, query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Fetch documents from all registered sources."""
        all_docs: list[dict[str, Any]] = []
        for source in self.sources:
            try:
                docs = await source.fetch(query, limit)
                # Ensure each doc has an id
                for i, doc in enumerate(docs):
                    if "id" not in doc:
                        doc["id"] = f"doc_{len(all_docs) + i}"
                all_docs.extend(docs)
            except Exception:
                continue  # Skip failed sources
        return all_docs

    def _build_knowledge_graph(
        self, docs: list[dict[str, Any]]
    ) -> nx.Graph:
        """Build a knowledge graph from documents.

        Nodes = documents. Edges = shared concepts/keywords.
        Edge weight = number of shared keywords.
        """
        G = nx.Graph()

        # Add nodes
        for doc in docs:
            G.add_node(
                doc["id"],
                text=doc.get("text", ""),
                source=doc.get("source", "unknown"),
                metadata=doc.get("metadata", {}),
            )

        # Add edges based on keyword overlap
        keyword_sets = {}
        for doc in docs:
            text = doc.get("text", "").lower()
            # Simple keyword extraction: words > 4 chars
            words = set(
                w.strip(".,!?;:\"'()[]{}") for w in text.split()
                if len(w.strip(".,!?;:\"'()[]{}")) > 4
            )
            keyword_sets[doc["id"]] = words

        doc_ids = [doc["id"] for doc in docs]
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                shared = keyword_sets[doc_ids[i]] & keyword_sets[doc_ids[j]]
                if shared:
                    weight = len(shared)
                    G.add_edge(doc_ids[i], doc_ids[j], weight=weight, shared_keywords=shared)

        return G

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract search keywords from query.

        Simple extraction: split on spaces, filter stopwords.
        Could be enhanced with NLP in future.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "because",
            "but", "and", "or", "if", "while", "about", "what", "which",
            "who", "whom", "this", "that", "these", "those", "it", "its",
        }
        words = query.lower().split()
        return [w.strip(".,!?;:\"'()[]{}") for w in words if w.lower() not in stopwords and len(w) > 2]

    def _synthesize(
        self,
        docs: list[dict[str, Any]],
        search_result: SearchResult,
        walk_result: WalkResult | None,
        kg: nx.Graph,
    ) -> list[ResearchFinding]:
        """Synthesize findings from search and walk results."""
        findings: list[ResearchFinding] = []
        doc_map = {doc["id"]: doc for doc in docs}

        # Primary findings from Grover search
        for i, item in enumerate(search_result.found_items):
            prob = 0.0
            if search_result.probabilities is not None and i < len(search_result.found_indices):
                prob = float(search_result.probabilities[search_result.found_indices[i]])

            # Find connections via knowledge graph
            doc_id = item.get("id")
            connections = []
            if doc_id and doc_id in kg:
                neighbors = sorted(
                    kg.neighbors(doc_id),
                    key=lambda n: kg[doc_id][n].get("weight", 0),
                    reverse=True,
                )[:5]
                for n in neighbors:
                    if n in doc_map:
                        connections.append(doc_map[n].get("text", str(n))[:100])

            findings.append(ResearchFinding(
                content=item.get("text", ""),
                source=item.get("source", "unknown"),
                relevance=prob,
                connections=connections,
                metadata=item.get("metadata", {}),
            ))

        # Secondary findings from quantum walk (hidden connections)
        if walk_result:
            for node in walk_result.visited_nodes[:5]:
                if node in doc_map and node not in {
                    item.get("id") for item in search_result.found_items
                }:
                    prob = walk_result.visit_probabilities.get(node, 0)
                    findings.append(ResearchFinding(
                        content=doc_map[node].get("text", ""),
                        source=f"quantum_walk:{doc_map[node].get('source', 'unknown')}",
                        relevance=prob,
                        metadata={"discovered_by": "quantum_walk"},
                    ))

        return findings

    def add_source(self, source: KnowledgeSource) -> None:
        """Register a knowledge source."""
        self.sources.append(source)
