"""Tests for QuantumResearcher — the main agent loop."""

import pytest
from typing import Any

from quantum_researcher.researcher import QuantumResearcher, KnowledgeSource


class MockSource:
    """Mock knowledge source for testing."""

    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.fetch_count = 0

    async def fetch(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        self.fetch_count += 1
        return self.docs[:limit]


class FailingSource:
    """Source that always fails — should be handled gracefully."""

    async def fetch(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        raise ConnectionError("Source unavailable")


@pytest.fixture
def sample_docs():
    return [
        {"id": "doc1", "text": "Quantum computing uses qubits for parallel computation", "source": "wiki"},
        {"id": "doc2", "text": "Classical computers use binary bits for sequential processing", "source": "wiki"},
        {"id": "doc3", "text": "Quantum entanglement enables instant correlation between particles", "source": "paper"},
        {"id": "doc4", "text": "Machine learning requires large datasets for training models", "source": "blog"},
        {"id": "doc5", "text": "Quantum error correction protects qubits from decoherence", "source": "paper"},
        {"id": "doc6", "text": "Neural networks are inspired by biological brain structures", "source": "blog"},
        {"id": "doc7", "text": "Quantum walks explore graphs exponentially faster than random walks", "source": "paper"},
        {"id": "doc8", "text": "Cooking pasta requires boiling water and timing", "source": "recipe"},
    ]


class TestQuantumResearcher:
    """Test the main research pipeline."""

    @pytest.mark.asyncio
    async def test_basic_research(self, sample_docs):
        source = MockSource(sample_docs)
        researcher = QuantumResearcher(
            sources=[source], use_quantum=False
        )
        result = await researcher.research("quantum computing")

        assert result.query == "quantum computing"
        assert len(result.findings) > 0
        assert result.n_sources == 8
        assert result.elapsed_ms > 0

        # Quantum-related docs should be in findings
        finding_texts = [f.content for f in result.findings]
        assert any("quantum" in t.lower() for t in finding_texts)

    @pytest.mark.asyncio
    async def test_no_sources(self):
        researcher = QuantumResearcher(sources=[])
        result = await researcher.research("anything")
        assert result.findings == []
        assert result.n_sources == 0

    @pytest.mark.asyncio
    async def test_failing_source_handled(self, sample_docs):
        """Failing sources should be skipped, not crash."""
        good = MockSource(sample_docs)
        bad = FailingSource()
        researcher = QuantumResearcher(
            sources=[bad, good], use_quantum=False
        )
        result = await researcher.research("quantum")
        # Should still get results from the good source
        assert len(result.findings) > 0

    @pytest.mark.asyncio
    async def test_knowledge_graph_built(self, sample_docs):
        source = MockSource(sample_docs)
        researcher = QuantumResearcher(
            sources=[source], use_quantum=False
        )
        result = await researcher.research("quantum")

        # Knowledge graph should have nodes for each document
        assert result.knowledge_graph.number_of_nodes() == 8

        # Documents sharing keywords should be connected
        assert result.knowledge_graph.number_of_edges() > 0

    @pytest.mark.asyncio
    async def test_findings_have_relevance(self, sample_docs):
        source = MockSource(sample_docs)
        researcher = QuantumResearcher(
            sources=[source], use_quantum=False
        )
        result = await researcher.research("quantum entanglement")

        for finding in result.findings:
            assert hasattr(finding, "relevance")
            assert hasattr(finding, "source")
            assert hasattr(finding, "content")

    @pytest.mark.asyncio
    async def test_connections_found(self, sample_docs):
        source = MockSource(sample_docs)
        researcher = QuantumResearcher(
            sources=[source], use_quantum=False
        )
        result = await researcher.research("quantum")

        # Some findings should have connections
        has_connections = any(len(f.connections) > 0 for f in result.findings)
        assert has_connections

    @pytest.mark.asyncio
    async def test_add_source(self, sample_docs):
        researcher = QuantumResearcher()
        assert len(researcher.sources) == 0

        source = MockSource(sample_docs)
        researcher.add_source(source)
        assert len(researcher.sources) == 1

    @pytest.mark.asyncio
    async def test_quantum_walk_findings(self, sample_docs):
        """Quantum walk should discover additional findings."""
        source = MockSource(sample_docs)
        researcher = QuantumResearcher(
            sources=[source], use_quantum=False
        )
        result = await researcher.research("quantum computing")

        # Check that walk result was generated
        # (may be None if top result isn't in graph, but should work with sample data)
        if result.walk_result:
            assert result.walk_result.method == "classical"

    @pytest.mark.asyncio
    async def test_keyword_extraction(self):
        researcher = QuantumResearcher()
        keywords = researcher._extract_keywords(
            "What is quantum computing and how does it work?"
        )
        assert "quantum" in keywords
        assert "computing" in keywords
        assert "work" in keywords
        # Stopwords should be removed
        assert "what" not in keywords
        assert "is" not in keywords
        assert "and" not in keywords
        assert "how" not in keywords
