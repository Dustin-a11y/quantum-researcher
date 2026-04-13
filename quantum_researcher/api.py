"""FastAPI endpoints for Quantum Researcher.

REST API for running quantum-enhanced research queries.
Standalone or mountable into existing FastAPI apps.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from quantum_researcher.oracle import QuantumOracle
from quantum_researcher.grover_search import GroverSearch
from quantum_researcher.quantum_walk import QuantumWalk

import networkx as nx


# --- Models ---


class SearchRequest(BaseModel):
    """Request for Grover search."""

    keywords: list[str] = Field(..., min_length=1, description="Search terms")
    items: list[dict[str, Any]] = Field(..., min_length=1, description="Items to search")
    text_key: str = Field("text", description="Key in items containing searchable text")
    use_quantum: bool = Field(True, description="Use quantum circuits if available")
    max_results: int = Field(20, ge=1, le=100)


class SearchResponse(BaseModel):
    """Response from Grover search."""

    found: list[dict[str, Any]]
    n_iterations: int
    n_items: int
    speedup: float
    elapsed_ms: float
    method: str


class WalkRequest(BaseModel):
    """Request for quantum walk."""

    edges: list[list[Any]] = Field(..., min_length=1, description="Edge list: [[src, tgt], ...]")
    start: str = Field(..., description="Starting node")
    target: str | None = Field(None, description="Optional target node")
    walk_time: float = Field(2.0, gt=0, le=20.0)
    n_steps: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.01, ge=0, le=1.0)


class WalkResponse(BaseModel):
    """Response from quantum walk."""

    visited_nodes: list[str]
    visit_probabilities: dict[str, float]
    paths_found: list[list[str]]
    n_steps: int
    elapsed_ms: float
    method: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    qiskit_available: bool


# --- App ---


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    from quantum_researcher import __version__
    from quantum_researcher.grover_search import _HAS_QISKIT

    app = FastAPI(
        title="Quantum Researcher API",
        description="Quantum-enhanced research — Grover search, quantum walks, oracle encoding",
        version=__version__,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            version=__version__,
            qiskit_available=_HAS_QISKIT,
        )

    @app.post("/search", response_model=SearchResponse)
    async def grover_search(req: SearchRequest):
        """Run Grover search over provided items."""
        if len(req.items) > 10000:
            raise HTTPException(400, "Max 10000 items")

        oracle = QuantumOracle.from_keywords(
            keywords=req.keywords,
            items=req.items,
            text_key=req.text_key,
            use_quantum=req.use_quantum,
        )
        grover = GroverSearch(oracle=oracle)
        result = grover.search()

        found = result.found_items[:req.max_results]

        return SearchResponse(
            found=found,
            n_iterations=result.n_iterations,
            n_items=result.n_items,
            speedup=result.speedup,
            elapsed_ms=result.elapsed_ms,
            method=result.method,
        )

    @app.post("/walk", response_model=WalkResponse)
    async def quantum_walk(req: WalkRequest):
        """Run quantum walk on provided graph."""
        if len(req.edges) > 50000:
            raise HTTPException(400, "Max 50000 edges")

        G = nx.Graph()
        for edge in req.edges:
            if len(edge) >= 2:
                G.add_edge(str(edge[0]), str(edge[1]))

        if req.start not in G:
            raise HTTPException(404, f"Start node '{req.start}' not in graph")

        walker = QuantumWalk(
            graph=G,
            walk_time=req.walk_time,
            n_steps=req.n_steps,
            threshold=req.threshold,
        )

        result = walker.walk(req.start, target=req.target)

        return WalkResponse(
            visited_nodes=[str(n) for n in result.visited_nodes],
            visit_probabilities={str(k): v for k, v in result.visit_probabilities.items()},
            paths_found=[[str(n) for n in p] for p in result.paths_found],
            n_steps=result.n_steps,
            elapsed_ms=result.elapsed_ms,
            method=result.method,
        )

    return app


app = create_app()
