"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from quantum_researcher.api import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "qiskit_available" in data


class TestSearchEndpoint:
    def test_basic_search(self, client):
        payload = {
            "keywords": ["quantum"],
            "items": [
                {"text": "quantum computing rocks"},
                {"text": "classical stuff"},
                {"text": "quantum mechanics paper"},
                {"text": "cooking recipe"},
            ],
        }
        resp = client.post("/search", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["found"]) >= 2
        assert data["method"] in ("quantum", "classical_simulation")
        assert data["n_items"] == 4
        assert data["elapsed_ms"] > 0

    def test_no_matches(self, client):
        payload = {
            "keywords": ["nonexistent"],
            "items": [
                {"text": "nothing here"},
                {"text": "also nothing"},
            ],
        }
        resp = client.post("/search", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] == []

    def test_empty_keywords_rejected(self, client):
        payload = {
            "keywords": [],
            "items": [{"text": "test"}],
        }
        resp = client.post("/search", json=payload)
        assert resp.status_code == 422

    def test_empty_items_rejected(self, client):
        payload = {
            "keywords": ["test"],
            "items": [],
        }
        resp = client.post("/search", json=payload)
        assert resp.status_code == 422

    def test_max_results(self, client):
        items = [{"text": f"quantum item {i}"} for i in range(20)]
        payload = {
            "keywords": ["quantum"],
            "items": items,
            "max_results": 3,
        }
        resp = client.post("/search", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["found"]) <= 3


class TestWalkEndpoint:
    def test_basic_walk(self, client):
        payload = {
            "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["A", "C"]],
            "start": "A",
        }
        resp = client.post("/walk", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "A" in data["visit_probabilities"]
        assert data["method"] == "classical"

    def test_with_target(self, client):
        payload = {
            "edges": [["A", "B"], ["B", "C"], ["C", "D"]],
            "start": "A",
            "target": "D",
        }
        resp = client.post("/walk", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        if data["paths_found"]:
            assert data["paths_found"][0][0] == "A"
            assert data["paths_found"][0][-1] == "D"

    def test_nonexistent_start(self, client):
        payload = {
            "edges": [["A", "B"]],
            "start": "Z",
        }
        resp = client.post("/walk", json=payload)
        assert resp.status_code == 404

    def test_empty_edges_rejected(self, client):
        payload = {
            "edges": [],
            "start": "A",
        }
        resp = client.post("/walk", json=payload)
        assert resp.status_code == 422
