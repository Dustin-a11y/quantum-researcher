"""Tests for data sources. DK 🦍"""
import pytest
import asyncio
from quantum_researcher.sources import (
    TextSource, JSONSource, SourceResult,
)


class TestTextSource:
    def test_basic(self):
        docs = [
            {"text": "Quantum computing is cool", "title": "QC"},
            {"text": "Classical computing is boring"},
        ]
        source = TextSource(docs)
        results = source.fetch_sync()
        assert len(results) == 2
        assert results[0].title == "QC"
        assert results[0].source == "text"
        assert "Quantum" in results[0].text

    def test_empty_text_skipped(self):
        docs = [{"text": ""}, {"text": "valid"}]
        source = TextSource(docs)
        results = source.fetch_sync()
        assert len(results) == 1

    def test_source_result_fields(self):
        r = SourceResult(text="hello", source="test", title="t", url="http://x")
        assert r.text == "hello"
        assert r.metadata == {}


class TestJSONSource:
    def test_basic(self):
        data = [
            {"content": "fact 1", "name": "F1"},
            {"content": "fact 2", "name": "F2"},
        ]
        source = JSONSource(data, text_key="content", title_key="name")
        results = source.fetch_sync()
        assert len(results) == 2
        assert results[0].title == "F1"

    def test_missing_text_key(self):
        data = [{"other": "no text key"}]
        source = JSONSource(data, text_key="text")
        results = source.fetch_sync()
        assert len(results) == 0

    def test_metadata_preserved(self):
        data = [{"text": "hi", "extra": "value", "count": 42}]
        source = JSONSource(data)
        results = source.fetch_sync()
        assert results[0].metadata["extra"] == "value"
        assert results[0].metadata["count"] == 42
