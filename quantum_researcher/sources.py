"""
Pluggable data sources for QuantumResearcher.
Each source returns List[Dict] with at minimum {'text': str, 'source': str}.

DK 🦍
"""

import re
import json
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class SourceResult:
    """Single item from a data source."""
    text: str
    source: str
    title: str = ""
    url: str = ""
    metadata: Dict = field(default_factory=dict)


class WebSource:
    """Fetch and extract text from web pages."""

    # Only allow http/https schemes
    ALLOWED_SCHEMES = ('http://', 'https://')
    MAX_URLS = 100

    def __init__(self, urls: List[str], extract_fn: Optional[Callable] = None):
        if len(urls) > self.MAX_URLS:
            raise ValueError(f"Max {self.MAX_URLS} URLs per source")
        # Validate URLs — no file://, ftp://, etc.
        for url in urls:
            if not any(url.lower().startswith(s) for s in self.ALLOWED_SCHEMES):
                raise ValueError(f"URL must start with http:// or https://: {url[:50]}")
        self.urls = urls
        self.extract_fn = extract_fn or self._default_extract

    @staticmethod
    def _default_extract(html: str) -> str:
        """Strip HTML tags, collapse whitespace."""
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.S)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.S)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def fetch(self) -> List[SourceResult]:
        """Fetch all URLs and extract text."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        results = []
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for url in self.urls:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        text = self.extract_fn(resp.text)
                        if text and len(text) > 50:
                            # Extract title
                            title_match = re.search(r'<title>([^<]+)</title>', resp.text, re.I)
                            title = title_match.group(1).strip() if title_match else url
                            results.append(SourceResult(
                                text=text[:10000],  # cap at 10k chars
                                source="web",
                                title=title,
                                url=url,
                            ))
                except Exception:
                    continue
        return results

    def fetch_sync(self) -> List[SourceResult]:
        """Synchronous wrapper."""
        import asyncio
        return asyncio.run(self.fetch())


class TextSource:
    """In-memory text documents."""

    def __init__(self, documents: List[Dict[str, str]]):
        """
        documents: list of {'text': str, 'title': str (optional), 'source': str (optional)}
        """
        self.documents = documents

    async def fetch(self) -> List[SourceResult]:
        return [
            SourceResult(
                text=doc["text"],
                source=doc.get("source", "text"),
                title=doc.get("title", doc["text"][:60]),
            )
            for doc in self.documents
            if doc.get("text")
        ]

    def fetch_sync(self) -> List[SourceResult]:
        import asyncio
        return asyncio.run(self.fetch())


class JSONSource:
    """Load data from JSON files or API responses."""

    def __init__(self, data: List[Dict], text_key: str = "text", title_key: str = "title"):
        self.data = data
        self.text_key = text_key
        self.title_key = title_key

    async def fetch(self) -> List[SourceResult]:
        results = []
        for item in self.data:
            text = item.get(self.text_key, "")
            if text:
                results.append(SourceResult(
                    text=str(text),
                    source="json",
                    title=str(item.get(self.title_key, text[:60])),
                    metadata={k: v for k, v in item.items()
                              if k not in (self.text_key, self.title_key)},
                ))
        return results

    def fetch_sync(self) -> List[SourceResult]:
        import asyncio
        return asyncio.run(self.fetch())


class FileSource:
    """Read text files from a directory."""

    # Allowed base directories to prevent path traversal
    ALLOWED_BASES = None  # Set to list of allowed dirs to restrict, e.g. ['/home/dt/data']
    MAX_FILE_SIZE = 1_000_000  # 1MB max per file

    def __init__(self, paths: List[str], encoding: str = "utf-8",
                 allowed_bases: Optional[List[str]] = None):
        self.paths = paths
        self.encoding = encoding
        self.allowed_bases = allowed_bases or self.ALLOWED_BASES

    def _is_allowed(self, path: str) -> bool:
        """Check path against allowed base directories."""
        import os
        if self.allowed_bases is None:
            return True  # No restriction
        real = os.path.realpath(path)
        return any(real.startswith(os.path.realpath(b)) for b in self.allowed_bases)

    async def fetch(self) -> List[SourceResult]:
        import os
        results = []
        for path in self.paths:
            if not self._is_allowed(path):
                continue
            if os.path.isfile(path):
                try:
                    if os.path.getsize(path) > self.MAX_FILE_SIZE:
                        continue
                    with open(path, encoding=self.encoding) as f:
                        text = f.read(self.MAX_FILE_SIZE)
                    if text.strip():
                        results.append(SourceResult(
                            text=text[:50000],
                            source="file",
                            title=os.path.basename(path),
                            url=path,
                        ))
                except Exception:
                    continue
            elif os.path.isdir(path):
                for fname in sorted(os.listdir(path))[:500]:  # cap directory listing
                    fpath = os.path.join(path, fname)
                    if not self._is_allowed(fpath):
                        continue
                    if os.path.isfile(fpath) and fname.endswith(('.txt', '.md', '.py', '.json')):
                        try:
                            if os.path.getsize(fpath) > self.MAX_FILE_SIZE:
                                continue
                            with open(fpath, encoding=self.encoding) as f:
                                text = f.read(self.MAX_FILE_SIZE)
                            if text.strip():
                                results.append(SourceResult(
                                    text=text[:50000],
                                    source="file",
                                    title=fname,
                                    url=fpath,
                                ))
                        except Exception:
                            continue
        return results

    def fetch_sync(self) -> List[SourceResult]:
        import asyncio
        return asyncio.run(self.fetch())


class APISource:
    """Fetch data from a REST API endpoint."""

    def __init__(self, url: str, headers: Optional[Dict] = None,
                 text_key: str = "text", results_key: str = "results"):
        self.url = url
        self.headers = headers or {}
        self.text_key = text_key
        self.results_key = results_key

    async def fetch(self) -> List[SourceResult]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required")

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self.url, headers=self.headers)
            if resp.status_code != 200:
                return []
            data = resp.json()

        items = data if isinstance(data, list) else data.get(self.results_key, [])
        results = []
        for item in items:
            text = item.get(self.text_key, "")
            if text:
                results.append(SourceResult(
                    text=str(text),
                    source="api",
                    title=str(item.get("title", text[:60])),
                    url=self.url,
                    metadata=item,
                ))
        return results

    def fetch_sync(self) -> List[SourceResult]:
        import asyncio
        return asyncio.run(self.fetch())
