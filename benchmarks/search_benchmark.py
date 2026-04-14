#!/usr/bin/env python3
"""
Quantum Researcher Search Benchmark
Compare Grover search vs classical on datasets of varying sizes.
Reports: accuracy, speedup factor, time per query.

DK 🦍
"""

import time
import random
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_researcher.oracle import QuantumOracle
from quantum_researcher.grover_search import GroverSearch
from quantum_researcher.embeddings import SimpleEmbedder, cosine_similarity
import numpy as np


def generate_dataset(n: int, n_relevant: int, dim: int = 128):
    """Generate a synthetic dataset with known relevant items."""
    embedder = SimpleEmbedder(dim=dim)
    # Create a query topic
    query_text = "quantum computing algorithm optimization"
    query_vec = embedder.embed(query_text)

    items = []
    relevant_indices = set(random.sample(range(n), min(n_relevant, n)))

    for i in range(n):
        if i in relevant_indices:
            # Relevant: similar to query
            text = random.choice([
                "quantum algorithm for search optimization",
                "quantum computing speedup over classical",
                "grover search algorithm quantum gates",
                "quantum oracle encoding for optimization",
                "quantum walk algorithm on graphs",
            ])
        else:
            text = random.choice([
                "cooking recipes for italian pasta",
                "sports news basketball tournament",
                "weather forecast for next week",
                "movie reviews action films",
                "gardening tips for spring planting",
                "financial markets stock trading",
                "history of ancient civilizations",
                "travel guide to european cities",
                "fitness workout routine planning",
                "photography basics for beginners",
            ])
        vec = embedder.embed(text)
        items.append({"id": i, "text": text, "embedding": vec})

    return items, relevant_indices, query_vec, query_text


def benchmark_grover(items, query_text, n_relevant):
    """Run Grover search and measure performance."""
    oracle = QuantumOracle.from_keywords(
        keywords=["quantum"], items=items, use_quantum=False
    )
    grover = GroverSearch(oracle=oracle, confidence_threshold=0.02)
    
    start = time.perf_counter()
    result = grover.search()
    elapsed = time.perf_counter() - start

    found_ids = set(item["id"] for item in result.found_items)
    return {
        "method": "grover",
        "found": len(found_ids),
        "time_s": elapsed,
        "iterations": result.n_iterations,
        "method_detail": result.method,
    }


def benchmark_classical_linear(items, query_text):
    """Linear scan + keyword match."""
    start = time.perf_counter()
    found = [item for item in items if "quantum" in item["text"].lower()]
    elapsed = time.perf_counter() - start
    return {
        "method": "classical_linear",
        "found": len(found),
        "time_s": elapsed,
    }


def benchmark_embedding_search(items, query_vec, threshold=0.5):
    """Embedding similarity search."""
    start = time.perf_counter()
    found = []
    for item in items:
        sim = cosine_similarity(query_vec, item["embedding"])
        if sim > threshold:
            found.append(item)
    elapsed = time.perf_counter() - start
    return {
        "method": "embedding_search",
        "found": len(found),
        "time_s": elapsed,
        "threshold": threshold,
    }


def run_benchmark():
    sizes = [64, 256, 1024, 4096]
    relevance_ratio = 0.05  # 5% relevant

    results = []
    print("=" * 70)
    print("QUANTUM RESEARCHER SEARCH BENCHMARK")
    print("=" * 70)

    for n in sizes:
        n_relevant = max(2, int(n * relevance_ratio))
        items, true_relevant, query_vec, query_text = generate_dataset(n, n_relevant)

        print(f"\n--- N={n}, relevant={n_relevant} ({relevance_ratio*100:.0f}%) ---")

        # Grover
        g = benchmark_grover(items, query_text, n_relevant)
        print(f"  Grover:     found={g['found']}, time={g['time_s']:.4f}s, "
              f"iterations={g.get('iterations','?')}")

        # Classical linear
        c = benchmark_classical_linear(items, query_text)
        print(f"  Classical:  found={c['found']}, time={c['time_s']:.6f}s")

        # Embedding
        e = benchmark_embedding_search(items, query_vec, threshold=0.3)
        print(f"  Embedding:  found={e['found']}, time={e['time_s']:.6f}s")

        speedup = c["time_s"] / g["time_s"] if g["time_s"] > 0 else float('inf')
        theoretical_speedup = n ** 0.5 / n_relevant if n_relevant > 0 else 0

        result = {
            "n": n,
            "n_relevant": n_relevant,
            "grover": g,
            "classical": c,
            "embedding": e,
            "speedup_vs_classical": round(speedup, 2),
            "theoretical_sqrt_speedup": round(theoretical_speedup, 2),
        }
        results.append(result)

        print(f"  Speedup vs classical: {speedup:.2f}x "
              f"(theoretical √N/M: {theoretical_speedup:.2f}x)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  N={r['n']:>5}: Grover {r['grover']['time_s']:.4f}s "
              f"| Classical {r['classical']['time_s']:.6f}s "
              f"| Speedup {r['speedup_vs_classical']:.2f}x")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_results.json")
    with open(out_path, "w") as f:
        # Convert numpy types for JSON
        clean = json.loads(json.dumps(results, default=str))
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
