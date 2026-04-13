"""CLI entry point for quantum-researcher."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="quantum-researcher",
        description="Quantum-enhanced research agent",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve_parser = sub.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8505)
    serve_parser.add_argument("--reload", action="store_true")

    # demo
    sub.add_parser("demo", help="Run a quick demo")

    # version
    sub.add_parser("version", help="Print version")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "quantum_researcher.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "demo":
        _run_demo()

    elif args.command == "version":
        from quantum_researcher import __version__
        print(f"quantum-researcher v{__version__}")

    else:
        parser.print_help()


def _run_demo():
    """Run a quick demo of Grover search + quantum walk."""
    import networkx as nx
    from quantum_researcher.oracle import QuantumOracle
    from quantum_researcher.grover_search import GroverSearch
    from quantum_researcher.quantum_walk import QuantumWalk

    print("=== Quantum Researcher Demo ===\n")

    # 1. Grover Search
    print("[1] Grover Search — finding needles in haystacks")
    items = [
        {"id": i, "text": f"Document {i} about {'quantum computing' if i % 7 == 0 else 'random topics'}"}
        for i in range(64)
    ]
    oracle = QuantumOracle.from_keywords(keywords=["quantum"], items=items)
    grover = GroverSearch(oracle=oracle)
    result = grover.search()

    print(f"    Items: {result.n_items}")
    print(f"    Found: {len(result.found_items)}")
    print(f"    Iterations: {result.n_iterations} (classical would need ~{result.n_items // 2})")
    print(f"    Speedup: {result.speedup:.1f}x")
    print(f"    Method: {result.method}")
    print(f"    Time: {result.elapsed_ms:.1f}ms\n")

    # 2. Quantum Walk
    print("[2] Quantum Walk — discovering hidden connections")
    G = nx.karate_club_graph()
    walker = QuantumWalk(graph=G, walk_time=3.0, n_steps=20)
    walk_result = walker.walk(start=0)

    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"    Visited: {len(walk_result.visited_nodes)} nodes")
    print(f"    Top connections: {walk_result.visited_nodes[:5]}")
    print(f"    Method: {walk_result.method}")
    print(f"    Time: {walk_result.elapsed_ms:.1f}ms\n")

    # 3. Connection finding
    print("[3] Finding connections from node 0")
    connections = walker.find_connections(0)
    for node, prob in connections[:5]:
        print(f"    Node {node}: probability {prob:.4f}")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
