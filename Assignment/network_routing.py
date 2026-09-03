"""
Smart Network Routing and Congestion Optimization System
CSA0607 - Design and Analysis of Algorithms
Implements: Graph model, Dijkstra, Bellman-Ford, Floyd-Warshall,
Greedy congestion-aware routing, congestion/bottleneck detection,
DP-based multi-request optimization, and a Hybrid Smart Routing Algorithm.
"""

import heapq
import time
import random
from collections import defaultdict

INF = float('inf')


class NetworkGraph:
    """Directed weighted graph representing a router network."""

    def __init__(self):
        self.nodes = set()
        self.adj = defaultdict(list)      # u -> [(v, cost)]
        self.congestion = {}              # (u, v) -> congestion in [0,1]

    def add_router(self, name):
        self.nodes.add(name)

    def add_link(self, u, v, cost, congestion=0.0, bidirectional=True):
        self.add_router(u)
        self.add_router(v)
        self.adj[u].append((v, cost))
        self.congestion[(u, v)] = congestion
        if bidirectional:
            self.adj[v].append((u, cost))
            self.congestion[(v, u)] = congestion

    def edges(self):
        for u in self.adj:
            for v, c in self.adj[u]:
                yield u, v, c


# ---------------------------------------------------------------- Dijkstra
def dijkstra(graph, source):
    dist = {n: INF for n in graph.nodes}
    prev = {n: None for n in graph.nodes}
    dist[source] = 0
    pq = [(0, source)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev, source, target):
    path, node, seen = [], target, set()
    while node is not None and node not in seen:
        seen.add(node)
        path.append(node)
        if node == source:
            break
        node = prev.get(node)
    path.reverse()
    return path if path and path[0] == source else []


# ------------------------------------------------------------- Bellman-Ford
def bellman_ford(graph, source):
    dist = {n: INF for n in graph.nodes}
    prev = {n: None for n in graph.nodes}
    dist[source] = 0
    edges = list(graph.edges())
    for _ in range(len(graph.nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break
    negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            negative_cycle = True
            break
    return dist, prev, negative_cycle


# ----------------------------------------------------------- Floyd-Warshall
def floyd_warshall(graph):
    nodes = list(graph.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    dist = [[INF] * n for _ in range(n)]
    nxt = [[None] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        nxt[i][i] = i
    for u, v, w in graph.edges():
        i, j = idx[u], idx[v]
        if w < dist[i][j]:
            dist[i][j] = w
            nxt[i][j] = j
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]
    return dist, nxt, nodes, idx


def fw_path(nxt, idx, nodes, u, v):
    if nxt[idx[u]][idx[v]] is None:
        return []
    path = [u]
    i, j = idx[u], idx[v]
    while i != j:
        i = nxt[i][j]
        path.append(nodes[i])
    return path


# ---------------------------------------------------- Congestion detection
def detect_congestion(graph, threshold=0.7):
    hot_links = [(u, v) for (u, v), c in graph.congestion.items() if c >= threshold]
    load = defaultdict(float)
    for (u, v), c in graph.congestion.items():
        load[u] += c
        load[v] += c
    degree = {n: len(graph.adj[n]) for n in graph.nodes}
    bottleneck_routers = [n for n in graph.nodes
                           if degree[n] > 0 and load[n] / degree[n] >= threshold]
    return hot_links, bottleneck_routers


# --------------------------------------------------- Greedy least-congested
def greedy_least_congested_path(graph, source, target):
    """At each hop greedily choose the unvisited neighbour with lowest congestion."""
    path = [source]
    visited = {source}
    current = source
    while current != target:
        candidates = [(graph.congestion.get((current, v), 1.0), v)
                      for v, _ in graph.adj[current] if v not in visited]
        if not candidates:
            return []  # dead end
        candidates.sort()
        _, nxt = candidates[0]
        path.append(nxt)
        visited.add(nxt)
        current = nxt
        if len(path) > len(graph.nodes):
            return []
    return path


# ----------------------------------------------------------- Hybrid routing
def hybrid_cost(graph, u, v, base_cost, alpha=0.6, beta=0.4):
    """Weighted blend of normalized distance cost and congestion penalty."""
    cong = graph.congestion.get((u, v), 0.0)
    return alpha * base_cost + beta * (cong * 10)


def hybrid_dijkstra(graph, source, target, alpha=0.6, beta=0.4):
    dist = {n: INF for n in graph.nodes}
    prev = {n: None for n in graph.nodes}
    dist[source] = 0
    pq = [(0, source)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in graph.adj[u]:
            hc = hybrid_cost(graph, u, v, w, alpha, beta)
            nd = d + hc
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return reconstruct_path(prev, source, target), dist[target]


# ------------------------------- DP optimization for multiple requests
def optimize_multiple_requests(graph, requests, capacity_per_link=3):
    """
    DP / greedy-optimization hybrid: assigns each (source,target) request
    a hybrid-cost path while tracking link usage so heavily-used links
    become progressively more expensive (congestion-aware load balancing).
    requests: list of (source, target)
    Returns list of (request, path, cost) and final link usage table.
    """
    usage = defaultdict(int)
    results = []
    # order requests by shortest baseline distance first (DP-style greedy
    # scheduling minimises total network cost - analogous to job scheduling
    # with weighted optimization)
    base = {}
    for s, t in requests:
        dist, _ = dijkstra(graph, s)
        base[(s, t)] = dist.get(t, INF)
    ordered = sorted(requests, key=lambda r: base[r])

    for s, t in ordered:
        # temporarily inflate congestion of overused links
        snapshot = dict(graph.congestion)
        for (u, v), used in usage.items():
            if used >= capacity_per_link:
                graph.congestion[(u, v)] = min(1.0, graph.congestion.get((u, v), 0) + 0.5)
        path, cost = hybrid_dijkstra(graph, s, t)
        graph.congestion = snapshot
        for i in range(len(path) - 1):
            usage[(path[i], path[i + 1])] += 1
        results.append(((s, t), path, cost))
    return results, usage


# ------------------------------------------------------------ Build network
def build_sample_network():
    g = NetworkGraph()
    links = [
        ("R1", "R2", 4, 0.30), ("R1", "R3", 2, 0.20), ("R2", "R3", 1, 0.85),
        ("R2", "R4", 5, 0.40), ("R3", "R4", 8, 0.55), ("R3", "R5", 10, 0.15),
        ("R4", "R5", 2, 0.90), ("R4", "R6", 6, 0.35), ("R5", "R6", 3, 0.25),
        ("R5", "R7", 7, 0.60), ("R6", "R7", 1, 0.45), ("R6", "R8", 4, 0.75),
        ("R7", "R8", 2, 0.20), ("R2", "R5", 9, 0.50), ("R3", "R6", 11, 0.10),
    ]
    for u, v, c, cong in links:
        g.add_link(u, v, c, cong)
    return g


def build_negative_weight_graph():
    """Small directed graph (no negative cycle) to demonstrate Bellman-Ford."""
    g = NetworkGraph()
    g.add_link("A", "B", 4, 0.2, bidirectional=False)
    g.add_link("A", "C", 5, 0.3, bidirectional=False)
    g.add_link("B", "C", -3, 0.1, bidirectional=False)   # variable/negative cost link (e.g. incentivised route)
    g.add_link("C", "D", 4, 0.2, bidirectional=False)
    g.add_link("B", "D", 6, 0.4, bidirectional=False)
    return g


# ----------------------------------------------------------------- Demo run
def run_demo():
    print("=" * 70)
    print("SMART NETWORK ROUTING AND CONGESTION OPTIMIZATION SYSTEM")
    print("=" * 70)

    g = build_sample_network()
    print(f"\nNetwork: {len(g.nodes)} routers, {len(list(g.edges())) // 2} bidirectional links")

    # 1. Dijkstra
    print("\n--- Dijkstra Shortest Path (Source: R1) ---")
    t0 = time.perf_counter()
    dist, prev = dijkstra(g, "R1")
    t_dij = time.perf_counter() - t0
    for target in ["R5", "R8"]:
        path = reconstruct_path(prev, "R1", target)
        print(f"R1 -> {target}: cost={dist[target]}, path={'->'.join(path)}")
    print(f"Execution time: {t_dij*1000:.4f} ms")

    # 2. Bellman-Ford
    print("\n--- Bellman-Ford (variable/negative cost links) ---")
    ng = build_negative_weight_graph()
    t0 = time.perf_counter()
    bf_dist, bf_prev, neg_cycle = bellman_ford(ng, "A")
    t_bf = time.perf_counter() - t0
    print(f"Negative cycle detected: {neg_cycle}")
    for target in ["C", "D"]:
        path = reconstruct_path(bf_prev, "A", target)
        print(f"A -> {target}: cost={bf_dist[target]}, path={'->'.join(path)}")
    print(f"Execution time: {t_bf*1000:.4f} ms")

    # 3. Floyd-Warshall
    print("\n--- Floyd-Warshall (All-Pairs Shortest Paths) ---")
    t0 = time.perf_counter()
    fdist, fnxt, fnodes, fidx = floyd_warshall(g)
    t_fw = time.perf_counter() - t0
    print("Sample distance matrix rows (R1, R4):")
    for u in ["R1", "R4"]:
        row = " ".join(f"{fdist[fidx[u]][fidx[v]]:>5.0f}" for v in fnodes)
        print(f"  {u}: {row}   (order: {fnodes})")
    print(f"Execution time: {t_fw*1000:.4f} ms")

    # 4. Congestion detection
    print("\n--- Congestion / Bottleneck Detection (threshold=0.7) ---")
    hot_links, bottlenecks = detect_congestion(g, 0.7)
    print(f"Hot links: {hot_links}")
    print(f"Bottleneck routers: {bottlenecks}")

    # 5. Greedy least-congested route
    print("\n--- Greedy Least-Congested Routing (R1 -> R8) ---")
    gpath = greedy_least_congested_path(g, "R1", "R8")
    print(f"Path: {'->'.join(gpath)}")

    # 6. Hybrid routing
    print("\n--- Hybrid Smart Routing (distance + congestion, R1 -> R8) ---")
    hpath, hcost = hybrid_dijkstra(g, "R1", "R8")
    print(f"Path: {'->'.join(hpath)}, hybrid cost={hcost:.2f}")

    # 7. DP-based optimization for multiple requests
    print("\n--- Multi-Request Optimization (DP-style load balancing) ---")
    requests = [("R1", "R8"), ("R2", "R7"), ("R1", "R6"), ("R3", "R8")]
    results, usage = optimize_multiple_requests(g, requests)
    for (s, t), path, cost in results:
        print(f"Request {s}->{t}: path={'->'.join(path)}, cost={cost:.2f}")
    print(f"Link usage after optimization: {dict(usage)}")

    # 8. Comparison table (correctness cross-check + timing over larger random graph)
    print("\n--- Algorithm Runtime Comparison (repeated 500x on sample network) ---")
    reps = 500
    t0 = time.perf_counter()
    for _ in range(reps):
        dijkstra(g, "R1")
    t_dij_avg = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        bellman_ford(g, "R1")
    t_bf_avg = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(50):
        floyd_warshall(g)
    t_fw_avg = (time.perf_counter() - t0) / 50
    print(f"Dijkstra avg: {t_dij_avg*1e6:.2f} us | Bellman-Ford avg: {t_bf_avg*1e6:.2f} us "
          f"| Floyd-Warshall avg: {t_fw_avg*1e6:.2f} us")

    print("\n" + "=" * 70)
    print("END OF DEMO RUN")
    print("=" * 70)

    return {
        "dijkstra_time_us": t_dij_avg * 1e6,
        "bf_time_us": t_bf_avg * 1e6,
        "fw_time_us": t_fw_avg * 1e6,
        "hybrid_path": hpath,
        "hybrid_cost": hcost,
        "greedy_path": gpath,
        "hot_links": hot_links,
        "bottlenecks": bottlenecks,
    }


if __name__ == "__main__":
    run_demo()
