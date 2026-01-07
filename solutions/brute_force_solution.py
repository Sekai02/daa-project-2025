"""
Brute force solution for power grid design problem.
Exhaustively evaluates all edge subsets within budget.
"""

from collections import deque
from typing import List, Tuple


class Dinic:
    """Dinic's maximum flow algorithm implementation."""
    __slots__ = ("n", "g", "lvl", "it")

    def __init__(self, n: int):
        """Initialize flow network with n nodes."""
        self.n = n
        self.g = [[] for _ in range(n)]
        self.lvl = [-1] * n
        self.it = [0] * n

    def add_edge(self, fr: int, to: int, cap: int) -> None:
        """Add directed edge with capacity."""
        fwd = [to, cap, None]
        rev = [fr, 0, fwd]
        fwd[2] = rev
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def _bfs(self, s: int, t: int) -> bool:
        """Build level graph using BFS."""
        for i in range(self.n):
            self.lvl[i] = -1
        q = deque([s])
        self.lvl[s] = 0
        while q:
            v = q.popleft()
            for to, cap, _ in self.g[v]:
                if cap > 0 and self.lvl[to] < 0:
                    self.lvl[to] = self.lvl[v] + 1
                    q.append(to)
        return self.lvl[t] >= 0

    def _dfs(self, v: int, t: int, f: int) -> int:
        """Find blocking flow using DFS."""
        if v == t:
            return f
        gv = self.g[v]
        i = self.it[v]
        while i < len(gv):
            e = gv[i]
            to = e[0]
            cap = e[1]
            if cap > 0 and self.lvl[to] == self.lvl[v] + 1:
                pushed = self._dfs(to, t, f if f < cap else cap)
                if pushed:
                    e[1] -= pushed
                    e[2][1] += pushed
                    return pushed
            i += 1
            self.it[v] = i
        return 0

    def max_flow(self, s: int, t: int) -> int:
        """Compute maximum flow from s to t."""
        flow = 0
        inf = 10 ** 18
        while self._bfs(s, t):
            for i in range(self.n):
                self.it[i] = 0
            while True:
                pushed = self._dfs(s, t, inf)
                if not pushed:
                    break
                flow += pushed
        return flow


def brute_force_solution(
    n: int,
    edges: List[Tuple[int, int]],
    plants: List[int],
    consumers: List[int],
    kappa: List[int],
    u: List[int],
    g: List[int],
    d: List[int],
    B: int,
) -> Tuple[List[int], int]:
    """
    Solve power grid design by exhaustive search over all edge subsets.
    Returns selected edges and worst-case unmet demand.
    """
    m = len(edges)
    if len(kappa) != m or len(u) != m:
        raise ValueError("edges, kappa, and u must have the same length")
    if len(plants) != len(g):
        raise ValueError("plants and g must have the same length")
    if len(consumers) != len(d):
        raise ValueError("consumers and d must have the same length")

    if not plants:
        return [0] * m, 0

    total_demand = 0
    for val in d:
        total_demand += val

    best_z = 10 ** 18
    best_cost = 10 ** 18
    best_mask = 0

    for mask in range(1 << m):
        cost = 0
        for i in range(m):
            if (mask >> i) & 1:
                cost += kappa[i]
                if cost > B:
                    break
        if cost > B:
            continue

        worst = 0
        for failed in plants:
            if worst >= best_z:
                break
            din = Dinic(n + 2)
            S = n
            T = n + 1
            for p, capg in zip(plants, g):
                din.add_edge(S, p, 0 if p == failed else capg)
            for c, dem in zip(consumers, d):
                din.add_edge(c, T, dem)
            for i, (a, b) in enumerate(edges):
                if (mask >> i) & 1:
                    din.add_edge(a, b, u[i])
            flow = din.max_flow(S, T)
            unmet = total_demand - flow
            if unmet > worst:
                worst = unmet

        if worst < best_z or (worst == best_z and (cost < best_cost or (cost == best_cost and mask < best_mask))):
            best_z = worst
            best_cost = cost
            best_mask = mask

    x = [1 if (best_mask >> i) & 1 else 0 for i in range(m)]
    return x, int(best_z)
