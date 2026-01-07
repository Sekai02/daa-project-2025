"""
Binary search solution for power grid design problem.
Uses binary search on objective value combined with DFS feasibility check.
"""

from collections import deque
from functools import lru_cache
from typing import List, Tuple, Dict


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


def binary_search_solution(
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
    Solve power grid design using binary search on objective value.
    Returns selected edges and worst-case unmet demand.
    """
    m = len(edges)
    if len(kappa) != m or len(u) != m:
        raise ValueError("edges, kappa, and u must have the same length")
    if len(plants) != len(g):
        raise ValueError("plants and g must have the same length")
    if len(consumers) != len(d):
        raise ValueError("consumers and d must have the same length")

    total_demand = 0
    for val in d:
        total_demand += val

    if not plants or m == 0:
        return [0] * m, int(total_demand)

    all_ones = (1 << m) - 1
    suffix_mask = [0] * (m + 1)
    for i in range(m + 1):
        if i == m:
            suffix_mask[i] = 0
        else:
            suffix_mask[i] = all_ones ^ ((1 << i) - 1)

    worst_cache: Dict[int, int] = {}

    def worst_unmet(mask: int) -> int:
        """Compute worst-case unmet demand for given edge selection mask."""
        v = worst_cache.get(mask)
        if v is not None:
            return v
        worst = 0
        for failed in plants:
            din = Dinic(n + 2)
            S = n
            T = n + 1
            for p, capg in zip(plants, g):
                din.add_edge(S, p, 0 if p == failed else capg)
            for c, dem in zip(consumers, d):
                din.add_edge(c, T, dem)
            mm = mask
            i = 0
            while mm:
                if mm & 1:
                    a, b = edges[i]
                    din.add_edge(a, b, u[i])
                mm >>= 1
                i += 1
            flow = din.max_flow(S, T)
            unmet = total_demand - flow
            if unmet > worst:
                worst = unmet
        worst_cache[mask] = worst
        return worst

    def exists_feasible(Z: int) -> bool:
        """Check if objective Z is achievable within budget."""
        @lru_cache(maxsize=None)
        def dfs(i: int, cost: int, inc_mask: int) -> bool:
            """DFS with pruning to find feasible edge selection."""
            if cost > B:
                return False
            opt_mask = inc_mask | suffix_mask[i]
            if worst_unmet(opt_mask) > Z:
                return False
            if worst_unmet(inc_mask) <= Z:
                return True
            if i == m:
                return False
            if dfs(i + 1, cost, inc_mask):
                return True
            c2 = cost + kappa[i]
            if c2 <= B and dfs(i + 1, c2, inc_mask | (1 << i)):
                return True
            return False

        return dfs(0, 0, 0)

    lo = 0
    hi = total_demand
    while lo < hi:
        mid = (lo + hi) // 2
        if exists_feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    z_star = lo

    best_cost = 10 ** 18
    best_mask = 0

    def search_best(i: int, cost: int, inc_mask: int) -> None:
        """Search for optimal solution achieving z_star."""
        nonlocal best_cost, best_mask
        if cost > B or cost > best_cost:
            return
        opt_mask = inc_mask | suffix_mask[i]
        if worst_unmet(opt_mask) > z_star:
            return
        if worst_unmet(inc_mask) <= z_star:
            if cost < best_cost or (cost == best_cost and inc_mask < best_mask):
                best_cost = cost
                best_mask = inc_mask
            return
        if i == m:
            return
        search_best(i + 1, cost, inc_mask)
        c2 = cost + kappa[i]
        if c2 <= B:
            search_best(i + 1, c2, inc_mask | (1 << i))

    search_best(0, 0, 0)
    x = [1 if (best_mask >> i) & 1 else 0 for i in range(m)]
    return x, int(z_star)
