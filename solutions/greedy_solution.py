"""
Greedy solution with cost-effectiveness heuristic for power grid design problem.
Selects edges based on improvement-to-cost ratio.
"""

from collections import deque
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


def greedy_solution(
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
    Solve power grid design using greedy cost-effectiveness heuristic.
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

    worst_cache: Dict[int, int] = {}

    def worst_unmet(mask: int) -> int:
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
            while mm:
                lb = mm & -mm
                i = (lb.bit_length() - 1)
                a, b = edges[i]
                din.add_edge(a, b, u[i])
                mm ^= lb
            flow = din.max_flow(S, T)
            unmet = total_demand - flow
            if unmet > worst:
                worst = unmet
        worst_cache[mask] = worst
        return worst

    mask = 0
    cost = 0
    cur = worst_unmet(mask)

    while True:
        best_i = -1
        best_w = cur
        best_impr = 0
        best_k = 1

        for i in range(m):
            if (mask >> i) & 1:
                continue
            ki = kappa[i]
            if cost + ki > B:
                continue
            cand_mask = mask | (1 << i)
            w = worst_unmet(cand_mask)
            impr = cur - w
            if impr <= 0:
                continue

            if best_i == -1:
                best_i, best_w, best_impr, best_k = i, w, impr, ki
                continue

            better = False
            if w < best_w:
                better = True
            elif w == best_w:
                if best_k == 0 and ki == 0:
                    if impr > best_impr or (impr == best_impr and i < best_i):
                        better = True
                elif ki == 0 and best_k != 0:
                    better = True
                elif ki != 0 and best_k == 0:
                    better = False
                else:
                    left = impr * best_k
                    right = best_impr * ki
                    if left > right:
                        better = True
                    elif left == right:
                        if impr > best_impr:
                            better = True
                        elif impr == best_impr:
                            if ki < best_k:
                                better = True
                            elif ki == best_k and i < best_i:
                                better = True

            if better:
                best_i, best_w, best_impr, best_k = i, w, impr, ki

        if best_i == -1:
            break
        mask |= 1 << best_i
        cost += kappa[best_i]
        cur = best_w
        if cur == 0:
            break

    if mask != 0:
        chosen = []
        mm = mask
        while mm:
            lb = mm & -mm
            chosen.append(lb.bit_length() - 1)
            mm ^= lb
        chosen.sort(key=lambda i: (-kappa[i], i))
        for i in chosen:
            cand_mask = mask & ~(1 << i)
            if worst_unmet(cand_mask) <= cur:
                mask = cand_mask
                cost -= kappa[i]

    x = [1 if (mask >> i) & 1 else 0 for i in range(m)]
    return x, int(worst_unmet(mask))
