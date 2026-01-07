"""
Simulated annealing metaheuristic solution for power grid design problem.
Uses probabilistic acceptance of worse solutions to escape local optima.
"""

import random
import math
from typing import List, Tuple
from collections import deque


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

from typing import List, Tuple, Dict

def _validate_inputs(n, edges, plants, consumers, kappa, u, g, d, B):
    m = len(edges)
    if len(kappa) != m or len(u) != m:
        raise ValueError("edges, kappa, and u must have the same length")
    if len(plants) != len(g):
        raise ValueError("plants and g must have the same length")
    if len(consumers) != len(d):
        raise ValueError("consumers and d must have the same length")
    return m

def _sum_list(a):
    s = 0
    for x in a:
        s += x
    return s

def _bit_iter(mm: int):
    while mm:
        lb = mm & -mm
        yield lb.bit_length() - 1
        mm ^= lb

def _worst_unmet_factory(n: int,
                         edges: List[Tuple[int, int]],
                         plants: List[int],
                         consumers: List[int],
                         u: List[int],
                         g: List[int],
                         d: List[int]):
    total_demand = _sum_list(d)
    cache: Dict[int, int] = {}

    def worst_unmet(mask: int) -> int:
        v = cache.get(mask)
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
            for i in _bit_iter(mask):
                a, b = edges[i]
                din.add_edge(a, b, u[i])
            flow = din.max_flow(S, T)
            unmet = total_demand - flow
            if unmet > worst:
                worst = unmet
        cache[mask] = worst
        return worst

    return total_demand, worst_unmet


def simulated_annealing_solution(
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
    Solve power grid design using simulated annealing metaheuristic.
    Returns selected edges and worst-case unmet demand.
    """
    m = _validate_inputs(n, edges, plants, consumers, kappa, u, g, d, B)
    total_demand, worst_unmet = _worst_unmet_factory(n, edges, plants, consumers, u, g, d)

    if not plants or m == 0:
        return [0] * m, int(total_demand)

    rng = random.Random(3)

    mask = 0
    cost = 0
    cur_z = worst_unmet(mask)

    for i in range(m):
        if cost + kappa[i] <= B:
            nm = mask | (1 << i)
            nz = worst_unmet(nm)
            if nz < cur_z:
                mask = nm
                cost += kappa[i]
                cur_z = nz

    best_mask, best_cost, best_z = mask, cost, cur_z

    big = total_demand + 1
    cur_E = cur_z * big + cost
    best_E = best_z * big + best_cost

    steps = min(20000, 2000 + 80 * m)
    T = float(max(1, cur_E // 10 + 1))
    alpha = 0.995 if steps <= 12000 else 0.997

    def random_neighbor(mask: int, cost: int):
        if m == 0:
            return mask, cost
        sel = []
        nsel = []
        mm = mask
        in_set = [0] * m
        while mm:
            lb = mm & -mm
            i = lb.bit_length() - 1
            in_set[i] = 1
            sel.append(i)
            mm ^= lb
        for i in range(m):
            if not in_set[i]:
                nsel.append(i)

        if not sel:
            i = nsel[rng.randrange(len(nsel))]
            if cost + kappa[i] <= B:
                return mask | (1 << i), cost + kappa[i]
            return mask, cost

        r = rng.random()
        if r < 0.25:
            j = sel[rng.randrange(len(sel))]
            return mask & ~(1 << j), cost - kappa[j]

        if r < 0.55 and nsel:
            i = nsel[rng.randrange(len(nsel))]
            if cost + kappa[i] <= B:
                return mask | (1 << i), cost + kappa[i]

        if nsel:
            j = sel[rng.randrange(len(sel))]
            i = nsel[rng.randrange(len(nsel))]
            nc = cost - kappa[j] + kappa[i]
            if nc <= B:
                nm = (mask & ~(1 << j)) | (1 << i)
                return nm, nc

        j = sel[rng.randrange(len(sel))]
        return mask & ~(1 << j), cost - kappa[j]

    for _ in range(steps):
        nm, nc = random_neighbor(mask, cost)
        nz = worst_unmet(nm)
        nE = nz * big + nc
        delta = nE - cur_E
        accept = False
        if delta <= 0:
            accept = True
        else:
            if T > 1e-12:
                p = math.exp(-float(delta) / T) if delta < 10**7 else 0.0
                if rng.random() < p:
                    accept = True

        if accept:
            mask, cost, cur_z, cur_E = nm, nc, nz, nE
            if cur_E < best_E or (cur_E == best_E and mask < best_mask):
                best_mask, best_cost, best_z, best_E = mask, cost, cur_z, cur_E

        T *= alpha
        if T < 1e-9:
            T = 1e-9

    x = [1 if (best_mask >> i) & 1 else 0 for i in range(m)]
    return x, int(best_z)


solve = simulated_annealing_solution
