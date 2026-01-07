"""
GRASP with swap-based local search solution for power grid design problem.
Combines greedy randomized construction with local search improvement.
"""

import random
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
    """Validate input parameters and return number of edges."""
    m = len(edges)
    if len(kappa) != m or len(u) != m:
        raise ValueError("edges, kappa, and u must have the same length")
    if len(plants) != len(g):
        raise ValueError("plants and g must have the same length")
    if len(consumers) != len(d):
        raise ValueError("consumers and d must have the same length")
    return m

def _sum_list(a):
    """Sum elements in list."""
    s = 0
    for x in a:
        s += x
    return s

def _bit_iter(mm: int):
    """Iterator over set bit positions in integer."""
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
    """Create worst-case unmet demand computation function with caching."""
    total_demand = _sum_list(d)
    cache: Dict[int, int] = {}

    def worst_unmet(mask: int) -> int:
        """Compute worst-case unmet demand for given edge selection mask."""
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


def _local_search_swap(m: int, kappa: List[int], B: int, worst_unmet, mask: int, cost: int, z: int, rng: random.Random) -> Tuple[int, int, int]:
    """Apply local search with swap moves to improve solution."""
    if m == 0:
        return mask, cost, z

    in_set = [0] * m
    mm = mask
    while mm:
        lb = mm & -mm
        i = lb.bit_length() - 1
        in_set[i] = 1
        mm ^= lb

    def better(z1, c1, mask1, z2, c2, mask2):
        if z1 != z2:
            return z1 < z2
        if c1 != c2:
            return c1 < c2
        return mask1 < mask2

    improved = True
    while improved:
        improved = False
        best_mask = mask
        best_cost = cost
        best_z = z

        rem_budget = B - cost
        for i in range(m):
            if in_set[i]:
                continue
            if kappa[i] <= rem_budget:
                nm = mask | (1 << i)
                nz = worst_unmet(nm)
                nc = cost + kappa[i]
                if better(nz, nc, nm, best_z, best_cost, best_mask):
                    best_mask, best_cost, best_z = nm, nc, nz

        for j in range(m):
            if not in_set[j]:
                continue
            nm = mask & ~(1 << j)
            nz = worst_unmet(nm)
            nc = cost - kappa[j]
            if better(nz, nc, nm, best_z, best_cost, best_mask):
                best_mask, best_cost, best_z = nm, nc, nz

        if m <= 80:
            for j in range(m):
                if not in_set[j]:
                    continue
                base_cost = cost - kappa[j]
                rem = B - base_cost
                for i in range(m):
                    if in_set[i]:
                        continue
                    if kappa[i] <= rem:
                        nm = (mask & ~(1 << j)) | (1 << i)
                        nz = worst_unmet(nm)
                        nc = base_cost + kappa[i]
                        if better(nz, nc, nm, best_z, best_cost, best_mask):
                            best_mask, best_cost, best_z = nm, nc, nz
        else:
            chosen = [i for i in range(m) if in_set[i]]
            not_chosen = [i for i in range(m) if not in_set[i]]
            if chosen and not_chosen:
                trials = min(4000, len(chosen) * len(not_chosen))
                for _ in range(trials):
                    j = chosen[rng.randrange(len(chosen))]
                    i = not_chosen[rng.randrange(len(not_chosen))]
                    base_cost = cost - kappa[j]
                    if base_cost + kappa[i] > B:
                        continue
                    nm = (mask & ~(1 << j)) | (1 << i)
                    nz = worst_unmet(nm)
                    nc = base_cost + kappa[i]
                    if better(nz, nc, nm, best_z, best_cost, best_mask):
                        best_mask, best_cost, best_z = nm, nc, nz

        if best_mask != mask:
            mask, cost, z = best_mask, best_cost, best_z
            in_set = [0] * m
            mm = mask
            while mm:
                lb = mm & -mm
                i = lb.bit_length() - 1
                in_set[i] = 1
                mm ^= lb
            improved = True

    return mask, cost, z


def grasp_solution(
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
    Solve power grid design using GRASP with local search.
    Returns selected edges and worst-case unmet demand.
    """
    m = _validate_inputs(n, edges, plants, consumers, kappa, u, g, d, B)
    total_demand, worst_unmet = _worst_unmet_factory(n, edges, plants, consumers, u, g, d)

    if not plants or m == 0:
        return [0] * m, int(total_demand)

    rng = random.Random(1)

    def better(z1, c1, mask1, z2, c2, mask2):
        if z1 != z2:
            return z1 < z2
        if c1 != c2:
            return c1 < c2
        return mask1 < mask2

    iters = 30
    if m <= 80:
        iters = min(200, 3 * m + 20)
    else:
        iters = min(120, m + 20)

    alpha = 0.3

    best_mask = 0
    best_cost = 0
    best_z = worst_unmet(0)

    for _ in range(iters):
        mask = 0
        cost = 0
        cur = worst_unmet(mask)

        while True:
            rem = B - cost
            cand = []
            for i in range(m):
                if (mask >> i) & 1:
                    continue
                if kappa[i] > rem:
                    continue
                nm = mask | (1 << i)
                nz = worst_unmet(nm)
                impr = cur - nz
                if impr <= 0:
                    continue
                if kappa[i] == 0:
                    score_num = impr
                    score_den = 1
                else:
                    score_num = impr
                    score_den = kappa[i]
                cand.append((nz, -score_num / score_den, i))

            if not cand:
                break

            cand.sort()
            rcl_sz = int(len(cand) * alpha)
            if rcl_sz < 1:
                rcl_sz = 1
            pick = cand[rng.randrange(rcl_sz)][2]

            mask |= 1 << pick
            cost += kappa[pick]
            cur = worst_unmet(mask)
            if cur == 0:
                break

        z0 = worst_unmet(mask)
        mask, cost, z0 = _local_search_swap(m, kappa, B, worst_unmet, mask, cost, z0, rng)

        if better(z0, cost, mask, best_z, best_cost, best_mask):
            best_z, best_cost, best_mask = z0, cost, mask

    x = [1 if (best_mask >> i) & 1 else 0 for i in range(m)]
    return x, int(best_z)


solve = grasp_solution
