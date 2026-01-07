"""
Tabu search metaheuristic solution for power grid design problem.
Uses tabu list to avoid cycling and explores neighborhood moves.
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


def tabu_search_solution(
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
    Solve power grid design using tabu search metaheuristic.
    Returns selected edges and worst-case unmet demand.
    """
    m = _validate_inputs(n, edges, plants, consumers, kappa, u, g, d, B)
    total_demand, worst_unmet = _worst_unmet_factory(n, edges, plants, consumers, u, g, d)

    if not plants or m == 0:
        return [0] * m, int(total_demand)

    rng = random.Random(2)

    def better(z1, c1, mask1, z2, c2, mask2):
        """Compare two solutions lexicographically."""
        if z1 != z2:
            return z1 < z2
        if c1 != c2:
            return c1 < c2
        return mask1 < mask2

    mask = 0
    cost = 0
    cur_z = worst_unmet(mask)

    used = [0] * m
    for i in range(m):
        if cost + kappa[i] <= B:
            nm = mask | (1 << i)
            nz = worst_unmet(nm)
            if nz < cur_z:
                mask = nm
                cost += kappa[i]
                cur_z = nz
                used[i] = 1

    best_mask, best_cost, best_z = mask, cost, cur_z

    tabu_until = {}
    iters = min(3000, 200 + 30 * m)
    tenure = 7 if m <= 120 else 10

    def is_tabu(i, t):
        """Check if edge i is tabu at iteration t."""
        v = tabu_until.get(i)
        return v is not None and v > t

    def apply_move(add_i, rem_j, t):
        """Apply move by adding/removing edges and update tabu list."""
        nonlocal mask, cost, cur_z
        if rem_j != -1:
            mask &= ~(1 << rem_j)
            cost -= kappa[rem_j]
        if add_i != -1:
            mask |= 1 << add_i
            cost += kappa[add_i]
        cur_z = worst_unmet(mask)
        if add_i != -1:
            tabu_until[add_i] = t + tenure
        if rem_j != -1:
            tabu_until[rem_j] = t + tenure

    for t in range(iters):
        in_set = []
        out_set = []
        mm = mask
        sel = [0] * m
        while mm:
            lb = mm & -mm
            i = lb.bit_length() - 1
            sel[i] = 1
            in_set.append(i)
            mm ^= lb
        for i in range(m):
            if not sel[i]:
                out_set.append(i)

        cand = []
        rem_budget = B - cost
        for i in out_set:
            if kappa[i] <= rem_budget:
                nm = mask | (1 << i)
                nz = worst_unmet(nm)
                cand.append((nz, cost + kappa[i], i, -1))

        if in_set:
            for j in in_set:
                nm = mask & ~(1 << j)
                nz = worst_unmet(nm)
                cand.append((nz, cost - kappa[j], -1, j))

        if in_set and out_set:
            trials = 0
            if m <= 90:
                for j in in_set:
                    base_cost = cost - kappa[j]
                    rem = B - base_cost
                    for i in out_set:
                        if kappa[i] <= rem:
                            nm = (mask & ~(1 << j)) | (1 << i)
                            nz = worst_unmet(nm)
                            cand.append((nz, base_cost + kappa[i], i, j))
            else:
                trials = min(6000, len(in_set) * len(out_set))
                for _ in range(trials):
                    j = in_set[rng.randrange(len(in_set))]
                    i = out_set[rng.randrange(len(out_set))]
                    base_cost = cost - kappa[j]
                    if base_cost + kappa[i] > B:
                        continue
                    nm = (mask & ~(1 << j)) | (1 << i)
                    nz = worst_unmet(nm)
                    cand.append((nz, base_cost + kappa[i], i, j))

        if not cand:
            break

        cand.sort(key=lambda x: (x[0], x[1], ((mask ^ ((0 if x[2] == -1 else (1 << x[2])) | (0 if x[3] == -1 else (1 << x[3])))))))
        chosen = None

        for nz, nc, add_i, rem_j in cand[: min(len(cand), 4000)]:
            move_tabu = False
            if add_i != -1 and is_tabu(add_i, t):
                move_tabu = True
            if rem_j != -1 and is_tabu(rem_j, t):
                move_tabu = True
            nm = mask
            if rem_j != -1:
                nm &= ~(1 << rem_j)
            if add_i != -1:
                nm |= 1 << add_i
            if move_tabu:
                if better(nz, nc, nm, best_z, best_cost, best_mask):
                    chosen = (add_i, rem_j, nz, nc, nm)
                    break
                continue
            chosen = (add_i, rem_j, nz, nc, nm)
            break

        if chosen is None:
            chosen = None
            for nz, nc, add_i, rem_j in cand[: min(len(cand), 4000)]:
                nm = mask
                if rem_j != -1:
                    nm &= ~(1 << rem_j)
                if add_i != -1:
                    nm |= 1 << add_i
                chosen = (add_i, rem_j, nz, nc, nm)
                break
            if chosen is None:
                break

        add_i, rem_j, nz, nc, nm = chosen
        apply_move(add_i, rem_j, t)

        if better(cur_z, cost, mask, best_z, best_cost, best_mask):
            best_z, best_cost, best_mask = cur_z, cost, mask

    x = [1 if (best_mask >> i) & 1 else 0 for i in range(m)]
    return x, int(best_z)


solve = tabu_search_solution
