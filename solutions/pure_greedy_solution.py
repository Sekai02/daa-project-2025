from collections import deque
from typing import List, Tuple, Dict


class Dinic:
    __slots__ = ("n", "g", "lvl", "it")

    def __init__(self, n: int):
        self.n = n
        self.g = [[] for _ in range(n)]
        self.lvl = [-1] * n
        self.it = [0] * n

    def add_edge(self, fr: int, to: int, cap: int) -> None:
        fwd = [to, cap, None]
        rev = [fr, 0, fwd]
        fwd[2] = rev
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def _bfs(self, s: int, t: int) -> bool:
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


def pure_greedy_solution(
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
            mm = mask
            while mm:
                lb = mm & -mm
                i = lb.bit_length() - 1
                a, b = edges[i]
                din.add_edge(a, b, u[i])
                mm ^= lb
            flow = din.max_flow(S, T)
            unmet = total_demand - flow
            if unmet > worst:
                worst = unmet
        cache[mask] = worst
        return worst

    mask = 0
    cost = 0
    cur = worst_unmet(mask)

    while True:
        best_i = -1
        best_w = cur

        for i in range(m):
            if (mask >> i) & 1:
                continue
            if cost + kappa[i] > B:
                continue
            nm = mask | (1 << i)
            w = worst_unmet(nm)
            if w < best_w:
                best_w = w
                best_i = i
            elif w == best_w and w < cur:
                if best_i == -1 or kappa[i] < kappa[best_i] or (kappa[i] == kappa[best_i] and i < best_i):
                    best_i = i

        if best_i == -1:
            break

        mask |= 1 << best_i
        cost += kappa[best_i]
        cur = best_w
        if cur == 0:
            break

    x = [1 if (mask >> i) & 1 else 0 for i in range(m)]
    return x, int(cur)


solve = pure_greedy_solution
