"""
Random instance generators for:
Robust Budget-Constrained Power Grid Design Under Single-Plant Failures.

Instance matches the solver signatures used in this project:
    (n, edges, plants, consumers, kappa, u, g, d, B)

- n: number of nodes (0..n-1)
- edges: List[(a,b)] directed candidate arcs
- plants: List[node_id]
- consumers: List[node_id]
- kappa: List[cost] per edge
- u: List[capacity] per edge
- g: List[g_p] generator capacity per plant (same order as plants)
- d: List[d_c] demand per consumer (same order as consumers)
- B: budget (nonnegative int)

Design goals:
- Simple configurable constraints via a dataclass.
- Sensible defaults for stress-testing.
- Optional "modes" to produce harder/structured instances.
- CLI to emit JSON or Python-literal for quick experimentation.

This generator does NOT solve the problem; it only creates inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import argparse
import json
import random
import sys


Edge = Tuple[int, int]


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for instance generation."""

    n: int = 20
    num_plants: int = 3
    num_consumers: int = 6

    m_edges: Optional[int] = 60
    edge_prob: Optional[float] = None
    allow_self_loops: bool = False

    cost_min: int = 1
    cost_max: int = 25
    cap_min: int = 1
    cap_max: int = 30

    demand_min: int = 5
    demand_max: int = 25
    plant_cap_min: int = 10
    plant_cap_max: int = 60

    zero_cost_edge_prob: float = 0.0

    budget: Optional[int] = None
    budget_ratio: float = 0.25
    budget_min: int = 0
    budget_max: Optional[int] = None

    ensure_each_consumer_has_incoming: bool = True
    ensure_each_plant_has_outgoing: bool = True
    ensure_min_connectivity: bool = True

    nudge_generation_for_robustness: bool = True
    robustness_slack: float = 1.10

    mode: str = "random"


@dataclass
class Instance:
    """Problem instance for power grid design."""

    n: int
    edges: List[Edge]
    plants: List[int]
    consumers: List[int]
    kappa: List[int]
    u: List[int]
    g: List[int]
    d: List[int]
    B: int
    meta: Dict[str, Any]

    def to_solver_args(self):
        """Convert instance to solver function arguments tuple."""
        return (self.n, self.edges, self.plants, self.consumers, self.kappa, self.u, self.g, self.d, self.B)


def _clamp(x: int, lo: int, hi: Optional[int]) -> int:
    """Clamp value x to range [lo, hi]."""
    if x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x


def _sample_distinct(rng: random.Random, n: int, k: int) -> List[int]:
    """Sample k distinct integers from range [0, n)."""
    if k < 0 or k > n:
        raise ValueError(f"cannot sample {k} distinct values from [0,{n})")
    arr = list(range(n))
    rng.shuffle(arr)
    return arr[:k]


def _make_roles(cfg: GeneratorConfig, rng: random.Random) -> Tuple[List[int], List[int], List[int]]:
    """Assign nodes to roles: plants, consumers, transit."""
    if cfg.n <= 0:
        raise ValueError("n must be > 0")
    if cfg.num_plants <= 0:
        raise ValueError("num_plants must be > 0 (single-plant failure scenarios are defined over plants)")
    if cfg.num_consumers <= 0:
        raise ValueError("num_consumers must be > 0")
    if cfg.num_plants + cfg.num_consumers > cfg.n:
        raise ValueError("num_plants + num_consumers must be <= n")

    picked = _sample_distinct(rng, cfg.n, cfg.num_plants + cfg.num_consumers)
    plants = picked[: cfg.num_plants]
    consumers = picked[cfg.num_plants :]
    is_special = [0] * cfg.n
    for p in plants:
        is_special[p] = 1
    for c in consumers:
        is_special[c] = 1
    transit = [i for i in range(cfg.n) if not is_special[i]]
    return plants, consumers, transit


def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    """Generate random integer in range [lo, hi]."""
    if lo > hi:
        lo, hi = hi, lo
    return rng.randint(lo, hi)


def _generate_demands(cfg: GeneratorConfig, rng: random.Random, consumers: List[int]) -> List[int]:
    """Generate demand values for consumers."""
    d = [_rand_int(rng, cfg.demand_min, cfg.demand_max) for _ in consumers]
    return d


def _generate_generation(cfg: GeneratorConfig, rng: random.Random, plants: List[int], demands: List[int]) -> List[int]:
    """Generate generation capacity values for plants."""
    g = [_rand_int(rng, cfg.plant_cap_min, cfg.plant_cap_max) for _ in plants]

    if not cfg.nudge_generation_for_robustness or len(plants) <= 1:
        return g

    total_demand = sum(demands)
    target = int(cfg.robustness_slack * total_demand)
    cur = sum(g) - max(g)
    if cur >= target:
        return g

    need = target - cur
    idx_sorted = sorted(range(len(g)), key=lambda i: g[i])
    g2 = g[:]
    i = 0
    while need > 0 and i < len(idx_sorted):
        pi = idx_sorted[i]
        room = cfg.plant_cap_max - g2[pi]
        if room > 0:
            add = min(room, need)
            g2[pi] += add
            need -= add
        i += 1
    return g2


def _all_possible_arcs(n: int, allow_self_loops: bool) -> List[Edge]:
    """Generate all possible directed arcs for n nodes."""
    arcs: List[Edge] = []
    for a in range(n):
        for b in range(n):
            if not allow_self_loops and a == b:
                continue
            arcs.append((a, b))
    return arcs


def _choose_edges_random(cfg: GeneratorConfig, rng: random.Random) -> List[Edge]:
    """Choose candidate edges based on configuration."""
    if cfg.m_edges is None and cfg.edge_prob is None:
        raise ValueError("Specify either m_edges or edge_prob.")
    if cfg.m_edges is not None and cfg.edge_prob is not None:
        raise ValueError("Specify only one of m_edges or edge_prob (not both).")

    candidates = _all_possible_arcs(cfg.n, cfg.allow_self_loops)

    if cfg.edge_prob is not None:
        p = cfg.edge_prob
        if p < 0.0 or p > 1.0:
            raise ValueError("edge_prob must be in [0,1]")
        edges = [e for e in candidates if rng.random() < p]
        if not edges and candidates:
            edges = [candidates[rng.randrange(len(candidates))]]
        return edges

    m = cfg.m_edges
    if m is None:
        raise ValueError("internal: m_edges unexpectedly None")
    if m < 0:
        raise ValueError("m_edges must be >= 0")
    m = min(m, len(candidates))
    rng.shuffle(candidates)
    edges = candidates[:m]
    return edges


def _ensure_backbone(cfg: GeneratorConfig, rng: random.Random, plants: List[int], consumers: List[int], transit: List[int], edges: List[Edge]) -> List[Edge]:
    """Add basic connectivity backbone to edge set."""
    edge_set = set(edges)

    def add(a: int, b: int):
        """Add edge to set if valid."""
        if not cfg.allow_self_loops and a == b:
            return
        edge_set.add((a, b))

    if not cfg.ensure_min_connectivity:
        return list(edge_set)

    hub: int
    if transit:
        hub = transit[0]
    else:
        hub = consumers[0]

    for p in plants:
        add(p, hub)

    for c in consumers:
        if hub != c:
            add(hub, c)

    if len(transit) >= 2:
        for i in range(len(transit) - 1):
            if rng.random() < 0.6:
                add(transit[i], transit[i + 1])

    return list(edge_set)


def _ensure_role_degrees(cfg: GeneratorConfig, rng: random.Random, plants: List[int], consumers: List[int], transit: List[int], edges: List[Edge]) -> List[Edge]:
    """Ensure each node has appropriate in/out degree based on role."""
    edge_set = set(edges)

    out_deg = [0] * cfg.n
    in_deg = [0] * cfg.n
    for a, b in edge_set:
        out_deg[a] += 1
        in_deg[b] += 1

    def add_edge(a: int, b: int):
        """Add edge to set and update degree counts."""
        if not cfg.allow_self_loops and a == b:
            return
        if (a, b) in edge_set:
            return
        edge_set.add((a, b))
        out_deg[a] += 1
        in_deg[b] += 1

    if cfg.ensure_each_consumer_has_incoming:
        for c in consumers:
            if in_deg[c] > 0:
                continue
            src_pool = transit if transit else plants
            a = src_pool[rng.randrange(len(src_pool))]
            add_edge(a, c)

    if cfg.ensure_each_plant_has_outgoing:
        for p in plants:
            if out_deg[p] > 0:
                continue
            dst_pool = transit if transit else consumers
            b = dst_pool[rng.randrange(len(dst_pool))]
            add_edge(p, b)

    return list(edge_set)


def _assign_edge_values(cfg: GeneratorConfig, rng: random.Random, edges: List[Edge], total_demand: int) -> Tuple[List[int], List[int]]:
    """Assign cost and capacity values to edges."""
    kappa: List[int] = []
    u: List[int] = []

    soft_cap_max = max(cfg.cap_max, max(1, total_demand // 2))

    for _ in edges:
        if cfg.zero_cost_edge_prob > 0.0 and rng.random() < cfg.zero_cost_edge_prob:
            k = 0
        else:
            k = _rand_int(rng, cfg.cost_min, cfg.cost_max)

        cap_hi = max(cfg.cap_min, min(soft_cap_max, cfg.cap_max))
        cap = _rand_int(rng, cfg.cap_min, cap_hi)

        kappa.append(k)
        u.append(cap)

    return kappa, u


def _derive_budget(cfg: GeneratorConfig, rng: random.Random, kappa: List[int]) -> int:
    """Derive budget value from configuration and edge costs."""
    if cfg.budget is not None:
        return _clamp(int(cfg.budget), cfg.budget_min, cfg.budget_max)

    total_cost = sum(kappa)
    base = int(cfg.budget_ratio * total_cost)
    jitter = int(0.08 * total_cost)
    if jitter > 0:
        base = base + rng.randint(-jitter, jitter)
    return _clamp(max(0, base), cfg.budget_min, cfg.budget_max)


def _mode_transform(cfg: GeneratorConfig, rng: random.Random, plants: List[int], consumers: List[int], transit: List[int], edges: List[Edge]) -> List[Edge]:
    """Apply mode-specific transformations to edge set."""
    mode = (cfg.mode or "random").lower()
    if mode == "random":
        return edges

    edge_set = set(edges)

    def add(a: int, b: int):
        """Add edge to set if valid."""
        if not cfg.allow_self_loops and a == b:
            return
        edge_set.add((a, b))

    if mode == "two_layer":
        edge_set = set()
        mid = transit[:] if transit else []
        if not mid:
            mid = consumers[:]

        for p in plants:
            for _ in range(max(1, len(mid) // 2)):
                add(p, mid[rng.randrange(len(mid))])

        for c in consumers:
            add(mid[rng.randrange(len(mid))], c)

        if len(mid) >= 2:
            for _ in range(min(10, len(mid) * 2)):
                a = mid[rng.randrange(len(mid))]
                b = mid[rng.randrange(len(mid))]
                if a != b:
                    add(a, b)

    elif mode == "bottleneck":
        if transit:
            bn = [transit[rng.randrange(len(transit))]]
            if len(transit) >= 2 and rng.random() < 0.6:
                bn2 = transit[rng.randrange(len(transit))]
                if bn2 != bn[0]:
                    bn.append(bn2)
        else:
            bn = [consumers[rng.randrange(len(consumers))]]

        edge_set = set()

        for p in plants:
            for b in bn:
                add(p, b)

        for c in consumers:
            add(bn[rng.randrange(len(bn))], c)

        for _ in range(max(1, cfg.n // 3)):
            a = rng.randrange(cfg.n)
            b = rng.randrange(cfg.n)
            if a != b:
                add(a, b)

    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}. Use random|two_layer|bottleneck.")

    return list(edge_set)


def generate_instance(cfg: GeneratorConfig, seed: Optional[int] = None) -> Instance:
    """Generate a random problem instance based on configuration."""
    rng = random.Random(seed)

    plants, consumers, transit = _make_roles(cfg, rng)
    demands = _generate_demands(cfg, rng, consumers)
    gen_caps = _generate_generation(cfg, rng, plants, demands)

    edges = _choose_edges_random(cfg, rng)

    edges = _mode_transform(cfg, rng, plants, consumers, transit, edges)

    edges = _ensure_backbone(cfg, rng, plants, consumers, transit, edges)
    edges = _ensure_role_degrees(cfg, rng, plants, consumers, transit, edges)

    edges = sorted(set(edges))

    total_demand = sum(demands)
    kappa, caps = _assign_edge_values(cfg, rng, edges, total_demand)
    B = _derive_budget(cfg, rng, kappa)

    meta = {
        "cfg": asdict(cfg),
        "seed": seed,
        "total_demand": total_demand,
        "total_generation": sum(gen_caps),
        "num_transit": len(transit),
        "num_edges": len(edges),
        "notes": "Generated for robust single-plant-failure unmet-demand objective.",
    }

    return Instance(
        n=cfg.n,
        edges=edges,
        plants=plants,
        consumers=consumers,
        kappa=kappa,
        u=caps,
        g=gen_caps,
        d=demands,
        B=B,
        meta=meta,
    )


def _instance_to_jsonable(inst: Instance) -> Dict[str, Any]:
    """Convert instance to JSON-serializable dictionary."""
    return {
        "n": inst.n,
        "edges": inst.edges,
        "plants": inst.plants,
        "consumers": inst.consumers,
        "kappa": inst.kappa,
        "u": inst.u,
        "g": inst.g,
        "d": inst.d,
        "B": inst.B,
        "meta": inst.meta,
    }


def _print_python_literal(inst: Instance) -> None:
    """Print instance as Python literals for REPL usage."""
    print(f"n = {inst.n}")
    print(f"edges = {inst.edges}")
    print(f"plants = {inst.plants}")
    print(f"consumers = {inst.consumers}")
    print(f"kappa = {inst.kappa}")
    print(f"u = {inst.u}")
    print(f"g = {inst.g}")
    print(f"d = {inst.d}")
    print(f"B = {inst.B}")
    print()
    print("args = (n, edges, plants, consumers, kappa, u, g, d, B)")
    print(f"meta = {json.dumps(inst.meta, indent=2)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    p = argparse.ArgumentParser(description="Generate random instances for the robust budget-constrained power grid design problem.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (int).")
    p.add_argument("--n", type=int, default=20, help="Number of nodes.")
    p.add_argument("--plants", type=int, default=3, help="Number of plants.")
    p.add_argument("--consumers", type=int, default=6, help="Number of consumers.")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--m", type=int, default=60, help="Number of candidate directed edges.")
    group.add_argument("--p_edge", type=float, default=None, help="Edge probability over ordered pairs (Erdos-Renyi).")

    p.add_argument("--mode", type=str, default="random", choices=["random", "two_layer", "bottleneck"], help="Generator mode.")
    p.add_argument("--budget", type=int, default=None, help="Absolute budget. If omitted, derived from budget_ratio.")
    p.add_argument("--budget_ratio", type=float, default=0.25, help="Budget ratio of total candidate cost if --budget not provided.")

    p.add_argument("--cost_min", type=int, default=1)
    p.add_argument("--cost_max", type=int, default=25)
    p.add_argument("--cap_min", type=int, default=1)
    p.add_argument("--cap_max", type=int, default=30)

    p.add_argument("--demand_min", type=int, default=5)
    p.add_argument("--demand_max", type=int, default=25)
    p.add_argument("--plant_cap_min", type=int, default=10)
    p.add_argument("--plant_cap_max", type=int, default=60)

    p.add_argument("--zero_cost_prob", type=float, default=0.0, help="Probability an edge has zero construction cost.")
    p.add_argument("--no_nudge_gen", action="store_true", help="Disable robustness nudging for generation capacities.")
    p.add_argument("--robust_slack", type=float, default=1.10, help="Target slack for (sum(g)-max(g)) vs total demand.")

    p.add_argument("--out", type=str, default="json", choices=["json", "py"], help="Output format.")
    return p


def main(argv: List[str]) -> int:
    """Main entry point for CLI."""
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    cfg = GeneratorConfig(
        n=args.n,
        num_plants=args.plants,
        num_consumers=args.consumers,
        m_edges=args.m if args.p_edge is None else None,
        edge_prob=args.p_edge,
        cost_min=args.cost_min,
        cost_max=args.cost_max,
        cap_min=args.cap_min,
        cap_max=args.cap_max,
        demand_min=args.demand_min,
        demand_max=args.demand_max,
        plant_cap_min=args.plant_cap_min,
        plant_cap_max=args.plant_cap_max,
        zero_cost_edge_prob=args.zero_cost_prob,
        budget=args.budget,
        budget_ratio=args.budget_ratio,
        nudge_generation_for_robustness=(not args.no_nudge_gen),
        robustness_slack=args.robust_slack,
        mode=args.mode,
    )

    inst = generate_instance(cfg, seed=args.seed)

    if args.out == "json":
        print(json.dumps(_instance_to_jsonable(inst), indent=2))
        return 0

    _print_python_literal(inst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
