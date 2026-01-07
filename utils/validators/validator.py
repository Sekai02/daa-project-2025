"""
Instance validator for robust budget-constrained power grid design problem.

Validates that generated instances meet basic constraints before stress testing.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from collections import deque
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../generators')))
from generator import Instance


@dataclass
class ValidationError:
    """Single validation error or warning."""

    severity: str  # "ERROR" or "WARNING"
    message: str
    field: Optional[str] = None

    def __str__(self) -> str:
        if self.field:
            return f"[{self.severity}] {self.field}: {self.message}"
        return f"[{self.severity}] {self.message}"


@dataclass
class ValidationResult:
    """Result of instance validation."""

    is_valid: bool
    errors: List[ValidationError]

    @property
    def warnings(self) -> List[ValidationError]:
        return [e for e in self.errors if e.severity == "WARNING"]

    @property
    def critical_errors(self) -> List[ValidationError]:
        return [e for e in self.errors if e.severity == "ERROR"]

    def __str__(self) -> str:
        if self.is_valid:
            if self.warnings:
                return f"Valid (with {len(self.warnings)} warnings)"
            return "Valid"
        return f"Invalid ({len(self.critical_errors)} errors, {len(self.warnings)} warnings)"


def validate_instance(instance: Instance) -> ValidationResult:
    """
    Validate a power grid instance for correctness.

    Checks:
    1. Basic constraints (list lengths, non-negative values, valid ranges)
    2. Graph connectivity (at least one plant can reach at least one consumer)
    3. Budget feasibility (at least one edge is affordable)

    Does NOT check:
    - Generation capacity vs demand (per user requirements)

    Args:
        instance: The Instance object to validate

    Returns:
        ValidationResult with is_valid flag and list of errors/warnings
    """
    errors: List[ValidationError] = []

    errors.extend(_validate_basic_constraints(instance))
    errors.extend(_validate_graph_connectivity(instance))
    errors.extend(_validate_budget_feasibility(instance))

    critical_errors = [e for e in errors if e.severity == "ERROR"]
    is_valid = len(critical_errors) == 0

    return ValidationResult(is_valid=is_valid, errors=errors)


def _validate_basic_constraints(instance: Instance) -> List[ValidationError]:
    """Validate basic constraints on instance parameters."""
    errors: List[ValidationError] = []

    # Check n > 0
    if instance.n <= 0:
        errors.append(ValidationError("ERROR", "must be positive", "n"))

    # Check list lengths
    m = len(instance.edges)
    if len(instance.kappa) != m:
        errors.append(ValidationError(
            "ERROR",
            f"length {len(instance.kappa)} does not match number of edges {m}",
            "kappa"
        ))
    if len(instance.u) != m:
        errors.append(ValidationError(
            "ERROR",
            f"length {len(instance.u)} does not match number of edges {m}",
            "u"
        ))

    num_plants = len(instance.plants)
    if num_plants <= 0:
        errors.append(ValidationError("ERROR", "must have at least one plant", "plants"))
    if len(instance.g) != num_plants:
        errors.append(ValidationError(
            "ERROR",
            f"length {len(instance.g)} does not match number of plants {num_plants}",
            "g"
        ))

    num_consumers = len(instance.consumers)
    if num_consumers <= 0:
        errors.append(ValidationError("ERROR", "must have at least one consumer", "consumers"))
    if len(instance.d) != num_consumers:
        errors.append(ValidationError(
            "ERROR",
            f"length {len(instance.d)} does not match number of consumers {num_consumers}",
            "d"
        ))

    # Check non-negativity
    if instance.B < 0:
        errors.append(ValidationError("ERROR", "budget must be non-negative", "B"))

    for i, cost in enumerate(instance.kappa):
        if cost < 0:
            errors.append(ValidationError("ERROR", f"cost at index {i} is negative: {cost}", "kappa"))

    for i, cap in enumerate(instance.u):
        if cap < 0:
            errors.append(ValidationError("ERROR", f"capacity at index {i} is negative: {cap}", "u"))

    for i, gen_cap in enumerate(instance.g):
        if gen_cap < 0:
            errors.append(ValidationError("ERROR", f"generation capacity at index {i} is negative: {gen_cap}", "g"))

    for i, demand in enumerate(instance.d):
        if demand < 0:
            errors.append(ValidationError("ERROR", f"demand at index {i} is negative: {demand}", "d"))

    # Check no overlap between plants and consumers
    plants_set = set(instance.plants)
    consumers_set = set(instance.consumers)
    overlap = plants_set & consumers_set
    if overlap:
        errors.append(ValidationError(
            "ERROR",
            f"nodes {sorted(overlap)} appear in both plants and consumers",
            "plants/consumers"
        ))

    # Check plants + consumers <= n
    if len(plants_set) + len(consumers_set) > instance.n:
        errors.append(ValidationError(
            "ERROR",
            f"total special nodes ({len(plants_set)} + {len(consumers_set)}) exceeds n={instance.n}",
            "n"
        ))

    # Check edge endpoints are in valid range
    for i, (a, b) in enumerate(instance.edges):
        if a < 0 or a >= instance.n:
            errors.append(ValidationError(
                "ERROR",
                f"edge {i} has source {a} out of range [0, {instance.n})",
                "edges"
            ))
        if b < 0 or b >= instance.n:
            errors.append(ValidationError(
                "ERROR",
                f"edge {i} has target {b} out of range [0, {instance.n})",
                "edges"
            ))

    # Check for duplicate node IDs in plants/consumers
    if len(plants_set) != len(instance.plants):
        errors.append(ValidationError("WARNING", "contains duplicate node IDs", "plants"))
    if len(consumers_set) != len(instance.consumers):
        errors.append(ValidationError("WARNING", "contains duplicate node IDs", "consumers"))

    return errors


def _validate_graph_connectivity(instance: Instance) -> List[ValidationError]:
    """
    Validate that at least one plant can reach at least one consumer.
    Uses BFS to check connectivity.
    """
    errors: List[ValidationError] = []

    if not instance.edges:
        errors.append(ValidationError(
            "ERROR",
            "no edges provided - no plant can reach any consumer",
            "connectivity"
        ))
        return errors

    if not instance.plants or not instance.consumers:
        return errors

    adj = _build_adjacency_list(instance.n, instance.edges)
    consumers_set = set(instance.consumers)

    any_plant_reaches_consumer = False
    unreachable_plants: List[int] = []

    for plant in instance.plants:
        if _bfs_reaches_any(adj, plant, consumers_set):
            any_plant_reaches_consumer = True
        else:
            unreachable_plants.append(plant)

    if not any_plant_reaches_consumer:
        errors.append(ValidationError(
            "ERROR",
            "no plant can reach any consumer - problem is trivially infeasible",
            "connectivity"
        ))
    elif unreachable_plants:
        errors.append(ValidationError(
            "WARNING",
            f"plants {unreachable_plants} cannot reach any consumer",
            "connectivity"
        ))

    return errors


def _validate_budget_feasibility(instance: Instance) -> List[ValidationError]:
    """Validate that at least one edge is affordable within budget."""
    errors: List[ValidationError] = []

    if not instance.kappa:
        return errors

    min_cost = min(instance.kappa)
    if min_cost > instance.B:
        errors.append(ValidationError(
            "WARNING",
            f"all edges cost more than budget (min cost: {min_cost}, budget: {instance.B})",
            "budget"
        ))

    return errors


def _build_adjacency_list(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Build adjacency list from edge list."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if 0 <= a < n and 0 <= b < n:
            adj[a].append(b)
    return adj


def _bfs_reaches_any(adj: List[List[int]], start: int, targets: Set[int]) -> bool:
    """
    Check if start node can reach any node in targets using BFS.

    Args:
        adj: Adjacency list
        start: Starting node
        targets: Set of target nodes

    Returns:
        True if any target is reachable from start
    """
    if start in targets:
        return True

    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()

        for neighbor in adj[node]:
            if neighbor in targets:
                return True

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False
