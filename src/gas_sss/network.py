"""Network data: nodes, pipes, ideal control branches and topology validation.

All quantities are SI: absolute pressure [Pa], mass flow [kg/s], length [m],
temperature [K]. Positive injection supplies gas; negative injection withdraws it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, pi
from typing import Hashable

import numpy as np
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Node:
    """A node with either a prescribed pressure or a prescribed injection.

    ``height`` [m] is the geodetic height; pipes take their elevation change from it.
    """

    id: Hashable
    pressure: float | None = None
    injection: float | None = None
    height: float = 0.0

    def __post_init__(self):
        hash(self.id)
        if (self.pressure is None) == (self.injection is None):
            raise ValueError("Specify exactly one of pressure or injection per node")
        value = self.pressure if self.pressure is not None else self.injection
        if not isfinite(self.height):
            raise ValueError("Node height must be finite [m]")
        if not isfinite(value):
            raise ValueError("Node boundary values must be finite")
        if self.pressure is not None and self.pressure <= 0:
            raise ValueError("Pressure must be positive and absolute [Pa]")


@dataclass(frozen=True)
class Pipe:
    """Isothermal pipe oriented start -> end; negative flow means reverse flow."""

    id: Hashable
    start: Hashable
    end: Hashable
    length: float
    diameter: float
    roughness: float = 0.0
    temperature: float = 283.15
    elevation_change: float = 0.0  # height(end) - height(start) [m], constant slope

    def __post_init__(self):
        for identifier in (self.id, self.start, self.end):
            hash(identifier)
        if self.start == self.end:
            raise ValueError("Self-loop pipes are unsupported")
        for name in ("length", "diameter", "temperature"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.roughness) or not 0 <= self.roughness < self.diameter:
            raise ValueError("Roughness must satisfy 0 <= roughness < diameter")
        if not isfinite(self.elevation_change) or abs(self.elevation_change) > self.length:
            raise ValueError("Elevation change must be finite and not exceed the pipe length")

    @property
    def slope(self) -> float:
        """sin(inclination), positive uphill from start to end."""
        return self.elevation_change / self.length

    @property
    def area(self) -> float:
        return pi * self.diameter**2 / 4


@dataclass(frozen=True)
class Control:
    """Ideal control branch oriented start -> end.

    kind "CS" (compressor) or "CV" (control valve): ``p_end - setpoint = 0``.
    kind "VA" (valve): open ``p_start - p_end = 0``; closed ``q = 0``.
    Reverse flow is allowed; equipment limits are not modelled.
    """

    id: Hashable
    start: Hashable
    end: Hashable
    kind: str
    setpoint: float | None = None
    is_open: bool | None = None

    def __post_init__(self):
        for identifier in (self.id, self.start, self.end):
            hash(identifier)
        if self.start == self.end:
            raise ValueError("Self-loop controls are unsupported")
        if self.kind not in {"CS", "CV", "VA"}:
            raise ValueError("Control kind must be CS, CV or VA")
        if self.kind == "VA":
            if type(self.is_open) is not bool or self.setpoint is not None:
                raise ValueError("VA requires an explicit boolean is_open and no setpoint")
        elif (self.is_open is not None or self.setpoint is None
              or not isfinite(self.setpoint) or self.setpoint <= 0):
            raise ValueError("CS/CV requires a finite positive setpoint [Pa] and no valve state")

    @property
    def closed(self) -> bool:
        return self.kind == "VA" and not self.is_open

    def residual(self, pi, pj, q):
        if self.kind in {"CS", "CV"}:
            return pj - self.setpoint
        return q if self.closed else pi - pj

    @property
    def derivatives(self):
        """(dR/dp_start, dR/dp_end, dR/dq)."""
        if self.kind in {"CS", "CV"}:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0) if self.closed else (1.0, -1.0, 0.0)


class Network:
    """Immutable network with signed incidence B[start, e] = -1, B[end, e] = +1.

    Mass balance: B @ q + s = 0. Every connected component needs a pressure
    boundary. Ill-posed control configurations are rejected, not regularized.
    """

    def __init__(self, nodes: list[Node], branches: list[Pipe | Control]):
        self.nodes = tuple(nodes)
        heights = {node.id: node.height for node in self.nodes}
        if any(heights.values()):
            resolved = []
            for b in branches:
                if isinstance(b, Pipe) and b.start in heights and b.end in heights:
                    dh = heights[b.end] - heights[b.start]
                    if b.elevation_change not in (0.0, dh) and abs(b.elevation_change - dh) > 1e-9:
                        raise ValueError(f"Pipe {b.id!r}: elevation_change contradicts node heights")
                    b = replace(b, elevation_change=dh)
                resolved.append(b)
            branches = resolved
        self.branches = tuple(branches)
        if not self.nodes:
            raise ValueError("A network must contain nodes")
        self.node_index = {node.id: i for i, node in enumerate(self.nodes)}
        if len(self.node_index) != len(self.nodes):
            raise ValueError("Duplicate node IDs")
        if len({b.id for b in self.branches}) != len(self.branches):
            raise ValueError("Duplicate branch IDs")
        adjacency = [set() for _ in self.nodes]
        starts, ends = [], []
        for b in self.branches:
            if b.start not in self.node_index or b.end not in self.node_index:
                raise ValueError(f"Branch {b.id!r} references an unknown node")
            i, j = self.node_index[b.start], self.node_index[b.end]
            starts.append(i)
            ends.append(j)
            if not isinstance(b, Control) or not b.closed:
                adjacency[i].add(j)
                adjacency[j].add(i)
        self.starts = np.array(starts, dtype=int)
        self.ends = np.array(ends, dtype=int)
        edges = np.arange(len(self.branches))
        self.incidence = coo_matrix((np.r_[-np.ones(edges.size), np.ones(edges.size)],
                                     (np.r_[self.starts, self.ends], np.r_[edges, edges])),
                                    shape=(len(self.nodes), len(self.branches))).tocsr()
        self.pressure_indices = np.array(
            [i for i, node in enumerate(self.nodes) if node.pressure is not None], dtype=int)
        self.free_indices = np.array(
            [i for i, node in enumerate(self.nodes) if node.pressure is None], dtype=int)
        pressure_nodes = set(self.pressure_indices.tolist())
        unseen = set(range(len(self.nodes)))
        while unseen:
            stack = [unseen.pop()]
            component = set(stack)
            while stack:
                for neighbor in adjacency[stack.pop()]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            if not component & pressure_nodes:
                raise ValueError("Every connected component needs a pressure boundary "
                                 "(closed valves disconnect flow)")
        self._validate_controls()
        for array in (self.incidence.data, self.incidence.indices, self.incidence.indptr,
                      self.starts, self.ends, self.pressure_indices, self.free_indices):
            array.setflags(write=False)

    def _validate_controls(self):
        """Reject undetermined ideal flows, duplicate pressure constraints and unanchored regions."""
        n = len(self.nodes)

        def disjoint_sets():
            parents = list(range(n + 1))

            def root(i):
                while parents[i] != i:
                    parents[i] = parents[parents[i]]
                    i = parents[i]
                return i

            def join(i, j):
                a, b = root(i), root(j)
                if a == b:
                    return False
                parents[a] = b
                return True
            return root, join

        controls = [b for b in self.branches if isinstance(b, Control)]
        if not controls:
            return
        root, join = disjoint_sets()
        for i in self.pressure_indices:
            join(int(i), n)
        for b, i, j in zip(self.branches, self.starts, self.ends):
            if isinstance(b, Control) and not b.closed and not join(i, j):
                raise ValueError(f"Undetermined ideal branch flows at {b.id!r}: "
                                 "ideal cycle or path between pressure boundaries")

        root, join = disjoint_sets()
        for b, i, j in zip(self.branches, self.starts, self.ends):
            if isinstance(b, Control) and b.kind == "VA" and b.is_open:
                join(i, j)
        anchors = {}
        for i in self.pressure_indices:
            anchors[root(i)] = (self.nodes[i].pressure, self.nodes[i].id)
        for b in controls:
            if b.kind == "VA":
                continue
            key = root(self.node_index[b.end])
            if key in anchors:
                value, identifier = anchors[key]
                problem = "Redundant" if value == b.setpoint else "Conflicting"
                raise ValueError(f"{problem} pressure constraints: {identifier!r} and {b.id!r}")
            anchors[key] = (b.setpoint, b.id)

        root, join = disjoint_sets()   # pipes and open valves couple pressures; CS/CV do not
        for b, i, j in zip(self.branches, self.starts, self.ends):
            if isinstance(b, Pipe) or (b.kind == "VA" and b.is_open):
                join(i, j)
        anchored = {root(i) for i in self.pressure_indices}
        anchored.update(root(self.node_index[b.end]) for b in controls if b.kind != "VA")
        missing = [self.nodes[i].id for i in range(n) if root(i) not in anchored]
        if missing:
            raise ValueError(f"Unanchored pressure region: {missing[:5]!r}")
