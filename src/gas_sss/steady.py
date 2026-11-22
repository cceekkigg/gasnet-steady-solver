"""Steady network system: assembly of residual and analytical sparse Jacobian.

Unknowns  y = [p_free (pressures of injection nodes), q (all branch flows)].
Residuals R = [mass balances at injection nodes, one equation per branch].

  mass:     (B q + s)_i = 0                                   [kg/s]
  pipe:     shooting residual of LocalPipePhysics             [Pa]
  CS / CV:  p_end - setpoint = 0                              [Pa]
  VA:       open p_start - p_end = 0 [Pa], closed q = 0       [kg/s]

Scaling: pressures and pressure-type residuals by p_ref = largest prescribed
pressure or setpoint; flows and mass-type residuals by q_ref = max(1, max|s_i|).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from .network import Control, Network, Pipe
from .newton import JacobianDomainError, NewtonOptions, NewtonResult, solve_newton
from . import initialization

START_METHODS = ("default", "uniform", "uniform_zero", "routed", "linearized")


@dataclass(frozen=True)
class SteadyResult:
    """Arrays follow the network's node and branch order (SI units).

    A failed solve keeps its last iterate for diagnosis; check ``success`` first.
    ``injections`` includes the injections recovered at pressure nodes.
    """

    network: Network
    pressures: np.ndarray
    flows: np.ndarray
    injections: np.ndarray
    branch_residuals: np.ndarray
    numerical: NewtonResult

    @property
    def success(self) -> bool:
        return self.numerical.success

    @property
    def pressure_by_node(self) -> dict:
        return dict(zip((node.id for node in self.network.nodes), self.pressures.tolist()))

    @property
    def flow_by_branch(self) -> dict:
        return dict(zip((b.id for b in self.network.branches), self.flows.tolist()))


class SteadyStateSolver:
    """Isothermal steady gas-network solver.

    network:           ``Network`` of nodes, pipes and ideal controls
    physics:           pipe law, e.g. ``LocalPipePhysics``
    options:           ``NewtonOptions`` (defaults = settings used in the experiments)
    minimum_pressure:  lower bound on free pressures [Pa]
    """

    def __init__(self, network: Network, physics, options: NewtonOptions | None = None,
                 minimum_pressure: float = 1.0):
        if not np.isfinite(minimum_pressure) or minimum_pressure <= 0:
            raise ValueError("minimum_pressure must be finite and positive [Pa]")
        if not callable(getattr(physics, "residual_derivatives", None)):
            raise ValueError("The pipe model must provide residual_derivatives")
        self.network, self.physics = network, physics
        self.options = options or NewtonOptions()
        self.minimum_pressure = minimum_pressure
        self.n_free = len(network.free_indices)
        self.n_unknowns = self.n_free + len(network.branches)
        self.active_edges = np.array([e for e, b in enumerate(network.branches)
                                      if not isinstance(b, Control) or not b.closed], dtype=int)
        self.diagnostics = {}

        # Cached sparsity pattern: mass rows from the incidence matrix, then three
        # entries (p_start, p_end, q) per branch row, omitting fixed pressures.
        free = network.incidence[network.free_indices].tocoo()
        self._mass_values = free.data.copy()
        rows, cols = free.row.tolist(), (free.col + self.n_free).tolist()
        pressure_column = np.full(len(network.nodes), -1, dtype=int)
        pressure_column[network.free_indices] = np.arange(self.n_free)
        edges, parts = [], []
        for e, (i, j) in enumerate(zip(network.starts, network.ends)):
            for part, col in enumerate((pressure_column[i], pressure_column[j], self.n_free + e)):
                if col >= 0:
                    rows.append(self.n_free + e)
                    cols.append(col)
                    edges.append(e)
                    parts.append(part)
        self._jac_rows, self._jac_cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
        self._jac_edges, self._jac_parts = np.asarray(edges, dtype=int), np.asarray(parts, dtype=int)

        self.boundary_pressures = np.array(
            [node.pressure if node.pressure is not None else np.nan for node in network.nodes])
        self.specified_injections = np.array(
            [node.injection if node.injection is not None else 0.0 for node in network.nodes])
        if (self.boundary_pressures[network.pressure_indices] < minimum_pressure).any():
            raise ValueError("A prescribed pressure is below minimum_pressure")
        setpoints = [b.setpoint for b in network.branches if isinstance(b, Control) and b.kind != "VA"]
        if any(p < minimum_pressure for p in setpoints):
            raise ValueError("A control setpoint is below minimum_pressure")
        self.pressure_scale = max([float(p) for p in self.boundary_pressures[network.pressure_indices]]
                                  + setpoints)
        self.flow_scale = max(1.0, float(np.max(np.abs(self.specified_injections))))
        self.variable_scale = np.r_[np.full(self.n_free, self.pressure_scale),
                                    np.full(len(network.branches), self.flow_scale)]
        self.residual_scale = np.r_[np.full(self.n_free, self.flow_scale), [
            (self.flow_scale if b.closed else self.pressure_scale) if isinstance(b, Control)
            else self.pressure_scale for b in network.branches]]

    def unpack(self, y):
        """All node pressures and branch flows from the unknown vector."""
        y = np.asarray(y, dtype=float)
        if y.shape != (self.n_unknowns,) or not np.isfinite(y).all():
            raise ValueError("Invalid steady unknown vector")
        pressures = self.boundary_pressures.copy()
        pressures[self.network.free_indices] = y[:self.n_free]
        if (pressures < self.minimum_pressure).any():
            raise ValueError("Pressure below minimum_pressure")
        return pressures, y[self.n_free:].copy()

    def residual(self, y):
        pressures, flows = self.unpack(y)
        net = self.network
        mass = (net.incidence @ flows + self.specified_injections)[net.free_indices]
        branches = [b.residual(pressures[i], pressures[j], q) if isinstance(b, Control)
                    else self.physics.residual(b, pressures[i], pressures[j], q)
                    for b, i, j, q in zip(net.branches, net.starts, net.ends, flows)]
        return np.concatenate((mass, np.asarray(branches, dtype=float)))

    def jacobian(self, y):
        """Analytical sparse Jacobian dR/dy (CSC) in physical units."""
        pressures, flows = self.unpack(y)
        net = self.network
        partials = np.empty((len(net.branches), 3))
        for e, (b, i, j, q) in enumerate(zip(net.branches, net.starts, net.ends, flows)):
            if isinstance(b, Control):
                partials[e] = b.derivatives
                continue
            try:
                partials[e] = self.physics.residual_derivatives(b, pressures[i], pressures[j], q)
            except ValueError as exc:
                raise JacobianDomainError(f"Pipe derivative failed: {exc}") from exc
        values = np.r_[self._mass_values, partials[self._jac_edges, self._jac_parts]]
        matrix = coo_matrix((values, (self._jac_rows, self._jac_cols)),
                            shape=(self.n_unknowns, self.n_unknowns)).tocsc()
        matrix.eliminate_zeros()
        return matrix

    def initial_guess(self, method: str = "default"):
        """Starting iterate (see ``gas_sss.initialization``).

        "default" (C0)       boundary-mean pressures, least-norm balanced flows
        "uniform" (C1)       highest known pressure per component, small flow seed
        "uniform_zero" (C1z) as C1 with exactly zero flows
        "routed" (C2)        injections routed along shortest-path trees
        "linearized" (C3)    linear squared-pressure network solve
        """
        if method not in START_METHODS:
            raise ValueError(f"Unknown initialization method {method!r}; choose from {START_METHODS}")
        self.diagnostics = {"method": method}
        if method == "default":
            return initialization.default(self)
        if method == "uniform":
            return initialization.uniform(self)
        if method == "uniform_zero":
            return initialization.uniform(self, seed=False)
        if method == "routed":
            return initialization.routed(self)
        return initialization.linearized(self)

    def solve(self, initial=None) -> SteadyResult:
        """Solve from ``initial`` (an unknown vector, a start-method name, or None = "default")."""
        if initial is None or isinstance(initial, str):
            y0 = self.initial_guess(initial or "default")
        else:
            y0 = np.asarray(initial, dtype=float)
        self.residual(y0)   # validate the starting state before iterating
        lower = np.r_[np.full(self.n_free, self.minimum_pressure),
                      np.full(len(self.network.branches), -np.inf)]
        numerical = solve_newton(self.residual, self.jacobian, y0, self.variable_scale,
                                 self.residual_scale, lower_bounds=lower, options=self.options)
        pressures, flows = self.unpack(numerical.x)
        injections = self.specified_injections.copy()
        net_flow = self.network.incidence @ flows
        injections[self.network.pressure_indices] = -net_flow[self.network.pressure_indices]
        residual = self.residual(numerical.x)
        return SteadyResult(self.network, pressures, flows, injections,
                            residual[self.n_free:], numerical)
