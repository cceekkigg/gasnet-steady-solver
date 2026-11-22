"""Isothermal steady pipe law with local gas properties and analytical sensitivities.

Along the pipe (no inertia) the squared pressure P = p^2 satisfies

    dP/dx = -p * f_D(Re) * q|q| / (D A^2 rho(p,T)) - 2 p rho(p,T) g sin(theta)

with local density, viscosity and Darcy factor. The branch residual [Pa] is
obtained by shooting from the lower-pressure (downstream) endpoint against the
flow direction:  q >= 0: R = p_i - p_i^pred(p_j, q);  q < 0: R = p_j^pred(p_i, q) - p_j.

Residual derivatives with respect to (p_i, p_j, q) are analytical: for a
horizontal pipe, dp_out/dp_in = h_out/h_in and dp_out/dq = h_out/q * int(q h_q/h dx),
integrated alongside the state; inclined pipes integrate the variational equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from scipy.integrate import solve_ivp

from .friction import friction_product, friction_product_partials

GRAVITY = 9.80665  # m/s^2


@dataclass(frozen=True)
class LocalPipePhysics:
    """Pipe law used by the network solver.

    gas:              property provider (see ``gas_sss.gas``)
    fixed_friction:   optional constant Darcy factor (verification only)
    integration_rtol, integration_atol [Pa^2]: DOP853 tolerances of the pipe integration
    """

    gas: object
    fixed_friction: float | None = None
    integration_rtol: float = 1e-10
    integration_atol: float = 1e-4

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in
                   (self.gas.gas_constant, self.integration_rtol, self.integration_atol)):
            raise ValueError("Gas constant and integration tolerances must be positive")
        if self.fixed_friction is not None and (not isfinite(self.fixed_friction) or self.fixed_friction <= 0):
            raise ValueError("Fixed Darcy factor must be positive")

    def viscosity(self, pressure, temperature):
        if not all(isfinite(v) and v > 0 for v in (pressure, temperature)):
            raise ValueError("Viscosity state must be finite and positive")
        value = self.gas.viscosity(pressure, temperature)
        if not isfinite(value) or value <= 0:
            raise ValueError("Viscosity must be finite and positive")
        return value

    def pressure_squared_gradient(self, pipe, pressure, q):
        """d(p^2)/dx at local pressure ``pressure`` and mass flow ``q``."""
        rho = self.gas.density(pressure, pipe.temperature)
        mu = self.viscosity(pressure, pipe.temperature)
        if not isfinite(rho) or rho <= 0:
            raise ValueError("Invalid local density")
        product = friction_product(q, pipe.diameter, pipe.area, mu, pipe.roughness, self.fixed_friction)
        gradient = -pressure * product / (pipe.diameter * pipe.area**2 * rho)
        if pipe.slope:
            gradient -= 2 * pressure * rho * GRAVITY * pipe.slope
        return gradient

    def _integrate(self, pipe, pressure, q, distance, sensitivities=False, variational=False):
        self.gas.density(pressure, pipe.temperature)
        if not isfinite(q):
            raise ValueError("Mass flow must be finite")

        def ode(x, squared):
            if squared[0] <= 0 or not np.isfinite(squared[0]):
                raise ValueError("Pipe integration reached nonpositive pressure")
            p = sqrt(squared[0])
            gradient = self.pressure_squared_gradient(pipe, p, q)
            if variational:
                # State [P, dP/dp0, dP/dq]; h_P by central difference in P = p^2.
                step = 1e-6 * squared[0]
                hp = (self.pressure_squared_gradient(pipe, sqrt(squared[0] + step), q) -
                      self.pressure_squared_gradient(pipe, sqrt(squared[0] - step), q)) / (2 * step)
                mu = self.viscosity(p, pipe.temperature)
                _, dq, _ = friction_product_partials(q, pipe.diameter, pipe.area, mu,
                                                     pipe.roughness, self.fixed_friction)
                rho = self.gas.density(p, pipe.temperature)
                hq = -p * dq / (pipe.diameter * pipe.area**2 * rho)
                return [gradient, hp * squared[1], hp * squared[2] + hq]
            if not sensitivities:
                return [gradient]
            mu = self.viscosity(p, pipe.temperature)
            product, dq, _ = friction_product_partials(q, pipe.diameter, pipe.area, mu,
                                                       pipe.roughness, self.fixed_friction)
            return [gradient, q * dq / product]

        initial = ([pressure**2, 2*pressure, 0.0] if variational else
                   [pressure**2, 0.0] if sensitivities else [pressure**2])
        atol = ([self.integration_atol, 1e-6, 1e-6] if variational else
                [self.integration_atol, 1e-8] if sensitivities else self.integration_atol)
        sol = solve_ivp(ode, (0, distance), initial, rtol=self.integration_rtol,
                        atol=atol, method="DOP853")
        if not sol.success or not np.isfinite(sol.y).all() or (sol.y[0] <= 0).any():
            raise ValueError("Local pipe-pressure integration failed")
        return sol

    def residual(self, pipe, pi, pj, q):
        """Branch residual [Pa]."""
        self.gas.density(pi, pipe.temperature)
        self.gas.density(pj, pipe.temperature)
        if q >= 0:
            return pi - sqrt(self._integrate(pipe, pj, q, -pipe.length).y[0, -1])
        return sqrt(self._integrate(pipe, pi, q, pipe.length).y[0, -1]) - pj

    def residual_and_derivatives(self, pipe, pi, pj, q):
        """Residual and analytical partials (dR/dp_i, dR/dp_j, dR/dq)."""
        self.gas.density(pi, pipe.temperature)
        self.gas.density(pj, pipe.temperature)
        p0, distance = (pj, -pipe.length) if q >= 0 else (pi, pipe.length)
        if pipe.slope:
            sol = self._integrate(pipe, p0, q, distance, variational=True)
            predicted = sqrt(sol.y[0, -1])
            initial_sensitivity, flow_sensitivity = sol.y[1:, -1] / (2 * predicted)
            if q >= 0:
                return pi - predicted, np.array([1.0, -initial_sensitivity, -flow_sensitivity])
            return predicted - pj, np.array([initial_sensitivity, -1.0, flow_sensitivity])
        if q == 0:   # exact laminar zero-flow limit
            mu = self.viscosity(p0, pipe.temperature)
            _, dq, _ = friction_product_partials(q, pipe.diameter, pipe.area, mu,
                                                 pipe.roughness, self.fixed_friction)
            rho = self.gas.density(p0, pipe.temperature)
            flow_sensitivity = -distance * dq / (2 * pipe.diameter * pipe.area**2 * rho)
            return pi - pj, np.array([1.0, -1.0, -flow_sensitivity])
        sol = self._integrate(pipe, p0, q, distance, sensitivities=True)
        predicted = sqrt(sol.y[0, -1])

        def gradient(p):
            return self.pressure_squared_gradient(pipe, p, q) / (2*p)

        out_gradient = gradient(predicted)
        endpoint_sensitivity = out_gradient / gradient(p0)
        flow_sensitivity = out_gradient / q * sol.y[1, -1]
        if q > 0:
            return pi - predicted, np.array([1.0, -endpoint_sensitivity, -flow_sensitivity])
        return predicted - pj, np.array([endpoint_sensitivity, -1.0, flow_sensitivity])

    def residual_derivatives(self, pipe, pi, pj, q):
        return self.residual_and_derivatives(pipe, pi, pj, q)[1]
