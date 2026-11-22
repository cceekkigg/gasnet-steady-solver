"""gasnet-steady-solver: steady-state solver for isothermal gas transport networks.

Newton's method with an analytically assembled sparse Jacobian, a sparse direct
linear solver, backtracking on the squared L2 merit of the scaled residual, and
convergence on the scaled residual infinity norm. All quantities are SI.

Optional modules: ``src.samples`` (validated N1–N5 cases, needs CoolProp and pyaga8)
and ``src.visualization`` (needs matplotlib).
"""

from .gas import Gerg2008Gas, HydrogenGas, IdealGas, MethaneGas, gas_preset
from .network import Control, Network, Node, Pipe
from .newton import JacobianDomainError, NewtonOptions, NewtonResult, solve_newton
from .pipe import LocalPipePhysics
from .steady import START_METHODS, SteadyResult, SteadyStateSolver

__version__ = "1.0.0"

__all__ = [
    "Node", "Pipe", "Control", "Network",
    "IdealGas", "MethaneGas", "HydrogenGas", "Gerg2008Gas", "gas_preset",
    "LocalPipePhysics",
    "NewtonOptions", "NewtonResult", "JacobianDomainError", "solve_newton",
    "SteadyStateSolver", "SteadyResult", "START_METHODS",
]
