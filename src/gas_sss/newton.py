"""Scaled Newton method with a sparse direct linear solver and backtracking.

Unknowns and residuals are made dimensionless with diagonal scalings,
z = x / x_ref and F(z) = R(x) / r_ref. Each iteration

  1. stops if ||F(z_k)||_inf <= residual_tolerance (convergence criterion),
  2. assembles the analytical sparse Jacobian J = diag(1/r_ref) dR/dx diag(x_ref),
  3. solves J d = -F(z_k) with a sparse direct LU factorization (SciPy spsolve),
  4. backtracks alpha = 1, 1/2, 1/4, ... until the squared L2 merit
     phi(z) = F(z)^T F(z) satisfies phi(z_k + alpha d) <= (1 - 1e-4 alpha) phi(z_k).

Trial points outside the variable bounds, or rejected by the residual callback
with ``ValueError`` (e.g. a gas state outside the property domain), count as
failed trials and are backtracked. The module has no gas-network concepts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve


class JacobianDomainError(ValueError):
    """The iterate is valid, but no admissible Jacobian evaluation exists there."""


@dataclass(frozen=True)
class NewtonOptions:
    """Numerical settings. The defaults are the settings used in the experiments."""

    max_iterations: int = 80
    residual_tolerance: float = 1e-8   # on the scaled residual infinity norm
    max_backtracks: int = 35           # step halvings per iteration
    armijo: float = 1e-4               # sufficient-decrease constant
    divergence_factor: float = 1e6     # fail if ||F||_inf > factor * ||F(z_0)||_inf

    def __post_init__(self):
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        if not isinstance(self.max_backtracks, int) or self.max_backtracks < 1:
            raise ValueError("max_backtracks must be a positive integer")
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.residual_tolerance, self.armijo, self.divergence_factor)):
            raise ValueError("Tolerances must be finite and positive")


@dataclass(frozen=True)
class NewtonResult:
    """Outcome, kept also for failed solves. ``history`` holds ||F||_inf per accepted iterate."""

    x: np.ndarray
    success: bool
    message: str
    iterations: int
    residual_norm: float
    history: tuple[float, ...]
    step_lengths: tuple[float, ...]
    backtracks: int
    residual_evaluations: int
    jacobian_evaluations: int
    jacobian_nnz: int
    jacobian_seconds: float
    linear_solve_seconds: float
    elapsed_seconds: float


def sparse_solve(matrix, rhs):
    """Sparse direct solve; a singular or nonfinite system raises LinAlgError."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            answer = spsolve(csc_matrix(matrix), rhs)
        except (MatrixRankWarning, RuntimeError) as exc:
            raise np.linalg.LinAlgError(f"Sparse factorization failed: {exc}") from exc
    if not np.isfinite(answer).all():
        raise np.linalg.LinAlgError("Nonfinite linear solution")
    return answer


def solve_newton(residual: Callable, jacobian: Callable, x0, variable_scale=None,
                 residual_scale=None, lower_bounds=None, upper_bounds=None,
                 options: NewtonOptions | None = None) -> NewtonResult:
    """Solve the square system R(x) = 0.

    residual(x) -> R(x) and jacobian(x) -> dR/dx (sparse or dense), both in physical units.
    Convergence failures return ``success=False``; invalid arguments raise ValueError.
    """
    options = options or NewtonOptions()
    started = perf_counter()
    counts = {"residual": 0, "jacobian": 0, "backtracks": 0, "nnz": 0}
    times = {"jacobian": 0.0, "linear": 0.0}
    step_lengths, history = [], []
    initial = np.asarray(x0, dtype=float)
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("x0 must be a finite one-dimensional array")
    n = initial.size

    def vector(value, default, name, positive=False):
        array = np.full(n, default, dtype=float) if value is None else np.asarray(value, dtype=float)
        if array.shape != (n,) or np.isnan(array).any():
            raise ValueError(f"Invalid {name} shape or values")
        if positive and (not np.isfinite(array).all() or (array <= 0).any()):
            raise ValueError(f"{name} must be finite and positive")
        return array

    xs = vector(variable_scale, 1, "variable_scale", True)
    rs = vector(residual_scale, 1, "residual_scale", True)
    lower = vector(lower_bounds, -np.inf, "lower_bounds") / xs
    upper = vector(upper_bounds, np.inf, "upper_bounds") / xs
    if (lower >= upper).any():
        raise ValueError("Lower bounds must be below upper bounds")
    z = initial / xs

    def trial(point):
        """Scaled residual, or None if the point is inadmissible."""
        if not np.isfinite(point).all() or (point < lower).any() or (point > upper).any():
            return None
        counts["residual"] += 1
        try:
            value = np.asarray(residual(point * xs), dtype=float)
        except ValueError:
            return None
        if value.shape != (n,) or not np.isfinite(value).all():
            return None
        return value / rs

    def finish(success, message, iteration):
        return NewtonResult((z * xs).copy(), success, message, iteration,
                            history[-1] if history else float("inf"), tuple(history),
                            tuple(step_lengths), counts["backtracks"], counts["residual"],
                            counts["jacobian"], counts["nnz"], times["jacobian"],
                            times["linear"], perf_counter() - started)

    f = trial(z)
    if f is None:
        return finish(False, "Invalid initial state or residual", 0)
    initial_norm = float(np.max(np.abs(f))) if n else 0.0
    for iteration in range(options.max_iterations + 1):
        norm = float(np.max(np.abs(f))) if n else 0.0
        history.append(norm)
        if norm <= options.residual_tolerance:
            return finish(True, "Scaled residual tolerance satisfied", iteration)
        if iteration == options.max_iterations:
            return finish(False, "Maximum iterations reached", iteration)
        if norm > options.divergence_factor * max(initial_norm, options.residual_tolerance):
            return finish(False, "Diverged: residual norm above divergence limit", iteration)

        tick = perf_counter()
        counts["jacobian"] += 1
        try:
            matrix = jacobian(z * xs)
        except JacobianDomainError as exc:
            times["jacobian"] += perf_counter() - tick
            return finish(False, str(exc), iteration)
        matrix = csc_matrix(matrix)
        if matrix.shape != (n, n):
            raise ValueError("Jacobian shape must match the unknown vector")
        matrix = (diags(1 / rs) @ matrix @ diags(xs)).tocsc()
        if not np.isfinite(matrix.data).all():
            raise ValueError("Jacobian must be finite")
        counts["nnz"] = matrix.nnz
        times["jacobian"] += perf_counter() - tick

        tick = perf_counter()
        try:
            direction = sparse_solve(matrix, -f)
        except np.linalg.LinAlgError:
            times["linear"] += perf_counter() - tick
            return finish(False, "Singular or nonfinite Jacobian linear solve", iteration)
        times["linear"] += perf_counter() - tick

        merit = float(f @ f)
        alpha = 1.0
        for _ in range(options.max_backtracks):
            candidate = z + alpha * direction
            candidate_f = trial(candidate)
            if candidate_f is not None and float(candidate_f @ candidate_f) <= (
                    1 - options.armijo * alpha) * merit:
                z, f = candidate, candidate_f
                step_lengths.append(alpha)
                break
            counts["backtracks"] += 1
            alpha *= 0.5
        else:
            return finish(False, "Backtracking failed to reduce residual", iteration)
    raise RuntimeError("Unreachable Newton termination")
