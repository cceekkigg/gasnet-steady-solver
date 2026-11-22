"""Reynolds number and Darcy friction factor with analytical flow derivatives.

Laminar 64/Re for Re <= 2300, Colebrook-White for Re >= 4000, and a smoothstep
blend of both in between. At zero flow the finite product f*q*|q| is used.
"""

from math import isfinite, log, log10, sqrt

from scipy.optimize import brentq


def reynolds_number(q, diameter, area, viscosity):
    if not isfinite(q) or not all(isfinite(v) and v > 0 for v in (diameter, area, viscosity)):
        raise ValueError("Re requires finite flow and positive diameter, area, viscosity")
    return abs(q) * diameter / (area * viscosity)


def colebrook_factor(reynolds, relative_roughness):
    """Solve Colebrook-White for x = 1/sqrt(f)."""
    if not isfinite(reynolds) or reynolds < 2300:
        raise ValueError("Colebrook requires Re >= 2300")
    if not isfinite(relative_roughness) or not 0 <= relative_roughness < 1:
        raise ValueError("Relative roughness must lie in [0,1)")
    x = brentq(lambda x: x + 2 * log10(relative_roughness / 3.7 + 2.51 * x / reynolds),
               0.1, 100, xtol=1e-12)
    return 1 / x**2


def darcy_factor(reynolds, relative_roughness):
    if not isfinite(reynolds) or reynolds < 0:
        raise ValueError("Reynolds number must be finite and nonnegative")
    if not isfinite(relative_roughness) or not 0 <= relative_roughness < 1:
        raise ValueError("Relative roughness must lie in [0,1)")
    if reynolds == 0:
        return float("inf")
    laminar = 64 / reynolds
    if reynolds <= 2300:
        return laminar
    turbulent = colebrook_factor(reynolds, relative_roughness)
    if reynolds >= 4000:
        return turbulent
    weight = (reynolds - 2300) / 1700
    weight = weight**2 * (3 - 2 * weight)
    return (1 - weight) * laminar + weight * turbulent


def friction_product(q, diameter, area, viscosity, roughness, fixed_friction=None):
    """Signed f_D * q * |q|, including exactly zero flow."""
    re = reynolds_number(q, diameter, area, viscosity)
    if not isfinite(roughness) or not 0 <= roughness < diameter:
        raise ValueError("Invalid absolute roughness")
    if fixed_friction is not None:
        if not isfinite(fixed_friction) or fixed_friction <= 0:
            raise ValueError("Fixed Darcy factor must be finite and positive")
        return fixed_friction * q * abs(q)
    if re <= 2300:
        return 64 * area * viscosity * q / diameter
    return darcy_factor(re, roughness / diameter) * q * abs(q)


def friction_product_partials(q, diameter, area, viscosity, roughness, fixed_friction=None):
    """Return (f*q*|q|, d/dq, d/dmu).

    Colebrook derivatives follow from implicit differentiation; the transition
    derivative includes the derivative of the smoothstep weight.
    """
    product = friction_product(q, diameter, area, viscosity, roughness, fixed_friction)
    if fixed_friction is not None:
        return product, 2 * fixed_friction * abs(q), 0.0
    re = reynolds_number(q, diameter, area, viscosity)
    if re <= 2300:
        return product, 64 * area * viscosity / diameter, 64 * area * q / diameter
    turbulent = colebrook_factor(re, roughness / diameter)
    x = 1 / sqrt(turbulent)
    argument = roughness / diameter / 3.7 + 2.51 * x / re
    c = 2 / log(10)
    dx_dre = (c * 2.51 * x / (re**2 * argument)) / (1 + c * 2.51 / (re * argument))
    dt_dre = -2 * dx_dre / x**3
    factor, df_dre = turbulent, dt_dre
    if re < 4000:
        t = (re - 2300) / 1700
        weight, dw_dre = t*t*(3 - 2*t), 6*t*(1 - t)/1700
        laminar = 64 / re
        factor = (1 - weight) * laminar + weight * turbulent
        df_dre = (1 - weight) * (-64 / re**2) + weight * dt_dre + dw_dre * (turbulent - laminar)
    return product, abs(q) * (2 * factor + re * df_dre), -q * abs(q) * df_dre * re / viscosity
