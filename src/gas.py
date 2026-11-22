"""Single-phase gas property providers for pure CH4 and pure H2 (SI units).

A provider exposes ``gas_constant`` [J/(kg K)] and ``z``, ``density`` [kg/m3] and
``viscosity`` [Pa s] as functions of absolute pressure [Pa] and temperature [K].
Inadmissible states raise ``ValueError``; the solver treats such trial states as
rejected and backtracks.

  MethaneGas    pure CH4: Papay or HEOS (Setzmann-Wagner) Z, HEOS viscosity
  HydrogenGas   pure normal H2: Papay or HEOS (Leachman) Z, HEOS (Muzny) viscosity
  Gerg2008Gas   GERG-2008 Z and density on top of one of the above
  IdealGas      Z = 1, constant viscosity; needs no optional dependencies

The real-gas providers need CoolProp, and GERG-2008 needs pyaga8
(``pip install -r src/requirements.txt``). They cache evaluations and hold mutable backend
states; do not share one instance across threads. Supported domain: 10 kPa–20 MPa, 250–350 K.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite

R_UNIVERSAL = 8.31446261815324  # J/(mol K)
MIN_PRESSURE, MAX_PRESSURE = 1e4, 20e6       # Pa
MIN_TEMPERATURE, MAX_TEMPERATURE = 250.0, 350.0  # K


def papay_z(pressure, temperature, critical_pressure, critical_temperature):
    """Papay compressibility factor from reduced pressure and temperature."""
    if not all(isfinite(v) and v > 0 for v in
               (pressure, temperature, critical_pressure, critical_temperature)):
        raise ValueError("Papay requires positive finite absolute pressures and temperatures")
    pr, tr = pressure / critical_pressure, temperature / critical_temperature
    z = 1 - 3.52 * pr * exp(-2.26 * tr) + 0.274 * pr**2 * exp(-1.878 * tr)
    if not isfinite(z) or z <= 0:
        raise ValueError("Papay returned nonpositive compressibility")
    return z


@dataclass(frozen=True)
class IdealGas:
    """Ideal gas (Z = 1) with constant viscosity; dependency-free verification provider."""

    molar_mass: float = 0.016            # kg/mol
    dynamic_viscosity: float = 1.058e-5  # Pa s

    def __post_init__(self):
        if not isfinite(self.molar_mass) or self.molar_mass <= 0:
            raise ValueError("Molar mass must be positive [kg/mol]")
        if not isfinite(self.dynamic_viscosity) or self.dynamic_viscosity <= 0:
            raise ValueError("Viscosity must be positive [Pa s]")

    @property
    def gas_constant(self):
        return R_UNIVERSAL / self.molar_mass

    @staticmethod
    def _check(pressure, temperature):
        if not isfinite(pressure) or not isfinite(temperature) or min(pressure, temperature) <= 0:
            raise ValueError("Gas state requires positive finite pressure and temperature")

    def z(self, pressure, temperature):
        self._check(pressure, temperature)
        return 1.0

    def density(self, pressure, temperature):
        return pressure / (self.z(pressure, temperature) * self.gas_constant * temperature)

    def viscosity(self, pressure, temperature):
        self._check(pressure, temperature)
        return self.dynamic_viscosity


class _CoolPropGas:
    """Shared pure-fluid provider: CoolProp HEOS state, Papay or HEOS Z, HEOS viscosity."""

    fluid = ""
    critical_pressure = critical_temperature = 0.0
    min_pressure, max_pressure = MIN_PRESSURE, MAX_PRESSURE
    min_temperature, max_temperature = MIN_TEMPERATURE, MAX_TEMPERATURE

    def __init__(self, z_model="papay"):
        if z_model not in ("papay", "heos"):
            raise ValueError("z_model must be papay or heos")
        try:
            import CoolProp.CoolProp as CP
        except ImportError as exc:
            raise ImportError('Real-gas providers require CoolProp: pip install -r src/requirements.txt') from exc
        self._cp = CP
        self._state = CP.AbstractState("HEOS", self.fluid)
        self.z_model = z_model
        self.molar_mass = self._state.molar_mass()
        self.gas_constant = self._state.gas_constant() / self.molar_mass
        self._evaluate = lru_cache(maxsize=8192)(self._evaluate_state)

    def _validate(self, p, t):
        if not (isfinite(p) and isfinite(t) and self.min_pressure <= p <= self.max_pressure
                and self.min_temperature <= t <= self.max_temperature):
            raise ValueError("Gas state outside supported domain: 10 kPa–20 MPa, 250–350 K")

    def _evaluate_state(self, p, t):
        self._validate(p, t)
        try:
            self._state.update(self._cp.PT_INPUTS, p, t)
            if self._state.phase() not in (self._cp.iphase_gas, self._cp.iphase_supercritical,
                                           self._cp.iphase_supercritical_gas):
                raise ValueError("Single gas/supercritical phase required")
            if self.z_model == "papay":
                z = papay_z(p, t, self.critical_pressure, self.critical_temperature)
                rho = p / (z * self.gas_constant * t)
            else:
                z, rho = self._state.compressibility_factor(), self._state.rhomass()
            values = (z, rho, self._state.viscosity(), self._state.speed_sound())
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid {self.fluid} state at p={p} Pa, T={t} K: {exc}") from exc
        if not all(isfinite(v) and v > 0 for v in values):
            raise ValueError(f"{self.fluid} backend returned nonphysical properties")
        return values

    def z(self, pressure, temperature):
        return self._evaluate(float(pressure), float(temperature))[0]

    def density(self, pressure, temperature):
        return self._evaluate(float(pressure), float(temperature))[1]

    def viscosity(self, pressure, temperature):
        return self._evaluate(float(pressure), float(temperature))[2]


class MethaneGas(_CoolPropGas):
    """Pure CH4."""

    fluid = "Methane"
    critical_pressure, critical_temperature = 4.5992e6, 190.564


class HydrogenGas(_CoolPropGas):
    """Pure normal H2."""

    fluid = "Hydrogen"
    critical_pressure, critical_temperature = 1296357.6060553084, 33.1443326883113


class Gerg2008Gas:
    """GERG-2008 Z and density (pyaga8) for a pure fluid; the base provider supplies
    the domain check, gas constant and viscosity."""

    _COMPONENT = {"Methane": "methane", "Hydrogen": "hydrogen"}

    def __init__(self, base):
        try:
            import pyaga8
        except ImportError as exc:
            raise ImportError('GERG-2008 requires pyaga8: pip install -r src/requirements.txt') from exc
        if getattr(base, "fluid", None) not in self._COMPONENT:
            raise ValueError("Gerg2008Gas expects a MethaneGas or HydrogenGas base provider")
        self.base = base
        composition = pyaga8.Composition()
        setattr(composition, self._COMPONENT[base.fluid], 1.0)
        self._state = pyaga8.Gerg2008()
        self._state.set_composition(composition)
        self.fluid, self.molar_mass, self.gas_constant = base.fluid, base.molar_mass, base.gas_constant
        self.min_pressure = base.min_pressure
        self.z_model = "gerg2008"
        self._evaluate = lru_cache(maxsize=8192)(self._evaluate_state)

    def _evaluate_state(self, p, t):
        self.base._validate(p, t)
        state = self._state
        state.temperature, state.pressure = t, p / 1000.0   # pyaga8 uses kPa
        state.calc_density(0)
        state.calc_properties()
        if abs(state.pressure * 1000 / p - 1) > 1e-8:
            raise ValueError(f"GERG-2008 pressure closure failed at p={p} Pa, T={t} K")
        mm = state.mm / 1000.0                      # kg/mol
        rho = state.d * 1000.0 * mm                 # mol/l -> kg/m3
        values = (float(state.z), rho, self.base.viscosity(p, t), float(state.w), mm / state.dp_dd)
        if not all(isfinite(v) and v > 0 for v in values):
            raise ValueError(f"GERG-2008 returned nonphysical properties at p={p} Pa, T={t} K")
        return values

    def z(self, pressure, temperature):
        return self._evaluate(float(pressure), float(temperature))[0]

    def density(self, pressure, temperature):
        return self._evaluate(float(pressure), float(temperature))[1]

    def viscosity(self, pressure, temperature):
        return self._evaluate(float(pressure), float(temperature))[2]


def gas_preset(fluid, law="gerg2008"):
    """Gas provider as used in the experiments.

    fluid: "CH4" or "H2".  law: "gerg2008" (GERG-2008 density, HEOS viscosity),
    "papay" or "heos" (the pure-fluid provider's own Z model).
    """
    providers = {"CH4": MethaneGas, "H2": HydrogenGas}
    if fluid not in providers:
        raise ValueError(f"Unknown fluid {fluid!r}; choose CH4 or H2")
    if law not in ("gerg2008", "papay", "heos"):
        raise ValueError("law must be gerg2008, papay or heos")
    base = providers[fluid](z_model="papay" if law == "gerg2008" else law)
    return Gerg2008Gas(base) if law == "gerg2008" else base
