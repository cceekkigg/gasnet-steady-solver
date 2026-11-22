"""Validated sample cases: synthetic networks N1–N5 with pure CH4 and pure H2.

Each sample carries reference solutions computed with GERG-2008 by MYNTS and
SIMONE, and the result of the thesis solver. Running a sample requires the
real-gas extra (CoolProp, pyaga8).

    from gas_sss.samples import list_samples, run_sample
    for sample_id in list_samples():
        sample, result = run_sample(sample_id)
        print(sample_id, compare(sample, result))
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import numpy as np

from .gas import gas_preset
from .network import Network, Node, Pipe
from .pipe import LocalPipePhysics
from .steady import SteadyStateSolver

REFERENCES = ("thesis", "mynts", "simone")


@lru_cache(maxsize=1)
def _data():
    text = resources.files("gas_sss").joinpath("data/samples.json").read_text(encoding="utf-8")
    return {s["id"]: s for s in json.loads(text)["samples"]}


def list_samples():
    """Sample ids, e.g. "N3.0_G1" (G1 = CH4, G3 = H2)."""
    return list(_data())


def get_sample(sample_id):
    """Raw sample definition: nodes, pipes, positions, gas and reference solutions."""
    try:
        return _data()[sample_id]
    except KeyError:
        raise KeyError(f"Unknown sample {sample_id!r}; available: {list_samples()}") from None


def build_network(sample):
    nodes = [Node(n["id"], pressure=n["pressure_Pa"]) if "pressure_Pa" in n
             else Node(n["id"], injection=n["injection_kg_s"]) for n in sample["nodes"]]
    pipes = [Pipe(p["id"], p["start"], p["end"], p["length_m"], p["diameter_m"], p["roughness_m"],
                  sample["temperature_K"]) for p in sample["pipes"]]
    return Network(nodes, pipes)


def positions(sample):
    """Node coordinates for plotting."""
    return {n["id"]: (n["x"], n["y"]) for n in sample["nodes"]}


def build_solver(sample, options=None):
    gas = gas_preset(sample["gas"], sample["gas_law"])
    return SteadyStateSolver(build_network(sample), LocalPipePhysics(gas), options,
                             minimum_pressure=gas.min_pressure)


def run_sample(sample_id, start="default", options=None):
    """Solve a sample; returns (sample definition, SteadyResult).

    With start="default" the sample's stored initial pressures (if any) override
    the default iterate, as in the thesis runs.
    """
    sample = get_sample(sample_id)
    solver = build_solver(sample, options)
    initial = solver.initial_guess(start)
    if start == "default" and sample.get("initial_pressures_Pa"):
        free = [solver.network.nodes[i].id for i in solver.network.free_indices]
        for node_id, value in sample["initial_pressures_Pa"].items():
            initial[free.index(node_id)] = max(value, solver.minimum_pressure)
    return sample, solver.solve(initial)


def compare(sample, result):
    """Maximum absolute differences to each reference: pressure [Pa] and pipe flow [kg/s].

    Only elements present in both the result and the reference are compared.
    """
    pressures, flows = result.pressure_by_node, result.flow_by_branch
    out = {}
    for name in REFERENCES:
        ref = sample["references"][name]
        dp = [abs(pressures[k] - v) for k, v in ref.get("pressure_Pa", {}).items() if k in pressures]
        dq = [abs(flows[k] - v) for k, v in ref.get("flow_kg_s", {}).items() if k in flows]
        out[name] = {"max_dp_Pa": float(np.max(dp)) if dp else float("nan"),
                     "max_dq_kg_s": float(np.max(dq)) if dq else float("nan"),
                     "n_compared": len(dp) + len(dq)}
    return out
