"""Solve a small looped network with a compressor and a closed valve.

    python -m src.examples.quickstart                    # ideal gas, no optional dependencies
    python -m src.examples.quickstart --gas CH4          # pure methane, GERG-2008 (CoolProp, pyaga8)
    python -m src.examples.quickstart --gas H2 --plot    # pure hydrogen, plus figure (matplotlib)
"""

import argparse

from src import (Control, IdealGas, LocalPipePhysics, Network, Node, Pipe,
                     SteadyStateSolver, gas_preset)

BAR = 1e5

parser = argparse.ArgumentParser()
parser.add_argument("--gas", choices=("ideal", "CH4", "H2"), default="ideal")
parser.add_argument("--start", default="default",
                    help="default | uniform | uniform_zero | routed | linearized")
parser.add_argument("--plot", action="store_true", help="save quickstart.png")
args = parser.parse_args()

nodes = [
    Node("supply", pressure=60 * BAR),            # pressure boundary [Pa]
    Node("A", injection=0.0),                     # junction
    Node("B", injection=-12.0),                   # withdrawal [kg/s]
    Node("C", injection=0.0),
    Node("D", injection=-8.0, height=25.0),       # geodetic height [m]
]
branches = [
    Pipe("P1", "supply", "A", length=40e3, diameter=0.6, roughness=2e-5, temperature=283.15),
    Pipe("P2", "A", "B", length=20e3, diameter=0.5, roughness=2e-5),
    Pipe("P3", "A", "C", length=25e3, diameter=0.4, roughness=2e-5),
    Pipe("P4", "B", "C", length=15e3, diameter=0.4, roughness=2e-5),
    Control("CS1", "C", "D", "CS", setpoint=65 * BAR),   # compressor: outlet pressure
    Control("V1", "B", "D", "VA", is_open=False),        # closed valve
]
if args.gas == "H2":   # hydrogen carries less mass at the same pressure drop
    nodes[2], nodes[4] = Node("B", injection=-3.0), Node("D", injection=-2.0, height=25.0)

gas = IdealGas() if args.gas == "ideal" else gas_preset(args.gas, "gerg2008")
solver = SteadyStateSolver(Network(nodes, branches), LocalPipePhysics(gas))
result = solver.solve(args.start)

run = result.numerical
print(f"{run.message}: {run.iterations} iterations, ||F||_inf = {run.residual_norm:.2e}")
print("residual history:", ", ".join(f"{v:.1e}" for v in run.history))
print("step lengths:    ", ", ".join(f"{a:g}" for a in run.step_lengths))
for node, p in result.pressure_by_node.items():
    print(f"  p[{node:>6}] = {p / BAR:8.3f} bar")
for branch, q in result.flow_by_branch.items():
    print(f"  q[{branch:>6}] = {q:8.3f} kg/s")

if args.plot:
    import matplotlib.pyplot as plt

    from src.visualization import plot_convergence, plot_network

    layout = {"supply": (0, 0), "A": (1, 0), "B": (2, 0.6), "C": (2, -0.6), "D": (3, 0)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=(1.6, 1),
                                      layout="constrained")
    plot_network(result, layout, ax=left, title=f"Quickstart network ({args.gas})")
    plot_convergence(result, ax=right)
    fig.savefig("quickstart.png", dpi=200)
    print("saved quickstart.png")
