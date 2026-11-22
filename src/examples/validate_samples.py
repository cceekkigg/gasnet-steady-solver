"""Run the validated N1–N5 samples (pure CH4 and H2, GERG-2008) and compare with references.

    python -m src.examples.validate_samples [--plot validation.png]

Needs CoolProp and pyaga8; --plot also needs matplotlib.
"""

import argparse

from src.samples import compare, list_samples, positions, run_sample

parser = argparse.ArgumentParser()
parser.add_argument("--plot", metavar="PNG", help="save network, convergence and parity plots")
args = parser.parse_args()

runs = {}
print(f"{'sample':19s} {'gas':3s} {'topology':21s} {'it':>2s}  {'vs thesis':>10s}  "
      f"{'vs MYNTS':>9s}  {'vs SIMONE':>9s}   max |dp|")
for sample_id in list_samples():
    sample, result = run_sample(sample_id)
    runs[sample_id] = (sample, result)
    c = compare(sample, result)
    print(f"{sample_id:19s} {sample['gas']:3s} {sample['topology']:21s} {result.numerical.iterations:2d}  "
          f"{c['thesis']['max_dp_Pa']:7.1e} Pa  {c['mynts']['max_dp_Pa'] / 1e5:5.3f} bar  "
          f"{c['simone']['max_dp_Pa'] / 1e5:5.3f} bar")

if args.plot:
    import matplotlib.pyplot as plt

    from src.visualization import plot_convergence, plot_network, plot_parity

    fig = plt.figure(figsize=(11, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    for k, sample_id in enumerate(("N3.0_G1", "N5.2_G3")):
        sample, result = runs[sample_id]
        plot_network(result, positions(sample), ax=fig.add_subplot(grid[0, k]),
                     title=f"{sample_id}: {sample['topology']}, {sample['gas']}")
    plot_convergence([r for _, r in runs.values()], labels=[s["gas"] for s, _ in runs.values()],
                     ax=fig.add_subplot(grid[0, 2]))
    for k, name in enumerate(("mynts", "simone")):
        pairs = {}
        for gas in ("CH4", "H2"):
            ref, own = [], []
            for sample, result in (v for v in runs.values() if v[0]["gas"] == gas):
                pressures = result.pressure_by_node
                for node, value in sample["references"][name]["pressure_Pa"].items():
                    if node in pressures:
                        ref.append(value)
                        own.append(pressures[node])
            pairs[gas] = (ref, own)
        plot_parity(pairs, ax=fig.add_subplot(grid[1, k]), title=f"Node pressures vs {name.upper()}")
    fig.savefig(args.plot, dpi=200)
    print("saved", args.plot)
