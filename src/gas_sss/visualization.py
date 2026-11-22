"""Basic matplotlib plots: network state, convergence history and validation parity.

Requires matplotlib (``pip install "gas-sss[plot]"``). Each function draws on a
given Axes (or a new figure) and returns the Axes.
"""

from __future__ import annotations

import math

import numpy as np

from .network import Control, Pipe

BAR = 1e5
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c9c8c3"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PRESSURE_RAMP = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")  # light -> dark


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Plotting requires matplotlib: pip install "gas-sss[plot]"') from exc
    return plt


def _axes(ax, size=(6, 4)):
    if ax is None:
        _, ax = _plt().subplots(figsize=size)
    return ax


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8)


def circular_layout(network):
    """Fallback node positions on a circle (network order)."""
    n = len(network.nodes)
    return {node.id: (math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
            for k, node in enumerate(network.nodes)}


def plot_network(result, positions=None, ax=None, labels=True, title=None):
    """Nodes coloured by pressure [bar]; pipe line width proportional to |flow|,
    arrow in the actual flow direction. Squares mark pressure boundaries,
    dashed lines mark controls."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _plt()
    ax = _axes(ax)
    net = result.network
    xy = positions or circular_layout(net)
    flows = np.abs(result.flows)
    qmax = flows.max() if flows.size and flows.max() > 0 else 1.0
    coords = np.array(list(xy.values()), dtype=float)
    span = float(np.ptp(coords, axis=0).max()) or 1.0
    groups = {}
    for e, b in enumerate(net.branches):
        groups.setdefault(frozenset((b.start, b.end)), []).append(e)
    for e, (b, q, aq) in enumerate(zip(net.branches, result.flows, flows)):
        a, c = np.array(xy[b.start], float), np.array(xy[b.end], float)
        if (b.start, b.end) != tuple(sorted((b.start, b.end), key=str)):
            a, c = c, a                                     # common orientation per node pair
        d = c - a
        normal = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1.0)
        group = groups[frozenset((b.start, b.end))]        # parallel branches: arcs
        bulge = (group.index(e) - (len(group) - 1) / 2) * 0.12 * span
        t = np.linspace(0, 1, 41)[:, None]                 # quadratic Bezier, peak = bulge
        control = (a + c) / 2 + 2 * bulge * normal
        curve = (1 - t)**2 * a + 2 * t * (1 - t) * control + t**2 * c
        if (b.start, b.end) != tuple(sorted((b.start, b.end), key=str)):
            curve = curve[::-1]                             # back to start -> end
        if q < 0:
            curve = curve[::-1]                             # actual flow direction
        width = 0.8 + 3.2 * aq / qmax
        style = "-" if isinstance(b, Pipe) else "--"
        ax.plot(curve[:, 0], curve[:, 1], style, color=MUTED, lw=width, solid_capstyle="round",
                zorder=1, alpha=0.35 if isinstance(b, Control) and b.closed else 1.0)
        xm, ym = curve[20]
        if aq > 1e-9:
            ax.annotate("", xy=tuple(curve[22]), xytext=(xm, ym),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0, mutation_scale=10), zorder=2)
        if labels:
            nx, ny = normal
            side = 1 if bulge >= 0 else -1
            ax.annotate(f"{q:.3g} kg/s", (xm, ym), textcoords="offset points",
                        xytext=(side * nx * 11, side * ny * 11), ha="center", va="center",
                        fontsize=7, color=MUTED, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    cmap = LinearSegmentedColormap.from_list("pressure", PRESSURE_RAMP)
    p = result.pressures / BAR
    lo, hi = p.min(), p.max()
    norm = plt.Normalize(lo - 1e-9, hi + 1e-9)
    fixed = set(net.pressure_indices.tolist())
    for marker, members in (("s", sorted(fixed)), ("o", [i for i in range(len(p)) if i not in fixed])):
        if members:
            ax.scatter([xy[net.nodes[i].id][0] for i in members], [xy[net.nodes[i].id][1] for i in members],
                       c=p[members], cmap=cmap, norm=norm, marker=marker, s=70,
                       edgecolors="white", linewidths=1.5, zorder=3)
    if labels:
        for i, node in enumerate(net.nodes):
            ax.annotate(f"{node.id}\n{p[i]:.3f} bar", xy[node.id], textcoords="offset points",
                        xytext=(0, -9), ha="center", va="top", fontsize=7, color=INK, zorder=4)
    bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, pad=0.02)
    bar.set_label("pressure [bar]", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    bar.outline.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.18)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    return ax


def plot_convergence(results, labels=None, ax=None, tolerance=1e-8):
    """Scaled residual infinity norm per Newton iteration (log scale).

    Runs with the same label share one colour and one legend entry (at most 8 labels).
    """
    ax = _axes(ax, (5, 3.2))
    results = results if isinstance(results, (list, tuple)) else [results]
    labels = labels or [f"run {k + 1}" for k in range(len(results))]
    names = list(dict.fromkeys(labels))
    if len(names) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} distinct labels; group runs by a shared label")
    markers, shown = "osD^vP*X", set()
    for res, label in zip(results, labels):
        k = names.index(label)
        history = getattr(res, "numerical", res).history
        ax.semilogy(range(len(history)), np.maximum(history, 1e-17), marker=markers[k], ms=4,
                    lw=1.2, alpha=0.85, color=SERIES[k], label=None if label in shown else label)
        shown.add(label)
    ax.axhline(tolerance, color=MUTED, lw=0.8, ls="--")
    ax.annotate("tolerance", (0, tolerance), textcoords="offset points", xytext=(2, 3),
                fontsize=7, color=MUTED)
    ax.set_xlabel("Newton iteration", fontsize=8, color=MUTED)
    ax.set_ylabel(r"$\|F\|_\infty$ (scaled residual)", fontsize=8, color=MUTED)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", color=RULE, lw=0.4)
    _clean(ax)
    if len(names) > 1:
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    return ax


def plot_parity(pairs, ax=None, unit="bar", title=None):
    """Parity plot. ``pairs``: {reference name: (reference values, solver values)} in Pa."""
    ax = _axes(ax, (3.6, 3.6))
    scale = BAR if unit == "bar" else 1.0
    lo, hi = np.inf, -np.inf
    for k, (name, (ref, own)) in enumerate(pairs.items()):
        ref, own = np.asarray(ref) / scale, np.asarray(own) / scale
        lo, hi = min(lo, ref.min(), own.min()), max(hi, ref.max(), own.max())
        ax.scatter(ref, own, s=22, marker="os^D"[k % 4], facecolors="none" if k else SERIES[k],
                   edgecolors=SERIES[k % len(SERIES)], lw=1.0, label=name, zorder=2)
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.8, zorder=1)
    ax.set_xlabel(f"reference pressure [{unit}]", fontsize=8, color=MUTED)
    ax.set_ylabel(f"gas-sss pressure [{unit}]", fontsize=8, color=MUTED)
    ax.set_aspect("equal")
    ax.grid(color=RULE, lw=0.4)
    _clean(ax)
    ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    return ax
