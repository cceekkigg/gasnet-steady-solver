"""Starting iterates for ``SteadyStateSolver``; none of them changes the equations.

All functions take a constructed solver and return a vector in its unknown
ordering [free pressures, all branch flows].

  default     C0  component-mean boundary pressures; least-norm balanced flows
  uniform     C1  highest known pressure (boundary or setpoint) per pressure-connected
                  component; every pipe flow set to a small positive seed
                  (seed=False: exactly zero flows, C1z)
  routed      C2  uniform pressures; each injection routed to its nearest pressure
                  node along the length-weighted shortest-path tree
  linearized  C3  linear network solve in squared pressure, P_i - P_j = K_e |q0_e| q_e,
                  with fully rough friction and a reference Z; two passes updating q0
"""

from __future__ import annotations

import heapq
from math import log10

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr, spsolve

from .network import Control, Pipe


def _flow_seed(solver, flows):
    pipes = np.array([isinstance(b, Pipe) for b in solver.network.branches])
    scale = np.max(np.abs(flows)) if flows.size and np.max(np.abs(flows)) > 0 else solver.flow_scale
    seed = max(1e-3, 1e-3 * scale)
    small = pipes & (np.abs(flows) < seed)
    flows = flows.copy()
    flows[small] = seed
    solver.diagnostics.update(seed_kg_s=float(seed), seeded_edges=int(small.sum()))
    return flows


def _pack(solver, pressures, flows):
    flows = flows.copy()
    for e, b in enumerate(solver.network.branches):
        if isinstance(b, Control) and b.closed:
            flows[e] = 0.0
    return np.concatenate((pressures[solver.network.free_indices], flows))


def _known_pressures(solver):
    p = solver.boundary_pressures.copy()
    for b, j in zip(solver.network.branches, solver.network.ends):
        if isinstance(b, Control) and b.kind != "VA":
            p[j] = b.setpoint
    return p


def _pressure_components(solver):
    """Node components joined by pipes and open valves."""
    net = solver.network
    parent = list(range(len(net.nodes)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for b, i, j in zip(net.branches, net.starts, net.ends):
        if isinstance(b, Pipe) or (b.kind == "VA" and b.is_open):
            parent[root(i)] = root(j)
    return np.array([root(i) for i in range(len(net.nodes))])


def default(solver):
    """C0: mean known pressure per pressure-connected component, least-norm flows."""
    net = solver.network
    pressure = _known_pressures(solver)
    components = _pressure_components(solver)
    for c in np.unique(components):
        members = np.flatnonzero(components == c)
        pressure[members[np.isnan(pressure[members])]] = np.nanmean(pressure[members])
    free_matrix = net.incidence[net.free_indices][:, solver.active_edges]
    rhs = -solver.specified_injections[net.free_indices]
    if solver.n_free:
        estimate = lsqr(free_matrix, rhs, atol=1e-12, btol=1e-12,
                        iter_lim=max(100, 4 * solver.n_unknowns))
        if estimate[1] not in (0, 1, 2, 4, 5):
            raise ValueError("Sparse initialization least-squares failed to converge")
        flows = estimate[0]
    else:
        flows = np.zeros(len(solver.active_edges))
    if flows.size:
        seed = max(1e-3, np.max(np.abs(flows)) * 1e-3)
        seeded = (np.abs(flows) < seed) & np.array([
            isinstance(net.branches[e], Pipe) for e in solver.active_edges])
        solver.diagnostics.update(seed_kg_s=float(seed), seeded_edges=int(seeded.sum()))
        flows[seeded] = seed
    all_flows = np.zeros(len(net.branches))
    all_flows[solver.active_edges] = flows
    return np.concatenate((pressure[net.free_indices], all_flows))


def uniform_pressures(solver):
    known = _known_pressures(solver)
    components = _pressure_components(solver)
    out = known.copy()
    for c in np.unique(components):
        members = components == c
        values = known[members]
        top = np.nanmax(values) if np.isfinite(values).any() else solver.pressure_scale
        out[members & np.isnan(known)] = top
    return out


def uniform(solver, seed=True):
    """C1 (seed=True) or C1z (seed=False)."""
    q = np.zeros(len(solver.network.branches))
    if seed:
        q = _flow_seed(solver, q)
    return _pack(solver, uniform_pressures(solver), q)


def routed(solver):
    """C2: shortest-path-tree routing (pipe length weights, controls weight 0)."""
    net = solver.network
    n = len(net.nodes)
    adj = [[] for _ in range(n)]
    for e, (b, i, j) in enumerate(zip(net.branches, net.starts, net.ends)):
        if isinstance(b, Control) and b.closed:
            continue
        w = b.length if isinstance(b, Pipe) else 0.0
        adj[i].append((j, e, w))
        adj[j].append((i, e, w))
    dist = np.full(n, np.inf)
    via = np.full(n, -1)
    heap = []
    for k in net.pressure_indices:
        dist[k] = 0.0
        heap.append((0.0, int(k)))
    heapq.heapify(heap)
    order = []
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        order.append(u)
        for v, e, w in adj[u]:
            if d + w < dist[v]:
                dist[v], via[v] = d + w, e
                heapq.heappush(heap, (d + w, v))
    flows = np.zeros(len(net.branches))
    load = -solver.specified_injections.copy()   # withdrawal > 0
    load[net.pressure_indices] = 0.0
    for u in reversed(order):                    # subtree sums, leaves first
        e = via[u]
        if e < 0:
            continue
        parent = net.starts[e] if net.ends[e] == u else net.ends[e]
        flows[e] = load[u] if net.ends[e] == u else -load[u]
        load[parent] += load[u]
    return _pack(solver, uniform_pressures(solver), _flow_seed(solver, flows))


def _rough_coefficient(pipe, gas, z):
    """K in P_i - P_j = K q|q| [Pa^2 s^2/kg^2], fully rough friction, horizontal pipe."""
    lam = (2 * log10(3.7 * pipe.diameter / max(pipe.roughness, 1e-7))) ** -2
    return lam * z * gas.gas_constant * pipe.temperature * pipe.length / (pipe.diameter * pipe.area ** 2)


def linearized(solver, passes=2):
    """C3: open valves contract their end nodes; CS/CV add one unknown flow and fix
    the outlet squared pressure to the setpoint. Valve flows follow from mass balance."""
    net = solver.network
    gas = solver.physics.gas
    n = len(net.nodes)
    base = default(solver)
    q0 = base[solver.n_free:].copy()
    known = _known_pressures(solver)
    p_ref = float(np.nanmean(solver.boundary_pressures[net.pressure_indices]))
    t_ref = next((b.temperature for b in net.branches if isinstance(b, Pipe)), 288.15)
    try:
        z = float(gas.z(p_ref, t_ref))
    except ValueError:
        z = 1.0
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for b, i, j in zip(net.branches, net.starts, net.ends):
        if isinstance(b, Control) and b.kind == "VA" and b.is_open:
            parent[root(i)] = root(j)
    groups = np.array([root(i) for i in range(n)])
    fixed_group = {groups[k]: solver.boundary_pressures[k] ** 2 for k in net.pressure_indices}
    free_groups = sorted(set(groups.tolist()) - set(fixed_group))
    col = {g: c for c, g in enumerate(free_groups)}
    pipe_edges = [e for e, b in enumerate(net.branches) if isinstance(b, Pipe)]
    ctrl_edges = [e for e, b in enumerate(net.branches) if isinstance(b, Control) and b.kind != "VA"]
    K = np.array([_rough_coefficient(net.branches[e], gas, z) for e in pipe_edges])
    m = len(free_groups)
    size = m + len(ctrl_edges)
    injection = np.zeros(m)
    for k in range(n):
        if groups[k] in col:
            injection[col[groups[k]]] += solver.specified_injections[k]
    P = None
    for _ in range(max(1, passes)):
        qmag = np.abs(q0[pipe_edges])
        qmin = max(1e-3, 1e-3 * (qmag.max() if qmag.size else 1.0))
        cond = 1.0 / (K * np.maximum(qmag, qmin))
        rows, cols, vals = [], [], []
        rhs = np.zeros(size)
        rhs[:m] = -injection
        for c_e, e in zip(cond, pipe_edges):   # q = c (P_i - P_j)
            gi, gj = groups[net.starts[e]], groups[net.ends[e]]
            if gi == gj:
                continue
            for g_row, sign in ((gi, -1.0), (gj, 1.0)):
                if g_row not in col:
                    continue
                r = col[g_row]
                for g_var, s2 in ((gi, 1.0), (gj, -1.0)):
                    coef = sign * c_e * s2
                    if g_var in col:
                        rows.append(r)
                        cols.append(col[g_var])
                        vals.append(coef)
                    else:
                        rhs[r] -= coef * fixed_group[g_var]
        for k, e in enumerate(ctrl_edges):
            gi, gj = groups[net.starts[e]], groups[net.ends[e]]
            cq = m + k
            for g_row, sign in ((gi, -1.0), (gj, 1.0)):
                if g_row in col:
                    rows.append(col[g_row])
                    cols.append(cq)
                    vals.append(sign)
            if gj in col:
                rows.append(cq)
                cols.append(col[gj])
                vals.append(1.0)
                rhs[cq] = net.branches[e].setpoint ** 2
            else:
                rows.append(cq)
                cols.append(cq)
                vals.append(1e-12)
        A = coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsc()
        try:
            x = spsolve(A, rhs)
            ok = np.isfinite(x).all()
        except Exception:  # noqa: BLE001 - a singular system falls back to least squares
            ok = False
        if not ok:
            x = lsqr(A, rhs, atol=1e-12, btol=1e-12, iter_lim=10 * size)[0]
            solver.diagnostics["linearized_lsqr_fallback"] = True
        P = np.array([fixed_group[groups[k]] if groups[k] in fixed_group else x[col[groups[k]]]
                      for k in range(n)])
        q0 = q0.copy()
        for c_e, e in zip(cond, pipe_edges):
            q0[e] = c_e * (P[net.starts[e]] - P[net.ends[e]])
        for k, e in enumerate(ctrl_edges):
            q0[e] = x[m + k]
    pressures = np.sqrt(np.maximum(P, (2 * solver.minimum_pressure) ** 2))
    for k in range(n):   # keep prescribed values exactly
        if np.isfinite(known[k]) and solver.boundary_pressures[k] == solver.boundary_pressures[k]:
            pressures[k] = solver.boundary_pressures[k]
    va = [e for e, b in enumerate(net.branches) if isinstance(b, Control) and b.kind == "VA" and b.is_open]
    flows = q0.copy()
    for e in va:
        flows[e] = 0.0
    if va:
        free = net.free_indices
        residual = (net.incidence @ flows + solver.specified_injections)[free]
        sub = net.incidence[free][:, va]
        flows[va] = lsqr(sub, -residual, atol=1e-12, btol=1e-12, iter_lim=max(100, 4 * len(va)))[0]
    solver.diagnostics.update(linearized_passes=passes, linearized_reference_z=z)
    return _pack(solver, pressures, _flow_seed(solver, flows))
