"""E86 (v84, external review round 53, review item: spatial coverage):
six pre-specified OD queries that do NOT involve node 10, full-background
direct SUE (setting C protocol).

Motivation.  The eight published queries all involve node 10; the paper
says so and calls the comparison local.  This experiment tests whether
the boundary-variation phenomenon extends beyond that local set,
without retraining any surrogate.

Pre-specified selection rule (uses ONLY published demands and free-flow
shortest-path times; no attribution result participates):
  1. candidates = all positive-demand OD pairs (o, t) with o != 10 and
     t != 10 (node IDs one-based here);
  2. rank candidates by published demand ascending and split into three
     equal-count terciles (lowest, middle, highest demand);
  3. within each tercile select the minimum- and the maximum-free-flow-
     shortest-path-time OD (ties by ascending (o, t)).
=> six ODs spanning three demand levels and, within each, the shortest
and longest trip distances.

Setting C protocol (unchanged from e83/e84): the six query demands move
from 0.2 q_base to 1.6 q_base along the straight line; every other OD
-- including the eight published queries -- stays at its published
demand; ALL 528 ODs jointly re-equilibrate at every quadrature point;
TSTT and its gradient use total flows; ten candidate paths per OD;
theta=1; 41 quadrature points; implicit sensitivities.  Reported per
query: absolute unrestricted range S_j, origin-cut account, both
normalized ranges (unrestricted and connected).

Gates (all must pass before any number enters the paper):
  (a) selection: six distinct ODs, none involving node 10, demands and
      distances spanning all three terciles (checked in-output);
  (b) path fixed-point residual <= 1e-7 at every node;
  (c) Jacobian conservation max |B J_j - mu_j| <= 1e-8 and background
      circulation max |B J^bg| <= 1e-8 at every node;
  (d) J = J^study + J^bg additivity <= 1e-9;
  (e) midpoint implicit-vs-central-FD max column error < 1e-3;
  (f) total conservation |B F - M_full q_full| <= 1e-6 at every node;
  (g) completeness |sum_j C_j - dT| / |dT| reported.

This experiment supports the claim that boundary variation is not
confined to the original eight queries.  It does NOT support an A/C
paired claim about background enlarging near-balanced accounts, which
would require matching zero-background runs.

Output: experiments/results/e86_v84_newod.json
        experiments/results/e86_v84_newod.npz
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import solve

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nets import load_sioux_falls, load_sioux_falls_trips  # noqa: E402
from sue import PathStructure  # noqa: E402
from run_e9_transport import TSTTT  # noqa: E402
from run_e83_v82_validation import EdgeSUE, attribution, cut_metrics  # noqa: E402
from run_e80_connected_cuts import (build_adj_bits, load_c_enum, enum_c,  # noqa: E402
                                    mask_to_nodes, connected)

RESULTS = HERE / 'results'
OUT = RESULTS / 'e86_v84_newod.json'
NPZ = RESULTS / 'e86_v84_newod.npz'
NODE10 = 9          # 0-based
STUDY_K = 6
GRID = 41


def select_ods(net):
    """The pre-specified tercile x distance rule."""
    trips = load_sioux_falls_trips()
    cands = sorted((od for od, d in trips.items()
                    if d > 0 and NODE10 not in od))
    demand = np.array([trips[od] for od in cands], dtype=float)
    # directed free-flow shortest-path times
    rows = np.array([u for u, v in net.edges])
    cols = np.array([v for u, v in net.edges])
    W = csr_matrix((net.free_flow_time, (rows, cols)), shape=(net.n, net.n))
    D = shortest_path(W, directed=True)
    dist = np.array([D[o, t] for o, t in cands])
    order = np.argsort(demand, kind='stable')
    terciles = np.array_split(order, 3)
    picked = []
    for k, idx in enumerate(terciles):
        sub = list(idx)
        near = min(sub, key=lambda i: (dist[i], cands[i]))
        far = max(sub, key=lambda i: (dist[i], -cands[i][0], -cands[i][1]))
        picked.append((k, 'short', cands[near], demand[near], dist[near]))
        picked.append((k, 'long', cands[far], demand[far], dist[far]))
    ods = [rec[2] for rec in picked]
    assert len(set(ods)) == STUDY_K, 'selection produced duplicates'
    assert all(NODE10 not in od for od in ods), 'node 10 leaked into selection'
    lo = [demand[i] for i in terciles[0]]
    hi = [demand[i] for i in terciles[2]]
    info = {
        'rule': 'positive-demand ODs without node 10; demand terciles; '
                'min/max free-flow shortest-path time within each tercile; '
                'ties by ascending (origin, destination)',
        'n_candidates': len(cands),
        'candidate_demand_min_max': [float(demand.min()), float(demand.max())],
        'tercile_demand_ranges': [
            [float(demand[terciles[k]].min()), float(demand[terciles[k]].max())]
            for k in range(3)],
        'selection': [
            {'tercile': rec[0], 'distance_class': rec[1], 'od': [rec[2][0] + 1, rec[2][1] + 1],
             'published_demand_veh_h': float(rec[3]),
             'free_flow_sp_time': float(rec[4])}
            for rec in picked],
        'demands_span_terciles': bool(min(hi) > max(lo)),
    }
    return ods, trips, info


def main():
    started = time.time()
    net = load_sioux_falls()
    study_ods, trips, sel_info = select_ods(net)
    print('selection:', json.dumps(sel_info['selection'], indent=None))
    bg_ods = [od for od, d in trips.items()
              if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    n_pos = sum(1 for d in trips.values() if d > 0)
    assert len(all_ods) == n_pos == 528, \
        f'OD count {len(all_ods)} vs positive {n_pos}'

    q_new = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    q0, q1 = 0.2 * q_new, 1.6 * q_new
    dQ = q1 - q0

    psfull = PathStructure(net, all_ods)
    solver = EdgeSUE(psfull, study_k=STUDY_K)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    B = net.B
    M_study = net.od_M(study_ods)
    M_full = net.od_M(all_ods)

    ts = np.linspace(0, 1, GRID)
    Fs, gs, Jall, Jss, Jbs = [], [], [], [], []
    max_jcons = max_bgcons = max_add = 0.0
    mid_fd_error = None
    warm = None
    for i, t in enumerate(ts):
        q = np.concatenate([q0 + t * dQ, q_bg])
        F, h, p, res = solver.solve(q, warm)
        warm = F
        J, Js, Jb = solver.sensitivity(F, h, p, q)
        max_jcons = max(max_jcons, float(np.max(np.abs(B @ J - M_study))))
        max_bgcons = max(max_bgcons, float(np.max(np.abs(B @ Jb))))
        max_add = max(max_add, float(np.max(np.abs(J - Js - Jb))))
        if i == GRID // 2:                     # gate (e): midpoint FD check
            fd = np.empty_like(J)
            for j in range(STUDY_K):
                step = .0025 * (dQ[j] / 1.4)
                qp = q.copy(); qm = q.copy()
                qp[j] += step; qm[j] -= step
                fd[:, j] = (solver.solve(qp, F)[0] - solver.solve(qm, F)[0]) / (2 * step)
            mid_fd_error = float(np.max(
                np.linalg.norm(fd - J, axis=0) / np.linalg.norm(J, axis=0)))
            assert mid_fd_error < 1e-3, f'implicit/FD mismatch {mid_fd_error}'
        Fs.append(F); gs.append(T.grad(F))
        Jall.append(J); Jss.append(Js); Jbs.append(Jb)
        print(f'  node {i + 1}/{GRID} t={t:.3f} res={res:.2e} '
              f'elapsed={time.time() - started:.1f}s', flush=True)
    F = np.stack(Fs); g = np.stack(gs)
    J = np.stack(Jall); Js = np.stack(Jss); Jb = np.stack(Jbs)

    # gates (b)-(f)
    assert solver.max_residual <= 1e-7, solver.max_residual
    assert max_jcons <= 1e-8, max_jcons
    assert max_bgcons <= 1e-8, max_bgcons
    assert max_add <= 1e-9, max_add
    cons = max(float(np.max(np.abs(B @ F[i] - M_full @ np.concatenate(
        [q0 + t * dQ, q_bg])))) for i, t in enumerate(ts))
    assert cons <= 1e-6, cons

    C = attribution(g, J, ts, dQ)
    Cs = attribution(g, Js, ts, dQ)
    Cb = attribution(g, Jb, ts, dQ)
    dT = T.value(F[-1]) - T.value(F[0])
    completeness = float(abs(C.sum() - dT) / abs(dT))

    adj, n = build_adj_bits(net)
    lib = load_c_enum()
    fullmask = (1 << n) - 1
    BC = B @ C
    rows = cut_metrics(B, C, study_ods)
    per_od = {}
    for j, (s, t) in enumerate(study_ods):
        key = f'{s + 1}->{t + 1}'
        row = dict(rows[j])
        cr = enum_c(lib, adj, n, s, t, BC[:, j:j + 1])
        lo, hi = float(cr['phi_min'][0]), float(cr['phi_max'][0])
        for mask, val in ((cr['min_mask'][0], lo), (cr['max_mask'][0], hi)):
            assert connected(adj, mask) and connected(adj, fullmask & ~mask)
            nd = np.array(mask_to_nodes(mask)) - 1
            assert s in nd and t not in nd
            assert abs(-float(BC[nd, j].sum()) - val) < 1e-6
        span = hi - lo
        assert span <= row['spread_abs'] * (1 + 1e-9)
        row['connected_range_abs'] = span
        row['connected_range_pct'] = 100 * span / abs(row['phi_s'])
        row['connected_fraction'] = span / row['spread_abs']
        row['n_connected_cuts'] = cr['n_sets']
        interior = [v for v in range(n) if v not in (s, t)]
        major = sorted(interior, key=lambda v: abs(BC[v, j]), reverse=True)[:3]
        row['major_nodes_1idx'] = [v + 1 for v in major]
        row['signed_node_decomposition'] = [
            {'node': v + 1, 'total': float(BC[v, j]),
             'study': float((B @ Cs)[v, j]), 'background': float((B @ Cb)[v, j])}
            for v in major]
        per_od[key] = row
        print(key, {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in row.items() if k in
                    ('phi_s', 'spread_abs', 'spread_pct',
                     'connected_range_pct')}, flush=True)

    out = {
        'protocol': {
            'setting': 'C: full background, all 528 ODs re-equilibrate at '
                       'every node; six pre-specified queries not involving '
                       'node 10 scale 0.2->1.6 of published demand',
            'theta': 1.0, 'paths_per_od': 10, 'nodes': GRID,
            'integration': 'uniform composite trapezoid',
            'study_ods_1idx': [[s + 1, t + 1] for s, t in study_ods],
            'background_ods': len(bg_ods),
            'claims_supported': 'boundary variation is not confined to the '
                                'eight published queries; NOT an A/C paired '
                                'background-enlargement claim',
        },
        'selection': sel_info,
        'gates': {
            'path_fixed_point_max_abs': solver.max_residual,
            'J_conservation_max_abs': max_jcons,
            'background_circulation_max_abs': max_bgcons,
            'J_additivity_max_abs': max_add,
            'midpoint_FD_max_column_relative_l2': mid_fd_error,
            'flow_conservation_max_abs': cons,
            'completeness_relative': completeness,
        },
        'per_od': per_od,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(NPZ, ts=ts, F=F, gT=g, J=J, J_study=Js,
                        J_background=Jb, C=C, C_study=Cs, C_background=Cb,
                        BC=BC, dQ=dQ)
    print(f'Completed e86 in {out["elapsed_seconds"]:.1f}s -> {OUT}',
          flush=True)


if __name__ == '__main__':
    main()
