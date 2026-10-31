"""E87 (v85, external review round 55, review items 3-4): quadrature
refinement and boundary-account intervals for the six pre-specified
queries away from node 10 (e86).

Part 1 (refinement, review item 4): re-run the e86 setting-C protocol
unchanged -- same six query ODs, same demand path 0.2->1.6, same 522
background ODs at published demand, all 528 ODs re-equilibrating at
every quadrature point, theta=1, ten paths per OD, implicit
sensitivities -- with the integration grid refined from 41 to 81
nested uniform points.  Per query, report full attribution-column
relative-l2 change (81-grid column in the denominator, the Table C.1
convention), origin-cut account, absolute unrestricted range S_j,
connected range, and both normalized ranges, with changes in absolute
percentage points.  The published Table 5 keeps its 41-point values;
this check states their significant digits.

Central finite-difference derivative cross-checks (implicit vs.\ re-solved
central differences) run at the path midpoint AND at both endpoints,
per review item 4 ("路径两个端点补充同类检查").

Part 2 (intervals, review item 3): boundary-account attainable
intervals from the published 41-point columns.  With b_j = B C_j,
phi_min = phi_j({s_j}) - sum_v (b_jv)_+ and phi_max = phi_j({s_j}) +
sum_v (-b_jv)_+ over interior nodes (an exact re-summary of the
identity behind Eq. (8), no new equilibrium solve).  The connected
interval comes from the existing exact enumeration.  Both intervals are
also computed on the 81-point chain as a refinement check.  The paper
reports unrestricted and connected intervals for the three queries
whose normalized range exceeds 1,000%.

Gates (all must pass before any number enters the paper):
  (a) selection reproduces e86's stored six ODs exactly;
  (b) even-index nodes of the 81-point chain reproduce e86's 41-point
      flows to <1e-6 relative l2 (same converged equilibrium chain);
  (c) path fixed-point residual <= 1e-7 at every node;
  (d) Jacobian conservation max |B J_j - mu_j| <= 1e-8 and background
      circulation max |B J^bg| <= 1e-8 at every node;
  (e) J = J^study + J^bg additivity <= 1e-9;
  (f) flow conservation |B F - M_full q| <= 1e-6 at every node;
  (g) implicit-vs-central-FD max column error < 1e-3 at midpoint and
      both endpoints;
  (h) unrestricted interval identity: phi_max - phi_min == S_j to
      1e-9 relative, and connected span <= unrestricted span.

Output: experiments/results/e87_v85_newod_refine.json
        experiments/results/e87_v85_newod_refine.npz
        console log experiments/results/e87_console.log
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nets import load_sioux_falls, load_sioux_falls_trips  # noqa: E402
from sue import PathStructure  # noqa: E402
from run_e9_transport import TSTTT  # noqa: E402
from run_e83_v82_validation import EdgeSUE, attribution, cut_metrics  # noqa: E402
from run_e80_connected_cuts import (build_adj_bits, load_c_enum, enum_c,  # noqa: E402
                                    mask_to_nodes, connected)
from run_e86_v84_newod import select_ods  # noqa: E402

RESULTS = HERE / 'results'
E86_NPZ = RESULTS / 'e86_v84_newod.npz'
E86_JSON = RESULTS / 'e86_v84_newod.json'
OUT = RESULTS / 'e87_v85_newod_refine.json'
NPZ = RESULTS / 'e87_v85_newod_refine.npz'
STUDY_K = 6
GRID = 81
FD_NODES = {0, GRID // 2, GRID - 1}     # endpoints and midpoint


def intervals(B, C, ods, adj, n, lib):
    """Unrestricted (closed form) and connected (enumerated) intervals."""
    BC = B @ C
    fullmask = (1 << n) - 1
    rows = []
    for j, (s, t) in enumerate(ods):
        b = BC[:, j]
        phi_s = -float(b[s])
        interior = [v for v in range(n) if v not in (s, t)]
        pos = sum(max(float(b[v]), 0.0) for v in interior)
        neg = sum(max(-float(b[v]), 0.0) for v in interior)
        phi_min, phi_max = phi_s - pos, phi_s + neg
        cr = enum_c(lib, adj, n, s, t, BC[:, j:j + 1])
        clo, chi = float(cr['phi_min'][0]), float(cr['phi_max'][0])
        for mask, val in ((cr['min_mask'][0], clo), (cr['max_mask'][0], chi)):
            assert connected(adj, mask) and connected(adj, fullmask & ~mask)
            nd = np.array(mask_to_nodes(mask)) - 1
            assert s in nd and t not in nd
            assert abs(-float(BC[nd, j].sum()) - val) < 1e-6
        rows.append({
            'phi_s': phi_s,
            'unrestricted': {'min': phi_min, 'max': phi_max,
                             'span': phi_max - phi_min},
            'connected': {'min': clo, 'max': chi, 'span': chi - clo,
                          'n_cuts': int(cr['n_sets'])},
        })
    return rows, BC


def main():
    started = time.time()
    net = load_sioux_falls()
    study_ods, trips, sel_info = select_ods(net)
    e86 = json.loads(E86_JSON.read_text())
    stored = [tuple(od) for od in e86['protocol']['study_ods_1idx']]
    fresh = [(s + 1, t + 1) for s, t in study_ods]
    assert fresh == stored, f'selection drift: {fresh} vs {stored}'

    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
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
    fd_errors = {}
    warm = None
    for i, t in enumerate(ts):
        q = np.concatenate([q0 + t * dQ, q_bg])
        F, h, p, res = solver.solve(q, warm)
        warm = F
        J, Js, Jb = solver.sensitivity(F, h, p, q)
        max_jcons = max(max_jcons, float(np.max(np.abs(B @ J - M_study))))
        max_bgcons = max(max_bgcons, float(np.max(np.abs(B @ Jb))))
        max_add = max(max_add, float(np.max(np.abs(J - Js - Jb))))
        if i in FD_NODES:                     # gate (g): midpoint + endpoints
            fd = np.empty_like(J)
            for j in range(STUDY_K):
                step = .0025 * (dQ[j] / 1.4)
                qp = q.copy(); qm = q.copy()
                qp[j] += step; qm[j] -= step
                fd[:, j] = (solver.solve(qp, F)[0] - solver.solve(qm, F)[0]) / (2 * step)
            fd_errors[f'node{i}_t{t:.4f}'] = float(np.max(
                np.linalg.norm(fd - J, axis=0) / np.linalg.norm(J, axis=0)))
            assert fd_errors[f'node{i}_t{t:.4f}'] < 1e-3, fd_errors
        Fs.append(F); gs.append(T.grad(F))
        Jall.append(J); Jss.append(Js); Jbs.append(Jb)
        print(f'  node {i + 1}/{GRID} t={t:.3f} res={res:.2e} '
              f'elapsed={time.time() - started:.1f}s', flush=True)
    F = np.stack(Fs); g = np.stack(gs)
    J = np.stack(Jall); Js = np.stack(Jss); Jb = np.stack(Jbs)

    # gates (b)-(f)
    z86 = np.load(E86_NPZ)
    even = F[::2]
    chain_gap = float(np.max(np.linalg.norm(even - z86['F'], axis=1)
                             / np.linalg.norm(z86['F'], axis=1)))
    assert chain_gap < 1e-6, f'81-chain does not reproduce e86 flows: {chain_gap}'
    assert solver.max_residual <= 1e-7, solver.max_residual
    assert max_jcons <= 1e-8, max_jcons
    assert max_bgcons <= 1e-8, max_bgcons
    assert max_add <= 1e-9, max_add
    cons = max(float(np.max(np.abs(B @ F[i] - M_full @ np.concatenate(
        [q0 + t * dQ, q_bg])))) for i, t in enumerate(ts))
    assert cons <= 1e-6, cons

    C81 = attribution(g, J, ts, dQ)
    dT = T.value(F[-1]) - T.value(F[0])
    completeness81 = float(abs(C81.sum() - dT) / abs(dT))
    C41 = z86['C']
    ts41 = z86['ts']
    assert np.allclose(ts41, ts[::2])

    adj, n = build_adj_bits(net)
    lib = load_c_enum()
    rows41, BC41 = intervals(B, C41, study_ods, adj, n, lib)
    rows81, BC81 = intervals(B, C81, study_ods, adj, n, lib)

    per_od = {}
    for j, (s, t) in enumerate(study_ods):
        key = f'{s + 1}->{t + 1}'
        r41, r81 = rows41[j], rows81[j]
        assert abs((r41['unrestricted']['span']) -
                   float(np.abs(BC41[[v for v in range(n) if v not in (s, t)], j]).sum())) < 1e-9
        col = float(np.linalg.norm(C81[:, j] - C41[:, j]) / np.linalg.norm(C81[:, j]))
        per_od[key] = {
            'column_41_vs_81_relative_l2': col,
            'phi_s_41': r41['phi_s'], 'phi_s_81': r81['phi_s'],
            'phi_s_change_pct': 100 * abs(r81['phi_s'] - r41['phi_s']) / abs(r81['phi_s']),
            'S_41': r41['unrestricted']['span'], 'S_81': r81['unrestricted']['span'],
            'S_change_pct': 100 * abs(r81['unrestricted']['span'] - r41['unrestricted']['span'])
                            / r81['unrestricted']['span'],
            'range_pct_41': 100 * r41['unrestricted']['span'] / abs(r41['phi_s']),
            'range_pct_81': 100 * r81['unrestricted']['span'] / abs(r81['phi_s']),
            'conn_range_pct_41': 100 * r41['connected']['span'] / abs(r41['phi_s']),
            'conn_range_pct_81': 100 * r81['connected']['span'] / abs(r81['phi_s']),
            'range_change_pp': abs(100 * r81['unrestricted']['span'] / abs(r81['phi_s'])
                                   - 100 * r41['unrestricted']['span'] / abs(r41['phi_s'])),
            'conn_range_change_pp': abs(100 * r81['connected']['span'] / abs(r81['phi_s'])
                                        - 100 * r41['connected']['span'] / abs(r41['phi_s'])),
            'interval_41': r41, 'interval_81': r81,
        }
        print(key, {k: round(v, 6) for k, v in per_od[key].items()
                    if isinstance(v, float)}, flush=True)

    # cross-check the closed-form 41-grid summary against the published numbers
    for j, (s, t) in enumerate(study_ods):
        key = f'{s + 1}->{t + 1}'
        pub = e86['per_od'][key]
        got = per_od[key]['interval_41']
        assert abs(got['phi_s'] - pub['phi_s']) < 1e-6 * max(1, abs(pub['phi_s']))
        assert abs(got['unrestricted']['span'] - pub['spread_abs']) \
            < 1e-9 * max(1, abs(pub['spread_abs']))
        assert abs(got['connected']['span'] - pub['connected_range_abs']) \
            < 1e-9 * max(1, abs(pub['connected_range_abs']))

    out = {
        'protocol': {
            'purpose': 'e86 refinement 41->81 nested nodes + endpoint/midpoint '
                       'FD checks + boundary-account intervals (no setting change)',
            'theta': 1.0, 'paths_per_od': 10, 'nodes_41': 41, 'nodes_81': GRID,
            'fd_check_nodes': sorted(FD_NODES),
            'integration': 'uniform composite trapezoid; 81-grid nests the 41-grid',
        },
        'gates': {
            'selection_matches_e86': True,
            'even_node_flow_vs_e86_max_relative_l2': chain_gap,
            'path_fixed_point_max_abs': solver.max_residual,
            'J_conservation_max_abs': max_jcons,
            'background_circulation_max_abs': max_bgcons,
            'J_additivity_max_abs': max_add,
            'flow_conservation_max_abs': cons,
            'FD_max_column_relative_l2': fd_errors,
            'completeness_relative_81': completeness81,
        },
        'per_od': per_od,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(NPZ, ts=ts, F=F, gT=g, J=J, J_study=Js, J_background=Jb,
                        C_81=C81, C_41=C41, BC_81=BC81, BC_41=BC41, dQ=dQ)
    print(f'Completed e87 in {out["elapsed_seconds"]:.1f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
