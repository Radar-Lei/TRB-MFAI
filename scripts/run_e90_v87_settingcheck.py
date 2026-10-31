"""E90 (v87, external review round 59, review item P2): model-setting
sensitivity for the six pre-specified queries away from node 10.

The published six-query results (e86/e87, Table 5 and Appendix C.8) use
theta=1 with ten candidate paths per OD. This experiment re-runs the
SAME setting-C protocol -- the six query demands moving simultaneously
along the same straight line from 0.2 to 1.6 of their published values,
the other 522 positive-demand ODs (including the eight main study ODs)
fixed at published demand, all 528 ODs re-equilibrating jointly at every
quadrature point, 41-point composite trapezoid, implicit sensitivities
-- under three changed model settings:

  (K=20, theta=1)   every truncated OD path set expanded from ten to
                    twenty candidates (all 528 ODs; first ten nested);
  (K=10, theta=0.5)
  (K=10, theta=2)   dispersion changes the fixed-path logit model.

Two robustness readings, per review:
  (a) per setting, recompute the unrestricted (closed form) and
      connected (exact enumeration) boundary-account intervals, and
      record whether each family still straddles zero for the three
      baseline sign-crossing queries;
  (b) freeze the baseline connected extremal sets U_- and U_+ (from the
      published 41-point columns) and evaluate phi_j at those same node
      sets under each setting -- does the SAME boundary keep its sign.

These are different fixed-path logit models; no convergence claim to an
unrestricted SUE is made. The baseline row of the paper keeps the
published e86/e87 values; this experiment does not re-run the baseline
chain, only re-enumerates the connected extrema of the published
columns (gate (i) below).

Gates (all must pass before any number enters the paper):
  (a) selection reproduces e86's stored six ODs exactly;
  (b) K=20 path sets nest the ten-path sets for every truncated OD
      (first ten candidates identical), and all 528 ODs are truncated;
  (c) path fixed-point residual <= 1e-7 at every node, every setting;
  (d) Jacobian conservation max |B J_j - mu_j| <= 1e-8 and background
      circulation max |B J^bg| <= 1e-8, every setting;
  (e) J = J^study + J^bg additivity <= 1e-9, every setting;
  (f) total flow conservation |B F - M_full q| <= 1e-6, every setting;
  (g) implicit-vs-central-FD max column error < 1e-3 at the path
      midpoint and both endpoints, every setting;
  (h) unrestricted interval identity phi_max - phi_min == S_j (1e-9
      relative) and connected span <= unrestricted span;
  (i) the baseline re-enumeration reproduces the published connected
      extrema of e87 (relative 1e-9) and the published origin-cut
      accounts and spans of e86 (relative 1e-9).

Output: experiments/results/e90_v87_settingcheck.json
        experiments/results/e90_v87_settingcheck.npz
        console log experiments/results/e90_console.log
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
from run_e83_v82_validation import EdgeSUE, attribution  # noqa: E402
from run_e80_connected_cuts import build_adj_bits, load_c_enum, enum_c, mask_to_nodes  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402

RESULTS = HERE / 'results'
E86_JSON = RESULTS / 'e86_v84_newod.json'
E87_JSON = RESULTS / 'e87_v85_newod_refine.json'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
OUT = RESULTS / 'e90_v87_settingcheck.json'
NPZ = RESULTS / 'e90_v87_settingcheck.npz'
STUDY_K = 6
GRID = 41
FD_NODES = {0, GRID // 2, GRID - 1}     # endpoints and midpoint
CONDITIONS = (('k20_theta1', 20, 1.0), ('k10_theta0.5', 10, 0.5),
              ('k10_theta2', 10, 2.0))


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
        # gate (h): the closed form re-summarizes Eq. (8)'s identity
        assert abs((phi_max - phi_min)
                   - float(np.abs(b[interior]).sum())) < 1e-9 * max(1.0, abs(phi_s))
        cr = enum_c(lib, adj, n, s, t, BC[:, j:j + 1])
        clo, chi = float(cr['phi_min'][0]), float(cr['phi_max'][0])
        assert chi - clo <= (phi_max - phi_min) * (1 + 1e-9)
        rows.append({
            'phi_s': phi_s,
            'unrestricted': {'min': phi_min, 'max': phi_max,
                             'span': phi_max - phi_min},
            'connected': {'min': clo, 'max': chi, 'span': chi - clo,
                          'n_cuts': int(cr['n_sets'])},
            'crossing': {'unrestricted': bool(phi_min < 0 < phi_max),
                         'connected': bool(clo < 0 < chi)},
            'U_minus': [v - 1 for v in mask_to_nodes(cr['min_mask'][0])],
            'U_plus': [v - 1 for v in mask_to_nodes(cr['max_mask'][0])],
        })
    return rows, BC


def phi_at(BC, j, nodes):
    """Account of a fixed node set (0-based node indices)."""
    return -float(BC[nodes, j].sum())


def main():
    started = time.time()
    net = load_sioux_falls()
    study_ods, trips, sel_info = select_ods(net)
    e86 = json.loads(E86_JSON.read_text())
    e87 = json.loads(E87_JSON.read_text())
    stored = [tuple(od) for od in e86['protocol']['study_ods_1idx']]
    fresh = [(s + 1, t + 1) for s, t in study_ods]
    assert fresh == stored, f'selection drift: {fresh} vs {stored}'   # gate (a)

    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    q_new = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    q0, q1 = 0.2 * q_new, 1.6 * q_new
    dQ = q1 - q0
    ts = np.linspace(0, 1, GRID)

    ps10 = PathStructure(net, all_ods)                     # k_sioux=10 default
    ps20 = PathStructure(net, all_ods, k_sioux=20)
    truncated = sum(n.startswith('kshortest') for n in ps10.notes)
    assert truncated == len(all_ods) == 528, truncated      # gate (b): all truncated
    for note, p10, p20 in zip(ps10.notes, ps10.od_path_idx, ps20.od_path_idx):
        if note.startswith('enumerate'):
            assert ps10.paths[p10[0]:p10[-1] + 1] == ps20.paths[p20[0]:p20[-1] + 1]
        else:
            paths10 = [ps10.paths[k] for k in p10]
            paths20 = [ps20.paths[k] for k in p20]
            assert len(paths10) == 10 and len(paths20) == 20
            assert paths20[:10] == paths10                 # gate (b): nesting

    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    B = net.B
    M_full = net.od_M(all_ods)
    adj, n = build_adj_bits(net)
    lib = load_c_enum()

    # ---- baseline: re-enumerate the published 41-point columns (no solve) ----
    z87 = np.load(E87_NPZ)
    BC_base = z87['BC_41']
    assert np.allclose(z87['ts'][::2], ts)
    # Run the published 41-point columns through the same intervals()
    # machinery as the new conditions, so every gate applies uniformly.
    base_rows, BC_base_check = intervals(B, z87['C_41'], study_ods, adj, n, lib)
    assert np.max(np.abs(BC_base_check - BC_base)) < 1e-9  # same published columns
    for j, (s, t) in enumerate(study_ods):                 # gate (i)
        key = f'{s + 1}->{t + 1}'
        pub86 = e86['per_od'][key]
        pub87 = e87['per_od'][key]['interval_41']
        got = base_rows[j]
        assert abs(got['phi_s'] - pub86['phi_s']) < 1e-9 * max(1, abs(pub86['phi_s']))
        assert abs(got['unrestricted']['span'] - pub86['spread_abs']) \
            < 1e-9 * max(1, abs(pub86['spread_abs']))
        assert abs(got['connected']['min'] - pub87['connected']['min']) \
            < 1e-9 * max(1, abs(pub87['connected']['min']))
        assert abs(got['connected']['max'] - pub87['connected']['max']) \
            < 1e-9 * max(1, abs(pub87['connected']['max']))
    baseline = {}
    for j, (s, t) in enumerate(study_ods):
        key = f'{s + 1}->{t + 1}'
        baseline[key] = base_rows[j]
    crossing_keys = [f'{s + 1}->{t + 1}' for j, (s, t) in enumerate(study_ods)
                     if base_rows[j]['crossing']['connected']]
    assert sorted(crossing_keys) == ['1->15', '1->19', '7->12'], crossing_keys

    # ---- changed settings ----
    cond_out = {}
    saved = {'ts': ts, 'dQ': dQ, 'BC_baseline': BC_base}
    for tag, k_paths, theta in CONDITIONS:
        t0 = time.time()
        ps = ps20 if k_paths == 20 else ps10
        solver = EdgeSUE(ps, study_k=STUDY_K)
        solver.theta = theta
        Fs, gs, Jall, Jss, Jbs = [], [], [], [], []
        max_jcons = max_bgcons = max_add = 0.0
        fd_errors = {}
        warm = None
        for i, t in enumerate(ts):
            q = np.concatenate([q0 + t * dQ, q_bg])
            F, h, p, res = solver.solve(q, warm)
            warm = F
            J, Js, Jb = solver.sensitivity(F, h, p, q)
            max_jcons = max(max_jcons, float(np.max(np.abs(B @ J - net.od_M(study_ods)))))
            max_bgcons = max(max_bgcons, float(np.max(np.abs(B @ Jb))))
            max_add = max(max_add, float(np.max(np.abs(J - Js - Jb))))
            if i in FD_NODES:                             # gate (g)
                fd = np.empty_like(J)
                for j in range(STUDY_K):
                    step = .0025 * (dQ[j] / 1.4)
                    qp = q.copy(); qm = q.copy()
                    qp[j] += step; qm[j] -= step
                    fd[:, j] = (solver.solve(qp, F)[0] - solver.solve(qm, F)[0]) / (2 * step)
                fd_errors[f'node{i}_t{t:.4f}'] = float(np.max(
                    np.linalg.norm(fd - J, axis=0) / np.linalg.norm(J, axis=0)))
                assert fd_errors[f'node{i}_t{t:.4f}'] < 1e-3, (tag, fd_errors)
            Fs.append(F); gs.append(T.grad(F))
            Jall.append(J); Jss.append(Js); Jbs.append(Jb)
        F = np.stack(Fs); g = np.stack(gs)
        J = np.stack(Jall); Js = np.stack(Jss); Jb = np.stack(Jbs)
        assert solver.max_residual <= 1e-7, (tag, solver.max_residual)      # gate (c)
        assert max_jcons <= 1e-8 and max_bgcons <= 1e-8, (tag, max_jcons, max_bgcons)
        assert max_add <= 1e-9, (tag, max_add)                              # gate (e)
        cons = max(float(np.max(np.abs(B @ F[i] - M_full @ np.concatenate(
            [q0 + t * dQ, q_bg])))) for i, t in enumerate(ts))
        assert cons <= 1e-6, (tag, cons)                                    # gate (f)

        C = attribution(g, J, ts, dQ)
        Cs = attribution(g, Js, ts, dQ)
        Cb = attribution(g, Jb, ts, dQ)
        dT = T.value(F[-1]) - T.value(F[0])
        completeness = float(abs(C.sum() - dT) / abs(dT))
        rows, BC = intervals(B, C, study_ods, adj, n, lib)
        per_od = {}
        for j, (s, t) in enumerate(study_ods):
            key = f'{s + 1}->{t + 1}'
            U_minus = baseline[key]['U_minus']
            U_plus = baseline[key]['U_plus']
            phi_m = phi_at(BC, j, U_minus)
            phi_p = phi_at(BC, j, U_plus)
            # compare signs against the baseline extrema each set attained
            sign_m0 = baseline[key]['connected']['min'] < 0
            sign_p0 = baseline[key]['connected']['max'] > 0
            per_od[key] = {
                'C_j': float(C[:, j].sum()),
                **rows[j],
                'phi_at_baseline': {
                    'U_minus': phi_m, 'sign_minus': '-' if phi_m < 0 else '+',
                    'U_plus': phi_p, 'sign_plus': '-' if phi_p < 0 else '+',
                    'keeps_baseline_sign': bool(((phi_m < 0) == sign_m0)
                                                 and ((phi_p > 0) == sign_p0)),
                },
            }
        cond_out[tag] = {
            'theta': theta, 'paths_per_od': k_paths,
            'gates': {
                'path_fixed_point_max_abs': solver.max_residual,
                'J_conservation_max_abs': max_jcons,
                'background_circulation_max_abs': max_bgcons,
                'J_additivity_max_abs': max_add,
                'flow_conservation_max_abs': cons,
                'FD_max_column_relative_l2': fd_errors,
                'completeness_relative': completeness,
                'solves': solver.solves,
            },
            'per_od': per_od,
            'elapsed_seconds': time.time() - t0,
        }
        for name, arr in (('C', C), ('C_study', Cs), ('C_background', Cb),
                          ('BC', BC), ('F', F)):
            saved[f'{tag}_{name}'] = arr
        print(f'{tag}: gates {cond_out[tag]["gates"]}', flush=True)
        print(f'{tag}: crossing(connected) '
              f'{[k for k in crossing_keys if cond_out[tag]["per_od"][k]["crossing"]["connected"]]}',
              flush=True)

    out = {
        'protocol': {
            'purpose': 'six-query model-setting sensitivity (review round 59, '
                       'item P2): same setting-C protocol, changed path count / '
                       'dispersion; no unrestricted-SUE convergence claim',
            'grid': GRID, 'fd_check_nodes': sorted(FD_NODES),
            'conditions': {tag: {'paths_per_od': k, 'theta': th}
                           for tag, k, th in CONDITIONS},
            'baseline': 'published e86/e87 41-point columns (not re-solved); '
                        'connected extrema re-enumerated (gate i)',
            'integration': 'uniform composite trapezoid',
        },
        'baseline': baseline,
        'crossing_queries': crossing_keys,
        'conditions': cond_out,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(NPZ, **saved)
    print(f'Completed e90 in {out["elapsed_seconds"]:.1f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
