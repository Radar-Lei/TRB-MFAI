"""E95 (v90, external review round 62, 87/100, item III-4 "the one new
experiment most worth adding"): single-OD demand-change check for the
six-query sign crossings.

Motivation.  The published six-query protocol moves ALL six query
demands simultaneously along a common straight line (each 0.2 -> 1.6 of
its published value) while the other 522 positive-demand ODs stay at
published demand.  Its OD attributions therefore answer to one specific
JOINT demand change; a reader may ask whether the three sign crossings
-- (1,19), (1,15), (7,12) admit connected boundary accounts of both
signs while C_j > 0 -- depend on the other five query demands moving at
the same time.

Protocol (pre-specified, no re-selection of base points).  For each of
the three sign-crossing queries plus the positive control (6,8):
  * q0 = published demands for all 528 ODs EXCEPT the focal OD, which
    sits at 0.2 x published;  q1 = same with the focal OD at 1.6 x
    published;  all other 527 demands (including the other five query
    ODs and the eight main study ODs) fixed at published values;
  * 41-point uniform trapezoid chain q(t) = q0 + t (q1 - q0),
    all 528 ODs re-equilibrated at every quadrature point, theta = 1,
    ten candidate paths per OD (the e86/e87 setting-C protocol, one
    moving coordinate instead of six);
  * implicit sensitivities, integrated attribution column for the focal
    OD.  Because only one OD moves, completeness makes the scalar
    contribution exactly the endpoint TSTT change,
    C_j = T(f(q^1)) - T(f(q^0)), with no apportionment across
    simultaneously moving ODs.

Reported per query (the review asks for four items):
  (1) C_j (column sum) and the endpoint TSTT difference it must equal;
  (2) the attainable connected-boundary interval [min, max];
  (3) phi_j at the JOINT-scenario extremal boundaries U_- and U_+
      (the published 41-point columns' connected argmin/argmax sets;
      for the three crossing queries these are the e91 node lists)
      evaluated on the new single-OD columns -- distinguishing "the
      original boundaries keep their signs" from "a sign crossing
      still exists somewhere in the connected family";
  (4) the unrestricted closed-form interval for context.

Gates (all must pass before any number enters the paper):
  (a) solver path fixed-point residual <= 1e-7 at every state of every
      chain;  flow conservation |B F - M_full q| <= 1e-6;
  (b) Jacobian conservation max |B J - mu_j| <= 1e-8 over the six
      study columns at every state (the focal column is the one used);
  (c) implicit-vs-re-solved-central-FD column check at both endpoints
      and the midpoint of each chain < 1e-3 (relative l2);
  (d) completeness: |C_j - (T(f(q1)) - T(f(q0)))| / |dT| <= 1e-4
      (trapezoid level, matching the published protocol's gap);
  (e) connected-family counts equal the published e86 counts (same
      (s, t), fixed topology);  attaining masks are valid connected
      separating sets recomputed by direct summation;
  (f) for the three crossing queries, the joint-scenario argmin/argmax
      masks from the frozen e87 41-point columns reproduce the e91
      U_-/U_+ node lists and their published phi values (< 1e-6).

NOT claimed: agreement with the joint scenario.  If a crossing
persists, the phenomenon does not hinge on the joint perturbation; if
one disappears, that is itself the scenario-dependence finding.  This
is a scope check on the interpretation, not a condition for any
theorem, and the base points above were fixed before running.

Output: experiments/results/e95_v90_singleod.json
        experiments/results/e95_v90_singleod.npz
        console log experiments/results/e95_console.log
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
from nets import load_sioux_falls  # noqa: E402
from sue import PathStructure  # noqa: E402
from run_e9_transport import TSTTT  # noqa: E402
from run_e83_v82_validation import EdgeSUE, attribution  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e80_connected_cuts import (build_adj_bits, load_c_enum, enum_c,  # noqa: E402
                                    mask_to_nodes, connected)

RESULTS = HERE / 'results'
E86_JSON = RESULTS / 'e86_v84_newod.json'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E91_JSON = RESULTS / 'e91_v87_negative_mechanism.json'
OUT = RESULTS / 'e95_v90_singleod.json'
NPZ = RESULTS / 'e95_v90_singleod.npz'

STUDY_K = 6
GRID = 41
FD_NODES = {0, GRID // 2, GRID - 1}      # endpoints and midpoint
FOCAL = ('1->19', '1->15', '7->12', '6->8')   # three crossings + control
CROSSING = ('1->19', '1->15', '7->12')


def joint_extremal_sets(lib, adj, n, C41, study_ods, e91, B):
    """Connected argmin/argmax masks of the published joint 41-point
    columns, gated against the e91 node lists for the crossing queries.

    Returns {key: {'U_minus': (mask, nodes0idx), 'U_plus': ...}}.
    """
    full = (1 << n) - 1
    BC41 = B @ C41
    out = {}
    for key in FOCAL:
        j = [f'{s + 1}->{t + 1}' for s, t in study_ods].index(key)
        s, t = study_ods[j]
        col = BC41[:, j:j + 1]
        cr = enum_c(lib, adj, n, s, t, col)
        for tag, field in (('U_minus', 'min_mask'), ('U_plus', 'max_mask')):
            mask = int(cr[field][0])
            nodes0 = np.array(mask_to_nodes(mask)) - 1
            assert connected(adj, mask) and connected(adj, full & ~mask)
            assert s in nodes0 and t not in nodes0
            val = float(cr['phi_min'][0] if tag == 'U_minus'
                        else cr['phi_max'][0])
            assert abs(-float(col[nodes0, 0].sum()) - val) < 1e-6, \
                (key, tag, val)
            if key in CROSSING:
                m = e91['mechanism'][key]
                pub_nodes = np.array(m[f'{tag}_nodes_1idx']) - 1
                assert np.array_equal(np.sort(nodes0), np.sort(pub_nodes)), \
                    (key, tag, nodes0, pub_nodes)
                pub_phi = m['phi_U_minus'] if tag == 'U_minus' \
                    else m['phi_U_plus']
                assert abs(val - pub_phi) < 1e-6, (key, tag, val, pub_phi)
            out.setdefault(key, {})[tag] = (mask, nodes0, val)
    return out


def run_focal(key, net, ps, solver, T, B, M_full, M_study, q_pub6, q_bg,
              study_ods, adj, n, lib, joint_sets):
    """One single-OD chain: solve, sensitize, integrate, enumerate."""
    started = time.time()
    keys6 = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    j = keys6.index(key)
    s, t = study_ods[j]
    pub_j = q_pub6[j]
    dQ = np.zeros(STUDY_K)
    dQ[j] = 1.4 * pub_j                     # 0.2 -> 1.6 of published

    ts = np.linspace(0.0, 1.0, GRID)
    Fs, gs, Js = [], [], []
    max_res = 0.0
    max_jcons = 0.0
    max_flowcons = 0.0
    fd_errors = {}
    warm = None
    for i, lam in enumerate(ts):
        q6 = q_pub6.copy()
        q6[j] = 0.2 * pub_j + lam * dQ[j]
        q = np.concatenate([q6, q_bg])
        F, h, p, res = solver.solve(q, warm)
        warm = F
        max_res = max(max_res, res)
        J, Js_, Jb = solver.sensitivity(F, h, p, q)
        max_jcons = max(max_jcons,
                        float(np.max(np.abs(B @ J - M_study))))
        max_flowcons = max(max_flowcons, float(np.max(np.abs(
            B @ F - M_full @ q))))
        if i in FD_NODES:                   # gate (c)
            fd = np.empty_like(J)
            step = .0025 * (dQ[j] / 1.4)
            qp = q.copy(); qm = q.copy()
            qp[j] += step; qm[j] -= step
            fd[:, j] = (solver.solve(qp, F)[0]
                        - solver.solve(qm, F)[0]) / (2 * step)
            rel = float(np.linalg.norm(fd[:, j] - J[:, j])
                       / np.linalg.norm(J[:, j]))
            fd_errors[f'node{i}_t{lam:.4f}'] = rel
            assert rel < 1e-3, (key, i, rel)
        Fs.append(F); gs.append(T.grad(F)); Js.append(J)
    F = np.stack(Fs); g = np.stack(gs); J = np.stack(Js)

    C = attribution(g, J, ts, dQ)            # (76, 6); only column j moves
    assert float(np.max(np.abs(C[:, [k for k in range(STUDY_K)
                                     if k != j]]))) == 0.0
    Cj = float(C[:, j].sum())
    dT = float(T.value(F[-1]) - T.value(F[0]))
    compl = abs(Cj - dT) / abs(dT)           # gate (d)

    BC = B @ C
    b = BC[:, j]
    phi_s = -float(b[s])
    interior = [v for v in range(n) if v not in (s, t)]
    pos = sum(max(float(b[v]), 0.0) for v in interior)
    neg = sum(max(-float(b[v]), 0.0) for v in interior)
    phi_min, phi_max = phi_s - pos, phi_s + neg

    cr = enum_c(lib, adj, n, s, t, b[:, None])
    clo, chi = float(cr['phi_min'][0]), float(cr['phi_max'][0])
    n_sets = int(cr['n_sets'])
    full = (1 << n) - 1
    for mask, val in ((cr['min_mask'][0], clo), (cr['max_mask'][0], chi)):
        assert connected(adj, mask) and connected(adj, full & ~mask)
        nd = np.array(mask_to_nodes(mask)) - 1
        assert s in nd and t not in nd
        assert abs(-float(BC[nd, j].sum()) - val) < 1e-6

    phi_joint = {}
    for tag in ('U_minus', 'U_plus'):
        mask, nodes0, joint_val = joint_sets[key][tag]
        phi_joint[tag] = {
            'phi_singleod': -float(BC[nodes0, j].sum()),
            'phi_joint_published': joint_val,
            'nodes_1idx': (nodes0 + 1).tolist(),
        }

    row = {
        'key': key,
        'published_demand_veh_h': float(pub_j),
        'Cj_column_sum': Cj,
        'dT_endpoint': dT,
        'completeness_relative': float(compl),
        'phi_s': phi_s,
        'unrestricted': {'min': phi_min, 'max': phi_max,
                         'span': phi_max - phi_min},
        'connected': {'min': clo, 'max': chi, 'span': chi - clo,
                      'n_sets': n_sets},
        'range_pct': 100.0 * (phi_max - phi_min) / abs(phi_s),
        'conn_range_pct': 100.0 * (chi - clo) / abs(phi_s),
        'connected_straddles_zero': bool(clo < 0.0 < chi),
        'phi_at_joint_boundaries': phi_joint,
        'gates': {'solver_max_residual': float(max_res),
                  'flow_conservation_max': float(max_flowcons),
                  'jacobian_conservation_max': float(max_jcons),
                  'fd_relative_errors': fd_errors},
        'elapsed_seconds': time.time() - started,
    }
    return row, C


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, trips, _ = select_ods(net)
    e86 = json.loads(E86_JSON.read_text())
    stored = [tuple(od) for od in e86['protocol']['study_ods_1idx']]
    fresh = [(s + 1, t + 1) for s, t in study_ods]
    assert fresh == stored, f'selection drift: {fresh} vs {stored}'

    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    q_pub6 = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    psfull = PathStructure(net, all_ods)
    solver = EdgeSUE(psfull, study_k=STUDY_K)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    M_study = net.od_M(study_ods)
    M_full = net.od_M(all_ods)

    z87 = np.load(E87_NPZ)
    C41 = z87['C_41']
    e91 = json.loads(E91_JSON.read_text())
    adj, n = build_adj_bits(net)
    lib = load_c_enum()

    # gate (f): joint extremal sets reproduce e91 where applicable
    joint_sets = joint_extremal_sets(lib, adj, n, C41, study_ods, e91, B)

    keys6 = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    per_od = {}
    cols = {}
    for key in FOCAL:
        row, C = run_focal(key, net, psfull, solver, T, B, M_full, M_study,
                           q_pub6, q_bg, study_ods, adj, n, lib, joint_sets)
        j = keys6.index(key)
        # gate (a)-(c) thresholds
        g = row['gates']
        assert g['solver_max_residual'] <= 1e-7, g
        assert g['flow_conservation_max'] <= 1e-6, g
        assert g['jacobian_conservation_max'] <= 1e-8, g
        # gate (d)
        assert row['completeness_relative'] <= 1e-4, row
        # gate (e): family count equals the published count for this OD
        pub_n = int(e86['per_od'][key]['n_connected_cuts'])
        assert row['connected']['n_sets'] == pub_n, (key, pub_n)
        per_od[key] = row
        cols[key] = C[:, j].copy()
        print(f"{key}: Cj={row['Cj_column_sum']:.1f} dT={row['dT_endpoint']:.1f} "
              f"conn=[{row['connected']['min']:.1f},{row['connected']['max']:.1f}] "
              f"phi(U-)={row['phi_at_joint_boundaries']['U_minus']['phi_singleod']:.1f} "
              f"phi(U+)={row['phi_at_joint_boundaries']['U_plus']['phi_singleod']:.1f} "
              f"({row['elapsed_seconds']:.0f}s)", flush=True)

    np.savez_compressed(NPZ,
                        **{f'C_{key.replace("->", "_")}': col
                           for key, col in cols.items()},
                        ts=np.linspace(0, 1, GRID))
    out = {
        'protocol': {
            'purpose': 'single-OD demand-change check for the six-query '
                       'sign crossings: each focal OD alone moves 0.2->1.6 '
                       'of published demand, other 527 ODs at published '
                       'demand, full re-equilibration (setting-C protocol)',
            'focal_ods': list(FOCAL),
            'control': '6->8',
            'grid': GRID,
            'claims_supported': 'whether each sign crossing and the '
                                'positive system contribution survive when '
                                'only one OD demand changes; NOT agreement '
                                'with the joint scenario in magnitude',
        },
        'per_od': per_od,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'written -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
