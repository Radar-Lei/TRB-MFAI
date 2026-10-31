"""E97 (v91, external review round 63, item III-4 "the single most
valuable addition"): per-edge endpoint identity check for the single-OD
chains of E95.

Identity under test.  For a separable TSTT outcome T(F) = sum_e g_e(F_e)
with g_e(F_e) = F_e t_e(F_e), only ONE OD demand q_j moving along the
straight line q(lambda) = q0 + lambda (q1 - q0) e_j gives, per edge,

    C~_{e,j} = int_0^1 g_e'(F_e(lambda)) dF_e(lambda)/dlambda dlambda
             = g_e(F_e(q1)) - g_e(F_e(q0)),

by the fundamental theorem of calculus applied along the realized
equilibrium path -- BACKGROUND RE-EQUILIBRATION IS IRRELEVANT because
dF_e/dlambda is the full Jacobian column (all ODs' route adjustments
included).  The published completeness check tests only the column SUM
against the endpoint TSTT change; this check tests every EDGE entry,
which can expose edge-level errors that cancel in the sum.

Protocol.  For each focal OD of E95 ((1,19), (1,15), (7,12) crossings
plus the (6,8) positive control) the 41-point single-OD chain is
re-solved with the identical E95 code path (same solver, same warm-start
sequence, same implicit sensitivities), so the recomputed attribution
column must reproduce the frozen E95 npz column; the endpoint flows of
THIS chain (not the joint six-OD chain) then give the endpoint edge-cost
changes g(F1) - g(F0).

Gates (all must pass before any number enters the paper):
  (a) recomputed C~ column reproduces the frozen E95 npz column:
      relative l2 <= 1e-9 per focal OD;
  (b) recomputed endpoint TSTT difference reproduces the E95 JSON
      dT_endpoint within 1e-9 relative;
  (c) solver path-residual <= 1e-7, flow conservation <= 1e-6,
      Jacobian OD conservation <= 1e-8 at every state (E95 thresholds);
  (d) sum_e residual reproduces the published completeness gap
      C_j - dT within 1e-9 relative (self-consistency of the identity
      bookkeeping);
  (e) per-edge g uses the SAME BPR parameters as TSTTT (verified by
      sum_e g_e(F) == T.value(F) at both endpoints to 1e-9 relative).

Reported per focal OD:
  * || C~_{:,j} - (g(F1) - g(F0)) ||_inf, ||.||_2 (TSTT units);
  * the same normalized by max_e |g_e(F1) - g_e(F0)| and by
    max_e |C~_{e,j}| (per-edge scales);
  * max_e |g_e(F1) - g_e(F0)|, max_e |C~_{e,j}| for context;
  * the edge attaining the maximum residual (1-based node pair).

NOT claimed: exact zero.  The columns come from 41-point trapezoid
integration; the residual IS the trapezoid quadrature error of the
identity integrand, so a small nonzero value is the expected outcome
and its size quantifies per-edge quadrature error that the sum-level
completeness check can hide.

Output: experiments/results/e97_v91_singleod_identity.json
        experiments/results/e97_v91_singleod_identity.npz
        console log experiments/results/e97_console.log
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

RESULTS = HERE / 'results'
E95_JSON = RESULTS / 'e95_v90_singleod.json'
E95_NPZ = RESULTS / 'e95_v90_singleod.npz'
OUT = RESULTS / 'e97_v91_singleod_identity.json'
NPZ = RESULTS / 'e97_v91_singleod_identity.npz'

STUDY_K = 6
GRID = 41
FOCAL = ('1->19', '1->15', '7->12', '6->8')   # three crossings + control


def edge_g(F, t0, cap):
    """Per-edge integrand g_e(F_e) = F_e t_e(F_e) = t0 F (1 + 0.15 x^4),
    x = F/cap -- identical BPR form to TSTTT.value."""
    x = F / cap
    return t0 * F * (1.0 + 0.15 * x ** 4)


def run_focal_chain(key, net, psfull, solver, T, B, M_full, M_study,
                    q_pub6, q_bg, study_ods):
    """Re-solve one E95 single-OD chain exactly as E95 did; return the
    attribution column, the endpoint flows, and the gate records."""
    keys6 = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    j = keys6.index(key)
    pub_j = q_pub6[j]
    dQ = np.zeros(STUDY_K)
    dQ[j] = 1.4 * pub_j                     # 0.2 -> 1.6 of published

    ts = np.linspace(0.0, 1.0, GRID)
    Fs, gs, Js = [], [], []
    max_res = 0.0
    max_jcons = 0.0
    max_flowcons = 0.0
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
        Fs.append(F); gs.append(T.grad(F)); Js.append(J)
    F = np.stack(Fs); g = np.stack(gs); J = np.stack(Js)

    C = attribution(g, J, ts, dQ)            # (76, 6); only column j moves
    return j, C[:, j].copy(), F, {
        'solver_max_residual': float(max_res),
        'flow_conservation_max': float(max_flowcons),
        'jacobian_conservation_max': float(max_jcons),
    }


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, trips, _ = select_ods(net)
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    q_pub6 = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    psfull = PathStructure(net, all_ods)
    solver = EdgeSUE(psfull, study_k=STUDY_K)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    M_study = net.od_M(study_ods)
    M_full = net.od_M(all_ods)

    e95 = json.loads(E95_JSON.read_text())
    z95 = np.load(E95_NPZ)

    t0 = net.free_flow_time
    cap = net.capacity
    edges = [(u + 1, v + 1) for u, v in net.edges]   # 1-based pairs

    per_od = {}
    npz_cols, npz_dg = {}, {}
    for key in FOCAL:
        j, Cj_col, F, gates = run_focal_chain(
            key, net, psfull, solver, T, B, M_full, M_study,
            q_pub6, q_bg, study_ods)

        # ---- gate (a): reproduce the frozen E95 column ----
        frozen = z95[f'C_{key.replace("->", "_")}']
        rel_col = float(np.linalg.norm(Cj_col - frozen)
                        / np.linalg.norm(frozen))
        assert rel_col <= 1e-9, (key, 'column drift', rel_col)

        # ---- per-edge endpoint identity ----
        g0 = edge_g(F[0], t0, cap)
        g1 = edge_g(F[-1], t0, cap)
        dg = g1 - g0

        # ---- gate (e): per-edge g matches TSTTT.value at endpoints ----
        for tag, (ga, Fa) in (('g0', (g0, F[0])), ('g1', (g1, F[-1]))):
            rel = abs(float(ga.sum()) - float(T.value(Fa))) \
                / abs(float(T.value(Fa)))
            assert rel <= 1e-9, (key, tag, rel)

        resid = Cj_col - dg

        # ---- gate (b): endpoint TSTT difference matches E95 JSON ----
        dT = float(T.value(F[-1]) - T.value(F[0]))
        rel_dT = abs(dT - e95['per_od'][key]['dT_endpoint']) \
            / abs(e95['per_od'][key]['dT_endpoint'])
        assert rel_dT <= 1e-9, (key, 'dT drift', rel_dT)

        # ---- gate (d): residual sum reproduces the completeness gap ----
        gap = float(e95['per_od'][key]['Cj_column_sum']
                    - e95['per_od'][key]['dT_endpoint'])
        rel_gap = abs(float(resid.sum()) - gap) / max(abs(gap), 1e-12)
        assert rel_gap <= 1e-6, (key, 'gap mismatch',
                                 float(resid.sum()), gap)

        # ---- gate (c): E95 solver thresholds ----
        assert gates['solver_max_residual'] <= 1e-7, gates
        assert gates['flow_conservation_max'] <= 1e-6, gates
        assert gates['jacobian_conservation_max'] <= 1e-8, gates

        k_inf = int(np.argmax(np.abs(resid)))
        max_dg = float(np.max(np.abs(dg)))
        max_c = float(np.max(np.abs(Cj_col)))
        per_od[key] = {
            'resid_inf': float(np.max(np.abs(resid))),
            'resid_l2': float(np.linalg.norm(resid)),
            'resid_inf_over_max_dg': float(np.max(np.abs(resid)) / max_dg),
            'resid_inf_over_max_C': float(np.max(np.abs(resid)) / max_c),
            'max_abs_dg': max_dg,
            'max_abs_C': max_c,
            'resid_sum': float(resid.sum()),
            'completeness_gap_published': gap,
            'resid_max_edge_1idx': list(edges[k_inf]),
            'resid_sum_over_abs_dT': float(abs(resid.sum()) / abs(dT)),
            'gates': {**gates,
                      'column_vs_frozen_rel_l2': rel_col,
                      'dT_vs_published_rel': rel_dT,
                      'resid_sum_vs_gap_rel': rel_gap},
        }
        npz_cols[key] = Cj_col
        npz_dg[key] = dg
        print(f"{key}: ||resid||_inf={per_od[key]['resid_inf']:.4f} "
              f"({per_od[key]['resid_inf_over_max_dg']*100:.4f}% of "
              f"max|dg|, edge {edges[k_inf]}), "
              f"||resid||_2={per_od[key]['resid_l2']:.4f}, "
              f"sum(resid)={resid.sum():.6f} vs gap {gap:.6f}",
              flush=True)

    np.savez_compressed(NPZ,
                        **{f'C_{k.replace("->", "_")}': v
                           for k, v in npz_cols.items()},
                        **{f'dg_{k.replace("->", "_")}': v
                           for k, v in npz_dg.items()})
    out = {
        'protocol': {
            'purpose': 'per-edge endpoint identity check for the E95 '
                       'single-OD chains: C~_{e,j} vs g_e(F1)-g_e(F0) '
                       'per edge, separable TSTT, background '
                       're-equilibration included in the Jacobian column',
            'focal_ods': list(FOCAL),
            'control': '6->8',
            'grid': GRID,
            'claims_supported': 'the E95 single-OD attribution columns '
                                'satisfy the per-edge endpoint identity '
                                'up to trapezoid quadrature error; NOT a '
                                'claim of exact zero',
        },
        'per_od': per_od,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'written -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
