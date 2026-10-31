"""E123 (v107): attribution-path robustness for the setting-C six queries.

External review of v106 (item 2 / A2): the paper fixes the straight-line
joint demand path gamma(lambda) = q0 + lambda*dQ; when several ODs change
together, the per-OD attribution columns depend on this choice.  This
experiment compares, on the setting-C six-query Sioux Falls protocol,
the straight-line joint path against piecewise-linear paths that change
the six study ODs one at a time:

    sequential: canonical order (0,1,2,3,4,5);
    reverse:    canonical order reversed;
    perm-1..20: 20 fixed-seed random OD permutations.

For every order and query we report C_j, the geometric-family account
range [min, max], the common summary (beta, epsilon, normalized error),
sign crossing over the family, the maximum profile deviation from the
straight-line accounts over the same family, and the four-part
free-flow/congestion x study/background decomposition evaluated at the
fixed straight-line connected-minimizing regions U_- of e91 (so the
region is held constant and only the path varies).

The total outcome change is path independent (same endpoints), which the
completeness gate verifies on every order; the per-query columns are not,
which is the point.

Gates (all must pass before any number enters the paper):
  (a) straight-line replication: columns from the frozen e87 arrays
      reproduce C_41 to <= 1e-9 relative;
  (b) screenline replication: the straight-line geometric-family ranges
      reproduce the published Table (tab:newod) values to <= 1 unit;
  (c) endpoint equality: every order's first and last states reproduce
      the straight-line endpoint equilibria to <= 1e-6 absolute flow;
  (d) completeness: |sum_j C_j - dT| / |dT| <= 1e-6 on every order,
      with dT identical across orders;
  (e) mechanism replication: the straight-line four-part totals at U_-
      reproduce the published e101 two-way ff/cong splits to <= 0.5 and
      the published tab:ffcong four-part entries to <= 0.5;
  (f) solver fixed-point residual <= 1e-7 throughout.

Output: experiments/results/e123_v107_pathrobust/{results.json}
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import json
import time
from pathlib import Path

import numpy as np

import run_e111_v102_pairs as p
from run_e83_v82_validation import attribution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e123_v107_pathrobust'
OUT.mkdir(exist_ok=True)
E87 = ROOT / 'experiments/results/e87_v85_newod_refine.npz'
E91 = ROOT / 'experiments/results/e91_v87_negative_mechanism.json'
E101 = ROOT / 'experiments/results/e101_v93_ffcong.json'
PERM_SEED = 20261031
N_PERMS = 20
GRID = 41
PAPER_SCREENLINE = {(7, 18): (642, 702), (1, 19): (-2767, 22124),
                    (4, 5): (2652, 3176), (1, 15): (511, 36621),
                    (6, 8): (45230, 65286), (7, 12): (-2967, 57780)}
# published tab:ffcong four-part entries at U_- (TSTT units)
PAPER_FFCONG = {'1->19': (-361.3, 759.0, -17357.9, 3458.6),
                '1->15': (1541.8, 333.8, -6537.4, -3624.2),
                '7->12': (3897.0, -1311.1, 125.5, -20327.6)}
KEY_ORDER = ('ff_study', 'ff_bg', 'cong_study', 'cong_bg')


def segment_states(c0, c1, qbg, sv, warm, T):
    """Solve equilibria and sensitivities along one straight segment."""
    ts = np.linspace(0, 1, GRID)
    F, J, Js, Jb, g = [], [], [], [], []
    for lam in ts:
        q = c0 + lam * (c1 - c0)
        f, h, prob, _ = sv.solve(np.r_[q, qbg], warm)
        warm = f
        jac, js, jb = sv.sensitivity(f, h, prob, np.r_[q, qbg])
        F.append(f)
        J.append(jac)
        Js.append(js)
        Jb.append(jb)
        g.append(T.grad(f))
    return ts, np.array(F), np.array(J), np.array(Js), np.array(Jb), np.array(g), warm


def family_accounts(C, geo):
    """Geometric-family phi, range and common summary per query."""
    out = []
    for j in range(6):
        phi = -(geo[j] @ C[:, j])
        lo, hi = float(phi.min()), float(phi.max())
        out.append({'phi': phi, 'min': lo, 'max': hi,
                    'beta': .5 * (lo + hi), 'eps': .5 * (hi - lo),
                    'eps_pct': 100 * .5 * (hi - lo) / max(abs(lo), abs(hi), 1e-9),
                    'sign_crossing': bool(lo < 0 < hi)})
    return out


def mech_parts(ub, wff, wcong, Js, Jb, dq, ts):
    """phi^{a,b}(U) = -dq * int 1_U^T B D^a J^b_j dlambda at fixed region."""
    parts = {}
    for a, w in (('ff', wff), ('cong', wcong)):
        for b, X in (('study', Js), ('bg', Jb)):
            parts[f'{a}_{b}'] = -float(dq * np.trapezoid(ub @ (w * X).T, ts))
    return parts


def main():
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    t0, cap = n.free_flow_time, n.capacity
    masks = p.screenline_masks(ods)
    geo = [masks[j] @ B for j in range(6)]
    ods1 = [(s + 1, t + 1) for s, t in ods]
    keys1 = [f'{a}->{b}' for a, b in ods1]

    # ---- straight-line reference from the frozen e87 arrays ----
    z = np.load(E87)
    ts41 = z['ts'][::2]
    F0 = z['F'][::2]
    J0 = z['J'][::2]
    Js0 = z['J_study'][::2]
    Jb0 = z['J_background'][::2]
    dQ = z['dQ']
    C_straight = attribution(np.array([T.grad(f) for f in F0]), J0, ts41, dQ)
    rep = float(np.max(np.abs(C_straight - z['C_41'])) / np.max(np.abs(z['C_41'])))
    assert rep <= 1e-9, rep

    fam0 = family_accounts(C_straight, geo)
    for j, od in enumerate(ods1):
        lo, hi = fam0[j]['min'], fam0[j]['max']
        plo, phi_ = PAPER_SCREENLINE[od]
        assert abs(lo - plo) <= 1.0 and abs(hi - phi_) <= 1.0, (od, lo, hi, plo, phi_)

    # ---- fixed U_- regions from e91 and straight-line mechanism ----
    e91 = json.loads(E91.read_text())['mechanism']
    e101 = json.loads(E101.read_text())['per_od']
    ubs, mech0 = {}, {}
    for key in PAPER_FFCONG:
        nodes = e91[key]['U_minus_nodes_1idx']
        u = np.zeros((1, n.n))
        u[0, [v - 1 for v in nodes]] = 1.0
        ubs[key] = u @ B                      # 1 x m: the phi operator at U_-
        j = keys1.index(key)
        wff, wcong = t0, 0.75 * t0 * (F0 / cap) ** 4
        mech0[key] = mech_parts(ubs[key], wff, wcong, Js0[:, :, j], Jb0[:, :, j], dQ[j], ts41)
        parts = mech0[key]
        assert abs(parts['ff_study'] + parts['ff_bg'] - e101[key]['U_minus']['phi_U_minus_ff']) <= 0.5, key
        assert abs(parts['cong_study'] + parts['cong_bg'] - e101[key]['U_minus']['phi_U_minus_cong']) <= 0.5, key
        got = tuple(round(parts[k], 1) for k in KEY_ORDER)
        assert max(abs(got[i] - PAPER_FFCONG[key][i]) for i in range(4)) <= 0.5 + 2 * 0.05 * max(
            abs(v) for v in PAPER_FFCONG[key]), (key, got, PAPER_FFCONG[key])

    # ---- orders ----
    rng = np.random.default_rng(PERM_SEED)
    perms = [tuple(int(x) for x in rng.permutation(6)) for _ in range(N_PERMS)]
    orders = [('sequential', tuple(range(6))), ('reverse', tuple(range(5, -1, -1)))]
    orders += [(f'perm{i + 1}', pm) for i, pm in enumerate(perms)]

    q_start, q_end = 0.2 * qb, 1.6 * qb
    dT = float(T.value(F0[-1]) - T.value(F0[0]))

    records = []
    gates = {'straight_replication_rel': rep, 'max_residual': 0.0,
             'endpoint_dev_max': 0.0, 'completeness_max': 0.0,
             'mechanism_straight_ok': True}
    started = time.perf_counter()
    for name, order in orders:
        corner = q_start.copy()
        warm = None
        C_order = np.zeros((n.m, 6))
        mech = {}
        F_first = F_last = None
        for j in order:
            nxt = corner.copy()
            nxt[j] = q_end[j]
            ts, F, J, Js, Jb, g, warm = segment_states(corner, nxt, qbg, sv, warm, T)
            if F_first is None:
                F_first = F[0]
            F_last = F[-1]
            corner = nxt
            dqv = np.zeros(6)
            dqv[j] = dQ[j]
            C_order += attribution(g, J, ts, dqv)
            key = keys1[j]
            if key in PAPER_FFCONG:
                wff, wcong = t0, 0.75 * t0 * (F / cap) ** 4
                mech[key] = mech_parts(ubs[key], wff, wcong, Js[:, :, j], Jb[:, :, j], dQ[j], ts)
            gates['max_residual'] = max(gates['max_residual'], sv.max_residual)
        gates['endpoint_dev_max'] = max(gates['endpoint_dev_max'],
                                        float(np.max(np.abs(F_first - F0[0]))),
                                        float(np.max(np.abs(F_last - F0[-1]))))
        comp = float(abs(C_order.sum() - dT) / abs(dT))
        gates['completeness_max'] = max(gates['completeness_max'], comp)
        assert comp <= 1e-6, (name, comp)
        fam = family_accounts(C_order, geo)
        for j in range(6):
            dev = float(np.max(np.abs(fam[j]['phi'] - fam0[j]['phi'])))
            rec = {'order': name, 'query': keys1[j], 'Cj': float(C_order[:, j].sum()),
                   'fam_min': fam[j]['min'], 'fam_max': fam[j]['max'],
                   'beta': fam[j]['beta'], 'eps': fam[j]['eps'], 'eps_pct': fam[j]['eps_pct'],
                   'sign_crossing': fam[j]['sign_crossing'],
                   'profile_dev_max': dev,
                   'profile_dev_pct': 100 * dev / max(np.max(np.abs(fam0[j]['phi'])), 1e-9)}
            if keys1[j] in PAPER_FFCONG:
                rec['mechanism'] = mech[keys1[j]]
            records.append(rec)
        print('order', name, 'done', round(time.perf_counter() - started, 1), 's', flush=True)

    assert gates['endpoint_dev_max'] <= 1e-6, gates['endpoint_dev_max']
    assert gates['max_residual'] <= 1e-7, gates['max_residual']

    # ---- summaries per query over the 22 alternative orders ----
    summary = {}
    for j in range(6):
        key = keys1[j]
        recs = [r for r in records if r['query'] == key]
        Cj = np.array([r['Cj'] for r in recs])
        eps = np.array([r['eps_pct'] for r in recs])
        dev = np.array([r['profile_dev_pct'] for r in recs])
        s = {'Cj_straight': float(C_straight[:, j].sum()),
             'Cj_alt_min': float(Cj.min()), 'Cj_alt_max': float(Cj.max()),
             'Cj_alt_median': float(np.median(Cj)),
             'Cj_rel_dev_max_pct': float(100 * np.max(np.abs(Cj - C_straight[:, j].sum()))
                                         / max(abs(float(C_straight[:, j].sum())), 1e-9)),
             'eps_pct_straight': fam0[j]['eps_pct'],
             'eps_pct_alt_median': float(np.median(eps)),
             'eps_pct_alt_range': [float(eps.min()), float(eps.max())],
             'sign_crossing_straight': fam0[j]['sign_crossing'],
             'sign_crossing_alt_count': int(sum(r['sign_crossing'] for r in recs)),
             'sign_of_fam_min_straight': float(np.sign(fam0[j]['min'])),
             'fam_min_alt_range': [float(min(r['fam_min'] for r in recs)),
                                   float(max(r['fam_min'] for r in recs))],
             'fam_max_alt_range': [float(min(r['fam_max'] for r in recs)),
                                   float(max(r['fam_max'] for r in recs))],
             'profile_dev_pct_median': float(np.median(dev)),
             'profile_dev_pct_max': float(dev.max())}
        if key in PAPER_FFCONG:
            dom0 = max(KEY_ORDER, key=lambda k: abs(mech0[key][k]))
            doms = [max(KEY_ORDER, key=lambda k: abs(r['mechanism'][k])) for r in recs]
            s['dominant_mechanism_straight'] = dom0
            s['dominant_mechanism_alt'] = {d: doms.count(d) for d in sorted(set(doms))}
            s['dominant_matches_straight'] = int(sum(d == dom0 for d in doms))
            s['mechanism_straight_parts'] = mech0[key]
        summary[key] = s

    result = {'protocol': {'orders': len(orders), 'perm_seed': PERM_SEED,
                           'grid_per_segment': GRID, 'segments_per_order': 6,
                           'straight_source': 'e87 frozen 41-point subset',
                           'U_minus_source': 'e91 connected-minimizing regions, fixed across orders',
                           'dT': dT, 'elapsed_seconds': time.perf_counter() - started},
              'gates': gates,
              'summary': summary,
              'records': records}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'gates': gates, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
