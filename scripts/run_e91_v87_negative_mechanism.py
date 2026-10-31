"""E91 (v87, external review round 59, review item P1): concrete
mechanism behind the negative connected boundary account, plus the
setting-C scalar system contributions C_j of the eight main queries.

Part A (mechanism, zero new solves). The published six-query columns
(e87 npz: gT, J_study, J_background, C_41 on the 81-node chain whose
even nodes are the published 41-point grid) are re-summarized:

  * study/background attribution split at the published grid:
    C_41^study = trapz(g * J^study), C_41^bg = trapz(g * J^bg), with
    additivity gates against the published total column;
  * for each of the three sign-crossing queries, the connected
    minimizing set U_- (exact enumeration, min_mask) and maximizing set
    U_+ at the baseline setting: node lists, phi_j(U_-), phi_j({s_j}),
    and the interior divergence sum  phi_s - phi_min  (the review's
    identity phi_j(U_-) = phi_j({s_j}) - sum_{v in U_- \\ s_j} b_jv);
  * inside U_- \\ {s_j}: per-node signed divergence with total/study/
    background components for the largest-magnitude nodes; the positive
    mass captured, the negative mass dragged in by connectivity, and
    the study/background split of the interior sum;
  * which of the globally largest positive-divergence interior nodes
    connectivity excludes from U_-.

Part B (eight-query C_j, zero new solves). The setting-C 41-point
columns of the eight main queries (e84 npz, C_k10_theta1) are summed
over links: C_j = sum_e C~_{e,j}. Gates: the columns reproduce the
published origin-cut accounts and spreads of e84's JSON, and the scalar
completeness |sum_j C_j - dT| / |dT| is reported against the endpoint
TSTT change of the same stored chain.

Gates (all must pass before any number enters the paper):
  (a) C_41 = C_41^study + C_41^bg to 1e-9 max-abs, and BC likewise;
  (b) baseline connected extrema reproduce the published e87 intervals
      (relative 1e-9);
  (c) every reported phi value equals -sum of the corresponding BC
      entries (recomputed by direct summation, 1e-6);
  (d) interior sum identity: phi_s - phi_min(U_-) equals the interior
      divergence sum of U_- (1e-6);
  (e) Part B: column sums reproduce e84's published phi_s and spread
      (relative 1e-9) and completeness is reported.

Output: experiments/results/e91_v87_negative_mechanism.json
        console log experiments/results/e91_console.log
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
from run_e9_transport import TSTTT  # noqa: E402
from run_e83_v82_validation import attribution  # noqa: E402
from run_e80_connected_cuts import build_adj_bits, load_c_enum, enum_c, mask_to_nodes  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402

RESULTS = HERE / 'results'
E84_NPZ = RESULTS / 'e84_v83_coverage.npz'
E84_JSON = RESULTS / 'e84_v83_coverage.json'
E87_JSON = RESULTS / 'e87_v85_newod_refine.json'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
OUT = RESULTS / 'e91_v87_negative_mechanism.json'
CROSSING = ('1->19', '1->15', '7->12')


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    n = net.n
    study_ods, trips, _ = select_ods(net)
    q_new = np.array([trips[od] for od in study_ods])
    dQ = 1.6 * q_new - 0.2 * q_new
    adj, _ = build_adj_bits(net)
    lib = load_c_enum()

    # ---------------- Part A: six-query mechanism at baseline ----------------
    z87 = np.load(E87_NPZ)
    e87 = json.loads(E87_JSON.read_text())
    ts81 = z87['ts']
    idx41 = np.arange(0, 81, 2)
    ts41 = ts81[idx41]
    g41 = z87['gT'][idx41]
    C41 = z87['C_41']
    BC = z87['BC_41']
    Cs41 = attribution(g41, z87['J_study'][idx41], ts41, dQ)
    Cb41 = attribution(g41, z87['J_background'][idx41], ts41, dQ)
    add_c = float(np.max(np.abs(C41 - Cs41 - Cb41)))          # gate (a)
    BCs, BCb = B @ Cs41, B @ Cb41
    add_bc = float(np.max(np.abs(BC - BCs - BCb)))
    assert add_c <= 1e-9 and add_bc <= 1e-9, (add_c, add_bc)

    mechanism = {}
    dev_c = dev_d = 0.0
    for j, (s, t) in enumerate(study_ods):
        key = f'{s + 1}->{t + 1}'
        cr = enum_c(lib, adj, n, s, t, BC[:, j:j + 1])
        clo, chi = float(cr['phi_min'][0]), float(cr['phi_max'][0])
        pub = e87['per_od'][key]['interval_41']['connected']       # gate (b)
        assert abs(clo - pub['min']) < 1e-9 * max(1, abs(pub['min']))
        assert abs(chi - pub['max']) < 1e-9 * max(1, abs(pub['max']))
        u_minus = [v - 1 for v in mask_to_nodes(cr['min_mask'][0])]
        u_plus = [v - 1 for v in mask_to_nodes(cr['max_mask'][0])]
        assert s in u_minus and t not in u_minus
        assert s in u_plus and t not in u_plus
        if key not in CROSSING:
            continue
        b = BC[:, j]
        bs, bb = BCs[:, j], BCb[:, j]
        phi_s = -float(b[s])
        interior = [v for v in u_minus if v != s]
        phi_m = -float(b[u_minus].sum())                          # gate (c)
        dev_c = max(dev_c, abs(phi_m - clo))
        assert dev_c < 1e-6
        interior_sum = phi_s - phi_m                              # gate (d)
        dev_d = max(dev_d, abs(interior_sum - float(b[interior].sum())))
        assert dev_d < 1e-6
        by_mag = sorted(interior, key=lambda v: abs(b[v]), reverse=True)
        node_rows = [{'node': v + 1,
                      'total': float(b[v]), 'study': float(bs[v]),
                      'background': float(bb[v])}
                     for v in by_mag[:8]]
        pos_inside = sum(max(float(b[v]), 0.0) for v in interior)
        neg_inside = sum(min(float(b[v]), 0.0) for v in interior)
        all_interior = [v for v in range(n) if v not in (s, t)]
        pos_total = sum(max(float(b[v]), 0.0) for v in all_interior)
        top_global = sorted(all_interior, key=lambda v: b[v], reverse=True)
        mechanism[key] = {
            'phi_s': phi_s, 'phi_U_minus': phi_m,
            'phi_U_plus': -float(b[u_plus].sum()),
            'U_minus_size': len(u_minus), 'U_minus_nodes_1idx': [v + 1 for v in u_minus],
            'U_plus_size': len(u_plus), 'U_plus_nodes_1idx': [v + 1 for v in u_plus],
            'interior_divergence_sum': interior_sum,
            'interior_sum_study': float(bs[interior].sum()),
            'interior_sum_background': float(bb[interior].sum()),
            'positive_mass_inside': pos_inside, 'negative_mass_inside': neg_inside,
            'positive_mass_all_interior': pos_total,
            'positive_share_captured': pos_inside / pos_total,
            'n_interior_nodes_in_U': len(interior),
            'largest_nodes_in_U': node_rows,
            'largest_positive_nodes_excluded': [
                {'node': v + 1, 'total': float(b[v])}
                for v in top_global if v not in u_minus and b[v] > 0][:5],
        }
        print(key, {k: (round(v, 1) if isinstance(v, float) else v)
                    for k, v in mechanism[key].items()
                    if not isinstance(v, list)}, flush=True)

    # ---------------- Part B: eight-query setting-C C_j ----------------
    z84 = np.load(E84_NPZ)
    e84 = json.loads(E84_JSON.read_text())
    C8 = z84['C_k10_theta1_C']
    F8 = z84['C_k10_theta1_F']
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    dT8 = float(T.value(F8[-1]) - T.value(F8[0]))
    queries84 = [tuple(od) for od in e84['protocol']['queries']]   # 1-indexed
    cj8 = {}
    for j, (s1, t1) in enumerate(queries84):
        key = f'{s1}->{t1}'
        pub = e84['C_k10_theta1']['per_od'][key]                   # gate (e)
        bc = B @ C8[:, j]
        phi_s = -float(bc[s1 - 1])
        spread = float(np.abs(np.array([bc[v] for v in range(n)
                                        if v not in (s1 - 1, t1 - 1)])).sum())
        assert abs(phi_s - pub['phi_s']) < 1e-9 * max(1, abs(pub['phi_s']))
        assert abs(spread - pub['spread_abs']) < 1e-9 * max(1, abs(pub['spread_abs']))
        cj8[key] = {'C_j': float(C8[:, j].sum()), 'phi_s': phi_s,
                    'spread_abs': spread}
    completeness8 = float(abs(C8.sum() - dT8) / abs(dT8))
    # same stored chain, so this must equal e84's own completeness gate
    assert abs(completeness8 - e84['C_k10_theta1']['gates']['completeness_relative']) < 1e-12

    out = {
        'protocol': {
            'purpose': 'negative-account mechanism extraction (P1) + eight-query '
                       'setting-C scalar contributions C_j (T2); re-summation of '
                       'published frozen arrays, no new equilibrium solves',
            'part_a_basis': 'e87 published 41-point columns (81-chain even nodes)',
            'part_b_basis': 'e84 setting-C 41-point columns (C_k10_theta1)',
            'crossing_queries': list(CROSSING),
        },
        'gates': {
            'C_additivity_max_abs': add_c, 'BC_additivity_max_abs': add_bc,
            'baseline_enum_matches_e87': True,
            'phi_recompute_max_abs_dev': dev_c,
            'interior_identity_max_abs_dev': dev_d,
            'part_B_columns_match_e84': True,
        },
        'mechanism': mechanism,
        'eight_query_Cj': {
            'per_od': cj8,
            'sum_Cj': float(C8.sum()),
            'dT_endpoint_TSTT': dT8,
            'completeness_relative': completeness8,
        },
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'Part B: sum C_j = {C8.sum():.1f}, dT = {dT8:.1f}, '
          f'completeness rel = {completeness8:.3e}', flush=True)
    print(f'Completed e91 in {out["elapsed_seconds"]:.1f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
