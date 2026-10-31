"""E96 (v90, external review round 62, 87/100, item III-1 "must fix"):
complete the surrogate-validation proposition C_j > 0 AND phi_j(U_-) < 0
on the SAME learned model and seed, formalize the dimensional cut error
epsilon_j^cut and the sign margin m_j, and locate the maximum
attribution-column error.

All quantities are re-summarized from frozen arrays -- no new training,
no new equilibrium solves:
  * learned attribution columns C_L (5, 76, 6) from the e93 npz;
  * direct-SUE 41-point columns C_41 / BC_41 from the e87 npz;
  * the e91 joint-scenario extremal node sets U_-/U_+;
  * the e80 connected-family enumeration for per-seed eps and intervals.

Reported:
  (1) C_j^learned = sum_e C_L[seed, e, j] per seed and per query, with
      the five-seed [min, max]; all-positive check; relative deviation
      from the direct C_j^SUE = sum_e C_41[:, j];
  (2) same-seed proposition: for every seed s and every sign-crossing
      query, C_j^learned(s) > 0 AND the seed's own connected interval
      straddles zero (recomputed from BC_L(s), cross-checked against the
      e93 JSON intervals at 1e-9);
  (3) epsilon_j^cut(s) = max_{U in connected family}
      |phi_L(U) - phi_SUE(U)| (dimensional, TSTT units; e93's E_cut_abs
      reproduced from frozen arrays) and the sign margin
      m_j(s) = min(-phi_SUE(U_-), phi_SUE(U_+)) - epsilon_j^cut(s);
      m_j(s) > 0 certifies that the learned account keeps opposite
      signs at the two fixed boundaries for that seed.  Worst seed
      (minimum m_j) reported per query;
  (4) attribution-column error table: per (seed, query) relative l2,
      absolute l2, and reference column norm; the maximum unit located
      (seed, OD) with its absolute error and reference norm, testing
      whether a small reference column amplifies the relative value;
  (5) normalized-vs-dimensional bookkeeping: E_j^cut =
      epsilon_j^cut / |phi_j^SUE({s_j})| (origin-cut denominator,
      zero-background family of the paper's Sec. 7 precision paragraph
      uses this same normalization), and epsilon/span ratios per query
      (medians over seeds), matching the published Table-5 convention.

Gates:
  (a) C_41 column sums reproduce the published Table-newod C_j values
      (e87/e86 arrays, 0.05 absolute);
  (b) recomputed learned intervals match the e93 JSON per-seed
      intervals to 1e-9, and recomputed epsilon matches e93 E_cut_abs
      to 1e-9;
  (c) learned completeness: |sum_j C_L - dT_L| / |dT_L| per seed
      reported (dT_L recomputed from the e93 f_L states via TSTTT);
  (d) singleton lower bound epsilon >= |phi_L({s}) - phi_SUE({s})|;
  (e) all thirty (seed, query) units present; five seeds; family counts
      match e86.

Claims supported: the surrogate preserves, per seed, the joint
proposition "positive system contribution with a negative connected
boundary account" for the three crossing queries, within the stated
margin; and its attribution columns are NOT uniformly accurate (the
max relative error is concentrated on the smallest reference column).
NOT claimed: attribution-level fidelity for all queries or general
528-dimensional perturbations.

Output: experiments/results/e96_v90_surrogate_cj.json
        console log experiments/results/e96_console.log
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
from run_e9_transport import TSTTT  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e80_connected_cuts import build_adj_bits, load_c_enum, enum_c  # noqa: E402

RESULTS = HERE / 'results'
E86_JSON = RESULTS / 'e86_v84_newod.json'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E91_JSON = RESULTS / 'e91_v87_negative_mechanism.json'
E93_JSON = RESULTS / 'e93_v89_fullbg_surrogate.json'
E93_NPZ = RESULTS / 'e93_v89_fullbg_surrogate.npz'
OUT = RESULTS / 'e96_v90_surrogate_cj.json'

SEEDS = (0, 1, 2, 3, 4)
CROSSING = ('1->19', '1->15', '7->12')


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, _, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    e86 = json.loads(E86_JSON.read_text())
    e91 = json.loads(E91_JSON.read_text())
    e93 = json.loads(E93_JSON.read_text())

    z87 = np.load(E87_NPZ)
    C_SUE = z87['C_41']
    BC_SUE = B @ C_SUE
    z93 = np.load(E93_NPZ)
    C_L = z93['C_L']                                   # (5, 76, 6)
    f_L = z93['f_L']                                   # (5, 41, 76)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)

    adj, n = build_adj_bits(net)
    lib = load_c_enum()

    # ---- gate (a): direct C_j reproduce the published values ----
    Cj_sue = C_SUE.sum(axis=0)
    published_Cj = {'7->18': 672.9, '1->19': 50097.7, '4->5': 2826.0,
                    '1->15': 77403.2, '6->8': 73964.9, '7->12': 117482.3}
    for j, key in enumerate(keys):
        assert abs(float(Cj_sue[j]) - published_Cj[key]) < 0.05, \
            (key, Cj_sue[j], published_Cj[key])

    # ---- per-seed quantities ----
    per_seed = {}
    col_err = {}
    for si, seed in enumerate(SEEDS):
        CLs = C_L[si]
        BCLs = B @ CLs
        # gate (c): learned completeness per seed
        dT_L = float(T.value(f_L[si, -1]) - T.value(f_L[si, 0]))
        compl = float(abs(CLs.sum() - dT_L) / abs(dT_L))

        od_out = {}
        for j, key in enumerate(keys):
            s, t = study_ods[j]
            # (1) learned scalar contribution
            cj_learned = float(CLs[:, j].sum())
            # (2) per-seed connected interval (recomputed)
            crL = enum_c(lib, adj, n, s, t, BCLs[:, j:j + 1])
            lo = float(crL['phi_min'][0])
            hi = float(crL['phi_max'][0])
            js = e93['per_seed'][si]['per_od'][key]
            assert abs(lo - js['learned_interval'][0]) < 1e-9, (key, lo)
            assert abs(hi - js['learned_interval'][1]) < 1e-9, (key, hi)
            assert int(crL['n_sets']) == int(e86['per_od'][key]
                                             ['n_connected_cuts'])
            # (3) dimensional cut error on the connected family
            D = (BCLs[:, j] - BC_SUE[:, j])[:, None]
            cr = enum_c(lib, adj, n, s, t, D)
            eps = max(abs(float(cr['phi_min'][0])),
                      abs(float(cr['phi_max'][0])))
            assert abs(eps - js['E_cut_abs']) < 1e-9, (key, eps)
            # gate (d): singleton lower bound
            E_single = abs(-float(BCLs[s, j]) - (-float(BC_SUE[s, j]))) \
                / abs(-float(BC_SUE[s, j]))
            assert eps >= E_single * abs(-float(BC_SUE[s, j])) - 1e-9

            phi_s_sue = -float(BC_SUE[s, j])
            # eps/span against the DIRECT connected span (Table-5
            # convention): direct interval from frozen columns
            crS = enum_c(lib, adj, n, s, t, BC_SUE[:, j:j + 1])
            span_direct = float(crS['phi_max'][0] - crS['phi_min'][0])

            # margins at the joint extremal boundaries
            entry = {
                'Cj_learned': cj_learned,
                'Cj_learned_rel_dev': float(
                    (cj_learned - float(Cj_sue[j])) / abs(float(Cj_sue[j]))),
                'learned_interval': [lo, hi],
                'straddles_zero': bool(lo < 0.0 < hi),
                'eps_cut': eps,
                'E_cut_over_origin': float(eps / abs(phi_s_sue)),
                'eps_over_span': float(eps / span_direct),
            }
            if key in CROSSING:
                m = e91['mechanism'][key]
                margins = {}
                for tag in ('U_minus', 'U_plus'):
                    nodes = np.array(m[f'{tag}_nodes_1idx']) - 1
                    margins[tag] = -float(BC_SUE[nodes, j].sum())
                m_j = float(min(-margins['U_minus'],
                                margins['U_plus']) - eps)
                entry['margin'] = {
                    'phi_sue_U_minus': margins['U_minus'],
                    'phi_sue_U_plus': margins['U_plus'],
                    'm_j': m_j,
                    'm_j_positive': bool(m_j > 0.0),
                    'same_seed_proposition': bool(
                        cj_learned > 0.0 and lo < 0.0 < hi),
                }
            od_out[key] = entry

            # (4) attribution-column error decomposition
            diff = CLs[:, j] - C_SUE[:, j]
            col_err[(seed, key)] = {
                'rel_pct': float(100.0 * np.linalg.norm(diff)
                                 / np.linalg.norm(C_SUE[:, j])),
                'abs_l2': float(np.linalg.norm(diff)),
                'ref_l2': float(np.linalg.norm(C_SUE[:, j])),
            }
        per_seed[seed] = {'completeness_relative': compl, 'per_od': od_out}
        print(f'seed {seed}: completeness {compl:.2e}', flush=True)

    # ---- summaries over seeds ----
    summary = {}
    for j, key in enumerate(keys):
        cjs = [per_seed[s]['per_od'][key]['Cj_learned'] for s in SEEDS]
        los = [per_seed[s]['per_od'][key]['learned_interval'][0]
               for s in SEEDS]
        his = [per_seed[s]['per_od'][key]['learned_interval'][1]
               for s in SEEDS]
        epss = [per_seed[s]['per_od'][key]['eps_cut'] for s in SEEDS]
        entry = {
            'Cj_learned_min': float(min(cjs)), 'Cj_learned_max': float(max(cjs)),
            'Cj_all_positive': bool(min(cjs) > 0.0),
            'Cj_sue': float(Cj_sue[j]),
            'learned_interval_envelope': [float(min(los)), float(max(his))],
            'every_seed_straddles_zero': bool(all(lo < 0.0 < hi
                                                  for lo, hi in zip(los, his))),
            'eps_cut_min': float(min(epss)), 'eps_cut_max': float(max(epss)),
            'eps_over_span_seed_medians': float(np.median(
                [per_seed[s]['per_od'][key]['eps_over_span'] for s in SEEDS])),
        }
        if key in CROSSING:
            ms = [per_seed[s]['per_od'][key]['margin']['m_j'] for s in SEEDS]
            entry['margin'] = {
                'per_seed': [float(x) for x in ms],
                'worst_seed_m': float(min(ms)),
                'all_positive': bool(min(ms) > 0.0),
                'same_seed_proposition_all_seeds': bool(all(
                    per_seed[s]['per_od'][key]['margin']
                    ['same_seed_proposition'] for s in SEEDS)),
            }
        summary[key] = entry

    # (4) located maximum attribution-column error
    best = max(col_err.items(), key=lambda kv: kv[1]['rel_pct'])
    (mx_seed, mx_key), mx = best
    ref_all = {k: col_err[(0, k)]['ref_l2'] for k in keys}
    col_err_table = {f'seed{s}|{k}': v for (s, k), v in col_err.items()}

    out = {
        'protocol': {
            'purpose': 'complete C_j>0 AND phi(U_-)<0 on the same learned '
                       'model/seed; formal epsilon^cut and margin m_j; '
                       'locate the max attribution-column error; frozen '
                       'arrays only, no retraining, no new solves',
            'sources': ['e93_v89_fullbg_surrogate.npz (C_L, f_L)',
                        'e87_v85_newod_refine.npz (C_41)',
                        'e91_v87_negative_mechanism.json (U sets)',
                        'e80 connected enumeration'],
            'claims_supported': 'same-seed sign proposition with margin; '
                                'attribution columns not uniformly accurate '
                                '(max relative error on smallest reference '
                                'column); no general attribution-precision '
                                'guarantee',
        },
        'per_seed': {str(s): {'completeness_relative':
                              per_seed[s]['completeness_relative'],
                              'per_od': per_seed[s]['per_od']}
                     for s in SEEDS},
        'summary': summary,
        'max_column_error': {
            'seed': mx_seed, 'od': mx_key, **mx,
            'ref_l2_all_queries': ref_all,
            'smallest_ref_query': min(ref_all, key=ref_all.get),
        },
        'column_error_table': col_err_table,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'written -> {OUT}', flush=True)
    print('max col err:', mx_key, 'seed', mx_seed,
          f"{mx['rel_pct']:.2f}% abs {mx['abs_l2']:.1f} "
          f"ref {mx['ref_l2']:.1f}", flush=True)


if __name__ == '__main__':
    main()
