"""E103 (v94, external review round 66, section IV priority statistic):
negative-account share r_j^- over each crossing query's EXACT connected
boundary family, plus the pre-specified simple-boundary families
extended from the three crossing queries to all six study queries.

Motivation (review): the paper reports that negative accounts EXIST in
the connected family and do NOT occur in the simple origin-centred
families, but not whether negatives are a tiny minority or a substantial
share of the connected family.  That share cannot be inferred from the
minimum and maximum.  This experiment re-aggregates the frozen e87
41-point attribution columns only: no new equilibrium solves, no
retraining, no new demand scenarios.

Definition (per review):
  r_j^- = #{U in U_j^conn : phi_j(U) < 0} / |U_j^conn|,
reported as the share of negative accounts in the ENUMERATED connected
family -- NOT a probability of an analyst selecting a negative boundary
(no boundary-selection model is assumed).

Gates:
  (a) frozen columns reproduce the published origin-cut accounts and
      C_j of Table 4 (0.05 absolute, as in e96/e99) for all six queries;
  (b) the pure-Python counting enumerator reproduces the C enumerator's
      n_sets and phi_min/phi_max exactly (1e-6 relative), for every
      query;
  (c) connected min/max reproduce e87's stored 41-point intervals;
  (d) phi(U) is computed twice per counted set stream: node-sum
      -sum_{v in U} BC[v,j]; spot-checked per attaining mask by direct
      signed crossing-link summation (1e-6);
  (e) the three crossing queries' pre-specified-family minima/maxima
      reproduce e99 (1e-9).

Output: experiments/results/e103_v94_negative_share.json
        console log experiments/results/e103_console.log
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
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e80_connected_cuts import (  # noqa: E402
    build_adj_bits, connected, load_c_enum, enum_c, mask_to_nodes)
from run_e99_v91_predefined_boundaries import (  # noqa: E402
    bfs_distances, dijkstra_fft, phi_nodesum, phi_linksum)

RESULTS = HERE / 'results'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E87_JSON = RESULTS / 'e87_v85_newod_refine.json'
E99_JSON = RESULTS / 'e99_v91_predefined_boundaries.json'
OUT = RESULTS / 'e103_v94_negative_share.json'

CROSSING = ('1->19', '1->15', '7->12')


def enum_count(adj, n, s, t, col):
    """Pure-Python enumeration of the connected two-sided family with
    negative-set counting.  Same ESU discipline as run_e80.enum_python:
    every connected U containing s (t excluded) with connected
    complement is generated exactly once; here we additionally count
    sets with phi < 0 and record the attaining masks of both extrema."""
    full = (1 << n) - 1
    n_sets = 0
    n_neg = 0
    best_min = np.inf
    best_max = -np.inf
    min_mask = max_mask = None

    def process(mask, acc):
        nonlocal n_sets, n_neg, best_min, best_max, min_mask, max_mask
        n_sets += 1
        if acc < 0.0:
            n_neg += 1
        if acc < best_min:
            best_min = acc
            min_mask = mask
        if acc > best_max:
            best_max = acc
            max_mask = mask

    def expand(S, cand, excl, acc):
        popped = 0
        while cand:
            b = cand & (-cand)
            v = b.bit_length() - 1
            cand ^= b
            if v == t:
                popped |= b
                continue
            S2 = S | b
            acc2 = acc - col[v]
            if connected(adj, full & ~S2):
                process(S2, acc2)
            expand(S2, (cand | adj[v]) & ~S2 & ~(excl | popped),
                   excl | popped, acc2)
            popped |= b

    if connected(adj, full & ~(1 << s)):
        process(1 << s, -col[s])
    expand(1 << s, adj[s] & ~((1 << s) | (1 << t)),
           1 << t, -col[s])
    return {'n_sets': n_sets, 'n_neg': n_neg,
            'phi_min': best_min, 'phi_max': best_max,
            'min_mask': min_mask, 'max_mask': max_mask}


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, _, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    z87 = np.load(E87_NPZ)
    C = z87['C_41']                                  # (76, 6)
    BC = B @ C
    e87 = json.load(open(E87_JSON))
    e99 = json.load(open(E99_JSON))

    adj, n = build_adj_bits(net)
    edges = list(net.edges)
    lib = load_c_enum()
    full = (1 << n) - 1

    # ---- gate (a): published origin-cut accounts and C_j (Table 4) ----
    published = {'7->18': (671.2, 672.9), '1->19': (1771.6, 50097.7),
                 '4->5': (2659.4, 2826.0), '1->15': (2948.1, 77403.2),
                 '6->8': (48126.8, 73964.9), '7->12': (9473.8, 117482.3)}
    for key, (phi_s_pub, cj_pub) in published.items():
        j = keys.index(key)
        s, t = study_ods[j]
        assert abs(-float(BC[s, j]) - phi_s_pub) < 0.05, (key, 'phi_s')
        assert abs(float(C[:, j].sum()) - cj_pub) < 0.05, (key, 'C_j')

    per_od = {}
    for key in keys:
        j = keys.index(key)
        s, t = study_ods[j]
        col = BC[:, j].copy()

        # ---- C enumerator (e80 machinery): n_sets and extrema ----
        res_c = enum_c(lib, adj, n, s, t, col[:, None])
        n_sets_c = int(res_c['n_sets'])
        cmin = float(res_c['phi_min'][0])
        cmax = float(res_c['phi_max'][0])

        # ---- pure-Python counting enumerator: gates (b) ----
        res_p = enum_count(adj, n, s, t, col)
        assert res_p['n_sets'] == n_sets_c, (key, res_p['n_sets'], n_sets_c)
        for a, b_ in ((res_p['phi_min'], cmin), (res_p['phi_max'], cmax)):
            assert abs(a - b_) <= 1e-6 * max(1.0, abs(b_)), (key, a, b_)

        # ---- gate (c): e87 stored 41-point interval ----
        i41 = e87['per_od'][key]['interval_41']
        assert abs(cmin - i41['connected']['min']) <= 1e-6 * max(1.0, abs(cmin)), key
        assert abs(cmax - i41['connected']['max']) <= 1e-6 * max(1.0, abs(cmax)), key
        assert n_sets_c == i41['connected']['n_cuts'], (key, n_sets_c)

        # ---- gate (d): attaining masks re-summed by direct link sum ----
        for msk, val in ((res_p['min_mask'], cmin), (res_p['max_mask'], cmax)):
            nodes0 = [v - 1 for v in mask_to_nodes(msk)]
            assert s in nodes0 and t not in nodes0, (key, 'mask endpoints')
            assert connected(adj, msk) and connected(adj, full & ~msk), (key, 'mask conn')
            phi_n = -float(col[nodes0].sum())
            phi_l = phi_linksum(BC, C, B, j, nodes0, edges)
            assert abs(phi_n - val) <= 1e-6 * max(1.0, abs(val)), (key, phi_n, val)
            assert abs(phi_l - phi_n) <= 1e-6, (key, phi_l, phi_n)

        share = res_p['n_neg'] / res_p['n_sets']
        per_od[key] = {
            'n_connected': n_sets_c,
            'n_negative': res_p['n_neg'],
            'negative_share': share,
            'phi_min': cmin,
            'phi_max': cmax,
            'origin_cut': -float(BC[s, j]),
            'C_j': float(C[:, j].sum()),
        }
        print(f"{key}: n_conn={n_sets_c}  n_neg={res_p['n_neg']}  "
              f"r-={share:.4%}  conn=[{cmin:,.1f}, {cmax:,.1f}]")

        # ---- pre-specified families for ALL six queries ----
        hop = bfs_distances(adj, n, s)
        fft = dijkstra_fft(net, adj, n, s)
        candidates = []
        for d in sorted({x for x in hop if x is not None}):
            nodes0 = [v for v in range(n) if hop[v] is not None
                      and hop[v] <= d]
            if t in nodes0:
                break
            candidates.append(('hop', float(d), nodes0))
        for r in sorted({x for x in fft if x < float('inf')}):
            nodes0 = [v for v in range(n) if fft[v] <= r + 1e-12]
            if t in nodes0:
                break
            candidates.append(('fft', float(r), nodes0))

        admissible = []
        n_rejected = 0
        seen = set()
        for family, radius, nodes0 in candidates:
            mask = 0
            for v in nodes0:
                mask |= 1 << v
            ok = (s in nodes0) and (t not in nodes0) \
                and connected(adj, mask) and connected(adj, full & ~mask)
            if not ok:
                n_rejected += 1
                continue
            phi_n = phi_nodesum(BC, j, nodes0)
            phi_l = phi_linksum(BC, C, B, j, nodes0, edges)
            assert abs(phi_n - phi_l) <= 1e-6, (key, family, radius)
            keyset = tuple(sorted(nodes0))
            dup = keyset in seen
            seen.add(keyset)
            admissible.append({
                'family': family, 'radius': radius,
                'n_nodes': len(nodes0), 'phi': phi_n,
                'dedup': dup,
            })
        phis = [a['phi'] for a in admissible]
        fam = {
            'n_candidates': len(candidates),
            'n_admissible': len(admissible),
            'n_rejected_inadmissible': n_rejected,
            'n_distinct_sets': len(seen),
            'any_negative': bool(phis and min(phis) < 0.0),
            'min_phi': min(phis) if phis else None,
            'max_phi': max(phis) if phis else None,
        }
        per_od[key]['predefined'] = fam
        print(f"  predefined: n_adm={fam['n_admissible']} "
              f"distinct={fam['n_distinct_sets']} "
              f"any_neg={fam['any_negative']} "
              f"range=[{fam['min_phi']:,.1f}, {fam['max_phi']:,.1f}]")

        # ---- gate (e): crossing queries reproduce e99 ----
        if key in CROSSING:
            e99od = e99['per_od'][key]
            assert fam['n_admissible'] == e99od['n_admissible'], key
            assert abs(fam['min_phi'] - e99od['min_phi']) <= 1e-9, key
            assert abs(fam['max_phi'] - e99od['max_phi']) <= 1e-9, key

    out = {
        'protocol': {
            'purpose': 'negative-account share over the exact connected '
                       'family and pre-specified simple-boundary families '
                       'for all six study queries, re-aggregated from the '
                       'frozen e87 41-point columns',
            'definition': 'r_j^- = #{U connected two-sided: phi_j(U)<0} '
                          '/ |connected family|; a share within the '
                          'enumerated family, NOT a boundary-selection '
                          'probability',
            'columns': 'e87 C_41 (setting C, six-query joint protocol, '
                       '41 quadrature points)',
            'claims_supported': 'how large the negative-account subset is '
                                'within each enumerated connected family; '
                                'the simple-family ranges for all six '
                                'queries; NOT analyst-selection odds',
            'no_new_solves': True,
        },
        'gates': {
            'table4_phi_s_Cj_max_abs': 0.05,
            'py_vs_c_n_sets_and_extrema': 'exact / 1e-6 rel',
            'e87_interval_match': '1e-6 rel',
            'attaining_masks_direct_link_sum': '1e-6',
            'e99_crossing_family_match': '1e-9',
        },
        'per_od': per_od,
        'elapsed_seconds': time.time() - started,
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"e103 written: {OUT} ({out['elapsed_seconds']:.1f}s)")


if __name__ == '__main__':
    main()
