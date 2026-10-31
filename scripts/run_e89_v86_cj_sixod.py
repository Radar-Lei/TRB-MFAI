"""E89 (v86, external review round 57, item A2): scalar system
contribution C_j for the six pre-specified queries away from node 10.

C_j = sum_e Ctilde_{e,j} is summed from the PUBLISHED 41-point
attribution columns (e87 npz C_41), cross-checked against the e86
41-point columns, with the e87 81-point refinement columns as a
stability reference.  No new equilibrium solve, no new training.  Signs
and magnitudes are taken as computed; none are presupposed.

For the main-text narrative this script also verifies the published
origin-cut accounts and the sign-straddling attainable intervals
(e87 json), and checks completeness: sum_j C_j against the endpoint
TSTT change T(F(q1)) - T(F(q0)) on the 81-point chain.

Gates (must pass before any number enters the paper):
  (a) e87 C_41 vs e86 C (76,6) agree to < 1e-10 relative l2 per column
      (both are the published 41-point chain);
  (b) phi_s derived from e87 BC_41 reproduces the published Table-5
      origin accounts to < 1e-9;
  (c) completeness: |sum_j C_j^(81) - (T(F1)-T(F0))| / |T(F1)-T(F0)|
      < 5e-7 on the 81-grid endpoints (published gap 1.07e-7; the
      41-grid gap 4.27e-7 also passes and is reported in the JSON);
  (d) the six published intervals straddle zero exactly for
      (1,19), (1,15), (7,12) in both families, and only for them.

Output: experiments/results/e89_v86_cj_sixod.json
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

RESULTS = HERE / 'results'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E87_JSON = RESULTS / 'e87_v85_newod_refine.json'
E86_NPZ = RESULTS / 'e86_v84_newod.npz'
OUT = RESULTS / 'e89_v86_cj_sixod.json'


def main():
    started = time.time()
    net = load_sioux_falls()
    T = TSTTT(net.free_flow_time, net.capacity, net.m)

    z87 = np.load(E87_NPZ)
    z86 = np.load(E86_NPZ)
    d87 = json.loads(E87_JSON.read_text())

    C41, C81 = z87['C_41'], z87['C_81']       # (76, 6) each
    BC41 = z87['BC_41']                        # (24, 6)
    F81 = z87['F']                             # (81, 76)
    ods_1idx = list(d87['per_od'].keys())      # '7->18', ...
    study_ods = [tuple(int(x) for x in k.split('->')) for k in ods_1idx]

    # gate (a): the two published 41-point column sets agree
    rel41 = float(np.max(np.linalg.norm(C41 - z86['C'], axis=0)
                         / np.linalg.norm(z86['C'], axis=0)))
    assert rel41 < 1e-10, f'gate (a) failed: {rel41:.2e}'

    # gate (b): origin accounts from BC_41 match the published table
    for j, od in enumerate(study_ods):
        s = od[0] - 1
        phi = float(-BC41[s, j])
        ref = d87['per_od'][f'{od[0]}->{od[1]}']['phi_s_41']
        assert abs(phi / ref - 1) < 1e-9, f'gate (b) failed at {od}'

    # scalar system contributions (the object under review)
    Cj41 = C41.sum(axis=0)                     # (6,)
    Cj81 = C81.sum(axis=0)

    # gate (c): completeness on the 81-point chain endpoints
    dT = float(T.value(F81[-1]) - T.value(F81[0]))
    gap = float(abs(Cj81.sum() - dT) / abs(dT))
    assert gap < 5e-7, f'gate (c) failed: completeness gap {gap:.2e}'
    gap41 = float(abs(Cj41.sum() - dT) / abs(dT))
    assert gap41 < 5e-7, f'gate (c) failed: 41-grid completeness gap {gap41:.2e}'

    rows = {}
    straddle = {}
    for j, od in enumerate(study_ods):
        key = f'{od[0]}->{od[1]}'
        rec = d87['per_od'][key]
        iv41 = rec['interval_41']
        fam = {}
        for name in ('unrestricted', 'connected'):
            lo, hi = iv41[name]['min'], iv41[name]['max']
            fam[name] = {'min': lo, 'max': hi,
                         'straddles_zero': bool(lo < 0.0 < hi)}
        straddle[key] = fam
        rows[key] = {
            'od_1idx': [od[0], od[1]],
            'C_j_41': float(Cj41[j]),
            'C_j_81': float(Cj81[j]),
            'C_j_change_pct': float(abs(Cj81[j] / Cj41[j] - 1) * 100),
            'phi_s_41': float(-BC41[od[0] - 1, j]),
            'interval_41': fam,
        }

    # gate (d): exactly (1,19), (1,15), (7,12) straddle zero, both families
    expected = {'1->19', '1->15', '7->12'}
    got = {k for k, f in straddle.items()
           if f['unrestricted']['straddles_zero']
           and f['connected']['straddles_zero']}
    assert got == expected, f'gate (d) failed: {sorted(got)}'

    out = {
        'protocol': {
            'purpose': 'scalar system contribution C_j = sum_e Ctilde_ej for '
                       'the six pre-specified queries (review round 57, '
                       'item A2); re-summarization of published columns, '
                       'no new solves',
            'columns': 'e87 published 41-point columns; e86 cross-check; '
                       'e87 81-point refinement stability',
            'completeness_denominator_dT_81': dT,
        },
        'gates': {
            'e87_vs_e86_C41_max_relative_l2': rel41,
            'phi_s_from_BC41_matches_published': True,
            'completeness_relative_81': gap,
            'completeness_relative_41': gap41,
            'straddling_queries_both_families': sorted(got),
        },
        'per_od': rows,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'Completed e89 in {out["elapsed_seconds"]:.1f}s -> {OUT}', flush=True)
    print(json.dumps({k: {'C_j_41': round(v['C_j_41'], 2),
                          'phi_s_41': round(v['phi_s_41'], 1),
                          'straddle_both': v['interval_41']['unrestricted']['straddles_zero']
                          and v['interval_41']['connected']['straddles_zero']}
                      for k, v in rows.items()}, indent=1), flush=True)


if __name__ == '__main__':
    main()
