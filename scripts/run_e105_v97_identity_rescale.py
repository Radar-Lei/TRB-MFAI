"""E105 (v97, external review round 69, item III): re-summarize the E97
per-edge endpoint identity residuals under two clearly separated metrics.

Background.  The v96 text reported the identity check as "matches its own
endpoint flows to 1e-9 relative l2 per column" alongside absolute per-edge
deviations and their ratio to the largest per-edge cost change.  The 1e-9
figure is the E97 REPRODUCIBILITY GATE (recomputed column vs frozen E95
column), not an error of the identity of Eq. (15); the round-69 external
review correctly noted the two numbers cannot describe the same comparison
(an l2 relative error below 1e-9 would contradict an l-inf relative error
of 8.0e-7 on 76 edges: r2 >= r_inf/sqrt(76) ~ 9.2e-8).

This script performs a PURE RE-SUMMARIZATION from the frozen E97 artifacts
(no equilibrium solves, no retraining, no recomputation of any published
quantity) and reports, per focal OD,

    r_{2,j}  = || C~_{:,j} - dg_j ||_2 / || dg_j ||_2
               (relative column error, dg_j = g(F1) - g(F0) per edge),
    a_{inf,j}= || C~_{:,j} - dg_j ||_inf
               (absolute max per-edge error, TSTT units),

together with the scalar cross-checks already gated in E97: the residual
sum vs the published completeness gap, and a_inf/max|dg|.

Gates:
  (a) recomputed r2 >= resid_inf_over_max_dg / sqrt(76)
      (consistency with the published 8.0e-7 statement);
  (b) a_inf and resid_l2 match the E97 JSON to 1e-12;
  (c) residual-sum vs completeness-gap relative difference <= 1e-6.

Output: experiments/results/e105_v97_identity_rescale.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
E97_JSON = RESULTS / 'e97_v91_singleod_identity.json'
E97_NPZ = RESULTS / 'e97_v91_singleod_identity.npz'
OUT = RESULTS / 'e105_v97_identity_rescale.json'

FOCAL = ('1->19', '1->15', '7->12', '6->8')
N_EDGES = 76


def main():
    started = time.time()
    d97 = json.loads(E97_JSON.read_text())
    z97 = np.load(E97_NPZ)

    per_od = {}
    for key in FOCAL:
        kk = key.replace('->', '_')
        C = z97[f'C_{kk}']
        dg = z97[f'dg_{kk}']
        resid = C - dg
        row = d97['per_od'][key]

        r2 = float(np.linalg.norm(resid) / np.linalg.norm(dg))
        a_inf = float(np.max(np.abs(resid)))
        resid_l2 = float(np.linalg.norm(resid))
        gap = float(row['completeness_gap_published'])
        sum_gap_rel = float(abs(resid.sum() - gap) / max(abs(gap), 1e-12))

        # gate (b): agree with E97 JSON to 1e-12
        assert abs(resid_l2 - row['resid_l2']) <= 1e-12, (key, 'l2')
        assert abs(a_inf - row['resid_inf']) <= 1e-12, (key, 'inf')

        # gate (a): consistency with the published max-relative statement
        lower = row['resid_inf_over_max_dg'] / np.sqrt(N_EDGES)
        assert r2 >= lower, (key, 'l2/inf consistency', r2, lower)

        # gate (c): scalar completeness cross-check
        assert sum_gap_rel <= 1e-6, (key, 'gap', sum_gap_rel)

        per_od[key] = {
            'r2_relative_column': r2,
            'a_inf_tstt': a_inf,
            'resid_l2_tstt': resid_l2,
            'norm_dg_l2_tstt': float(np.linalg.norm(dg)),
            'a_inf_over_max_dg': float(row['resid_inf_over_max_dg']),
            'resid_sum': float(resid.sum()),
            'completeness_gap': gap,
            'sum_vs_gap_rel': sum_gap_rel,
        }
        print(f"{key}: r2={r2:.4g}, a_inf={a_inf:.4g} TSTT, "
              f"a_inf/max|dg|={row['resid_inf_over_max_dg']:.3g}, "
              f"sum-vs-gap rel={sum_gap_rel:.2g}", flush=True)

    out = {
        'protocol': {
            'purpose': 're-summarized endpoint-identity metrics with '
                       'separated scopes: relative column error vs absolute '
                       'max per-edge error; pure frozen-artifact '
                       're-summarization of E97 (no solves, no retraining)',
            'focal_ods': list(FOCAL),
            'source_artifacts': [str(E97_JSON.name), str(E97_NPZ.name)],
            'metric_definitions': {
                'r2_relative_column':
                    '||C~_{:,j} - dg_j||_2 / ||dg_j||_2, '
                    'dg_j = g(F1) - g(F0) per edge (Eq. 15 object)',
                'a_inf_tstt':
                    '||C~_{:,j} - dg_j||_inf, TSTT units',
            },
            'claims_supported':
                'the v96 sentence conflated the E97 reproducibility gate '
                '(recomputed vs frozen column, <=1e-9) with the identity '
                'error; the identity error is r2 in '
                f"{min(v['r2_relative_column'] for v in per_od.values()):.2g}"
                '--'
                f"{max(v['r2_relative_column'] for v in per_od.values()):.2g}"
                ' relative l2, consistent with the published '
                'a_inf/max|dg| <= 8.1e-7 via r2 >= that/sqrt(76) (rounded values are never stated as bounds; raw max 8.0434e-7)',
        },
        'per_od': per_od,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'written -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
