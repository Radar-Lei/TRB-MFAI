"""E136 (v109): paired bootstrap uncertainty for the Anaheim learning gain.

External review of v108 (P0-3): Anaheim's response-vs-flow improvement is
smaller than Sioux Falls', so uncertainty evidence matters more than a
third decimal.  This script consumes the frozen e131 per-path rows (no
retraining, no re-selection) and reports, for each of column_pct and
geometry_pct:

  * pooled path-level paired median difference (response - flow), seeds
    averaged within each path first, matching the e131 protocol;
  * 95% bootstrap confidence interval over the 48 paths (10,000
    resamples, fixed seed);
  * improvement count (paths with response < flow);
  * per-seed path-level medians and direction consistency.

A (path, query) level variant is reported as a secondary summary.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / 'results/e131_v108_anaheim_eval'
OUT = HERE / 'results/e136_v109_anaheim_bootstrap'
OUT.mkdir(exist_ok=True)
N_BOOT = 10_000
METRICS = ('column_pct', 'geometry_pct')
N_PATHS = 48


def path_level(rows, method, seed=None):
    """Average active queries within each path (protocol order)."""
    acc = {}
    for r in rows:
        if r['method'] != method:
            continue
        if seed is not None and r['seed'] != seed:
            continue
        acc.setdefault(r['path'], []).append(r)
    return {pi: {k2: float(np.mean([x[k2] for x in v])) for k2 in METRICS}
            for pi, v in acc.items()}


def paired(rows, seed=None):
    fl = path_level(rows, 'flow', seed)
    rs = path_level(rows, 'response', seed)
    out = {}
    for k2 in METRICS:
        out[k2] = np.array([rs[pi][k2] - fl[pi][k2]
                            for pi in sorted(fl) if pi in rs])
    return out


def boot_ci(d, rng):
    n = len(d)
    meds = np.empty(N_BOOT)
    for b in range(N_BOOT):
        meds[b] = np.median(d[rng.integers(0, n, n)])
    return float(np.quantile(meds, .025)), float(np.quantile(meds, .975))


def main():
    rows = json.loads((SRC / 'rows.json').read_text())
    rng = np.random.default_rng(20261031)

    pooled = paired(rows)
    pooled_summary = {}
    for k2, d in pooled.items():
        lo, hi = boot_ci(d, rng)
        pooled_summary[k2] = {
            'median_difference': float(np.median(d)),
            'ci95': [lo, hi],
            'improved_paths': int(np.sum(d < 0)),
            'paths': int(len(d)),
            'ci_excludes_zero': bool(hi < 0 or lo > 0),
        }

    per_seed = []
    direction = {k2: 0 for k2 in METRICS}
    for seed in range(5):
        pr = paired(rows, seed)
        entry = {'seed': seed}
        for k2, d in pr.items():
            lo, hi = boot_ci(d, rng)
            entry[k2] = {
                'median_difference': float(np.median(d)),
                'ci95': [lo, hi],
                'improved_paths': int(np.sum(d < 0)),
                'paths': int(len(d)),
            }
            if np.median(d) < 0:
                direction[k2] += 1
        per_seed.append(entry)

    # secondary: (path, query) level pairing, pooled over seeds
    acc = {}
    for r in rows:
        acc.setdefault((r['method'], r['path'], r['query']), []).append(r)
    pq = {}
    for k2 in METRICS:
        dif = []
        for (mm, pi, j), v in acc.items():
            if mm != 'flow':
                continue
            if ('response', pi, j) not in acc:
                continue
            f = float(np.mean([x[k2] for x in v]))
            g = float(np.mean([x[k2] for x in acc[('response', pi, j)]]))
            dif.append(g - f)
        dif = np.array(dif)
        lo, hi = boot_ci(dif, rng)
        pq[k2] = {
            'median_difference': float(np.median(dif)),
            'ci95': [lo, hi],
            'improved_pairs': int(np.sum(dif < 0)),
            'pairs': int(len(dif)),
        }

    out = {
        'protocol': ('e131 frozen rows; paths are the resampling unit; seeds '
                     'averaged within path before pairing (primary) and within '
                     '(path, query) before seed averaging (secondary); '
                     f'{N_BOOT} bootstrap resamples, seed 20261031'),
        'pooled_path_level': pooled_summary,
        'per_seed_path_level': per_seed,
        'seeds_with_negative_median': {k2: f'{direction[k2]}/5' for k2 in METRICS},
        'secondary_path_query_level': pq,
    }
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'pooled': pooled_summary,
                      'seeds_neg': out['seeds_with_negative_median']}, indent=2))


if __name__ == '__main__':
    main()
