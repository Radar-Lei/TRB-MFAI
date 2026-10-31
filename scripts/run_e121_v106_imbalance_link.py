"""E121 (v106): node-imbalance norms of the node representation versus realized
spatial ranges and common-summary errors.

External review of v105 (P1): the load-level experiment shows median
normalized summary errors rising with baseline load, but the explanation
is descriptive. The node representation supplies theory quantities for
the same columns: with the endpoint attribution column C_j (single-OD
identity), b_j = B C_j, d_j = (I - P_{mu_j}) b_j, the unrestricted range
S_j = sum |b_jv| over interior nodes and the Cauchy-Schwarz bound
S_j <= sqrt(n-2) ||d_j||_2 (Eq. rangebound). This experiment recomputes
||d_j||_2 and S_j for the 18 stratified queries at baseline levels
0.7/1.0/1.3 (levels 1.0 from e114 halfwidth 0.2; 0.7 and 1.3 from e116)
and relates them to the stored geometric-family ranges and normalized
common-summary errors. No new equilibrium solves; gate: recomputed S_j
must equal the stored unrestricted_spread of each row.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e121_v106_imbalance_link'
OUT.mkdir(exist_ok=True)


def analyse(B, C, s, t, row):
    b = B @ C
    interior = [v for v in range(B.shape[0]) if v not in (s, t)]
    S = float(np.abs(b[interior]).sum())
    mu = np.zeros(B.shape[0])
    mu[t], mu[s] = 1, -1
    d = b - mu * (mu @ b) / (mu @ mu)
    dn = float(np.linalg.norm(d))
    gmin, gmax = row['geometric_min'], row['geometric_max']
    mag = max(abs(gmin), abs(gmax))
    rel = (gmax - gmin) / 2 / mag if mag > 0 else 0.0
    return {'od': row['od'], 'level': row.get('level', 1.0),
            'd_norm': dn, 'S_unrestricted': S,
            'bound_sqrt_n_minus_2': float(np.sqrt(B.shape[0] - 2)),
            'S_over_d_norm': float(S / dn) if dn > 0 else 0.0,
            'gate_S_vs_stored': float(abs(S - row['unrestricted_spread'])),
            'geometric_spread': row['geometric_spread'],
            'geometric_over_unrestricted': row['geometric_to_all_spread'],
            'normalized_summary_error': float(rel)}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    e114 = json.loads((ROOT / 'experiments/results/e114_v103/results.json').read_text())
    e116 = json.loads((ROOT / 'experiments/results/e116_v105_loadlevels/results.json').read_text())
    a114 = np.load(ROOT / 'experiments/results/e114_v103/arrays.npz')
    a116 = np.load(ROOT / 'experiments/results/e116_v105_loadlevels/arrays.npz')
    B = a114['B']
    rows = []
    for row in e114['rows']:
        if row['halfwidth'] != .2:
            continue
        s, t = row['od'][0] - 1, row['od'][1] - 1
        rows.append(analyse(B, a114[f"C_{row['index']}_0.2"], s, t, row))
    for row in e116['rows']:
        s, t = row['od'][0] - 1, row['od'][1] - 1
        rows.append(analyse(B, a116[f"C_{row['level']}_{row['index']}"], s, t, row))
    maxgate = max(r['gate_S_vs_stored'] for r in rows)
    dn = np.array([r['d_norm'] for r in rows])
    gs = np.array([r['geometric_spread'] for r in rows])
    ne = np.array([r['normalized_summary_error'] for r in rows])
    Sx = np.array([r['S_unrestricted'] for r in rows])
    lev = np.array([r['level'] for r in rows])
    by_level = {str(l): {'median_d_norm': float(np.median(dn[lev == l])),
                         'median_S': float(np.median(Sx[lev == l])),
                         'median_normalized_error': float(np.median(ne[lev == l])),
                         'median_geometric_spread': float(np.median(gs[lev == l]))}
                for l in (0.7, 1.0, 1.3)}
    out = {'rows': rows, 'by_level': by_level,
           'spearman_d_norm_vs_geometric_spread': spearman(dn, gs),
           'spearman_d_norm_vs_normalized_error': spearman(dn, ne),
           'spearman_S_vs_geometric_spread': spearman(Sx, gs),
           'bound_tightness_median_S_over_d_norm': float(np.median(Sx / dn)),
           'gate_max_S_vs_stored': maxgate,
           'source': 'e114_v103 halfwidth-0.2 columns (level 1.0) + e116_v105_loadlevels (0.7, 1.3); no new solves'}
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
