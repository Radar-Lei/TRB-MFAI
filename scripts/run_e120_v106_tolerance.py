"""E120 (v106): tolerance-compression frontier over ordered boundary accounts.

External review of v105 (P1): the 5% reporting tolerance is a stated
descriptive choice; a reviewer will ask how conclusions depend on it.
This experiment reuses the stored per-region account sequences (no new
equilibrium solves) and sweeps tau in {2.5, 5, 7.5, 10}%:
  - number of queries whose family admits one common value (Eq. normerror
    relative error <= tau);
  - minimum number of contiguous segments (greedy longest-feasible-prefix,
    same procedure and progress ordering as e118) and the compression
    1 - segments/regions;
  - median equal-count-partition worst-segment error ratio (greedy vs
    equal-count at the same segment count).
Data: e114 (level 1.0, halfwidth 0.2), e116 (levels 0.7 and 1.3), and
the Anaheim distance-progress profiles of v105. Ordering for Sioux Falls
is the e118 free-flow-progress order of the farthest contained node.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
from pathlib import Path
import numpy as np
import networkx as nx
from nets import load_sioux_falls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e120_v106_tolerance'
OUT.mkdir(exist_ok=True)
TOLS = (0.025, 0.05, 0.075, 0.10)


def greedy_segments(a, eps):
    segs, i, n = [], 0, len(a)
    while i < n:
        j = i
        lo = hi = a[i]
        while j + 1 < n:
            nlo, nhi = min(lo, a[j + 1]), max(hi, a[j + 1])
            if nhi - nlo > 2 * eps or (nlo < 0 < nhi):
                break
            lo, hi = nlo, nhi
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def equal_max_eps(a, K):
    n = len(a)
    out = []
    for k in range(K):
        i, j = int(round(k * n / K)), int(round((k + 1) * n / K)) - 1
        s = a[i:j + 1]
        out.append((s.max() - s.min()) / 2)
    return max(out)


def sweep(a_sorted):
    mag = max(abs(a_sorted.min()), abs(a_sorted.max()))
    rel = (a_sorted.max() - a_sorted.min()) / 2 / mag if mag > 0 else 0.0
    per = []
    for tau in TOLS:
        eps = tau * mag
        segs = greedy_segments(a_sorted, eps)
        worst = max((a_sorted[i:j + 1].max() - a_sorted[i:j + 1].min()) / 2 for i, j in segs)
        eq = equal_max_eps(a_sorted, len(segs))
        per.append({'tau': tau, 'single_value_ok': bool(rel <= tau), 'segments': len(segs),
                    'compression': float(1 - len(segs) / len(a_sorted)),
                    'greedy_worst_epsilon': float(worst), 'equal_count_worst_epsilon': float(eq),
                    'equal_over_greedy': float(eq / worst) if worst > 0 else 0.0,
                    'equal_over_tau': float(eq / eps) if eps > 0 else 0.0})
    return rel, per


def main():
    net = load_sioux_falls()
    g = nx.Graph()
    for (u, v), w in zip(net.edges, net.free_flow_time):
        g.add_edge(u, v, weight=min(w, g[u][v]['weight']) if g.has_edge(u, v) else w)
    e114 = json.loads((ROOT / 'experiments/results/e114_v103/results.json').read_text())
    e116 = json.loads((ROOT / 'experiments/results/e116_v105_loadlevels/results.json').read_text())
    protocol = json.loads((ROOT / 'experiments/results/e114_v103/protocol.json').read_text())
    regions = [tuple(U) for U in protocol['all_region_masks']]
    order_cache = {}
    sioux = []
    sources = [({'level': 1.0}, row) for row in e114['rows'] if row['halfwidth'] == .2] + \
              [({'level': row['level']}, row) for row in e116['rows']]
    for meta, row in sources:
        s, t = row['od'][0] - 1, row['od'][1] - 1
        key = (s, t)
        if key not in order_cache:
            ds = nx.single_source_dijkstra_path_length(g, s, weight='weight')
            dt = nx.single_source_dijkstra_path_length(g, t, weight='weight')
            p = {v: (ds[v] - dt[v] + ds[t]) / (2 * ds[t]) for v in range(net.n)}
            masks = [U for U in regions if s in U and t not in U]
            pos = np.array([max(p[v] for v in U) for U in masks])
            order_cache[key] = np.argsort(pos, kind='stable')
        a = np.array(row['geometric_accounts'])[order_cache[key]]
        rel, per = sweep(a)
        sioux.append({'od': row['od'], 'level': meta['level'], 'regions': len(a),
                      'relative_error': float(rel), 'tolerance_sweep': per})
    levels = {}
    for lev in (0.7, 1.0, 1.3):
        rows = [r for r in sioux if r['level'] == lev]
        levels[str(lev)] = {
            'queries': len(rows),
            **{f'tau{int(tau * 1000)}': {
                'single_value_queries': int(sum(r['tolerance_sweep'][ti]['single_value_ok'] for r in rows)),
                'median_segments': float(np.median([r['tolerance_sweep'][ti]['segments'] for r in rows])),
                'median_compression': float(np.median([r['tolerance_sweep'][ti]['compression'] for r in rows])),
                'median_equal_over_greedy': float(np.median([r['tolerance_sweep'][ti]['equal_over_greedy']
                                                             for r in rows])),
                'median_equal_over_tau': float(np.median([r['tolerance_sweep'][ti]['equal_over_tau']
                                                         for r in rows])),
            } for ti, tau in enumerate(TOLS)},
        }
    prof = json.loads((ROOT / 'experiments/results/v105_anaheim_profile/profile.json').read_text())
    ana = []
    for row in prof['rows']:
        rel, per = sweep(np.array(row['accounts']))
        ana.append({'query': row['query'], 'regions': len(row['accounts']),
                    'relative_error': float(rel), 'tolerance_sweep': per})
    ana_sum = {f'tau{int(tau * 1000)}': {
        'single_value_queries': int(sum(r['tolerance_sweep'][ti]['single_value_ok'] for r in ana)),
        'segments_per_query': [r['tolerance_sweep'][ti]['segments'] for r in ana],
        'median_compression': float(np.median([r['tolerance_sweep'][ti]['compression'] for r in ana])),
        'median_equal_over_greedy': float(np.median([r['tolerance_sweep'][ti]['equal_over_greedy'] for r in ana])),
        'median_equal_over_tau': float(np.median([r['tolerance_sweep'][ti]['equal_over_tau'] for r in ana])),
    } for ti, tau in enumerate(TOLS)}
    out = {'tolerances': list(TOLS), 'sioux_levels': levels, 'sioux_rows': sioux,
           'anaheim': ana_sum, 'anaheim_rows': ana,
           'data_sources': 'e114_v103 (level 1.0, halfwidth 0.2), e116_v105_loadlevels (0.7, 1.3), '
                           'v105_anaheim_profile; ordering identical to e118; no new solves'}
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'sioux_levels': levels, 'anaheim': {k: {kk: vv for kk, vv in v.items()
                                                              if kk != 'segments_per_query'}
                                                          for k, v in ana_sum.items()}}, indent=1))


if __name__ == '__main__':
    main()
