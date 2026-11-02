"""E141 (v110): the completed Anaheim (25,2) explanation case (external
review of v109, items 3.3c and 3.5).

The v109 Anaheim case reports the four-segment summary only as counts
(Table 4 style).  This experiment assembles the full delivered object and
the missing "where" layer, all from existing frozen data (no equilibrium
is re-solved):

  1. Complete four-segment output table for query (25,2): per segment the
     progress range, the number of summarized boundary values, the
     representative value, the actual worst within-segment error (<= eps),
     from the e126 row.
  2. Crossing-link table at the three prespecified positions and the
     plateau boundary: for each position, the region size, the crossing
     links with their signed attribution contributions (from the frozen
     41-state column), the top contributors listed by node pair, and the
     free-flow/congestion split of the account from the existing
     decomposition curves.  Anaheim TNTP ships no node coordinates, so the
     location layer is reported as crossing links between node IDs rather
     than as a geographic map.
  3. Boundary-count bookkeeping N_unique / N_chain / L_total: for Anaheim
     the single ordered family per query (regions -> segments), for Sioux
     Falls the per-query direction counts, distinct-chain counts and
     per-chain region counts from the fam_v108 families (the same
     deduplication e127 aggregates over).

Outputs results/e141_v110_anaheim_case/results.json consumed by the paper
tables; every number cites its frozen source file in 'sources'.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'results/e141_v110_anaheim_case'
OUT.mkdir(exist_ok=True)
E112 = HERE / 'results/e112_v102'
E113 = HERE / 'results/e113_v103'
E126 = HERE / 'results/e126_v108_persdirection'
QUERY = (25, 2)  # 1-based, as in the paper


def anaheim_family():
    """Rebuild the (25,2) distance-based family exactly as v105 (same code
    path, deterministic): progress order, admissible boundary indices,
    midgap progress levels, prespecified positions."""
    from run_e115_v103_scan import connectivity
    a = np.load(E112 / 'arrays.npz')
    edges = a['edges']
    old = json.loads((E112 / 'results.json').read_text())
    ods = [tuple(np.array(x['query']) - 1) for x in old['rows']]
    n = a['B'].shape[0]
    g = nx.Graph()
    for line in (ROOT / 'experiments/data/Anaheim/Anaheim_net.tntp').read_text().splitlines():
        p = line.split()
        if p and p[0].isdigit():
            u, v, w = int(p[0]) - 1, int(p[1]) - 1, float(p[4])
            g.add_edge(u, v, weight=min(w, g[u][v]['weight']) if g.has_edge(u, v) else w)
    adj = [set(g.neighbors(v)) for v in range(n)]
    s, t = ods[5]
    ds = nx.single_source_dijkstra_path_length(g, s, weight='weight')
    dt = nx.single_source_dijkstra_path_length(g, t, weight='weight')
    progress = np.array([(ds[v] - dt[v] + ds[t]) / (2 * ds[t]) for v in range(n)])
    order = np.argsort(progress, kind='stable')
    p = progress[order]
    ends = np.r_[np.flatnonzero(np.diff(p) > 1e-12) + 1, n]
    starts = np.r_[0, ends[:-1]]
    good = connectivity(order, ends, adj)[:-1] & \
        connectivity(order[::-1], n - starts[::-1], adj)[::-1][1:]
    rank = np.argsort(order)
    good &= (rank[s] < ends[:-1]) & (rank[t] >= ends[:-1])
    e = ends[:-1][good]
    levels = (p[e - 1] + p[e]) / 2
    return s, t, edges, order, e, levels, progress


def main():
    # ---------- segments table ----------
    e126 = json.loads((E126 / 'results.json').read_text())
    row = next(x for x in e126['anaheim']['rows'] if tuple(x['query']) == QUERY)
    a = np.array(row['accounts'])
    lv = np.array(row['progress_levels'])
    seg_rows = []
    for det, (lo, hi) in zip(row['segments_detail'], row['segment_progress_ranges']):
        i, j = det['start'], det['end']
        seg = a[i:j + 1]
        seg_rows.append({
            'progress_range': [float(lo), float(hi)],
            'boundaries': int(j - i + 1),
            'beta': float(det['beta']),
            'worst_error': float(np.max(np.abs(seg - det['beta']))),
            'account_range_in_segment': [float(seg.min()), float(seg.max())],
        })
        assert np.max(np.abs(seg - det['beta'])) <= row['eps_absolute'] + 1e-9

    # ---------- crossing links at the key positions ----------
    s, t, edges, order, e, levels, progress = anaheim_family()
    c = np.load(E113 / 'columns_41.npz')
    C = c['C'][:, 5]
    B = csr_matrix(np.load(E112 / 'arrays.npz')['B'])
    account = -np.cumsum((B @ C)[order])[e - 1]
    assert np.max(np.abs(account - a)) <= 1e-6 * max(1, float(np.max(np.abs(a))))

    prof = json.loads((HERE / 'results/v105_anaheim_profile/profile.json').read_text())
    prow = next(r for r in prof['rows'] if tuple(r['query']) == QUERY)
    curves = prow['curves']
    pos_idx = [int(np.argmin(np.abs(levels - x))) for x in
               (0.25, 0.5, 0.7238444018556495, 0.8176371175434433,
                0.9114298332312369)]
    link_ids = {}
    for k, (u, v) in enumerate(edges):
        link_ids.setdefault((u, v), []).append(k)

    def crossing(k):
        """Region U = order[:e[k]] (origin side); crossing links with signed
        attribution contributions (+1 outgoing from U, -1 entering U)."""
        U = set(order[:e[k]].tolist())
        contrib = []
        for ei, (u, v) in enumerate(edges):
            if (u in U) != (v in U):
                sign = 1.0 if u in U else -1.0
                contrib.append((ei, u, v, sign * float(C[ei])))
        contrib.sort(key=lambda x: -abs(x[3]))
        return U, contrib

    positions = []
    names = ['early', 'middle', 'plateau_start', 'plateau_end', 'destination']
    for name, k in zip(names, pos_idx):
        U, contrib = crossing(k)
        top = [{'from': int(u) + 1, 'to': int(v) + 1, 'edge': int(ei),
                'contribution': float(cv)}
               for ei, u, v, cv in contrib[:5]]
        positions.append({
            'name': name, 'progress': float(levels[k]),
            'region_nodes': int(e[k]),
            'account': float(account[k]),
            'crossing_links': len(contrib),
            'top_links': top,
            'crossing_total': float(sum(cv for *_, cv in contrib)),
            'components': {key: float(curves[key][k]) for key in curves},
        })
        # closure: the crossing contributions sum to the account
        assert abs(positions[-1]['crossing_total'] - account[k]) <= \
            1e-6 * max(1., abs(account[k]))

    # ---------- N_unique / N_chain / L_total bookkeeping ----------
    fam_stats = {'anaheim': []}
    for r in e126['anaheim']['rows']:
        fam_stats['anaheim'].append({
            'query': r['query'], 'N_unique': r['regions'], 'N_chain': 1,
            'L': r['segments']})
    from fam_v108 import direction_levels, query_families
    import run_e111_v102_pairs as p
    n, ods, *_ = p.base.setup()
    chains = direction_levels(net=n)
    sf = []
    for j, (s_, t_) in enumerate(ods):
        fams = query_families(s_, t_, chains=chains, net=n)
        seen, keep = set(), []
        sizes = []
        for i, f in enumerate(fams):
            key = tuple(tuple(U) for U in f['masks'])
            if key not in seen:
                seen.add(key)
                keep.append(i)
                sizes.append(len(f['masks']))
        sf.append({'query': [int(s_) + 1, int(t_) + 1],
                   'directions': len(fams), 'N_chain': len(keep),
                   'per_chain_regions_min': int(min(sizes)),
                   'per_chain_regions_max': int(max(sizes)),
                   'N_chain_positions_total': int(sum(sizes))})
    fam_stats['sioux_falls'] = sf

    out = {
        'sources': {
            'segments_and_family': 'e126_v108_persdirection/results.json (row (25,2))',
            'attribution_column': 'e113_v103/columns_41.npz (41-state column, query index 5)',
            'decomposition_curves': 'v105_anaheim_profile/profile.json',
            'network': 'e112_v102/arrays.npz + data/Anaheim/Anaheim_net.tntp',
            'sf_chain_stats': 'fam_v108 direction families (as aggregated by e127)'},
        'query': list(QUERY), 'eps_absolute': row['eps_absolute'],
        'segments_table': seg_rows, 'positions': positions,
        'family_bookkeeping': fam_stats,
        'protocol': ('no equilibrium re-solved; crossing contributions are '
                     'signed C_e entries of the frozen 41-state column '
                     '(+1 link leaving the region, -1 entering); closure to '
                     'the reported account asserted at each position'),
    }
    (OUT / 'results.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'segments': len(seg_rows), 'positions': len(positions),
                      'sf_chains': [(x['query'], x['N_chain']) for x in sf]},
                     indent=1))


if __name__ == '__main__':
    main()
