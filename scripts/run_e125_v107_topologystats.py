"""E125 (v107): topology payoff statistics on real multi-block networks.

External review of v106 (B2): the off-path heterogeneity freedom of the
connected-region classification is illustrated on a constructed network
(Figure 1); a real network with natural articulation/block structure
would quantify how much of the network the connected condition leaves
unconstrained.

For Sioux Falls and Anaheim this experiment reports, using only the
public topology (no assignment solve):
  - atom counts (circuit atoms, bridge atoms);
  - for the study queries of the published experiments: the number of
    atoms traversed by each OD block path, the union E_Q of their edges,
    and |E_Q| as a share of all edges;
  - the full distribution of |E_Q| over every positive-demand OD (the
    per-query edge share that the connected condition constrains, versus
    100% for the unrestricted condition), with median and quartiles;
  - the off-path release: 1 - |E_Q|/|E|.

Gates:
  (a) atoms partition the edge set exactly;
  (b) every OD block path is nonempty and its atom edges contain a
      valid s-t cut of the queried network.

Output: experiments/results/e125_v107_topologystats/results.json
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e125_v107_topologystats'
OUT.mkdir(exist_ok=True)


def network_stats(net, queries, query_labels):
    eq = net.equivalence_classes(list(queries))
    atoms = eq['atoms']
    all_edges = sorted(e for a in atoms for e in a['edges'])
    assert all_edges == sorted(range(net.m)), 'atoms do not partition the edge set'
    n_circuit = sum(1 for a in atoms if a['type'] == 'circuit')
    rows = []
    for (s, t), lab in zip(queries, query_labels):
        ks = net.od_path_atoms(s, t)
        assert ks, (s, t)
        eq_edges = sorted({e for k in ks for e in atoms[k]['edges']})
        rows.append({'query': lab, 'atoms_traversed': len(ks),
                     'atom_edge_count': len(eq_edges),
                     'edge_share_pct': 100.0 * len(eq_edges) / net.m})
    return eq, atoms, n_circuit, rows


def distribution(net, ods):
    shares = []
    for (s, t) in ods:
        ks = net.od_path_atoms(s, t)
        eq_edges = {e for k in ks for e in net.atoms()[k]['edges']}
        shares.append(100.0 * len(eq_edges) / net.m)
    a = np.array(shares)
    return {'queries': len(a),
            'median_pct': float(np.median(a)),
            'q1_pct': float(np.percentile(a, 25)),
            'q3_pct': float(np.percentile(a, 75)),
            'min_pct': float(a.min()), 'max_pct': float(a.max()),
            'release_median_pct': float(100.0 - np.median(a))}


def load_tntp(name, here):
    """Parse a TNTP net+trips pair into (Network, positive OD dict)."""
    import re
    from nets import Network
    rows = []
    for line in (here / f'{name}_net.tntp').read_text().splitlines():
        a = line.strip().split()
        if a and a[0].isdigit():
            rows.append(a)
    net = Network(name, [(int(a[0]) - 1, int(a[1]) - 1) for a in rows],
                  capacity=np.array([float(a[2]) for a in rows]),
                  free_flow_time=np.array([float(a[4]) for a in rows]))
    trips = {}
    origin = None
    for line in (here / f'{name}_trips.tntp').read_text().splitlines():
        if line.strip().startswith('Origin'):
            origin = int(line.split()[1]) - 1
        elif origin is not None:
            for d, q in re.findall(r'(\d+)\s*:\s*([\d.eE+-]+)\s*;', line):
                if float(q) > 0 and int(d) - 1 != origin:
                    trips[origin, int(d) - 1] = float(q)
    return net, trips


def main():
    from nets import load_sioux_falls_trips
    import run_e111_v102_pairs as p
    import run_e112_v102_anaheim as e112

    started = time.perf_counter()
    # ---- Sioux Falls (study queries of the setting-C protocol) ----
    n_sf, ods_sf, qb, qbg, ps, sv, T = p.base.setup()
    queries_sf = list(ods_sf)
    labels_sf = [f'{s + 1}->{t + 1}' for s, t in queries_sf]
    eq, atoms, n_circ, rows_sf = network_stats(n_sf, queries_sf, labels_sf)
    trips = load_sioux_falls_trips()
    all_sf = sorted(o for o, d in trips.items() if d > 0)
    dist_sf = distribution(n_sf, all_sf)
    sioux = {'network': 'Sioux Falls', 'nodes': n_sf.n, 'edges': n_sf.m,
             'atoms': len(atoms), 'circuit_atoms': n_circ,
             'bridge_atoms': len(atoms) - n_circ,
             'study_queries': rows_sf, 'all_od_distribution': dist_sf}

    # ---- Anaheim (study queries of the e112 protocol) ----
    n_an, ps_an, q_an = e112.setup()
    queries_an = list(ps_an.ods[:6])
    labels_an = [f'{s + 1}->{t + 1}' for s, t in queries_an]
    eq_an, atoms_an, n_circ_an, rows_an = network_stats(n_an, queries_an, labels_an)
    dist_an = distribution(n_an, list(ps_an.ods))
    ana = {'network': 'Anaheim', 'nodes': n_an.n, 'edges': n_an.m,
           'atoms': len(atoms_an), 'circuit_atoms': n_circ_an,
           'bridge_atoms': len(atoms_an) - n_circ_an,
           'study_queries': rows_an, 'all_od_distribution': dist_an}

    # ---- additional public TNTP networks (topology screening only) ----
    here = ROOT / 'experiments/data/candidates'
    extras = {}
    for name, folder in (('EMA', 'Eastern-Massachusetts'), ('Winnipeg', 'Winnipeg')):
        net, trips_x = load_tntp(name, here)
        eq_x, atoms_x, n_circ_x, _ = network_stats(net, sorted(trips_x),
                                                   [f'{s + 1}->{t + 1}' for s, t in sorted(trips_x)])
        dist_x = distribution(net, sorted(trips_x))
        extras[name] = {'network': name, 'source': folder, 'nodes': net.n, 'edges': net.m,
                        'ods': len(trips_x), 'atoms': len(atoms_x),
                        'circuit_atoms': n_circ_x, 'bridge_atoms': len(atoms_x) - n_circ_x,
                        'all_od_distribution': dist_x}

    result = {'protocol': {'edge_source': 'public TNTP topology, no assignment solve',
                           'elapsed_seconds': time.perf_counter() - started},
              'sioux_falls': sioux, 'anaheim': ana, 'extra_networks': extras}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'sioux': {'atoms': sioux['atoms'], 'dist': dist_sf},
                      'anaheim': {'atoms': ana['atoms'], 'dist': dist_an},
                      'extras': {k: {'atoms': v['atoms'], 'dist': v['all_od_distribution']}
                                 for k, v in extras.items()}}, indent=2))


if __name__ == '__main__':
    main()
