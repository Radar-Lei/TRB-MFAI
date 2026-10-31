"""E99 (v91, external review round 63, item III-2 optional enhancement):
pre-specified simple boundary families for the three sign-crossing
queries, defined WITHOUT reference to any attribution value.

Motivation.  The published negative accounts are attained by connected
minimizing regions found by exhaustive enumeration (13-15 nodes, 26+
crossing links).  Finding a negative value over ALL connected regions
establishes EXISTENCE, not that commonly drawn boundaries typically
produce negative accounts.  The review asks for a family of boundaries
an analyst might draw for other reasons -- here, origin-centred network
distance neighbourhoods -- evaluated on the frozen published attribution
columns, with both signs reported honestly whatever the outcome.

Boundary families (both pre-specified by network structure alone):
  * hop balls: U_d = {v : undirected hop distance from s_j <= d},
    d = 1, 2, ...; each d gives one candidate;
  * free-flow-time balls: U_r = {v : shortest free-flow travel time
    from s_j <= r}, r ranging over the distinct attainable distance
    values; each threshold gives one candidate.

Admissibility filter (same as the paper's connected family): s_j in U,
t_j not in U, U's induced subgraph connected, complement connected.
Every candidate is reported if admissible; inadmissible candidates are
counted, not silently dropped.

Gates:
  (a) the frozen e87 41-point columns reproduce the published origin-cut
      accounts and C_j of Table 4 (0.05 absolute, as in e96);
  (b) each reported phi_j(U) is computed twice -- as the node-sum
      -sum_{v in U} (B C~_{:,j})_v and by direct summation of signed
      crossing-link attributions (-C~_e for entering, +C~_e for leaving)
      -- agreeing to 1e-6;
  (c) hop balls and free-flow balls are connected by construction;
      verified with the e80 bit-mask connected() on both sides;
  (d) Dijkstra distances use the published free-flow times, and the
      ball at the largest radius equals the full node set (excluded by
      the t not in U filter) -- i.e. the families terminate.

Reported: for each crossing query, every admissible boundary with its
family, radius, node count, phi_j(U) (TSTT units), and sign; whether
any pre-specified boundary attains a negative account; and the extremes
over each family.  Whatever the signs, the finding is reported as-is:
the purpose is to separate "the attainable extreme range" from "the
variation available to commonly drawn boundaries".

NOT claimed: representativeness of these families for any particular
planning practice; they are simple pre-specified structures, and a
negative result would bound what they can show, not the phenomenon.

Output: experiments/results/e99_v91_predefined_boundaries.json
        console log experiments/results/e99_console.log
"""
from __future__ import annotations

import heapq
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
from run_e80_connected_cuts import build_adj_bits, connected  # noqa: E402

RESULTS = HERE / 'results'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
OUT = RESULTS / 'e99_v91_predefined_boundaries.json'

CROSSING = ('1->19', '1->15', '7->12')


def bfs_distances(adj, n, src):
    dist = [None] * n
    dist[src] = 0
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in range(n):
                if (adj[u] >> v) & 1 and dist[v] is None:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def dijkstra_fft(net, adj, n, src):
    """Shortest free-flow travel time from src over the undirected
    network with edge weight = free-flow time of the directed arc."""
    edges = net.edges
    t0 = net.free_flow_time
    w = {}
    for e, (u, v) in enumerate(edges):
        wt = float(t0[e])
        w.setdefault((u, v), []).append(wt)
        w.setdefault((v, u), []).append(wt)
    dist = [float('inf')] * n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in range(n):
            if (adj[u] >> v) & 1:
                nd = d + min(w[(u, v)])
                if nd < dist[v] - 1e-15:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
    return dist


def phi_nodesum(BC, j, nodes0):
    return -float(BC[nodes0, j].sum())


def phi_linksum(BC, C, B, j, nodes0, edges):
    """Direct signed crossing-link summation: -C~_e entering, +C~_e
    leaving; entering means the arc's head is in U and tail is not."""
    U = set(nodes0)
    total = 0.0
    for e, (u, v) in enumerate(edges):
        if u in U and v not in U:
            total += float(C[e, j])          # leaving: +C~
        elif v in U and u not in U:
            total -= float(C[e, j])          # entering: -C~
    return total


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, _, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    z87 = np.load(E87_NPZ)
    C = z87['C_41']                                  # (76, 6)
    BC = B @ C

    adj, n = build_adj_bits(net)
    edges = list(net.edges)

    # ---- gate (a): published origin-cut accounts and C_j ----
    published = {'1->19': (1771.6, 50097.7), '1->15': (2948.1, 77403.2),
                 '7->12': (9473.8, 117482.3)}
    for key in CROSSING:
        j = keys.index(key)
        s, t = study_ods[j]
        assert abs(-float(BC[s, j])) - published[key][0] < 0.05, key
        assert abs(float(C[:, j].sum()) - published[key][1]) < 0.05, key

    per_od = {}
    for key in CROSSING:
        j = keys.index(key)
        s, t = study_ods[j]

        hop = bfs_distances(adj, n, s)
        fft = dijkstra_fft(net, adj, n, s)

        candidates = []
        # hop balls: one per distinct radius up to (but excluding) the
        # first radius whose ball contains t
        for d in sorted({x for x in hop if x is not None}):
            nodes0 = [v for v in range(n) if hop[v] is not None
                      and hop[v] <= d]
            if t in nodes0:
                break
            candidates.append(('hop', d, nodes0))
        # free-flow-time balls: one per distinct attainable threshold
        for r in sorted({x for x in fft if x < float('inf')}):
            nodes0 = [v for v in range(n) if fft[v] <= r + 1e-12]
            if t in nodes0:
                break
            candidates.append(('fft', r, nodes0))

        admissible = []
        n_rejected = 0
        for family, radius, nodes0 in candidates:
            mask = 0
            for v in nodes0:
                mask |= 1 << v
            full = (1 << n) - 1
            ok = (s in nodes0) and (t not in nodes0) \
                and connected(adj, mask) and connected(adj, full & ~mask)
            if not ok:
                n_rejected += 1
                continue
            phi_n = phi_nodesum(BC, j, nodes0)
            phi_l = phi_linksum(BC, C, B, j, nodes0, edges)
            assert abs(phi_n - phi_l) <= 1e-6, (key, family, radius)  # gate (b)
            admissible.append({
                'family': family,
                'radius': radius,
                'n_nodes': len(nodes0),
                'nodes_1idx': [v + 1 for v in nodes0],
                'phi': phi_n,
                'phi_negative': bool(phi_n < 0.0),
            })
        # gate (d): families terminate before swallowing the network
        assert all(a['n_nodes'] < n for a in admissible)

        phis = [a['phi'] for a in admissible]
        per_od[key] = {
            'origin_s_1idx': s + 1,
            'destination_t_1idx': t + 1,
            'phi_s_published': published[key][0],
            'n_candidates': len(candidates) + 1,   # + the t-swallowing one
            'n_admissible': len(admissible),
            'n_rejected_inadmissible': n_rejected,
            'boundaries': admissible,
            'any_negative': bool(any(p < 0 for p in phis)),
            'min_phi': float(min(phis)) if phis else None,
            'max_phi': float(max(phis)) if phis else None,
        }
        print(f"{key}: {len(admissible)} admissible boundaries "
              f"(hop+fft), any_negative="
              f"{per_od[key]['any_negative']}, "
              f"phi range [{min(phis):,.1f}, {max(phis):,.1f}]"
              if phis else f"{key}: none", flush=True)
        for a in admissible:
            print(f"    {a['family']} r={a['radius']}: |U|={a['n_nodes']:2d} "
                  f"phi={a['phi']:+,.1f}{'  <-- NEGATIVE' if a['phi_negative'] else ''}",
                  flush=True)

    out = {
        'protocol': {
            'purpose': 'pre-specified origin-centred boundary families '
                       '(hop balls and free-flow-time balls) for the '
                       'three sign-crossing queries, evaluated on the '
                       'frozen published 41-point columns; no boundary '
                       'was selected using attribution values',
            'families': ['undirected hop ball from s_j',
                         'free-flow-time ball from s_j'],
            'admissibility': 's in U, t not in U, U and complement both '
                             'connected (the paper\'s connected family)',
            'claims_supported': 'which signs the pre-specified '
                                'boundaries produce; distinguishes the '
                                'attainable extreme range from the '
                                'variation available to simple '
                                'boundaries; NOT representativeness of '
                                'any planning practice',
        },
        'per_od': per_od,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'written -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
