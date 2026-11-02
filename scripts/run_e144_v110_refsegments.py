"""E144: reference-profile segment counts for the E140 tolerance sweep.

Codex round-1 P1: fig_resolution plotted the reference compression error at
the *learned model's* segment count.  The reference profile compressed at the
same tolerance has its own segment count, which E140 computed but discarded.
This script recomputes the reference segmentation from the frozen reference
columns (no model, no solver), gates every recomputed E_summary_ref against
the stored E140 shard values, and writes L_ref per (network, tau) with the
same aggregation as E140's collect (per-path mean over queries, median over
paths; SF sums over deduplicated direction chains).
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import pickle
from pathlib import Path

import numpy as np

from run_e140_v110_resolution import TAUS, greedy_segments, summarize

R = Path(__file__).resolve().parent / 'results'
E140 = R / 'e140_v110_resolution'
E111 = R / 'e111_v102'
E119 = R / 'e119_v106_expanded_paths'
DATA = R / 'e129_v108_anaheim_data'
OUT = R / 'e144_v110_refsegments'
OUT.mkdir(exist_ok=True)


def ref_pass(aS_list, tau):
    """Compress the reference accounts at eps = tau * family reference scale,
    with E140's own greedy rule (segments never straddle zero)."""
    fam_ref_scale = max(float(np.max(np.abs(a))) for a in aS_list)
    eps = tau * fam_ref_scale
    cnt, err = [], 0.0
    for as_ in aS_list:
        segs = greedy_segments(as_, eps)
        cnt.append(len(segs))
        h = summarize(as_, segs)
        err = max(err, float(np.max(np.abs(h - as_))))
    return cnt, err


def sioux_falls():
    import run_e111_v102_pairs as p
    from run_e83_v82_validation import attribution
    from fam_v108 import direction_levels, query_families

    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    chains = direction_levels(net=n)
    geo, dedup = [], {}
    for j, (s, t) in enumerate(ods):
        fams = query_families(s, t, chains=chains, net=n)
        seen, keep = set(), []
        for i, f in enumerate(fams):
            key = tuple(tuple(U) for U in f['masks'])
            if key not in seen:
                seen.add(key)
                keep.append(i)
        dedup[j] = keep
        geo.append([np.array([[v in U for v in range(n.n)] for U in f['masks']])
                    @ B for f in fams])

    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Cnew = np.load(E119 / 'reference_columns.npz')['C']
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jold, ts = d_old['J'], d_old['ts']
    Cref = np.concatenate([
        np.array([attribution(np.array([T.grad(f) for f in Fref[pi]]),
                              Jold[pi], ts, Q[pi, -1] - Q[pi, 0])
                  for pi in range(24)]), Cnew])
    N = len(Q)

    # gate source: stored shard rows (E_summary_ref is method/seed-independent)
    gate = {(r['path'], r['query']): r['tau_grid']
            for r in json.loads((E140 / 'sf_flow_0.json').read_text())}

    rows = []  # (path, query, {tau: (L_row, err)})
    for pi in range(N):
        for j, (s, t) in enumerate(ods):
            if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                continue
            aS = [-(g @ Cref[pi][:, j]) for g in geo[j]]
            sw = {}
            for tau in TAUS:
                cnt, err = ref_pass(aS, tau)
                L = sum(cnt[i] for i in dedup[j])
                stored = gate[(pi, j)][f'{tau:g}']['E_summary_ref']
                assert abs(err - stored) <= 1e-9 * max(1., abs(stored)), \
                    ('sf gate', pi, j, tau, err, stored)
                sw[tau] = L
            rows.append((pi, j, sw))
        print(f'sf ref path{pi}', flush=True)

    out = {}
    for tau in TAUS:
        per_path = {}
        for pi, j, sw in rows:
            per_path.setdefault(pi, []).append(sw[tau])
        out[f'{tau:g}'] = {'L_ref': float(np.median(
            [np.mean(v) for v in per_path.values()]))}
    return out


def anaheim():
    import run_e130_v108_anaheim_train as tr
    from run_e83_v82_validation import TSTTT

    d = np.load(DATA / 'demands.npz')
    ref = np.load(DATA / 'reference.npz')
    with (DATA / 'setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    fams = json.loads((DATA / 'families.json').read_text())
    Q, dQP, Cref = d['QP'], d['dQP'], ref['Cref']
    B = net.B
    geo = [np.array([[1.0 if v in U else 0.0 for v in range(net.n)]
                     for U in f['masks']]) @ B for f in fams]

    gate = {(r['path'], r['query']): r['tau_grid']
            for r in json.loads((E140 / 'an_flow_0.json').read_text())}

    rows = []
    for pi in range(48):
        for j in range(6):
            if abs(dQP[pi][j]) < 1e-10:
                continue
            aS = [-(geo[j] @ Cref[pi][:, j])]
            sw = {}
            for tau in TAUS:
                cnt, err = ref_pass(aS, tau)
                stored = gate[(pi, j)][f'{tau:g}']['E_summary_ref']
                assert abs(err - stored) <= 1e-9 * max(1., abs(stored)), \
                    ('an gate', pi, j, tau, err, stored)
                sw[tau] = cnt[0]
            rows.append((pi, j, sw))
        print(f'an ref path{pi}', flush=True)

    out = {}
    for tau in TAUS:
        per_path = {}
        for pi, j, sw in rows:
            per_path.setdefault(pi, []).append(sw[tau])
        out[f'{tau:g}'] = {'L_ref': float(np.median(
            [np.mean(v) for v in per_path.values()]))}
    return out


if __name__ == '__main__':
    result = {'sioux_falls': sioux_falls(), 'anaheim': anaheim(),
              'protocol': ('reference-profile segment counts at the E140 '
                           'tolerances; greedy per-direction segmentation of '
                           'aS at eps=tau*max|aS|; every recomputed '
                           'E_summary_ref gated against the stored E140 '
                           'shard values to 1e-9 relative; aggregation '
                           'identical to E140 collect (path mean over '
                           'queries, median over paths; SF sums over '
                           'deduplicated direction chains)')}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
