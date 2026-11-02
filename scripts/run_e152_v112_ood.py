"""E152 (v112): response supervision under out-of-domain base demands.

External review of v111: the derivative-label centers and the main test
paths draw base demands from [0.65, 1.3] x q_base, so the response-
supervision gains could be specific to that labelled band.  This experiment
evaluates the frozen E111 checkpoints (flow and response, five seeds) on
two OOD base-demand bands, generating 24 paths per band with the E111 test
generator's rules unchanged:

  low band   base ~ U[0.40, 0.60] x q_base   (seed 112120)
  high band  base ~ U[1.35, 1.55] x q_base  (seed 112121)

per band: active-query groups k in {1,2,3,6} x 6 replicates; deltas
~ U[0.15, 0.30] x q_base on the active queries; replicates with odd r flip
the sign of every even-indexed delta (mixed signs, as E111); 41 states per
path; reference SUE flows and sensitivities per state; reference attribution
columns by the shared trapezoid rule.

Metrics per (method, seed, path, active query): column error; family
boundary error over the query's geometric screenline family (E_model
normalized by the reference family maximum); unrestricted extrema error
(closed-form node-divergence extrema: max over the two endpoints of
|extremum(learned) - extremum(reference)| as a percentage of the reference
unrestricted span); plus flow error and the 5%-tolerance end-to-end summary
error for context.  Aggregation follows Table tab:pathtable: mean over
active queries and seeds within a path, median over the 24 paths of each
band.

Outputs: reference.npz, rows.json, results.json.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_e111_v102_pairs as p
import run_e127_v108_totalerror as e127
from fam_v108 import direction_levels, query_families
from run_e83_v82_validation import attribution

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e152_v112_ood'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'

BANDS = {'low': (0.40, 0.60, 112120), 'high': (1.35, 1.55, 112121)}
METHODS = ('flow', 'response')
TS = np.linspace(0, 1, 41)


def reference(band):
    """24 OOD test paths for one band: SUE flows, sensitivities, columns."""
    lo, hi, seed = BANDS[band]
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    rng = np.random.default_rng(seed)
    qs, ff, jj, groups = [], [], [], []
    for k in [1, 2, 3, 6]:
        for r in range(6):
            a = rng.uniform(lo, hi, 6) * qb
            b = a.copy()
            active = [(r + i) % 6 for i in range(k)]
            delta = rng.uniform(.15, .3, k) * qb[active]
            if r % 2:
                delta[::2] *= -1
            b[active] += delta
            Q = a + TS[:, None] * (b - a)
            F, J = [], []
            warm = None
            for q in Q:
                f, h, pr, _ = sv.solve(np.r_[q, qbg], warm)
                warm = f
                jac, *_ = sv.sensitivity(f, h, pr, np.r_[q, qbg])
                F.append(f)
                J.append(jac)
            qs.append(Q)
            ff.append(F)
            jj.append(J)
            groups.append(k)
            print(f'{band} path group {k} rep {r} done', flush=True)
    F = np.array(ff)
    J = np.array(jj)
    Q = np.array(qs)
    Cref = np.array([attribution(np.array([T.grad(f0) for f0 in F[pi]]), J[pi], TS,
                                 Q[pi, -1] - Q[pi, 0]) for pi in range(24)])
    np.savez_compressed(OUT / f'reference_{band}.npz', Q=Q, F=F, J=J, C=Cref,
                        group=np.array(groups), ts=TS)
    print(f'{band}: reference complete (solver residual {sv.max_residual:.1e})', flush=True)


def evaluate():
    """Frozen E111 checkpoints on both OOD bands."""
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    chains = direction_levels(net=n)
    fams = [query_families(s, t, chains=chains, net=n) for s, t in ods]
    geo = []
    for fam_j in fams:
        geo.append([np.array([[v in U for v in range(n.n)] for U in f['masks']]) @ B
                    for f in fam_j])

    rows = []
    started = time.perf_counter()
    for band in BANDS:
        d = np.load(OUT / f'reference_{band}.npz')
        Q, Fref, Jref, Cref, ts = d['Q'], d['F'], d['J'], d['C'], d['ts']
        for method in METHODS:
            for seed in range(5):
                m = p.base.old.build_fullnet_model(ps, qbg, seed)
                m.load_state_dict(torch.load(E111 / f'{method}_{seed}.pt',
                                             weights_only=False)['best'])
                for pi in range(len(Q)):
                    f, jac, C = e127.learned_columns(m, Q[pi], ts, T)
                    dQ = Q[pi, -1] - Q[pi, 0]
                    flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                                         / np.linalg.norm(Fref[pi], axis=1))
                    for j, (s, t) in enumerate(ods):
                        if abs(dQ[j]) < 1e-10:
                            continue
                        cL, cS = C[:, j], Cref[pi][:, j]
                        # family boundary error (E_model / reference family max)
                        aL = [-(g @ cL) for g in geo[j]]
                        aS = [-(g @ cS) for g in geo[j]]
                        E_model = max(np.max(np.abs(al - as_)) for al, as_ in zip(aL, aS))
                        famscale = float(max(np.max(np.abs(as_)) for as_ in aS))
                        # unrestricted extrema via node divergence (closed form):
                        # phi(U) = -sum_{v in U} (B c)_v, unrestricted family lets
                        # every interior vertex choose a side independently.
                        divL, divS = B @ cL, B @ cS
                        inter = np.ones(n.n, dtype=bool)
                        inter[[s, t]] = False
                        maxL = -divL[s] - np.minimum(divL[inter], 0).sum()
                        minL = -divL[s] - np.maximum(divL[inter], 0).sum()
                        maxS = -divS[s] - np.minimum(divS[inter], 0).sum()
                        minS = -divS[s] - np.maximum(divS[inter], 0).sum()
                        spanS = float(np.abs(divS[inter]).sum())
                        err_ext = max(abs(maxL - maxS), abs(minL - minS))
                        # 5%-tolerance end-to-end summary error (context)
                        epsL = 0.05 * max(np.max(np.abs(al)) for al in aL)
                        hatL = []
                        for al in aL:
                            segs = e127.greedy_segments(al, epsL)
                            hatL.append(e127.summarize(al, segs))
                        E_total = max(np.max(np.abs(h - as_)) for h, as_ in zip(hatL, aS))
                        rows.append({
                            'band': band, 'method': method, 'seed': seed,
                            'path': pi, 'query': j, 'group': int(d['group'][pi]),
                            'flow_pct': float(flow),
                            'column_pct': float(100 * np.linalg.norm(cL - cS)
                                                / max(np.linalg.norm(cS), 1e-6)),
                            'family_pct': float(100 * E_model / max(famscale, 1e-9)),
                            'unrestricted_pct': float(100 * err_ext / max(spanS, 1e-9)),
                            'E_total_pct': float(100 * E_total / max(famscale, 1e-9)),
                        })
                print('eval', band, method, seed,
                      f'{time.perf_counter()-started:.0f}s', flush=True)
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))

    # aggregation: Table tab:pathtable convention
    METRICS = ('flow_pct', 'column_pct', 'family_pct', 'unrestricted_pct', 'E_total_pct')
    summary = {}
    for band in BANDS:
        for method in METHODS:
            rr = [r for r in rows if r['band'] == band and r['method'] == method]
            per_path = {}
            for r in rr:
                per_path.setdefault(r['path'], []).append(r)
            summary[f'{band}:{method}'] = {
                k: float(np.median([np.mean([x[k] for x in v]) for v in per_path.values()]))
                for k in METRICS}
    # paired per-path differences (response minus flow) for CI reporting
    paired = {}
    rng = np.random.default_rng(0)
    for band in BANDS:
        for k in METRICS:
            fv, rv = [], []
            rr = [r for r in rows if r['band'] == band]
            for pi in sorted({r['path'] for r in rr}):
                f_ = np.mean([x[k] for x in rr if x['path'] == pi and x['method'] == 'flow'])
                r_ = np.mean([x[k] for x in rr if x['path'] == pi and x['method'] == 'response'])
                fv.append(f_)
                rv.append(r_)
            fv, rv = np.array(fv), np.array(rv)
            diff = rv - fv
            boots = []
            for _ in range(2000):
                ix = rng.integers(0, len(diff), len(diff))
                boots.append(np.median(diff[ix]))
            paired[f'{band}:{k}'] = {
                'median_diff': float(np.median(diff)),
                'bootstrap_ci95': [float(np.percentile(boots, 2.5)),
                                   float(np.percentile(boots, 97.5))]}
    (OUT / 'results.json').write_text(json.dumps(
        {'protocol': {'bands': {k: list(v) for k, v in BANDS.items()},
                      'paths_per_band': 24,
                      'generator': 'e111 test rules (groups 1/2/3/6 x 6, deltas U[.15,.30]q, odd reps flip even deltas)',
                      'models': 'frozen e111 checkpoints (flow, response; seeds 0-4)'},
         'summary': summary, 'paired': paired}, indent=2))
    print(json.dumps({'summary': summary, 'paired': paired}, indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('phase', choices=['reference', 'eval'])
    a.add_argument('--band', choices=list(BANDS), default=None)
    v = a.parse_args()
    if v.phase == 'reference':
        bands = [v.band] if v.band else list(BANDS)
        for band in bands:
            reference(band)
    else:
        evaluate()
