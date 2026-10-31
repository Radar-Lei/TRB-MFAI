"""E128 (v108): unified four-objective comparison on the 96 test demand paths.

External review of v107 item 3.4: the main results use 96 paths but the
four-objective comparison (flow / edge / cut / response) was still
reported on the original 24 paths.  This experiment evaluates the frozen
v104 selection of all four objectives on the same 96 paths and the same
metrics as e119; no model is retrained and no selection is revisited.
It also reports the five-seed paired improvement of response supervision
over flow supervision at path level.

Gate: the recomputed flow/response medians reproduce the e119 stored
96-path overall medians to <= 0.02 percentage points.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_e111_v102_pairs as p
from run_e83_v82_validation import attribution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e128_v108_fourmethods'
OUT.mkdir(exist_ok=True)
OLD = ROOT / 'experiments/results/v104_learning'
E111 = ROOT / 'experiments/results/e111_v102'
E119 = ROOT / 'experiments/results/e119_v106_expanded_paths'
METHODS = ('flow', 'edge', 'cut', 'response')
METRICS = ('flow_pct', 'column_pct', 'system_pct', 'geometry_pct')


def learned_columns(m, Q, ts, T):
    q = torch.tensor(Q)
    f = m(q).detach().numpy()
    js = []
    for j in range(6):
        v = torch.zeros_like(q)
        v[:, j] = 1
        _, jv = torch.func.jvp(m, (q,), (v,))
        js.append(jv.detach().numpy())
    jac = np.stack(js, axis=2)
    C = attribution(np.array([T.grad(f0) for f0 in f]), jac, ts, Q[-1] - Q[0])
    return f, C


def main():
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    masks = p.screenline_masks(ods)
    geo = [masks[j] @ n.B for j in range(6)]

    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Cnew = np.load(E119 / 'reference_columns.npz')['C']
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jref = np.concatenate([d_old['J'], d_new['J']])
    groups = np.concatenate([d_old['group'], d_new['group']])
    ts = d_old['ts']
    Cold = np.array([attribution(np.array([T.grad(f) for f in Fref[pi]]), Jref[pi], ts,
                                 Q[pi, -1] - Q[pi, 0]) for pi in range(24)])
    Cref = np.concatenate([Cold, Cnew])
    N = len(Q)

    selection = {(s['method'], s['seed']): s['step']
                 for s in json.loads((OLD / 'results.json').read_text())['selection']}
    rows = []
    started = time.perf_counter()
    for method in METHODS:
        for seed in range(5):
            m = p.base.old.build_fullnet_model(ps, qbg, seed)
            step = selection[(method, seed)]
            ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else \
                E111 / f'{method}_{seed}.pt'
            m.load_state_dict(torch.load(ck, weights_only=False)['best'])
            for pi in range(N):
                f, C = learned_columns(m, Q[pi], ts, T)
                flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                                     / np.linalg.norm(Fref[pi], axis=1))
                for j, (s, t) in enumerate(ods):
                    if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                        continue
                    e = C[:, j] - Cref[pi][:, j]
                    r = Cref[pi][:, j]
                    rows.append({
                        'method': method, 'seed': seed, 'path': pi, 'query': j,
                        'group': int(groups[pi]),
                        'flow_pct': float(flow),
                        'column_pct': float(100 * np.linalg.norm(e)
                                            / max(np.linalg.norm(r), 1e-6)),
                        'system_pct': float(100 * abs(e.sum())
                                            / max(abs(r.sum()), 1e-6)),
                        'geometry_pct': float(100 * np.max(np.abs(geo[j] @ e))
                                              / max(np.max(np.abs(geo[j] @ r)), 1e-6)),
                    })
            print('evaluated', method, seed, f'{time.perf_counter() - started:.0f}s',
                  flush=True)
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))

    # ---- path-level (seeds averaged within path), medians over paths ----
    vals = {}
    for met_m in METHODS:
        acc = {}
        for r in rows:
            if r['method'] == met_m:
                acc.setdefault(r['path'], []).append(r)
        vals[met_m] = {pi: {k2: float(np.mean([x[k2] for x in v])) for k2 in METRICS}
                       for pi, v in acc.items()}
    overall = {met_m: {k2: float(np.median([v[k2] for v in vals[met_m].values()]))
                       for k2 in METRICS} for met_m in METHODS}
    by_group = {}
    for met_m in METHODS:
        for g in (1, 2, 3, 6):
            idx = [pi for pi in vals[met_m] if groups[pi] == g]
            by_group[f'{met_m}_g{g}'] = {
                k2: float(np.median([vals[met_m][pi][k2] for pi in idx]))
                for k2 in METRICS}
    # replication gate vs e119 overall (flow, response)
    e119 = json.loads((E119 / 'results.json').read_text())['overall']
    gate = {}
    for met_m in ('flow', 'response'):
        for k2 in ('column_pct', 'geometry_pct', 'system_pct'):
            gate[f'{met_m}_{k2}'] = {
                'e119': e119[met_m][k2], 'e128': round(overall[met_m][k2], 4),
                'abs_dev': round(abs(overall[met_m][k2] - e119[met_m][k2]), 6)}
            assert abs(overall[met_m][k2] - e119[met_m][k2]) <= .02 + 1e-9, gate
    # five-seed paired improvement (response vs flow), per seed, path level:
    # active queries averaged within a path first, matching the protocol
    seed_table = []
    for seed in range(5):
        for k2 in ('column_pct', 'geometry_pct'):
            acc = {}
            for r in rows:
                if r['seed'] == seed and r['method'] in ('flow', 'response'):
                    acc.setdefault((r['method'], r['path']), []).append(r[k2])
            pathval = {k: np.mean(v) for k, v in acc.items()}
            dif = [pathval[('response', pi)] - pathval[('flow', pi)]
                   for pi in range(N)
                   if ('flow', pi) in pathval and ('response', pi) in pathval]
            seed_table.append({'seed': seed, 'metric': k2,
                               'median_flow': float(np.median(
                                   [v for (mm, pi), v in pathval.items() if mm == 'flow'])),
                               'median_response': float(np.median(
                                   [v for (mm, pi), v in pathval.items() if mm == 'response'])),
                               'improved_paths': int(np.sum(np.array(dif) < 0)),
                               'paths': len(dif),
                               'improved_share': float(np.mean(np.array(dif) < 0))})
    out = {'overall': overall, 'by_group': by_group, 'replication_gate': gate,
           'five_seed_paired': seed_table,
           'protocol': 'v104 frozen selection; 96 paths of e119; identical metrics; no retraining'}
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(overall, indent=2))
    print('gate ok; five-seed sample:', seed_table[:2])


if __name__ == '__main__':
    main()
