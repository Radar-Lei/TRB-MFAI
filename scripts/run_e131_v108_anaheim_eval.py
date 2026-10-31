"""E131 (v108): evaluate the Anaheim learned assignments (flow vs response)
on the 48 test demand paths.

Mirrors the Sioux Falls e119/e127 evaluation: frozen selection, five
seeds averaged within a path before any summary, metrics flow_pct /
column_pct / system_pct / geometry_pct against the e129 reference
columns, geometry over the ordered nested-prefix family of each study
query.  Additionally computes the total-error budget of the learned
segmented summaries (E_model, E_summary, E_total over the same ordered
family, greedy minimum-segment summary at the 5% family tolerance) so
the second network reports the same delivered explanation quantities as
Sioux Falls, and the five-seed paired improvement of response over flow
supervision.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e131_v108_anaheim_eval'
OUT.mkdir(exist_ok=True)
DATA = HERE / 'results/e129_v108_anaheim_data'
TRAIN = HERE / 'results/e130_v108_anaheim_train'
METHODS = ('flow', 'response')
METRICS = ('flow_pct', 'column_pct', 'system_pct', 'geometry_pct',
           'E_model', 'E_summary', 'E_total', 'segments')
TOL = 0.05
N_PATHS = 48


def greedy_segments(a, eps):
    segs = []
    i = 0
    n = len(a)
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


def summarize(a, segs):
    out = np.empty_like(a)
    for i, j in segs:
        s = a[i:j + 1]
        out[i:j + 1] = (s.max() + s.min()) / 2
    return out


def main():
    torch.set_num_threads(1)
    from run_e83_v82_validation import attribution, TSTTT
    import run_e130_v108_anaheim_train as tr

    d = np.load(DATA / 'demands.npz')
    ref = np.load(DATA / 'reference.npz')
    with (DATA / 'setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    fams = json.loads((DATA / 'families.json').read_text())
    Q, dQP, group, ts = d['QP'], d['dQP'], d['group'], d['ts']
    Cref = ref['Cref']
    # merge test-path reference flows from shards (for flow_pct)
    Fp = [None] * N_PATHS
    for f in sorted(DATA.glob('paths_*.npz')):
        z = np.load(f)
        for fr, i in zip(z['F'], z['pids']):
            Fp[int(i)] = fr
    assert all(x is not None for x in Fp)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    B = net.B
    geo = []
    for f in fams:
        R = np.array([[1.0 if v in U else 0.0 for v in range(net.n)] for U in f['masks']])
        geo.append(R @ B)

    rows = []
    started = time.perf_counter()
    for method in METHODS:
        for seed in range(5):
            ck = torch.load(TRAIN / f'{method}_{seed}.pt', weights_only=False)
            pre = torch.load(TRAIN / f'pre_seed{seed}.pt', weights_only=False)
            m = tr.build_model(ps, d['q'][6:], seed, pre['norm']['mu_q'], pre['norm']['sd_q'])
            m.load_state_dict(ck['best'])
            for pi in range(N_PATHS):
                qt = torch.tensor(Q[pi])
                f = m(qt).detach().numpy()
                js = []
                for j in range(6):
                    v = torch.zeros_like(qt)
                    v[:, j] = 1
                    _, jv = torch.func.jvp(m, (qt,), (v,))
                    js.append(jv.detach().numpy())
                jac = np.stack(js, axis=2)
                C = attribution(np.array([T.grad(f0) for f0 in f]), jac, ts, dQP[pi])
                flow = 100 * np.mean(np.linalg.norm(f - Fp[pi], axis=1)
                                     / np.linalg.norm(Fp[pi], axis=1))
                for j in range(6):
                    if abs(dQP[pi][j]) < 1e-10:
                        continue
                    e = C[:, j] - Cref[pi][:, j]
                    r = Cref[pi][:, j]
                    col = float(100 * np.linalg.norm(e) / max(np.linalg.norm(r), 1e-6))
                    sysm = float(100 * abs(e.sum()) / max(abs(r.sum()), 1e-6))
                    ge = float(100 * np.max(np.abs(geo[j] @ e))
                               / max(np.max(np.abs(geo[j] @ r)), 1e-6))
                    # total-error budget on the ordered family
                    aL = -(geo[j] @ C[:, j])
                    aS = -(geo[j] @ r)
                    E_model = float(np.max(np.abs(aL - aS)))
                    epsL = TOL * max(np.max(np.abs(aL)), 1e-12)
                    segs = greedy_segments(aL, epsL)
                    hat = summarize(aL, segs)
                    E_summary = float(np.max(np.abs(hat - aL)))
                    E_total = float(np.max(np.abs(hat - aS)))
                    assert E_summary <= epsL + 1e-6 * max(1., epsL)
                    assert E_total <= E_summary + E_model + 1e-6 * max(1., E_model)
                    rows.append({'method': method, 'seed': seed, 'path': pi, 'query': j,
                                 'group': int(group[pi]), 'flow_pct': float(flow),
                                 'column_pct': col, 'system_pct': sysm, 'geometry_pct': ge,
                                 'E_model': E_model, 'E_summary': E_summary,
                                 'E_total': E_total, 'segments': len(segs),
                                 'regions': len(fams[j]['masks'])})
            print('evaluated', method, seed, f'{time.perf_counter() - started:.0f}s',
                  flush=True)
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))

    vals = {}
    for met_m in METHODS:
        acc = {}
        for r in rows:
            if r['method'] == met_m:
                acc.setdefault(r['path'], []).append(r)
        vals[met_m] = {pi: {k2: float(np.mean([x[k2] for x in v]))
                            for k2 in METRICS if k2 != 'flow_pct'} |
                       {'flow_pct': float(np.mean([x['flow_pct'] for x in v]))}
                       for pi, v in acc.items()}
    overall = {met_m: {k2: float(np.median([v[k2] for v in vals[met_m].values()]))
                       for k2 in METRICS} for met_m in METHODS}
    by_group = {}
    for met_m in METHODS:
        for g in (1, 2, 3, 6):
            idx = [pi for pi in vals[met_m] if group[pi] == g]
            by_group[f'{met_m}_g{g}'] = {
                k2: float(np.median([vals[met_m][pi][k2] for pi in idx]))
                for k2 in ('column_pct', 'geometry_pct')}
    # five-seed paired improvement
    seed_table = []
    for seed in range(5):
        acc = {}
        for r in rows:
            if r['seed'] == seed:
                acc.setdefault((r['method'], r['path'], r['query']), r)
        for k2 in ('column_pct', 'geometry_pct'):
            dif = [acc[('response', pi, j)][k2] - acc[('flow', pi, j)][k2]
                   for pi in range(N_PATHS) for j in range(6)
                   if ('flow', pi, j) in acc and ('response', pi, j) in acc]
            seed_table.append({
                'seed': seed, 'metric': k2,
                'median_flow': float(np.median([v[k2] for (mm, _, _), v in acc.items() if mm == 'flow'])),
                'median_response': float(np.median([v[k2] for (mm, _, _), v in acc.items() if mm == 'response'])),
                'improved_share': float(np.mean(np.array(dif) < 0))})
    out = {'overall': overall, 'by_group': by_group, 'five_seed_paired': seed_table,
           'protocol': 'e130 frozen selection; 48 Anaheim test paths; '
                       'metrics identical to Sioux Falls e119/e127'}
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(overall, indent=2))
    print('five-seed:', [x for x in seed_table if x['metric'] == 'column_pct'])


if __name__ == '__main__':
    main()
