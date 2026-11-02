"""E138 (v109): normalized Anaheim total-error budget for the merged
cross-network summary table.

External review of v108 (P1-5): Tables 13+14 merge into one cross-network
summary, which requires Anaheim E-terms on the same per-query
normalization as tab:totalerror (percent of the reference family
maximum) plus the two compression baselines (single common value;
equal-count segmentation with the same count).

Mirrors run_e131_v108_anaheim_eval.py exactly (same checkpoints, states,
families, tolerance and per-row metrics) and adds per row:

    fam_scale_reference = max_U |a^S_j(U)|  (reference family maximum)
    E_single  = max_U |beta(a^L) - a^S_j(U)|,  beta = midrange of a^L
    E_equal   = max_U |hat a^L_eq - a^S_j(U)|, equal-count blocks with
                the greedy segment count, midrange representatives
                (identical definitions to e127 on Sioux Falls)

Unified aggregation identical to e127's table_unified_normalized:
per-row E / fam_scale_reference, mean within path, median over the 48
test paths, times 100.

Gates: every recomputed E_model/E_summary/E_total/segments value must
match the stored e131 row to 1e-9 relative (same inputs, same code
path); E_summary <= eps^L and E_total <= E_summary + E_model asserted.
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
DATA = HERE / 'results/e129_v108_anaheim_data'
E131 = HERE / 'results/e131_v108_anaheim_eval'
TRAIN = HERE / 'results/e130_v108_anaheim_train'
OUT = HERE / 'results/e138_v109_crossnetwork'
OUT.mkdir(exist_ok=True)
METHODS = ('flow', 'response')
N_PATHS = 48
TOL = 0.05


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


def equal_blocks(nn, k):
    return [(int(round(x * nn / k)), int(round((x + 1) * nn / k)) - 1)
            for x in range(k)]


def main():
    torch.set_num_threads(1)
    from run_e83_v82_validation import attribution, TSTTT
    import run_e130_v108_anaheim_train as tr

    d = np.load(DATA / 'demands.npz')
    ref = np.load(DATA / 'reference.npz')
    with (DATA / 'setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    fams = json.loads((DATA / 'families.json').read_text())
    Q, dQP = d['QP'], d['dQP']
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    Cref = ref['Cref']
    B = net.B
    geo = []
    for f in fams:
        R = np.array([[1.0 if v in U else 0.0 for v in range(net.n)] for U in f['masks']])
        geo.append(R @ B)
    old_rows = json.loads((E131 / 'rows.json').read_text())
    old_map = {(r['method'], r['seed'], r['path'], r['query']): r for r in old_rows}

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
                    v[:, j] = 1.0
                    _, jv = torch.func.jvp(m, (qt,), (v,))
                    js.append(jv.detach().numpy())
                jac = np.stack(js, axis=2)
                C = attribution(np.array([T.grad(f0) for f0 in f]), jac, d['ts'], dQP[pi])
                for j in range(6):
                    if abs(dQP[pi][j]) < 1e-10:
                        continue
                    r = Cref[pi][:, j]
                    aL = -(geo[j] @ C[:, j])
                    aS = -(geo[j] @ r)
                    E_model = float(np.max(np.abs(aL - aS)))
                    epsL = TOL * max(np.max(np.abs(aL)), 1e-12)
                    segs = greedy_segments(aL, epsL)
                    k = len(segs)
                    hat = summarize(aL, segs)
                    E_summary = float(np.max(np.abs(hat - aL)))
                    E_total = float(np.max(np.abs(hat - aS)))
                    assert E_summary <= epsL + 1e-6 * max(1., epsL)
                    assert E_total <= E_summary + E_model + 1e-6 * max(1., E_model)
                    beta = (aL.max() + aL.min()) / 2
                    E_single = float(np.max(np.abs(beta - aS)))
                    hatE = summarize(aL, equal_blocks(len(aL), k))
                    E_equal = float(np.max(np.abs(hatE - aS)))
                    scale = float(np.max(np.abs(aS)))
                    # gate: identical to the stored e131 row
                    o = old_map[(method, seed, pi, j)]
                    for key, new in (('E_model', E_model), ('E_summary', E_summary),
                                     ('E_total', E_total)):
                        assert abs(new - o[key]) <= 1e-9 * max(1., abs(o[key])), \
                            (method, seed, pi, j, key, new, o[key])
                    assert k == o['segments']
                    rows.append({'method': method, 'seed': seed, 'path': pi, 'query': j,
                                 'E_model': E_model, 'E_summary': E_summary,
                                 'E_total': E_total, 'E_single': E_single,
                                 'E_equal': E_equal, 'segments': k,
                                 'fam_scale_reference': scale})
            print('evaluated', method, seed, f'{time.perf_counter() - started:.0f}s',
                  flush=True)
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))

    METRICS = ('E_model', 'E_summary', 'E_total', 'E_single', 'E_equal')
    unified = {}
    for met_m in METHODS:
        per_path = {}
        for r in rows:
            if r['method'] == met_m:
                per_path.setdefault(r['path'], []).append(r)
        unified[met_m] = {}
        for k2 in METRICS:
            unified[met_m][k2] = float(np.median(
                [100 * np.mean([x[k2] / x['fam_scale_reference'] for x in ps_])
                 for ps_ in per_path.values()]))
        unified[met_m]['segments'] = float(np.median(
            [np.mean([x['segments'] for x in ps_]) for ps_ in per_path.values()]))
    (OUT / 'results.json').write_text(json.dumps(
        {'protocol': ('e131 checkpoints/states/families recomputed with per-row '
                      'gates; unified per-query normalization identical to e127 '
                      '(E / fam_scale_reference, mean within path, median over '
                      'the 48 test paths, x100)'),
         'table_unified_normalized': unified}, indent=2))
    print(json.dumps(unified, indent=1))


if __name__ == '__main__':
    main()
