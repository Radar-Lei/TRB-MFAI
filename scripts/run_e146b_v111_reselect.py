"""E146b (v111): conservation-ablation re-evaluation.

Codex round-1 fixes (P1-3, P2-8):
(a) The paper's uniform selection rule pools the pre-trained initial
    model (update 0) with every 100th continuation update whose
    validation flow error is at most 1.1 times the initial model's
    value, and takes the minimum validation boundary MSE. E146's
    original implementation selected only among continuation
    checkpoints. This script recomputes the selection for every direct
    run from the saved validation logs (the update-0 candidate is the
    run's original_validation) and evaluates the 96 test paths with the
    newly selected state (the saved 'initial' state when update 0
    wins, the saved 'best' continuation checkpoint otherwise).
(b) Every row additionally reports the integrated flow-attribution
    residual of Eq. (eq:residualidentity), R_j = B g_j - mu_j dq_j
    with g_j the trapezoid-integrated learned flow Jacobian column
    (same quadrature as the outcome attribution), normalized as
    ||R_j||_2 / (|dq_j| * ||mu_j||_2).

Exact-mechanism rows reuse the published checkpoints unchanged and are
re-evaluated only to add the residual column. Writes rows.json and
results.json into results/e146_v111_conservation (overwriting), with
the per-run selection audit in results.json['selection_audit'].
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import time
from multiprocessing import Pool

import numpy as np
import torch

import run_e111_v102_pairs as p
import run_e127_v108_totalerror as e127
import run_e146_v111_conservation as e146
from fam_v108 import direction_levels, query_families
from run_e83_v82_validation import attribution

HERE = e146.HERE
OUT = e146.OUT
E111 = e146.E111
E119 = e146.E119
OLD = e146.OLD
METHODS = ('flow', 'response')
ARCHS = ('exact', 'penalty', 'projection')


def reselect(arch, method, seed):
    """Uniform rule incl. update 0, from the saved validation logs."""
    cost = json.loads((OUT / f'{arch}_{method}_{seed}_cost.json').read_text())
    orig = cost['original_validation']
    cands = [(0, orig['K_mse'], orig['flow_relative'])]
    for lg in cost['logs']:
        v = lg['validation']
        if v['flow_relative'] <= 1.1 * orig['flow_relative']:
            cands.append((lg['step'], v['K_mse'], v['flow_relative']))
    best = min(cands, key=lambda c: (c[1], c[0]))
    changed = best[0] != cost['selected_step']
    return best[0], changed, {'candidates': len(cands),
                              'old_step': cost['selected_step'],
                              'new_step': best[0],
                              'new_K_mse': best[1],
                              'initial_K_mse': orig['K_mse']}


def eval_run(args):
    arch, method, seed = args
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    Ms, b = e146.balance_terms(n, ods, qbg)
    chains = direction_levels(net=n)
    fams = [query_families(s, t, chains=chains, net=n) for s, t in ods]
    geo = [[np.array([[v in U for v in range(n.n)] for U in f['masks']]) @ B
            for f in fam_j] for fam_j in fams]
    dedup_idx = e127.dedup_indices(ods, chains, n)

    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jref = np.concatenate([d_old['J'], d_new['J']])
    ts = d_old['ts']
    Cref = np.concatenate([
        np.array([attribution(np.array([T.grad(f0) for f0 in Fref[pi]]), Jref[pi], ts,
                              Q[pi, -1] - Q[pi, 0]) for pi in range(24)]),
        np.load(E119 / 'reference_columns.npz')['C']])
    mu_norm = {j: float(np.linalg.norm(Ms[:, j])) for j in range(6)}

    if arch == 'exact':
        selection = {(s['method'], s['seed']): s['step']
                     for s in json.loads((OLD / 'results.json').read_text())['selection']}
        m = p.base.old.build_fullnet_model(ps, qbg, seed)
        step = selection[(method, seed)]
        ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else \
            E111 / f'{method}_{seed}.pt'
        m.load_state_dict(torch.load(ck, weights_only=False)['best'])
    else:
        step, changed, _ = reselect(arch, method, seed)
        ck = torch.load(OUT / f'{arch}_{method}_{seed}.pt', weights_only=False)
        pre = torch.load(OUT / f'pre_{arch}_{seed}.pt', weights_only=False)
        m, _, _ = e146.build_direct_model(n, ps, qbg, seed, arch,
                                          pre['norm']['mu_q'], pre['norm']['sd_q'],
                                          width=pre['width'])
        m.set_balance(Ms, b)
        m.load_state_dict(ck['initial'] if step == 0 else ck['best'])

    rows = []
    for pi in range(len(Q)):
        f, jac, C = e127.learned_columns(m, Q[pi], ts, T)
        dQ = Q[pi, -1] - Q[pi, 0]
        flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                             / np.linalg.norm(Fref[pi], axis=1))
        cons_res = float(np.max(np.abs(f @ B.T - Q[pi] @ Ms.T - b)))
        dres = max(float(np.max(np.abs(jac[s][:, j] @ B.T - Ms[:, j])))
                   / mu_norm[j] for s in range(len(f)) for j in range(6))
        # integrated flow-attribution residual R_j (Eq. residualidentity)
        gL = np.trapezoid(jac, ts, axis=0) * dQ[None, :]
        r_attr = max(float(np.linalg.norm(B @ gL[:, j] - Ms[:, j] * dQ[j])
                           / (abs(dQ[j]) * mu_norm[j]))
                     for j in range(6) if abs(dQ[j]) > 1e-10)
        for j, (s, t) in enumerate(ods):
            if abs(dQ[j]) < 1e-10:
                continue
            cL, cS = C[:, j], Cref[pi][:, j]
            aL = [-(g @ cL) for g in geo[j]]
            aS = [-(g @ cS) for g in geo[j]]
            E_model = max(np.max(np.abs(al - as_)) for al, as_ in zip(aL, aS))
            epsL = 0.05 * max(np.max(np.abs(al)) for al in aL)
            seg_counts, hatL = [], []
            for al in aL:
                sg = e127.greedy_segments(al, epsL)
                seg_counts.append(len(sg))
                hatL.append(e127.summarize(al, sg))
            rows.append({
                'arch': arch, 'method': method, 'seed': seed,
                'path': pi, 'query': j, 'selected_step': int(step),
                'flow_pct': float(flow),
                'increment_pct': float(100 * np.linalg.norm(
                    (f[-1] - f[0]) - (Fref[pi, -1] - Fref[pi, 0]))
                    / max(np.linalg.norm(Fref[pi, -1] - Fref[pi, 0]), 1e-6)),
                'cons_residual_max': cons_res,
                'derivative_residual_max': dres,
                'attribution_residual_max': r_attr,
                'column_pct': float(100 * np.linalg.norm(cL - cS)
                                    / max(np.linalg.norm(cS), 1e-6)),
                'system_pct': float(100 * abs((cL - cS).sum())
                                    / max(abs(cS.sum()), 1e-6)),
                'E_model': float(E_model),
                'E_summary': float(max(np.max(np.abs(h - al))
                                       for h, al in zip(hatL, aL))),
                'E_total': float(max(np.max(np.abs(h - as_))
                                     for h, as_ in zip(hatL, aS))),
                'segments': int(sum(seg_counts)),
                'per_direction_segments': seg_counts,
                'fam_scale_reference': float(max(np.max(np.abs(as_)) for as_ in aS)),
            })
    return rows


def main():
    audit = {}
    jobs = []
    for arch in ARCHS:
        for method in METHODS:
            for seed in range(5):
                if arch != 'exact':
                    step, changed, info = reselect(arch, method, seed)
                    audit[f'{arch}:{method}:{seed}'] = {**info, 'changed': changed}
                    print(f'reselect {arch} {method} {seed}: '
                          f'{info["old_step"]} -> {info["new_step"]}', flush=True)
                jobs.append((arch, method, seed))
    if (OUT / 'rows.json').exists() and 'attribution_residual_max' in \
            json.loads((OUT / 'rows.json').read_text())[0]:
        rows = json.loads((OUT / 'rows.json').read_text())
        print('reusing completed rows.json', flush=True)
    else:
        with Pool(processes=min(10, len(jobs))) as pool:
            all_rows = pool.map(eval_run, jobs)
        rows = [r for rr in all_rows for r in rr]
        (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))

    # gates: exact rows must reproduce published values
    ex = [r for r in rows if r['arch'] == 'exact']
    pub = json.loads((E119 / 'results.json').read_text())['overall']
    gates = {}
    for method in METHODS:
        got = float(np.median([
            np.mean([r['column_pct'] for r in ex
                     if r['method'] == method and r['path'] == pi])
            for pi in range(96)]))
        gates[f'exact_{method}_column_pct'] = {'published': round(pub[method]['column_pct'], 4),
                                               'recomputed': round(got, 4)}
        assert abs(got - pub[method]['column_pct']) < 5e-3, (method, got)
    gates['projection_conservation_max'] = float(max(
        r['cons_residual_max'] for r in rows if r['arch'] == 'projection'))
    gates['exact_conservation_max'] = float(max(
        r['cons_residual_max'] for r in rows if r['arch'] == 'exact'))

    METRICS = ('flow_pct', 'increment_pct', 'cons_residual_max',
               'derivative_residual_max', 'attribution_residual_max',
               'column_pct', 'system_pct', 'E_model', 'E_summary', 'E_total')
    n0 = p.base.setup()[0]
    chains = direction_levels(net=n0)
    ods0 = p.base.setup()[1]
    dedup_idx = e127.dedup_indices(ods0, chains, n0)
    summary = {}
    for arch in ARCHS:
        for method in METHODS:
            rr = [r for r in rows if r['arch'] == arch and r['method'] == method]
            per_path = {}
            for r in rr:
                per_path.setdefault(r['path'], []).append(r)
            ent = {k: float(np.median([np.mean([x[k] for x in v])
                                       for v in per_path.values()]))
                   for k in METRICS}
            segd = []
            for v in per_path.values():
                tot = [sum(x['per_direction_segments'][i]
                           for i in dedup_idx[x['query']]['first_indices']) for x in v]
                segd.append(float(np.mean(tot)))
            ent['segments_distinct_chains'] = float(np.median(segd))
            ent['E_total_normalized'] = float(np.median([
                100 * np.mean([x['E_total'] / x['fam_scale_reference'] for x in v])
                for v in per_path.values()]))
            summary[f'{arch}:{method}'] = ent
    prev = json.loads((OUT / 'results.json').read_text())
    prev['summary'] = summary
    prev['gates'] = gates
    prev['selection_audit'] = audit
    prev['protocol']['selection'] = ('uniform rule incl. the pre-trained initial model '
                                     '(update 0); recomputed by run_e146b_v111_reselect.py')
    prev['protocol']['attribution_residual'] = ('max_j ||B g_j - mu_j dq_j||_2 / '
                                                '(|dq_j| ||mu_j||_2), g_j trapezoid-integrated '
                                                'learned Jacobian (Eq. residualidentity)')
    (OUT / 'results.json').write_text(json.dumps(prev, indent=2))
    changed = {k: v for k, v in audit.items() if v['changed']}
    print(json.dumps({'changed_runs': changed,
                      'summary': {k: {kk: round(vv, 4) for kk, vv in v.items()}
                                  for k, v in summary.items()}}, indent=1), flush=True)


if __name__ == '__main__':
    main()
