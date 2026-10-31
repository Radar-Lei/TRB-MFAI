"""E119 (v106): expand the learned-assignment test set from 24 to 96 demand paths.

External review of v105 (P0): 24 fixed paths is a thin basis for the
learning claim. This experiment generates 72 additional paths with the
same e111 generation rule under a fresh RNG seed (18 per active-query
group for k in {1,2,3,6}; total 96 = 24 per group with the original 24),
solves their 41-state reference equilibria and sensitivities, and
evaluates the already-selected v104 checkpoints (methods flow and
response, five seeds, selection frozen from v104_learning/results.json;
no retraining, no re-tuning). Report paired path-level differences,
95% path-level bootstrap intervals, leave-one-path-out stability and
per-group medians. Paths are the independent unit; seeds are averaged
within a path before any summary. Metrics and aggregation mirror
run_v104_learning.py exactly.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json, time
from pathlib import Path
import numpy as np
import torch

import run_e111_v102_pairs as p
from run_e83_v82_validation import attribution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e119_v106_expanded_paths'
OUT.mkdir(exist_ok=True)
OLD = ROOT / 'experiments/results/v104_learning'
E111 = ROOT / 'experiments/results/e111_v102'
NEW_SEED = 102113
REPS_PER_GROUP = 18          # new paths per group; 18*4 = 72 new, 96 total
BOOT_SEED = 20261030
BOOT_REPS = 10000
METHODS = ('flow', 'response')
METRICS = ('flow_pct', 'column_pct', 'system_pct', 'geometry_pct')
PAPER_OVERALL = {'flow': {'column_pct': 21.61, 'system_pct': 2.33, 'geometry_pct': 7.91},
                 'response': {'column_pct': 14.22, 'system_pct': 3.31, 'geometry_pct': 5.81}}
PAPER_GROUPS_COLUMN = {'1': (9.65, 5.43), '2': (22.58, 16.41), '3': (19.43, 13.63), '6': (23.01, 14.44)}


def generate_new_paths():
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    rng = np.random.default_rng(NEW_SEED)
    ts = np.linspace(0, 1, 41)
    qs, ff, jj, groups = [], [], [], []
    started = time.perf_counter()
    for k in (1, 2, 3, 6):
        for r in range(REPS_PER_GROUP):
            a = rng.uniform(.65, 1.3, 6) * qb
            b = a.copy()
            active = [(r + i) % 6 for i in range(k)]
            delta = rng.uniform(.15, .3, k) * qb[active]
            if r % 2:
                delta[::2] *= -1
            b[active] += delta
            Q = a + ts[:, None] * (b - a)
            F, J, warm = [], [], None
            for q in Q:
                f, h, prob, _ = sv.solve(np.r_[q, qbg], warm)
                warm = f
                jac, *_ = sv.sensitivity(f, h, prob, np.r_[q, qbg])
                F.append(f)
                J.append(jac)
            qs.append(Q)
            ff.append(F)
            jj.append(J)
            groups.append(k)
        print('reference group', k, 'done', flush=True)
    np.savez_compressed(OUT / 'paths_new.npz', Q=qs, F=ff, J=jj, ts=ts, group=groups)
    return time.perf_counter() - started


def reference_columns(n, T, F, J, ts, Q):
    return np.array([attribution(np.array([T.grad(f) for f in F[pi]]), J[pi], ts, Q[pi, -1] - Q[pi, 0])
                     for pi in range(len(Q))])


def evaluate_models(n, ods, T, Q, ref, ts, masks):
    selection = {(M := s['method'], s['seed']): s['step']
                 for s in json.loads((OLD / 'results.json').read_text())['selection']}
    rows = []
    started = time.perf_counter()
    for method in METHODS:
        for seed in range(5):
            m = p.base.old.build_fullnet_model(ps_global, qbg_global, seed)
            step = selection[(method, seed)]
            ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else E111 / f'{method}_{seed}.pt'
            m.load_state_dict(torch.load(ck, weights_only=False)['best'])
            for pi in range(len(Q)):
                q = torch.tensor(Q[pi])
                f = m(q).detach().numpy()
                js = []
                for j in range(6):
                    v = torch.zeros_like(q)
                    v[:, j] = 1
                    _, jv = torch.func.jvp(m, (q,), (v,))
                    js.append(jv.detach().numpy())
                jac = np.stack(js, axis=2)
                C = attribution(np.array([T.grad(f0) for f0 in f]), jac, ts, Q[pi, -1] - Q[pi, 0])
                flow = 100 * np.mean(np.linalg.norm(f - Fref_global[pi], axis=1) / np.linalg.norm(Fref_global[pi], axis=1))
                for j, (s, t) in enumerate(ods):
                    if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                        continue
                    e = C[:, j] - ref[pi, :, j]
                    r = ref[pi, :, j]
                    geo = masks[j] @ n.B
                    rows.append({'method': method, 'seed': seed, 'path': pi,
                                 'flow_pct': float(flow),
                             'column_pct': float(100 * np.linalg.norm(e) / max(np.linalg.norm(r), 1e-6)),
                             'system_pct': float(100 * abs(e.sum()) / max(abs(r.sum()), 1e-6)),
                             'geometry_pct': float(100 * np.max(abs(geo @ e)) / max(np.max(abs(geo @ r)), 1e-6))})
            print('evaluated', method, seed, flush=True)
    return rows, time.perf_counter() - started


def path_level(rows, n_paths):
    vals = {}
    for m in METHODS:
        acc = {pi: [] for pi in range(n_paths)}
        for row in rows:
            if row['method'] == m:
                acc[row['path']].append(row)
        for pi in range(n_paths):
            if acc[pi]:
                vals[(m, pi)] = {met: float(np.mean([x[met] for x in acc[pi]])) for met in METRICS}
    return vals


def main():
    global ps_global, qbg_global, Fref_global
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    ps_global, qbg_global = ps, qbg
    protocol = {'new_seed': NEW_SEED, 'reps_per_group': REPS_PER_GROUP, 'new_paths': REPS_PER_GROUP * 4,
                'total_paths': 24 + REPS_PER_GROUP * 4, 'states_per_path': 41, 'bootstrap_seed': BOOT_SEED,
                'bootstrap_reps': BOOT_REPS, 'generation_rule': 'identical to e111 with fresh seed',
                'models': 'v104 selection frozen; no retraining'}
    # 1. new reference paths
    if (OUT / 'paths_new.npz').exists():
        protocol['reference_seconds'] = None
    else:
        protocol['reference_seconds'] = generate_new_paths()
    d = np.load(OUT / 'paths_new.npz')
    Q, Fnew, Jnew, ts, newgroup = d['Q'], d['F'], d['J'], d['ts'], d['group']
    Fref_global = Fnew
    ref = reference_columns(n, T, Fnew, Jnew, ts, Q)
    np.savez_compressed(OUT / 'reference_columns.npz', C=ref)
    # gates on the new references (supply includes the 522 fixed background demands)
    from run_e86_v84_newod import load_sioux_falls_trips
    fulltrips = load_sioux_falls_trips()
    study6 = list(map(tuple, ods))
    bgods = [o for o in sorted(fulltrips) if fulltrips[o] > 0 and o[0] != o[1] and o not in study6]
    Mb = n.od_M(bgods)
    qbgv = np.array([fulltrips[o] for o in bgods])
    Ms = n.od_M(study6)
    conservation = max(float(np.max(np.abs(n.B @ Fnew[pi].T - (Ms @ Q[pi].T + (Mb @ qbgv)[:, None]))))
                       for pi in range(len(Q)))
    completeness = []
    for pi in range(len(Q)):
        dt = T.value(Fnew[pi, -1]) - T.value(Fnew[pi, 0])
        completeness.append(float(abs(ref[pi].sum() - dt) / max(abs(dt), 1e-9)))
    # 2. learned evaluation on new paths
    masks = p.screenline_masks(ods)
    rows_new, eval_seconds = evaluate_models(n, ods, T, Q, ref, ts, masks)
    protocol['evaluation_seconds'] = eval_seconds
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (OUT / 'rows_new.json').write_text(json.dumps(rows_new, indent=2))
    # 3. combine with the original 24 paths from v104 (identical metrics)
    r_old = json.loads((OLD / 'results.json').read_text())
    rows_old = [row for row in r_old['rows'] if row['method'] in METHODS]
    vals_old = path_level(rows_old, 24)
    vals_new = path_level(rows_new, len(Q))
    group_old = np.load(E111 / 'paths.npz')['group']
    groups = np.concatenate([group_old, newgroup])
    vals = {}
    for pi in range(24):
        vals[pi] = vals_old[('flow', pi)], vals_old[('response', pi)]
    for pi in range(len(Q)):
        vals[24 + pi] = vals_new[('flow', pi)], vals_new[('response', pi)]
    # 4. summaries
    out = {'gates': {'max_reference_conservation': conservation,
                     'max_reference_completeness_relative': float(max(completeness))},
           'groups': {}, 'paired': {}, 'bootstrap': {}, 'loo': {}}
    N = 24 + len(Q)
    for g in (1, 2, 3, 6):
        idx = np.flatnonzero(groups == g)
        out['groups'][str(g)] = {'paths': int(len(idx)),
                                 'flow': {met: float(np.median([vals[pi][0][met] for pi in idx])) for met in METRICS},
                                 'response': {met: float(np.median([vals[pi][1][met] for pi in idx])) for met in METRICS}}
    overall = {m: {met: float(np.median([vals[pi][0 if m == 'flow' else 1][met] for pi in range(N)])) for met in METRICS}
               for m in METHODS}
    out['overall'] = overall
    rng = np.random.default_rng(BOOT_SEED)
    for met in ('column_pct', 'geometry_pct', 'system_pct'):
        dif = np.array([vals[pi][1][met] - vals[pi][0][met] for pi in range(N)])
        boot = np.array([np.median(rng.choice(dif, N, replace=True)) for _ in range(BOOT_REPS)])
        out['paired'][met] = {'median_difference': float(np.median(dif)),
                              'q1': float(np.percentile(dif, 25)), 'q3': float(np.percentile(dif, 75)),
                              'improved_path_share': float(np.mean(dif < 0)), 'improved_paths': int(np.sum(dif < 0)),
                              'bootstrap_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        for g in (1, 2, 3, 6):
            idx = np.flatnonzero(groups == g)
            dg = dif[idx]
            bootg = np.array([np.median(rng.choice(dg, len(idx), replace=True)) for _ in range(BOOT_REPS)])
            out['bootstrap'][f'{met}_group{g}'] = {'median_difference': float(np.median(dg)),
                                                   'improved_paths': int(np.sum(dg < 0)), 'paths': int(len(idx)),
                                                   'bootstrap_ci95': [float(np.percentile(bootg, 2.5)),
                                                                      float(np.percentile(bootg, 97.5))]}
        loos = []
        for pi in range(N):
            keep = np.delete(dif, pi)
            loos.append(float(np.median(keep)))
        out['loo'][met] = {'median_min': float(min(loos)), 'median_max': float(max(loos))}
    # 5. replication gate against the published 24-path numbers
    r24 = {m2: {met2: float(np.median([vals[pi][0 if m2 == 'flow' else 1][met2] for pi in range(24)]))
                for met2 in METRICS} for m2 in METHODS}
    repl = {m: {met: {'paper': PAPER_OVERALL[m][met], 'recomputed': round(r24[m][met], 4),
                      'abs_dev': round(abs(r24[m][met] - PAPER_OVERALL[m][met]), 6)}
                for met in PAPER_OVERALL[m]} for m in METHODS}
    for g, (fc, rc) in PAPER_GROUPS_COLUMN.items():
        idx = np.flatnonzero(groups[:24] == int(g))
        got_f = float(np.median([vals[pi][0]['column_pct'] for pi in idx]))
        got_r = float(np.median([vals[pi][1]['column_pct'] for pi in idx]))
        repl.setdefault('groups_column', {})[g] = {'paper': [fc, rc], 'recomputed': [round(got_f, 4), round(got_r, 4)]}
    out['replication_gate_24'] = repl
    np.savez_compressed(OUT / 'path_values.npz',
                        flow=np.array([[vals[pi][0][met] for met in METRICS] for pi in range(N)]),
                        response=np.array([[vals[pi][1][met] for met in METRICS] for pi in range(N)]),
                        group=groups, metrics=METRICS)
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'overall': overall, 'paired': {k: {kk: vv for kk, vv in v.items() if kk != 'bootstrap_ci95'}
                                                      for k, v in out['paired'].items()},
                      'gates': out['gates'], 'replication': repl}, indent=2))


if __name__ == '__main__':
    main()
