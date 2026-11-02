"""E147 (v111): random-subset repetitions of the label-budget curve.

External review of v110 (Section 6.3): the label-budget curve of E137 has
no subset-level uncertainty. This experiment repeats the budget training
with R random query-balanced nested subsets: for replication r in
{0,1,2}, each query's 64 training centers are permuted with
seed 20261101+r and nested prefixes of 6/12/24/48 centers per query form
the budgets (nesting keeps larger budgets supersets of smaller ones, as
in E137's frozen-prefix design). Five training seeds per (replication,
budget) as everywhere else; the full-budget point (64 per query, 384
centers) is the single E134 run, whose subset uncertainty is zero by
construction (one label set). Everything else follows E137: same
initialization, 1200 updates, batch schedule, validation cadence,
uniform selection rule, lambda fixed to the E134 validation choice.

Reported: per-budget median and interquartile range over the three
replications of the path-median metrics, and the center counts reaching
50%/80%/90% of the full-budget improvement over flow-only training.
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
import run_e134_v109_pareto as e134
from run_e83_v82_validation import attribution

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e147_v111_labelrep'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'
E137 = HERE / 'results/e137_v109_labelbudget'

BUDGETS = (6, 12, 24, 48)
REPS = (0, 1, 2)
SUBSET_SEED = 20261101
STEPS = 1200
LR = 3e-5
BS = 32
BJ = 8


def subset_order(rep, query, n=64):
    """Nested replication r: fixed permutation of the 64 per-query centers."""
    return np.random.default_rng(SUBSET_SEED + rep + 97 * query).permutation(n)


def train(seed, per_query, rep):
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    z = np.load(p.base.old.DATASET)
    lab = np.load(E111 / 'labels.npz')
    lam = json.loads((e134.OUT / 'selection.json').read_text())[
        'methods']['response']['lambda']
    m = p.base.old.build_fullnet_model(ps, qbg, seed)
    m.load_state_dict(torch.load(p.base.old.CKPT_DIR / f'seed{seed}.pt',
                                 weights_only=False)['best'])
    initial = {k: v.detach().clone() for k, v in m.state_dict().items()}
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    x = torch.tensor(z['Q'][:2500])
    f = torch.tensor(z['F'][:2500])
    a = torch.tensor(lab['a'])
    b = torch.tensor(lab['b'])
    zt = torch.tensor(lab['z'])
    qm = torch.tensor(lab['qm'])
    jm = torch.tensor(lab['J'])
    fs = torch.tensor(lab['fscale'])
    ks = torch.tensor(lab['kscale'])
    B = torch.tensor(n.B)
    t0 = torch.tensor(n.free_flow_time)
    cap = torch.tensor(n.capacity)
    vi = np.where(~lab['train'])[0]
    rng = np.random.default_rng(7102 + seed)
    # random nested subset: per-query permutation, nested prefixes
    ids_j = [np.where((lab['j'] == j) & lab['train'])[0] for j in range(6)]
    train_ids = []
    for j in range(6):
        ids = ids_j[j]
        key = (lab['a'][ids, j] * 7919 + lab['b'][ids, j])  # per-center identity key
        order = np.lexsort((lab['b'][ids, j], lab['a'][ids, j]))
        assert len(set(np.round(key[order], 12))) == len(ids)
        train_ids.append(ids[order][subset_order(rep, j)][:per_query])
    js = torch.tensor([max(float(np.sqrt(np.mean(
        np.linalg.norm(lab['J'][ids], axis=1) ** 2))), 1e-6)
        for ids in train_ids])

    def g(ft):
        return ft * t0 * (1 + .15 * (ft / cap) ** 4)

    def validate():
        with torch.no_grad():
            fa = m(a[vi])
            fb = m(b[vi])
            err = g(fb) - g(fa) - zt[vi]
            values = []
            for j, (s, t) in enumerate(ods):
                values.extend((p.kt(err[lab['j'][vi] == j], B, s, t)
                               / ks[j]).square().tolist())
            flow = torch.cat([fa - torch.tensor(lab['F0'][vi]),
                              fb - torch.tensor(lab['F1'][vi])])
            ref = torch.tensor(np.concatenate([lab['F0'][vi], lab['F1'][vi]]))
            fr = float((torch.linalg.vector_norm(flow, dim=1)
                        / torch.linalg.vector_norm(ref, dim=1)).mean())
        return {'K_mse': float(np.mean(values)), 'flow_relative': fr}

    original = validate()
    logs = []
    eligible_best = float('inf')
    eligible_state = None
    eligible_step = 0
    for step in range(1, STEPS + 1):
        ix = rng.integers(0, 2500, BS)
        j = (step - 1) % 6
        ij = rng.choice(train_ids[j], BJ)
        opt.zero_grad()
        lf = (((m(x[ix]) - f[ix]) / fs) ** 2).mean()
        v = torch.zeros_like(qm[ij])
        v[:, j] = 1
        _, jp = torch.func.jvp(m, (qm[ij],), (v,))
        lr_l = (jp - jm[ij]).square().sum(1).mean() / js[j] ** 2
        loss = lf + lam * lr_l
        loss.backward()
        opt.step()
        if step % p.EVERY == 0:
            val = validate()
            eligible = val['flow_relative'] <= 1.1 * original['flow_relative']
            if eligible and val['K_mse'] < eligible_best:
                eligible_best = val['K_mse']
                eligible_state = {k: v2.detach().clone()
                                  for k, v2 in m.state_dict().items()}
                eligible_step = step
            logs.append({'step': step, 'flow_loss': float(lf.detach()),
                         'aux_loss': float(lr_l.detach()), 'validation': val,
                         'eligible': bool(eligible),
                         'elapsed_seconds': time.time() - st})
            if step % 600 == 0:
                print(f'e147 r{rep} q{per_query} seed{seed} step{step} '
                      f'({time.time() - st:.0f}s)', flush=True)
    elig_logs = [l for l in logs if l['eligible']]
    pool = [{'step': 0, 'validation': original}] + elig_logs
    winner = min(pool, key=lambda l: l['validation']['K_mse'])
    state = initial if winner['step'] == 0 else eligible_state
    assert winner['step'] == 0 or winner['step'] == eligible_step
    tag = f'r{rep}_q{per_query}_{seed}'
    torch.save({'initial': initial, 'selected': state,
                'selected_step': winner['step']}, OUT / f'{tag}.pt')
    (OUT / f'{tag}_cost.json').write_text(json.dumps(
        {'rep': rep, 'per_query': per_query, 'lam': lam, 'seed': seed,
         'seconds': time.time() - st, 'initial_validation': original,
         'selected_step': winner['step'], 'logs': logs}, indent=2))
    print(f'{tag} done selected_step={winner["step"]}', flush=True)


def evaluate():
    """96-path metrics for every (rep, budget); full budget reuses E134/E137."""
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    masks = p.screenline_masks(ods)
    geo = [masks[j] @ n.B for j in range(6)]
    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Cnew = np.load(E119 / 'reference_columns.npz')['C']
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jold = d_old['J']
    ts = d_old['ts']
    Cref = np.concatenate([
        np.array([attribution(np.array([T.grad(f) for f in Fref[pi]]), Jold[pi],
                              ts, Q[pi, -1] - Q[pi, 0]) for pi in range(24)]),
        Cnew])
    N = len(Q)
    lam = json.loads((e134.OUT / 'selection.json').read_text())[
        'methods']['response']['lambda']
    for rep in REPS:
        for pq in BUDGETS:
            rows = []
            for seed in range(5):
                m = p.base.old.build_fullnet_model(ps, qbg, seed)
                ck = torch.load(OUT / f'r{rep}_q{pq}_{seed}.pt', weights_only=False)
                step = ck['selected_step']
                m.load_state_dict(ck['initial'] if step == 0 else ck['selected'])
                for pi in range(N):
                    f, C = e134.learned_columns(m, Q[pi], ts, T)
                    flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                                         / np.linalg.norm(Fref[pi], axis=1))
                    for j, (s, t) in enumerate(ods):
                        if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                            continue
                        e = C[:, j] - Cref[pi][:, j]
                        r = Cref[pi][:, j]
                        rows.append({
                            'rep': rep, 'per_query': pq, 'seed': seed,
                            'path': pi, 'query': j,
                            'flow_pct': float(flow),
                            'column_pct': float(100 * np.linalg.norm(e)
                                                / max(np.linalg.norm(r), 1e-6)),
                            'geometry_pct': float(100 * np.max(np.abs(geo[j] @ e))
                                                  / max(np.max(np.abs(geo[j] @ r)), 1e-6)),
                        })
            (OUT / f'eval_r{rep}_q{pq}.json').write_text(json.dumps(rows, indent=2))
            print('evaluated rep', rep, 'budget', pq, flush=True)


def collect():
    """Per-budget medians + IQR over replications; label-efficiency levels."""
    flow_level = 21.17768587460666          # e119 overall flow column error
    lam = json.loads((e134.OUT / 'selection.json').read_text())[
        'methods']['response']['lambda']
    full = json.loads((E137 / 'results.json').read_text())['curve'][-1]
    assert full['per_query'] == 64
    full_impr = {'column_pct': flow_level - full['column_pct'],
                 'geometry_pct': 8.742454456202914 - full['geometry_pct']}
    curve = []
    for pq in list(BUDGETS) + [64]:
        entries = []
        if pq < 64:
            for rep in REPS:
                rows = json.loads((OUT / f'eval_r{rep}_q{pq}.json').read_text())
                vals = {}
                for r in rows:
                    vals.setdefault(r['path'], []).append(r)
                per_path = {pi: {k: float(np.mean([x[k] for x in v]))
                                 for k in ('flow_pct', 'column_pct', 'geometry_pct')}
                            for pi, v in vals.items()}
                entries.append({k: float(np.median([v[k] for v in per_path.values()]))
                                for k in ('flow_pct', 'column_pct', 'geometry_pct')})
        else:
            entries.append({k: full[k] for k in
                            ('flow_pct', 'column_pct', 'geometry_pct')})
        agg = {}
        for k in ('flow_pct', 'column_pct', 'geometry_pct'):
            xs = [e[k] for e in entries]
            agg[k] = {'median': float(np.median(xs)),
                      'q1': float(np.percentile(xs, 25)),
                      'q3': float(np.percentile(xs, 75)),
                      'values': xs}
        curve.append({'per_query': pq, 'total_centers': 6 * pq,
                      'n_replications': len(entries), **agg})
    # improvement levels vs flow-only reference
    levels = {}
    for k, flow_ref in (('column_pct', flow_level), ('geometry_pct', 8.742454456202914)):
        full_gain = flow_ref - full[k]
        need = {}
        for frac in (0.5, 0.8, 0.9):
            target = flow_ref - frac * full_gain
            hit = None
            for c in curve:
                if c[k]['median'] <= target:
                    hit = c['total_centers']
                    break
            need[f'{int(frac * 100)}pct'] = hit
        levels[k] = {'flow_reference': flow_ref, 'full_improvement': full_gain,
                     'centers_reaching': need}
    out = {'protocol': {'subset_seed': SUBSET_SEED, 'replications': list(REPS),
                        'subset_rule': ('per-query permutation of the 64 frozen training '
                                        'centers, nested prefixes per budget; query-balanced '
                                        'by construction'),
                        'training': 'e137 protocol unchanged (lambda from e134 validation choice)'},
           'curve': curve,
           'levels': levels}
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'curve': [{k: v[k]['median'] for k in
                                 ('column_pct', 'geometry_pct', 'flow_pct')}
                                for v in curve], 'levels': levels}, indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('phase', choices=['train', 'eval', 'collect'])
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--budget', type=int, choices=BUDGETS)
    a.add_argument('--rep', type=int, choices=REPS)
    v = a.parse_args()
    if v.phase == 'train':
        train(v.seed, v.budget, v.rep)
    elif v.phase == 'eval':
        evaluate()
    else:
        collect()
