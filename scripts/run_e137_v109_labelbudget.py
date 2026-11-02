"""E137 (v109): response-label budget curve on Sioux Falls.

External review of v108 (P1-4): the learning contribution should show how
much response supervision suffices, not a single 384-center point.  This
experiment retrains the response objective with per-query derivative-center
budgets {6, 12, 24, 48} (36/72/144/288 centers in total; the full 64 per
query, 384 centers, is the E134 response run at the same lambda and is
reused, not retrained).  Budget subsets are prefixes of the frozen e111
per-query training-center order.  Everything else follows E134: same
initialization, 1200 steps, batch schedule, validation cadence, uniform
v104 selection rule, lambda fixed to the E134 validation choice.

Label cost per budget: one reference flow solve plus one sensitivity
solve per center; the 96 shared validation centers are a fixed cost
common to every method and budget.

Outputs: per-budget selected steps, 96-path test metrics, and the label
accounting used for the budget-curve figure.
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
from run_e83_v82_validation import attribution
import run_e134_v109_pareto as e134

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e137_v109_labelbudget'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'

BUDGETS = (6, 12, 24, 48, 64)      # training centers per query
STEPS = 1200
LR = 3e-5
BS = 32
BJ = 8


def train(seed, per_query, lam):
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    z = np.load(p.base.old.DATASET)
    lab = np.load(E111 / 'labels.npz')
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
    es = torch.tensor(lab['escale'])
    ks = torch.tensor(lab['kscale'])
    full_ids = [np.where((lab['j'] == j) & lab['train'])[0] for j in range(6)]
    train_ids = [ids[:per_query] for ids in full_ids]
    # response scale from the budget's own prefix centers only (the full
    # jscale would leak reference information beyond the label budget)
    js = torch.tensor([max(float(np.sqrt(np.mean(
        np.linalg.norm(lab['J'][ids], axis=1) ** 2))), 1e-6)
        for ids in train_ids])
    B = torch.tensor(n.B)
    t0 = torch.tensor(n.free_flow_time)
    cap = torch.tensor(n.capacity)
    vi = np.where(~lab['train'])[0]
    rng = np.random.default_rng(7102 + seed)

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
            print(f'e137 q{per_query} seed{seed} step{step} {val} '
                  f'({time.time() - st:.0f}s)', flush=True)
    elig_logs = [l for l in logs if l['eligible']]
    pool = [{'step': 0, 'validation': original}] + elig_logs
    winner = min(pool, key=lambda l: l['validation']['K_mse'])
    state = initial if winner['step'] == 0 else eligible_state
    assert winner['step'] == 0 or winner['step'] == eligible_step
    tag = f'q{per_query}_{seed}'
    torch.save({'initial': initial, 'selected': state,
                'selected_step': winner['step']}, OUT / f'{tag}.pt')
    (OUT / f'{tag}_cost.json').write_text(json.dumps(
        {'per_query': per_query, 'lam': lam, 'seed': seed,
         'seconds': time.time() - st, 'initial_validation': original,
         'selected_step': winner['step'], 'logs': logs}, indent=2))
    print(f'{tag} done selected_step={winner["step"]}', flush=True)


def evaluate(per_query):
    """96-path test metrics for one budget; the 64/query point reuses E134."""
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
    groups = np.concatenate([d_old['group'], d_new['group']])
    ts = d_old['ts']
    Cref = [attribution(np.array([T.grad(f) for f in Fref[pi]]),
                        Jold[pi], ts, Q[pi, -1] - Q[pi, 0]) for pi in range(24)]
    Cref = np.concatenate([np.array(Cref), Cnew])
    N = len(Q)
    lam = json.loads((e134.OUT / 'selection.json').read_text())[
        'methods']['response']['lambda']
    rows = []
    for seed in range(5):
        m = p.base.old.build_fullnet_model(ps, qbg, seed)
        if per_query == 64:
            ck = torch.load(e134.OUT / f'response_{e134.lam_tag(lam)}_{seed}.pt',
                            weights_only=False)
            step = ck['selected_step']
            m.load_state_dict(ck['initial'] if step == 0 else ck['selected'])
        else:
            ck = torch.load(OUT / f'q{per_query}_{seed}.pt', weights_only=False)
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
                    'per_query': per_query, 'seed': seed, 'path': pi,
                    'query': j, 'selected_step': int(step),
                    'flow_pct': float(flow),
                    'column_pct': float(100 * np.linalg.norm(e)
                                        / max(np.linalg.norm(r), 1e-6)),
                    'geometry_pct': float(100 * np.max(np.abs(geo[j] @ e))
                                          / max(np.max(np.abs(geo[j] @ r)), 1e-6)),
                })
        print('evaluated budget', per_query, 'seed', seed, flush=True)
    (OUT / f'eval_q{per_query}.json').write_text(json.dumps(rows, indent=2))


def collect():
    lam = json.loads((e134.OUT / 'selection.json').read_text())[
        'methods']['response']['lambda']
    curve = []
    for pq in BUDGETS:
        if pq == 64:
            # full-budget point: the E134 response run at the same lambda
            fp = e134.OUT / f'eval_response_{e134.lam_tag(lam)}.json'
        else:
            fp = OUT / f'eval_q{pq}.json'
        rows = json.loads(fp.read_text())
        vals = {}
        for r in rows:
            vals.setdefault(r['path'], []).append(r)
        per_path = {pi: {k: float(np.mean([x[k] for x in v]))
                         for k in ('flow_pct', 'column_pct', 'geometry_pct')}
                    for pi, v in vals.items()}
        if pq == 64:
            sel = json.loads((e134.OUT / 'selection.json').read_text())
            steps = sorted({r['selected_step']
                            for r in sel['methods']['response']['per_seed']})
        else:
            steps = sorted({r['selected_step'] for r in rows})
        curve.append({
            'per_query': pq, 'total_centers': 6 * pq, 'lam': lam,
            'selected_steps': steps,
            'column_pct': float(np.median([v['column_pct']
                                           for v in per_path.values()])),
            'geometry_pct': float(np.median([v['geometry_pct']
                                             for v in per_path.values()])),
            'flow_pct': float(np.median([v['flow_pct']
                                         for v in per_path.values()])),
        })
        print(curve[-1], flush=True)
    out = {
        'curve': curve,
        'label_cost': {
            'per_center': 'one reference flow solve + one sensitivity solve',
            'train_centers': {f'{6 * pq}': 6 * pq for pq in BUDGETS},
            'shared_validation_centers': 96,
        },
        'protocol': ('prefix subsets of the frozen e111 per-query center '
                     'order; identical budgets, selection rule and lambda '
                     '(E134 validation choice) across budgets; 96 frozen '
                     'test paths'),
    }
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(curve, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval', 'collect'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--per-query', type=int, choices=[6, 12, 24, 48])
    v = ap.parse_args()
    if v.cmd == 'train':
        lam = json.loads((e134.OUT / 'selection.json').read_text())[
            'methods']['response']['lambda']
        train(v.seed, v.per_query, lam)
    elif v.cmd == 'eval':
        evaluate(v.per_query)
    else:
        collect()
