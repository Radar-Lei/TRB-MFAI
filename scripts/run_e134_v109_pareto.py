"""E134 (v109): fair lambda-sweep comparison of all six supervision objectives.

External review of v108 (P0-1): the v102 four-objective comparison tuned no
loss weight (edge 1, cut 0.01, response 1) and three edge/cut seeds fell
back to the initial checkpoint, so 'response supervision is best' could
reflect hyperparameter choice rather than the objective.  This experiment
gives every objective the same budget and the same weight search:

  objectives (auxiliary term added to the flow loss, loss = flow + lam*aux):
    flow       - none (frozen e111 training reused, no retraining)
    response   - unweighted Jacobian MSE  ||J^L_j - J^S_j||^2 / js^2
    wresponse  - weighted response MSE    ||D^L J^L_j - D^S J^S_j||^2 / js^2
                with D = diag(t(f)) the TSTT marginal weights evaluated at
                each model's own flows (reference weights at reference flows)
    kboundary  - pointwise boundary seminorm (K_j(D^L J^L_j - D^S J^S_j))^2
                / kws^2 on the same training centers
    edge       - endpoint attribution difference g(m(b))-g(m(a))-z, /es^2
    cut        - endpoint bundle aux = edge + 0.01*cut-seminorm (the v102
                internal edge-to-cut ratio is retained), /es^2 and /ks^2
  lambda grid: {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}

  All runs start from the same five e111 pretrained checkpoints, share
  steps (1200), lr (3e-5), batch sizes (32 flow + 8 derivative centers),
  validation cadence (every 100 steps) and the frozen v104 selection rule:
  among {the initial checkpoint and every validation state with validation
  flow error <= 1.1x the initial} take the minimum validation K-MSE.  The
  lambda of each objective is chosen on validation data only: among
  lambdas whose seed-median selected validation flow error stays within
  1.1x the seed-median initial error, take the smallest seed-median
  validation K-MSE.  Test paths enter only the final report.

  Gates: (i) the response lambda=1 column must reproduce the e119 stored
  96-path medians to <= 0.02 percentage points; (ii) the re-selected flow
  control must match the frozen v104 selection.
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

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e134_v109_pareto'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'
V104 = HERE / 'results/v104_learning'

LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SWEEP_METHODS = ('response', 'wresponse', 'kboundary', 'edge', 'cut')
ALL_METHODS = ('flow',) + SWEEP_METHODS
STEPS = 1200
LR = 3e-5
BS = 32
BJ = 8
CUT_WEIGHT = .01
METRICS = ('flow_pct', 'column_pct', 'system_pct', 'geometry_pct', 'boundary_pct')


def lam_tag(lam):
    return f'{lam:g}'


def scales():
    """Weighted-response scales over the 384 training centers (frozen labels).
    Each center stores its own query's derivative column (labels.npz J is
    (n_centers, n_links)), so centers are grouped by their query index."""
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    lab = np.load(E111 / 'labels.npz')
    t0 = n.free_flow_time
    cap = n.capacity
    wS = t0[None, :] * (1 + .75 * (lab['Fm'] / cap[None, :]) ** 4)
    kw = np.zeros(6)
    for j, (s, t) in enumerate(ods):
        ix = (lab['j'] == j) & lab['train']
        disc = wS[ix] * lab['J'][ix]
        kw[j] = max(float(np.sqrt(np.mean(p.kn(disc, n.B, s, t) ** 2))), 1e-6)
    return kw


def uniform_winner(hist):
    """The frozen v104 rule: min validation K-MSE over
    {step 0 (initial)} + {validation states with flow error <= 1.1x initial}."""
    original = hist['initial_validation']
    eligible = [l for l in hist['logs']
                if l['validation']['flow_relative']
                <= 1.1 * original['flow_relative']]
    pool = [{'step': 0, 'validation': original}] + eligible
    winner = min(pool, key=lambda l: l['validation']['K_mse'])
    return winner['step'], winner['validation']


def train(seed, method, lam):
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    z = np.load(p.base.old.DATASET)
    lab = np.load(E111 / 'labels.npz')
    kw = np.load(OUT / 'kwscale.npy') if (OUT / 'kwscale.npy').exists() else None
    if kw is None:
        kw = scales()
        np.save(OUT / 'kwscale.npy', kw)
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
    fm = torch.tensor(lab['Fm'])
    fs = torch.tensor(lab['fscale'])
    es = torch.tensor(lab['escale'])
    ks = torch.tensor(lab['kscale'])
    js = torch.tensor(lab['jscale'])
    kws = torch.tensor(kw)
    B = torch.tensor(n.B)
    t0 = torch.tensor(n.free_flow_time)
    cap = torch.tensor(n.capacity)
    train_ids = [np.where((lab['j'] == j) & lab['train'])[0] for j in range(6)]
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
    fallback = float('inf')
    fallback_state = None
    fallback_step = 0
    for step in range(1, STEPS + 1):
        ix = rng.integers(0, 2500, BS)
        j = (step - 1) % 6
        ij = rng.choice(train_ids[j], BJ)
        opt.zero_grad()
        lf = (((m(x[ix]) - f[ix]) / fs) ** 2).mean()
        loss = lf
        laux = torch.tensor(0.)
        if method in ('edge', 'cut'):
            err = g(m(b[ij])) - g(m(a[ij])) - zt[ij]
            le = err.square().sum(1).mean() / es[j] ** 2
            loss = loss + lam * le
            laux = le.detach()
            if method == 'cut':
                s, t = ods[j]
                lc = (p.kt(err, B, s, t) / ks[j]).square().mean()
                loss = loss + lam * CUT_WEIGHT * lc
                laux = laux + CUT_WEIGHT * lc.detach()
        elif method in ('response', 'wresponse', 'kboundary'):
            v = torch.zeros_like(qm[ij])
            v[:, j] = 1
            pr, jp = torch.func.jvp(m, (qm[ij],), (v,))
            if method == 'response':
                lr_l = (jp - jm[ij]).square().sum(1).mean() / js[j] ** 2
            else:
                wL = t0 * (1 + .75 * (pr / cap) ** 4)
                wS = t0 * (1 + .75 * (fm[ij] / cap) ** 4)
                disc = wL * jp - wS * jm[ij]
                if method == 'wresponse':
                    lr_l = disc.square().sum(1).mean() / js[j] ** 2
                else:
                    s, t = ods[j]
                    lr_l = (p.kt(disc, B, s, t) / kws[j]).square().mean()
            loss = loss + lam * lr_l
            laux = lr_l.detach()
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
            if not eligible and val['flow_relative'] < fallback:
                fallback = val['flow_relative']
                fallback_state = {k: v2.detach().clone()
                                  for k, v2 in m.state_dict().items()}
                fallback_step = step
            logs.append({'step': step, 'flow_loss': float(lf.detach()),
                         'aux_loss': float(laux), 'validation': val,
                         'eligible': bool(eligible),
                         'elapsed_seconds': time.time() - st})
            print(f'{method} lam{lam_tag(lam)} seed{seed} step{step} '
                  f'{val} ({time.time() - st:.0f}s)', flush=True)
    hist = {'method': method, 'lam': lam, 'seed': seed,
            'seconds': time.time() - st, 'initial_validation': original,
            'logs': logs}
    wstep, wval = uniform_winner(hist)
    # the states the uniform rule can choose from
    state = initial if wstep == 0 else eligible_state
    assert wstep == 0 or (wstep == eligible_step), (wstep, eligible_step)
    hist['selected_step'] = wstep
    tag = f'{method}_{lam_tag(lam)}_{seed}'
    torch.save({'initial': initial, 'selected': state, 'selected_step': wstep,
                'eligible_state': eligible_state, 'eligible_step': eligible_step,
                'fallback_state': fallback_state, 'fallback_step': fallback_step},
               OUT / f'{tag}.pt')
    (OUT / f'{tag}_cost.json').write_text(json.dumps(hist, indent=2))
    print(f'{tag} done selected_step={wstep}', flush=True)


def select():
    """Validation-only lambda choice per method; writes selection.json."""
    sel = {}
    flow_per_seed = []
    for seed in range(5):
        hist = json.loads((E111 / f'flow_{seed}_cost.json').read_text())
        hist['initial_validation'] = hist['original_validation']
        wstep, wval = uniform_winner(hist)
        flow_per_seed.append({'seed': seed, 'selected_step': wstep,
                              'selected': wval,
                              'initial': hist['initial_validation']})
        v104 = json.loads((V104 / 'results.json').read_text())['selection']
        frozen = next(s['step'] for s in v104
                      if s['method'] == 'flow' and s['seed'] == seed)
        assert wstep == frozen, ('flow selection mismatch', seed, wstep, frozen)
    flow_baseline = {
        'K_mse': float(np.median([r['initial']['K_mse'] for r in flow_per_seed])),
        'flow_relative': float(np.median(
            [r['initial']['flow_relative'] for r in flow_per_seed])),
    }
    sel['flow'] = {'lambda': None, 'per_seed': flow_per_seed}
    for method in SWEEP_METHODS:
        cand = []
        for lam in LAMBDAS:
            per_seed = []
            ok = True
            for seed in range(5):
                fp = OUT / f'{method}_{lam_tag(lam)}_{seed}_cost.json'
                if not fp.exists():
                    ok = False
                    break
                h = json.loads(fp.read_text())
                wstep, wval = uniform_winner(h)
                per_seed.append({'seed': seed, 'selected_step': wstep,
                                 'selected': wval,
                                 'initial': h['initial_validation']})
            if not ok:
                continue
            med_k = float(np.median([r['selected']['K_mse'] for r in per_seed]))
            med_f = float(np.median(
                [r['selected']['flow_relative'] for r in per_seed]))
            cand.append({'lam': lam, 'median_selected_K_mse': med_k,
                         'median_selected_flow_relative': med_f,
                         'initial_model_seeds': [r['seed'] for r in per_seed
                                                 if r['selected_step'] == 0],
                         'selected_steps': [r['selected_step'] for r in per_seed],
                         'per_seed': per_seed})
        threshold = 1.1 * flow_baseline['flow_relative']
        eligible_lams = [c for c in cand
                         if c['median_selected_flow_relative'] <= threshold]
        pool = eligible_lams if eligible_lams else cand
        winner = min(pool, key=lambda c: (c['median_selected_K_mse'],
                                          c['median_selected_flow_relative']))
        sel[method] = {'lambda': winner['lam'], 'candidates': cand,
                       'flow_constraint_threshold': threshold,
                       'constrained_choice': bool(eligible_lams),
                       'per_seed': winner['per_seed']}
        print(method, 'lambda* =', winner['lam'],
              'medK', round(winner['median_selected_K_mse'], 5),
              'medFlow', round(winner['median_selected_flow_relative'], 5),
              'steps', winner['selected_steps'], flush=True)
    (OUT / 'selection.json').write_text(json.dumps(
        {'flow_baseline': flow_baseline, 'methods': sel,
         'protocol': ('uniform v104 selection rule (pool includes the initial '
                      'checkpoint) for every method; lambda chosen on '
                      'validation only: among lambdas with seed-median selected '
                      'validation flow error <= 1.1x the initial seed-median, '
                      'minimize seed-median validation K-MSE; ties broken by '
                      'flow error')}, indent=2))


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


def load_model(method, lam, seed):
    """Build the model selected by selection.json for (method[, lam], seed)."""
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    m = p.base.old.build_fullnet_model(ps, qbg, seed)
    sel = json.loads((OUT / 'selection.json').read_text())['methods']
    if method == 'flow':
        entry = next(r for r in sel['flow']['per_seed'] if r['seed'] == seed)
        step = entry['selected_step']
        ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else \
            E111 / f'flow_{seed}.pt'
        m.load_state_dict(torch.load(ck, weights_only=False)['best'])
        return m, step
    cand = next(c for c in sel[method]['candidates'] if c['lam'] == lam)
    entry = next(r for r in cand['per_seed'] if r['seed'] == seed)
    step = entry['selected_step']
    ck = torch.load(OUT / f'{method}_{lam_tag(lam)}_{seed}.pt', weights_only=False)
    m.load_state_dict(ck['initial'] if step == 0 else ck['selected'])
    return m, step


def evaluate(method, lam):
    """96-path evaluation of one (method, lambda); writes eval_<tag>.json."""
    tag = 'flow' if method == 'flow' else f'{method}_{lam_tag(lam)}'
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
    rows = []
    st = time.perf_counter()
    for seed in range(5):
        m, step = load_model(method, lam, seed)
        for pi in range(N):
            f, C = learned_columns(m, Q[pi], ts, T)
            flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                                 / np.linalg.norm(Fref[pi], axis=1))
            for j, (s, t) in enumerate(ods):
                if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                    continue
                e = C[:, j] - Cref[pi][:, j]
                r = Cref[pi][:, j]
                kr = p.kn(r, n.B, s, t)
                rows.append({
                    'method': method, 'lam': lam, 'seed': seed, 'path': pi,
                    'query': j, 'group': int(groups[pi]),
                    'flow_pct': float(flow),
                    'column_pct': float(100 * np.linalg.norm(e)
                                        / max(np.linalg.norm(r), 1e-6)),
                    'system_pct': float(100 * abs(e.sum())
                                        / max(abs(r.sum()), 1e-6)),
                    'geometry_pct': float(100 * np.max(np.abs(geo[j] @ e))
                                          / max(np.max(np.abs(geo[j] @ r)), 1e-6)),
                    'boundary_pct': float(100 * p.kn(e, n.B, s, t)
                                          / max(kr, 1e-6)),
                })
        print(f'eval {tag} seed{seed} step{step} '
              f'{time.perf_counter() - st:.0f}s', flush=True)
    (OUT / f'eval_{tag}.json').write_text(json.dumps(rows, indent=2))


def collect():
    """Aggregate eval files: final table at lambda*, Pareto points for all
    lambdas, five-seed paired improvement, replication gate vs e119."""
    sel = json.loads((OUT / 'selection.json').read_text())
    rows_all = []
    for fp in sorted(OUT.glob('eval_*.json')):
        rows_all.extend(json.loads(fp.read_text()))

    def summarize(rows):
        vals = {}
        for r in rows:
            vals.setdefault(r['path'], []).append(r)
        per_path = {pi: {k: float(np.mean([x[k] for x in v])) for k in METRICS}
                    for pi, v in vals.items()}
        return ({k: float(np.median([v[k] for v in per_path.values()]))
                 for k in METRICS}, per_path)

    final = {}
    pareto = []
    for method in ALL_METHODS:
        lam = None if method == 'flow' else sel['methods'][method]['lambda']
        tag = 'flow' if method == 'flow' else f'{method}_{lam_tag(lam)}'
        rows = [r for r in rows_all
                if r['method'] == method
                and (method == 'flow' or r['lam'] == lam)]
        assert rows, ('missing eval', tag)
        med, _ = summarize(rows)
        final[method] = {
            'lam': lam, **med,
            'selected_steps': [r['selected_step'] for r in
                               sel['methods'][method]['per_seed']]}
        if method == 'flow':
            pareto.append({'method': 'flow', 'lam': None, **med})
        else:
            for l2 in LAMBDAS:
                rl = [r for r in rows_all
                      if r['method'] == method and r['lam'] == l2]
                if not rl:
                    continue
                ml, _ = summarize(rl)
                pareto.append({'method': method, 'lam': l2, **ml})

    improvement = {}
    for method in SWEEP_METHODS:
        lam = sel['methods'][method]['lambda']
        tab = []
        for seed in range(5):
            acc = {}
            for r in rows_all:
                if r['seed'] != seed:
                    continue
                if r['method'] == 'flow' or (r['method'] == method
                                             and r['lam'] == lam):
                    acc.setdefault((r['method'], r['path']), []).append(r)
            for k2 in ('column_pct', 'geometry_pct'):
                pv = {kk: float(np.mean([x[k2] for x in v]))
                      for kk, v in acc.items()}
                dif = [pv[(method, pi)] - pv[('flow', pi)] for pi in range(96)
                       if (method, pi) in pv and ('flow', pi) in pv]
                tab.append({'seed': seed, 'metric': k2,
                            'improved_paths': int(np.sum(np.array(dif) < 0)),
                            'paths': len(dif),
                            'improved_share': float(np.mean(np.array(dif) < 0))})
        improvement[method] = tab

    gate = {}
    e119 = json.loads((E119 / 'results.json').read_text())['overall']
    mr, _ = summarize([r for r in rows_all
                       if r['method'] == 'response' and r['lam'] == 1.0])
    mf, _ = summarize([r for r in rows_all if r['method'] == 'flow'])
    for meth, mm in (('response', mr), ('flow', mf)):
        for k2 in ('column_pct', 'geometry_pct', 'system_pct'):
            gate[f'{meth}_{k2}'] = {'e119': e119[meth][k2],
                                    'e134': round(mm[k2], 4),
                                    'abs_dev': round(abs(mm[k2]
                                                         - e119[meth][k2]), 6)}
            assert abs(mm[k2] - e119[meth][k2]) <= .02 + 1e-9, gate

    out = {'final': final, 'pareto': pareto, 'improvement': improvement,
           'replication_gate': gate,
           'protocol': ('identical budgets, initialization, batch schedule and '
                        'selection rule across objectives; lambda on validation '
                        'only; 96 frozen test paths reported once per '
                        'configuration')}
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(final, indent=2))
    print('gate ok:', json.dumps(gate, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'select', 'eval', 'collect'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--method', choices=SWEEP_METHODS)
    ap.add_argument('--lam', type=float)
    ap.add_argument('--eval-method', choices=ALL_METHODS)
    ap.add_argument('--eval-lam', type=float)
    v = ap.parse_args()
    if v.cmd == 'train':
        train(v.seed, v.method, v.lam)
    elif v.cmd == 'select':
        select()
    elif v.cmd == 'eval':
        evaluate(v.eval_method, v.eval_lam)
    else:
        collect()
