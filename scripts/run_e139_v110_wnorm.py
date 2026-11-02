"""E139 (v110): scale-normalized weighted-response objective (external review
of v109, item 3.1).

The v109 weighted-response loss normalized the weighted discrepancy by the
UNWEIGHTED reference scale u_j (jscale), so the weighted and unweighted
objectives shared a lambda grid but not an internal scale: under a unit
change T -> cT the weighted loss scales by c^2 while the flow and unweighted
losses do not, making the effective regularization range of the weighted
objective unit-dependent.

This experiment reruns ONLY the weighted-response objective with its own
reference scale

    v_j^2 = E_train[ || D^S(q) J^S_j(q) ||^2 ],

with D^S the TSTT marginal weights at each training center's reference
flows (t0*(1+0.75*(F/cap)^4), the same weights the discrepancy itself
uses).  Everything else is frozen exactly as in e134: the five e111
pretrained initializations, steps (1200), lr (3e-5), batches (32 flow +
8 derivative centers, rng 7102+seed), validation cadence, the uniform v104
selection rule with candidate pool including step 0, the 1.1x initial flow
constraint, the lambda grid, and the 96 frozen test paths.  The
protection floor max(v_j, 1e-6) carries the weighted units and is computed
from training labels only.

Gates:
  (i) flow and response(lambda*) 96-path medians must equal the stored
      e134 values exactly (rows are merged from e134, not retrained);
  (ii) at tau=5% nothing changes (this experiment retrains nothing that
      e127/e138 consume);
  (iii) vscale/jscale ratios are reported as the units check.

Outputs: selection.json (validation-only lambda choice under the same
rule), eval_wresponse_<lam>.json per lambda, results.json with the final
six-objective table (five methods merged from e134 + normalized wresponse),
paired improvement vs flow, Pareto rows, and the scale-comparison block.
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
OUT = HERE / 'results/e139_v110_wnorm'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'
E134 = HERE / 'results/e134_v109_pareto'

LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
STEPS = 1200
LR = 3e-5
BS = 32
BJ = 8
METRICS = ('flow_pct', 'column_pct', 'system_pct', 'geometry_pct', 'boundary_pct')


def lam_tag(lam):
    return f'{lam:g}'


def scales():
    """Weighted-response scale v_j over the 384 training centers (frozen
    labels): RMS of ||D^S J^S_j|| with D^S the TSTT marginal weights at the
    center's reference flows."""
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    lab = np.load(E111 / 'labels.npz')
    t0 = n.free_flow_time
    cap = n.capacity
    wS = t0[None, :] * (1 + .75 * (lab['Fm'] / cap[None, :]) ** 4)
    vs = np.zeros(6)
    js = np.zeros(6)
    for j, (s, t) in enumerate(ods):
        ix = (lab['j'] == j) & lab['train']
        disc = wS[ix] * lab['J'][ix]
        vs[j] = max(float(np.sqrt(np.mean(np.sum(disc ** 2, axis=1)))), 1e-6)
        js[j] = max(float(np.sqrt(np.mean(np.sum(lab['J'][ix] ** 2, axis=1)))),
                    1e-6)
    return vs, js


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


def train(seed, lam):
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    z = np.load(p.base.old.DATASET)
    lab = np.load(E111 / 'labels.npz')
    if not (OUT / 'vscale.npy').exists():
        vs, js = scales()
        np.save(OUT / 'vscale.npy', vs)
        np.save(OUT / 'jscale_check.npy', js)
    vs = np.load(OUT / 'vscale.npy')
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
    vscale = torch.tensor(vs)
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
                ks = torch.tensor(lab['kscale'])[j]
                values.extend((p.kt(err[lab['j'][vi] == j], B, s, t)
                               / ks).square().tolist())
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
        pr, jp = torch.func.jvp(m, (qm[ij],), (v,))
        wL = t0 * (1 + .75 * (pr / cap) ** 4)
        wS = t0 * (1 + .75 * (fm[ij] / cap) ** 4)
        disc = wL * jp - wS * jm[ij]
        laux = disc.square().sum(1).mean() / vscale[j] ** 2
        loss = lf + lam * laux
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
                         'aux_loss': float(laux.detach()), 'validation': val,
                         'eligible': bool(eligible),
                         'elapsed_seconds': time.time() - st})
            print(f'wnorm lam{lam_tag(lam)} seed{seed} step{step} '
                  f'{val} ({time.time() - st:.0f}s)', flush=True)
    hist = {'method': 'wresponse_norm', 'lam': lam, 'seed': seed,
            'seconds': time.time() - st, 'initial_validation': original,
            'logs': logs}
    wstep, wval = uniform_winner(hist)
    state = initial if wstep == 0 else eligible_state
    assert wstep == 0 or (wstep == eligible_step), (wstep, eligible_step)
    hist['selected_step'] = wstep
    tag = f'wresponse_{lam_tag(lam)}_{seed}'
    torch.save({'initial': initial, 'selected': state, 'selected_step': wstep,
                'eligible_state': eligible_state, 'eligible_step': eligible_step},
               OUT / f'{tag}.pt')
    (OUT / f'{tag}_cost.json').write_text(json.dumps(hist, indent=2))
    print(f'{tag} done selected_step={wstep}', flush=True)


def select():
    """Validation-only lambda choice for the normalized weighted objective;
    flow baseline and comparison lambdas are imported from e134."""
    e134 = json.loads((E134 / 'selection.json').read_text())
    flow_baseline = e134['flow_baseline']
    sel = {'flow': e134['methods']['flow']}
    cand = []
    for lam in LAMBDAS:
        per_seed = []
        ok = True
        for seed in range(5):
            fp = OUT / f'wresponse_{lam_tag(lam)}_{seed}_cost.json'
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
    sel['wresponse_norm'] = {'lambda': winner['lam'], 'candidates': cand,
                             'flow_constraint_threshold': threshold,
                             'constrained_choice': bool(eligible_lams),
                             'per_seed': winner['per_seed']}
    sel['wresponse_jscale_v109'] = {
        'lambda': e134['methods']['wresponse']['lambda'],
        'note': ('v109 jscale-normalized weighted objective, retained for '
                 'the scale-comparison block only')}
    vs = np.load(OUT / 'vscale.npy')
    js = np.load(OUT / 'jscale_check.npy')
    js_e111 = np.load(E111 / 'labels.npz')['jscale']
    assert np.allclose(js, js_e111, rtol=1e-12), 'jscale reproduction failed'
    (OUT / 'selection.json').write_text(json.dumps(
        {'flow_baseline': flow_baseline, 'methods': sel,
         'vscale': vs.tolist(), 'jscale': js.tolist(),
         'vscale_jscale_ratio': (vs / js).tolist(),
         'protocol': ('normalized weighted objective: ||D^L J^L - D^S J^S||^2 '
                      'divided by v_j^2 = E_train||D^S J^S||^2 with TSTT '
                      'marginal weights at reference flows; everything else '
                      'frozen as e134; lambda on validation only under the '
                      'same rule')}, indent=2))
    print('wresponse_norm lambda* =', winner['lam'],
          'medK', round(winner['median_selected_K_mse'], 5),
          'medFlow', round(winner['median_selected_flow_relative'], 5),
          'steps', winner['selected_steps'], flush=True)
    print('vscale/jscale ratio', np.round(vs / js, 3).tolist(), flush=True)


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


def load_model(lam, seed):
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    m = p.base.old.build_fullnet_model(ps, qbg, seed)
    sel = json.loads((OUT / 'selection.json').read_text())['methods']
    cand = next(c for c in sel['wresponse_norm']['candidates'] if c['lam'] == lam)
    entry = next(r for r in cand['per_seed'] if r['seed'] == seed)
    step = entry['selected_step']
    ck = torch.load(OUT / f'wresponse_{lam_tag(lam)}_{seed}.pt', weights_only=False)
    m.load_state_dict(ck['initial'] if step == 0 else ck['selected'])
    return m, step


def evaluate(lam):
    """96-path evaluation of the normalized weighted objective at lambda."""
    tag = f'wresponse_{lam_tag(lam)}'
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
        m, step = load_model(lam, seed)
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
                    'method': 'wresponse_norm', 'lam': lam, 'seed': seed,
                    'path': pi, 'query': j, 'group': int(groups[pi]),
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
    """Final six-objective table: five methods merged from stored e134 evals
    (gate: medians must match e134 results.json exactly) + normalized
    weighted objective at its lambda*; Pareto rows and paired improvement."""
    e134 = json.loads((E134 / 'results.json').read_text())
    sel = json.loads((OUT / 'selection.json').read_text())
    lam = sel['methods']['wresponse_norm']['lambda']
    rows_new = []
    for fp in sorted(OUT.glob('eval_*.json')):
        rows_new.extend(json.loads(fp.read_text()))

    def summarize(rows):
        vals = {}
        for r in rows:
            vals.setdefault(r['path'], []).append(r)
        per_path = {pi: {k: float(np.mean([x[k] for x in v])) for k in METRICS}
                    for pi, v in vals.items()}
        return ({k: float(np.median([v[k] for v in per_path.values()]))
                 for k in METRICS}, per_path)

    final = {}
    pareto = [{'method': 'flow', 'lam': None,
               **{k: e134['final']['flow'][k] for k in METRICS}}]
    for method in ('flow', 'response', 'kboundary', 'edge', 'cut'):
        e = e134['final'][method]
        final[method] = {k: e[k] for k in ('lam',) + METRICS}
        if method != 'flow':
            pareto.append({'method': method, 'lam': e['lam'],
                           **{k: e[k] for k in METRICS}})
    # full frozen e134 weight sweeps for the four untouched objectives, so
    # the Pareto figure carries every objective's whole curve (the e134
    # wresponse sweep is superseded by the renormalized rerun below)
    for p in e134['pareto']:
        if p['method'] in ('flow', 'wresponse'):
            continue
        if p['lam'] == e134['final'][p['method']]['lam']:
            continue
        pareto.append(dict(p))
    # gate: e134 rows re-summarized from the raw eval files must match the
    # stored medians bit-for-bit before they are merged into the new table
    for method in ('response', 'kboundary', 'edge', 'cut'):
        l2 = e134['final'][method]['lam']
        tag = f'{method}_{lam_tag(l2)}'
        raw = json.loads((E134 / f'eval_{tag}.json').read_text())
        med, _ = summarize(raw)
        for k2 in METRICS:
            assert abs(med[k2] - e134['final'][method][k2]) < 1e-9, \
                ('e134 merge gate', method, k2)
    rows_at_lam = [r for r in rows_new if r['lam'] == lam]
    assert rows_at_lam, ('missing eval at lambda*', lam)
    med, per_path = summarize(rows_at_lam)
    final['wresponse'] = {'lam': lam, **med, 'normalization': 'v_j (own scale)'}
    for l2 in LAMBDAS:
        rl = [r for r in rows_new if r['lam'] == l2]
        if not rl:
            continue
        ml, _ = summarize(rl)
        pareto.append({'method': 'wresponse', 'lam': l2, 'normalization': 'v_j',
                       **ml})

    improvement = []
    for seed in range(5):
        acc = {}
        for r in rows_new:
            if r['seed'] != seed or r['lam'] != lam:
                continue
            acc.setdefault(('wresponse', r['path']), []).append(r)
        e134rows = json.loads(
            (E134 / 'eval_flow.json').read_text()) if False else None
        flow_rows = [r for r in json.loads((E134 / 'eval_flow.json').read_text())
                     if r['seed'] == seed]
        for r in flow_rows:
            acc.setdefault(('flow', r['path']), []).append(r)
        for k2 in ('column_pct', 'geometry_pct'):
            pv = {kk: float(np.mean([x[k2] for x in v])) for kk, v in acc.items()}
            dif = [pv[('wresponse', pi)] - pv[('flow', pi)] for pi in range(96)
                   if ('wresponse', pi) in pv and ('flow', pi) in pv]
            improvement.append({'seed': seed, 'metric': k2,
                                'improved_paths': int(np.sum(np.array(dif) < 0)),
                                'paths': len(dif),
                                'improved_share': float(np.mean(np.array(dif) < 0))})

    out = {
        'final': final, 'pareto': pareto, 'improvement_wresponse': improvement,
        'scale_comparison': {
            'vscale': sel['vscale'], 'jscale': sel['jscale'],
            'ratio': sel['vscale_jscale_ratio'],
            'lambda_jscale_v109': e134['final']['wresponse']['lam'],
            'final_jscale_v109': {k: e134['final']['wresponse'][k]
                                  for k in METRICS},
        },
        'protocol': ('weighted objective renormalized by its own training '
                     'scale v_j^2=E_train||D^S J^S||^2; all other methods '
                     'merged from e134 with exact-median gates; identical '
                     'budgets, initialization, batch schedule, selection rule; '
                     'lambda on validation only'),
    }
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(final, indent=2))
    print('scale ratio', np.round(sel['vscale_jscale_ratio'], 3).tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'select', 'eval', 'collect'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--lam', type=float)
    v = ap.parse_args()
    if v.cmd == 'train':
        train(v.seed, v.lam)
    elif v.cmd == 'select':
        select()
    elif v.cmd == 'eval':
        evaluate(v.lam)
    else:
        collect()
