"""E150 (v112): exact-parameterization rows under the scaled pretraining loss.

External review of v111 (Section 4 / Appendix conservation ablation): the
exact path-share rows of the ablation reuse models pretrained on the
unscaled loss while the direct (penalty/projection) rows use the per-link
scaled loss, so the exact-versus-direct gap confounds parameterization with
pretraining recipe.  Plan B of the review: add an exact architecture
pretrained on the same per-link scaled loss as the direct models, and report
exact-unscaled / exact-scaled / penalty-scaled / projection-scaled.

Everything except the pretraining objective is shared: same dataset and
split, same optimizer (Adam 3e-3), batch 256, at most 12000 steps, validation
every 50 steps with the min-validation checkpoint (validation metric is the
unscaled relative L2 flow error for all architectures, as in E93/E146 --- the
scaling changes the training loss only); continuation follows the E111
protocol unchanged (1200 updates, shared labels and minibatch sequences, the
uniform selection rule with the pre-trained initial model in the candidate
pool); evaluation recomputes the E146 metrics on the same 96 test paths.

Gates: the exact-unscaled and direct rows must reproduce the stored E146
rows exactly (they are copied, not retrained); exact-scaled rows must
satisfy the path-share conservation identities (node residual ~1e-11).

Outputs: pre_exact_scaled_{seed}.pt, exact_scaled_{method}_{seed}.pt
(+cost json), rows.json (exact_scaled rows only), results.json (merged
four-architecture summary).
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
import run_e146_v111_conservation as e146

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e150_v112_exactscaled'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E146 = HERE / 'results/e146_v111_conservation'
OLD = HERE / 'results/v104_learning'

METHODS = ('flow', 'response')
STEPS = 1200
LR = 3e-5
BS = 32
BJ = 8
PRE_LR = 3e-3
PRE_BATCH = 256
PRE_STEPS = 12000
PRE_EVERY = 50


def pretrain(seed):
    """Exact path-share model on the per-link scaled MSE (the direct-model
    objective), E93 schedule, unscaled relative-L2 validation selection."""
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    z = np.load(p.base.old.DATASET)
    Q, F = z['Q'], z['F']
    n_tr = int(z['split'][0])
    q_mu, q_sd = Q[:n_tr].mean(0), Q[:n_tr].std(0)
    # per-link scale identical to the continuation protocol (fscale)
    fscale = np.sqrt(np.mean(F[:n_tr] ** 2, axis=0)) * 0.01
    m = p.base.old.build_fullnet_model(ps, qbg, seed, q_mu, q_sd)
    opt = torch.optim.Adam(m.parameters(), lr=PRE_LR)
    Xt = torch.tensor(Q[:n_tr])
    Ft = torch.tensor(F[:n_tr])
    fst = torch.tensor(fscale)
    Xv = torch.tensor(Q[n_tr:n_tr + int(z['split'][1])])
    Fv = torch.tensor(F[n_tr:n_tr + int(z['split'][1])])
    ds = torch.utils.data.TensorDataset(Xt, Ft)
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=PRE_BATCH, shuffle=True,
                                     generator=g)

    def val_error():
        m.eval()
        with torch.no_grad():
            f = m(Xv)
            rel = (f - Fv).norm(dim=1) / Fv.norm(dim=1)
            return float(rel.mean())

    best_val, best_state, best_step = float('inf'), None, -1
    step = 0
    while step < PRE_STEPS:
        for xb, yb in dl:
            opt.zero_grad()
            pred = m(xb)
            loss = (((pred - yb) / fst) ** 2).mean()
            loss.backward()
            opt.step()
            step += 1
            if step % PRE_EVERY == 0:
                ve = val_error()
                if ve < best_val:
                    best_val, best_step = ve, step
                    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                if step % 2000 == 0:
                    print(f'pre exact_scaled {seed} step {step} loss {float(loss):.3e} '
                          f'val {ve:.4%} best {best_val:.4%} ({time.time()-st:.0f}s)', flush=True)
            if not np.isfinite(float(loss)):
                raise RuntimeError(f'pretrain diverged: exact_scaled seed {seed} step {step}')
            if step >= PRE_STEPS:
                break
    torch.save({'best': best_state, 'best_val': best_val, 'best_step': best_step,
                'norm': {'mu_q': q_mu, 'sd_q': q_sd}},
               OUT / f'pre_exact_scaled_{seed}.pt')
    (OUT / f'pre_exact_scaled_{seed}_cost.json').write_text(json.dumps(
        {'seconds': time.time() - st, 'best_val': best_val, 'best_step': best_step},
        indent=2))
    print(f'pre exact_scaled {seed}: best val {best_val:.4%} at {best_step}', flush=True)


def cont(seed, method):
    """E111 continuation from the exact-scaled pretrained checkpoint."""
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    z = np.load(p.base.old.DATASET)
    lab = np.load(E111 / 'labels.npz')
    ck = torch.load(OUT / f'pre_exact_scaled_{seed}.pt', weights_only=False)
    m = p.base.old.build_fullnet_model(ps, qbg, seed,
                                       ck['norm']['mu_q'], ck['norm']['sd_q'])
    m.load_state_dict(ck['best'])
    initial = {k: v.detach().clone() for k, v in m.state_dict().items()}
    initial_val_flow = None
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    x = torch.tensor(z['Q'][:2500])
    f = torch.tensor(z['F'][:2500])
    a = torch.tensor(lab['a'])
    bq = torch.tensor(lab['b'])
    zt = torch.tensor(lab['z'])
    qm = torch.tensor(lab['qm'])
    jm = torch.tensor(lab['J'])
    fs = torch.tensor(lab['fscale'])
    ks = torch.tensor(lab['kscale'])
    full_ids = [np.where((lab['j'] == j) & lab['train'])[0] for j in range(6)]
    js = torch.tensor([max(float(np.sqrt(np.mean(
        np.linalg.norm(lab['J'][ids], axis=1) ** 2))), 1e-6) for ids in full_ids])
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
            fb = m(bq[vi])
            err = g(fb) - g(fa) - zt[vi]
            values = []
            for j, (s, t) in enumerate(ods):
                values.extend((p.kt(err[lab['j'][vi] == j], B, s, t) / ks[j]).square().tolist())
            flow = torch.cat([fa - torch.tensor(lab['F0'][vi]), fb - torch.tensor(lab['F1'][vi])])
            ref = torch.tensor(np.concatenate([lab['F0'][vi], lab['F1'][vi]]))
            fr = float((torch.linalg.vector_norm(flow, dim=1) / torch.linalg.vector_norm(ref, dim=1)).mean())
        return {'K_mse': float(np.mean(values)), 'flow_relative': fr}

    original = validate()
    initial_val_flow = original['flow_relative']
    logs = []
    best = None
    eligible_best = float('inf')
    fallback = float('inf')
    beststep = 0
    for step in range(1, STEPS + 1):
        ix = rng.integers(0, 2500, BS)
        j = (step - 1) % 6
        ij = rng.choice(full_ids[j], BJ)
        opt.zero_grad()
        lf = (((m(x[ix]) - f[ix]) / fs) ** 2).mean()
        loss = lf
        if method == 'response':
            v = torch.zeros_like(qm[ij])
            v[:, j] = 1
            _, jp = torch.func.jvp(m, (qm[ij],), (v,))
            loss = loss + (jp - jm[ij]).square().sum(1).mean() / js[j] ** 2
        loss.backward()
        opt.step()
        if step % 100 == 0:
            val = validate()
            eligible = val['flow_relative'] <= 1.1 * original['flow_relative']
            take = (eligible and val['K_mse'] < eligible_best) or \
                   (eligible_best == float('inf') and val['flow_relative'] < fallback)
            if take:
                best = {k: v.detach().clone() for k, v in m.state_dict().items()}
                beststep = step
                fallback = val['flow_relative']
                if eligible:
                    eligible_best = val['K_mse']
            logs.append({'step': step, 'flow_loss': float(lf.detach()),
                         'validation': val, 'elapsed_seconds': time.time() - st})
    # candidate pool includes the pre-trained initial model (uniform rule):
    # if no continuation state qualified, keep the initial checkpoint.
    if best is None:
        best = initial
        beststep = 0
    torch.save({'best': best, 'selected_step': beststep,
                'initial': initial, 'initial_val_flow': initial_val_flow},
               OUT / f'exact_scaled_{method}_{seed}.pt')
    (OUT / f'exact_scaled_{method}_{seed}_cost.json').write_text(json.dumps(
        {'seconds': time.time() - st, 'selected_step': beststep,
         'original_validation': original, 'logs': logs}, indent=2))
    print(f'cont exact_scaled {method} {seed}: step {beststep} '
          f'({time.time()-st:.0f}s)', flush=True)


def eval_scaled():
    """E146 metrics for the exact-scaled checkpoints (96 paths)."""
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    Ms, b = e146.balance_terms(n, ods, qbg)
    chains = direction_levels(net=n)
    fams = [query_families(s, t, chains=chains, net=n) for s, t in ods]
    geo = []
    for fam_j in fams:
        geo.append([np.array([[v in U for v in range(n.n)] for U in f['masks']]) @ B
                    for f in fam_j])
    dedup_idx = e127.dedup_indices(ods, chains, n)

    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(HERE / 'results/e119_v106_expanded_paths/paths_new.npz')
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jref = np.concatenate([d_old['J'], d_new['J']])
    ts = d_old['ts']
    Cref = np.concatenate([
        np.array([attribution(np.array([T.grad(f0) for f0 in Fref[pi]]), Jref[pi], ts,
                              Q[pi, -1] - Q[pi, 0]) for pi in range(24)]),
        np.load(HERE / 'results/e119_v106_expanded_paths/reference_columns.npz')['C']])
    N = len(Q)
    mu_norm = {j: np.linalg.norm(Ms[:, j]) for j in range(6)}

    rows = []
    started = time.perf_counter()
    for method in METHODS:
        for seed in range(5):
            ck = torch.load(OUT / f'exact_scaled_{method}_{seed}.pt', weights_only=False)
            pre = torch.load(OUT / f'pre_exact_scaled_{seed}.pt', weights_only=False)
            m = p.base.old.build_fullnet_model(ps, qbg, seed,
                                               pre['norm']['mu_q'], pre['norm']['sd_q'])
            m.load_state_dict(ck['best'])
            for pi in range(N):
                f, jac, C = e127.learned_columns(m, Q[pi], ts, T)
                gL = np.array([T.grad(f0) for f0 in f])
                gS = np.array([T.grad(f0) for f0 in Fref[pi]])
                dQ = Q[pi, -1] - Q[pi, 0]
                flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                                     / np.linalg.norm(Fref[pi], axis=1))
                cons_res = float(np.max(np.abs(f @ B.T - Q[pi] @ Ms.T - b)))
                dres = max(float(np.max(np.abs(jac[s][:, j] @ B.T - Ms[:, j])))
                           / mu_norm[j] for s in range(len(f)) for j in range(6))
                for j, (s, t) in enumerate(ods):
                    if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                        continue
                    cL, cS = C[:, j], Cref[pi][:, j]
                    aL = [-(g @ cL) for g in geo[j]]
                    aS = [-(g @ cS) for g in geo[j]]
                    E_model = max(np.max(np.abs(al - as_)) for al, as_ in zip(aL, aS))
                    epsL = 0.05 * max(np.max(np.abs(al)) for al in aL)
                    seg_counts, hatL = [], []
                    for al in aL:
                        segs = e127.greedy_segments(al, epsL)
                        seg_counts.append(len(segs))
                        hatL.append(e127.summarize(al, segs))
                    E_summary = max(np.max(np.abs(h - al)) for h, al in zip(hatL, aL))
                    E_total = max(np.max(np.abs(h - as_)) for h, as_ in zip(hatL, aS))
                    famscale = float(max(np.max(np.abs(as_)) for as_ in aS))
                    rows.append({
                        'arch': 'exact_scaled', 'method': method, 'seed': seed,
                        'path': pi, 'query': j,
                        'flow_pct': float(flow),
                        'increment_pct': float(100 * np.linalg.norm(
                            (f[-1] - f[0]) - (Fref[pi, -1] - Fref[pi, 0]))
                            / max(np.linalg.norm(Fref[pi, -1] - Fref[pi, 0]), 1e-6)),
                        'cons_residual_max': cons_res,
                        'derivative_residual_max': dres,
                        'column_pct': float(100 * np.linalg.norm(cL - cS)
                                            / max(np.linalg.norm(cS), 1e-6)),
                        'system_pct': float(100 * abs((cL - cS).sum())
                                            / max(abs(cS.sum()), 1e-6)),
                        'E_model': float(E_model), 'E_summary': float(E_summary),
                        'E_total': float(E_total), 'segments': int(sum(seg_counts)),
                        'per_direction_segments': seg_counts,
                        'fam_scale_reference': famscale,
                    })
            print('eval exact_scaled', method, seed,
                  f'{time.perf_counter()-started:.0f}s', flush=True)
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))
    return rows


def collect():
    """Merge E146 rows with the exact-scaled rows into the four-recipe table."""
    rows = json.loads((E146 / 'rows.json').read_text())
    rows += json.loads((OUT / 'rows.json').read_text())
    N = 96
    METRICS = ('flow_pct', 'increment_pct', 'cons_residual_max',
               'derivative_residual_max', 'column_pct', 'system_pct',
               'E_model', 'E_summary', 'E_total')
    summary = {}
    for arch in ('exact', 'exact_scaled', 'penalty', 'projection'):
        for method in METHODS:
            rr = [r for r in rows if r['arch'] == arch and r['method'] == method]
            per_path = {}
            for r in rr:
                per_path.setdefault(r['path'], []).append(r)
            ent = {k: float(np.median([np.mean([x[k] for x in v]) for v in per_path.values()]))
                   for k in METRICS}
            ent['E_total_normalized'] = float(np.median([
                100 * np.mean([x['E_total'] / x['fam_scale_reference'] for x in v])
                for v in per_path.values()]))
            summary[f'{arch}:{method}'] = ent
    gates = {
        'exact_scaled_conservation_max': float(max(
            r['cons_residual_max'] for r in rows if r['arch'] == 'exact_scaled')),
        'n_rows': len(rows),
    }
    (OUT / 'results.json').write_text(json.dumps(
        {'protocol': {'pretrain': ('exact path-share architecture, per-link scaled MSE '
                                   '(the direct-model objective), E93 schedule (Adam 3e-3, batch 256, '
                                   '<=12000 steps, validation every 50, min-validation checkpoint on the '
                                   'unscaled relative-L2 metric)'),
                      'continuation': 'e111 protocol unchanged (1200 updates, shared labels, uniform selection rule)',
                      'rows_exact_direct': 'copied from frozen e146 rows.json'},
         'gates': gates, 'summary': summary}, indent=2))
    print(json.dumps({'gates': gates,
                      'summary': {k: {m: round(v[m], 4) for m in v} for k, v in summary.items()}},
                     indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('phase', choices=['pretrain', 'cont', 'eval', 'collect'])
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--method', choices=METHODS, default='flow')
    v = a.parse_args()
    if v.phase == 'pretrain':
        pretrain(v.seed)
    elif v.phase == 'cont':
        cont(v.seed, v.method)
    elif v.phase == 'eval':
        eval_scaled()
    else:
        collect()
