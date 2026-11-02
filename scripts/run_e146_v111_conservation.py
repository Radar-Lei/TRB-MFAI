"""E146 (v111): conservation-mechanism ablation on Sioux Falls.

External review of v110 (Section 6.2): the paper claims that exact output
conservation transfers to flow-valued integrated gradients while approximate
conservation need not, but the architectures were never ablated directly.
This experiment trains, under the shared pre-specified protocol, three
architectures:

  exact       the conservative path-share parameterization of Eq. (consparam)
              (published models reused: e93 pretraining + e111 continuation);
  penalty     a direct edge-flow MLP of matched parameter count plus a
              conservation penalty term in the loss;
  projection  the same direct MLP followed by a fixed differentiable
              least-norm projection onto the conservation manifold.

each under flow-only and response supervision, and evaluates them on the
same 96 test demand paths with the same metrics as E119/E127, adding the
node-balance residual and the pointwise derivative-balance residual that
separate the two conservation mechanisms.

Protocol controls (fixed before training): matched parameter count;
identical pretraining data, optimizer, schedule and step cap as the exact
architecture (E93), with the per-link scaled mean-square objective of the
stated pretraining protocol (the unscaled objective destabilizes the
unbounded direct output layer); identical continuation protocol,
derivative labels, validation sets and selection rule as E111; penalty
weight fixed at 1.0 on a nodal-throughput-normalized residual; no
test-path information in any choice. The exact-architecture rows are
recomputed from the frozen checkpoints and must reproduce the published
E119/E127 values.

Outputs: pre_{arch}_{seed}.pt, {arch}_{method}_{seed}.pt (+cost json),
rows.json, results.json.
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

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e146_v111_conservation'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'
OLD = HERE / 'results/v104_learning'

ARCHS = ('exact', 'penalty', 'projection')
METHODS = ('flow', 'response')
STEPS = 1200
LR = 3e-5
BS = 32
BJ = 8
PENALTY_W = 1.0          # pre-specified, on the normalized residual
PRE_LR = 3e-3
PRE_BATCH = 256
PRE_STEPS = 12000
PRE_EVERY = 50


def build_direct_model(n, ps, q_bg, seed, arch, mu_q=None, sd_q=None, width=None,
                       f_mean=None):
    """Direct edge-flow MLP; parameter count matched to the exact model.

    forward: f_raw = MLP(standardized q);  penalty returns f_raw;
    projection returns f_raw - (B f_raw - M q - b) P^T with
    P = B^T pinv(B B^T), which satisfies Bf = Mq + b exactly because the
    residual is orthogonal to 1 (= lies in range(B)).  The final-layer
    bias can be initialized to a reference flow field (output
    conditioning; checkpoint state overrides it after training).
    """
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = np.zeros(6) if mu_q is None else np.asarray(mu_q, float)
    sd = np.ones(6) if sd_q is None else np.asarray(sd_q, float)
    K, m = 6, ps.m
    if width is None:
        # parameter count of the exact architecture (6-128-128-128-5280)
        target = (K * 128 + 128) + 2 * (128 * 128 + 128) + (128 * ps.P + ps.P)
        # direct: K*W+W + 2*(W*W+W) + W*m+m  ->  2W^2 + (K+3+m)W + m
        a, b, c = 2.0, float(K + 3 + m), float(m)
        width = int(round((-b + np.sqrt(b * b + 4 * a * (target - c))) / (2 * a)))
    n_param = (K * width + width) + 2 * (width * width + width) + (width * m + m)

    class DirectNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], K
            for _ in range(3):
                layers += [nn.Linear(d, width), nn.Tanh()]
                d = width
            layers.append(nn.Linear(d, m))
            if f_mean is not None:
                with torch.no_grad():
                    layers[-1].bias.copy_(torch.tensor(np.asarray(f_mean, float)))
            self.net = nn.Sequential(*layers)
            B = torch.tensor(n.B, dtype=torch.float64)
            self.register_buffer('B', B)
            # study-OD incidence and constant background balance supplied later
            self.register_buffer('Mst', torch.zeros(n.n, K, dtype=torch.float64))
            self.register_buffer('bvec', torch.zeros(n.n, dtype=torch.float64))
            if arch == 'projection':
                G = torch.linalg.pinv(B @ B.T)
                self.register_buffer('Pproj', (B.T @ G).contiguous())
            self.register_buffer('mu_q', torch.tensor(mu, dtype=torch.float64))
            self.register_buffer('sd_q', torch.tensor(sd, dtype=torch.float64))

        def set_balance(self, Mst, bvec):
            self.Mst.copy_(torch.tensor(Mst, dtype=torch.float64))
            self.bvec.copy_(torch.tensor(bvec, dtype=torch.float64))

        def forward(self, q):
            z = self.net((q - self.mu_q) / self.sd_q)
            if not hasattr(self, 'Pproj'):
                return z
            r = z @ self.B.T - q @ self.Mst.T - self.bvec
            return z - r @ self.Pproj.T
    return DirectNet().double(), width, n_param


def balance_terms(n, ods, qbg):
    """Study incidence M and constant background balance b = M_bg q_bg."""
    from run_e86_v84_newod import load_sioux_falls_trips
    fulltrips = load_sioux_falls_trips()
    study6 = list(map(tuple, ods))
    bgods = [o for o in sorted(fulltrips) if fulltrips[o] > 0 and o[0] != o[1]
             and o not in study6]
    Mb = n.od_M(bgods)
    qbgv = np.array([fulltrips[o] for o in bgods])
    Ms = n.od_M(study6)
    return Ms, Mb @ qbgv


def cons_scale(Q, Ms, b):
    """Nodal-throughput scale s_v = RMS over states of |(Mq+b)_v|, floored
    at the network-average RMS (nodes with near-zero net balance get the
    average scale, so the normalized residual stays finite)."""
    thr = np.abs(Q @ Ms.T + b)
    rms = np.sqrt(np.mean(thr ** 2, axis=0))
    return np.maximum(rms, rms.mean())


def penalty_loss(f, q, B, Ms, b, scon):
    r = f @ B.T - q @ Ms.T - b
    return ((r / scon) ** 2).mean()


def pretrain(seed, arch):
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    Ms, b = balance_terms(n, ods, qbg)
    z = np.load(p.base.old.DATASET)
    Q, F = z['Q'], z['F']
    n_tr = int(z['split'][0])
    q_mu, q_sd = Q[:n_tr].mean(0), Q[:n_tr].std(0)
    scon = cons_scale(Q[:n_tr], Ms, b)
    # per-link scale identical to the continuation protocol (fscale)
    fscale = np.sqrt(np.mean(F[:n_tr] ** 2, axis=0)) * 0.01
    m, width, n_param = build_direct_model(n, ps, qbg, seed, arch, q_mu, q_sd,
                                           f_mean=F[:n_tr].mean(0))
    m.set_balance(Ms, b)
    mstd = sum(v.numel() for v in m.parameters())
    # reference parameter count of the exact architecture
    me = p.base.old.build_fullnet_model(ps, qbg, seed)
    n_exact = sum(v.numel() for v in me.parameters())
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
    Bt = torch.tensor(n.B)
    Mst = torch.tensor(Ms)
    bt = torch.tensor(b)
    sct = torch.tensor(scon)

    def val_error():
        m.eval()
        with torch.no_grad():
            f = m(Xv)
            rel = (f - Fv).norm(dim=1) / Fv.norm(dim=1)
            res = (f @ Bt.T - Xv @ Mst.T - bt).abs().max().item()
            return float(rel.mean()), res

    best_val, best_state, best_step = float('inf'), None, -1
    step = 0
    while step < PRE_STEPS:
        for xb, yb in dl:
            opt.zero_grad()
            pred = m(xb)
            loss = (((pred - yb) / fst) ** 2).mean()
            if arch == 'penalty':
                loss = loss + PENALTY_W * penalty_loss(pred, xb, Bt, Mst, bt, sct)
            loss.backward()
            opt.step()
            step += 1
            if step % PRE_EVERY == 0:
                ve, res = val_error()
                if ve < best_val:
                    best_val, best_step = ve, step
                    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                if step % 2000 == 0:
                    print(f'pre {arch} {seed} step {step} loss {float(loss):.3e} '
                          f'val {ve:.4%} resid {res:.2e} ({time.time()-st:.0f}s)', flush=True)
            if not np.isfinite(float(loss)):
                raise RuntimeError(f'pretrain diverged: {arch} seed {seed} step {step}')
            if step >= PRE_STEPS:
                break
    torch.save({'best': best_state, 'best_val': best_val, 'best_step': best_step,
                'norm': {'mu_q': q_mu, 'sd_q': q_sd}, 'width': width,
                'n_params': n_param, 'n_params_exact': n_exact},
               OUT / f'pre_{arch}_{seed}.pt')
    (OUT / f'pre_{arch}_{seed}_cost.json').write_text(json.dumps(
        {'seconds': time.time() - st, 'best_val': best_val, 'best_step': best_step,
         'val_residual_final': res, 'width': width, 'n_params': n_param,
         'n_params_exact': n_exact, 'params_direct_actual': mstd}, indent=2))
    print(f'pre {arch} {seed}: best val {best_val:.4%} at {best_step}', flush=True)


def cont(seed, arch, method):
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    Ms, b = balance_terms(n, ods, qbg)
    z = np.load(p.base.old.DATASET)
    lab = np.load(E111 / 'labels.npz')
    ck = torch.load(OUT / f'pre_{arch}_{seed}.pt', weights_only=False)
    m, width, n_param = build_direct_model(n, ps, qbg, seed, arch,
                                           ck['norm']['mu_q'], ck['norm']['sd_q'],
                                           width=ck['width'])
    m.set_balance(Ms, b)
    m.load_state_dict(ck['best'])
    initial = {k: v.detach().clone() for k, v in m.state_dict().items()}
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
    Bt = torch.tensor(n.B)
    Mst = torch.tensor(Ms)
    bt = torch.tensor(b)
    sct = torch.tensor(cons_scale(z['Q'][:2500], Ms, b))
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
                values.extend((p.kt(err[lab['j'][vi] == j], Bt, s, t) / ks[j]).square().tolist())
            flow = torch.cat([fa - torch.tensor(lab['F0'][vi]), fb - torch.tensor(lab['F1'][vi])])
            ref = torch.tensor(np.concatenate([lab['F0'][vi], lab['F1'][vi]]))
            fr = float((torch.linalg.vector_norm(flow, dim=1) / torch.linalg.vector_norm(ref, dim=1)).mean())
            res = float((torch.cat([fa, fb]) @ Bt.T
                         - torch.cat([a[vi], bq[vi]]) @ Mst.T - bt).abs().max())
        return {'K_mse': float(np.mean(values)), 'flow_relative': fr, 'cons_residual': res}

    original = validate()
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
        if arch == 'penalty':
            loss = loss + PENALTY_W * penalty_loss(m(x[ix]), x[ix], Bt, Mst, bt, sct)
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
    torch.save({'best': best, 'selected_step': beststep, 'initial': initial},
               OUT / f'{arch}_{method}_{seed}.pt')
    (OUT / f'{arch}_{method}_{seed}_cost.json').write_text(json.dumps(
        {'seconds': time.time() - st, 'selected_step': beststep,
         'original_validation': original, 'logs': logs}, indent=2))
    print(f'cont {arch} {method} {seed}: step {beststep} '
          f'({time.time()-st:.0f}s)', flush=True)


def evaluate():
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    Ms, b = balance_terms(n, ods, qbg)
    chains = direction_levels(net=n)
    fams = [query_families(s, t, chains=chains, net=n) for s, t in ods]
    geo = []
    for fam_j in fams:
        geo.append([np.array([[v in U for v in range(n.n)] for U in f['masks']]) @ B
                    for f in fam_j])
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
    N = len(Q)
    mu_norm = {j: np.linalg.norm(Ms[:, j]) for j in range(6)}  # = sqrt(2)

    selection = {(s['method'], s['seed']): s['step']
                 for s in json.loads((OLD / 'results.json').read_text())['selection']}
    rows = []
    gates = {}
    started = time.perf_counter()
    for arch in ARCHS:
        for method in METHODS:
            for seed in range(5):
                if arch == 'exact':
                    m = p.base.old.build_fullnet_model(ps, qbg, seed)
                    step = selection[(method, seed)]
                    ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else \
                        E111 / f'{method}_{seed}.pt'
                    m.load_state_dict(torch.load(ck, weights_only=False)['best'])
                else:
                    ck = torch.load(OUT / f'{arch}_{method}_{seed}.pt', weights_only=False)
                    pre = torch.load(OUT / f'pre_{arch}_{seed}.pt', weights_only=False)
                    m, _, _ = build_direct_model(n, ps, qbg, seed, arch,
                                                 pre['norm']['mu_q'], pre['norm']['sd_q'],
                                                 width=pre['width'])
                    m.set_balance(Ms, b)
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
                            'arch': arch, 'method': method, 'seed': seed,
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
                print('eval', arch, method, seed, f'{time.perf_counter()-started:.0f}s',
                      flush=True)
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))
    # gates: exact rows must reproduce published E119 values
    ex = [r for r in rows if r['arch'] == 'exact']
    pub = json.loads((E119 / 'results.json').read_text())['overall']
    for method in METHODS:
        got = float(np.median([
            np.mean([r['column_pct'] for r in ex
                     if r['method'] == method and r['path'] == pi])
            for pi in range(N)]))
        gates[f'exact_{method}_column_pct'] = {'published': round(pub[method]['column_pct'], 4),
                                                'recomputed': round(got, 4)}
        assert abs(got - pub[method]['column_pct']) < 5e-3, (method, got)
    gates['projection_conservation_max'] = float(max(
        r['cons_residual_max'] for r in rows if r['arch'] == 'projection'))
    gates['exact_conservation_max'] = float(max(
        r['cons_residual_max'] for r in rows if r['arch'] == 'exact'))
    # aggregate: path unit (mean active queries and seeds within path), medians
    METRICS = ('flow_pct', 'increment_pct', 'cons_residual_max',
               'derivative_residual_max', 'column_pct', 'system_pct',
               'E_model', 'E_summary', 'E_total')
    summary = {}
    for arch in ARCHS:
        for method in METHODS:
            rr = [r for r in rows if r['arch'] == arch and r['method'] == method]
            per_path = {}
            for r in rr:
                per_path.setdefault(r['path'], []).append(r)
            ent = {k: float(np.median([np.mean([x[k] for x in v]) for v in per_path.values()]))
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
    (OUT / 'results.json').write_text(json.dumps(
        {'protocol': {'penalty_weight': PENALTY_W,
                      'penalty_normalization': 'nodal throughput RMS over the 2500 training states',
                      'pretrain': ('e93 schedule (Adam 3e-3, batch 256, <=12000 steps, validation every 50, '
                                   'min-validation checkpoint) with per-link scaled MSE (the stated pretraining objective)'),
                      'continuation': 'e111 protocol (1200 updates, lr 3e-5, batches 32/8, shared labels, uniform selection rule)',
                      'width_matching': 'direct-MLP width solves 2W^2+(6+3+76)W+76 = exact parameter count'},
         'gates': gates, 'summary': summary}, indent=2))
    print(json.dumps({'gates': gates, 'summary': {k: {m: round(v[m], 4) if isinstance(v[m], float) else v[m] for m in v} for k, v in summary.items()}}, indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('phase', choices=['pretrain', 'cont', 'eval'])
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--arch', choices=ARCHS[1:], default='penalty')
    a.add_argument('--method', choices=METHODS, default='flow')
    v = a.parse_args()
    if v.phase == 'pretrain':
        pretrain(v.seed, v.arch)
    elif v.phase == 'cont':
        cont(v.seed, v.arch, v.method)
    else:
        evaluate()
