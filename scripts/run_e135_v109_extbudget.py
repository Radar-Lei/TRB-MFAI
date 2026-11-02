"""E135 (v109): extended training budget with validation convergence curves.

External review of v108 (P0-2): every response checkpoint stopped at the
1200-step continuation cap, so the flow-vs-response comparison could be an
artifact of the budget rather than the objective.  This experiment repeats
the frozen finetuning protocol of e111 (Sioux Falls) and e130 (Anaheim)
for the two primary objectives, flow and response, with the continuation
budget extended from 1200 to 3600 updates.  Everything else is unchanged:
same pretrained checkpoints, same seeds, same lr, same batch schedule,
same validation cadence, same per-network selection rule, test paths
untouched.  The per-run random streams start at the same seeds, so the
first 1200 steps reproduce the frozen trajectories; the extension only
adds candidate checkpoints.

Selection rules (frozen per network):
  Sioux Falls -- the v104 rule: minimum validation K-MSE over
    {the initial checkpoint} + {validation states with flow error
    <= 1.1x initial};
  Anaheim -- the e130 rule: minimum validation K-MSE among states with
    validation flow error <= 1.1x initial, fallback minimum flow error.

Outputs per run: the selected state dictionary and a log of validation
flow error and validation boundary error (K-MSE) at every 100 steps --
the convergence curves reported in the paper.

Gate: the Sioux Falls step-100..1200 log entries must reproduce the e111
logs bit-for-bit (same random stream, same environment).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import run_e111_v102_pairs as p

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e135_v109_extbudget'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E130 = HERE / 'results/e130_v108_anaheim_train'
DATA_AN = HERE / 'results/e129_v108_anaheim_data'

STEPS = 3600
LR = 3e-5
BS = 32
BJ = 8
EVERY = 100
METHODS = ('flow', 'response')


# ----------------------------- Sioux Falls -----------------------------

def train_sf(seed, method):
    st = time.time()
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = torch.tensor(n.B)
    t0 = torch.tensor(n.free_flow_time)
    cap = torch.tensor(n.capacity)
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
    js = torch.tensor(lab['jscale'])
    ks = torch.tensor(lab['kscale'])
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
    for step in range(1, STEPS + 1):
        ix = rng.integers(0, 2500, BS)
        j = (step - 1) % 6
        ij = rng.choice(train_ids[j], BJ)
        opt.zero_grad()
        lf = (((m(x[ix]) - f[ix]) / fs) ** 2).mean()
        loss = lf
        lr_l = torch.tensor(0.)
        if method == 'response':
            v = torch.zeros_like(qm[ij])
            v[:, j] = 1
            _, jp = torch.func.jvp(m, (qm[ij],), (v,))
            lr_l = (jp - jm[ij]).square().sum(1).mean() / js[j] ** 2
            loss = loss + lr_l
        loss.backward()
        opt.step()
        if step % EVERY == 0:
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
            print(f'sf {method} seed{seed} step{step} {val} '
                  f'({time.time() - st:.0f}s)', flush=True)
    # frozen v104 selection rule over the extended budget
    elig_logs = [l for l in logs if l['eligible']]
    pool = [{'step': 0, 'validation': original}] + elig_logs
    winner = min(pool, key=lambda l: l['validation']['K_mse'])
    state = initial if winner['step'] == 0 else eligible_state
    assert winner['step'] == 0 or winner['step'] == eligible_step
    tag = f'sf_{method}_{seed}'
    torch.save({'initial': initial, 'selected': state,
                'selected_step': winner['step']}, OUT / f'{tag}.pt')
    (OUT / f'{tag}_cost.json').write_text(json.dumps(
        {'network': 'sioux_falls', 'method': method, 'seed': seed,
         'steps': STEPS, 'seconds': time.time() - st,
         'initial_validation': original, 'selected_step': winner['step'],
         'selected_validation': winner['validation'], 'logs': logs}, indent=2))
    # determinism gate vs the frozen 1200-step trajectories
    old = json.loads((E111 / f'{method}_{seed}_cost.json').read_text())['logs']
    dev = max(abs(l['validation']['K_mse'] - lo['validation']['K_mse'])
              for l, lo in zip(logs[:12], old[:12]))
    print(f'{tag} done selected_step={winner["step"]} prefix_dev={dev:.2e}',
          flush=True)
    assert dev < 1e-12, ('prefix mismatch vs e111', tag, dev)


# ------------------------------- Anaheim -------------------------------

def build_anaheim_model(ps, q_bg, seed, mu_q, sd_q):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    class AnaheimNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers, dd = [], 6
            for _ in range(3):
                layers += [torch.nn.Linear(dd, 128), torch.nn.Tanh()]
                dd = 128
            layers.append(torch.nn.Linear(dd, ps.P))
            self.net = torch.nn.Sequential(*layers)
            self.register_buffer('Delta', torch.tensor(ps.Delta, dtype=torch.float64))
            self.register_buffer('path_od', torch.tensor(ps.path_od, dtype=torch.long))
            self.register_buffer('q_bg', torch.tensor(q_bg, dtype=torch.float64))
            self.register_buffer('mu_q', torch.tensor(np.asarray(mu_q), dtype=torch.float64))
            self.register_buffer('sd_q', torch.tensor(np.asarray(sd_q), dtype=torch.float64))

        def forward(self, q_Q):
            z = self.net((q_Q - self.mu_q) / self.sd_q)
            b = q_Q.shape[0]
            po = self.path_od[None, :].expand(b, -1)
            zmax = torch.full((b, ps.K), -float('inf'), dtype=z.dtype).scatter_reduce_(
                1, po, z, reduce='amax', include_self=True)
            w = torch.exp(z - zmax.gather(1, po))
            den = torch.zeros(b, ps.K, dtype=z.dtype).scatter_add_(1, po, w)
            shares = w / den.gather(1, po)
            q_full = torch.cat([q_Q, self.q_bg[None, :].expand(b, -1)], dim=1)
            h = q_full.gather(1, po) * shares
            return h @ self.Delta.t()

    return AnaheimNet().double()


def train_anaheim(seed, method):
    d = np.load(DATA_AN / 'demands.npz')
    ref = np.load(DATA_AN / 'reference.npz')
    with (DATA_AN / 'setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    B = torch.tensor(net.B, dtype=torch.float64)
    t0 = torch.tensor(net.free_flow_time, dtype=torch.float64)
    cap = torch.tensor(net.capacity, dtype=torch.float64)
    ods6 = ps.ods[:6]
    Q, F = d['QF'], ref['F']
    FC, JC = ref['FC'], ref['JC']
    qbg = d['q'][6:]
    N_TRAIN, N_VAL = 800, 200
    NC_TRAIN = 384
    ck = torch.load(E130 / f'pre_seed{seed}.pt', weights_only=False)
    m = build_anaheim_model(ps, qbg, seed, ck['norm']['mu_q'], ck['norm']['sd_q'])
    m.load_state_dict(ck['best'])
    initial = {k: v.detach().clone() for k, v in m.state_dict().items()}
    fs = torch.tensor(np.sqrt(np.mean(F[:N_TRAIN] ** 2, axis=0)) * .01,
                      dtype=torch.float64)
    jscale = np.zeros(6)
    for j in range(6):
        jscale[j] = max(float(np.sqrt(np.mean(
            np.sum(JC[:NC_TRAIN, :, j] ** 2, axis=1)))), 1e-6)
    kscale = np.zeros(6)
    gC = t0.numpy()[None, :] * (1 + .75 * (FC / cap.numpy()[None, :]) ** 4)
    for j, (s, t) in enumerate(ods6):
        dB = net.B @ (gC[:NC_TRAIN] * JC[:NC_TRAIN][:, :, j]).T
        inter = [v2 for v2 in range(net.n) if v2 not in (s, t)]
        k = .5 * (np.abs(dB[t] - dB[s]) + np.abs(dB[inter]).sum(0))
        kscale[j] = max(float(np.sqrt(np.mean(k ** 2))), 1e-6)
    x = torch.tensor(Q[:N_TRAIN])
    f = torch.tensor(F[:N_TRAIN])
    xv = torch.tensor(Q[N_TRAIN:N_TRAIN + N_VAL])
    fv = torch.tensor(F[N_TRAIN:N_TRAIN + N_VAL])
    qm = torch.tensor(d['QC'])
    jm = torch.tensor(JC)
    js_t, ks_t = torch.tensor(jscale), torch.tensor(kscale)
    train_ids = np.arange(NC_TRAIN)
    vi = np.arange(NC_TRAIN, 480)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = np.random.default_rng(7102 + seed)

    def validate():
        with torch.no_grad():
            fr = float(((m(xv) - fv).norm(dim=1) / fv.norm(dim=1)).mean())
            wS = t0 * (1 + .75 * (torch.tensor(FC[vi]) / cap) ** 4)
            jS = torch.tensor(JC[vi])
            vals = []
            for j, (s, t) in enumerate(ods6):
                qv = qm[vi]
                v = torch.zeros_like(qv)
                v[:, j] = 1
                pr, jv = torch.func.jvp(m, (qv,), (v,))
                wL = t0 * (1 + .75 * (pr / cap) ** 4)
                disc = wL * jv - wS * jS[:, :, j]
                dBv = torch.einsum('ve,be->vb', B, disc)
                inter = [v2 for v2 in range(net.n) if v2 not in (s, t)]
                k = .5 * (torch.abs(dBv[t] - dBv[s]) + torch.abs(dBv[inter]).sum(0))
                vals.extend((k / ks_t[j]).square().tolist())
        return {'K_mse': float(np.mean(vals)), 'flow_relative': fr}

    original = validate()
    logs = []
    eligible_best = float('inf')
    eligible_state = None
    eligible_step = 0
    fallback = float('inf')
    fallback_state = None
    fallback_step = 0
    st = time.time()
    for step in range(1, STEPS + 1):
        ix = rng.integers(0, N_TRAIN, BS)
        j = (step - 1) % 6
        ij = rng.choice(train_ids, BJ)
        opt.zero_grad()
        lf = (((m(x[ix]) - f[ix]) / fs) ** 2).mean()
        loss = lf
        lr_l = torch.tensor(0.)
        if method == 'response':
            v = torch.zeros_like(qm[ij])
            v[:, j] = 1
            _, jp = torch.func.jvp(m, (qm[ij],), (v,))
            lr_l = (jp - jm[ij][:, :, j]).square().sum(1).mean() / js_t[j] ** 2
            loss = loss + lr_l
        loss.backward()
        opt.step()
        if step % EVERY == 0:
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
                         'aux_loss': float(lr_l.detach()), 'validation': val,
                         'eligible': bool(eligible),
                         'elapsed_seconds': time.time() - st})
            print(f'anaheim {method} seed{seed} step{step} {val} '
                  f'({time.time() - st:.0f}s)', flush=True)
    # frozen e130 rule over the extended budget
    if eligible_state is not None:
        state, sel_step, sel_val = eligible_state, eligible_step, \
            {'K_mse': eligible_best}
    else:
        state, sel_step = (fallback_state if fallback_state is not None
                           else initial), (fallback_step
                           if fallback_state is not None else 0)
        sel_val = None
    tag = f'an_{method}_{seed}'
    torch.save({'initial': initial, 'selected': state,
                'selected_step': sel_step}, OUT / f'{tag}.pt')
    (OUT / f'{tag}_cost.json').write_text(json.dumps(
        {'network': 'anaheim', 'method': method, 'seed': seed, 'steps': STEPS,
         'seconds': time.time() - st, 'initial_validation': original,
         'selected_step': sel_step,
         'selected_validation': sel_val, 'logs': logs}, indent=2))
    print(f'{tag} done selected_step={sel_step}', flush=True)


def eval_sf():
    """96-path test evaluation of the 3600-update Sioux Falls selections."""
    from run_e83_v82_validation import attribution
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    masks = p.screenline_masks(ods)
    geo = [masks[j] @ n.B for j in range(6)]
    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(HERE / 'results/e119_v106_expanded_paths/paths_new.npz')
    Cnew = np.load(HERE / 'results/e119_v106_expanded_paths/reference_columns.npz')['C']
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
    for method in METHODS:
        for seed in range(5):
            ck = torch.load(OUT / f'sf_{method}_{seed}.pt', weights_only=False)
            m = p.base.old.build_fullnet_model(ps, qbg, seed)
            m.load_state_dict(ck['initial'] if ck['selected_step'] == 0
                              else ck['selected'])
            for pi in range(N):
                q = torch.tensor(Q[pi])
                f = m(q).detach().numpy()
                js = []
                for j in range(6):
                    v = torch.zeros_like(q)
                    v[:, j] = 1
                    _, jv = torch.func.jvp(m, (q,), (v,))
                    js.append(jv.detach().numpy())
                jac = np.stack(js, axis=2)
                C = attribution(np.array([T.grad(f0) for f0 in f]), jac, ts,
                                Q[pi, -1] - Q[pi, 0])
                flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                                     / np.linalg.norm(Fref[pi], axis=1))
                for j, (s, t) in enumerate(ods):
                    if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                        continue
                    e = C[:, j] - Cref[pi][:, j]
                    r = Cref[pi][:, j]
                    rows.append({
                        'method': method, 'seed': seed, 'path': pi,
                        'query': j, 'group': int(groups[pi]),
                        'flow_pct': float(flow),
                        'column_pct': float(100 * np.linalg.norm(e)
                                            / max(np.linalg.norm(r), 1e-6)),
                        'system_pct': float(100 * abs(e.sum())
                                            / max(abs(r.sum()), 1e-6)),
                        'geometry_pct': float(100 * np.max(np.abs(geo[j] @ e))
                                              / max(np.max(np.abs(geo[j] @ r)), 1e-6)),
                        'boundary_pct': float(100 * p.kn(e, n.B, s, t)
                                              / max(p.kn(r, n.B, s, t), 1e-6)),
                    })
            print('evaluated', method, seed, flush=True)
    (OUT / 'sf_rows.json').write_text(json.dumps(rows, indent=2))
    vals = {}
    for r in rows:
        vals.setdefault((r['method'], r['path']), []).append(r)
    per_path = {k: {m2: float(np.mean([x[m2] for x in v]))
                    for m2 in ('flow_pct', 'column_pct', 'system_pct',
                               'geometry_pct', 'boundary_pct')}
                for k, v in vals.items()}
    summary = {}
    for method in METHODS:
        summary[method] = {m2: float(np.median([v[m2] for (me, _), v
                                                in per_path.items() if me == method]))
                           for m2 in ('flow_pct', 'column_pct', 'system_pct',
                                      'geometry_pct', 'boundary_pct')}
    (OUT / 'sf_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def eval_anaheim():
    """48-path test evaluation of the 3600-update Anaheim selections,
    including the total-error budget quantities of e131."""
    from run_e83_v82_validation import attribution, TSTTT
    torch.set_num_threads(1)
    d = np.load(DATA_AN / 'demands.npz')
    ref = np.load(DATA_AN / 'reference.npz')
    with (DATA_AN / 'setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    fams = json.loads((DATA_AN / 'families.json').read_text())
    Q, dQP, group, ts = d['QP'], d['dQP'], d['group'], d['ts']
    Cref = ref['Cref']
    Fp = [None] * 48
    for f2 in sorted(DATA_AN.glob('paths_*.npz')):
        zz = np.load(f2)
        for fr, i in zip(zz['F'], zz['pids']):
            Fp[int(i)] = fr
    assert all(x is not None for x in Fp)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    B = net.B
    geo = []
    for f2 in fams:
        R = np.array([[1.0 if v in U else 0.0 for v in range(net.n)]
                      for U in f2['masks']])
        geo.append(R @ B)

    def greedy_segments(a, eps):
        segs = []
        i = 0
        nn = len(a)
        while i < nn:
            j2 = i
            lo = hi = a[i]
            while j2 + 1 < nn:
                nlo, nhi = min(lo, a[j2 + 1]), max(hi, a[j2 + 1])
                if nhi - nlo > 2 * eps or (nlo < 0 < nhi):
                    break
                lo, hi = nlo, nhi
                j2 += 1
            segs.append((i, j2))
            i = j2 + 1
        return segs

    def summarize_seg(a, segs):
        o = np.empty_like(a)
        for i, j2 in segs:
            s = a[i:j2 + 1]
            o[i:j2 + 1] = (s.max() + s.min()) / 2
        return o

    rows = []
    for method in METHODS:
        for seed in range(5):
            ck = torch.load(OUT / f'an_{method}_{seed}.pt', weights_only=False)
            pre = torch.load(E130 / f'pre_seed{seed}.pt', weights_only=False)
            m = build_anaheim_model(ps, d['q'][6:], seed,
                                    pre['norm']['mu_q'], pre['norm']['sd_q'])
            m.load_state_dict(ck['initial'] if ck['selected_step'] == 0
                              else ck['selected'])
            for pi in range(48):
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
                    aL = -(geo[j] @ C[:, j])
                    aS = -(geo[j] @ r)
                    E_model = float(np.max(np.abs(aL - aS)))
                    epsL = .05 * max(np.max(np.abs(aL)), 1e-12)
                    segs = greedy_segments(aL, epsL)
                    hat = summarize_seg(aL, segs)
                    rows.append({
                        'method': method, 'seed': seed, 'path': pi, 'query': j,
                        'group': int(group[pi]), 'flow_pct': float(flow),
                        'column_pct': float(100 * np.linalg.norm(e)
                                            / max(np.linalg.norm(r), 1e-6)),
                        'system_pct': float(100 * abs(e.sum())
                                            / max(abs(r.sum()), 1e-6)),
                        'geometry_pct': float(100 * np.max(np.abs(geo[j] @ e))
                                              / max(np.max(np.abs(geo[j] @ r)), 1e-6)),
                        'E_model': E_model,
                        'E_summary': float(np.max(np.abs(hat - aL))),
                        'E_total': float(np.max(np.abs(hat - aS))),
                        'segments': len(segs),
                    })
            print('evaluated', method, seed, flush=True)
    (OUT / 'an_rows.json').write_text(json.dumps(rows, indent=2))
    vals = {}
    for r in rows:
        vals.setdefault((r['method'], r['path']), []).append(r)
    keys = ('flow_pct', 'column_pct', 'system_pct', 'geometry_pct',
            'E_model', 'E_summary', 'E_total', 'segments')
    per_path = {k: {m2: float(np.mean([x[m2] for x in v])) for m2 in keys}
                for k, v in vals.items()}
    summary = {}
    for method in METHODS:
        summary[method] = {m2: float(np.median([v[m2] for (me, _), v
                                                in per_path.items() if me == method]))
                           for m2 in keys}
    (OUT / 'an_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval'], nargs='?', default='train')
    ap.add_argument('network', choices=['sf', 'anaheim'])
    ap.add_argument('seed', type=int, nargs='?')
    ap.add_argument('method', choices=METHODS, nargs='?')
    a = ap.parse_args()
    if a.cmd == 'train':
        if a.network == 'sf':
            train_sf(a.seed, a.method)
        else:
            train_anaheim(a.seed, a.method)
    else:
        eval_sf() if a.network == 'sf' else eval_anaheim()
