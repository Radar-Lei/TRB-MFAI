"""E130 (v108): Anaheim conservative learned assignment, flow vs response
supervision, five seeds (external review strongly recommended item).

Two stages mirroring the Sioux Falls protocol exactly in structure:
  Stage A (pretrain): flow-only training from scratch on 800 flow states,
    scaled MSE, Adam lr 3e-3, batch 128, <= 4000 steps, best-validation
    checkpoint retained per seed (the analog of the e93 checkpoints).
  Stage B (finetune): continue with the method objective, Adam lr 3e-5,
    batch 32 flow samples + 8 derivative centers per step, 1200 steps,
    validation every 100 steps.  Methods: 'flow' (control) and 'response'
    (adds the unweighted Jacobian loss against reference sensitivity
    columns).  Selection: minimum validation K-MSE (boundary-seminorm of
    the weighted response discrepancy, per-query normalized) among states
    with validation flow error <= 1.1x the stage-A checkpoint; fallback
    minimum flow error.  Test paths never enter training or selection.

The model is the same conservative parameterization as Sioux Falls:
per-OD softmax path shares z_theta(q)|Pi_k multiplied by the raw demand,
flows h @ Delta^T over the 14060 centroid-restricted free-flow paths of
the e112 Anaheim structure (1406 ODs, 914 links), background demands
fixed with endogenous route choice.
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

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e130_v108_anaheim_train'
OUT.mkdir(exist_ok=True)
DATA = HERE / 'results/e129_v108_anaheim_data'

STEPS_A = 4000
STEPS_B = 1200
LR_A = 3e-3
LR_B = 3e-5
BS = 32
BJ = 8
EVERY = 100
N_TRAIN, N_VAL = 800, 200
NC_TRAIN = 384


def load_all():
    d = np.load(DATA / 'demands.npz')
    ref = np.load(DATA / 'reference.npz')
    with (DATA / 'setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    return d, ref, net, ps, q


def build_model(ps, q_bg, seed, mu_q, sd_q):
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



def pretrain(seed):
    d, ref, net, ps, q = load_all()
    Q, F = d['QF'], ref['F']
    mu, sd = Q[:N_TRAIN].mean(0), Q[:N_TRAIN].std(0)
    qbg = d['q'][6:]
    m = build_model(ps, qbg, seed, mu, sd)
    fs = torch.tensor(np.sqrt(np.mean(F[:N_TRAIN] ** 2, axis=0)) * .01, dtype=torch.float64)
    x = torch.tensor(Q[:N_TRAIN])
    f = torch.tensor(F[:N_TRAIN])
    xv = torch.tensor(Q[N_TRAIN:N_TRAIN + N_VAL])
    fv = torch.tensor(F[N_TRAIN:N_TRAIN + N_VAL])
    opt = torch.optim.Adam(m.parameters(), lr=LR_A)
    rng = np.random.default_rng(3000 + seed)
    best_val, best_state, best_step = float('inf'), None, 0
    st = time.time()
    for step in range(1, STEPS_A + 1):
        ix = rng.integers(0, N_TRAIN, 128)
        opt.zero_grad()
        loss = (((m(x[ix]) - f[ix]) / fs) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                fv_pred = m(xv)
                val = float(((fv_pred - fv).norm(dim=1) / fv.norm(dim=1)).mean())
            if val < best_val:
                best_val, best_step = val, step
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            if step % 500 == 0:
                print(f'A seed{seed} step{step} val_flow_rel={val:.5f} '
                      f'({time.time() - st:.0f}s)', flush=True)
        if step % 1000 == 0:
            torch.save({'best': best_state, 'step': best_step, 'val': best_val,
                        'norm': {'mu_q': mu, 'sd_q': sd}}, OUT / f'pre_seed{seed}.pt')
    torch.save({'best': best_state, 'step': best_step, 'val': best_val,
                'norm': {'mu_q': mu, 'sd_q': sd}}, OUT / f'pre_seed{seed}.pt')
    (OUT / f'pre_seed{seed}.json').write_text(json.dumps(
        {'seed': seed, 'best_step': best_step, 'best_val_flow_relative': best_val,
         'seconds': time.time() - st}, indent=2))
    print(f'A seed{seed} done best_step={best_step} val={best_val:.5f}', flush=True)


def finetune(seed, method):
    d, ref, net, ps, q = load_all()
    B = torch.tensor(net.B, dtype=torch.float64)
    t0 = torch.tensor(net.free_flow_time, dtype=torch.float64)
    cap = torch.tensor(net.capacity, dtype=torch.float64)
    ods6 = ps.ods[:6]
    Q, F = d['QF'], ref['F']
    FC, JC = ref['FC'], ref['JC']
    qbg = d['q'][6:]
    ck = torch.load(OUT / f'pre_seed{seed}.pt', weights_only=False)
    m = build_model(ps, qbg, seed, ck['norm']['mu_q'], ck['norm']['sd_q'])
    m.load_state_dict(ck['best'])
    fs = torch.tensor(np.sqrt(np.mean(F[:N_TRAIN] ** 2, axis=0)) * .01, dtype=torch.float64)

    # Derivative centers are generic demand points; the step rotation
    # supervises one study column at a time.  Scales are computed over all
    # training centers for each column.
    jscale = np.zeros(6)
    for j in range(6):
        jscale[j] = max(float(np.sqrt(np.mean(np.sum(JC[:NC_TRAIN, :, j] ** 2, axis=1)))), 1e-6)
    kscale = np.zeros(6)
    gC = t0.numpy()[None, :] * (1 + .75 * (FC / cap.numpy()[None, :]) ** 4)
    for j, (s, t) in enumerate(ods6):
        dB = net.B @ (gC[:NC_TRAIN] * JC[:NC_TRAIN][:, :, j]).T    # (n, n_ix)
        inter = [v for v in range(net.n) if v not in (s, t)]
        k = .5 * (np.abs(dB[t] - dB[s]) + np.abs(dB[inter]).sum(0))
        kscale[j] = max(float(np.sqrt(np.mean(k ** 2))), 1e-6)
    x = torch.tensor(Q[:N_TRAIN])
    f = torch.tensor(F[:N_TRAIN])
    xv = torch.tensor(Q[N_TRAIN:N_TRAIN + N_VAL])
    fv = torch.tensor(F[N_TRAIN:N_TRAIN + N_VAL])
    qm = torch.tensor(d['QC'])
    jm = torch.tensor(JC)
    fs_t, js_t, ks_t = fs, torch.tensor(jscale), torch.tensor(kscale)
    train_ids = np.arange(NC_TRAIN)
    vi = np.arange(NC_TRAIN, 480)

    opt = torch.optim.Adam(m.parameters(), lr=LR_B)
    rng = np.random.default_rng(7102 + seed)

    def validate():
        # K-MSE: boundary seminorm of the weighted response discrepancy
        # h_j = D^L J^L_j - D^S J^S_j, with TSTT marginal weights t(f)
        # evaluated at each model's own flows (reference weights and
        # responses from the frozen e129 centers).
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
    logs, best, best_step = [], None, 0
    eligible_best, fallback = float('inf'), float('inf')
    st = time.time()
    for step in range(1, STEPS_B + 1):
        ix = rng.integers(0, N_TRAIN, BS)
        j = (step - 1) % 6
        ij = rng.choice(train_ids, BJ)
        opt.zero_grad()
        lf = (((m(x[ix]) - f[ix]) / fs_t) ** 2).mean()
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
            take = (eligible and val['K_mse'] < eligible_best) or \
                   (eligible_best == float('inf') and val['flow_relative'] < fallback)
            if take:
                best = {k2: v2.detach().clone() for k2, v2 in m.state_dict().items()}
                best_step = step
                fallback = val['flow_relative']
                if eligible:
                    eligible_best = val['K_mse']
            logs.append({'step': step, 'flow_loss': float(lf.detach()),
                         'response_loss': float(lr_l.detach()), 'validation': val,
                         'elapsed_seconds': time.time() - st})
            print(f'B seed{seed} {method} step{step} {val} ({time.time() - st:.0f}s)',
                  flush=True)
        if step % 300 == 0:
            torch.save({'best': best, 'selected_step': best_step},
                       OUT / f'{method}_{seed}.pt')
    torch.save({'best': best, 'selected_step': best_step}, OUT / f'{method}_{seed}.pt')
    (OUT / f'{method}_{seed}_cost.json').write_text(json.dumps(
        {'seconds': time.time() - st, 'selected_step': best_step,
         'original_validation': original, 'logs': logs}, indent=2))
    print(f'B seed{seed} {method} done selected_step={best_step}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pretrain', 'finetune'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--method', choices=['flow', 'response'])
    a = ap.parse_args()
    if a.cmd == 'pretrain':
        pretrain(a.seed)
    else:
        finetune(a.seed, a.method)
