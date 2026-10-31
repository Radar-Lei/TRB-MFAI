"""E93 (v89, external review round 61, 87/100, item II-2 "the one
experiment most worth adding"): full-background six-query conservative
learned surrogate, validated against direct SUE on the six-query
setting-C protocol.

Review context.  The v88 paper validates its conservative learned
surrogate only in the zero-background setting (eight queries around
node 10), while its strongest full-background sign-flip evidence (the
six pre-specified queries away from node 10) comes from direct SUE.
This experiment fills the gap: a surrogate whose input is the
six-dimensional query demand q_Q and whose path-share head covers ALL
528 ODs, so background OD demands stay fixed while their route shares
respond to q_Q (the reviewer's "six-dim input does not mean only six
ODs can reroute").  A surrogate that froze background path flows would
model the frozen-background setting B, not C; the head here spans the
full 5,280-path candidate set.

Model (extends the paper's M3 construction to the full network):
  h_k(q_Q) = q_k(q_Q) * softmax(z_k(q_Q))|Pi_k,  k = 1..528,
  f(q_Q) = Delta h(q_Q),
with z a 3x128 tanh MLP on standardized q_Q and a linear head with
5,280 logits; q_k(q_Q) is the input demand for the six query ODs and
the published demand for the other 522.  By construction B f = M_full
q_full exactly (learned conservation gate), with no background
injection term b.

Training data follow the paper's B.3 sampling principle: q_k = q_base,k
* s_k with s_k iid LogNormal(0, 0.5^2) clipped to [0.2, 2]; the other
522 ODs stay at published demand; targets are full-network fixed-path
logit-SUE link flows (theta = 1, ten candidate paths per OD, the same
EdgeSUE solver and tolerance as e86).  Split 2500/857/857, dataset
seed 20261025.  Optimization adapts B.3's protocol to the 66x larger
head: Adam lr 3e-3, minibatch 256 (B.3 used full batch on an 80-logit
head), at most 12,000 steps, validation every 50 steps, the
minimum-validation state retained per seed 0..4; the test split is
used only for final reporting.

Evaluation uses EXACTLY the 41 demand states of the published direct
SUE chain (e87 81-point chain, even nodes) and the same integration
protocol (implicit sensitivities for SUE, autodiff for the surrogate,
composite trapezoid, ts uniform).  Reported per query j:
  * paired flow error ||f_L - F_SUE||_2/||F_SUE||_2 at the 41 states;
  * paired Jacobian column error over the six input dims;
  * attribution-column error;
  * boundary-account error E_j^cut = max_{U in family}
    |phi_L(U) - phi_SUE(U)| / |phi_SUE({s_j})| over the SAME connected
    family (e80 exact enumeration), per seed and summarized;
  * the learned connected interval [min_U phi_L, max_U phi_L];
  * sign preservation at the direct-SUE extremal boundaries U_- and
    U_+ (e91 node lists), per seed, with the margin test
    phi_SUE(U_-) < -eps_j^cut and phi_SUE(U_+) > +eps_j^cut, where
    eps_j^cut is the per-seed absolute worst boundary disagreement --
    the condition under which the learned account provably keeps the
    opposite signs at those two boundaries.

Gates (all must pass before any number enters the paper):
  (a) dataset: solver max path fixed-point residual <= 1e-7 over all
      4,214 solves; conservation |B F - M_full q_full| <= 1e-6;
  (b) model: learned conservation |B f_L - M_full q_full| <= 1e-10 at
      the 41 evaluation states (exact by construction, float64);
      per-OD shares sum to q_k within 1e-12;
  (c) reference reproduction: phi_SUE({s_j}) from the frozen e87
      columns matches Table newod (0.05 absolute); connected-family
      counts match e86 n_connected_cuts; e91 U_-/U_+ admissible with
      phi_SUE(U_-) matching e91 (1e-6);
  (d) evaluation: learned column completeness |sum_j C_L_j -
      (T(f_L(q1)) - T(f_L(q0)))| / |dT| reported; E_j^cut singleton
      lower bound E >= |phi_L({s})-phi_SUE({s})|/|phi_SUE({s})| holds;
      unrestricted-family counterpart of E_j^cut computed as a
      cross-check (expected same order);
  (e) training: five seeds trained; best-validation checkpoints
      retained; test split untouched before final reporting.

Claims supported: on this six-query full-background protocol, a
conservative learned surrogate trained only on the six query demands
reproduces the direct-SUE boundary accounts (paired error scale), its
connected intervals straddle zero for the three sign-crossing queries,
and the direct-SUE sign at the extremal boundaries is preserved by the
learned model within the stated margin.  NOT claimed: surrogate
fidelity for arbitrary 528-dimensional demand perturbations, other
networks, or endogenous path sets.

Output: experiments/results/e93_v89_fullbg_surrogate.json
        experiments/results/e93_v89_fullbg_surrogate.npz
        experiments/results/e93_dataset.npz (training data)
        experiments/results/e93_ckpt/seed{0..4}.pt
        console log experiments/results/e93_console.log
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nets import load_sioux_falls  # noqa: E402
from sue import PathStructure  # noqa: E402
from run_e9_transport import TSTTT  # noqa: E402
from run_e83_v82_validation import EdgeSUE, attribution  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e80_connected_cuts import (build_adj_bits, load_c_enum, enum_c,  # noqa: E402
                                    connected, mask_to_nodes)

RESULTS = HERE / 'results'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E86_JSON = RESULTS / 'e86_v84_newod.json'
E91_JSON = RESULTS / 'e91_v87_negative_mechanism.json'
DATASET = RESULTS / 'e93_dataset.npz'
CKPT_DIR = RESULTS / 'e93_ckpt'
OUT = RESULTS / 'e93_v89_fullbg_surrogate.json'
NPZ = RESULTS / 'e93_v89_fullbg_surrogate.npz'

N_TRAIN, N_VAL, N_TEST = 2500, 857, 857
DATA_SEED = 20261025
LR = 3e-3
BATCH = 256
MAX_STEPS = 12000
EVAL_EVERY = 50
SEEDS = (0, 1, 2, 3, 4)
CROSSING = ('1->19', '1->15', '7->12')
GRID = 41


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
def build_dataset():
    """Full-network SUE solves at sampled six-demand states."""
    net = load_sioux_falls()
    study_ods, trips, _ = select_ods(net)
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    ps = PathStructure(net, all_ods)
    solver = EdgeSUE(ps, study_k=6)
    B, M_full = net.B, net.od_M(all_ods)

    q_base = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    rng = np.random.default_rng(DATA_SEED)
    n_total = N_TRAIN + N_VAL + N_TEST
    s = np.exp(rng.normal(0.0, 0.5, size=(n_total, 6)))
    s = np.clip(s, 0.2, 2.0)
    Q = s * q_base[None, :]

    started = time.time()
    Fs = np.empty((n_total, ps.m))
    cons_max = 0.0
    warm = None
    for i in range(n_total):
        q = np.concatenate([Q[i], q_bg])
        F, h, p, res = solver.solve(q, warm)
        warm = F
        Fs[i] = F
        cons_max = max(cons_max, float(np.max(np.abs(B @ F - M_full @ q))))
        if (i + 1) % 250 == 0:
            print(f'  solve {i + 1}/{n_total} res={solver.max_residual:.1e} '
                  f'elapsed={time.time() - started:.0f}s', flush=True)
    assert solver.max_residual <= 1e-7, solver.max_residual
    assert cons_max <= 1e-6, cons_max
    np.savez_compressed(DATASET, Q=Q, F=Fs, q_base=q_base, q_bg=q_bg,
                        split=np.array([N_TRAIN, N_VAL, N_TEST]),
                        data_seed=DATA_SEED)
    print(f'dataset written: {DATASET} (residual {solver.max_residual:.1e}, '
          f'conservation {cons_max:.1e}, {time.time() - started:.0f}s)',
          flush=True)
    return {'n_total': n_total, 'solver_max_residual': solver.max_residual,
            'conservation_max_abs': cons_max,
            'elapsed_seconds': time.time() - started}


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def build_fullnet_model(ps, q_bg, seed, mu_q=None, sd_q=None):
    """Construct the model; mu_q/sd_q default to zeros/ones (identity
    standardization) when not supplied (construction-time placeholder
    replaced by the checkpoint's norm at load)."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = np.zeros(6) if mu_q is None else np.asarray(mu_q)
    sd = np.ones(6) if sd_q is None else np.asarray(sd_q)

    class FullNetSimplex(nn.Module):
        """q_Q (batch, 6) raw demands -> link flows (batch, m).

        Standardized input interface for the logits network; the RAW
        demand multiplies the shares (intrinsic to the conservation
        identity), as in the paper's NormPathSimplexMLP.
        """

        def __init__(self, mu_q, sd_q):
            super().__init__()
            K, P, m = 6, ps.P, ps.m
            layers, d = [], K
            for _ in range(3):
                layers += [nn.Linear(d, 128), nn.Tanh()]
                d = 128
            layers.append(nn.Linear(d, P))
            self.net = nn.Sequential(*layers)
            self.register_buffer('Delta', torch.tensor(ps.Delta,
                                                       dtype=torch.float64))
            self.register_buffer('path_od',
                                 torch.tensor(ps.path_od, dtype=torch.long))
            self.register_buffer('q_bg',
                                 torch.tensor(q_bg, dtype=torch.float64))
            self.register_buffer('mu_q', torch.tensor(np.asarray(mu_q),
                                                      dtype=torch.float64))
            self.register_buffer('sd_q', torch.tensor(np.asarray(sd_q),
                                                      dtype=torch.float64))

        def forward(self, q_Q):                     # (batch, 6) raw demands
            z = self.net((q_Q - self.mu_q) / self.sd_q)   # (batch, P)
            b = q_Q.shape[0]
            po = self.path_od[None, :].expand(b, -1)  # (batch, P)
            zmax = torch.full((b, ps.K), -float('inf'),
                              dtype=z.dtype).scatter_reduce_(
                1, po, z, reduce='amax', include_self=True)
            w = torch.exp(z - zmax.gather(1, po))
            den = torch.zeros(b, ps.K, dtype=z.dtype).scatter_add_(1, po, w)
            shares = w / den.gather(1, po)
            q_full = torch.cat([q_Q, self.q_bg[None, :].expand(
                b, -1)], dim=1)                      # (batch, 528)
            h = q_full.gather(1, po) * shares        # (batch, 5280)
            return h @ self.Delta.t()                # (batch, m)

    return FullNetSimplex(mu, sd).double()


def train_seed(seed):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    z = np.load(DATASET)
    Q, F = z['Q'], z['F']
    n_tr, n_va, _ = z['split']
    q_mu, q_sd = Q[:n_tr].mean(0), Q[:n_tr].std(0)
    net = load_sioux_falls()
    study_ods, trips, _ = select_ods(net)
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    q_bg = np.array([trips[od] for od in bg_ods])
    all_ods = list(study_ods) + bg_ods
    ps = PathStructure(net, all_ods)

    model = build_fullnet_model(ps, q_bg, seed, q_mu, q_sd)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(Q[:n_tr], dtype=torch.float64)          # raw demands
    Ft = torch.tensor(F[:n_tr], dtype=torch.float64)
    Xv = torch.tensor(Q[n_tr:n_tr + n_va], dtype=torch.float64)
    Fv = torch.tensor(F[n_tr:n_tr + n_va], dtype=torch.float64)
    ds = TensorDataset(Xt, Ft)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g)

    def val_error():
        model.eval()
        with torch.no_grad():
            f = model(Xv)
            rel = ((f - Fv).norm(dim=1) / Fv.norm(dim=1))
            return float(rel.mean())

    best_val, best_state, best_step = float('inf'), None, -1
    started = time.time()
    step = 0
    while step < MAX_STEPS:
        for xb, yb in dl:
            model.train()
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                ve = val_error()
                if ve < best_val:
                    best_val, best_step = ve, step
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                if step % 1000 == 0:
                    print(f'  seed {seed} step {step} loss {float(loss):.4e} '
                          f'val {ve:.4%} best {best_val:.4%} '
                          f'({time.time() - started:.0f}s)', flush=True)
            if step >= MAX_STEPS:
                break
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save({'best': best_state, 'best_val': best_val,
                'best_step': best_step,
                'norm': {'mu_q': q_mu, 'sd_q': q_sd}}, CKPT_DIR / f'seed{seed}.pt')
    print(f'seed {seed}: best val {best_val:.4%} at step {best_step} '
          f'-> {CKPT_DIR / f"seed{seed}.pt"}', flush=True)
    return {'seed': seed, 'best_val_rel_l2_mean': best_val,
            'best_step': best_step, 'elapsed_seconds': time.time() - started}


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def evaluate():
    import torch
    torch.set_num_threads(1)
    net = load_sioux_falls()
    B = net.B
    study_ods, trips, _ = select_ods(net)
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    ps = PathStructure(net, all_ods)
    solver = EdgeSUE(ps, study_k=6)
    M_full = net.od_M(all_ods)
    q_bg = np.array([trips[od] for od in bg_ods])
    q_base = np.array([trips[od] for od in study_ods])
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    dQ = q1 - q0
    T = TSTTT(net.free_flow_time, net.capacity, net.m)

    z87 = np.load(E87_NPZ)
    e86 = json.loads(E86_JSON.read_text())
    e91 = json.loads(E91_JSON.read_text())
    ts81 = z87['ts']
    idx41 = np.arange(0, len(ts81), 2)
    assert len(idx41) == GRID
    ts41 = ts81[idx41]
    F41, g41 = z87['F'][idx41], z87['gT'][idx41]
    C_SUE, BC_SUE = z87['C_41'], z87['BC_41']
    J_SUE = z87['J'][idx41]

    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    adj, n = build_adj_bits(net)
    lib = load_c_enum()
    full = (1 << n) - 1

    # gate (c): reference reproduction
    for j, key in enumerate(keys):
        phi_s = -float(BC_SUE[study_ods[j][0], j])
        pub = e86['per_od'][key]['phi_s']
        assert abs(phi_s - pub) < 0.05, (key, phi_s, pub)
    U_masks = {}
    for key in CROSSING:
        m = e91['mechanism'][key]
        j = keys.index(key)
        s, t = study_ods[j]
        for tag in ('U_minus', 'U_plus'):
            nodes = np.array(m[f'{tag}_nodes_1idx']) - 1
            mask = 0
            for v in nodes:
                mask |= 1 << int(v)
            assert connected(adj, mask) and connected(adj, full & ~mask)
            assert (mask >> s) & 1 and not (mask >> t) & 1
            U_masks[(key, tag)] = (mask, nodes)
        phi_m = -float(BC_SUE[U_masks[(key, 'U_minus')][1], j].sum())
        assert abs(phi_m - m['phi_U_minus']) < 1e-6, (key, phi_m)

    Q41 = q0[None, :] + ts41[:, None] * dQ[None, :]
    per_seed = []
    cols_L, fL_all = [], []
    for seed in SEEDS:
        ckpt = torch.load(CKPT_DIR / f'seed{seed}.pt', weights_only=False,
                          map_location='cpu')
        model = build_fullnet_model(ps, q_bg, seed)
        model.load_state_dict(ckpt['best'])   # includes mu_q/sd_q buffers
        model.eval()
        Zt = torch.tensor(Q41, dtype=torch.float64)      # raw demands
        with torch.no_grad():
            fL = model(Zt).numpy()                       # (41, m)
        fL_all.append(fL)

        # flow error (paired, 41 states)
        flow_err = np.linalg.norm(fL - F41, axis=1) / np.linalg.norm(F41, axis=1)

        # learned conservation gate
        cons = max(float(np.max(np.abs(
            B @ fL[i] - M_full @ np.concatenate([Q41[i], q_bg]))))
            for i in range(GRID))
        assert cons <= 1e-10, cons

        # Jacobians (m x 6, links x query demands) and learned TSTT
        # gradients, per evaluation state
        JL = np.empty((GRID, ps.m, 6))
        gL = np.empty((GRID, ps.m))

        def _single(q6):
            return model(q6[None, :])[0]                 # (m,)

        for i in range(GRID):
            x = Zt[i].clone().requires_grad_(True)
            JL[i] = torch.autograd.functional.jacobian(
                _single, x).numpy()                      # (m, 6)
            gL[i] = T.grad(fL[i])
        C_L = attribution(gL, JL, ts41, dQ)              # (m, 6)
        BC_L = B @ C_L
        cols_L.append(C_L)

        # paired Jacobian column error (per query column, e83 convention:
        # norm along the link dimension only)
        jac_err = (np.linalg.norm(JL - J_SUE, axis=1) /
                   np.linalg.norm(J_SUE, axis=1))        # (41, 6)

        # paired attribution-column error
        col_err = (np.linalg.norm(C_L - C_SUE, axis=0) /
                   np.linalg.norm(C_SUE, axis=0))

        # learned completeness
        dT_L = T.value(fL[-1]) - T.value(fL[0])
        compl_L = float(abs(C_L.sum() - dT_L) / abs(dT_L))

        per_od = {}
        for j, key in enumerate(keys):
            s, t = study_ods[j]
            D = (BC_L[:, j] - BC_SUE[:, j])[:, None]     # difference column
            cr = enum_c(lib, adj, n, s, t, D)
            eps_abs = max(abs(float(cr['phi_min'][0])),
                          abs(float(cr['phi_max'][0])))
            phi_s_sue = -float(BC_SUE[s, j])
            E_cut = eps_abs / abs(phi_s_sue)
            # singleton lower bound gate
            E_single = abs(-float(BC_L[s, j]) - phi_s_sue) / abs(phi_s_sue)
            assert E_cut >= E_single - 1e-12, (key, E_cut, E_single)

            crL = enum_c(lib, adj, n, s, t, BC_L[:, j:j + 1])
            lo_L = float(crL['phi_min'][0])
            hi_L = float(crL['phi_max'][0])
            assert int(crL['n_sets']) == int(e86['per_od'][key]['n_connected_cuts'])

            signs = {}
            for tag in ('U_minus', 'U_plus'):
                mask, nodes = U_masks[(key, tag)] if key in CROSSING \
                    else (None, None)
                if nodes is None:
                    signs[tag] = None
                    continue
                phi_L_U = -float(BC_L[nodes, j].sum())
                phi_S_U = -float(BC_SUE[nodes, j].sum())
                signs[tag] = {
                    'phi_learned': phi_L_U, 'phi_sue': phi_S_U,
                    'margin_ok': bool(phi_S_U < -eps_abs if tag == 'U_minus'
                                      else phi_S_U > eps_abs),
                    'sign_kept': bool(np.sign(phi_L_U) == np.sign(phi_S_U))}

            # unrestricted counterpart of E_cut: extrema of the difference
            # column d over the unrestricted family via the closed form
            # (eq:cutextrema applied to d, origin value -d_s)
            d = BC_L[:, j] - BC_SUE[:, j]
            interior = [v for v in range(n) if v not in (s, t)]
            d_int = d[interior]
            min_unr = -d[s] - float(np.maximum(d_int, 0.0).sum())
            max_unr = -d[s] + float(np.maximum(-d_int, 0.0).sum())
            eps_unr = max(abs(min_unr), abs(max_unr))

            per_od[key] = {
                'phi_s_sue': phi_s_sue,
                'phi_s_learned': -float(BC_L[s, j]),
                'E_cut_pct': 100.0 * E_cut,
                'E_cut_abs': eps_abs,
                'E_cut_unrestricted_pct': 100.0 * eps_unr / abs(phi_s_sue),
                'learned_interval': [lo_L, hi_L],
                'n_connected': int(crL['n_sets']),
                'signs': signs,
                'column_err_pct': 100.0 * float(col_err[j]),
            }
        per_seed.append({
            'seed': seed, 'flow_err_mean': float(flow_err.mean()),
            'flow_err_max': float(flow_err.max()),
            'jac_err_median': float(np.median(jac_err)),
            'jac_err_max': float(jac_err.max()),
            'col_err_median': float(np.median(col_err)),
            'col_err_max': float(np.max(col_err)),
            'completeness_learned_relative': compl_L,
            'conservation_max_abs': cons,
            'per_od': per_od,
        })
        print(f'seed {seed}: flow {flow_err.mean():.4%} '
              f'jac_med {np.median(jac_err):.4%} '
              f'E_cut max {max(v["E_cut_pct"] for v in per_od.values()):.3f}%',
              flush=True)

    # test-split flow error (final reporting only)
    z = np.load(DATASET)
    n_tr, n_va, n_te = z['split']
    test_flow = []
    for seed in SEEDS:
        ckpt = torch.load(CKPT_DIR / f'seed{seed}.pt', weights_only=False,
                          map_location='cpu')
        model = build_fullnet_model(ps, q_bg, seed)
        model.load_state_dict(ckpt['best'])
        model.eval()
        Zt = torch.tensor(z['Q'][n_tr + n_va:], dtype=torch.float64)
        with torch.no_grad():
            fL = model(Zt).numpy()
        rel = np.linalg.norm(fL - z['F'][n_tr + n_va:], axis=1) / \
            np.linalg.norm(z['F'][n_tr + n_va:], axis=1)
        test_flow.append(float(rel.mean()))

    out = {
        'protocol': {
            'purpose': 'full-background six-query conservative surrogate vs '
                       'direct SUE on the identical 41-state setting-C chain',
            'model': '3x128 tanh on standardized 6-dim query demand; '
                     'linear head 5280 path logits; h_k = q_k softmax(z_k) '
                     'for all 528 ODs (background demands fixed, shares '
                     'respond); f = Delta h; exact conservation',
            'training': f'LogNormal(0,0.5^2) clip [0.2,2] x q_base per query '
                        f'dim, {N_TRAIN}/{N_VAL}/{N_TEST} split, seed '
                        f'{DATA_SEED}; Adam lr {LR}, batch {BATCH}, '
                        f'{MAX_STEPS} steps, val every {EVAL_EVERY}',
            'evaluation': 'e87 published 41-point chain (81-chain even '
                          'nodes); trapezoid integration; autodiff Jacobians',
            'claims_supported': 'surrogate boundary accounts match direct '
                                'SUE on this protocol; connected intervals '
                                'and extremal-boundary signs; NOT arbitrary '
                                '528-dim demand fidelity',
        },
        'test_flow_error_mean_per_seed': test_flow,
        'test_flow_error_median': float(np.median(test_flow)),
        'per_seed': per_seed,
    }
    OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(NPZ, Q41=Q41, ts41=ts41,
                        C_L=np.stack(cols_L),
                        f_L=np.stack(fL_all))
    print(f'evaluation written -> {OUT}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['data', 'train', 'eval', 'all'])
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    if args.phase in ('data', 'all'):
        if DATASET.exists():
            print('dataset exists, skipping generation', flush=True)
        else:
            build_dataset()
    if args.phase in ('train', 'all'):
        seeds = [args.seed] if args.seed is not None else list(SEEDS)
        for s in seeds:
            if (CKPT_DIR / f'seed{s}.pt').exists():
                print(f'seed {s} checkpoint exists, skipping', flush=True)
            else:
                train_seed(s)
    if args.phase in ('eval', 'all'):
        evaluate()


if __name__ == '__main__':
    main()
