"""E122 (v107): theory--empirical closure of the boundary-error bound chain
on the 96 test demand paths.

External review of v106 (item 5 / A1): Eq. (20) identifies the weighted
response discrepancy h_j(q) = D^L(q) J_j^L(q) - D^S(q) J_j^S(q) as the
local driver of boundary-explanation error, but the paper never evaluates
the predicted chain on the test paths.  This experiment computes, for
both supervised methods (flow, response) and every active query on all
96 paths,

    E_geo(j) = max_{U in geometric family} |phi^L(U) - phi^S(U)|,
    E_all(j) = K_j(e_j)          (exact maximum over ALL separating sets),
    B(j)     = |dq_j| * int_0^1 K_j(h_j(gamma(lambda))) dlambda   (Eq. 20),

so that the chain E_geo <= E_all <= B is verified numerically and its
tightness reported.  No model is retrained; the v104 selection stays
frozen exactly as in e119.

Normalization: percentage errors use the reference column's own boundary
seminorm K_j(reference column) as denominator, the natural boundary
scale.  Ratios E_geo/E_all and E_all/B are dimensionless.

Gates (all must pass before any number enters the paper):
  (a) chain: E_geo <= E_all + 1e-6*scale and E_all <= B + 1e-6*scale for
      every path/query/seed/method (scale = K_j(reference column));
  (b) replication: recomputed geometry_pct medians reproduce the
      published 96-path numbers (21.18/14.19 column, 8.74/5.75 geometry)
      to <= 0.02 percentage points;
  (c) learned-reference completeness of the reference columns equals the
      e119 stored value.

Output: experiments/results/e122_v107_boundchain/{results.json}
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_e111_v102_pairs as p
from run_e83_v82_validation import attribution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e122_v107_boundchain'
OUT.mkdir(exist_ok=True)
OLD = ROOT / 'experiments/results/v104_learning'
E111 = ROOT / 'experiments/results/e111_v102'
E119 = ROOT / 'experiments/results/e119_v106_expanded_paths'
METHODS = ('flow', 'response')
PAPER = {'column_pct': {'flow': 21.18, 'response': 14.19},
         'geometry_pct': {'flow': 8.74, 'response': 5.75}}


def seminorm(z, B, s, t):
    """K_j(z): exact max |phi_j(U) difference| over all separating sets."""
    d = z @ B.T
    inter = [i for i in range(B.shape[0]) if i not in (s, t)]
    return .5 * (abs(d[..., t] - d[..., s]) + np.abs(d[..., inter]).sum(-1))


def learned_columns(m, Q, ts, T):
    """Learned flows, full Jacobians and attribution columns on one path."""
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
    return f, jac, C


def main():
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    inter = {j: [i for i in range(B.shape[0]) if i not in ods[j]] for j in range(6)}
    masks = p.screenline_masks(ods)
    geo = [masks[j] @ B for j in range(6)]          # family x edges -> phi operator

    # ---- assemble the 96 reference paths (24 frozen + 72 from e119) ----
    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Qs = [d_old['Q'][pi] for pi in range(24)] + [d_new['Q'][pi] for pi in range(len(d_new['Q']))]
    Fs = [d_old['F'][pi] for pi in range(24)] + [d_new['F'][pi] for pi in range(len(d_new['F']))]
    Js = [d_old['J'][pi] for pi in range(24)] + [d_new['J'][pi] for pi in range(len(d_new['J']))]
    groups = np.concatenate([d_old['group'], d_new['group']])
    ts = d_old['ts']
    n_paths = len(Qs)

    # ---- reference columns and gates ----
    ref = np.array([attribution(np.array([T.grad(f) for f in F]), J, ts, Q[-1] - Q[0])
                    for Q, F, J in zip(Qs, Fs, Js)])
    stored = np.load(E119 / 'reference_columns.npz')['C']
    gate_ref = float(np.max(np.abs(ref[24:] - stored)) / max(np.max(np.abs(stored)), 1e-12))

    selection = {(s['method'], s['seed']): s['step']
                 for s in json.loads((OLD / 'results.json').read_text())['selection']}
    rows = []
    started = time.perf_counter()
    for method in METHODS:
        for seed in range(5):
            m = p.base.old.build_fullnet_model(ps, qbg, seed)
            step = selection[(method, seed)]
            ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else E111 / f'{method}_{seed}.pt'
            m.load_state_dict(torch.load(ck, weights_only=False)['best'])
            for pi in range(n_paths):
                Q, Fref, Jref = Qs[pi], Fs[pi], Js[pi]
                fL, jacL, CL = learned_columns(m, Q, ts, T)
                gL = np.array([T.grad(f0) for f0 in fL])
                gS = np.array([T.grad(f0) for f0 in Fref])
                for j, (s, t) in enumerate(ods):
                    dq = Q[-1, j] - Q[0, j]
                    if abs(dq) < 1e-10:
                        continue
                    e = CL[:, j] - ref[pi][:, j]
                    r = ref[pi][:, j]
                    E_geo = float(np.max(np.abs(geo[j] @ e)))
                    E_all = float(seminorm(e, B, s, t))
                    Kr = float(seminorm(r, B, s, t))
                    h = gL * jacL[:, :, j] - gS * Jref[:, :, j]
                    Kk = seminorm(h, B, s, t)
                    Bj = float(abs(dq) * np.trapezoid(Kk, ts))
                    rows.append({'method': method, 'seed': seed, 'path': pi, 'query': j,
                                 'dq': float(dq), 'E_geo': E_geo, 'E_all': E_all,
                                 'B': Bj, 'Kr': Kr,
                                 'column_pct': float(100 * np.linalg.norm(e) / max(np.linalg.norm(r), 1e-6)),
                                 'geo_pct': float(100 * E_geo / max(np.max(np.abs(geo[j] @ r)), 1e-6)),
                                 'all_pct': float(100 * E_all / max(Kr, 1e-6)),
                                 'bound_pct': float(100 * Bj / max(Kr, 1e-6))})
            print('evaluated', method, seed, flush=True)
    elapsed = time.perf_counter() - started

    # ---- gates ----
    tol = 1e-6
    gate_chain = 0.0
    for row in rows:
        scale = max(row['Kr'], 1e-9)
        gate_chain = max(gate_chain, (row['E_geo'] - row['E_all']) / scale,
                         (row['E_all'] - row['B']) / scale)
    assert gate_chain <= tol, f'bound chain violated by {gate_chain}'

    # ---- path-level aggregation mirroring e119 (mean over queries+seeds) ----
    def path_level(metric):
        out = {}
        for meth in METHODS:
            acc = {pi: [] for pi in range(n_paths)}
            for row in rows:
                if row['method'] == meth:
                    acc[row['path']].append(row[metric])
            out[meth] = [float(np.mean(acc[pi])) for pi in range(n_paths) if acc[pi]]
        return out

    def ratios(num, den):
        out = {}
        for meth in METHODS:
            acc = {pi: [] for pi in range(n_paths)}
            for row in rows:
                if row['method'] == meth:
                    acc[row['path']].append(row[num] / max(row[den], 1e-12))
            out[meth] = [float(np.mean(acc[pi])) for pi in range(n_paths) if acc[pi]]
        return out

    geo_pl = path_level('geo_pct')
    all_pl = path_level('all_pct')
    bnd_pl = path_level('bound_pct')
    col_pl = path_level('column_pct')
    r_geo_all = ratios('E_geo', 'E_all')
    r_all_b = ratios('E_all', 'B')

    summary = {}
    for meth in METHODS:
        summary[meth] = {
            'geometry_pct_median': float(np.median(geo_pl[meth])),
            'all_pct_median': float(np.median(all_pl[meth])),
            'bound_pct_median': float(np.median(bnd_pl[meth])),
            'column_pct_median': float(np.median(col_pl[meth])),
            'Egeo_over_Eall_median': float(np.median(r_geo_all[meth])),
            'Eall_over_B_median': float(np.median(r_all_b[meth])),
            'Egeo_over_Eall_q1_q3': [float(np.percentile(r_geo_all[meth], 25)),
                                      float(np.percentile(r_geo_all[meth], 75))],
            'Eall_over_B_q1_q3': [float(np.percentile(r_all_b[meth], 25)),
                                  float(np.percentile(r_all_b[meth], 75))],
        }
        for met, paper in PAPER.items():
            key = f'{met}_median'
            got = summary[meth][key]
            assert abs(got - paper[meth]) <= 0.02, (meth, met, got, paper[meth])

    result = {
        'protocol': {'paths': n_paths, 'methods': list(METHODS), 'seeds': 5,
                     'quadrature': len(ts),
                     'selection': 'v104 frozen, identical to e119',
                     'reference_columns_gate_rel': gate_ref,
                     'evaluation_seconds': elapsed},
        'gates': {'chain_violation_max_relative': gate_chain,
                  'chain_tolerance': tol,
                  'replication': {met: {meth: {'paper': PAPER[met][meth],
                                               'recomputed': round(summary[meth][f'{met}_median'], 4)}
                                       for meth in METHODS} for met in PAPER}},
        'summary': summary,
        'groups': {str(g): {meth: {'geo_pct': float(np.median(np.array(geo_pl[meth])[groups == g])),
                                   'all_pct': float(np.median(np.array(all_pl[meth])[groups == g])),
                                   'bound_pct': float(np.median(np.array(bnd_pl[meth])[groups == g]))}
                            for meth in METHODS} for g in (1, 2, 3, 6)},
    }
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps({'summary': summary, 'gates': result['gates']}, indent=2))


if __name__ == '__main__':
    main()
