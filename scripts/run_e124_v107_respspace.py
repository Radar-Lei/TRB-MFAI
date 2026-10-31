"""E124 (v107): response-space (Level-2) diagnostics for the boundary
transferability hierarchy.

External review of v106 (item 1 / A3): the universal structural condition
quantifies over ALL conservative responses, while an actual fixed-path
assignment model realizes only the affine slice x_j^ref + T_swap with
T_swap = Delta @ ker(A_OD) (Appendix app:membership).  This experiment
moves that hierarchy into numbers:

  Level 1  structural, unrestricted family   eta_all  (weight spread within atoms)
           structural, connected family      eta_conn (atoms crossing the queried
                                                     block paths only)
  Level 2  model-specific response space     B D N_swap = 0  (all swap directions
                                           and  have zero weighted divergence)
                                           B D x_j^ref in span{mu_j}
  Level 3  realized attribution              family account ranges

Networks: the Figure-1 synthetic two-bridges network in its published
post-merge weight state (queried atoms uniform, off-path pendant atom
heterogeneous), and Sioux Falls at load levels 0.7/1.0/1.3, plus Anaheim
at the e112 midpoint state.

Diagnostics (all dimensionless):
  eta_all, eta_conn      absolute weight spread, and normalized by mean weight;
  rho_swap = sigma_max(B D N_swap) / max_j ||B D xhat_j^ref||_2
                         weighted-divergence leakage of a unit rerouting,
                         relative to a unit reference response;
  rho_ref_j = dist(B D x_j^ref, span{mu_j}) / ||B D x_j^ref||;
  Level-3 realized       unrestricted and geometric-family ranges as
                         percentages of the origin-cut account.

Gates:
  (a) membership: J_f e_j - x_j^ref lies in T_swap to <= 1e-6 relative
      (Sioux Falls at level 1.0 for the six study queries; Anaheim at the
      e112 midpoint for its six study queries);
  (b) synthetic ground truth: eta_all = 1 (pendant weights 3,4,5),
      eta_conn = 0, rho_swap = 0, rho_ref = 0 exactly;
  (c) conservation: B x_j^ref = mu_j and B N_swap = 0 to machine precision.

Output: experiments/results/e124_v107_respspace/{results.json}
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

import run_e111_v102_pairs as p
from nets import two_bridges
from sue import PathStructure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e124_v107_respspace'
OUT.mkdir(exist_ok=True)
E87 = ROOT / 'experiments/results/e87_v85_newod_refine.npz'
E112 = ROOT / 'experiments/results/e112_v102'


def swap_basis(ps):
    """Orthonormal basis of T_swap = Delta @ ker(A_OD) as columns.

    The basis is the left singular subspace of Delta @ Nnull; the rank
    threshold is relative to the largest singular value (a plain QR
    without column pivoting misjudges the rank on this matrix).
    """
    A_OD = np.zeros((ps.K, ps.P))
    A_OD[ps.path_od, np.arange(ps.P)] = 1.0
    Nnull = null_space(A_OD)                      # P x (P - K)
    T = ps.Delta @ Nnull                          # m x (P - K)
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    rank = int(np.sum(s > 1e-9 * max(s[0], 1e-30)))
    return U[:, :rank], rank


def eta_values(net, w, ods):
    """eta_all and eta_conn at weight vector w for the queried ODs."""
    eq = net.equivalence_classes(list(ods))
    atoms = eq['atoms']
    eta_all = 0.0
    for atom in atoms:
        e = atom['edges']
        eta_all = max(eta_all, float(np.max(np.abs(w[e] - w[e].mean()))))
    queried = set()
    for (s, t) in ods:
        queried.update(net.od_path_atoms(s, t))
    eta_conn = 0.0
    for k in queried:
        e = atoms[k]['edges']
        eta_conn = max(eta_conn, float(np.max(np.abs(w[e] - w[e].mean()))))
    eq_edges = sorted(e for k in queried for e in atoms[k]['edges'])
    return eta_all, eta_conn, eq, len(eq_edges)


def level2(net, w, ps, ods, B):
    """rho_swap, per-query rho_ref and membership residual."""
    Nswap, rank = swap_basis(ps)
    D = w[:, None]
    BDn = B @ (w[:, None] * Nswap)
    sigma_swap = float(np.linalg.svd(BDn, compute_uv=False)[0]) if BDn.size else 0.0
    rho_ref, sigma_ref, resid = [], 0.0, []
    for j, (s, t) in enumerate(ods):
        p0 = ps.od_path_idx[j][0]
        xref = ps.Delta[:, p0]
        assert np.max(np.abs(B @ xref - (np.eye(net.n)[t] - np.eye(net.n)[s]))) < 1e-9
        bdx = B @ (w * xref)
        mu = np.zeros(net.n)
        mu[t], mu[s] = 1.0, -1.0
        rho_ref.append(float(np.linalg.norm(bdx - (bdx @ mu) * mu / (mu @ mu))
                             / max(np.linalg.norm(bdx), 1e-12)))
        sigma_ref = max(sigma_ref, float(np.linalg.norm(bdx) / np.linalg.norm(xref)))
        if hasattr(ps, '_Jfe'):
            v = ps._Jfe[:, j] - xref
            Pn = Nswap @ (Nswap.T @ v)
            resid.append(float(np.linalg.norm(v - Pn) / max(np.linalg.norm(v), 1e-12)))
    rho_swap = sigma_swap / max(sigma_ref, 1e-12)
    return {'rank_Tswap': rank, 'dim_Tswap_of_m': [rank, net.m],
            'rho_swap': rho_swap, 'rho_ref': rho_ref,
            'membership_residual_max': max(resid) if resid else None}


def synthetic():
    net = two_bridges()
    w = np.ones(net.m)
    w[[11, 12, 13]] = [3.0, 4.0, 5.0]              # pendant atom D, published state
    ods = [(0, 4), (4, 8)]
    ps = PathStructure(net, ods)
    eta_all, eta_conn, eq, nq = eta_values(net, w, ods)
    B = net.B
    l2 = level2(net, w, ps, ods, B)
    # conservation gate on the swap basis
    Nswap = swap_basis(ps)[0]
    assert float(np.max(np.abs(B @ Nswap))) < 1e-9
    return {'network': 'two_bridges (Figure 1, post-merge weights)',
            'nodes': net.n, 'edges': net.m, 'queries': [[s, t] for s, t in ods],
            'paths': ps.P, 'atoms': len(eq['atoms']),
            'queried_atom_edges': nq,
            'eta_all': eta_all, 'eta_conn': eta_conn,
            'level2': {k: l2[k] for k in ('rho_swap', 'rho_ref', 'rank_Tswap')},
            'level3_note': 'Level 2 holds exactly, so every model response has one '
                           'unrestricted common account; the universal condition '
                           'fails (eta_all = 1 on the pendant atom).'}


def sioux_falls():
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    out = {'network': 'Sioux Falls (setting-C protocol path structure)',
           'nodes': n.n, 'edges': n.m, 'queries': [[s + 1, t + 1] for s, t in ods],
           'paths': ps.P, 'levels': {}}
    eq = n.equivalence_classes(list(ods))
    out['atoms'] = len(eq['atoms'])
    out['atom_types'] = [a['type'] for a in eq['atoms']]
    out['queried_atom_edges'] = n.m
    warm = None
    for level in (0.7, 1.0, 1.3):
        f, h, prob, _ = sv.solve(np.r_[level * qb, level * qbg], warm)
        warm = f
        w = T.grad(f)
        eta_all, eta_conn, _, _ = eta_values(n, w, ods)
        ps._Jfe = sv.sensitivity(f, h, prob, np.r_[level * qb, level * qbg])[0]
        l2 = level2(n, w, ps, ods, B)
        out['levels'][str(level)] = {
            'eta_all': eta_all, 'eta_conn': eta_conn,
            'eta_all_normalized': eta_all / float(np.mean(w)),
            'level2': l2}
        print('sioux level', level, 'done', flush=True)
    del ps._Jfe
    # realized Level 3 from the frozen e87 columns
    z = np.load(E87)
    BC = z['BC_41']
    C = z['C_41']
    masks = p.screenline_masks(ods)
    unres, geo = [], []
    for j, (s, t) in enumerate(ods):
        inter = [v for v in range(n.n) if v not in (s, t)]
        phi_s = -BC[s, j]
        unres.append(100 * float(np.abs(BC[inter, j]).sum()) / max(abs(phi_s), 1e-9))
        phi = -(masks[j] @ B) @ C[:, j]
        geo.append(100 * .5 * (phi.max() - phi.min()) / max(abs(phi.min()), abs(phi.max())))
    out['level3_realized'] = {'unrestricted_pct': [float(x) for x in unres],
                              'unrestricted_pct_median': float(np.median(unres)),
                              'geometric_pct_median': float(np.median(geo))}
    return out


def anaheim():
    import run_e112_v102_anaheim as e112
    net, ps, q = e112.setup()
    z = np.load(E112 / 'arrays.npz')
    B, g, C, J = z['B'], z['g'], z['C'], z['J']
    mid = len(z['ts']) // 2
    w = g[mid]
    ods = ps.ods[:6]
    eta_all, eta_conn, eq, nq = eta_values(net, w, ods)
    ps._Jfe = J[mid]
    l2 = level2(net, w, ps, ods, B)
    del ps._Jfe
    unres = []
    for j, (s, t) in enumerate(ods):
        d = B @ C[:, j]
        inter = [v for v in range(net.n) if v not in (s, t)]
        unres.append(100 * float(np.abs(d[inter]).sum()) / max(abs(-d[s]), 1e-9))
    return {'network': 'Anaheim (e112 protocol)', 'nodes': net.n, 'edges': net.m,
            'queries': [[s + 1, t + 1] for s, t in ods], 'paths': ps.P,
            'atoms': len(eq['atoms']), 'queried_atom_edges': nq,
            'eta_all': eta_all, 'eta_conn': eta_conn,
            'eta_all_normalized': eta_all / float(np.mean(w)),
            'level2': l2,
            'level3_realized': {'unrestricted_pct': [float(x) for x in unres],
                                'unrestricted_pct_median': float(np.median(unres))}}


def main():
    started = time.perf_counter()
    synth = synthetic()
    print('synthetic done', flush=True)
    sioux = sioux_falls()
    print('sioux done', flush=True)
    ana = anaheim()
    print('anaheim done', flush=True)

    # gates
    assert abs(synth['eta_all'] - 1.0) < 1e-12 and synth['eta_conn'] == 0.0
    assert synth['level2']['rho_swap'] < 1e-9 and max(synth['level2']['rho_ref']) < 1e-9
    assert sioux['levels']['1.0']['level2']['membership_residual_max'] < 1e-6
    assert ana['level2']['membership_residual_max'] < 1e-6

    result = {'protocol': {'synthetic_weights': 'A,b1,B,b2,C = 1; pendant D = 3,4,5 '
                                                '(connected-example post-merge state)',
                           'sioux_levels': [0.7, 1.0, 1.3],
                           'anaheim_state': 'e112 21-state midpoint',
                           'elapsed_seconds': time.perf_counter() - started},
              'synthetic': synth, 'sioux_falls': sioux, 'anaheim': ana}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('synthetic',)}, indent=2))
    print(json.dumps({'sioux_levels': {k: {'eta_all': v['eta_all'],
                                           'eta_norm': v['eta_all_normalized'],
                                           'rho_swap': v['level2']['rho_swap'],
                                           'rho_ref_max': max(v['level2']['rho_ref'])}
                                       for k, v in sioux['levels'].items()},
                      'sioux_level3': sioux['level3_realized'],
                      'anaheim': {'eta_all': ana['eta_all'],
                                  'eta_conn': ana['eta_conn'],
                                  'rho_swap': ana['level2']['rho_swap'],
                                  'rho_ref_max': max(ana['level2']['rho_ref']),
                                  'atoms': ana['atoms'],
                                  'level3': ana['level3_realized']}}, indent=2))


if __name__ == '__main__':
    main()
