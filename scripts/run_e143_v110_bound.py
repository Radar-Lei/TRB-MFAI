"""E143 (v110): informativeness check of the response-bounded approximate
common-value error bound (external review of v109, optional item in Sec. 4).

For a query j with block-path edge set E_j^path, a scalar reference weight
omega_j(lambda), weight spread eta_j(lambda) = max_{e in E_path}|w_e - omega_j|,
path response magnitude R_j(lambda) = sum_{e in E_path}|J_ej|, and
beta_j^0 = Delta q_j * int omega_j, the proposed bound is

    max_{U in U_j^conn} |phi_j(U) - beta_j^0|
        <= |Delta q_j| * int_0^1 eta_j(lambda) R_j(lambda) dlambda.

This script computes both sides on the Anaheim joint 0.8->1.2 attribution
path (21 frozen states, e112 arrays) for the six study queries, with
omega_j the mean path-edge weight, and compares the bound with the actual
maximum deviation of the reported distance-family accounts from beta_j^0
(v105 profile).  The review's decision rule: include the theorem only if
the bound is informative; this script measures the ratio.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e143_v110_bound'
OUT.mkdir(exist_ok=True)


def main():
    a = np.load(HERE / 'results/e112_v102/arrays.npz')
    F, J, g, ts = a['F'], a['J'], a['g'], a['ts']
    prof = json.loads((HERE / 'results/v105_anaheim_profile/profile.json').read_text())
    d = np.load(HERE / 'results/e129_v108_anaheim_data/demands.npz')
    qb = d['qb']
    with (HERE / 'results/e129_v108_anaheim_data/setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    edges = a['edges']
    dq = 0.4 * qb  # 0.8 -> 1.2 joint path

    rows = []
    for j, prow in enumerate(prof['rows']):
        s, t = int(prow['query'][0]) - 1, int(prow['query'][1]) - 1
        ks = net.od_path_atoms(s, t)
        eq = net.equivalence_classes([(s, t)])
        atoms = eq['atoms']
        ep = sorted({e for k in ks for e in atoms[k]['edges']})
        w = g[:, ep]                          # (21, |E_path|) marginal weights
        omega = w.mean(axis=1)                # scalar reference weight per state
        eta = np.max(np.abs(w - omega[:, None]), axis=1)
        R = np.sum(np.abs(J[:, ep, j]), axis=1)
        bound = abs(dq[j]) * float(np.trapezoid(eta * R, ts))
        beta0 = dq[j] * float(np.trapezoid(omega, ts))
        accounts = np.array(prow['accounts'])
        actual = float(np.max(np.abs(accounts - beta0)))
        rows.append({
            'query': prow['query'], 'path_edges': len(ep),
            'atoms_traversed': len(ks),
            'eta_mean': float(eta.mean()), 'R_mean': float(R.mean()),
            'beta0': beta0, 'bound': bound, 'actual_family_dev': actual,
            'ratio_bound_actual': bound / max(actual, 1e-12),
            'family_range': [prow['family_min'], prow['family_max']],
        })
        print(prow['query'], 'edges', len(ep), 'atoms', len(ks),
              'beta0 %.1f bound %.3g actual %.3g ratio %.3g'
              % (beta0, bound, actual, bound / max(actual, 1e-12)), flush=True)
    (OUT / 'results.json').write_text(json.dumps(
        {'protocol': ('omega_j = mean path-edge TSTT marginal weight per '
                      'state; eta/R/bound per the review formula; actual = '
                      'max deviation of the reported distance-family '
                      'accounts from beta_j^0; 21 frozen states of the '
                      'joint 0.8->1.2 Anaheim path'),
         'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
