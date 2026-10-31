"""E101 (v93, external review round 65, review priority IV.2):
free-flow vs congestion decomposition of the negative boundary accounts.

Motivation.  The paper explains the sign crossing through entering/leaving
links and the own/background response split (e91).  The review asks one
further question with direct traffic meaning: is the negative boundary
account driven by heterogeneous free-flow times or by the congestion
premium of the marginal social cost?

Method.  For the TSTT outcome with BPR costs,
    g_e'(F) = t0_e + 0.75 t0_e (F/c_e)^4,
so every attribution column splits exactly, at the SAME equilibrium
states and responses as the published columns:
    C~_j = C~_j^ff + C~_j^cong,
with integrands t0_e * J_ej and 0.75 t0_e (F_e(lambda)/c_e)^4 * J_ej.
No re-solving and no new boundary selection: the decomposition is
reported at the published connected minimizing boundaries U_- (e91).

Gates (all must pass before any number enters the paper):
  (a) reproduction: attribution(gT41, J41, ts41, dQ) reproduces the
      published C_41 columns to <= 1e-9 relative;
  (b) exact split: C_ff + C_cong - C_total <= 1e-9 relative, elementwise;
  (c) boundary reproduction: -1_U^T B C_total at each published U_- and
      U_+ matches e91's phi to <= 1e-6 absolute.

Output: experiments/results/e101_v93_ffcong.json
        experiments/results/e101_v93_ffcong.npz   (C_ff, C_cong columns)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from nets import load_sioux_falls  # noqa: E402
from run_e9_transport import TSTTT  # noqa: E402
from run_e83_v82_validation import attribution  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402

RESULTS = HERE / 'results'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E91_JSON = RESULTS / 'e91_v87_negative_mechanism.json'
OUT = RESULTS / 'e101_v93_ffcong.json'
NPZ = RESULTS / 'e101_v93_ffcong.npz'
CROSSING = ('1->19', '1->15', '7->12')


def main():
    net = load_sioux_falls()
    B = net.B
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    study_ods, _, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]

    z = np.load(E87_NPZ)
    ts81 = z['ts']
    idx41 = np.arange(0, len(ts81), 2)
    ts41 = ts81[idx41]
    F41 = z['F'][idx41]
    gT41 = z['gT'][idx41]
    J41 = z['J'][idx41]
    C41 = z['C_41']
    dQ = z['dQ']
    t0 = net.free_flow_time
    cap = net.capacity

    # gate (a): reproduce the published 41-point columns
    C_tot = attribution(gT41, J41, ts41, dQ)
    rep_rel = float(np.max(np.abs(C_tot - C41)) /
                    np.max(np.abs(C41)))
    assert rep_rel <= 1e-9, rep_rel

    # the exact split: integrand t0 + 0.75 t0 (F/c)^4
    g_ff = np.repeat(t0[None, :], len(ts41), axis=0)
    x4 = (F41 / cap[None, :]) ** 4
    g_cong = 0.75 * t0[None, :] * x4
    assert np.max(np.abs(g_ff + g_cong - gT41)) <= 1e-9
    C_ff = attribution(g_ff, J41, ts41, dQ)
    C_cong = attribution(g_cong, J41, ts41, dQ)
    split_rel = float(np.max(np.abs(C_ff + C_cong - C_tot)) /
                      np.max(np.abs(C_tot)))
    assert split_rel <= 1e-9, split_rel

    e91 = json.loads(E91_JSON.read_text())
    per_od = {}
    for key in CROSSING:
        j = keys.index(key)
        s, t = study_ods[j]
        mech = e91['mechanism'][key]
        for tag in ('U_minus', 'U_plus'):
            nodes = np.array(mech[f'{tag}_nodes_1idx']) - 1
            rec = {}
            for name, col in (('ff', C_ff[:, j]), ('cong', C_cong[:, j]),
                              ('total', C_tot[:, j])):
                BC = B @ col
                rec[f'phi_{tag}_{name}'] = -float(BC[nodes].sum())
            # origin-cut and interior decomposition, each part
            BC = B @ C_tot[:, j]
            rec['phi_s_total'] = -float(BC[s])
            rec['interior_total'] = float(
                sum(-BC[v] for v in range(net.n) if v not in (s, t)))
            for name, col in (('ff', C_ff[:, j]), ('cong', C_cong[:, j])):
                BCp = B @ col
                rec[f'phi_s_{name}'] = -float(BCp[s])
                rec[f'interior_{name}'] = float(
                    sum(-BCp[v] for v in range(net.n) if v not in (s, t)))
            # gate (c): reproduce e91's published accounts
            pub = mech[f'phi_{tag}']
            assert abs(rec[f'phi_{tag}_total'] - pub) <= 1e-6, \
                (key, tag, rec[f'phi_{tag}_total'], pub)
            per_od.setdefault(key, {})[tag] = rec

    out = {
        'protocol': {
            'purpose': 'free-flow vs congestion decomposition of the '
                       'boundary accounts at the published U_- and U_+, '
                       'same 41-point setting-C chain, states, and '
                       'responses as the published columns',
            'split': "g'_e = t0_e + 0.75 t0_e (F_e/c_e)^4; C~_j^ff uses "
                     "the t0_e integrand, C~_j^cong the congestion "
                     "premium; both with the same trapezoid rule",
            'no_reselection': 'both components reported at the published '
                              'e91 extremal boundaries; no per-component '
                              'boundary re-selection',
            'claims_supported': 'mechanism attribution of the negative '
                                'accounts between free-flow-time '
                                'heterogeneity and the congestion '
                                'premium; either outcome is reported',
        },
        'gates': {'reproduction_relative': rep_rel,
                  'split_relative': split_rel,
                  'boundary_reproduction_abs': '<=1e-6 vs e91 phi'},
        'per_od': per_od,
    }
    OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(NPZ, C_ff=C_ff, C_cong=C_cong, C_tot=C_tot)
    print(json.dumps({'gates': out['gates'],
                      'per_od': {k: {t: {kk: round(vv, 1) for kk, vv in
                                         r.items() if vv is not None}
                                     for t, r in v.items()}
                                 for k, v in per_od.items()}}, indent=2))
    print(f'-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
