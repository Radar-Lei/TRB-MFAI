"""E102 (v93, external review round 65, review priority IV.3, optional):
learned-surrogate evaluation of the small single-OD demand bands.

The direct-equilibrium counterparts exist (e100): +-10% and +-20% bands
around the published demand, one moving OD, all 528 ODs jointly
re-equilibrated, endpoint-identity columns.  This script evaluates the
five trained full-background surrogates (e93 checkpoints) on the same
scenarios.  Because one demand moves, the per-edge endpoint identity
(Eq. (14)) applies to each conservative surrogate as well:

    C~^L_{e,j} = g_e(F^{L,1}_e) - g_e(F^{L,0}_e),

so no retraining and no path integration are needed -- two endpoint
forward passes per seed per scenario.

Reported per scenario and seed: learned C_j, learned accounts at the
FIXED direct-equilibrium extremal boundaries (published U_- and U_+),
and errors against the direct endpoint columns and accounts.  Sign
outcomes are reported as they come out -- a sign failure would indicate
surrogate explanation inaccuracy on this scenario, not a failure of the
direct structural phenomenon (which e100 already established).

Gates:
  (a) direct-reference reproduction: recomputing the direct accounts
      from the e100 npz columns reproduces e100's reported C_j and
      connected extrema to <= 1e-6 relative;
  (b) learned conservation <= 1e-10 (exact by construction, checked).

Output: experiments/results/e102_v93_smallband_learned.json
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
from sue import PathStructure  # noqa: E402
from run_e9_transport import TSTTT  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e93_v89_fullbg_surrogate import (build_fullnet_model, CKPT_DIR,
                                          DATASET, SEEDS)

RESULTS = HERE / 'results'
E100_JSON = RESULTS / 'e100_v92_small_demand.json'
E100_NPZ = RESULTS / 'e100_v92_small_demand.npz'
E91_JSON = RESULTS / 'e91_v87_negative_mechanism.json'
OUT = RESULTS / 'e102_v93_smallband_learned.json'

FOCAL = ['1->19', '1->15', '7->12', '6->8']
BANDS = [('0.9->1.1', 0.9, 1.1), ('0.8->1.2', 0.8, 1.2)]


def edge_g(F, t0, cap):
    """g_e(F) = t0 F (1 + 0.15 (F/c)^4), elementwise on the last axis."""
    x = F / cap
    return t0 * F * (1.0 + 0.15 * x ** 4)


def main():
    import torch
    torch.set_num_threads(1)
    net = load_sioux_falls()
    B = net.B
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    t0, cap = net.free_flow_time, net.capacity
    study_ods, trips, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    q_bg = np.array([trips[od] for od in bg_ods])
    all_ods = list(study_ods) + bg_ods
    ps = PathStructure(net, all_ods)
    q_base = np.array([trips[od] for od in study_ods])

    z100 = np.load(E100_NPZ)
    e100 = json.loads(E100_JSON.read_text())
    e91 = json.loads(E91_JSON.read_text())

    # gate (a): reproduce e100's reported accounts from its own columns
    for chain, rec in e100['per_chain'].items():
        od = rec['od'].replace('->', '_')
        dg = z100[f'dg_{od}_{chain.split("|")[1].split("->")[0]}']
        C_ref = float(dg.sum())
        assert abs(C_ref - rec['Cj_dT']) <= 1e-6 * abs(rec['Cj_dT'])
        nodes_m = np.array(rec['connected_min_nodes_1idx']) - 1
        nodes_p = np.array(rec['connected_max_nodes_1idx']) - 1
        assert abs(-float((B @ dg)[nodes_m].sum())
                   - rec['connected_min']) <= 1e-6 * max(1.0, abs(rec['connected_min']))
        assert abs(-float((B @ dg)[nodes_p].sum())
                   - rec['connected_max']) <= 1e-6 * max(1.0, abs(rec['connected_max']))

    # published U_- / U_+ node masks per focal OD (0-based)
    U = {}
    for od in FOCAL:
        rec = e100['per_chain'][[c for c in e100['per_chain']
                                 if c.startswith(od + '|')][0]]
        U[od] = (np.array(rec['connected_min_nodes_1idx']) - 1,
                 np.array(rec['connected_max_nodes_1idx']) - 1)
        if od in ('1->19', '1->15', '7->12'):
            m91 = np.array(e91['mechanism'][od]['U_minus_nodes_1idx']) - 1
            assert np.array_equal(np.sort(m91), np.sort(U[od][0])), od

    j_of = {od: keys.index(od) for od in FOCAL}
    per_chain = {}
    for chain_key, rec in e100['per_chain'].items():
        od = rec['od']
        band = chain_key.split('|')[1]
        s0f, s1f = (float(x) for x in band.split('->'))
        q0v = q_base.copy()
        q1v = q_base.copy()
        q0v[j_of[od]] = s0f * q_base[j_of[od]]
        q1v[j_of[od]] = s1f * q_base[j_of[od]]
        dg_S = z100[f"dg_{od.replace('->', '_')}_{band.split('->')[0]}"]
        U_m, U_p = U[od]

        per_seed = []
        for seed in SEEDS:
            ckpt = torch.load(CKPT_DIR / f'seed{seed}.pt', weights_only=False,
                              map_location='cpu')
            model = build_fullnet_model(ps, q_bg, seed)
            model.load_state_dict(ckpt['best'])
            model.eval()
            with torch.no_grad():
                f0 = model(torch.tensor(q0v[None, :], dtype=torch.float64))[0].numpy()
                f1 = model(torch.tensor(q1v[None, :], dtype=torch.float64))[0].numpy()
            cons = max(float(np.max(np.abs(
                B @ f0 - net.od_M(all_ods) @ np.concatenate([q0v, q_bg])))),
                float(np.max(np.abs(
                    B @ f1 - net.od_M(all_ods) @ np.concatenate([q1v, q_bg])))))
            assert cons <= 1e-10, (chain_key, seed, cons)
            dg_L = edge_g(f1, t0, cap) - edge_g(f0, t0, cap)
            BC_L = B @ dg_L
            entry = {
                'seed': seed,
                'Cj_learned': float(dg_L.sum()),
                'phi_U_minus': -float(BC_L[U_m].sum()),
                'phi_U_plus': -float(BC_L[U_p].sum()),
                'col_err_vs_direct_rel': float(
                    np.linalg.norm(dg_L - dg_S) / np.linalg.norm(dg_S)),
                'phi_U_minus_err_abs': abs(-float(BC_L[U_m].sum())
                                           - rec['connected_min']),
                'phi_U_plus_err_abs': abs(-float(BC_L[U_p].sum())
                                          - rec['connected_max']),
                'Cj_err_pct': 100.0 * abs(float(dg_L.sum()) - rec['Cj_dT'])
                / abs(rec['Cj_dT']),
            }
            per_seed.append(entry)
        learned_sign_crossing = all(
            e['phi_U_minus'] < 0 < e['phi_U_plus'] for e in per_seed)
        per_chain[chain_key] = {
            'od': od, 'band': band,
            'direct': {'Cj_dT': rec['Cj_dT'],
                       'connected_min': rec['connected_min'],
                       'connected_max': rec['connected_max']},
            'per_seed': per_seed,
            'all_seeds_sign_crossing': learned_sign_crossing,
            'Cj_positive_all_seeds': all(e['Cj_learned'] > 0
                                         for e in per_seed),
        }
        print(f"{chain_key}: learned C_j "
              f"[{min(e['Cj_learned'] for e in per_seed):.0f}, "
              f"{max(e['Cj_learned'] for e in per_seed):.0f}] "
              f"U_- [{min(e['phi_U_minus'] for e in per_seed):.0f}, "
              f"{max(e['phi_U_minus'] for e in per_seed):.0f}] "
              f"U_+ [{min(e['phi_U_plus'] for e in per_seed):.0f}, "
              f"{max(e['phi_U_plus'] for e in per_seed):.0f}] "
              f"crossing={learned_sign_crossing}", flush=True)

    out = {
        'protocol': {
            'purpose': 'learned-surrogate evaluation of the small single-OD '
                       'demand bands via the endpoint identity; fixed '
                       'direct-equilibrium extremal boundaries; errors '
                       'against the e100 direct endpoint columns',
            'scenarios': 'focal OD demand at 0.9->1.1 or 0.8->1.2 of its '
                         'published value, all other demands at published '
                         'values; identical to the e100 direct scenarios',
            'claims_supported': 'whether the conservative surrogates '
                                'reproduce the direct sign conclusions on '
                                'these scenarios, and their quantitative '
                                'endpoint-column accuracy; either outcome '
                                'is reported',
        },
        'gates': {'direct_reference_reproduction': 'e100 C_j and connected '
                                                   'extrema reproduced <=1e-6 rel',
                  'learned_conservation': '<=1e-10'},
        'per_chain': per_chain,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
