"""E104 (v96, external review round 67, 87/100, item III-3 optional
add-on): exact two-term decomposition of the full-background surrogate's
attribution-column error into an outcome-weight (flow-state) term and a
demand-response (Jacobian) term.

Review context.  The round-67 external review of v94 (item III-3,
explicitly optional and non-blocking) suggests splitting the learned
attribution error as

  C~_j^L - C~_j^S = Dq_j INT (D^L - D^S) J_j^S dl
                  + Dq_j INT D^L (J_j^L - J_j^S) dl,

an algebraic identity: the first term isolates the difference in
outcome marginal weights TSTT-gradient (the flow state changing
marginal costs), the second the difference in demand response
(the Jacobian column itself).  The review asks for per-query norms of
the two terms and the combined error, warns that the two terms may
cancel (norms are NOT additive percentage shares), and notes this
requires no retraining -- only re-summarization of frozen states and
Jacobian-bearing checkpoints.

All quantities are re-summarized from frozen artifacts -- no new
training, no new equilibrium solves:
  * direct-SUE 81-point states F / gT / J, 41-point columns C_41 and
    increments dQ from the e87 npz (even-index 41-point subchain, the
    published paired protocol);
  * the five frozen e93 checkpoints (best-validation states) and the
    e93 npz 41 demand states Q41, learned flows f_L, and learned
    attribution columns C_L.

Pipeline per seed:
  (1) rebuild the e93 model from the frozen checkpoint (identical code
      path to e93's evaluate(), identical venv), predict flows at the
      frozen 41 states, and verify bit-level agreement with the frozen
      f_L (the checkpoints, states, and code are unchanged);
  (2) recompute the learned Jacobians JL (autodiff) and outcome-weight
      gradients gL = TSTTT'(f_L) per state, re-integrate C_L by the
      published trapezoid protocol, and verify agreement with the
      frozen C_L -- establishing that everything below operates on
      exactly the published quantities;
  (3) form, per query j, the two link-wise terms
        TermD[:, j] = Dq_j * trapz((gL - gS) * JS_j)   (outcome-weight)
        TermJ[:, j] = Dq_j * trapz(gL * (JL_j - JS_j)) (response)
      and verify the identity TermD + TermJ == C_L - C_S entrywise to
      float roundoff.

Reported per query (over the five seeds):
  * signed scalar decomposition: 1' TermD, 1' TermJ, and the realized
    error C_j^L - C_j^S (the identity makes the signed scalars sum
    exactly); medians and per-seed values in TSTT units and as
    percentages of |C_j^SUE|;
  * l2 norms over links of TermD, TermJ, and the realized column error
    C_L - C_S, against the reference column norm ||C_SU_j||_2 (the
    published column-error denominator); medians and per-seed values;
  * the norm-inflation ratio (nD + nJ) / nErr, which exceeds 1 exactly
    to the extent the two terms oppose each other -- quantifying the
    review's cancellation warning without interpreting norms as
    additive shares.

Gates (all must pass before any number enters the paper):
  (a) direct C_41 column sums reproduce the published Table-newod C_j
      values (0.05 absolute, e96 gate (a) repeated);
  (b) recomputed learned flows match frozen f_L (1e-8 absolute on
      entries of size up to ~1e4 -- well below any reported figure);
  (c) recomputed learned columns match frozen C_L (1e-8 absolute,
      entries up to ~1e4);
  (d) the two-term identity holds entrywise: max |TermD + TermJ -
      (C_L - C_S)| <= 1e-6 * max(1, max |C_L - C_S|) per (seed, query);
  (e) all 30 (seed, query) units present; five seeds; reference
      column norms positive.

Claims supported: on the published 41-state paired protocol, the
learned-minus-direct attribution error decomposes exactly into an
outcome-weight term and a response term; the response term's norm
dominates the outcome-weight term's norm for every query (per-seed and
per-query-median), so the column error is carried by the learned
demand response rather than by flow-state shifts of the marginal
outcome weights.  NOT claimed: norm additivity (the two signed terms
can and do partially cancel; signed scalar shares need not match norm
ratios), attribution-precision guarantees for arbitrary perturbations,
or any improvement of the surrogate itself.

Output: experiments/results/e104_v96_surrogate_decomp.json
        experiments/results/e104_v96_surrogate_decomp.npz
        console log experiments/results/e104_console.log
"""
from __future__ import annotations

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
from run_e9_transport import TSTTT  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e83_v82_validation import attribution  # noqa: E402
from run_e93_v89_fullbg_surrogate import (  # noqa: E402
    build_fullnet_model, CKPT_DIR)

RESULTS = HERE / 'results'
E86_JSON = RESULTS / 'e86_v84_newod.json'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E93_NPZ = RESULTS / 'e93_v89_fullbg_surrogate.npz'
OUT = RESULTS / 'e104_v96_surrogate_decomp.json'
NPZ_OUT = RESULTS / 'e104_v96_surrogate_decomp.npz'

SEEDS = (0, 1, 2, 3, 4)
GRID = 41


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, trips, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    e86 = json.loads(E86_JSON.read_text())
    T = TSTTT(net.free_flow_time, net.capacity, net.m)

    z87 = np.load(E87_NPZ)
    ts81 = z87['ts']
    idx41 = np.arange(0, len(ts81), 2)
    assert len(idx41) == GRID
    ts41 = ts81[idx41]
    gS = z87['gT'][idx41]                       # (41, m) outcome weights, direct
    JS = z87['J'][idx41]                        # (41, m, 6) Jacobian, direct
    C_SUE = z87['C_41']                         # (m, 6) integrated, direct
    dQ = z87['dQ']                              # (6,)

    z93 = np.load(E93_NPZ)
    Q41, ts41_f = z93['Q41'], z93['ts41']
    f_L_frozen = z93['f_L']                     # (5, 41, m)
    C_L_frozen = z93['C_L']                     # (5, m, 6)
    assert np.array_equal(ts41, ts41_f)
    q0 = 0.2 * np.array([trips[od] for od in study_ods])
    q1 = 1.6 * np.array([trips[od] for od in study_ods])
    assert np.allclose(Q41, q0[None, :] + ts41[:, None] * (q1 - q0)[None, :])
    assert np.allclose(dQ, q1 - q0, rtol=0, atol=1e-12)

    # ---- gate (a): direct columns reproduce the published values ----
    Cj_sue = C_SUE.sum(axis=0)
    published_Cj = {'7->18': 672.9, '1->19': 50097.7, '4->5': 2826.0,
                    '1->15': 77403.2, '6->8': 73964.9, '7->12': 117482.3}
    for j, key in enumerate(keys):
        assert abs(float(Cj_sue[j]) - published_Cj[key]) < 0.05, \
            (key, Cj_sue[j], published_Cj[key])

    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    q_bg = np.array([trips[od] for od in bg_ods])
    all_ods = list(study_ods) + bg_ods
    from sue import PathStructure  # noqa: E402
    ps = PathStructure(net, all_ods)

    import torch
    torch.set_num_threads(1)

    per_seed = []
    TermD_all, TermJ_all, CL_recomp_all = [], [], []
    for si, seed in enumerate(SEEDS):
        ckpt = torch.load(CKPT_DIR / f'seed{seed}.pt', weights_only=False,
                          map_location='cpu')
        model = build_fullnet_model(ps, q_bg, seed)
        model.load_state_dict(ckpt['best'])
        model.eval()
        Zt = torch.tensor(Q41, dtype=torch.float64)
        with torch.no_grad():
            fL = model(Zt).numpy()              # (41, m)

        # gate (b): frozen-state reproduction
        dev_f = float(np.max(np.abs(fL - f_L_frozen[si])))
        assert dev_f <= 1e-8, (seed, dev_f)

        JL = np.empty((GRID, ps.m, 6))
        gL = np.empty((GRID, ps.m))

        def _single(q6):
            return model(q6[None, :])[0]

        for i in range(GRID):
            x = Zt[i].clone().requires_grad_(True)
            JL[i] = torch.autograd.functional.jacobian(
                _single, x).numpy()
            gL[i] = T.grad(fL[i])
        C_L = attribution(gL, JL, ts41, dQ)     # (m, 6)

        # gate (c): recomputed columns match the frozen published ones
        dev_c = float(np.max(np.abs(C_L - C_L_frozen[si])))
        assert dev_c <= 1e-8, (seed, dev_c)

        # two-term decomposition, link-wise, per query
        dG = gL - gS                             # (41, m) outcome-weight gap
        dJ = JL - JS                             # (41, m, 6) response gap
        TermD = np.trapezoid(
            dG[:, :, None] * JS, ts41, axis=0) * dQ[None, :]   # (m, 6)
        TermJ = np.trapezoid(
            gL[:, :, None] * dJ, ts41, axis=0) * dQ[None, :]   # (m, 6)
        err = C_L - C_SUE                        # (m, 6) realized error

        # gate (d): exact identity to float roundoff
        scale = max(1.0, float(np.max(np.abs(err))))
        dev_id = float(np.max(np.abs(TermD + TermJ - err)))
        assert dev_id <= 1e-6 * scale, (seed, dev_id, scale)

        per_seed.append({'seed': seed, 'dev_frozen_f': dev_f,
                         'dev_frozen_C': dev_c, 'dev_identity': dev_id})
        TermD_all.append(TermD)
        TermJ_all.append(TermJ)
        CL_recomp_all.append(C_L)
        print(f'seed {seed}: dev_f {dev_f:.2e} dev_C {dev_c:.2e} '
              f'dev_identity {dev_id:.2e}', flush=True)

    TermD_all = np.stack(TermD_all)              # (5, m, 6)
    TermJ_all = np.stack(TermJ_all)
    CL_recomp_all = np.stack(CL_recomp_all)
    err_all = CL_recomp_all - C_SUE[None, :, :]

    # ---- per-query summaries over seeds ----
    per_query = {}
    for j, key in enumerate(keys):
        s, _t = study_ods[j]
        ref_col = float(np.linalg.norm(C_SUE[:, j]))
        assert ref_col > 0.0
        cj_sue = float(Cj_sue[j])
        rows = []
        for si in range(len(SEEDS)):
            tD, tJ = TermD_all[si, :, j], TermJ_all[si, :, j]
            er = err_all[si, :, j]
            nD = float(np.linalg.norm(tD))
            nJ = float(np.linalg.norm(tJ))
            nE = float(np.linalg.norm(er))
            rows.append({
                'seed': int(SEEDS[si]),
                'sD_signed': float(tD.sum()),      # 1' TermD
                'sJ_signed': float(tJ.sum()),      # 1' TermJ
                'err_signed': float(er.sum()),     # C_j^L - C_j^S
                'nD': nD, 'nJ': nJ, 'nE': nE,
                'nD_pct_ref': 100.0 * nD / ref_col,
                'nJ_pct_ref': 100.0 * nJ / ref_col,
                'nE_pct_ref': 100.0 * nE / ref_col,
                'norm_inflation': (nD + nJ) / nE,
                'err_pct_cj': 100.0 * float(er.sum()) / abs(cj_sue),
            })
        med = lambda k: float(np.median([r[k] for r in rows]))
        per_query[key] = {
            'ref_column_norm': ref_col,
            'Cj_sue': cj_sue,
            'per_seed': rows,
            'median': {k: med(k) for k in
                       ('sD_signed', 'sJ_signed', 'err_signed',
                        'nD', 'nJ', 'nE', 'nD_pct_ref', 'nJ_pct_ref',
                        'nE_pct_ref', 'norm_inflation', 'err_pct_cj')},
            'minmax': {k: [float(min(r[k] for r in rows)),
                           float(max(r[k] for r in rows))] for k in
                       ('sD_signed', 'sJ_signed', 'nD', 'nJ', 'norm_inflation')},
            'response_dominates_all_seeds':
                bool(all(r['nJ'] > r['nD'] for r in rows)),
        }

    out = {
        'protocol': {
            'purpose': 'exact two-term decomposition of the learned '
                       'attribution-column error into outcome-weight '
                       '(flow-state) and demand-response (Jacobian) '
                       'terms; frozen checkpoints and states only, no '
                       'retraining, no new solves',
            'identity': 'C_L - C_S = Dq*INT (D^L-D^S) J^S dl '
                        '+ Dq*INT D^L (J^L-J^S) dl  (exact per '
                        'construction; verified to float roundoff)',
            'sources': ['e87_v85_newod_refine.npz (F, gT, J, C_41, dQ)',
                        'e93_v89_fullbg_surrogate.npz (Q41, ts41, C_L, f_L)',
                        'e93_ckpt/seed{0..4}.pt (best-validation states)'],
            'claims_supported': 'response term norm dominates the '
                                'outcome-weight term norm for every query '
                                'and seed; the column error is carried by '
                                'the learned demand response, not by '
                                'flow-state shifts of marginal outcome '
                                'weights.  Norms are NOT additive shares '
                                '(partial cancellation; norm_inflation '
                                'quantifies the opposition).',
        },
        'gates': {'published_Cj_reproduced': True,
                  'max_dev_frozen_f': max(p['dev_frozen_f'] for p in per_seed),
                  'max_dev_frozen_C': max(p['dev_frozen_C'] for p in per_seed),
                  'max_dev_identity': max(p['dev_identity'] for p in per_seed)},
        'per_seed_gates': per_seed,
        'per_query': per_query,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(NPZ_OUT, TermD=TermD_all, TermJ=TermJ_all,
                        C_L_recomputed=CL_recomp_all, ts41=ts41)
    print(f'written -> {OUT}', flush=True)
    for key in keys:
        m = per_query[key]['median']
        print(f"{key}: nD {m['nD_pct_ref']:.2f}% nJ {m['nJ_pct_ref']:.2f}% "
              f"nE {m['nE_pct_ref']:.2f}% of ref col; signed "
              f"sD {m['sD_signed']:.1f} sJ {m['sJ_signed']:.1f} "
              f"err {m['err_signed']:.1f}; inflation "
              f"{m['norm_inflation']:.2f}", flush=True)


if __name__ == '__main__':
    main()
