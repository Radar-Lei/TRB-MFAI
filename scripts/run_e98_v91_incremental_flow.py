"""E98 (v91, external review round 63, item III-3 "the numerical
analysis most worth adding"): demand-change FLOW INCREMENT error of the
full-background surrogate, on a denominator matched to the attribution
task.

Motivation.  The published full-background surrogate validation reports
a total-flow error of 0.037-0.065% (denominator: the full-network flow
level) alongside attribution-column errors of 10.6-12.3% medians
(denominator: the query's own contribution column).  The two percentages
do not share a scale: in the full-background setting the total-flow
denominator can be far larger than the attribution denominator, so the
pair cannot tell how much of the gap is derivative-fitting error and how
much is a denominator mismatch.  The review asks for the increment error

    E_dF = || (f^L(q1) - f^L(q0)) - (f^S(q1) - f^S(q0)) ||_2
           / || f^S(q1) - f^S(q0) ||_2,

with L the learned surrogate and S the direct SUE, both evaluated on the
identical six-query 41-state setting-C chain.  This asks the question
closest to the interpretation task: does the model reproduce the flow
CHANGE caused by THIS demand change, not just the large flow level?

All quantities are re-summarized from frozen arrays -- no new training,
no new equilibrium solves:
  * f^S: e87 npz field F (81, 76); the published 41-point chain is the
    81-point chain's even nodes, so f^S(q0) = F[0], f^S(q1) = F[80];
  * f^L: e93 npz field f_L (5, 41, 76), seed s giving f_L[s, 0] and
    f_L[s, 40].

Gates:
  (a) chain alignment: e93 ts41 equals e87 ts[::2] exactly;
  (b) endpoint demands: e93 Q41 row 0 is 0.2 x published and row 40 is
      1.6 x published for each of the six query ODs (background demands
      constant across the chain);
  (c) learned flows are conservative at both endpoints of every seed:
      max |B f_L - M_full q| <= 1e-6 (e93 reported 3.6e-11 chain-wide);
  (d) the chain-mean level error reproduces the e93 JSON
      flow_err_mean per seed within 1e-9 (same definition, same data);
  (e) the SUE endpoint increment is nondegenerate: ||dF^S||_2 > 0 and
      reported for scale.

Reported per seed and summarized over the five seeds:
  * E_dF as defined above;
  * for context, the endpoint LEVEL errors ||f^L(q)-f^S(q)||_2 /
    ||f^S(q)||_2 at both endpoints and the chain mean level error;
  * the SUE increment norm, the endpoint flow norms, and the ratio
    ||dF^S||_2 / ||f^S(q0)||_2 (increment scale vs level scale) --
    this ratio is the denominator asymmetry the review points out.

NOT claimed: attribution accuracy.  E_dF is a flow-space diagnostic on
one demand direction; it does not replace the per-OD Jacobian or
attribution-column errors, and it is not a bound on either.

Output: experiments/results/e98_v91_incremental_flow.json
        console log experiments/results/e98_console.log
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
from run_e86_v84_newod import select_ods  # noqa: E402

RESULTS = HERE / 'results'
E86_JSON = RESULTS / 'e86_v84_newod.json'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E93_JSON = RESULTS / 'e93_v89_fullbg_surrogate.json'
E93_NPZ = RESULTS / 'e93_v89_fullbg_surrogate.npz'
OUT = RESULTS / 'e98_v91_incremental_flow.json'

SEEDS = (0, 1, 2, 3, 4)


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, trips, _ = select_ods(net)
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    q_pub6 = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    M_full = net.od_M(all_ods)

    e86 = json.loads(E86_JSON.read_text())
    z87 = np.load(E87_NPZ)
    F81 = z87['F']                                  # (81, 76)
    ts81 = z87['ts']
    z93 = np.load(E93_NPZ)
    f_L = z93['f_L']                                # (5, 41, 76)
    Q41 = z93['Q41']                                # (41, 6)
    ts41 = z93['ts41']

    # ---- gate (a): the 41-chain is the 81-chain's even nodes ----
    assert np.array_equal(ts41, ts81[::2]), 'chain misalignment'

    # ---- gate (b): endpoint demands are 0.2x / 1.6x published ----
    assert np.allclose(Q41[0], 0.2 * q_pub6, atol=1e-12), Q41[0]
    assert np.allclose(Q41[-1], 1.6 * q_pub6, atol=1e-12), Q41[-1]
    for i in range(len(Q41)):                       # background constant
        assert np.allclose(Q41[i], 0.2 * q_pub6
                           + (ts41[i] / ts41[-1]) * 1.4 * q_pub6,
                           atol=1e-12), i

    # ---- SUE endpoint flows and the increment ----
    fS0, fS1 = F81[0], F81[-1]
    dS = fS1 - fS0
    nS = float(np.linalg.norm(dS))                  # gate (e)
    assert nS > 0.0

    # ---- gate (c): learned endpoint flows conservative ----
    def full_q(q6):
        return np.concatenate([q6, q_bg])
    q0_full = full_q(Q41[0])
    q1_full = full_q(Q41[-1])
    for si in range(len(SEEDS)):
        for F in (f_L[si, 0], f_L[si, -1]):
            res = float(np.max(np.abs(B @ F - M_full @ q0_full)))
            res1 = float(np.max(np.abs(B @ F - M_full @ q1_full)))
            assert min(res, res1) <= 1e-6, (si, res, res1)

    per_seed = {}
    for si, seed in enumerate(SEEDS):
        fL0, fL1 = f_L[si, 0], f_L[si, -1]
        dL = fL1 - fL0
        e_inc = float(np.linalg.norm(dL - dS) / nS)
        lvl0 = float(np.linalg.norm(fL0 - fS0) / np.linalg.norm(fS0))
        lvl1 = float(np.linalg.norm(fL1 - fS1) / np.linalg.norm(fS1))
        # chain-mean level error, same definition as e93 flow_err_mean
        chain_lvl = float(np.mean([
            np.linalg.norm(f_L[si, i] - F81[2 * i])
            / np.linalg.norm(F81[2 * i]) for i in range(41)]))
        per_seed[seed] = {
            'E_dF': e_inc,
            'level_error_q0': lvl0,
            'level_error_q1': lvl1,
            'chain_mean_level_error': chain_lvl,
            'gates': {'conservation_endpoint_max': float(np.max([
                np.max(np.abs(B @ f_L[si, k] - M_full @ full_q(Q41[k])))
                for k in (0, -1)]))},
        }
        # ---- gate (d): chain-mean level error reproduces e93 JSON ----
        pub = float(e93_json_flow_err(E93_JSON, si))
        assert abs(chain_lvl - pub) <= 1e-9, (seed, chain_lvl, pub)
        print(f"seed {seed}: E_dF={e_inc*100:.3f}%  "
              f"level(q0)={lvl0*100:.3f}%  level(q1)={lvl1*100:.3f}%  "
              f"chain-mean={chain_lvl*100:.3f}%", flush=True)

    inc = [per_seed[s]['E_dF'] for s in SEEDS]
    out = {
        'protocol': {
            'purpose': 'demand-change flow-increment error E_dF of the '
                       'full-background surrogate vs direct SUE on the '
                       'identical 41-state six-query setting-C chain; '
                       'denominator matched to the increment, not the '
                       'flow level',
            'definition': 'E_dF = ||(f_L(q1)-f_L(q0)) - (f_S(q1)-f_S(q0))'
                          '||_2 / ||f_S(q1)-f_S(q0)||_2',
            'seeds': list(SEEDS),
            'claims_supported': 'flow-space increment fidelity of the '
                                'surrogate on this demand direction; NOT '
                                'attribution accuracy and not a bound on '
                                'the attribution errors',
        },
        'sue_reference': {
            'dF_norm_l2': nS,
            'fS0_norm_l2': float(np.linalg.norm(fS0)),
            'fS1_norm_l2': float(np.linalg.norm(fS1)),
            'increment_over_level_pct': float(100.0 * nS
                                              / np.linalg.norm(fS0)),
        },
        'per_seed': per_seed,
        'summary': {
            'E_dF_median': float(np.median(inc)),
            'E_dF_min': float(min(inc)),
            'E_dF_max': float(max(inc)),
        },
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"summary: E_dF median {out['summary']['E_dF_median']*100:.3f}% "
          f"range [{out['summary']['E_dF_min']*100:.3f}%, "
          f"{out['summary']['E_dF_max']*100:.3f}%]", flush=True)
    print(f"written -> {OUT}", flush=True)


def e93_json_flow_err(path, seed_index):
    d = json.loads(Path(path).read_text())
    return d['per_seed'][seed_index]['flow_err_mean']


if __name__ == '__main__':
    main()
