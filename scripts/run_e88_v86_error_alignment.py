"""E88 (v86, external review round 57, item A1): unified-sample flow-vs-
sensitivity error comparison on the 41-point setting-A chain.

The published v85 comparison juxtaposed the surrogate's test flow error
(0.139%; mean over 857 test demands within each seed, then median over
seeds) with the maximum learned-vs-direct Jacobian column error (9.56%;
pooled over 5x41x8 seed-point-OD columns).  Those are different samples
AND different summary statistics, so they must not be read as a strictly
aligned error-magnitude comparison.  This experiment re-summarizes BOTH
error families on the SAME demand states: the 41 zero-background
quadrature points of the attribution path, for each of the five frozen
surrogate seeds.

  E_F[s,i]   = ||f_learned(q_i) - F_SUE(q_i)||_2 / ||F_SUE(q_i)||_2
               (unit: seed x quadrature point; 5x41 = 205 samples)
  E_J[s,i,j] = ||J_learned[s,i] - J_SUE[i]||_2 / ||J_SUE[i]||_2
               (unit: seed x quadrature point x OD; 5x41x8 = 1640
               samples; identical to the published Jacobian definition)

Both families report median, 95th percentile, and maximum (pooled), plus
per-seed maxima.  The test-split flow error is NOT recomputed here; it
keeps its own published definition and is reported separately.

Gates (must pass before any number enters the paper):
  (a) the reloaded learned Jacobians reproduce e83's stored J_learned
      to < 1e-12 relative l2 (same checkpoints, same 41-point path);
  (b) every reloaded learned flow conserves demand exactly:
      max |B f_learned(q) - M q| <= 1e-8;
  (c) the recomputed Jacobian-error median and maximum reproduce the
      published 0.516% / 9.56% to display precision.

Output: experiments/results/e88_v86_error_alignment.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Must be set before BLAS/torch imports when launched directly.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import torch

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_e9_transport import _sioux_setup  # noqa: E402
from models import build_model  # noqa: E402
from attribution import jacobians_along_path  # noqa: E402

RESULTS = HERE / 'results'
E83_NPZ = RESULTS / 'e83_v82_validation.npz'
OUT = RESULTS / 'e88_v86_error_alignment.json'


def summarize(x: np.ndarray) -> dict:
    v = np.asarray(x, dtype=float).ravel()
    return {
        'n_samples': int(v.size),
        'median_pct': float(100 * np.median(v)),
        'p95_pct': float(100 * np.percentile(v, 95)),
        'max_pct': float(100 * v.max()),
        'min_pct': float(100 * v.min()),
    }


def main():
    started = time.time()
    torch.set_num_threads(1)
    net, ods, qbase, ps, classes, ec = _sioux_setup()
    B, M = ec.B, ec.M
    q0 = 0.2 * qbase
    dQ = 1.6 * qbase - q0

    z = np.load(E83_NPZ)
    ts, F_sue, J_sue, J_learned_stored = z['ts'], z['F_zero'], z['J_zero'], z['J_learned']
    assert ts.shape[0] == 41 and F_sue.shape == (41, ps.m) and \
        J_learned_stored.shape == (5, 41, ps.m, 8)

    F_learned = []
    J_learned = []
    flow_cons_gap = 0.0
    for seed in range(5):
        ckpt = torch.load(RESULTS / 'e5_ckpt' / f'M3|norm|raw|{seed}.pt',
                          weights_only=False, map_location='cpu')
        model = build_model('M3', ps, seed,
                            norm={'mu_q': np.zeros(8), 'sd_q': np.ones(8)})
        model.load_state_dict(ckpt['best'])
        model.eval()
        jl, fl, tlearn, dql = jacobians_along_path(model, q0, q0 + dQ, steps=40)
        assert np.allclose(ts, tlearn) and np.allclose(dQ, dql)
        J_learned.append(jl)
        F_learned.append(fl)
        # gate (b): exact conservation of every loaded learned flow
        for i in range(41):
            q = q0 + ts[i] * dQ
            flow_cons_gap = max(flow_cons_gap,
                                float(np.max(np.abs(B @ fl[i] - M @ q))))
    J_learned = np.stack(J_learned)          # (5, 41, m, 8)
    F_learned = np.stack(F_learned)          # (5, 41, m)

    # gate (a): byte-level replica of the stored learned Jacobians
    rel_j = float(np.linalg.norm(J_learned - J_learned_stored)
                  / np.linalg.norm(J_learned_stored))
    assert rel_j < 1e-12, f'gate (a) failed: J_learned drift {rel_j:.2e}'
    assert flow_cons_gap <= 1e-8, \
        f'gate (b) failed: learned flow conservation gap {flow_cons_gap:.2e}'
    print(f'gates (a)/(b) passed: J drift {rel_j:.2e}, '
          f'flow conservation {flow_cons_gap:.2e}', flush=True)

    # same-sample error families
    E_F = (np.linalg.norm(F_learned - F_sue[None], axis=2)
           / np.linalg.norm(F_sue[None], axis=2))            # (5, 41)
    E_J = (np.linalg.norm(J_learned - J_sue[None], axis=2)
           / np.linalg.norm(J_sue[None], axis=2))            # (5, 41, 8)
    assert E_J.shape == (5, 41, 8) and np.isfinite(E_J).all()

    jac_summary = summarize(E_J)
    # gate (c): published values 0.516% / 9.56% at display precision
    assert abs(jac_summary['median_pct'] - 0.516) < 5e-4, jac_summary
    assert abs(jac_summary['max_pct'] - 9.56) < 5e-3, jac_summary
    print(f'gate (c) passed: Jacobian median/max '
          f'{jac_summary["median_pct"]:.3f}/{jac_summary["max_pct"]:.2f}%',
          flush=True)

    out = {
        'protocol': {
            'purpose': 'unified-sample flow-vs-sensitivity error comparison '
                       '(review round 57, item A1); no new training, no new '
                       'equilibrium solves',
            'setting': 'A: zero background; direct fixed-path logit SUE '
                       '(theta=1, ten paths per study OD, implicit '
                       'sensitivities) vs the five frozen surrogate seeds',
            'demand_states': 'the 41 zero-background quadrature points of the '
                             'published attribution path (0.2->1.6 q_base)',
            'flow_units': 'seed x quadrature point (5x41 = 205)',
            'jacobian_units': 'seed x quadrature point x OD (5x41x8 = 1640)',
            'definitions': {
                'E_F': '||f_learned(q)-F_SUE(q)||_2 / ||F_SUE(q)||_2',
                'E_J': '||J_learned-J_SUE||_2 / ||J_SUE||_2 (per OD column, '
                       'identical to the published Jacobian definition)',
            },
        },
        'gates': {
            'learned_J_vs_e83_stored_relative_l2': rel_j,
            'learned_flow_conservation_max_abs': flow_cons_gap,
            'jacobian_median_matches_published_0p516': True,
            'jacobian_max_matches_published_9p56': True,
        },
        'flow_error_same_path': summarize(E_F),
        'jacobian_error_same_path': jac_summary,
        'per_seed': {
            'flow_error_max_pct': [float(100 * E_F[s].max()) for s in range(5)],
            'flow_error_median_pct': [float(100 * np.median(E_F[s])) for s in range(5)],
            'jacobian_error_max_pct': [float(100 * E_J[s].max()) for s in range(5)],
        },
        'max_locations': {
            'flow': {
                'seed': int(np.unravel_index(E_F.argmax(), E_F.shape)[0]),
                'quadrature_point_1idx': int(np.unravel_index(E_F.argmax(), E_F.shape)[1] + 1),
            },
            'jacobian': 'published localization (e85 part 2): seed 3, '
                        '(16,10), quadrature point 38 of 41 (t=0.925)',
        },
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'Completed e88 in {out["elapsed_seconds"]:.1f}s -> {OUT}', flush=True)
    print(json.dumps({'flow': out['flow_error_same_path'],
                      'jacobian': out['jacobian_error_same_path'],
                      'per_seed_flow_max': out['per_seed']['flow_error_max_pct']},
                     indent=1), flush=True)


if __name__ == '__main__':
    main()
