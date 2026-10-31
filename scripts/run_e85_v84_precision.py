"""E85 (v84, external review round 53, review item 4): targeted precision
check on the near-zero zero-background direct-SUE cut ranges, plus two
postprocessing extracts from the frozen e83 arrays.

Part 1 (precision): setting A (zero background, direct implicit SUE,
theta=1, ten paths per study OD) chains at 41, 81, and 161 nested
trapezoid nodes.  The published Table-2 'SUE A' column comes from the
e83 41-node implicit chain (C_zero in e83_v82_validation.npz).  A small
full-column relative error does not automatically grant the same
significant digits to the near-zero unrestricted range of ODs (10,11)
and (11,10): that range is an absolute-value sum of node differences of
the attribution column.  This part reports, per grid, phi_j({s_j}), the
absolute range S_j, 100*S_j/|phi_j|, the {10}->{10,17} boundary
accounts, and full-column relative-l2 differences between grids.

Gates (all must pass before any number enters the paper):
  (a) the fresh 41-node chain reproduces e83's J_zero/C_zero to <1e-12
      relative (same warm-start protocol, byte-level replica);
  (b) Jacobian conservation max|B J_j - mu_j| <= 1e-8 on every node;
  (c) path fixed-point residual <= 1e-7 at every node.

Part 2 (Jacobian-error localization): the largest learned-surrogate
Jacobian column error against direct SUE (published max 9.56%) localized
to (seed, OD, quadrature point) from the frozen e83 arrays.

Part 3 (A/C same-model boundary comparison): direct-SUE zero-background
(setting A) accounts at U={10} and U'={10,17} for ODs (10,16) and
(10,11), against the setting-C accounts of Table 4 (cross-checked with
e84_v83_coverage.json C_k10_theta1 to <1e-9 relative); relative changes
computed for both settings.

Output: experiments/results/e85_v84_precision.json
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
from run_e9_transport import _sioux_setup, TSTTT  # noqa: E402
from run_e83_v82_validation import EdgeSUE, attribution  # noqa: E402

RESULTS = HERE / 'results'
E83_NPZ = RESULTS / 'e83_v82_validation.npz'
E84_JSON = RESULTS / 'e84_v83_coverage.json'
OUT = RESULTS / 'e85_v84_precision.json'

SHORT_ODS = {(9, 10), (10, 9)}          # (10,11) and (11,10), 0-based
BOUNDARY_ODS = {(9, 15), (9, 10)}      # (10,16) and (10,11), 0-based


def col_err(a, b):
    """Relative l2 over links, per column."""
    return np.linalg.norm(a - b, axis=0) / np.linalg.norm(b, axis=0)


def zero_chain(solver, q0, dQ, nodes, T):
    """Setting-A implicit chain, e83 zero protocol (warm from previous node)."""
    ts = np.linspace(0.0, 1.0, nodes)
    Fs, Js = [], []
    warm = None
    for t in ts:
        q = q0 + t * dQ
        F, h, p, res = solver.solve(q, warm)
        warm = F
        J, _, _ = solver.sensitivity(F, h, p, q)
        Fs.append(F)
        Js.append(J)
    F = np.stack(Fs)
    J = np.stack(Js)
    g = np.stack([T.grad(f) for f in F])
    C = attribution(g, J, ts, dQ)
    return ts, F, J, g, C, float(solver.max_residual)


def part1(net, ods, qbase, ps, ec):
    """Nested-grid precision chains for setting A."""
    B, M = ec.B, ec.M
    n = B.shape[0]
    q0, q1 = 0.2 * qbase, 1.6 * qbase
    dQ = q1 - q0
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    z = np.load(E83_NPZ)

    chains = {}
    for nodes in (41, 81, 161):
        solver = EdgeSUE(ps)
        t0 = time.time()
        ts, F, J, g, C, res = zero_chain(solver, q0, dQ, nodes, T)
        # gate (b): implicit sensitivities conserve OD demand exactly on
        # the path-flow construction
        jcons = float(np.max(np.abs(B @ J - M)))
        assert jcons <= 1e-8, f'Jacobian conservation gate failed: {jcons}'
        # gate (c): path fixed-point residual
        assert res <= 1e-7, f'path residual gate failed: {res}'
        chains[nodes] = {'ts': ts, 'C': C, 'res': res, 'jcons': jcons,
                         'seconds': time.time() - t0}
        print(f'  A chain {nodes}n done in {chains[nodes]["seconds"]:.1f}s '
              f'(path_res={res:.2e}, J_cons={jcons:.2e})', flush=True)

    # gate (a): 41-node replica of the frozen e83 zero chain
    relJ = float(np.max(
        np.linalg.norm(chains[41]['C'] - z['C_zero'], axis=0)
        / np.linalg.norm(z['C_zero'], axis=0)))
    assert relJ < 1e-12, f'gate (a) failed: C 41-node drift {relJ:.2e}'
    print(f'gate (a) passed: 41-node C replicates e83 C_zero ({relJ:.2e})',
          flush=True)

    out = {'grids': {}}
    for nodes, ch in chains.items():
        C = ch['C']
        BC = B @ C
        rows = {}
        for j, (s, t) in enumerate(ods):
            col = BC[:, j]
            interior = np.ones(n, dtype=bool)
            interior[[s, t]] = False
            spread = float(np.abs(col[interior]).sum())
            phi_s = float(-col[s])
            row = {
                'phi_s': phi_s,
                'spread_abs': spread,
                'spread_pct': 100.0 * spread / abs(phi_s),
            }
            if (s, t) in BOUNDARY_ODS:
                phi_U = float(-col[s])
                phi_Up = float(-(col[s] + col[16]))
                row['boundary_10_to_10_17'] = {
                    'U': [10], 'Uprime': [10, 17],
                    'phi_U': phi_U, 'phi_Uprime': phi_Up,
                    'delta_total': phi_Up - phi_U,
                    'delta_rel_pct': 100.0 * abs(phi_Up - phi_U) / abs(phi_U),
                }
            rows[f'{s + 1}->{t + 1}'] = row
        out['grids'][str(nodes)] = {
            'nodes': nodes, 'per_od': rows,
            'path_fixed_point_max_abs': ch['res'],
            'J_conservation_max_abs': ch['jcons'],
            'elapsed_seconds': ch['seconds'],
        }

    # grid-to-grid full-column differences and near-zero range stability
    diffs = {}
    for a, b in ((41, 81), (41, 161), (81, 161)):
        Ca, Cb = chains[a]['C'], chains[b]['C']
        diffs[f'{a}_vs_{b}'] = {
            'full_column_relative_l2_per_od': {
                f'{s + 1}->{t + 1}': float(col_err(Ca[:, j], Cb[:, j]))
                for j, (s, t) in enumerate(ods)},
        }
    near = {}
    for j, (s, t) in enumerate(ods):
        if (s, t) not in SHORT_ODS:
            continue
        key = f'{s + 1}->{t + 1}'
        near[key] = {
            'phi_s': {k: out['grids'][k]['per_od'][key]['phi_s']
                      for k in out['grids']},
            'spread_abs': {k: out['grids'][k]['per_od'][key]['spread_abs']
                           for k in out['grids']},
            'spread_pct': {k: out['grids'][k]['per_od'][key]['spread_pct']
                           for k in out['grids']},
        }
    out['grid_differences'] = diffs
    out['near_zero_rows'] = near
    return out


def part2(z, ods):
    """Localize the learned-vs-direct Jacobian column errors."""
    Jl = z['J_learned']          # (5, 41, m, 8)
    Jz = z['J_zero']             # (41, m, 8)
    errs = (np.linalg.norm(Jl - Jz[None], axis=2)
            / np.linalg.norm(Jz[None], axis=2))    # (5, 41, 8)
    seed, node, j = np.unravel_index(int(np.argmax(errs)), errs.shape)
    flat = errs.ravel()
    return {
        'median_pct': float(100 * np.median(flat)),
        'max_pct': float(100 * flat.max()),
        'max_location': {
            'seed': int(seed),
            'quadrature_point_1idx': int(node + 1),
            'quadrature_t': float(z['ts'][node]),
            'od': f'{ods[j][0] + 1}->{ods[j][1] + 1}',
        },
        'per_seed_max_pct': [float(100 * errs[s].max()) for s in range(5)],
    }


def part3(B, z, e84, ods):
    """A/C same-model boundary comparison at U={10}, U'={10,17}."""
    BCz = B @ z['C_zero']
    BCc, BCs, BCb = z['BC'], z['BC_study'], z['BC_background']
    ref = e84['C_k10_theta1']['per_od']
    out = {}
    for st in ((9, 15), (9, 10)):          # (10,16), (10,11), 0-based
        j = ods.index(st)
        s, t = st
        key = f'{s + 1}->{t + 1}'
        # cross-check the C side against the published Table-4 source
        r = ref[key]['boundary_10_to_10_17']
        for name, mine, theirs in (
                ('phi_U', -float(BCc[s, j]), r['phi_U']),
                ('phi_Uprime', -float(BCc[s, j] + BCc[16, j]), r['phi_Uprime']),
                ('delta_study', -float(BCs[16, j]), r['delta_study']),
                ('delta_background', -float(BCb[16, j]), r['delta_background']),
                ('delta_total', -float(BCc[16, j]), r['delta_total'])):
            assert abs(mine / theirs - 1) < 1e-9, \
                f'e83/e84 boundary mismatch ({key}/{name}): {mine} vs {theirs}'
        def bset(BC):
            phi_U = float(-BC[s, j])
            phi_Up = float(-(BC[s, j] + BC[16, j]))
            return {'phi_U': phi_U, 'phi_Uprime': phi_Up,
                    'delta': phi_Up - phi_U,
                    'delta_rel_pct': 100.0 * abs(phi_Up - phi_U) / abs(phi_U)}
        out[key] = {'A_zero': bset(BCz), 'C_reroute': bset(BCc),
                    'C_study': bset(BCs), 'C_background': bset(BCb)}
    return out


def main():
    started = time.time()
    net, ods, qbase, ps, classes, ec = _sioux_setup()
    z = np.load(E83_NPZ)
    e84 = json.loads(E84_JSON.read_text())
    out = {'protocol': {
        'setting': 'A: zero background, direct fixed-path logit SUE, '
                   'theta=1, ten paths per study OD, implicit sensitivities',
        'grids': [41, 81, 161],
        'integration': 'uniform composite trapezoid, nested grids',
    }}
    print('== Part 1: near-zero precision chains ==', flush=True)
    out['precision'] = part1(net, ods, qbase, ps, ec)
    print('== Part 2: Jacobian error localization ==', flush=True)
    out['jacobian_error_localization'] = part2(z, ods)
    print('== Part 3: A/C boundary comparison ==', flush=True)
    out['boundary_A_vs_C'] = part3(ec.B, z, e84, ods)
    out['elapsed_seconds'] = time.time() - started
    OUT.write_text(json.dumps(out, indent=2))
    print(f'Completed e85 in {out["elapsed_seconds"]:.1f}s -> {OUT}',
          flush=True)


if __name__ == '__main__':
    main()
