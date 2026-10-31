"""E100 (v92, external review round 64, section IV enhancement): does the
sign crossing survive SMALL demand changes?

The published single-OD chains (E95/E97) move one focal demand from
0.2 to 1.6 of its published value -- a large change.  E100 keeps the
E95 protocol (one moving coordinate, the other 527 positive demands at
published values, all 528 ODs re-equilibrated jointly, theta=1, ten
paths per OD, setting-C machinery) but shrinks the demand interval to
two pre-specified bands:

    0.9 q_pub -> 1.1 q_pub        ( +-10% )
    0.8 q_pub -> 1.2 q_pub        ( +-20% )

Focal ODs: the three sign-crossing queries (1,19), (1,15), (7,12) plus
the positive control (6,8).  For a separable TSTT outcome and a single
moving demand, Eq. (14) of the paper gives the attribution column
EXACTLY from the two endpoint equilibria,

    C~_{e,j} = g_e(F_e(q^1)) - g_e(F_e(q^0)),   g_e(F)=F t_e(F),

so no quadrature chain is needed: two direct solves per (query,
interval) chain.  One 41-point integral chain (1->19, 0.8->1.2) is
re-solved as a spot check that the endpoint column and the integrated
column agree to quadrature error (the E97 identity machinery).

Reported per (query, interval): system contribution C_j = dT (endpoint
TSTT change, equal to the column sum by construction), the
connected-boundary interval [min, max] over the SAME exact connected
family as the published results (e80 C enumerator), the unrestricted
extrema (closed form), the account of the PUBLISHED baseline minimal
boundary U_- (node set frozen from e91) on the new column, and the
sign-crossing verdict (C_j > 0 with connected min < 0).  If a crossing
disappears at a small interval, that is reported as-is: it would say
the sign result depends on the demand scenario, not that the
classification or the published existence statements fail.

Gates (all must pass before any number enters the paper):
  (a) E95/E97 solver thresholds at every solved state: path fixed-point
      residual <= 1e-7, flow conservation |B F - M q| <= 1e-6;
  (b) per-edge g sums to TSTTT.value at both endpoints (1e-9 relative)
      -- the column sum equals the endpoint TSTT change exactly;
  (c) every phi is computed twice, node-sum and signed crossing-link
      sum, agreeing to 1e-6 (e99 gate b);
  (d) connected extrema: attaining masks verified in-family with values
      recomputed by direct summation (e80 gate c); connected interval
      nested inside the unrestricted extrema (e80 gate a);
  (e) the published U_- node set is in-family for its query and its
      phi matches direct evaluation;
  (f) spot check: the 41-point integrated column reproduces the
      endpoint column within E97-scale quadrature error (reported; gate
      at 0.1 TSTT units absolute and 1e-4 of max|dg| relative).

Output: experiments/results/e100_v92_small_demand.json
        experiments/results/e100_v92_small_demand.npz
        console log experiments/results/e100_console.log
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
from sue import PathStructure  # noqa: E402
from run_e9_transport import TSTTT  # noqa: E402
from run_e83_v82_validation import EdgeSUE, attribution  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e80_connected_cuts import (  # noqa: E402
    build_adj_bits, connected, load_c_enum, enum_c, mask_to_nodes)

RESULTS = HERE / 'results'
OUT = RESULTS / 'e100_v92_small_demand.json'
NPZ = RESULTS / 'e100_v92_small_demand.npz'
E91 = RESULTS / 'e91_v87_negative_mechanism.json'

STUDY_K = 6
GRID = 41
FOCAL = ('1->19', '1->15', '7->12', '6->8')   # three crossings + control
INTERVALS = ((0.9, 1.1), (0.8, 1.2))
SPOT = ('1->19', (0.8, 1.2))                  # integral spot check


def edge_g(F, t0, cap):
    """g_e(F)=F t_e(F), identical BPR form to TSTTT.value."""
    x = F / cap
    return t0 * F * (1.0 + 0.15 * x ** 4)


def phi_linksum(Cj, edges, nodes0):
    """Signed crossing-link sum: +C~ leaving, -C~ entering (e99)."""
    U = set(nodes0)
    total = 0.0
    for e, (u, v) in enumerate(edges):
        if u in U and v not in U:
            total += float(Cj[e])
        elif v in U and u not in U:
            total -= float(Cj[e])
    return total


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    study_ods, trips, _ = select_ods(net)
    keys6 = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    q_pub6 = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    psfull = PathStructure(net, all_ods)
    solver = EdgeSUE(psfull, study_k=STUDY_K)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    M_full = net.od_M(all_ods)

    t0 = np.asarray(net.free_flow_time)
    cap = np.asarray(net.capacity)
    edges = list(net.edges)
    n = net.n
    full_mask = (1 << n) - 1
    adj, _ = build_adj_bits(net)
    lib = load_c_enum()

    e91 = json.loads(E91.read_text())['mechanism']

    def solve_state(key, scale):
        """Joint equilibrium with focal demand at `scale` x published."""
        j = keys6.index(key)
        q6 = q_pub6.copy()
        q6[j] = scale * q_pub6[j]
        q = np.concatenate([q6, q_bg])
        F, h, p, path_res = solver.solve(q)
        flow_res = float(np.max(np.abs(B @ F - M_full @ q)))
        assert path_res <= 1e-7, (key, scale, 'path residual', path_res)
        assert flow_res <= 1e-6, (key, scale, 'flow conservation', flow_res)
        return j, q, F, h, p, {'path_residual': float(path_res),
                               'flow_conservation': flow_res}

    def endpoint_column(key, lo, hi):
        """C~_{:,j} = g(F1) - g(F0) plus gate records."""
        j, q0, F0, _, _, g0rec = solve_state(key, lo)
        _, q1, F1, _, _, g1rec = solve_state(key, hi)
        g0 = edge_g(F0, t0, cap)
        g1 = edge_g(F1, t0, cap)
        dg = g1 - g0
        # gate (b): per-edge g matches TSTTT at both endpoints; the
        # column sum is then exactly the endpoint TSTT change
        for tag, (ga, Fa) in (('g0', (g0, F0)), ('g1', (g1, F1))):
            rel = abs(float(ga.sum()) - float(T.value(Fa))) \
                / abs(float(T.value(Fa)))
            assert rel <= 1e-9, (key, lo, hi, tag, rel)
        dT = float(T.value(F1) - T.value(F0))
        rel_sum = abs(float(dg.sum()) - dT) / max(abs(dT), 1e-12)
        assert rel_sum <= 1e-9, (key, lo, hi, 'column sum vs dT', rel_sum)
        return j, dg, dT, F0, F1, q0, q1, {
            'endpoints': [g0rec, g1rec],
            'colsum_vs_dT_rel': float(rel_sum),
        }

    per_chain = {}
    npz_out = {}
    for key in FOCAL:
        s, t = study_ods[keys6.index(key)]
        jcol = None
        for lo, hi in INTERVALS:
            j, dg, dT, F0, F1, q0, q1, rec = endpoint_column(key, lo, hi)
            jcol = j
            BC = B @ dg
            col = BC.copy()

            # unrestricted extrema, closed form (eq:cutextrema)
            interior = np.ones(n, dtype=bool)
            interior[[s, t]] = False
            phi_s = float(-col[s])
            pos_div = float(np.maximum(col[interior], 0.0).sum())
            neg_div = float(np.maximum(-col[interior], 0.0).sum())
            unres_min = phi_s - pos_div
            unres_max = phi_s + neg_div

            # connected extrema, exact C enumeration (e80 machinery)
            res = enum_c(lib, adj, n, s, t, col[:, None])
            conn_min = float(res['phi_min'][0])
            conn_max = float(res['phi_max'][0])
            # gate (a'): connected nested inside unrestricted
            assert conn_min >= unres_min - 1e-6 * max(1.0, abs(unres_min))
            assert conn_max <= unres_max + 1e-6 * max(1.0, abs(unres_max))
            # gate (d): attaining masks in-family, values by direct sum
            conn_nodes = {}
            for tag, msk, val in (('min', res['min_mask'][0], conn_min),
                                  ('max', res['max_mask'][0], conn_max)):
                nodes = [v - 1 for v in mask_to_nodes(msk)]
                assert s in nodes and t not in nodes, (key, tag)
                assert connected(adj, msk) and connected(adj, full_mask & ~msk)
                direct = -float(col[nodes].sum())
                link = phi_linksum(dg, edges, nodes)   # gate (c)
                assert abs(direct - val) <= 1e-6 * max(1.0, abs(val)), \
                    (key, tag, direct, val)
                assert abs(link - direct) <= 1e-6, (key, tag, link, direct)
                conn_nodes[tag] = [v + 1 for v in nodes]

            # published baseline U_- (and U_+ for context) on the new column
            fixed = {}
            if key in e91:
                for tag in ('U_minus', 'U_plus'):
                    nodes1 = e91[key][f'{tag}_nodes_1idx']
                    nodes0 = [v - 1 for v in nodes1]
                    mask = 0
                    for v in nodes0:
                        mask |= 1 << v
                    assert s in nodes0 and t not in nodes0, (key, tag)
                    assert connected(adj, mask) \
                        and connected(adj, full_mask & ~mask), (key, tag)
                    phi_n = -float(col[nodes0].sum())
                    phi_l = phi_linksum(dg, edges, nodes0)
                    assert abs(phi_n - phi_l) <= 1e-6, (key, tag)
                    fixed[tag] = phi_n

            chain_id = f'{key}|{lo}->{hi}'
            per_chain[chain_id] = {
                'od': key,
                'interval': [lo, hi],
                'q0_q1_veh_h': [float(q0[jcol]), float(q1[jcol])],
                'Cj_dT': dT,
                'phi_s': phi_s,
                'unrestricted_min': unres_min,
                'unrestricted_max': unres_max,
                'connected_min': conn_min,
                'connected_max': conn_max,
                'connected_min_nodes_1idx': conn_nodes['min'],
                'connected_max_nodes_1idx': conn_nodes['max'],
                'phi_published_U_minus': fixed.get('U_minus'),
                'phi_published_U_plus': fixed.get('U_plus'),
                'sign_crossing': bool(dT > 0.0 and conn_min < 0.0),
                'gates': rec,
            }
            npz_out[f'dg_{key.replace("->", "_")}_{lo}'] = dg
            print(f"{chain_id}: C_j={dT:+,.1f}  connected "
                  f"[{conn_min:+,.1f}, {conn_max:+,.1f}]  "
                  f"U_-: {fixed.get('U_minus', float('nan')):+,.1f}  "
                  f"crossing={per_chain[chain_id]['sign_crossing']}",
                  flush=True)

    # ---- gate (f): integral-vs-endpoint spot check on one chain ----
    key, (lo, hi) = SPOT
    j = keys6.index(key)
    pub_j = q_pub6[j]
    dQ = np.zeros(STUDY_K)
    dQ[j] = (hi - lo) * pub_j
    ts = np.linspace(0.0, 1.0, GRID)
    Fs, gs, Js = [], [], []
    warm = None
    for lam in ts:
        q6 = q_pub6.copy()
        q6[j] = lo * pub_j + lam * dQ[j]
        q = np.concatenate([q6, q_bg])
        F, h, p, path_res = solver.solve(q, warm)
        warm = F
        assert path_res <= 1e-7, (key, 'spot path residual', path_res)
        assert float(np.max(np.abs(B @ F - M_full @ q))) <= 1e-6
        J, _, _ = solver.sensitivity(F, h, p, q)
        Fs.append(F)
        gs.append(T.grad(F))
        Js.append(J)
    Cint = attribution(np.stack(gs), np.stack(Js), ts, dQ)[:, j]
    j2, dg, dT, F0, F1, q0, q1, rec = endpoint_column(key, lo, hi)
    assert float(np.max(np.abs(Fs[0] - F0))) <= 1e-6   # same endpoint state
    assert float(np.max(np.abs(Fs[-1] - F1))) <= 1e-6
    resid = Cint - dg
    spot = {
        'chain': f'{key}|{lo}->{hi}',
        'grid': GRID,
        'resid_inf': float(np.max(np.abs(resid))),
        'resid_inf_over_max_dg': float(np.max(np.abs(resid))
                                       / np.max(np.abs(dg))),
        'dT_integral': float(Cint.sum()),
        'dT_endpoint': dT,
        'completeness_rel': float(abs(float(Cint.sum()) - dT) / abs(dT)),
    }
    assert spot['resid_inf'] <= 0.1, spot
    assert spot['resid_inf_over_max_dg'] <= 1e-4, spot
    print(f"spot {key} {lo}->{hi}: ||int-endpoint||_inf="
          f"{spot['resid_inf']:.2e} "
          f"({spot['resid_inf_over_max_dg']:.2e} of max|dg|)", flush=True)

    crossings = {c: per_chain[c]['sign_crossing'] for c in per_chain}
    out = {
        'protocol': {
            'purpose': 'small-interval endpoint test: do the sign '
                       'crossings survive +-10% and +-20% demand bands '
                       'around the published demand, single moving OD, '
                       'all 528 ODs jointly re-equilibrated (E95 '
                       'protocol, Eq. (14) endpoint columns)',
            'focal_ods': list(FOCAL),
            'intervals': [list(iv) for iv in INTERVALS],
            'columns': 'endpoint identity Eq. (14), g_e(F1)-g_e(F0); '
                       'one 41-point integral spot check',
            'connected_family': 'same exact e80 enumeration as the '
                                'published connected intervals',
            'claims_supported': 'persistence or disappearance of the '
                                'sign crossings under small demand '
                                'changes; either outcome is reported',
        },
        'per_chain': per_chain,
        'spot_check': spot,
        'elapsed_seconds': time.time() - started,
    }
    OUT.write_text(json.dumps(out, indent=2))
    np.savez_compressed(NPZ, **npz_out)
    print(f'written -> {OUT}', flush=True)
    print(f'crossings: {crossings}', flush=True)


if __name__ == '__main__':
    main()
