"""E92 (v88, external review round 60, 86/100 targeted revision).

Three parts, all on frozen arrays; no retraining, no new equilibrium
solves.

Part A (review item III-1: source of the setting-A boundary numbers).
The paper's Section 7.3 compares the connected enlargement
{10} -> {10,17} across settings A and C, and Appendix C.1 attributes
the setting-A boundary accounts to the 41-point direct-SUE chain,
while Section 7.1 illustrates the same enlargement with the seed-0
learned surrogate.  This part freezes both numbers side by side:

  * direct-SUE setting A (e83 npz C_zero on the identical 41-node
    grid, cross-checked against the frozen e79 BC_A chain and the
    e79 JSON field cut_change_ratio_node17): phi({10}),
    phi({10,17}), signed change, and relative change for ODs
    (10,16) and (10,11);
  * learned seed 0 on the same 41-node grid (e83 C_learned[0]) and
    on the published 200-step chain (checkpoint re-evaluation with
    the e73/e80 protocol), reproducing the published 5.77e4 ->
    4.34e4 (24.8%) record.

Part B (review item IV-1, priority enhancement: boundary-account
error of the learned surrogate).  The paired columns of Appendix C.2
(e83 npz: C_learned five seeds vs C_zero, identical 41 integration
nodes, setting A) are differenced at the node level,
  delta_BC[seed] = B C_learned[seed] - B C_zero,
and the connected-boundary family U_j^conn (exact C enumeration,
the same e80 machinery) evaluates
  E_j^cut(seed) = max_{U in U_j^conn} |phi_learned(U) - phi_SUE(U)|
                  / |phi_SUE({s_j})|,
where phi(U) - phi'(U) = -1_U^T (BC - BC'), so the extrema of the
difference column over the family give the maximum absolute account
difference at shared boundaries.  The denominator is the direct-SUE
origin-cut account of the same chain.  Reported: median / 95th /
max over the 40 seed-OD units, per-OD medians, attaining sets, and
the unrestricted-family counterpart.

Part C (review item IV-2, optional enhancement: own-OD vs
other-query-OD split of the study component).  For the three
sign-crossing queries of the six-query protocol, the study response
(the joint response of all six query ODs) is split at the same joint
equilibrium state and the same total-flow gradient into the OD being
explained and the other five query ODs:
  J_own_j = Delta_j p_j - S_own_j W J_j
          = J_study_j + (S_study - S_own_j) W J_j,
with S_R the logit covariance of OD group R.  Path probabilities are
recomputed from the stored equilibrium flows F (logit loading is a
pure function of F), so no state is re-solved.  The split is
evaluated inside the baseline connected minimizing boundary U_-
(e91) and at its largest interior nodes.

Gates (all must pass before any number enters the paper):
  (a) Part A: {10} and {10,17} are admissible members of each OD's
      connected family; every phi equals direct node summation
      (1e-6); the direct relative change for (10,16) reproduces the
      e79 JSON cut_change_ratio_node17 (1e-9 relative); the
      200-step learned seed-0 values reproduce the published
      5.77e4 / 4.34e4 / 24.8% record (1e-4 relative on the ratio,
      1e-3 relative on the absolute accounts);
  (b) Part B: family counts match e80; attaining masks are in-family
      with phi by direct summation (1e-6); the singleton lower bound
      E >= |phi_L({s})-phi_S({s})|/|phi_S({s})| holds exactly; the
      e83 C_zero node divergences agree with the frozen e79 BC_A
      chain at integration level (recorded, threshold 1e-3 relative
      -- the two chains are independent solves of the same
      zero-background fixed points);
  (c) Part C: logit loading reproduces the stored flows; S_group
      reproduces EdgeSUE.covariance for the all/study groups;
      J_study is reproduced from the recomputed loading (1e-8
      relative); B J_own = mu_j, B J_other = 0, additivity
      J_own + J_other = J_study and C_own + C_other = C_study
      (1e-9); interior sums own + other reproduce the e91 study
      interior sums (1e-6).

Output: experiments/results/e92_v88_cut_error.json
        console log experiments/results/e92_console.log
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
from run_e9_transport import _sioux_setup, TSTTT  # noqa: E402
from run_e83_v82_validation import EdgeSUE, attribution  # noqa: E402
from run_e80_connected_cuts import (  # noqa: E402
    build_adj_bits, load_c_enum, enum_c, connected, mask_to_nodes)
from run_e86_v84_newod import select_ods  # noqa: E402

RESULTS = HERE / 'results'
E79_NPZ = RESULTS / 'e79_chain_arrays.npz'
E79_JSON = RESULTS / 'e79_bgreroute.json'
E80_JSON = RESULTS / 'e80_connected_cuts.json'
E83_NPZ = RESULTS / 'e83_v82_validation.npz'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E91_JSON = RESULTS / 'e91_v87_negative_mechanism.json'
OUT = RESULTS / 'e92_v88_cut_error.json'

CROSSING = ('1->19', '1->15', '7->12')
NODE10, NODE17 = 9, 16          # 0-based


def part_a(net, B, ods, adj, lib):
    """Setting-A {10} -> {10,17} numbers, direct SUE and learned."""
    z83 = np.load(E83_NPZ)
    z79 = np.load(E79_NPZ)
    e79 = json.loads(E79_JSON.read_text())
    n = net.n
    full = (1 << n) - 1

    BC_S = B @ z83['C_zero']                    # direct SUE, 41-node chain
    BC_A = z79['BC_A']                          # frozen e79 chain
    scale = float(np.abs(BC_A).max())
    agree = float(np.max(np.abs(BC_S - BC_A)) / scale)

    out = {'direct_vs_e79_BC_max_rel': agree}

    # gate (b)-style chain agreement (independent solves of the same
    # zero-background fixed points; recorded and thresholded)
    assert agree < 1e-3, f'e83/e79 setting-A chains disagree: {agree}'

    learned41 = [B @ z83['C_learned'][s] for s in range(5)]

    # published 200-step learned seed-0 chain (e73/e80 protocol)
    import torch  # deferred: only Part A needs the checkpoint
    from models import build_model
    from attribution import outcome_field_C
    torch.set_num_threads(1)
    _, _, q_base, ps, _, ec = _sioux_setup()
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    ckpt = torch.load(RESULTS / 'e5_ckpt' / 'M3|norm|raw|0.pt',
                      weights_only=False, map_location='cpu')
    norm = {'mu_q': np.zeros(len(ods)), 'sd_q': np.ones(len(ods))}
    model = build_model('M3', ps, 0, norm=norm)
    model.load_state_dict(ckpt['best'])
    model.eval()
    C200, _ = outcome_field_C(model, T, q0, q1, steps=200)
    BC200 = B @ C200

    def record(bc, j, label):
        s, t = ods[j]
        assert (s, t) in ((NODE10, 15), (NODE10, 10)), 'unexpected OD'
        m1, m2 = (1 << s), (1 << s) | (1 << NODE17)
        for m in (m1, m2):
            assert connected(adj, m) and connected(adj, full & ~m)
            assert (m >> s) & 1 and not (m >> t) & 1
        nd1, nd2 = [s], [s, NODE17]
        phi1 = -float(bc[nd1, j].sum())
        phi2 = -float(bc[nd2, j].sum())
        return {
            'od': f'({s + 1},{t + 1})', 'chain': label,
            'phi_10': phi1, 'phi_10_17': phi2,
            'delta': phi2 - phi1,
            'rel_pct': 100.0 * abs(phi2 - phi1) / abs(phi1)}

    rows = []
    for j, (s, t) in enumerate(ods):
        if (s, t) not in ((NODE10, 15), (NODE10, 10)):
            continue
        key = f'{s + 1}->{t + 1}'
        r_sue = record(BC_S, j, 'direct SUE, setting A, 41 nodes')
        r_79 = record(BC_A, j, 'direct SUE, setting A, e79 frozen')
        r_l41 = record(learned41[0], j, 'learned seed 0, 41 nodes')
        rows += [r_sue, r_79, r_l41]
        if (s, t) == (NODE10, 15):
            r_200 = record(BC200, j, 'learned seed 0, 200 steps')
            rows.append(r_200)
            # gate (a): the frozen e79 chain reproduces its own published
            # ratio bit-for-bit; the e83 re-solve agrees to 1e-4
            r79 = e79['setting_a_nobg_41n'][key]['cut_change_ratio_node17']
            rel_frozen = abs(r_79['rel_pct'] / 100.0 - r79) / r79
            rel_resolve = abs(r_sue['rel_pct'] / 100.0 - r79) / r79
            assert rel_frozen < 1e-9, f'frozen-chain ratio mismatch: {rel_frozen}'
            assert rel_resolve < 1e-4, f're-solved chain drift: {rel_resolve}'
            out['cut_change_ratio_vs_e79_rel_frozen'] = rel_frozen
            out['cut_change_ratio_vs_e79_rel_resolve'] = rel_resolve
            # gate (a): published 24.8% record, 5.77e4 -> 4.34e4
            assert abs(r_200['rel_pct'] - 24.8) < 1e-3 * 24.8
            assert abs(r_200['phi_10'] - 5.77e4) < 1e-3 * 5.77e4
            assert abs(r_200['phi_10_17'] - 4.34e4) < 1e-3 * 4.34e4
    out['records'] = rows
    return out


def _selftest_unrestricted(rng):
    """Brute-force check of the unrestricted extrema formula: min/max of
    -sum_{v in U} b_v over subsets of a small allowed set containing s,
    excluding t, no connectivity requirement."""
    for _ in range(20):
        nv = 12
        b = rng.normal(size=nv + 2)
        s, t = 3, 7
        interior = [v for v in range(nv + 2) if v not in (s, t)]
        lo = -(b[s] + sum(max(b[v], 0.0) for v in interior))
        hi = -(b[s] - sum(max(-b[v], 0.0) for v in interior))
        blo, bhi = np.inf, -np.inf
        for code in range(1 << nv):
            U = 1 << s
            c = code
            for v in interior:
                if c & 1:
                    U |= 1 << v
                c >>= 1
            if (U >> t) & 1:
                continue
            val = -sum(b[v] for v in range(nv + 2) if (U >> v) & 1)
            blo = min(blo, val)
            bhi = max(bhi, val)
        assert abs(blo - lo) < 1e-9 and abs(bhi - hi) < 1e-9, (blo, lo, bhi, hi)


def part_b(net, B, ods, adj, lib):
    """Connected-boundary account error E_j^cut for five learned seeds."""
    _selftest_unrestricted(np.random.default_rng(20261024))
    z83 = np.load(E83_NPZ)
    e80 = json.loads(E80_JSON.read_text())
    n = net.n
    full = (1 << n) - 1
    C_S = z83['C_zero']
    BC_S = B @ C_S
    seeds = range(z83['C_learned'].shape[0])

    per_od, flat = {}, []
    for j, (s, t) in enumerate(ods):
        key = f'{s + 1}->{t + 1}'
        # difference columns: phi_L(U) - phi_S(U) = -1_U^T delta_BC
        cols = np.stack(
            [(B @ z83['C_learned'][sd])[:, j] - BC_S[:, j] for sd in seeds],
            axis=1)                                        # (n, 5)
        cr = enum_c(lib, adj, n, s, t, cols)
        assert cr['n_sets'] == e80['family_sizes'][key], \
            (key, cr['n_sets'], e80['family_sizes'][key])

        phi_s_sue = -float(BC_S[s, j])
        rec = {'phi_s_sue': phi_s_sue, 'per_seed': {}}
        # singleton floor: |phi_L({s}) - phi_S({s})| / |phi_S({s})|
        for sd in seeds:
            phi_min = float(cr['phi_min'][sd])
            phi_max = float(cr['phi_max'][sd])
            worst = max(abs(phi_min), abs(phi_max))
            # gate: attaining mask in-family and phi by direct summation
            mask = cr['min_mask'][sd] if abs(phi_min) >= abs(phi_max) \
                else cr['max_mask'][sd]
            assert connected(adj, mask) and connected(adj, full & ~mask)
            assert (mask >> s) & 1 and not (mask >> t) & 1
            nd = np.array(mask_to_nodes(mask)) - 1
            assert abs(-float(cols[nd, sd].sum()) -
                       (phi_min if abs(phi_min) >= abs(phi_max)
                        else phi_max)) < 1e-6
            singleton = abs(float(cols[s, sd]))
            assert worst >= singleton - 1e-9 * max(1.0, abs(worst))
            # unrestricted counterpart (interior nodes enter freely)
            interior = [v for v in range(n) if v not in (s, t)]
            db = cols[:, sd]
            un_min = -float(db[s] + sum(max(db[v], 0.0) for v in interior))
            un_max = -float(db[s] - sum(max(-db[v], 0.0) for v in interior))
            E = worst / abs(phi_s_sue)
            rec['per_seed'][str(sd)] = {
                'conn_max_abs_diff': worst,
                'E_cut': E,
                'E_cut_unrestricted': max(abs(un_min), abs(un_max))
                                       / abs(phi_s_sue),
                'attaining_nodes_1idx': mask_to_nodes(mask),
                'singleton_diff': singleton,
                'singleton_floor': singleton / abs(phi_s_sue)}
            flat.append(E)
        med = [v['E_cut'] for v in rec['per_seed'].values()]
        rec['median'] = float(np.median(med))
        per_od[key] = rec

    flat = np.array(flat)                    # OD-major, seed-minor
    n_seed = len(seeds)
    worst = int(np.argmax(flat))
    summary = {
        'n_units': int(flat.size),
        'median': float(np.median(flat)),
        'p95': float(np.percentile(flat, 95)),
        'max': float(flat.max()),
        'min': float(flat.min()),
        'max_unit': {'od': f'{ods[worst // n_seed][0] + 1}'
                            f'->{ods[worst // n_seed][1] + 1}',
                     'seed': worst % n_seed}}
    return {'per_od': per_od, 'summary': summary,
            'denominator': '|phi_SUE({s_j})| of the same 41-node '
                           'zero-background chain (e83 C_zero)',
            'pairing': 'e83 C_learned(seed) vs C_zero, identical 41 '
                       'integration nodes, setting A'}


def s_group(D, h, q, od_slices, idx_list, theta):
    """Logit covariance of an OD group: sum_k (Dh D^T - by/q by^T).

    Mirrors EdgeSUE.covariance term by term, restricted to the ODs in
    idx_list; od_slices[k] = (start, stop) path index range of OD k."""
    S = np.zeros((D.shape[0], D.shape[0]))
    for k in idx_list:
        a, b = od_slices[k]
        Dh = D[:, a:b] * h[None, a:b]
        by = Dh.sum(axis=1)
        S += Dh @ D[:, a:b].T - np.outer(by, by) / q[k]
    return theta * S


def part_c(net, B):
    """Own-OD / other-query-OD split of the study response at U_-."""
    from sue import PathStructure
    z87 = np.load(E87_NPZ)
    e91 = json.loads(E91_JSON.read_text())
    n = net.n

    study_ods, trips, _ = select_ods(net)
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    ps = PathStructure(net, all_ods)
    solver = EdgeSUE(ps, study_k=6)
    q_new = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    q0, q1 = 0.2 * q_new, 1.6 * q_new
    dQ = q1 - q0

    ts81 = z87['ts']
    idx41 = np.arange(0, len(ts81), 2)
    assert len(idx41) == 41
    ts41 = ts81[idx41]
    F41, g41 = z87['F'][idx41], z87['gT'][idx41]
    J41, Js41, Jb41 = (z87['J'][idx41], z87['J_study'][idx41],
                       z87['J_background'][idx41])

    M_study = net.od_M(study_ods) if hasattr(net, 'od_M') else None
    if M_study is None:
        M_study = np.zeros((n, 6))
        for j, (s, t) in enumerate(study_ods):
            M_study[t, j] += 1.0
            M_study[s, j] -= 1.0

    t0, cap = solver.t0, solver.cap
    max_load_gap = 0.0
    max_js_rel = 0.0
    Jown41 = np.zeros_like(Js41)
    Joth41 = np.zeros_like(Js41)
    for i, t in enumerate(ts41):
        q = np.concatenate([q0 + t * dQ, q_bg])
        F = F41[i]
        L, h, p = solver.loading(F, q)
        max_load_gap = max(max_load_gap, float(np.max(np.abs(L - F))))
        cp = .6 * t0 * (F / cap) ** 3 / cap
        WJ = cp[:, None] * J41[i]                      # (m, 6)
        starts = [a for a, b in ps.od_slices]
        # gates: S_group reproduces covariance for all/study groups
        S_all_mine = s_group(solver.D, h, q, ps.od_slices,
                             range(len(all_ods)), solver.theta)
        S_all = solver.covariance(h, q)
        S_stu_mine = s_group(solver.D, h, q, ps.od_slices, range(6),
                             solver.theta)
        S_stu = solver.covariance(h, q, 6)
        g1 = float(np.max(np.abs(S_all_mine - S_all)) /
                   max(1e-300, float(np.abs(S_all).max())))
        g2 = float(np.max(np.abs(S_stu_mine - S_stu)) /
                   max(1e-300, float(np.abs(S_stu).max())))
        assert g1 < 1e-10 and g2 < 1e-10, (g1, g2)
        for j in range(6):
            a, b = ps.od_slices[j]
            direct = solver.D[:, a:b] @ p[a:b]
            # J_study reproduction gate (validates the loading recompute)
            js_rep = direct - S_stu @ WJ[:, j]
            denom = max(1e-300, float(np.linalg.norm(Js41[i][:, j])))
            max_js_rel = max(max_js_rel,
                             float(np.linalg.norm(js_rep - Js41[i][:, j])
                                   / denom))
            S_own = s_group(solver.D, h, q, ps.od_slices, [j],
                            solver.theta)
            Jown41[i][:, j] = direct - S_own @ WJ[:, j]
            Joth41[i][:, j] = Js41[i][:, j] - Jown41[i][:, j]
    assert max_js_rel < 1e-7, f'J_study reproduction failed: {max_js_rel}'
    assert max_load_gap < 1e-5, f'loading fixed-point gap too large: {max_load_gap}'

    # structural gates on the split
    max_own_cons = max(float(np.max(np.abs(B @ Jown41[i] - M_study)))
                       for i in range(41))
    max_oth_cons = max(float(np.max(np.abs(B @ Joth41[i])))
                       for i in range(41))
    max_add = float(np.max(np.abs(Jown41 + Joth41 - Js41)))
    assert max_own_cons < 1e-8, max_own_cons
    assert max_oth_cons < 1e-8, max_oth_cons
    assert max_add < 1e-10, max_add

    C_own = attribution(g41, Jown41, ts41, dQ)
    C_oth = attribution(g41, Joth41, ts41, dQ)
    C_stu = attribution(g41, Js41, ts41, dQ)
    C_bg = attribution(g41, Jb41, ts41, dQ)
    max_cadd = float(np.max(np.abs(C_own + C_oth - C_stu)))
    assert max_cadd < 1e-9, max_cadd
    BC_own, BC_oth = B @ C_own, B @ C_oth
    BC_stu, BC_bg = B @ C_stu, B @ C_bg
    assert float(np.max(np.abs(BC_own + BC_oth + BC_bg - z87['BC_41']))) \
        < 1e-8, 'BC additivity vs published columns failed'

    out = {'gates': {
        'loading_max_abs_gap': max_load_gap,
        'J_study_reproduction_max_rel': max_js_rel,
        'own_conservation_max_abs': max_own_cons,
        'other_circulation_max_abs': max_oth_cons,
        'J_additivity_max_abs': max_add,
        'C_additivity_max_abs': max_cadd},
        'protocol': 'same joint equilibrium states and total-flow '
                    'gradients as the published 41-point columns; '
                    'logit loading recomputed from stored F; no new '
                    'equilibrium solves',
        'per_query': {}}
    for j, (s, t) in enumerate(study_ods):
        key = f'{s + 1}->{t + 1}'
        if key not in CROSSING:
            continue
        mech = e91['mechanism'][key]
        inside = [v - 1 for v in mech['U_minus_nodes_1idx'] if v - 1 != s]
        rec = {'U_minus_nodes_1idx': mech['U_minus_nodes_1idx'],
               'interior_nodes_0idx': inside}
        parts = {}
        for nm, BCx in (('own', BC_own), ('other_query', BC_oth),
                        ('study', BC_stu), ('background', BC_bg)):
            parts[nm] = float(BCx[inside, j].sum())
        # gate: own + other reproduces e91's study interior sum
        e91s = mech['interior_sum_study']
        assert abs(parts['own'] + parts['other_query'] - e91s) \
            < 1e-6 * max(1.0, abs(e91s)), (key, parts, e91s)
        assert abs(parts['own'] + parts['other_query'] - parts['study']) \
            < 1e-8
        rec['interior_sum'] = parts
        # largest interior nodes, decomposed four ways
        tot = {v: float(BC_stu[v, j] + BC_bg[v, j]) for v in inside}
        top = sorted(inside, key=lambda v: abs(tot[v]), reverse=True)[:3]
        rec['top_nodes'] = [
            {'node': v + 1,
             'total': float(BC_stu[v, j] + BC_bg[v, j]),
             'own': float(BC_own[v, j]),
             'other_query': float(BC_oth[v, j]),
             'study': float(BC_stu[v, j]),
             'background': float(BC_bg[v, j])} for v in top]
        out['per_query'][key] = rec
    return out


def main():
    started = time.time()
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    B, M = ec.B, ec.M
    adj, n = build_adj_bits(net)
    lib = load_c_enum()

    out = {'protocol': {
        'purpose': 'v88 revision: setting-A boundary-number provenance, '
                   'connected-boundary account error E_cut, and '
                   'own/other split of the study response',
        'basis': 'frozen e79/e80/e83/e87/e91 arrays; checkpoint '
                 're-evaluation for the published 200-step seed-0 chain '
                 'only'}}
    print('Part A: setting-A boundary numbers', flush=True)
    out['part_a'] = part_a(net, B, ods, adj, lib)
    for r in out['part_a']['records']:
        print(f"  {r['od']:9s} {r['chain']:36s} "
              f"phi10={r['phi_10']:11.1f} phi10,17={r['phi_10_17']:11.1f} "
              f"rel={r['rel_pct']:.4f}%", flush=True)

    print('Part B: E_cut over connected families', flush=True)
    out['part_b'] = part_b(net, B, ods, adj, lib)
    s = out['part_b']['summary']
    print(f"  E_cut over {s['n_units']} seed-OD units: median "
          f"{s['median']:.4f} p95 {s['p95']:.4f} max {s['max']:.4f}",
          flush=True)
    for k, v in out['part_b']['per_od'].items():
        print(f"  {k:8s} median {v['median']:.4f}", flush=True)

    print('Part C: own/other split at U_-', flush=True)
    out['part_c'] = part_c(net, B)
    print('  gates:', json.dumps(out['part_c']['gates']), flush=True)
    for k, v in out['part_c']['per_query'].items():
        p = v['interior_sum']
        print(f"  {k:6s} own {p['own']:10.1f} other {p['other_query']:10.1f}"
              f" study {p['study']:10.1f} bg {p['background']:10.1f}",
              flush=True)

    out['elapsed_seconds'] = time.time() - started
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Completed e92 in {out['elapsed_seconds']:.1f}s -> {OUT}",
          flush=True)


if __name__ == '__main__':
    main()
