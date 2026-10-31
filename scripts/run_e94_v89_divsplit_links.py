"""E94 (v89, external review round 61, 87/100 targeted revision).

Two frozen-array analyses; no retraining, no new equilibrium solves.

Part A (review item I-1, first priority: positive-coverage vs net
divergence).  The v88 text reports, for the three sign-crossing
queries, that the connected minimizing region U_- "captures 95.5% /
94.8% / 85.7% of the query's total positive interior divergence",
immediately beside a table whose Interior column is the NET divergence
sum over U_- minus the origin.  These are different quantities, and a
reader can divide the net sum by the total positive mass (73.1%,
58.5%, 49.5%) and conclude the printed percentages are wrong.  This
part recomputes, from the raw published 41-point node-divergence
columns, the full three-way split

  positive captured inside U_-   sum_{v in U_-^o} (b_v)_+
  negative captured inside U_-   sum_{v in U_-^o} (-b_v)_+
  net interior sum               sum_{v in U_-^o}  b_v

together with the total positive interior mass, both shares
(positive share = the printed percentages; net share = what a reader
dividing the table column by the total obtains), and verifies the
identity implied by Eq. (9) (eq:cutextrema),

  phi_j(U) - phi_j^min
    = sum_{v in I_j \\ U} (b_v)_+  +  sum_{v in I_j cap U} (-b_v)_+,
  I_j = V \\ {s_j, t_j},

for every connected U of each of the three queries at BOTH extrema,
where phi_j^min = phi_j({s_j}) - sum_{I_j} (b_v)_+ is the
unrestricted minimum.  This identity is what the revision will print:
the gap between a connected region and the unrestricted minimum
decomposes exactly into (i) positive divergence the region does not
contain and (ii) negative divergence the region does contain.

Part B (review item III-1, third priority: sign-flip mechanism down to
links).  For OD (1,19) at its published connected minimizer U_-, the
account is decomposed over the crossing links,

  phi_j(U_-) = sum_{e in delta^+(U)} C~_{e,j} - sum_{e in delta^-(U)} C~_{e,j},

and each crossing link's total attribution is further split into
own-OD / other-query-OD / background components with the e92 Part-C
machinery (path probabilities recomputed from the stored equilibrium
flows; no state re-solved).  Reported: every crossing link with its
signed contribution and its three components; the top negative and
positive contributors; and the aggregate out-crossing vs in-crossing
sums, so the mechanism (large positive attributions on entering links
vs negative attributions on leaving links vs both) is read off the
arrays rather than inferred from the node summary.

Gates (all must pass before any number enters the paper):
  (a) Part A: phi(U_-) equals -1_U^T BC within 1e-6 for all three
      queries; every e91 number (phi_s, phi_U_-, positive/negative
      mass inside, positive mass all, interior sums) reproduces
      within 1e-6; the identity holds within 1e-8 relative at U_- and
      U_+ for all three queries AND at every connected mask of a
      500-mask random subsample per query (max |LHS-RHS| <= 1e-6
      absolute against the scale of the positive mass);
  (b) Part B: the crossing-link sum reproduces phi(U_-) within 1e-6;
      loading and J_study reproduction gates as in e92 Part C
      (1e-5 / 1e-7); own/other conservation B J_own = mu_j,
      B J_other = 0 within 1e-8; C additivity own+other+background =
      published column within 1e-8; every reported link crosses the
      boundary by construction.

Output: experiments/results/e94_v89_divsplit_links.json
        console log experiments/results/e94_console.log
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
from run_e83_v82_validation import EdgeSUE, attribution  # noqa: E402
from run_e86_v84_newod import select_ods  # noqa: E402
from run_e80_connected_cuts import (build_adj_bits, load_c_enum, enum_c,  # noqa: E402
                                    connected, mask_to_nodes)
from run_e92_v88_cut_error import s_group  # noqa: E402
from sue import PathStructure  # noqa: E402

RESULTS = HERE / 'results'
E87_NPZ = RESULTS / 'e87_v85_newod_refine.npz'
E91_JSON = RESULTS / 'e91_v87_negative_mechanism.json'
OUT = RESULTS / 'e94_v89_divsplit_links.json'

CROSSING = ('1->19', '1->15', '7->12')
GATE_PHI = 1e-6
GATE_E91 = 1e-6
GATE_REL = 1e-8


def part_a(net, B):
    """Three-way split and the Eq.-(9) identity at the extrema."""
    z87 = np.load(E87_NPZ)
    e91 = json.loads(E91_JSON.read_text())
    n = net.n
    BC = z87['BC_41']
    ts81 = z87['ts']
    idx41 = np.arange(0, len(ts81), 2)
    assert len(idx41) == 41
    study_ods, _, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]

    adj, _ = build_adj_bits(net)
    lib = load_c_enum()
    full = (1 << n) - 1

    out = {}
    id_gates = {'max_abs_dev': 0.0, 'max_rel_dev': 0.0,
                'n_random_masks_checked': 0}
    rng = np.random.default_rng(20261025)
    for key in CROSSING:
        j = keys.index(key)
        s, t = study_ods[j]
        m = e91['mechanism'][key]
        b = BC[:, j]
        interior = np.array([v for v in range(n) if v not in (s, t)])
        U = np.array(m['U_minus_nodes_1idx']) - 1     # 0-based, incl. s
        Uo = np.array([v for v in U if v in set(interior.tolist())])

        pos_all = float(np.maximum(b[interior], 0.0).sum())
        pos_in = float(np.maximum(b[Uo], 0.0).sum())
        neg_in = float(np.maximum(-b[Uo], 0.0).sum())
        net_in = float(b[Uo].sum())
        phi_s = float(-b[s])
        phi_U = float(-b[U].sum())
        phi_min = phi_s - pos_all

        # e91 cross-checks
        for mine, theirs in ((phi_s, m['phi_s']), (phi_U, m['phi_U_minus']),
                             (net_in, m['interior_divergence_sum']),
                             (pos_in, m['positive_mass_inside']),
                             (pos_all, m['positive_mass_all_interior']),
                             (neg_in, -m['negative_mass_inside'])):
            assert abs(mine - theirs) < GATE_E91, (key, mine, theirs)

        def identity_dev(nodes_U):
            Uo_ = np.array([v for v in nodes_U
                            if v in set(interior.tolist())])
            lhs = (-b[nodes_U].sum()) - phi_min
            missed_pos = pos_all - float(np.maximum(b[Uo_], 0.0).sum())
            captured_neg = float(np.maximum(-b[Uo_], 0.0).sum())
            return lhs, missed_pos, captured_neg, lhs - (missed_pos + captured_neg)

        # at U_- and U_+
        Up = np.array(m['U_plus_nodes_1idx']) - 1
        devs = []
        for nodes_U in (U, Up):
            lhs, mp, cn, dev = identity_dev(nodes_U)
            devs.append(dev)
            id_gates['max_abs_dev'] = max(id_gates['max_abs_dev'], abs(dev))
            id_gates['max_rel_dev'] = max(
                id_gates['max_rel_dev'],
                abs(dev) / max(1e-300, pos_all))
        # at random connected family members (connectivity-aware family)
        cr = enum_c(lib, adj, n, s, t, BC[:, j:j + 1])
        fam = []
        code = cr['first_mask'][0] if 'first_mask' in cr else None
        # walk the family via repeated enumeration interface: enum_c returns
        # extrema only, so instead enumerate admissible masks around random
        # node subsets and keep the connected ones (bounded effort, gate-only)
        tries = 0
        while len(fam) < 500 and tries < 20000:
            tries += 1
            k = rng.integers(1, n - 1)
            pick = rng.permutation(interior)[:k]
            mask = 1 << s
            for v in pick:
                mask |= 1 << int(v)
            if (mask >> t) & 1:
                continue
            if not (connected(adj, mask) and connected(adj, full & ~mask)):
                continue
            fam.append(mask)
        for mask in fam:
            nodes_U = np.array(mask_to_nodes(mask)) - 1
            _, _, _, dev = identity_dev(nodes_U)
            id_gates['max_abs_dev'] = max(id_gates['max_abs_dev'], abs(dev))
            id_gates['max_rel_dev'] = max(
                id_gates['max_rel_dev'], abs(dev) / max(1e-300, pos_all))
        id_gates['n_random_masks_checked'] += len(fam)
        assert max(abs(d) for d in devs) < 1e-6, (key, devs)
        # Both gates must hold independently (AND, not OR): the identity
        # deviation is small in absolute terms and relative to the
        # positive mass it decomposes.
        assert id_gates['max_abs_dev'] < 1e-6, (key, id_gates)
        assert id_gates['max_rel_dev'] < GATE_REL, (key, id_gates)

        out[key] = {
            'phi_s': phi_s, 'phi_U_minus': phi_U, 'phi_min_unrestricted': phi_min,
            'interior_nodes_in_U_minus': len(Uo),
            'positive_inside': pos_in, 'negative_inside_abs': neg_in,
            'net_inside': net_in,
            'positive_all_interior': pos_all,
            'missed_positive': pos_all - pos_in,
            'positive_share': pos_in / pos_all,
            'net_share_of_positive': net_in / pos_all,
            'phi_U_minus_minus_phi_min': phi_U - phi_min,
            'identity_at_U_minus': {
                'missed_positive': pos_all - pos_in,
                'captured_negative': neg_in,
                'sum': pos_all - pos_in + neg_in,
                'dev': phi_U - phi_min - (pos_all - pos_in + neg_in)},
        }
        print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in out[key].items()
                               if k != 'identity_at_U_minus'}), flush=True)
    out['identity_gates'] = id_gates
    assert id_gates['n_random_masks_checked'] >= 1000
    return out


def part_b(net, B):
    """(1,19) cross-boundary link decomposition at U_-."""
    z87 = np.load(E87_NPZ)
    e91 = json.loads(E91_JSON.read_text())
    n, edges = net.n, net.edges
    ts81 = z87['ts']
    idx41 = np.arange(0, len(ts81), 2)
    ts41, F41, g41 = ts81[idx41], z87['F'][idx41], z87['gT'][idx41]
    J41, Js41, Jb41 = (z87['J'][idx41], z87['J_study'][idx41],
                       z87['J_background'][idx41])
    C_tot = z87['C_41']
    BC = z87['BC_41']

    study_ods, trips, _ = select_ods(net)
    keys = [f'{s + 1}->{t + 1}' for s, t in study_ods]
    j = keys.index('1->19')
    s, t = study_ods[j]

    U = set(v - 1 for v in e91['mechanism']['1->19']['U_minus_nodes_1idx'])
    assert s in U and t not in U

    # --- own / other split of J (e92 Part-C machinery, no re-solve) ---
    bg_ods = [od for od, d in trips.items() if d > 0 and od not in study_ods]
    all_ods = list(study_ods) + bg_ods
    ps = PathStructure(net, all_ods)
    solver = EdgeSUE(ps, study_k=6)
    q_new = np.array([trips[od] for od in study_ods])
    q_bg = np.array([trips[od] for od in bg_ods])
    q0, q1 = 0.2 * q_new, 1.6 * q_new
    dQ = q1 - q0

    M_study = net.od_M(study_ods)
    t0, cap = solver.t0, solver.cap
    max_load_gap = max_js = 0.0
    Jown41 = np.zeros_like(Js41)
    Joth41 = np.zeros_like(Js41)
    for i, lam in enumerate(ts41):
        q = np.concatenate([q0 + lam * dQ, q_bg])
        F = F41[i]
        L, h, p = solver.loading(F, q)
        max_load_gap = max(max_load_gap, float(np.max(np.abs(L - F))))
        cp = .6 * t0 * (F / cap) ** 3 / cap
        WJ = cp[:, None] * J41[i]
        S_stu = solver.covariance(h, q, 6)
        for jj in range(6):
            a, b_ = ps.od_slices[jj]
            direct = solver.D[:, a:b_] @ p[a:b_]
            if jj == j:
                js_rep = direct - S_stu @ WJ[:, j]
                max_js = max(max_js, float(
                    np.linalg.norm(js_rep - Js41[i][:, j]) /
                    max(1e-300, np.linalg.norm(Js41[i][:, j]))))
            S_own = s_group(solver.D, h, q, ps.od_slices, [jj], solver.theta)
            Jown41[i][:, jj] = direct - S_own @ WJ[:, jj]
            Joth41[i][:, jj] = Js41[i][:, jj] - Jown41[i][:, jj]
    assert max_load_gap < 1e-5, max_load_gap
    assert max_js < 1e-7, max_js
    assert float(np.max(np.abs(B @ Jown41 - M_study))) < 1e-8
    assert float(np.max(np.abs(B @ Joth41))) < 1e-8

    C_own = attribution(g41, Jown41, ts41, dQ)
    C_oth = attribution(g41, Joth41, ts41, dQ)
    C_bg = attribution(g41, Jb41, ts41, dQ)
    assert float(np.max(np.abs(C_own + C_oth + C_bg - C_tot))) < 1e-8

    # --- crossing links of U_- ---
    rows = []
    sum_check = 0.0
    for e, (u, v) in enumerate(edges):
        out_c, in_c = u in U, v in U
        if out_c == in_c:
            continue
        sign = 1.0 if out_c else -1.0       # delta^+ contributes +, delta^- -
        contrib = sign * C_tot[e, j]
        sum_check += contrib
        rows.append({
            'link_1idx': [u + 1, v + 1],
            'direction': 'out' if out_c else 'in',
            'signed_contribution': float(contrib),
            'C_total': float(C_tot[e, j]),
            'C_own': float(C_own[e, j]),
            'C_other_query': float(C_oth[e, j]),
            'C_background': float(C_bg[e, j]),
        })
    phi_U = float(-BC[list(U), j].sum())
    assert abs(sum_check - phi_U) < GATE_PHI, (sum_check, phi_U)

    rows_by_neg = sorted(rows, key=lambda r: r['signed_contribution'])
    rows_by_pos = sorted(rows, key=lambda r: -r['signed_contribution'])
    out_sum = sum(r['signed_contribution'] for r in rows if r['direction'] == 'out')
    in_sum = sum(r['signed_contribution'] for r in rows if r['direction'] == 'in')

    return {
        'od': '(1,19)', 'U_minus_1idx': sorted(v + 1 for v in U),
        'phi_U_minus': phi_U,
        'n_crossing_links': len(rows),
        'out_crossing_sum': out_sum, 'in_crossing_sum': in_sum,
        'sum_check_dev': sum_check - phi_U,
        'all_links': rows,
        'most_negative_links': rows_by_neg[:5],
        'most_positive_links': rows_by_pos[:3],
        'gates': {
            'loading_max_abs_gap': max_load_gap,
            'J_study_reproduction_max_rel': max_js,
            'own_conservation_max_abs': float(np.max(np.abs(B @ Jown41 - M_study))),
            'other_circulation_max_abs': float(np.max(np.abs(B @ Joth41))),
            'C_additivity_max_abs': float(np.max(np.abs(C_own + C_oth + C_bg - C_tot))),
            'crossing_sum_vs_phi_dev': abs(sum_check - phi_U),
        },
    }


def main():
    started = time.time()
    net = load_sioux_falls()
    B = net.B
    out = {'protocol': {
        'purpose': 'v89 review items I-1 (positive vs net divergence) and '
                   'III-1 (cross-boundary link mechanism at U_- for (1,19)); '
                   'frozen e87 41-point published columns, no new solves',
        'basis': 'e87_v85_newod_refine.npz C_41/BC_41 (81-chain even nodes), '
                 'e91 U_-/U_+ node lists',
    }}
    out['part_a_divergence_split'] = part_a(net, B)
    out['part_b_cross_links'] = part_b(net, B)
    out['elapsed_seconds'] = time.time() - started
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Completed e94 in {out['elapsed_seconds']:.1f}s -> {OUT}",
          flush=True)


if __name__ == '__main__':
    main()
