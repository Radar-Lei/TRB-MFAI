"""E145 (v111): learned spatial-profile repair case figure.

External review of v110 (Section 4.3): the learning section reports only
aggregate errors; one pre-specified case should show where the flow-only
model misexplains the boundary profile and what response supervision
repairs.

Pre-specified selection rule (no test-driven choice):
 1. path: among the 96 test paths, the one whose paired family-error
    improvement (E_model under flow supervision minus E_model under
    response supervision, averaged over active queries and seeds within
    the path, from the frozen E127 rows) is closest to the median
    improvement over all 96 paths (ties: smaller path index);
 2. query: that path's active query with the largest flow-only E_model;
 3. chain: the projection-direction chain of the query's geometric
    family containing the largest |a_flow - a_ref| boundary value;
 4. profiles: entrywise five-seed medians of the attribution columns
    (reference from the frozen reference arrays; flow and response from
    the frozen selected checkpoints).

Outputs: figures/fig_v111_repair.{pdf,png} and results.json with the
displayed numbers (per-boundary accounts, segment summaries, error
columns, boundary composition, side statistics).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path

import numpy as np
import torch

import run_e111_v102_pairs as p
import run_e127_v108_totalerror as e127
from fam_v108 import direction_levels, query_families
from run_e83_v82_validation import attribution

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'results/e145_v111_repair'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'
E127 = HERE / 'results/e127_v108_totalerror'
OLD = HERE / 'results/v104_learning'
TOL = 0.05


def select_case():
    rows = json.loads((E127 / 'rows.json').read_text())
    paths = sorted({r['path'] for r in rows})
    improvement = {}
    for pi in paths:
        flow = [r['E_model'] for r in rows if r['path'] == pi and r['method'] == 'flow']
        resp = [r['E_model'] for r in rows if r['path'] == pi and r['method'] == 'response']
        improvement[pi] = float(np.mean(flow) - np.mean(resp))
    vals = np.array([improvement[pi] for pi in paths])
    med = float(np.median(vals))
    pi_star = min(paths, key=lambda pi: (abs(improvement[pi] - med), pi))
    # query: largest flow-only E_model on that path (mean over seeds)
    q_rows = [r for r in rows if r['path'] == pi_star and r['method'] == 'flow']
    j_star = max({r['query'] for r in q_rows},
                 key=lambda j: (np.mean([r['E_model'] for r in q_rows
                                         if r['query'] == j]), -j))
    return pi_star, j_star, improvement, med


def main():
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    pi_star, j_star, improvement, med = select_case()
    s, t = ods[j_star]
    print(f'selected path {pi_star} (median improvement {med:.3f}, '
          f'this path {improvement[pi_star]:.3f}), query {j_star} = OD ({s},{t})',
          flush=True)

    # reference columns for the 96 paths
    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jref = np.concatenate([d_old['J'], d_new['J']])
    ts = d_old['ts']
    Cref_all = np.concatenate([
        np.array([attribution(np.array([T.grad(f) for f in Fref[pi]]), Jref[pi], ts,
                              Q[pi, -1] - Q[pi, 0]) for pi in range(24)]),
        np.load(E119 / 'reference_columns.npz')['C']])

    # learned columns: five seeds per method, entrywise median
    selection = {(x['method'], x['seed']): x['step']
                 for x in json.loads((OLD / 'results.json').read_text())['selection']}
    columns = {}
    flows = {}
    for method in ('flow', 'response'):
        Cs = []
        for seed in range(5):
            m = p.base.old.build_fullnet_model(ps, qbg, seed)
            step = selection[(method, seed)]
            ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else \
                E111 / f'{method}_{seed}.pt'
            m.load_state_dict(torch.load(ck, weights_only=False)['best'])
            f, _, C = e127.learned_columns(m, Q[pi_star], ts, T)
            Cs.append(C[:, j_star])
            if seed == 2:
                flows[method] = f
        columns[method] = np.median(np.stack(Cs), axis=0)
    cS = Cref_all[pi_star][:, j_star]

    # geometric family chains for the query
    chains = direction_levels(net=n)
    fams = query_families(s, t, chains=chains, net=n)
    geo = [np.array([[v in U for v in range(n.n)] for U in f['masks']]) @ B
           for f in fams]

    def accounts(c):
        return [-(g @ c) for g in geo]

    aS = accounts(cS)
    aflow = accounts(columns['flow'])
    aresp = accounts(columns['response'])
    # display chain: the direction with the most boundaries (geometric,
    # attribution-independent; ties broken by smaller angle)
    k_star = max(range(len(fams)),
                 key=lambda k: (len(fams[k]['masks']), -fams[k]['angle']))
    fam = fams[k_star]
    levels = fam['levels']
    aS_c = aS[k_star]
    af_c = aflow[k_star]
    ar_c = aresp[k_star]
    # family-wide worst flow-only boundary (for the map panel and the
    # error-panel reference line)
    err_fam = [[float(abs(x - y)) for x, y in zip(al, as_)]
               for al, as_ in zip(aflow, aS)]
    err_fam_response = [[float(abs(x - y)) for x, y in zip(al, as_)]
                        for al, as_ in zip(aresp, aS)]
    k_fam, ib_fam = max(((k, int(np.argmax(e))) for k, e in enumerate(err_fam)),
                        key=lambda ki: err_fam[ki[0]][ki[1]])

    # segment summaries at the family tolerance of each model's own family max
    segs = {}
    for name, a_all in (('reference', aS), ('flow', aflow), ('response', aresp)):
        eps = TOL * max(np.max(np.abs(a)) for a in a_all)
        sg = e127.greedy_segments(a_all[k_star], eps)
        segs[name] = {'eps': float(eps), 'n_segments_family': int(sum(
            len(e127.greedy_segments(a, eps)) for a in a_all)),
            'chain_segments': [(int(i), int(j)) for i, j in sg],
            'chain_summary': e127.summarize(a_all[k_star], sg).tolist()}

    # case-level side statistics computed from the SAME median columns the
    # figure displays (family-wide E_model/E_summary/E_total, segment count
    # summed over all chains of the family)
    side = {}
    for name, a_all in (('flow', aflow), ('response', aresp)):
        fam_err = max(np.max(np.abs(al - as_)) for al, as_ in zip(a_all, aS))
        eps = TOL * max(np.max(np.abs(al)) for al in a_all)
        hats, seg_tot = [], 0
        for al in a_all:
            sg = e127.greedy_segments(al, eps)
            seg_tot += len(sg)
            hats.append(e127.summarize(al, sg))
        side[name] = {'E_model': float(fam_err),
                      'E_summary': float(max(np.max(np.abs(h - al))
                                             for h, al in zip(hats, a_all))),
                      'E_total': float(max(np.max(np.abs(h - as_))
                                           for h, as_ in zip(hats, aS))),
                      'segments': int(seg_tot)}

    # per-boundary errors along the chain and the repair interval
    ef = np.abs(af_c - aS_c)
    er = np.abs(ar_c - aS_c)
    ib = int(np.argmax(ef))                    # worst flow-only boundary on the chain
    gain = ef - er
    i_gain = int(np.argmax(gain))              # largest per-boundary repair
    fam_max_err = err_fam[k_fam][ib_fam]

    # reference account split at the worst boundary: free-flow vs congestion
    # weight components of the reference attribution column
    Fp = Fref[pi_star]
    gff = np.array([n.free_flow_time[e] for e in range(n.m)])[:, None] * np.ones_like(
        Jref[pi_star][:, :, j_star:j_star + 1])
    # integrated free-flow part: dF/dlambda weighted by t0
    dQ = Q[pi_star, -1, j_star] - Q[pi_star, 0, j_star]
    c_ff = attribution(np.repeat(np.array([[n.free_flow_time[e] for e in range(n.m)]]
                                          for _ in range(len(ts))), 1)[:, :m0]
                       if False else np.tile(n.free_flow_time, (len(ts), 1)),
                       Jref[pi_star][:, :, j_star:j_star + 1], ts, np.array([dQ]))[:, 0]
    c_cong = cS - c_ff
    U = fams[k_fam]['masks'][ib_fam]
    Uv = np.zeros(n.n)
    Uv[list(U)] = 1
    # oriented crossing account contributions: -1_U^T B c  per component
    orient = -(geo[k_fam][ib_fam])
    ff_b = float(orient @ c_ff)
    cg_b = float(orient @ c_cong)
    closure = abs(ff_b + cg_b - float(orient @ cS))

    # per-edge signed contributions at the worst boundary (for the map
    # panel); the sign already encodes leaving(+)/entering(-), so arrow
    # color and linestyle follow the value directly
    contrib = orient * cS
    crossing = [(e, float(contrib[e])) for e in range(n.m) if abs(orient[e]) > 0]

    result = {
        'selection_rule': {
            'path': 'median paired family-error improvement among the 96 paths',
            'query': 'largest flow-only E_model on the selected path',
            'chain': ('display: direction with the most boundaries (geometric, '
                      'attribution-independent); the family-wide worst flow-only '
                      'boundary is marked in the error panel and mapped in (c)'),
            'profiles': 'entrywise five-seed medians of attribution columns'},
        'path': int(pi_star),
        'group': int(np.concatenate([d_old['group'], d_new['group']])[pi_star]),
        'query': int(j_star),
        'od': [int(s) + 1, int(t) + 1],
        'od_zero_based_internal': [int(s), int(t)],
        'median_improvement_all_paths': med,
        'this_path_improvement': improvement[pi_star],
        'chain_angle': fam['angle'], 'n_directions': len(fams),
        'n_levels': len(levels),
        'side': side,
        'chain_levels': [float(v) for v in levels],
        'accounts_reference': aS_c.tolist(),
        'accounts_flow': af_c.tolist(),
        'accounts_response': ar_c.tolist(),
        'error_flow': ef.tolist(), 'error_response': er.tolist(),
        'worst_boundary_index': ib,
        'worst_boundary_level': float(levels[ib]),
        'family_worst': {'angle': fams[k_fam]['angle'],
                         'index': ib_fam,
                         'flow_error': fam_max_err,
                         'response_error': float(err_fam_response[k_fam][ib_fam])},
        'largest_repair_index': i_gain,
        'segments': segs,
        'worst_boundary_split': {'free_flow': ff_b, 'congestion': cg_b,
                                 'total': float(orient @ cS), 'closure': float(closure)},
        'crossing_contributions': crossing,
        'worst_boundary_nodes': sorted(int(v) + 1 for v in U),
    }
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))

    print(json.dumps({'worst_boundary_split': result['worst_boundary_split'],
                      'segments': {k: v['n_segments_family'] for k, v in segs.items()},
                      'side': result['side']}, indent=2))


if __name__ == '__main__':
    main()
