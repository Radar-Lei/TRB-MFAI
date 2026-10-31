"""E127 (v108): total explanation error of learned segmented summaries and
the response/weight error decomposition on the 96 test demand paths.

External review of v107 items 3.2/3.4: the compression error (Section 5)
and the learned-reference error (Section 7) are verified separately, but
a reader uses the learned model's *segmented* summary.  For every test
path, active query, method (flow, response) and frozen seed this
experiment computes, over the direction-grouped Sioux Falls boundary
families of e126,

    E_model   = max_U |a^L(U) - a^S(U)|,
    E_summary = max_U |hat a^L(U) - a^L(U)|   (hat a^L = min-segment
                greedy summary of the learned profile at the family
                tolerance eps^L = 5% of the largest learned account),
    E_total   = max_U |hat a^L(U) - a^S(U)|,

with E_total <= E_summary + E_model as the total error budget, plus two
baselines (single common value; equal-count segments with the same
per-direction counts).  It also recomputes the response-vs-weight
decomposition of the column error,

    e_j = attribution(g^L, J^L_j - J^S_j) + attribution(g^L - g^S, J^S_j),

on the same 96 paths (replacing the 24-path decomposition), and gates
E_model against the stored per-row E_geo of e122 (same family union) and
the segmentation guarantee E_summary <= eps^L.

No model is retrained; the v104 selection stays frozen exactly as in
e119/e122.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_e111_v102_pairs as p
from fam_v108 import direction_levels, query_families
from run_e83_v82_validation import attribution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e127_v108_totalerror'
OUT.mkdir(exist_ok=True)
OLD = ROOT / 'experiments/results/v104_learning'
E111 = ROOT / 'experiments/results/e111_v102'
E119 = ROOT / 'experiments/results/e119_v106_expanded_paths'
E122 = ROOT / 'experiments/results/e122_v107_boundchain'
METHODS = ('flow', 'response')
TOL = 0.05


def greedy_segments(a, eps):
    segs = []
    i = 0
    n = len(a)
    while i < n:
        j = i
        lo = hi = a[i]
        while j + 1 < n:
            nlo, nhi = min(lo, a[j + 1]), max(hi, a[j + 1])
            if nhi - nlo > 2 * eps or (nlo < 0 < nhi):
                break
            lo, hi = nlo, nhi
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def summarize(a, segs):
    out = np.empty_like(a)
    for i, j in segs:
        s = a[i:j + 1]
        out[i:j + 1] = (s.max() + s.min()) / 2
    return out


def learned_columns(m, Q, ts, T):
    q = torch.tensor(Q)
    f = m(q).detach().numpy()
    js = []
    for j in range(6):
        v = torch.zeros_like(q)
        v[:, j] = 1
        _, jv = torch.func.jvp(m, (q,), (v,))
        js.append(jv.detach().numpy())
    jac = np.stack(js, axis=2)
    C = attribution(np.array([T.grad(f0) for f0 in f]), jac, ts, Q[-1] - Q[0])
    return f, jac, C


def dedup_indices(ods, chains, net):
    """Per query: first-occurrence indices of distinct direction chains."""
    out = {}
    for j, (s, t) in enumerate(ods):
        fams = query_families(s, t, chains=chains, net=net)
        seen, keep = set(), []
        for i, f in enumerate(fams):
            key = tuple(tuple(U) for U in f['masks'])
            if key not in seen:
                seen.add(key)
                keep.append(i)
        out[j] = {'directions': len(fams), 'distinct_chains': len(keep),
                  'first_indices': keep}
    return out


def aggregate(rows, dedup_idx):
    """Paper aggregations of the frozen per-row results: absolute path
    medians (historical) and the unified per-query normalization of
    tab:totalerror (per-row E / fam_scale_reference, mean within path,
    median over the 96 paths), with segment counts over distinct
    direction chains."""
    METRICS = ('E_model', 'E_summary', 'E_total', 'E_single', 'E_equal', 'segments')
    vals = {}
    for met_m in METHODS:
        acc = {}
        for r in rows:
            if r['method'] == met_m:
                acc.setdefault((r['path'], r['query']), []).append(r)
        vals[met_m] = {k: {k2: float(np.mean([x[k2] for x in v])) for k2 in METRICS}
                       for k, v in acc.items()}
    summary = {}
    for met_m in METHODS:
        per_path = {}
        for (pi, j), v in vals[met_m].items():
            per_path.setdefault(pi, []).append(v)
        summary[met_m] = {k2: float(np.median([np.mean([v[k2] for v in ps_])
                                               for ps_ in per_path.values()]))
                          for k2 in METRICS}
    unified = {}
    for met_m in METHODS:
        per_path = {}
        for r in rows:
            if r['method'] != met_m:
                continue
            per_path.setdefault(r['path'], []).append(r)
        unified[met_m] = {}
        for k2 in METRICS[:-1]:
            unified[met_m][k2] = float(np.median(
                [100 * np.mean([x[k2] / x['fam_scale_reference'] for x in ps_])
                 for ps_ in per_path.values()]))
        segd = []
        for ps_ in per_path.values():
            tot = [sum(x['per_direction_segments'][i]
                       for i in dedup_idx[x['query']]['first_indices'])
                   for x in ps_]
            segd.append(float(np.mean(tot)))
        unified[met_m]['segments_distinct_chains'] = float(np.median(segd))
    paired = []
    for (pi, j), v in vals['flow'].items():
        if (pi, j) in vals['response']:
            paired.append(vals['response'][(pi, j)]['E_total'] - v['E_total'])
    rdom = float(np.mean([r['response_norm'] > r['weight_norm'] for r in rows]))
    return {
        'tolerance': TOL,
        'tolerance_definition': '5% of the largest learned account magnitude in the family',
        'summary_path_level_median': summary,
        'table_unified_normalized': unified,
        'table_note': ('tab:totalerror values: per-row E/fam_scale_reference, '
                       'mean within path, median over the 96 paths; segments '
                       'summed over distinct direction chains (duplicates of '
                       'the same nested chain removed)'),
        'paired_E_total_difference': {
            'median': float(np.median(paired)),
            'improved_share': float(np.mean(np.array(paired) < 0)),
            'n': len(paired),
        },
        'response_dominant_share_rows': rdom,
        'max_decomposition_closure': float(max(r['decomposition_closure'] for r in rows)),
        'dedup_directions_per_query': {
            str(j): {'directions': v['directions'],
                     'distinct_chains': v['distinct_chains']}
            for j, v in dedup_idx.items()},
        'gates': {'e122_E_geo_match': 'asserted per row',
                  'E_summary_leq_eps': 'asserted per row',
                  'E_total_leq_budget': 'asserted per row'},
    }


def main():
    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    chains = direction_levels(net=n)
    fams = [query_families(s, t, chains=chains, net=n) for s, t in ods]
    geo = []
    for fam_j in fams:
        rows = [np.array([[v in U for v in range(n.n)] for U in f['masks']])
                for f in fam_j]
        geo.append([r @ B for r in rows])       # per direction: (levels, m)

    # ---- 96 reference paths and columns ----
    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Cnew = np.load(E119 / 'reference_columns.npz')['C']
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jref = np.concatenate([d_old['J'], d_new['J']])
    groups = np.concatenate([d_old['group'], d_new['group']])
    ts = d_old['ts']
    Cold = np.array([attribution(np.array([T.grad(f) for f in Fref[pi]]), Jref[pi], ts,
                                 Q[pi, -1] - Q[pi, 0]) for pi in range(24)])
    Cref = np.concatenate([Cold, Cnew])
    N = len(Q)

    # ---- e122 stored E_geo for the gate ----
    e122_rows = json.loads((E122 / 'rows.json').read_text())
    e122_map = {(r['method'], r['seed'], r['path'], r['query']): r['E_geo']
                for r in e122_rows}

    selection = {(s['method'], s['seed']): s['step']
                 for s in json.loads((OLD / 'results.json').read_text())['selection']}
    rows = []
    started = time.perf_counter()
    for method in METHODS:
        for seed in range(5):
            m = p.base.old.build_fullnet_model(ps, qbg, seed)
            step = selection[(method, seed)]
            ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else \
                E111 / f'{method}_{seed}.pt'
            m.load_state_dict(torch.load(ck, weights_only=False)['best'])
            for pi in range(N):
                f, jac, C = learned_columns(m, Q[pi], ts, T)
                gL = np.array([T.grad(f0) for f0 in f])
                gS = np.array([T.grad(f0) for f0 in Fref[pi]])
                dQ = Q[pi, -1] - Q[pi, 0]
                flow = 100 * np.mean(np.linalg.norm(f - Fref[pi], axis=1)
                                     / np.linalg.norm(Fref[pi], axis=1))
                for j, (s, t) in enumerate(ods):
                    if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                        continue
                    cL, cS = C[:, j], Cref[pi][:, j]
                    aL = [-(g @ cL) for g in geo[j]]
                    aS = [-(g @ cS) for g in geo[j]]
                    E_model = max(np.max(np.abs(al - as_))
                                  for al, as_ in zip(aL, aS))
                    # gate against e122 E_geo (identical family union)
                    ref_egeo = e122_map.get((method, seed, pi, j))
                    if ref_egeo is not None:
                        assert abs(E_model - ref_egeo) <= 1e-6 * max(1., ref_egeo), \
                            (method, seed, pi, j, E_model, ref_egeo)
                    epsL = TOL * max(np.max(np.abs(al)) for al in aL)
                    seg_counts, hatL = [], []
                    for al in aL:
                        segs = greedy_segments(al, epsL)
                        seg_counts.append(len(segs))
                        hatL.append(summarize(al, segs))
                    E_summary = max(np.max(np.abs(h - al))
                                    for h, al in zip(hatL, aL))
                    E_total = max(np.max(np.abs(h - as_))
                                  for h, as_ in zip(hatL, aS))
                    assert E_summary <= epsL + 1e-9 * max(1., epsL)
                    assert E_total <= E_summary + E_model + 1e-6 * max(1., E_model)
                    # baselines: single common value; equal-count same counts
                    allL = np.concatenate(aL)
                    beta = (allL.max() + allL.min()) / 2
                    E_single = np.max(np.abs(beta - np.concatenate(aS)))
                    hatE = []
                    for al, k in zip(aL, seg_counts):
                        nn = len(al)
                        eq = [(int(round(x * nn / k)), int(round((x + 1) * nn / k)) - 1)
                              for x in range(k)]
                        hatE.append(summarize(al, eq))
                    E_equal = max(np.max(np.abs(h - as_))
                                  for h, as_ in zip(hatE, aS))
                    # response vs weight decomposition of the column error
                    resp = attribution(gL, jac[:, :, j:j + 1] - Jref[pi][:, :, j:j + 1],
                                       ts, dQ[j:j + 1])[:, 0]
                    wt = attribution(gL - gS, Jref[pi][:, :, j:j + 1],
                                     ts, dQ[j:j + 1])[:, 0]
                    closure = np.max(np.abs((resp + wt) - (cL - cS)))
                    rows.append({
                        'method': method, 'seed': seed, 'path': pi, 'query': j,
                        'flow_pct': float(flow),
                        'E_model': float(E_model), 'E_summary': float(E_summary),
                        'E_total': float(E_total), 'segments': int(sum(seg_counts)),
                        'per_direction_segments': seg_counts,
                        'E_single': float(E_single), 'E_equal': float(E_equal),
                        'eps_learned': float(epsL),
                        'fam_scale_reference': float(
                            max(np.max(np.abs(as_)) for as_ in aS)),
                        'response_norm': float(np.linalg.norm(resp)),
                        'weight_norm': float(np.linalg.norm(wt)),
                        'decomposition_closure': float(closure),
                    })
            print('evaluated', method, seed, f'{time.perf_counter() - started:.0f}s',
                  flush=True)
    (OUT / 'rows.json').write_text(json.dumps(rows, indent=2))

    # ---- path-level aggregation (seeds averaged within path, as e119) ----
    METRICS = ('E_model', 'E_summary', 'E_total', 'E_single', 'E_equal', 'segments')
    chains_d = direction_levels(net=n)
    dedup_idx = dedup_indices(ods, chains_d, n)
    out = aggregate(rows, dedup_idx)
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out['table_unified_normalized'], indent=2))
    print('paired E_total diff:', out['paired_E_total_difference'],
          'response-dominant share:', out['response_dominant_share_rows'])


def from_rows():
    rows = json.loads((OUT / 'rows.json').read_text())
    n_net, ods_d, *_ = p.base.setup()
    chains_d = direction_levels(net=n_net)
    dedup_idx = dedup_indices(ods_d, chains_d, n_net)
    out = aggregate(rows, dedup_idx)
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out['table_unified_normalized'], indent=2))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--from-rows':
        from_rows()
    else:
        main()
