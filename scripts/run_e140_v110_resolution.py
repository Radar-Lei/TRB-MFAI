"""E140 (v110): final-explanation error versus reported-segment count, plus
the incremental-flow error E_dF (external review of v109, items 3.2 and 3.5-4).

The v109 total-error table fixes the tolerance at 5% of the learned family
maximum and reports E_model, E_summary, E_total and the segment count.  The
review asks for the resolution sweep: at tau in {0.5,1,2.5,5,10}% of the
same learned family scale, recompute the segmentation and report
E_model (tau-independent), E_summary(tau), E_total(tau), L(tau) for both
networks and both main methods (flow, response), all under the common
per-query normalization of tab:totalerror (E / max_U |a^S|, mean within
path, median over paths).  The same pass records E_summary^S(tau) --- the
compression error of the REFERENCE profile at the same tolerance --- as the
oracle-model floor, and the incremental-flow error

    E_dF = || (f^L(q1)-f^L(q0)) - (f^S(q1)-f^S(q0)) ||_2
           / || f^S(q1)-f^S(q0) ||_2

on endpoints of every test path (absolute numerator and reference
increment recorded for the small-denominator reporting).

Everything downstream of the frozen checkpoints is recomputed exactly as
e127 (Sioux Falls: direction-grouped families, dedup of duplicate chains at
aggregation) and e138 (Anaheim: single ordered family).

Gates: at tau=5% every per-row E_summary/E_total and the segment count
must match the stored e127/e138 row to 1e-9 relative; the recomputed
e127/e138 unified medians must reproduce the published table values.

Sharding: one shard per (network, method, seed); collect() merges, gates
and aggregates.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e140_v110_resolution'
OUT.mkdir(exist_ok=True)
E111 = HERE / 'results/e111_v102'
E119 = HERE / 'results/e119_v106_expanded_paths'
V104 = HERE / 'results/v104_learning'
E127 = HERE / 'results/e127_v108_totalerror'
DATA = HERE / 'results/e129_v108_anaheim_data'
TRAIN = HERE / 'results/e130_v108_anaheim_train'
E138 = HERE / 'results/e138_v109_crossnetwork'
METHODS = ('flow', 'response')
TAUS = (0.005, 0.01, 0.025, 0.05, 0.10)


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


def sweep_tau(aL_list, aS_list):
    """Per-tau metrics for one row: learned segmentation error, total error,
    segment counts per direction, and the reference-profile compression
    floor.  aL_list/aS_list list the per-direction account arrays."""
    fam_learned = max(float(np.max(np.abs(al))) for al in aL_list)
    out = {}
    for tau in TAUS:
        epsL = tau * fam_learned
        hatL, cnt, E_s, E_t = [], [], 0.0, 0.0
        for al in aL_list:
            segs = greedy_segments(al, epsL)
            cnt.append(len(segs))
            h = summarize(al, segs)
            hatL.append(h)
            E_s = max(E_s, float(np.max(np.abs(h - al))))
        for h, as_ in zip(hatL, aS_list):
            E_t = max(E_t, float(np.max(np.abs(h - as_))))
        # oracle floor: compress the reference profile itself at the same tau
        fam_ref_scale = max(float(np.max(np.abs(a))) for a in aS_list)
        epsS = tau * fam_ref_scale
        E_s_ref = 0.0
        for as_ in aS_list:
            segs = greedy_segments(as_, epsS)
            h = summarize(as_, segs)
            E_s_ref = max(E_s_ref, float(np.max(np.abs(h - as_))))
        out[f'{tau:g}'] = {'E_summary': E_s, 'E_total': E_t,
                           'segments_per_direction': cnt,
                           'E_summary_ref': E_s_ref}
    return out


def shard_sf(method, seed):
    """One (method, seed) pass over the 96 Sioux Falls test paths."""
    import run_e111_v102_pairs as p
    from run_e83_v82_validation import attribution
    from fam_v108 import direction_levels, query_families

    torch.set_num_threads(1)
    n, ods, qb, qbg, ps, sv, T = p.base.setup()
    B = n.B
    chains = direction_levels(net=n)
    fams = [query_families(s, t, chains=chains, net=n) for s, t in ods]
    geo = []
    for fam_j in fams:
        rr = [np.array([[v in U for v in range(n.n)] for U in f['masks']])
              for f in fam_j]
        geo.append([r @ B for r in rr])

    d_old = np.load(E111 / 'paths.npz')
    d_new = np.load(E119 / 'paths_new.npz')
    Cnew = np.load(E119 / 'reference_columns.npz')['C']
    Q = np.concatenate([d_old['Q'], d_new['Q']])
    Fref = np.concatenate([d_old['F'], d_new['F']])
    Jold = d_old['J']
    ts = d_old['ts']
    Cref = np.concatenate([
        np.array([attribution(np.array([T.grad(f) for f in Fref[pi]]),
                              Jold[pi], ts, Q[pi, -1] - Q[pi, 0])
                  for pi in range(24)]), Cnew])
    N = len(Q)

    e127_rows = json.loads((E127 / 'rows.json').read_text())
    gate_map = {(r['method'], r['seed'], r['path'], r['query']): r
                for r in e127_rows if r['method'] == method and r['seed'] == seed}
    selection = {(s['method'], s['seed']): s['step']
                 for s in json.loads((V104 / 'results.json').read_text())['selection']}
    m = p.base.old.build_fullnet_model(ps, qbg, seed)
    step = selection[(method, seed)]
    ck = p.base.old.CKPT_DIR / f'seed{seed}.pt' if step == 0 else \
        E111 / f'{method}_{seed}.pt'
    m.load_state_dict(torch.load(ck, weights_only=False)['best'])

    rows = []
    st = time.perf_counter()
    for pi in range(N):
        q = torch.tensor(Q[pi])
        f = m(q).detach().numpy()
        js = []
        for j in range(6):
            v = torch.zeros_like(q)
            v[:, j] = 1
            _, jv = torch.func.jvp(m, (q,), (v,))
            js.append(jv.detach().numpy())
        jac = np.stack(js, axis=2)
        C = attribution(np.array([T.grad(f0) for f0 in f]), jac, ts,
                        Q[pi, -1] - Q[pi, 0])
        dFL = f[-1] - f[0]
        dFS = Fref[pi, -1] - Fref[pi, 0]
        dflow = {'E_dF': float(np.linalg.norm(dFL - dFS)
                               / max(np.linalg.norm(dFS), 1e-12)),
                 'abs_num': float(np.linalg.norm(dFL - dFS)),
                 'ref_dF_norm': float(np.linalg.norm(dFS))}
        for j, (s, t) in enumerate(ods):
            if abs(Q[pi, -1, j] - Q[pi, 0, j]) < 1e-10:
                continue
            aL = [-(g @ C[:, j]) for g in geo[j]]
            aS = [-(g @ Cref[pi][:, j]) for g in geo[j]]
            E_model = max(float(np.max(np.abs(al - as_)))
                          for al, as_ in zip(aL, aS))
            sw = sweep_tau(aL, aS)
            # gate against the stored e127 row at tau=5%
            o = gate_map[(method, seed, pi, j)]
            g5 = sw['0.05']
            for key, new in (('E_summary', g5['E_summary']),
                             ('E_total', g5['E_total'])):
                assert abs(new - o[key]) <= 1e-9 * max(1., abs(o[key])), \
                    (method, seed, pi, j, key, new, o[key])
            assert sum(g5['segments_per_direction']) == o['segments']
            rows.append({'method': method, 'seed': seed, 'path': pi, 'query': j,
                         'E_model': E_model, 'tau_grid': sw,
                         'fam_scale_reference': float(
                             max(np.max(np.abs(a)) for a in aS)),
                         'dflow': dflow})
        print(f'sf {method} seed{seed} path{pi} {time.perf_counter() - st:.0f}s',
              flush=True)
    (OUT / f'sf_{method}_{seed}.json').write_text(json.dumps(rows, indent=1))


def shard_anaheim(method, seed):
    """One (method, seed) pass over the 48 Anaheim test paths."""
    torch.set_num_threads(1)
    from run_e83_v82_validation import attribution, TSTTT
    import run_e130_v108_anaheim_train as tr

    d = np.load(DATA / 'demands.npz')
    ref = np.load(DATA / 'reference.npz')
    with (DATA / 'setup.pkl').open('rb') as fh:
        net, ps, q = pickle.load(fh)
    fams = json.loads((DATA / 'families.json').read_text())
    Q, dQP = d['QP'], d['dQP']
    Cref = ref['Cref']
    Fp = [None] * 48
    for fn in sorted(DATA.glob('paths_*.npz')):
        z = np.load(fn)
        for fr, i in zip(z['F'], z['pids']):
            Fp[int(i)] = fr
    assert all(x is not None for x in Fp)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    B = net.B
    geo = []
    for f in fams:
        R = np.array([[1.0 if v in U else 0.0 for v in range(net.n)]
                      for U in f['masks']])
        geo.append(R @ B)
    e138_rows = json.loads((E138 / 'rows.json').read_text())
    gate_map = {(r['method'], r['seed'], r['path'], r['query']): r
                for r in e138_rows if r['method'] == method and r['seed'] == seed}

    ck = torch.load(TRAIN / f'{method}_{seed}.pt', weights_only=False)
    pre = torch.load(TRAIN / f'pre_seed{seed}.pt', weights_only=False)
    m = tr.build_model(ps, d['q'][6:], seed, pre['norm']['mu_q'], pre['norm']['sd_q'])
    m.load_state_dict(ck['best'])
    rows = []
    st = time.perf_counter()
    for pi in range(48):
        qt = torch.tensor(Q[pi])
        f = m(qt).detach().numpy()
        js = []
        for j in range(6):
            v = torch.zeros_like(qt)
            v[:, j] = 1.0
            _, jv = torch.func.jvp(m, (qt,), (v,))
            js.append(jv.detach().numpy())
        jac = np.stack(js, axis=2)
        C = attribution(np.array([T.grad(f0) for f0 in f]), jac, d['ts'], dQP[pi])
        dFL = f[-1] - f[0]
        dFS = Fp[pi][-1] - Fp[pi][0]
        dflow = {'E_dF': float(np.linalg.norm(dFL - dFS)
                               / max(np.linalg.norm(dFS), 1e-12)),
                 'abs_num': float(np.linalg.norm(dFL - dFS)),
                 'ref_dF_norm': float(np.linalg.norm(dFS))}
        for j in range(6):
            if abs(dQP[pi][j]) < 1e-10:
                continue
            r = Cref[pi][:, j]
            aL = [-(geo[j] @ C[:, j])]
            aS = [-(geo[j] @ r)]
            E_model = float(np.max(np.abs(aL[0] - aS[0])))
            sw = sweep_tau(aL, aS)
            o = gate_map[(method, seed, pi, j)]
            g5 = sw['0.05']
            for key, new in (('E_model', E_model),
                             ('E_summary', g5['E_summary']),
                             ('E_total', g5['E_total'])):
                assert abs(new - o[key]) <= 1e-9 * max(1., abs(o[key])), \
                    (method, seed, pi, j, key, new, o[key])
            assert g5['segments_per_direction'][0] == o['segments']
            rows.append({'method': method, 'seed': seed, 'path': pi, 'query': j,
                         'E_model': E_model, 'tau_grid': sw,
                         'fam_scale_reference': float(np.max(np.abs(aS[0]))),
                         'dflow': dflow})
        print(f'an {method} seed{seed} path{pi} {time.perf_counter() - st:.0f}s',
              flush=True)
    (OUT / f'an_{method}_{seed}.json').write_text(json.dumps(rows, indent=1))


def collect():
    import run_e111_v102_pairs as p
    from fam_v108 import direction_levels, query_families
    n, ods, *_ = p.base.setup()
    chains = direction_levels(net=n)
    dedup = {}
    for j, (s, t) in enumerate(ods):
        fams = query_families(s, t, chains=chains, net=n)
        seen, keep = set(), []
        for i, f in enumerate(fams):
            key = tuple(tuple(U) for U in f['masks'])
            if key not in seen:
                seen.add(key)
                keep.append(i)
        dedup[j] = keep

    result = {}
    for net, prefix, dedup_on in (('sioux_falls', 'sf', True),
                                  ('anaheim', 'an', False)):
        rows = []
        for method in METHODS:
            for seed in range(5):
                rows.extend(json.loads(
                    (OUT / f'{prefix}_{method}_{seed}.json').read_text()))
        net_out = {}
        for method in METHODS:
            mr = [r for r in rows if r['method'] == method]
            per_path = {}
            for r in mr:
                per_path.setdefault(r['path'], []).append(r)
            # tau-independent: E_model (normalized, path-median)
            net_out[method] = {}
            net_out[method]['E_model'] = float(np.median(
                [100 * np.mean([x['E_model'] / x['fam_scale_reference']
                                for x in ps_]) for ps_ in per_path.values()]))
            for tau in TAUS:
                key = f'{tau:g}'
                Es, Et, Esr, Ls = [], [], [], []
                for ps_ in per_path.values():
                    Es.append(100 * np.mean([x['tau_grid'][key]['E_summary']
                                             / x['fam_scale_reference']
                                             for x in ps_]))
                    Et.append(100 * np.mean([x['tau_grid'][key]['E_total']
                                             / x['fam_scale_reference']
                                             for x in ps_]))
                    Esr.append(100 * np.mean([x['tau_grid'][key]['E_summary_ref']
                                              / x['fam_scale_reference']
                                              for x in ps_]))
                    if dedup_on:
                        tot = [sum(x['tau_grid'][key]['segments_per_direction'][i]
                                   for i in dedup[x['query']]) for x in ps_]
                    else:
                        tot = [x['tau_grid'][key]['segments_per_direction'][0]
                               for x in ps_]
                    Ls.append(float(np.mean(tot)))
                net_out[method][f'tau_{key}'] = {
                    'E_summary': float(np.median(Es)),
                    'E_total': float(np.median(Et)),
                    'E_summary_ref_floor': float(np.median(Esr)),
                    'L': float(np.median(Ls)),
                }
            # incremental flow over paths (per seed first, then median over
            # path-seed units; endpoints identical for all queries of a path)
            dfl = {}
            for r in mr:
                dfl.setdefault((r['seed'], r['path']), r['dflow'])
            net_out[method]['E_dF'] = {
                'median_pct': float(np.median(
                    [100 * v['E_dF'] for v in dfl.values()])),
                'median_abs_num': float(np.median(
                    [v['abs_num'] for v in dfl.values()])),
                'median_ref_dF_norm': float(np.median(
                    [v['ref_dF_norm'] for v in dfl.values()])),
                'min_ref_dF_norm': float(np.min(
                    [v['ref_dF_norm'] for v in dfl.values()])),
            }
        result[net] = net_out

    # reference-profile segment counts (E144): merge so a fresh collect()
    # reproduces the figure inputs without a manual patch step
    e144 = HERE / 'results/e144_v110_refsegments/results.json'
    assert e144.exists(), 'run run_e144_v110_refsegments.py first (L_ref)'
    lref = json.loads(e144.read_text())
    for net in ('sioux_falls', 'anaheim'):
        result[net]['L_ref'] = {t: v['L_ref']
                                for t, v in lref[net].items()}

    # gate: 5% entries must reproduce the published e127/e138 unified values
    e127 = json.loads((E127 / 'results.json').read_text())['table_unified_normalized']
    for method in METHODS:
        g = result['sioux_falls'][method]['tau_0.05']
        gm = result['sioux_falls'][method]['E_model']
        ref = e127[method]
        for k2, gv, rv in (('E_model', gm, ref['E_model']),
                           ('E_summary', g['E_summary'], ref['E_summary']),
                           ('E_total', g['E_total'], ref['E_total'])):
            assert abs(gv - rv) < 5e-3, ('sf gate', method, k2, gv, rv)
        assert abs(g['L'] - ref['segments_distinct_chains']) < 1.0, \
            ('sf seg gate', method, g['L'], ref['segments_distinct_chains'])
    e138u = json.loads((E138 / 'results.json').read_text())['table_unified_normalized']
    for method in METHODS:
        g = result['anaheim'][method]['tau_0.05']
        gm = result['anaheim'][method]['E_model']
        ref = e138u[method]
        for k2, gv in (('E_model', gm), ('E_summary', g['E_summary']),
                       ('E_total', g['E_total'])):
            assert abs(gv - ref[k2]) < 5e-3, ('an gate', method, k2, gv, ref)
        assert abs(g['L'] - ref['segments']) < 1.0, ('an seg gate', method)

    (OUT / 'results.json').write_text(json.dumps(
        {'protocol': ('tau sweep on the frozen e127/e138 rows: eps(tau) = '
                      'tau x max|a^L| per family (learned scale), greedy '
                      'absolute rule identical to e127/e138; unified '
                      'normalization E/fam_scale_reference, mean within '
                      'path, median over paths; L sums distinct direction '
                      'chains on Sioux Falls; oracle floor compresses the '
                      'reference profile at the same tau; E_dF from path '
                      'endpoint flow predictions; per-row gates at tau=5% '
                      'asserted during sharding'),
         'table': result}, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['sf', 'an', 'collect'])
    ap.add_argument('--method', choices=METHODS)
    ap.add_argument('--seed', type=int)
    v = ap.parse_args()
    if v.cmd == 'sf':
        shard_sf(v.method, v.seed)
    elif v.cmd == 'an':
        shard_anaheim(v.method, v.seed)
    else:
        collect()
