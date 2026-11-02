"""E151 (v112): matched-complexity end-to-end comparison with paired
bootstrap intervals.

External review of v111: Table tab:pathtable reports the final summary
error at a single working tolerance (5%), where Anaheim's flow-vs-response
difference (4.57 vs 4.55) reads as marginal even though the error budget
explains it.  The review asks for the comparison across the pre-specified
tolerance grid with complexity matched: final error at equal segment count,
segments required at equal error, and paired bootstrap intervals.

Data: the frozen E140 shards (per method, seed, path, query: E_model and
tau_grid with E_summary / E_total / segments).  No model is re-evaluated;
per-path normalized values are recomputed exactly as E140 aggregates
(E_total / fam_scale_reference, mean over active queries and seeds within
path; L summed over deduplicated distinct direction chains on Sioux Falls,
single-family length on Anaheim), so the tau=5% medians reproduce the
published table.

Outputs (results/e151_v112_frontier/results.json):
  frontier[net][method][tau] = {L_med, E_total_med, ci95}
  matched[net][tau] = {L_flow, L_response, E_flow, E_response,
      diff_med, diff_ci95 (paired per-path bootstrap of the median)}
  iso_error[net][level] = segments each method needs to reach the level
      (smallest median L with median E_total <= level) and the saving.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E140 = HERE / 'results/e140_v110_resolution'
OUT = HERE / 'results/e151_v112_frontier'
OUT.mkdir(exist_ok=True)
TAUS = ('0.005', '0.01', '0.025', '0.05', '0.1')
NETS = {'sioux_falls': 'sf', 'anaheim': 'an'}
NBOOT = 5000


def dedup_map():
    """Distinct-chain indices per query (E140 collect's dedup)."""
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
    return dedup


def path_map(net, dedup):
    """{method: {path: {tau: (E_total_pct, L)}}}, E140 aggregation."""
    rows = []
    for method in ('flow', 'response'):
        for seed in range(5):
            rows += json.loads((E140 / f'{NETS[net]}_{method}_{seed}.json').read_text())
    acc = {'flow': {}, 'response': {}}
    for r in rows:
        fs = r['fam_scale_reference']
        for tau in TAUS:
            g = r['tau_grid'][tau]
            if net == 'sioux_falls':
                L = float(sum(g['segments_per_direction'][i]
                              for i in dedup[r['query']]))
            else:
                L = float(g['segments_per_direction'][0])
            acc[r['method']].setdefault((r['path'], tau), []).append(
                (100 * g['E_total'] / fs, L))
    out = {'flow': {}, 'response': {}}
    for method in acc:
        bypath = {}
        for (path, tau), vals in acc[method].items():
            bypath.setdefault(path, {}).setdefault(tau, []).extend(vals)
        for path, taus in bypath.items():
            out[method][path] = {tau: (float(np.mean([v[0] for v in vals])),
                                       float(np.mean([v[1] for v in vals])))
                                 for tau, vals in taus.items()}
    return out


def boot_ci(vals, rng):
    vals = np.asarray(vals, float)
    meds = [float(np.median(vals[rng.integers(0, len(vals), len(vals))]))
            for _ in range(NBOOT)]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def main():
    rng = np.random.default_rng(112151)
    dedup = dedup_map()
    results = {'frontier': {}, 'matched': {}, 'iso_error': {}}
    for net in NETS:
        pm = path_map(net, dedup)
        paths = sorted(pm['flow'])
        frontier = {}
        for method in ('flow', 'response'):
            frontier[method] = {}
            for tau in TAUS:
                ev = [pm[method][p][tau][0] for p in paths]
                lv = [pm[method][p][tau][1] for p in paths]
                frontier[method][tau] = {
                    'E_total_med': float(np.median(ev)),
                    'E_total_ci95': boot_ci(ev, rng),
                    'L_med': float(np.median(lv)),
                    'L_ci95': boot_ci(lv, rng),
                    'n_paths': len(paths)}
        results['frontier'][net] = frontier
        matched = {}
        for tau in TAUS:
            evf = np.array([pm['flow'][p][tau][0] for p in paths])
            evr = np.array([pm['response'][p][tau][0] for p in paths])
            diff = evr - evf
            matched[tau] = {
                'L_flow': frontier['flow'][tau]['L_med'],
                'L_response': frontier['response'][tau]['L_med'],
                'E_flow': frontier['flow'][tau]['E_total_med'],
                'E_response': frontier['response'][tau]['E_total_med'],
                'diff_med': float(np.median(diff)),
                'diff_ci95': boot_ci(diff, rng)}
        results['matched'][net] = matched
        targets = [8.8, 11.0] if net == 'sioux_falls' else [1.4, 4.6]
        iso = {}
        for level in targets:
            segs = {}
            for method in ('flow', 'response'):
                cand = [(frontier[method][t]['L_med'], frontier[method][t]['E_total_med'])
                        for t in TAUS]
                ok = [L for L, E in cand if E <= level]
                segs[method] = min(ok) if ok else None
            iso[str(level)] = {
                'segments_flow': segs['flow'], 'segments_response': segs['response'],
                'saving': (segs['flow'] - segs['response']
                           if segs['flow'] is not None and segs['response'] is not None
                           else None)}
        results['iso_error'][net] = iso
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    # gate: tau=5% medians must reproduce the published table values
    pub = {'sioux_falls': {'flow': 11.35, 'response': 8.68},
           'anaheim': {'flow': 4.57, 'response': 4.55}}
    for net in NETS:
        for method in ('flow', 'response'):
            got = results['frontier'][net][method]['0.05']['E_total_med']
            assert abs(got - pub[net][method]) < 0.02, (net, method, got)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
