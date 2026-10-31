"""E133 (v108): tolerance-compression frontier within fixed projection
directions (per-direction rework of e120 for Sioux Falls).

e120 ordered all separating regions of a Sioux Falls query into one
progress sequence, mixing screenline directions.  This experiment
sweeps the same tolerances tau in {2.5, 5, 7.5, 10}% with the e126
direction-grouped chains: each of the 36 projection normals is its own
ordered profile (nested chain by projection level, duplicates
deduplicated), segmented independently at the family tolerance
(5%-style rule on the family's largest account magnitude).  Reported
per query and load level: whether one common value covers the whole
family; the median and maximum per-direction segment count; the median
per-direction compression; the median equal-count worst-error ratio.
The mixed-ordering counts of e120 are retained for comparison.
Anaheim families are single-direction and are unchanged.  No new
equilibrium solves.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path
import numpy as np

from fam_v108 import direction_levels, query_families
from nets import load_sioux_falls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e133_v108_pertol'
OUT.mkdir(exist_ok=True)
TOLS = (0.025, 0.05, 0.075, 0.10)


def greedy_segments(a, eps):
    segs, i, n = [], 0, len(a)
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


def equal_max_eps(a, K):
    n = len(a)
    worst = 0.0
    for k in range(K):
        i, j = int(round(k * n / K)), int(round((k + 1) * n / K)) - 1
        s = a[i:j + 1]
        worst = max(worst, (s.max() - s.min()) / 2)
    return worst


def main():
    net = load_sioux_falls()
    chains = direction_levels(net=net)
    e114 = json.loads((ROOT / 'experiments/results/e114_v103/results.json').read_text())
    e116 = json.loads((ROOT / 'experiments/results/e116_v105_loadlevels/results.json').read_text())
    e120 = json.loads((ROOT / 'experiments/results/e120_v106_tolerance/results.json').read_text())
    e114_regions = [tuple(U) for U in json.loads(
        (ROOT / 'experiments/results/e114_v103/protocol.json').read_text())['all_region_masks']]
    mixed = {(r['od'][0], r['od'][1], r['level']): r for r in e120['sioux_rows']}

    sources = [({'level': 1.0}, row) for row in e114['rows'] if row['halfwidth'] == .2] + \
              [({'level': row['level']}, row) for row in e116['rows']]
    rows = []
    for meta, row in sources:
        s, t = row['od'][0] - 1, row['od'][1] - 1
        fams = query_families(s, t, chains=chains, net=net)
        masks = [U for U in e114_regions if s in U and t not in U]
        acct = {U: a for U, a in zip(masks, row['geometric_accounts'])}
        assert len({tuple(m) for f in fams for m in f['masks']}) == len(masks)
        uniq = {}
        for f in fams:
            uniq.setdefault(tuple(f['masks']), f)
        fam_mag = max(abs(a) for a in acct.values())
        fam_rel = (max(acct.values()) - min(acct.values())) / 2 / fam_mag if fam_mag > 0 else 0.
        per = []
        for tau in TOLS:
            eps = tau * fam_mag
            seg_counts, comp, eqrat = [], [], []
            for f in uniq.values():
                a = np.array([acct[m] for m in f['masks']])
                segs = greedy_segments(a, eps)
                seg_counts.append(len(segs))
                comp.append(1 - len(segs) / len(a))
                worst = max((a[i:j + 1].max() - a[i:j + 1].min()) / 2 for i, j in segs)
                eqrat.append(equal_max_eps(a, len(segs)) / worst if worst > 0 else 0.)
            per.append({
                'tau': tau, 'single_value_ok': bool(fam_rel <= tau),
                'distinct_chains': len(uniq),
                'median_segments_per_direction': float(np.median(seg_counts)),
                'max_segments_per_direction': int(max(seg_counts)),
                'total_segments': int(sum(seg_counts)),
                'median_compression_per_direction': float(np.median(comp)),
                'median_equal_over_greedy': float(np.median(eqrat)),
            })
        m = mixed[(row['od'][0], row['od'][1], meta['level'])]
        rows.append({'od': row['od'], 'level': meta['level'], 'regions': len(masks),
                     'relative_error': float(fam_rel), 'tolerance_sweep': per,
                     'mixed_order_segments': {f"tau{int(x['tau'] * 1000)}": x['segments']
                                              for x in m['tolerance_sweep']}})
    levels = {}
    for lev in (0.7, 1.0, 1.3):
        rs = [r for r in rows if r['level'] == lev]
        levels[str(lev)] = {'queries': len(rs), **{
            f'tau{int(tau * 1000)}': {
                'single_value_queries': int(sum(r['tolerance_sweep'][ti]['single_value_ok'] for r in rs)),
                'median_segments_per_direction': float(np.median(
                    [r['tolerance_sweep'][ti]['median_segments_per_direction'] for r in rs])),
                'max_segments_per_direction': int(max(
                    r['tolerance_sweep'][ti]['max_segments_per_direction'] for r in rs)),
                'median_total_segments': float(np.median(
                    [r['tolerance_sweep'][ti]['total_segments'] for r in rs])),
                'median_compression_per_direction': float(np.median(
                    [r['tolerance_sweep'][ti]['median_compression_per_direction'] for r in rs])),
                'median_equal_over_greedy': float(np.median(
                    [r['tolerance_sweep'][ti]['median_equal_over_greedy'] for r in rs])),
            } for ti, tau in enumerate(TOLS)}}
    out = {'tolerances': list(TOLS), 'sioux_levels': levels, 'sioux_rows': rows,
           'anaheim': e120['anaheim'], 'anaheim_rows': e120['anaheim_rows'],
           'data_sources': 'e114_v103 (level 1.0), e116_v105_loadlevels (0.7, 1.3), '
                           'v105_anaheim_profile (unchanged, single-direction); '
                           'direction chains of e126; no new solves'}
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(levels, indent=1))


if __name__ == '__main__':
    main()
