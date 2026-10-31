"""E126 (v108): per-direction spatial summaries and full segment outputs.

External review of v107 item 3.3: the Sioux Falls summary ordered all
separating regions of a query into one progress sequence, mixing
screenline directions, so adjacent array positions are not adjacent
traffic cross-sections.  This experiment re-segments the existing e114
(+/-20%) accounts within fixed projection directions: each of the 36
normals contributes its own nested chain ordered by projection level,
segmented independently with the same greedy rule and the same
family-level tolerance (5% of the largest account magnitude, no strict
sign mixing within a segment).  Query-level totals sum the per-direction
segment counts; the mixed progress ordering of e118 is retained as the
comparison.  For Anaheim the native progress-ordered nested-prefix family
already has a single direction; the full per-segment values, positions
and errors are emitted for the main application figure and table.
No equilibrium is re-solved.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path
import numpy as np

from fam_v108 import direction_levels, query_families, family_masks_union
from nets import load_sioux_falls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e126_v108_persdirection'
OUT.mkdir(exist_ok=True)
TOL = 0.05


def greedy_segments(a, eps):
    """Fewest contiguous segments with range<=2*eps and no strict sign mixing."""
    segs = []
    i = 0
    n = len(a)
    while i < n:
        j = i
        lo = hi = a[i]
        while j + 1 < n:
            nlo, nhi = min(lo, a[j + 1]), max(hi, a[j + 1])
            if nhi - nlo > 2 * eps:
                break
            if (nlo < 0 < nhi):
                break
            lo, hi = nlo, nhi
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def seg_detail(a, segs):
    out = []
    for i, j in segs:
        s = a[i:j + 1]
        out.append({'start': int(i), 'end': int(j),
                    'beta': float((s.max() + s.min()) / 2),
                    'epsilon': float((s.max() - s.min()) / 2)})
    return out


def equal_segments(a, K):
    n = len(a)
    return [(int(round(k * n / K)), int(round((k + 1) * n / K)) - 1) for k in range(K)]


def main():
    net = load_sioux_falls()
    chains = direction_levels(net=net)
    e114 = json.loads((ROOT / 'experiments/results/e114_v103/results.json').read_text())
    e118 = json.loads((ROOT / 'experiments/results/e118_v105_segmented/results.json').read_text())
    mixed = {tuple(r['od']): r for r in e118['sioux_falls']['rows']}

    rows = []
    mismatches = 0
    for row in e114['rows']:
        if row['halfwidth'] != .2:
            continue
        s, t = row['od'][0] - 1, row['od'][1] - 1
        fams = query_families(s, t, chains=chains, net=net)
        # account lookup keyed by the mask tuple, rebuilt exactly as e114
        # ordered its masks ([U for U in sorted regions if s in U and t not in U])
        acct = {}
        e114_regions = [tuple(U) for U in json.loads(
            (ROOT / 'experiments/results/e114_v103/protocol.json').read_text())['all_region_masks']]
        e114_masks = [U for U in e114_regions if s in U and t not in U]
        for U, a in zip(e114_masks, row['geometric_accounts']):
            acct[U] = float(a)
        union = family_masks_union(fams)
        if union != set(e114_masks):
            mismatches += 1
            raise AssertionError(f'mask set mismatch for OD {row["od"]}: '
                                 f'{len(union)} vs {len(e114_masks)}')
        fam_max = max(abs(a) for a in acct.values())
        fam_min = min(acct.values())
        fam_max_mag = max(abs(fam_max), abs(fam_min))
        eps = TOL * fam_max_mag
        drows = []
        for f in fams:
            a = np.array([acct[m] for m in f['masks']])
            segs = greedy_segments(a, eps)
            drows.append({
                'angle': f['angle'], 'side': f['side'], 'regions': len(a),
                'segments': len(segs),
                'levels': [float(x) for x in f['levels']],
                'masks': [list(m) for m in f['masks']],
                'accounts': a.tolist(),
                'detail': seg_detail(a, segs),
                'segment_level_ranges': [[float(f['levels'][i]), float(f['levels'][j])]
                                         for i, j in segs],
            })
        total_seg = sum(d['segments'] for d in drows)
        # several projection normals induce the same region sequence;
        # query-level aggregation deduplicates chains first
        uniq = {}
        for d in drows:
            uniq.setdefault(tuple(tuple(m) for m in d['masks']), d)
        distinct = list(uniq.values())
        seg_per_chain = [d['segments'] for d in distinct]
        reg_per_chain = [d['regions'] for d in distinct]
        m = mixed[tuple(row['od'])]
        # single common value over the whole family (minimax over all regions)
        beta_single = (fam_max + fam_min) / 2
        rows.append({
            'od': row['od'], 'regions': len(e114_masks),
            'eps_absolute': float(eps), 'fam_max_mag': float(fam_max_mag),
            'directions': len(drows), 'per_direction': drows,
            'segments_direction_total': int(total_seg),
            'segments_mixed_order': int(m['segments']),
            'distinct_chains': len(distinct),
            'segments_per_distinct_chain': sorted(seg_per_chain),
            'regions_per_distinct_chain': sorted(reg_per_chain),
            'median_segments_per_direction': float(np.median(seg_per_chain)),
            'max_segments_per_direction': int(max(seg_per_chain)),
            'median_regions_per_direction': float(np.median(reg_per_chain)),
            'single_value_beta': float(beta_single),
            'single_value_epsilon': float((fam_max - fam_min) / 2),
            'max_segment_epsilon': float(max(
                e['epsilon'] for d in drows for e in d['detail'])),
        })
        print(f"OD {row['od']}: dirs={len(drows)} seg_dir={total_seg} "
              f"seg_mixed={m['segments']} regions={len(e114_masks)}", flush=True)

    # --- Anaheim full segment outputs (native progress order) ---
    prof = json.loads((ROOT / 'experiments/results/v105_anaheim_profile/profile.json').read_text())
    arows = []
    for row in prof['rows']:
        a = np.array(row['accounts'])
        eps = TOL * max(abs(a.min()), abs(a.max()))
        segs = greedy_segments(a, eps)
        det = seg_detail(a, segs)
        eq = seg_detail(a, equal_segments(a, len(segs)))
        arows.append({
            'query': row['query'], 'regions': len(a),
            'fam_min': float(a.min()), 'fam_max': float(a.max()),
            'eps_absolute': float(eps), 'segments': len(segs),
            'progress_levels': row['progress_levels'],
            'accounts': row['accounts'],
            'segments_detail': det,
            'segment_progress_ranges': [[row['progress_levels'][i], row['progress_levels'][j]]
                                        for i, j in segs],
            'equal_count_max_epsilon': float(max(e['epsilon'] for e in eq)),
            'greedy_max_epsilon': float(max(e['epsilon'] for e in det)),
            'system': row['system'],
        })
        print(f"Anaheim {row['query']}: {len(segs)} segments / {len(a)} regions", flush=True)

    out = {
        'sioux_falls': {
            'tolerance': TOL,
            'tolerance_definition': 'absolute tolerance = 5% of the largest account magnitude in the family',
            'ordering': 'within fixed projection direction (36 normals), nested chain by projection level; directions segmented independently',
            'rows': rows,
        },
        'anaheim': {
            'tolerance': TOL,
            'ordering': 'native progress order (single direction)',
            'rows': arows,
        },
    }
    (OUT / 'results.json').write_text(json.dumps(out, indent=2))
    tot_dir = [r['segments_direction_total'] for r in rows]
    tot_mix = [r['segments_mixed_order'] for r in rows]
    print('SF total segments per query, direction vs mixed:',
          list(zip(tot_dir, tot_mix)))


if __name__ == '__main__':
    main()
