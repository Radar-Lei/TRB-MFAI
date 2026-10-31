"""E132 (v108): mixed absolute-relative tolerance sensitivity for the
minimum-segment summaries (external review item 3.3, low-cost item).

The primary tolerance is absolute at 5% of the largest account magnitude
in each family, which permits errors far larger than 5% at
small-magnitude positions.  A mixed tolerance

    |hat a_i - a_i| <= eps_abs + tau * |a_i|

keeps relative control at small magnitudes.  A candidate segment I has a
common representative value iff max_{i in I}(a_i - t_i) <= min_{i in I}
(a_i + t_i) with t_i = eps_abs + tau*|a_i|; feasibility is inherited by
subintervals, so the same greedy longest-feasible-prefix rule returns the
minimum number of segments.  This experiment sweeps (eps_abs, tau) around
the primary rule on the existing Sioux Falls direction chains (e126) and
the Anaheim progress families, and reports segment counts together with
the worst relative error at the smallest-magnitude positions.  No
equilibrium is re-solved.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/e132_v108_mixedtol'
OUT.mkdir(exist_ok=True)

# (eps_abs as fraction of family max magnitude, tau): the first entry is the
# primary absolute rule (tau=0); the others add relative control.
GRID = [(0.05, 0.0), (0.05, 0.05), (0.05, 0.10), (0.025, 0.05), (0.025, 0.10),
        (0.01, 0.10)]


def greedy_mixed(a, t):
    """Fewest contiguous segments under |beta - a_i| <= t_i for all i in I.

    A segment [i, j] is feasible iff max(a-t) <= min(a+t) over it; the
    representative is any value in [max(a-t), min(a+t)] (midpoint used).
    Feasibility is inherited by subintervals, so greedy is optimal.
    """
    segs = []
    i = 0
    n = len(a)
    while i < n:
        j = i
        lo = a[i] - t[i]
        hi = a[i] + t[i]
        while j + 1 < n:
            nlo = max(lo, a[j + 1] - t[j + 1])
            nhi = min(hi, a[j + 1] + t[j + 1])
            if nlo > nhi:
                break
            lo, hi = nlo, nhi
            j += 1
        segs.append((i, j, (lo + hi) / 2))
        i = j + 1
    return segs


def rel_errors(a, segs):
    """Worst relative deviation of the representative at each position."""
    hat = np.empty_like(a)
    for i, j, beta in segs:
        hat[i:j + 1] = beta
    small = np.abs(a) < .1 * np.max(np.abs(a))
    denom_small = np.where(small, np.maximum(np.abs(a), 1e-12), np.nan)
    denom_all = np.maximum(np.abs(a), 1e-12)
    return {
        'max_rel_overall': float(np.max(np.abs(hat - a) / denom_all)),
        'max_rel_small_positions': float(np.nanmax(np.abs(hat - a) / denom_small))
        if small.any() else None,
        'max_abs': float(np.max(np.abs(hat - a))),
    }


def main():
    e126 = json.loads(
        (ROOT / 'experiments/results/e126_v108_persdirection/results.json').read_text())

    sf_rows = []
    for row in e126['sioux_falls']['rows']:
        scale = row['fam_max_mag']
        # dedup chains exactly as e126
        uniq = {}
        for d in row['per_direction']:
            uniq.setdefault(tuple(tuple(m) for m in d['masks']), d)
        chains = list(uniq.values())
        entry = {'od': row['od'], 'regions': row['regions'],
                 'chains': len(chains), 'grid': {}}
        for fa, tau in GRID:
            eps_abs = fa * scale
            tot, rels = 0, []
            for d in chains:
                a = np.array(d['accounts'])
                t = eps_abs + tau * np.abs(a)
                segs = greedy_mixed(a, t)
                tot += len(segs)
                rels.append(rel_errors(a, segs))
            entry['grid'][f'fa{fa}_tau{tau}'] = {
                'segments': int(tot),
                'median_segments_per_chain': float(tot / len(chains)),
                'max_rel_overall': max(r['max_rel_overall'] for r in rels),
                'max_rel_small': max((r['max_rel_small_positions'] or 0) for r in rels),
            }
        sf_rows.append(entry)
        print('SF', row['od'], {k: v['segments'] for k, v in entry['grid'].items()},
              flush=True)

    an_rows = []
    for row in e126['anaheim']['rows']:
        a = np.array(row['accounts'])
        scale = max(abs(a.min()), abs(a.max()))
        entry = {'query': row['query'], 'regions': row['regions'], 'grid': {}}
        for fa, tau in GRID:
            t = fa * scale + tau * np.abs(a)
            segs = greedy_mixed(a, t)
            rr = rel_errors(a, segs)
            entry['grid'][f'fa{fa}_tau{tau}'] = {
                'segments': len(segs), **rr}
        an_rows.append(entry)
        print('Anaheim', row['query'], {k: v['segments'] for k, v in entry['grid'].items()},
              flush=True)

    (OUT / 'results.json').write_text(json.dumps({
        'grid': [{'eps_abs_frac': fa, 'tau': tau} for fa, tau in GRID],
        'sioux_falls': sf_rows, 'anaheim': an_rows,
    }, indent=2))
    # headline medians
    headline = {}
    for name, rows in (('sioux_falls', sf_rows), ('anaheim', an_rows)):
        head = {}
        for key in rows[0]['grid']:
            relkey = 'max_rel_small' if 'max_rel_small' in rows[0]['grid'][key] \
                else 'max_rel_small_positions'
            head[key] = {
                'median_segments': float(np.median([r['grid'][key]['segments'] for r in rows])),
                'median_max_rel_small': float(np.median(
                    [r['grid'][key][relkey] for r in rows])),
                'max_max_rel_small': float(np.max(
                    [r['grid'][key][relkey] for r in rows])),
            }
        headline[name] = head
        print(name, json.dumps(head, indent=1))
    (OUT / 'headline.json').write_text(json.dumps(headline, indent=2))


if __name__ == '__main__':
    main()
