"""E153 (v112): iso-segment matched comparison on the pre-specified
tolerance grid.

Codex round-1 review of v112 (MAJOR-1): the tab:matched rows pair the two
methods at equal tolerance, and the per-tolerance segment counts differ
slightly (Sioux Falls 0.5%: 232.87 vs 234.58; 38 of 48 Anaheim paths have
unequal paired mean segments), so the intervals support "lower error at
equal tolerance", not yet the claim "lower final error at every matched
segment count".  This run implements the matched-budget comparison
literally: at each tolerance the common budget is the smaller of the two
methods' per-path segment counts, L* = min(L_flow, L_response); each
method's (L, E) points over the five tolerances are linearly interpolated
to L*, and the paired difference E_response(L*) - E_flow(L*) is bootstrapped
per path exactly as in E151.

Data: the frozen E140 shards via E151's path_map (no model re-evaluated);
the tau=5% per-tolerance columns reproduce E151's matched table, which
reproduces the published values.

Outputs (results/e153_v112_isosegment/results.json):
  matched_iso[net][tau] = {L_star_med, interp_frac, diff_med, diff_ci95,
      diff_same_tol_med}  (diff at interpolated common budget; the last
      column restates the equal-tolerance median for side-by-side reading)
Codex round-2 (P1): budgets outside a method's grid support are
excluded rather than endpoint-clamped (a clamped endpoint is not a
matched budget); each tolerance reports n_compared, the number of
paths whose L* lies in both grids' common support.

Gate: on every tolerance with n_compared >= 10, sign(iso diff) ==
sign(equal-tolerance diff) and |iso median| >= 0.8 * |equal-tolerance
median|.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from run_e151_v112_frontier import TAUS, NETS, NBOOT, dedup_map, path_map, boot_ci

OUT = HERE / 'results/e153_v112_isosegment'
OUT.mkdir(exist_ok=True)


def interp_E(points, L):
    """Linear interpolation of E over sorted (L, E) points, clamped."""
    pts = sorted(points)
    Ls = np.array([p[0] for p in pts])
    Es = np.array([p[1] for p in pts])
    return float(np.interp(L, Ls, Es))


def main():
    rng = np.random.default_rng(112153)
    dedup = dedup_map()
    results = {'matched_iso': {}, 'protocol': {
        'budget': 'L* = min(L_flow, L_response) per path and tolerance; '
                  'each method interpolated linearly on its own (L, E) grid '
                  'over the five tolerances; paired bootstrap of the median '
                  'difference, 5000 resamples'}}
    gates = {}
    for net in NETS:
        pm = path_map(net, dedup)
        paths = sorted(pm['flow'])
        iso = {}
        for tau in TAUS:
            Lstars, diffs_iso, diffs_tol = [], [], []
            clamped = 0
            for p in paths:
                # path_map entries are (E_total_pct, L); flip to (L, E)
                pf = [(v[1], v[0]) for v in (pm['flow'][p][t] for t in TAUS)]
                pr = [(v[1], v[0]) for v in (pm['response'][p][t] for t in TAUS)]
                Lf, Lr = pm['flow'][p][tau][1], pm['response'][p][tau][1]
                Lstar = min(Lf, Lr)
                lo_f, hi_f = min(v[0] for v in pf), max(v[0] for v in pf)
                lo_r, hi_r = min(v[0] for v in pr), max(v[0] for v in pr)
                in_support = lo_f <= Lstar <= hi_f and lo_r <= Lstar <= hi_r
                if not in_support:
                    clamped += 1
                    continue
                Ef = interp_E(pf, Lstar)
                Er = interp_E(pr, Lstar)
                Lstars.append(Lstar)
                diffs_iso.append(Er - Ef)
                diffs_tol.append(pm['response'][p][tau][0] - pm['flow'][p][tau][0])
            iso[tau] = {
                'L_star_med': float(np.median(Lstars)) if Lstars else None,
                'n_compared': len(diffs_iso),
                'clamped_paths': clamped,
                'diff_med': float(np.median(diffs_iso)) if diffs_iso else None,
                'diff_ci95': boot_ci(diffs_iso, rng) if diffs_iso else None,
                'diff_same_tol_med': float(np.median(diffs_tol))}
            v = iso[tau]
            gates[f'{net}:{tau}'] = bool(
                v['diff_med'] is not None and v['n_compared'] >= 10
                and np.sign(v['diff_med']) == np.sign(v['diff_same_tol_med'])
                and abs(v['diff_med']) >= 0.8 * abs(v['diff_same_tol_med']))
        results['matched_iso'][net] = iso
    results['gates'] = gates
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    assert all(gates.values()), gates
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
