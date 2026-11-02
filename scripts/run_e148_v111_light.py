"""E148 (v111): two light computations requested by the external review.

(a) Topology-release distribution (Section 5.2 option B): per-OD
    constrained-edge shares on the four public networks, reported as
    10/25/50/75/90 percent quantiles, the OD fractions with share below
    25/50/75%, and the most peripheral OD (minimum share, its traversed
    atom count) as the corridor/peripheral counterpoint to the median.

(b) Equal-wall-clock comparison (Section 6.4): the tripled-budget
    convergence runs of E135 logged validation boundary error and elapsed
    wall-clock at every validation point; this converts the equal-update
    comparison into equal-wall-clock medians at the shorter method's
    total time on each network.

No model is retrained; both parts read frozen arrays only.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'results/e148_v111_light'
OUT.mkdir(exist_ok=True)
E135 = HERE / 'results/e135_v109_extbudget'


def full_shares(net, ods):
    from run_e125_v107_topologystats import distribution  # noqa: F401 (consistency)
    shares, n_atoms = [], []
    for (s, t) in ods:
        ks = net.od_path_atoms(s, t)
        eq_edges = {e for k in ks for e in net.atoms()[k]['edges']}
        shares.append(100.0 * len(eq_edges) / net.m)
        n_atoms.append(len(ks))
    return np.array(shares), np.array(n_atoms)


def topology_distribution():
    from nets import load_sioux_falls_trips
    import run_e111_v102_pairs as p
    import run_e112_v102_anaheim as e112
    from run_e125_v107_topologystats import load_tntp

    out = {}
    # Sioux Falls
    n_sf, ods_sf, qb, qbg, ps, sv, T = p.base.setup()
    trips = load_sioux_falls_trips()
    all_sf = sorted(o for o, d in trips.items() if d > 0)
    a, na = full_shares(n_sf, all_sf)
    out['sioux_falls'] = summarize(a, na, all_sf, n_sf)
    # Anaheim
    n_an, ps_an, q_an = e112.setup()
    ods_an = list(ps_an.ods)
    a, na = full_shares(n_an, ods_an)
    out['anaheim'] = summarize(a, na, ods_an, n_an)
    # extras
    here = ROOT / 'experiments/data/candidates'
    for name in ('EMA', 'Winnipeg'):
        net, trips_x = load_tntp(name, here)
        ods = sorted(trips_x)
        a, na = full_shares(net, ods)
        out[name.lower()] = summarize(a, na, ods, net)
    return out


def summarize(a, n_atoms, ods, net):
    imin = int(np.argmin(a))
    return {
        'nodes': net.n, 'edges': net.m, 'queries': int(len(a)),
        'quantile_pct': {str(q): float(np.percentile(a, q))
                         for q in (10, 25, 50, 75, 90)},
        'min_pct': float(a.min()), 'max_pct': float(a.max()),
        'below_share_pct': {str(th): float(np.mean(a < th))
                            for th in (25.0, 50.0, 75.0)},
        'peripheral_example': {
            'od': [int(ods[imin][0]) + 1, int(ods[imin][1]) + 1],
            'share_pct': float(a[imin]),
            'atoms_traversed': int(n_atoms[imin])},
        'per_od_shares': a.tolist(),
    }


def wallclock():
    """Equal-wall-clock medians from the E135 tripled-budget logs."""
    out = {}
    for net, tag in (('Sioux Falls', 'sf'), ('Anaheim', 'an')):
        curves = {}
        for method in ('flow', 'response'):
            per_seed = []
            for seed in range(5):
                c = json.loads((E135 / f'{tag}_{method}_{seed}_cost.json').read_text())
                t = np.array([l['elapsed_seconds'] for l in c['logs']])
                k = np.array([l['validation']['K_mse'] for l in c['logs']])
                fr = np.array([l['validation']['flow_relative'] for l in c['logs']])
                per_seed.append((t, k, fr))
            curves[method] = per_seed
        med_final = {m: float(np.median([c[0][-1] for c in curves[m]]))
                     for m in curves}
        t_star = min(med_final.values())
        slow = max(med_final, key=med_final.get)
        fast = min(med_final, key=med_final.get)
        at_t = {}
        for m in ('flow', 'response'):
            vals_k, vals_f = [], []
            for t, k, fr in curves[m]:
                kk = float(np.interp(t_star, t, k))
                ff = float(np.interp(t_star, t, fr))
                vals_k.append(kk)
                vals_f.append(ff)
            at_t[m] = {'K_mse_median': float(np.median(vals_k)),
                       'K_mse_q1_q3': [float(np.percentile(vals_k, 25)),
                                       float(np.percentile(vals_k, 75))],
                       'flow_median': float(np.median(vals_f))}
        # per-step cost ratio (first-to-last step slope)
        rate = {m: float(np.median([(c[0][-1] - c[0][0]) /
                                    (curves[m][i][0].size * 100 - 100)
                                    for i, c in enumerate(curves[m])]))
                for m in ('flow', 'response')}
        # crossover: earliest time at which the seed-median response curve
        # falls below the flow method's median final validation error
        med_k_flow_final = float(np.median([c[1][-1] for c in curves['flow']]))
        ts_r = np.linspace(0, max(c[0][-1] for c in curves['response']), 400)
        med_curve_r = np.median([np.interp(ts_r, c[0], c[1]) for c in curves['response']], axis=0)
        below = np.flatnonzero(med_curve_r <= med_k_flow_final)
        crossover_s = float(ts_r[below[0]]) if below.size else None
        out[net] = {
            'final_wallclock_median_s': med_final,
            'equal_wallclock_point_s': t_star,
            'shorter_method': fast,
            'validation_K_mse_at_equal_wallclock': at_t,
            'seconds_per_update_median': rate,
            'per_update_cost_ratio_response_over_flow': rate['response'] / rate['flow'],
            'response_reaches_flow_final_error_at_s': crossover_s,
            'flow_final_median_K_mse': med_k_flow_final,
        }
    return out


if __name__ == '__main__':
    dist = topology_distribution()
    wc = wallclock()
    result = {'topology_distribution': dist, 'wallclock': wc,
              'protocol': {'shares': 'per positive-demand OD, percent of edges in the '
                                     'queried block atoms (the connected-condition set)',
                           'wallclock': ('E135 tripled-budget continuation logs; validation '
                                         'K-MSE interpolated at the shorter method median '
                                         'total time; no retraining')}}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    slim = {k: {kk: vv for kk, vv in v.items() if kk != 'per_od_shares'}
            for k, v in dist.items()}
    print(json.dumps({'topology': slim, 'wallclock': wc}, indent=2))
