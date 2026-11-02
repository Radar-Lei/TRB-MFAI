"""Fig (v109): response-label budget curve on Sioux Falls.

E137 retrains response supervision with per-query derivative-center
budgets {6, 12, 24, 48, 64} (36--384 centers in total) at the
validation-selected lambda.  Median attribution-column,
geometric-boundary and flow errors on the 96 test paths against the
number of training centers; the shared validation-center cost (96) is
common to every method and excluded from the per-budget cost.

Data: experiments/results/e137_v109_labelbudget/results.json
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _repo_root():
    for anc in Path(__file__).resolve().parents:
        if (anc / 'experiments' / 'results').is_dir():
            return anc
    return Path.cwd()


ROOT = _repo_root()
r = json.loads((ROOT / 'experiments/results/e137_v109_labelbudget/results.json').read_text())
curve = r['curve']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
TEAL, OCHRE, GREY = '#26747e', '#b55c36', '#888888'

fig, ax = plt.subplots(figsize=(4.6, 3.2))
xs = [c['total_centers'] for c in curve]
ax.plot(xs, [c['column_pct'] for c in curve], 'o-', color=TEAL, lw=1.6,
        ms=4.6, label='attribution-column error')
ax.plot(xs, [c['geometry_pct'] for c in curve], 's--', color=OCHRE,
        lw=1.5, ms=4.2, label='geometric-boundary error')
ax.plot(xs, [c['flow_pct'] for c in curve], '^:', color=GREY, lw=1.4,
        ms=4.2, label='flow error')
ax.set_xscale('log')
ax.set_xticks(xs, [str(x) for x in xs])
ax.set_xlabel('training derivative centers (labels)')
ax.set_ylabel('median error over 96 test paths (%)')
ax.set_title('Response supervision vs. label budget', fontsize=10.5)
ax.legend(fontsize=8, frameon=True, framealpha=.9, edgecolor='none')
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v109_labelbudget.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v109_labelbudget.png', dpi=190,
            bbox_inches='tight')
print('wrote fig_v109_labelbudget')
