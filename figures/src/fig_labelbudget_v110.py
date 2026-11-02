"""Fig (v110): response-label budget curve on Sioux Falls.

As fig_labelbudget_v109 with two review-driven changes (external review
of v109, Fig. 7): the flow-only column error enters as a reference line,
so the fraction of the improvement recovered at each budget is directly
readable; and the near-zero flow-error curve moves to the caption (it
carried no information on the shared axis).

Data: experiments/results/e137_v109_labelbudget/results.json
(flow-only column error from e134 via e139 merge).
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
flow_col = json.loads(
    (ROOT / 'experiments/results/e139_v110_wnorm/results.json').read_text()
)['final']['flow']['column_pct']
flow_geo = json.loads(
    (ROOT / 'experiments/results/e139_v110_wnorm/results.json').read_text()
)['final']['flow']['geometry_pct']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
TEAL, OCHRE, GREY = '#26747e', '#b55c36', '#888888'

fig, ax = plt.subplots(figsize=(4.6, 3.2))
xs = [c['total_centers'] for c in curve]
ax.plot(xs, [c['column_pct'] for c in curve], 'o-', color=TEAL, lw=1.6,
        ms=4.6, label='attribution-column error')
ax.plot(xs, [c['geometry_pct'] for c in curve], 's--', color=OCHRE,
        lw=1.5, ms=4.2, label='family-boundary error')
ax.axhline(flow_col, color=TEAL, ls=':', lw=1.2)
ax.axhline(flow_geo, color=OCHRE, ls=':', lw=1.2)
ax.text(xs[0], flow_col + .35, 'flow-only level', fontsize=8,
        color=TEAL, va='bottom')
ax.set_xscale('log')
ax.set_xticks(xs, [str(x) for x in xs])
ax.set_xlabel('training derivative centers (labels)')
ax.set_ylabel('median error over 96 test paths (%)')
ax.set_ylim(0, 22.6)
ax.set_title('Response supervision vs. label budget', fontsize=10.5)
ax.legend(fontsize=8, frameon=True, framealpha=.9, edgecolor='none',
          loc='upper right')
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v110_labelbudget.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v110_labelbudget.png', dpi=190,
            bbox_inches='tight')
print('wrote fig_v110_labelbudget')
