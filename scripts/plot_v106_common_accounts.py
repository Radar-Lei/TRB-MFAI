"""v106 Figures: split the v105 common-account figure into two single-column figures.

External review of v105 (P0): the main text keeps only the error-ranking
panel; the complete normalized-interval panel moves to the appendix
(Section coverage). Data and colours are unchanged from v105.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / 'experiments/results/v104_spatial/coverage.json').read_text())
rows = [x for x in r['rows'] if x['halfwidth'] == .2]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
MIX, SAME = '#b55c36', '#26747e'
ods = [f"({x['od'][0]},{x['od'][1]})" for x in rows]

# --- main-text figure: error ranking only ---
fig, ax = plt.subplots(figsize=(4.7, 4.5))
order = sorted(range(len(rows)), key=lambda i: rows[i]['relative_epsilon'])
for j, i in enumerate(order):
    x = rows[i]
    color = MIX if x['sign'] == 'mixed' else SAME
    ax.barh(j, 100 * x['relative_epsilon'], color=color, height=.62)
ax.set_yticks(range(len(order)), [ods[i] for i in order])
ax.invert_yaxis()
ax.axvline(5, color=MIX, ls='--', lw=1)
ax.set_xlabel('Maximum absolute summary error (% of family maximum)')
ax.text(6, .3, '5% tolerance', fontsize=9, color=MIX)
ax.set_xlim(0, 56)
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v106_summary_error_ranking.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v106_summary_error_ranking.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# --- appendix figure: complete normalized intervals ---
fig, ax = plt.subplots(figsize=(5.8, 4.6))
for i, x in enumerate(rows):
    color = MIX if x['sign'] == 'mixed' else SAME
    m = max(abs(x['geometric_min']), abs(x['geometric_max']))
    ax.plot([100 * x['geometric_min'] / m, 100 * x['geometric_max'] / m], [i, i],
            color=color, lw=3)
    ax.plot(100 * x['beta'] / m, i, 'o', color=color, ms=4)
ax.axvline(0, color='.6', lw=.8)
ax.set_yticks(range(len(rows)), ods)
ax.invert_yaxis()
ax.set_xlabel('Account interval and midpoint (% of family maximum)')
ax.set_ylabel('OD query')
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v106_coverage_intervals.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v106_coverage_intervals.png', dpi=170, bbox_inches='tight')
print('wrote fig_v106_summary_error_ranking and fig_v106_coverage_intervals')
