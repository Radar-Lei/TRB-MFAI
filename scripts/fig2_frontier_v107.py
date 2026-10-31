"""Fig 2 (v107): adaptive summary selection and its cost, two panels.

Panel A: maximum absolute summary error over the 18 input-stratified
queries at +/-20% (Eq. normerror), reordered by error, with the 5%
tolerance marking which queries receive one common value.
Panel B: the alternative output's cost as the tolerance varies -- median
minimal contiguous segments per query on Sioux Falls (three load
levels) and per-query segment range with compression on Anaheim.

Data: experiments/results/v104_spatial/coverage.json (Panel A) and
experiments/results/e120_v106_tolerance/results.json (Panel B); numbers
are unchanged from the published tables.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
r = json.loads((ROOT / 'experiments/results/v104_spatial/coverage.json').read_text())
t = json.loads((ROOT / 'experiments/results/e120_v106_tolerance/results.json').read_text())
rows = [x for x in r['rows'] if x['halfwidth'] == .2]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
MIX, SAME = '#b55c36', '#26747e'
GREY = '#9AA0A6'
LEVEL_COLORS = {'0.7': '#7fb3a3', '1.0': '#26747e', '1.3': '#14454d'}

fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.7), width_ratios=[1, 1.12])

# --- Panel A: error ranking ---
ax = axs[0]
ods = [f"({x['od'][0]},{x['od'][1]})" for x in rows]
order = sorted(range(len(rows)), key=lambda i: rows[i]['relative_epsilon'])
for j, i in enumerate(order):
    x = rows[i]
    color = MIX if x['sign'] == 'mixed' else SAME
    ax.barh(j, 100 * x['relative_epsilon'], color=color, height=.62)
ax.set_yticks(range(len(order)), [ods[i] for i in order])
ax.invert_yaxis()
ax.axvline(5, color=MIX, ls='--', lw=1)
ax.set_xlabel('maximum absolute summary error (% of family maximum)')
ax.text(5.6, -0.72, '5% tolerance', fontsize=9, color=MIX, va='center')
ax.set_xlim(0, 56)
ax.set_title('(a)  one-value adequacy', fontsize=10.5, loc='left')

# --- Panel B: tolerance frontier ---
ax = axs[1]
taus = [100 * x for x in t['tolerances']]
for lv in ('0.7', '1.0', '1.3'):
    segs = [t['sioux_levels'][lv][k]['median_segments'] for k in
            ('tau25', 'tau50', 'tau75', 'tau100')]
    ax.plot(taus, segs, 'o-', color=LEVEL_COLORS[lv], lw=1.6, ms=4.6,
            label=f'Sioux Falls, level {lv}')
segs_q = [t['anaheim'][k]['segments_per_query'] for k in ('tau25', 'tau50', 'tau75', 'tau100')]
seg_lo = [min(s) for s in segs_q]
seg_hi = [max(s) for s in segs_q]
comp = [100 * t['anaheim'][k]['median_compression'] for k in ('tau25', 'tau50', 'tau75', 'tau100')]
ax.fill_between(taus, seg_lo, seg_hi, color='#b55c36', alpha=.14, lw=0)
ax.plot(taus, seg_hi, 's--', color='#b55c36', lw=1.4, ms=4.2,
        label='Anaheim (segment range)')
for x_, c_, y_ in zip(taus, comp, seg_hi):
    ax.annotate(f'{c_:.0f}%', (x_, y_), textcoords='offset points',
                xytext=(2, 5), fontsize=8.2, color='#b55c36')
ax.set_xlabel('tolerance $\\tau$ (%)')
ax.set_ylabel('minimal contiguous segments per query')
ax.set_yscale('log')
ax.set_xticks(taus)
ax.set_title('(b)  cost of the adaptive profile', fontsize=10.5, loc='left')
ax.legend(fontsize=8, loc='upper right', frameon=True, framealpha=.88,
          edgecolor='none')

fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v107_summary_frontier.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v107_summary_frontier.png', dpi=190, bbox_inches='tight')
print('wrote fig_v107_summary_frontier')
