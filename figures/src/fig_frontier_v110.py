"""Fig (v110): spatial resolution selected by the error tolerance.

External review of v109 (Fig. 5): the vertical bars with 18 tilted OD
labels were crowded.  Panel (a) is redrawn as a horizontal bar chart ---
one row per query, labels horizontal on the y axis; a query that admits a
single common value at the 5% family tolerance is marked.  Panel (b) is
retained from v109.

Data: experiments/results/e133_v108_pertol/results.json (sioux_rows
carry per-query tolerance sweeps; tau = 0.05 rows).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _repo_root():
    for anc in Path(__file__).resolve().parents:
        if (anc / 'experiments' / 'results').is_dir():
            return anc
    return Path.cwd()


ROOT = _repo_root()
t = json.loads((ROOT / 'experiments/results/e133_v108_pertol/results.json').read_text())

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
MIX, SAME = '#b55c36', '#26747e'
LEVEL_COLORS = {'0.7': '#7fb3a3', '1.0': '#26747e', '1.3': '#14454d'}

rows10 = [r for r in t['sioux_rows'] if r['level'] == 1.0]
for r in rows10:
    sweep = next(s for s in r['tolerance_sweep'] if s['tau'] == 0.05)
    r['seg05'] = sweep['median_segments_per_direction']
    r['single'] = sweep['single_value_ok']
order = sorted(range(len(rows10)), key=lambda i: (rows10[i]['single'],
                                                  rows10[i]['seg05']))
fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.9), width_ratios=[1, 1.12])

# --- Panel A: horizontal bars, one row per query ---
ax = axs[0]
labels, vals, cols = [], [], []
for i in order:
    x = rows10[i]
    labels.append(f"({x['od'][0]},{x['od'][1]})")
    vals.append(x['seg05'])
    cols.append(SAME if x['single'] else MIX)
ypos = np.arange(len(vals))[::-1]
bars = ax.barh(ypos, vals, color=cols, height=.62)
for y, b, x in zip(ypos, bars, [rows10[i] for i in order]):
    if x['single']:
        ax.text(b.get_width() + .12, y, 'one value', va='center',
                fontsize=8.2, color=SAME)
ax.axvline(1, color='#666666', ls=':', lw=.9)
ax.set_yticks(ypos, labels, fontsize=8)
ax.set_xlabel('median segments per direction at $\\tau=5\\%$')
ax.set_xlim(0, 8.6)
ax.set_title('(a)  resolution selected at $\\tau=5\\%$',
             fontsize=10.5, loc='left')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=SAME, label='one common value'),
                   Patch(color=MIX, label='error-controlled profile')],
          fontsize=8, loc='lower right', frameon=True, framealpha=.88,
          edgecolor='none')

# --- Panel B: profile complexity versus tolerance (unchanged) ---
ax = axs[1]
taus = [100 * x for x in t['tolerances']]
for lv in ('0.7', '1.0', '1.3'):
    segs = [t['sioux_levels'][lv][k]['median_segments_per_direction']
            for k in ('tau25', 'tau50', 'tau75', 'tau100')]
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
                xytext=(0, -14), ha='center', fontsize=8.2,
                color='#b55c36')
ax.set_xlabel('tolerance $\\tau$ (%)')
ax.set_ylabel('profile complexity (segments)')
ax.set_yscale('log')
ax.set_yticks([1, 2, 5, 10], ['1', '2', '5', '10'])
ax.set_xticks(taus)
ax.set_title('(b)  profile complexity versus tolerance', fontsize=10.5,
             loc='left')
ax.legend(fontsize=8, loc='lower left', frameon=True, framealpha=.88,
          edgecolor='none')

fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v110_summary_frontier.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v110_summary_frontier.png', dpi=190, bbox_inches='tight')
print('wrote fig_v110_summary_frontier')
