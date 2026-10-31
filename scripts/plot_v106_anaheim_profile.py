"""v106 Figures: Anaheim spatial response profiles, main-text and appendix.

External review of v105 (P0): the main text keeps only the representative
query (25,2) -- its account profile and the four-part same-state
decomposition as two side-by-side panels at full width; the profiles of
the remaining five queries move to an appendix figure. Data are unchanged
from v105 (v105_anaheim_profile/profile.json).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / 'experiments/results/v105_anaheim_profile/profile.json').read_text())
rows = r['rows']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
TEAL, OCHRE = '#26747e', '#b55c36'

rep = max(rows, key=lambda x: abs(x['selected_accounts'][-1] - x['selected_accounts'][0]))
rest = [row for row in rows if row['query'] != rep['query']]

# --- main-text figure: representative profile + decomposition ---
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
lv = np.array(rep['progress_levels'])
a = np.array(rep['accounts'])
sel = [int(np.argmin(np.abs(lv - x))) for x in (.25, .5, .75)]
ax = axs[0]
ax.plot(lv, a, color=TEAL, lw=1.8)
ax.plot(lv[sel], a[sel], 'o', color=OCHRE, ms=5.5, zorder=3)
ax.axhline(0, color='.6', lw=.7)
ax.set_title(f"account profile, query ({rep['query'][0]},{rep['query'][1]})", fontsize=10.5)
ax.set_xlabel('free-flow progress')
ax.set_ylabel('account (veh-min/h)')
ax.tick_params(labelsize=8.5)

ax = axs[1]
styles = {'ff_study': (TEAL, '-'), 'ff_bg': (TEAL, '--'),
          'cong_study': (OCHRE, '-'), 'cong_bg': (OCHRE, '--')}
labels = {'ff_study': 'free-flow, study', 'ff_bg': 'free-flow, background',
          'cong_study': 'congestion, study', 'cong_bg': 'congestion, background'}
for key, (color, ls) in styles.items():
    ax.plot(lv, rep['curves'][key], color=color, ls=ls, lw=1.7, label=labels[key])
ax.plot(lv, a, color='k', lw=.9, alpha=.55, label='total account')
ax.axhline(0, color='.6', lw=.7)
ax.plot(lv[sel], a[sel], 'o', color=OCHRE, ms=5.5, zorder=3)
ax.set_xlabel('free-flow progress along the OD direction')
ax.set_ylabel('contribution to account (veh-min/h)')
ax.set_title(f"four-part decomposition, query ({rep['query'][0]},{rep['query'][1]})", fontsize=10.5)
ax.legend(fontsize=8, ncol=2, loc='upper center')
ax.tick_params(labelsize=8.5)
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v106_anaheim_profile.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v106_anaheim_profile.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# --- appendix figure: remaining five profiles ---
fig, axs = plt.subplots(2, 3, figsize=(6.8, 4.3), layout='constrained')
for ax in axs.flat[len(rest):]:
    ax.axis('off')
for i, (ax, row) in enumerate(zip(axs.flat, rest)):
    lv = np.array(row['progress_levels'])
    a = np.array(row['accounts'])
    ax.plot(lv, a, color=TEAL, lw=1.6)
    sel = [int(np.argmin(np.abs(lv - x))) for x in (.25, .5, .75)]
    ax.plot(lv[sel], a[sel], 'o', color=OCHRE, ms=4.5, zorder=3)
    ax.axhline(0, color='.6', lw=.7)
    ax.set_title(f"({row['query'][0]},{row['query'][1]})", fontsize=9.5)
    ax.set_xlabel('free-flow progress', fontsize=8.5)
    if i % 3 == 0:
        ax.set_ylabel('account (veh-min/h)', fontsize=8.5)
    ax.tick_params(labelsize=7.5)
fig.savefig(ROOT / 'figures/fig_v106_anaheim_rest.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v106_anaheim_rest.png', dpi=170, bbox_inches='tight')
print('wrote fig_v106_anaheim_profile (rep', rep['query'], ') and fig_v106_anaheim_rest (5 queries)')
