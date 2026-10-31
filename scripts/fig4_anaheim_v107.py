"""Fig 4 (v107): Anaheim spatial response profile, representative query.

Two panels at full width: the account profile over the distance-based
family with the three prespecified positions, and the four-part
same-state decomposition. Styling: shared x-range, bold black total
curve, two line styles (study solid / background dashed) crossed with
two markers and colors (free-flow teal circles / congestion ochre
squares), legend outside the data area, enlarged fonts.

Data are unchanged from v106 (v105_anaheim_profile/profile.json).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
r = json.loads((ROOT / 'experiments/results/v105_anaheim_profile/profile.json').read_text())
rows = r['rows']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                     'axes.spines.top': False, 'axes.spines.right': False})
TEAL, OCHRE = '#26747e', '#b55c36'

rep = max(rows, key=lambda x: abs(x['selected_accounts'][-1] - x['selected_accounts'][0]))
fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.6))
lv = np.array(rep['progress_levels'])
a = np.array(rep['accounts'])
sel = [int(np.argmin(np.abs(lv - x))) for x in (.25, .5, .75)]

ax = axs[0]
ax.plot(lv, a, color=TEAL, lw=2.0)
ax.plot(lv[sel], a[sel], 'o', color=OCHRE, ms=6, zorder=3)
ax.axhline(0, color='.6', lw=.7)
ax.set_xlim(0, 1)
ax.set_title(f"account profile, query ({rep['query'][0]},{rep['query'][1]})", fontsize=11)
ax.set_xlabel('free-flow progress')
ax.set_ylabel('account (veh-min/h)')
ax.tick_params(labelsize=9)

ax = axs[1]
styles = {'ff_study': (TEAL, '-', 'o'), 'ff_bg': (TEAL, '--', 'o'),
          'cong_study': (OCHRE, '-', 's'), 'cong_bg': (OCHRE, '--', 's')}
labels = {'ff_study': 'free-flow, study', 'ff_bg': 'free-flow, background',
          'cong_study': 'congestion, study', 'cong_bg': 'congestion, background'}
for key, (color, ls, mk) in styles.items():
    kw = {} if ls == '-' else {'dashes': (5, 2.2)}
    ax.plot(lv, rep['curves'][key], color=color, ls=ls, lw=1.9, marker=mk,
            markevery=5, ms=4.6, label=labels[key], **kw)
ax.plot(lv, a, color='k', lw=2.6, alpha=.9, label='total account', zorder=2)
ax.axhline(0, color='.6', lw=.7)
ax.plot(lv[sel], a[sel], 'o', color=OCHRE, ms=6, zorder=3)
ax.set_xlim(0, 1)
ax.set_xlabel('free-flow progress along the OD direction')
ax.set_ylabel('contribution to account (veh-min/h)')
ax.set_title(f"four-part decomposition, query ({rep['query'][0]},{rep['query'][1]})", fontsize=11)
ax.tick_params(labelsize=9)
pos_dot = plt.Line2D([], [], color=OCHRE, marker='o', ls='none', ms=6,
                     label='prespecified positions')
handles, labs = ax.get_legend_handles_labels()
handles.append(pos_dot)
labs.append('prespecified positions')
fig.legend(handles, labs, fontsize=8.8, ncol=5, loc='lower center',
           bbox_to_anchor=(.5, -0.055), frameon=False, columnspacing=1.2,
           handletextpad=.5)
fig.tight_layout(rect=(0, .045, 1, 1))
fig.savefig(ROOT / 'figures/fig_v107_anaheim_profile.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v107_anaheim_profile.png', dpi=190, bbox_inches='tight')
plt.close(fig)

# appendix figure: remaining five profiles (unchanged styling, enlarged fonts)
rest = [row for row in rows if row['query'] != rep['query']]
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.4), layout='constrained')
for ax in axs.flat[len(rest):]:
    ax.axis('off')
for i, (ax, row) in enumerate(zip(axs.flat, rest)):
    lv = np.array(row['progress_levels'])
    a = np.array(row['accounts'])
    ax.plot(lv, a, color=TEAL, lw=1.7)
    sel = [int(np.argmin(np.abs(lv - x))) for x in (.25, .5, .75)]
    ax.plot(lv[sel], a[sel], 'o', color=OCHRE, ms=5, zorder=3)
    ax.axhline(0, color='.6', lw=.7)
    ax.set_title(f"({row['query'][0]},{row['query'][1]})", fontsize=10)
    ax.set_xlabel('free-flow progress', fontsize=9)
    if i % 3 == 0:
        ax.set_ylabel('account (veh-min/h)', fontsize=9)
    ax.tick_params(labelsize=8)
fig.savefig(ROOT / 'figures/fig_v107_anaheim_rest.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v107_anaheim_rest.png', dpi=190, bbox_inches='tight')
print('wrote fig_v107_anaheim_profile (rep', rep['query'], ') and fig_v107_anaheim_rest')
