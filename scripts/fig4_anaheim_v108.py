"""Fig (v108): Anaheim spatial response profile with its delivered summary.

Two panels at full width. Left: the account profile over the
distance-based family (grey), the minimum four-segment summary at the
family tolerance overlaid as a teal step with a +/-epsilon band, and
the three prespecified positions. Right: the four-part same-state
decomposition (styling unchanged from v107).

Data: experiments/results/v105_anaheim_profile/profile.json and
experiments/results/e126_v108_persdirection/results.json (query (25,2)).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
prof = json.loads((ROOT / 'experiments/results/v105_anaheim_profile/profile.json').read_text())
e126 = json.loads((ROOT / 'experiments/results/e126_v108_persdirection/results.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                     'axes.spines.top': False, 'axes.spines.right': False})
TEAL, OCHRE, GREY = '#26747e', '#b55c36', '#8a8f94'

rep = max(prof['rows'], key=lambda x: abs(x['selected_accounts'][-1] - x['selected_accounts'][0]))
seg_row = next(x for x in e126['anaheim']['rows'] if tuple(x['query']) == tuple(rep['query']))
fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.7))
lv = np.array(rep['progress_levels'])
a = np.array(rep['accounts'])
sel = [int(np.argmin(np.abs(lv - x))) for x in (.25, .5, .75)]
eps = seg_row['eps_absolute']
ranges = seg_row['segment_progress_ranges']
betas = [d['beta'] for d in seg_row['segments_detail']]

# --- Panel A: profile + delivered segment summary ---
ax = axs[0]
ax.fill_between([0, 1], [-eps, -eps], [eps, eps], color=GREY, alpha=0, lw=0)
for (lo, hi), beta in zip(ranges, betas):
    xs = [lo, hi] if hi > lo else [lo - 0.006, hi + 0.006]
    ax.fill_between(xs, beta - eps, beta + eps, color=TEAL, alpha=.13, lw=0, zorder=1)
    ax.plot(xs, [beta, beta], color=TEAL, lw=2.6, solid_capstyle='butt', zorder=3)
    ax.plot([hi, hi], [beta - eps, beta + eps], color=TEAL, lw=.9, alpha=.55, zorder=2)
ax.plot(lv, a, color=GREY, lw=1.1, zorder=2, label='account profile')
ax.plot(lv[sel], a[sel], 'o', color=OCHRE, ms=6, zorder=4)
ax.axhline(0, color='.6', lw=.7)
ax.set_xlim(0, 1)
ax.set_ylim(min(a.min(), -1.2 * eps), a.max() + 1.2 * eps)
ax.set_title(f"account profile and delivered summary, query ({rep['query'][0]},{rep['query'][1]})", fontsize=10)
ax.set_xlabel('free-flow progress')
ax.set_ylabel('account (veh-min/h)')
ax.tick_params(labelsize=9)
step_line = plt.Line2D([], [], color=TEAL, lw=2.6, label='4-segment summary')
band_patch = plt.Rectangle((0, 0), 1, 1, color=TEAL, alpha=.13, lw=0,
                           label=f'$\\pm\\epsilon$ band ($\\epsilon$={eps:.0f})')
pos_dot = plt.Line2D([], [], color=OCHRE, marker='o', ls='none', ms=6,
                     label='prespecified positions')
ax.legend(handles=[step_line, band_patch, plt.Line2D([], [], color=GREY, lw=1.1,
                                                     label='account profile'), pos_dot],
          fontsize=8, loc='upper left', frameon=True, framealpha=.88, edgecolor='none')

# --- Panel B: decomposition (unchanged from v107) ---
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
ax.set_title(f"four-part decomposition, query ({rep['query'][0]},{rep['query'][1]})", fontsize=10)
ax.tick_params(labelsize=9)
ax.legend(fontsize=8, loc='upper left', frameon=True, framealpha=.88, edgecolor='none')

fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v108_anaheim_profile.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v108_anaheim_profile.png', dpi=190, bbox_inches='tight')
print('wrote fig_v108_anaheim_profile')
