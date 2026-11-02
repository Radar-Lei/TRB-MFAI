"""Fig (v109): validation convergence of the extended-budget training.

E135 repeats the frozen flow/response finetuning with the continuation
budget tripled from 1200 to 3600 updates.  One panel per network:
seed-median validation boundary error (K-MSE, log scale) and validation
flow error (relative, log scale) against training step, with the
previous 1200-step budget marked.  The curves answer whether the
flow-vs-response comparison is an artifact of the training cap.

Data: experiments/results/e135_v109_extbudget/{sf,an}_{flow,response}_{s}_cost.json
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
D = ROOT / 'experiments/results/e135_v109_extbudget'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
TEAL, OCHRE = '#26747e', '#b55c36'

fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.1))
for ax, net, title, lab_y in ((axs[0], 'sf', 'Sioux Falls', 0.055),
                              (axs[1], 'an', 'Anaheim', None)):
    for method, color in (('flow', OCHRE), ('response', TEAL)):
        curves_k, curves_f = [], []
        for seed in range(5):
            h = json.loads((D / f'{net}_{method}_{seed}_cost.json').read_text())
            curves_k.append([l['validation']['K_mse'] for l in h['logs']])
            curves_f.append([l['validation']['flow_relative']
                             for l in h['logs']])
        steps = [l['step'] for l in h['logs']]
        k = np.median(curves_k, axis=0)
        f = np.median(curves_f, axis=0)
        ax.plot(steps, k, '-', color=color, lw=1.6,
                label=f'{method} — boundary error')
        ax.plot(steps, f, '--', color=color, lw=1.3,
                label=f'{method} — flow error')
    ax.axvline(1200, color='#888888', ls=':', lw=1)
    # label position: below the flow-K curve (empty band) on Sioux Falls,
    # at the top edge on Anaheim, so neither placement crosses a curve
    ylab_pos = lab_y if lab_y is not None else ax.get_ylim()[1]
    va = 'center' if lab_y is not None else 'top'
    ax.text(1206, ylab_pos, 'previous budget', fontsize=7.6,
            color='#666666', va=va, ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.85,
                      pad=1.5))
    ax.set_yscale('log')
    ax.set_xlabel('continuation updates')
    ax.set_title(title, fontsize=10.5)
axs[0].set_ylabel('validation error (log)')
axs[0].legend(fontsize=7.2, loc='center', frameon=True,
              framealpha=.9, edgecolor='none')
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v109_convergence.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v109_convergence.png', dpi=190,
            bbox_inches='tight')
print('wrote fig_v109_convergence')
