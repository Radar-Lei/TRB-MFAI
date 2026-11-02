"""Figure for E147: label-efficiency curves with subset IQR bands.

Reads experiments/results/e147_v111_labelrep/results.json (medians and
per-replication values) and writes figures/fig_v111_labelbudget.{pdf,png}.
Style follows the paper's v-series figures (teal column error, ochre
family-boundary error, grey reference levels, log x axis).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = json.loads((HERE / 'results/e147_v111_labelrep/results.json').read_text())

TEAL, OCHRE, GREY = '#0f766e', '#b45309', '#6b7280'

curve = R['curve']
x = np.array([c['total_centers'] for c in curve])
col = np.array([c['column_pct']['median'] for c in curve])
col_q1 = np.array([c['column_pct']['q1'] for c in curve])
col_q3 = np.array([c['column_pct']['q3'] for c in curve])
fam = np.array([c['geometry_pct']['median'] for c in curve])
fam_q1 = np.array([c['geometry_pct']['q1'] for c in curve])
fam_q3 = np.array([c['geometry_pct']['q3'] for c in curve])
lvl = R['levels']

fig, ax = plt.subplots(figsize=(5.6, 3.7))
ax.fill_between(x, col_q1, col_q3, color=TEAL, alpha=0.14, lw=0)
ax.fill_between(x, fam_q1, fam_q3, color=OCHRE, alpha=0.14, lw=0)
ax.plot(x, col, '-o', color=TEAL, lw=1.8, ms=4.5, label='Column error')
ax.plot(x, fam, '-s', color=OCHRE, lw=1.8, ms=4.5,
        label='Family-boundary error')
ax.axhline(lvl['column_pct']['flow_reference'], color=TEAL, ls=':', lw=1.1)
ax.axhline(lvl['geometry_pct']['flow_reference'], color=OCHRE, ls=':', lw=1.1)
ax.annotate('flow-only level', xy=(x[0], lvl['column_pct']['flow_reference']),
            xytext=(4, 5), textcoords='offset points', fontsize=8, color=GREY)
ax.set_xscale('log')
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in x])
ax.minorticks_off()
ax.set_xlabel('Training derivative centers')
ax.set_ylabel('Median error (%)')
ax.legend(frameon=True, framealpha=0.85, facecolor='white', edgecolor='none',
          fontsize=9, loc='upper right')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v111_labelbudget.pdf')
fig.savefig(ROOT / 'figures/fig_v111_labelbudget.png', dpi=300)
print('written fig_v111_labelbudget.{pdf,png}')
