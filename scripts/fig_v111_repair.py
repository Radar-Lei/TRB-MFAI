"""Regenerate figures/fig_v111_repair.{pdf,png} from E145 results.json.

Visual-QA fixes (2026-11-02): panel (b) 'family max' annotation moved to
the right end of the dashed line (was colliding with the legend);
panel (a) reference curve drawn on top with heavier weight so it stays
visible where the response-supervised curve tracks it.
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
R = json.loads((HERE / 'results/e145_v111_repair/results.json').read_text())
from run_e107_v100_screenline import load_network  # noqa: E402
nodes, coords, g, arcs = load_network()

fig = plt.figure(figsize=(13.2, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05],
                      wspace=0.30, left=0.055, right=0.985, bottom=0.16, top=0.90)
x = np.asarray(R['chain_levels'])
aS = np.asarray(R['accounts_reference'])
af = np.asarray(R['accounts_flow'])
ar = np.asarray(R['accounts_response'])
ef = np.asarray(R['error_flow'])
er = np.asarray(R['error_response'])
ib = R['worst_boundary_index']
fam_max_err = R['family_worst']['flow_error']
TEAL, OCHRE, GREY = '#0f766e', '#b45309', '#6b7280'

ax = fig.add_subplot(gs[0, 0])
ax.plot(x, aS, '-', color=GREY, lw=2.2, label='Reference', zorder=4)
ax.plot(x, af, '-', color=OCHRE, lw=1.5, label='Flow-only', zorder=2)
ax.plot(x, ar, '-', color=TEAL, lw=1.5, label='Response-supervised', zorder=3)
for name, col in (('reference', GREY), ('flow', OCHRE), ('response', TEAL)):
    xs, ys = [], []
    sm = R['segments'][name]
    for i, j2 in sm['chain_segments']:
        xs += [x[i], x[j2]]
        ys += [sm['chain_summary'][i], sm['chain_summary'][j2]]
    ax.plot(xs, ys, linestyle=(0, (4, 2)), color=col, lw=1.0, alpha=0.9, zorder=2)
ax.set_xlabel('Screenline position along the chain (projection level)')
ax.set_ylabel('Boundary attribution (TSTT units)')
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.set_title('(a) Boundary-attribution profile', fontsize=9.5)

ax = fig.add_subplot(gs[0, 1])
ax.plot(x, ef, '-', color=OCHRE, lw=1.6, label='Flow-only')
ax.plot(x, er, '-', color=TEAL, lw=1.6, label='Response-supervised')
ax.axvline(x[ib], color='grey', lw=0.8, ls=':')
ax.annotate('worst boundary\non this chain', xy=(x[ib], ef[ib]),
            xytext=(6, 10), textcoords='offset points', fontsize=7.5,
            color='#374151')
ax.axhline(fam_max_err, color='#9ca3af', lw=0.9, ls='--')
ax.annotate(f'family max ({fam_max_err:.0f})', xy=(x[-1], fam_max_err),
            xytext=(-4, -11), textcoords='offset points', fontsize=7,
            color='#6b7280', ha='right')
ax.fill_between(x, er, ef, where=ef > er, color=TEAL, alpha=0.12,
                label='Repaired error')
ax.set_xlabel('Screenline position along the chain (projection level)')
ax.set_ylabel(r'$|a^{L}(U)-a^{S}(U)|$ (TSTT units)')
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.set_title('(b) Boundary error against the reference', fontsize=9.5)

ax = fig.add_subplot(gs[0, 2])
Uset = set(R['worst_boundary_nodes'])
s, t = R['od']          # one-based display IDs (matches coords/labels)
for (u, v) in g.edges():
    col = '#9ca3af' if (u in Uset) == (v in Uset) else '#111827'
    lw = 0.9 if (u in Uset) == (v in Uset) else 1.6
    ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]],
            color=col, lw=lw, zorder=1)
for v in nodes:
    inside = v in Uset
    ax.plot(*coords[v], 'o', ms=13 if v in (s, t) else 8,
            color='#14b8a6' if inside else '#e5e7eb',
            markeredgecolor='#0f766e' if inside else '#9ca3af',
            markeredgewidth=0.8, zorder=3)
vmax = max(abs(c) for _, c in R['crossing_contributions']) or 1.0
for e, cval in R['crossing_contributions']:
    if abs(cval) < 0.02 * vmax:
        continue
    u, v = arcs[e][0], arcs[e][1]
    dx, dy = coords[v][0] - coords[u][0], coords[v][1] - coords[u][1]
    # cval already carries the leaving(+)/entering(-) orientation: its sign
    # is the signed contribution to the boundary attribution
    ax.annotate('', xy=(coords[u][0] + 0.72 * dx, coords[u][1] + 0.72 * dy),
                xytext=(coords[u][0] + 0.28 * dx, coords[u][1] + 0.28 * dy),
                arrowprops=dict(arrowstyle='-|>', lw=0.5 + 3.5 * abs(cval) / vmax,
                                color='#2563eb' if cval > 0 else '#dc2626',
                                linestyle='-' if cval > 0 else (0, (4, 2)),
                                alpha=0.95), zorder=4)
ax.plot(*coords[s], 'o', ms=15, mfc='none', mec='#0f766e', mew=1.6, zorder=5)
ax.plot(*coords[t], 's', ms=12, mfc='none', mec='#7c2d12', mew=1.6, zorder=5)
ax.annotate(f'{s}', coords[s], xytext=(5, 4), textcoords='offset points', fontsize=8)
ax.annotate(f'{t}', coords[t], xytext=(5, 4), textcoords='offset points', fontsize=8)
ax.set_title('(c) Worst flow-only boundary on the network', fontsize=9.5)
ax.set_xlabel('Longitude (equirectangular, $\\times\\cos\\bar\\phi$)', fontsize=8.5)
ax.set_ylabel('Latitude', fontsize=8.5)
ax.tick_params(labelsize=7.5)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.savefig(ROOT / 'figures/fig_v111_repair.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v111_repair.png', dpi=200, bbox_inches='tight')
print('rewritten fig_v111_repair.{pdf,png}')
