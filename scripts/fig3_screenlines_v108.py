"""Fig 3 (v108): contrasting screenline accounts, side-by-side panels.

Two straight screenlines for OD (1,19), setting C: the geometric
family's minimum (upstream) and maximum (downstream) accounts under one
positive network-wide contribution. Panels sit side by side at full
width; signed crossing contributions use arrow direction, color and
line style jointly (solid = positive, dashed = negative); the account
value sits in the panel's upper-left corner; principal contributions
are listed under each panel.

Data are unchanged from the v106 figure (e107 screenline records,
e86 columns); the two lines follow different directions and their regions do
not nest; labels A/B avoid an upstream-downstream reading.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, 'experiments')
from run_e107_v100_screenline import load_network
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nodes, coords, g, arcs = load_network()
r = json.loads(Path('experiments/results/e107_v100_screenline.json').read_text())['per_od']['1->19']
columns = np.load('experiments/results/e86_v84_newod.npz')['C']
evidence = {}
P = np.array([coords[v] for v in nodes])


def crossing_progress(U):
    for angle in np.arange(0, 180, 5):
        norm = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        pr = P @ norm
        for t in (np.sort(pr)[:-1] + np.sort(pr)[1:]) / 2:
            S = {v for v, p in zip(nodes, pr) if p < t}
            if S == U or set(nodes) - S == U:
                s_pt, t_pt = np.array(coords[1]), np.array(coords[19])
                seg = t_pt - s_pt
                return float((t - norm @ s_pt) / (norm @ seg))
    raise AssertionError('separating line not found')


cp_min = crossing_progress(set(r['argmin_nodes_1idx']))
cp_max = crossing_progress(set(r['argmax_nodes_1idx']))
assert cp_min < cp_max, (cp_min, cp_max)

fig, axs = plt.subplots(1, 2, figsize=(7.48, 4.15))
panels = [('argmin_nodes_1idx', 'Screenline A', r['phi_min']),
          ('argmax_nodes_1idx', 'Screenline B', r['phi_max'])]
for ax, (tag, title, val) in zip(axs, panels):
    U = set(r[tag])
    crossing = []
    for e, (u, v) in enumerate(arcs):
        if (u in U) != (v in U):
            crossing.append((e, u, v, (1 if u in U else -1) * float(columns[e, 1])))
    crossing.sort(key=lambda z: abs(z[3]), reverse=True)
    total = sum(z[3] for z in crossing)
    assert abs(total - val) < 1e-8
    evidence[tag] = {'phi': total, 'nodes': sorted(U),
                     'crossing_links': [{'from': u, 'to': v, 'signed_contribution': value}
                                        for e, u, v, value in crossing]}
    mag = np.array([abs(z[3]) for z in crossing])
    lw = 1.0 + 2.4 * mag / mag.max()
    for u, v in g.edges:
        ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]],
                color='#c9c9c9', lw=.6, zorder=1)
    for (e, u, v, value), w in zip(crossing, lw):
        ax.annotate('', xy=coords[v], xytext=coords[u],
                    arrowprops={'arrowstyle': '->', 'shrinkA': 2, 'shrinkB': 2,
                                'color': '#b83232' if value < 0 else '#185f96',
                                'lw': w,
                                'linestyle': '-' if value > 0 else (0, (4, 2))},
                    zorder=2)
    labelled = {1, 19} | {u for e, u, w_, val_ in crossing[:3]} | {w_ for e, u, w_, val_ in crossing[:3]}
    for v in nodes:
        x, y = coords[v]
        ax.scatter(x, y, c='#3b77a6' if v in U else '#dddddd', s=54, zorder=3,
                   edgecolors='#444444' if v in (1, 19) else 'none', linewidths=1.4)
        if v in labelled:
            ax.annotate(str(v), (x, y),
                        xytext=({1: (5, 5), 10: (-22, -5), 11: (-20, 7), 15: (-19, -9),
                                 19: (7, 1), 6: (5, 4), 8: (8, -8), 13: (-17, -5),
                                 24: (7, -7)}.get(v, (4, 3))),
                        textcoords='offset points', fontsize=8.8, zorder=4)
    found = False
    for angle in np.arange(0, 180, 5):
        norm = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        pr = P @ norm
        for t in (np.sort(pr)[:-1] + np.sort(pr)[1:]) / 2:
            S = {v for v, p in zip(nodes, pr) if p < t}
            if S == U or set(nodes) - S == U:
                point = norm * t
                d = np.array([-norm[1], norm[0]])
                extent = np.ptp(P, axis=0).max() * 1.2
                line = point[None, :] + np.array([-extent, extent])[:, None] * d
                ax.plot(line[:, 0], line[:, 1], '--', color='#b83232', lw=1.4)
                found = True
                break
        if found:
            break
    assert found
    valtex = ('-' if val < 0 else '') + f'{abs(val):,.1f}'.replace(',', '{,}')
    ax.text(0.015, 0.985, r'$\phi_j(U)=' + valtex + '$', transform=ax.transAxes,
            fontsize=10.5, va='top', ha='left',
            bbox=dict(fc='white', ec='none', alpha=.78, pad=1.6))
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(P[:, 0].min() - .13 * np.ptp(P[:, 0]), P[:, 0].max() + .20 * np.ptp(P[:, 0]))
    ax.set_ylim(P[:, 1].min() - .09 * np.ptp(P[:, 1]), P[:, 1].max() + .16 * np.ptp(P[:, 1]))
    ax.set_aspect('equal')
    ax.axis('off')
    lines = [f'{i}. {u} $\\to$ {v}: {value:+,.1f}' for i, (e, u, v, value) in enumerate(crossing[:3], 1)]
    ax.text(.5, -0.06, 'Principal signed crossing contributions\n' + '\n'.join(lines),
            transform=ax.transAxes, fontsize=8.4, va='top', ha='center',
            linespacing=1.35)

fig.tight_layout(w_pad=1.2)
fig.savefig('figures/fig_screenlines_v108.pdf', bbox_inches='tight')
fig.savefig('figures/fig_screenlines_v108.png', dpi=190, bbox_inches='tight')
Path('revision-stage/v107/SCREENLINE_LINKS.json').parent.mkdir(parents=True, exist_ok=True)
Path('revision-stage/v107/SCREENLINE_LINKS.json').write_text(json.dumps(evidence, indent=2))
print('wrote fig_screenlines_v108; upstream progress', round(cp_min, 3),
      'phi', round(r['phi_min'], 1), '| downstream progress', round(cp_max, 3),
      'phi', round(r['phi_max'], 1))
