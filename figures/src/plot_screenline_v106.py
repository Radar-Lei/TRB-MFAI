"""v106 Figure (screenlines): vertical two-panel contrast for OD (1,19).

External review of v105 (P0): stack the two panels vertically at full
width, emphasize the actual crossing links (all of them, width encoding
|contribution|) and the key nodes, rename the panels Upstream/Downstream
screenline (the sign stays on the reported account), and drop the in-figure
suptitle (the caption carries the framing). Data and regions are unchanged
from v103: the prespecified geometric family's minimum and maximum
screenline accounts for OD (1,19), setting C.
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

umin = set(r['argmin_nodes_1idx'])
umax = set(r['argmax_nodes_1idx'])
# upstream/downstream = the position of the separating line itself along
# the OD corridor: the crossing progress of the separating line with the
# straight origin-destination segment (codex round-1: the earlier naming
# used the farthest-node hop progress, which orders these two lines the
# other way and is not what the panel names mean)
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

cp_min = crossing_progress(umin)
cp_max = crossing_progress(umax)
assert cp_min < cp_max, (cp_min, cp_max)
panels = [('argmin_nodes_1idx', 'Upstream screenline', r['phi_min'], 'phi_min'),
          ('argmax_nodes_1idx', 'Downstream screenline', r['phi_max'], 'phi_max')]

fig, axs = plt.subplots(2, 1, figsize=(5.8, 6.0))
for ax, (tag, title, val, key) in zip(axs, panels):
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
    P = np.array([coords[v] for v in nodes])
    for u, v in g.edges:
        ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]],
                color='#c9c9c9', lw=.6, zorder=1)
    for (e, u, v, value), w in zip(crossing, lw):
        ax.annotate('', xy=coords[v], xytext=coords[u],
                    arrowprops={'arrowstyle': '->', 'shrinkA': 2, 'shrinkB': 2,
                                'color': '#b83232' if value < 0 else '#185f96', 'lw': w},
                    zorder=2)
    labelled = {1, 19} | {u for e, u, w_, val_ in crossing[:3]} | {w_ for e, u, w_, val_ in crossing[:3]}
    for v in nodes:
        x, y = coords[v]
        ax.scatter(x, y, c='#3b77a6' if v in U else '#dddddd', s=52, zorder=3,
                   edgecolors='#444444' if v in (1, 19) else 'none', linewidths=1.4)
        if v in labelled:
            ax.annotate(str(v), (x, y),
                        xytext=({1: (5, 5), 10: (-22, -5), 11: (-20, 7), 15: (-19, -9),
                                 19: (7, 1), 6: (5, 4), 8: (8, -8), 13: (-17, -5),
                                 24: (7, -7)}.get(v, (4, 3))),
                        textcoords='offset points', fontsize=8.6, zorder=4)
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
    ax.set_title(f'{title}: ' + r'$\phi_j(U)=' + valtex + '$', fontsize=10.5)
    ax.set_xlim(P[:, 0].min() - .13 * np.ptp(P[:, 0]), P[:, 0].max() + .20 * np.ptp(P[:, 0]))
    ax.set_ylim(P[:, 1].min() - .09 * np.ptp(P[:, 1]), P[:, 1].max() + .09 * np.ptp(P[:, 1]))
    ax.set_aspect('equal')
    ax.axis('off')
    lines = [f'{i}. {u} $\\to$ {v}: {value:+,.1f}' for i, (e, u, v, value) in enumerate(crossing[:3], 1)]
    ax.text(.30, -0.04, 'Principal signed crossing contributions\n' + '\n'.join(lines),
            transform=ax.transAxes, fontsize=8.6, va='top', linespacing=1.35)

fig.tight_layout(h_pad=2.4)
fig.savefig('figures/fig_screenlines_v106.pdf', bbox_inches='tight')
fig.savefig('figures/fig_screenlines_v106.png', dpi=160, bbox_inches='tight')
Path('revision-stage/v106/SCREENLINE_LINKS.json').write_text(json.dumps(evidence, indent=2))
print('wrote fig_screenlines_v106; upstream (min-account) line crossing progress',
      round(cp_min, 3), 'phi', round(r['phi_min'], 1),
      '| downstream (max-account) line crossing progress',
      round(cp_max, 3), 'phi', round(r['phi_max'], 1))
