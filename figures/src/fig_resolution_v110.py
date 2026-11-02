"""Fig (v110, new): final-explanation error versus reported-segment count.

External review of v109 (item 3.2): the useful plot is not error versus
tolerance but the deliverable trade-off --- on the x axis the number of
segments a reader must absorb, on the y axis the final error of the
delivered summary against the reference explanation.  Two panels (Sioux
Falls / Anaheim), two learning methods (flow-only ochre, response teal),
each a tau-sweep curve tau in {0.5,1,2.5,5,10}% of the learned family
scale; the grey dotted curve is the reference compression (the
compression error of the reference profile itself at each tolerance,
identical for both methods, plotted at its own segment count from
E144).  Common per-query normalization: percent of the reference
family maximum, mean within path, median over paths.

Data: experiments/results/e140_v110_resolution/results.json
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _repo_root():
    for anc in Path(__file__).resolve().parents:
        if (anc / 'experiments' / 'results').is_dir():
            return anc
    return Path.cwd()


ROOT = _repo_root()
d = json.loads((ROOT / 'experiments/results/e140_v110_resolution/results.json').read_text())['table']
TAUS = ('0.005', '0.01', '0.025', '0.05', '0.1')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
TEAL, OCHRE, GREY = '#26747e', '#b55c36', '#8a8f94'

fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.5))
for ax, net, title in ((axs[0], 'sioux_falls', 'Sioux Falls (96 paths)'),
                       (axs[1], 'anaheim', 'Anaheim (48 paths)')):
    for method, color, name in (('flow', OCHRE, 'flow supervision'),
                                ('response', TEAL, 'response supervision')):
        e = d[net][method]
        L = [e[f'tau_{t}']['L'] for t in TAUS]
        Et = [e[f'tau_{t}']['E_total'] for t in TAUS]
        ax.plot(L, Et, 'o-', color=color, lw=1.7, ms=4.8, label=name)
        for k, (t, x, y) in enumerate(zip(TAUS, L, Et)):
            dx, dy, ha = (4, -9, 'left')
            # Anaheim: flow and response points nearly coincide, so lift the
            # response labels clear of the flow labels and the floor curve
            if net == 'anaheim' and method == 'response':
                dy = 8 if k in (0, 2) else -14
            ax.annotate(f'{100 * float(t):g}%', (x, y),
                        textcoords='offset points', xytext=(dx, dy),
                        ha=ha, fontsize=7.2, color=color)
    fl = [d[net]['flow'][f'tau_{t}']['E_summary_ref_floor'] for t in TAUS]
    Lf = [d[net]['L_ref'][t] for t in TAUS] if 'L_ref' in d[net] else \
        [v['L_ref'] for v in json.loads(
            (ROOT / 'experiments/results/e144_v110_refsegments/'
             'results.json').read_text())[net].values()]
    ax.plot(Lf, fl, ':', color=GREY, lw=1.5, marker='x', ms=4.5,
            label='reference compression')
    ax.set_xlabel('reported segments $L$')
    ax.set_ylabel('final error $E_{\\mathrm{total}}$ (%)')
    ax.set_title(title, fontsize=10.5)
    ax.margins(x=.12, y=.14)
axs[0].legend(fontsize=8, loc='upper right', frameon=True, framealpha=.9,
              edgecolor='none')
fig.tight_layout()
fig.savefig(ROOT / 'figures/fig_v110_resolution.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v110_resolution.png', dpi=190,
            bbox_inches='tight')
print('wrote fig_v110_resolution')
