"""Fig (v110): explanation-flow trade-off across six supervision objectives.

As fig_pareto_v109, with the weighted-response curve replaced by the
scale-normalized objective of E139 (its own reference scale v_j), and the
flow-error axes relabeled at two-decimal ticks (external review of v109,
Fig. 6: tick crowding).  The normalized weighted objective now trains
across the whole low-to-mid weight range, so its curve spans the frontier
band instead of collapsing to the initial checkpoint at large weights.

Data: experiments/results/e139_v110_wnorm/results.json (five methods
merged from e134 with exact-median gates + normalized weighted response).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

def _repo_root():
    for anc in Path(__file__).resolve().parents:
        if (anc / 'experiments' / 'results').is_dir():
            return anc
    return Path.cwd()


ROOT = _repo_root()
r = json.loads((ROOT / 'experiments/results/e139_v110_wnorm/results.json').read_text())
final = r['final']
pareto = r['pareto']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
COLORS = {'flow': '#444444', 'response': '#26747e', 'wresponse': '#4f7cab',
          'kboundary': '#7a5ea0', 'edge': '#b55c36', 'cut': '#d08a5e'}
NAMES = {'flow': 'Flow only', 'response': 'Response',
         'wresponse': 'Weighted response (own scale)', 'kboundary': 'Boundary seminorm',
         'edge': 'Endpoint-edge', 'cut': 'Endpoint-cut'}

fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.7), sharex=True)
for ax, metric, ylab in ((axs[0], 'geometry_pct',
                          'family-boundary error (%)'),
                         (axs[1], 'boundary_pct',
                          'unrestricted boundary error (%)')):
    for method in ('response', 'wresponse', 'kboundary', 'edge', 'cut'):
        pts = [(p['flow_pct'], p[metric]) for p in pareto
               if p['method'] == method]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        hollow = (method == 'cut')
        ax.plot(xs, ys, 'o-', color=COLORS[method],
                ms=7.6 if hollow else 4.4, lw=1.1,
                alpha=.95 if hollow else .72, label=NAMES[method],
                mfc='none' if hollow else None)
        star = final[method]
        # the endpoint-cut star sits inside a larger open star: the two
        # selected weights nearly coincide on both panels
        ax.plot(star['flow_pct'], star[metric], '*', color=COLORS[method],
                ms=20 if method == 'cut' else 14, mec='white', mew=.7,
                zorder=4 if method == 'cut' else 5,
                mfc='none' if hollow else None)
    fl = final['flow']
    ax.plot(fl['flow_pct'], fl[metric], 's', color=COLORS['flow'], ms=6,
            label=NAMES['flow'], zorder=5)
    ax.margins(x=.10, y=.10)
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3f}'))
    ax.set_xlabel('flow error (%)')
    ax.set_ylabel(ylab)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=8, ncol=3,
           loc='lower center', frameon=False, handlelength=1.4,
           columnspacing=1.6, borderpad=.2)
fig.savefig(ROOT / 'figures/fig_v110_pareto.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/fig_v110_pareto.png', dpi=190,
            bbox_inches='tight')
print('wrote fig_v110_pareto')
