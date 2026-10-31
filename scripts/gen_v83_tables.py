"""Extend the versioned v82 sections using actual v83 coverage results."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'experiments/results';P=ROOT/'paper'
x=json.loads((R/'e84_v83_coverage.json').read_text())
v82=json.loads((R/'e83_v82_validation.json').read_text())
old=json.loads((R/'e79_bgreroute.json').read_text())
a=np.load(R/'e84_v83_coverage.npz')
base=x['C_k10_theta1']['per_od'];exp=x['C_k20_theta1']['per_od']
order=['17->10','10->17','16->10','10->16','15->10','10->15','10->11','11->10']
def od(k):return '$('+k.replace('->',',')+')$'
def fmt(v,digits=4):return f'{v:,.{digits}f}'
def maxchange(field):return max(abs(exp[k][field]-base[k][field]) for k in order)
def maxrel(field):return max(abs(exp[k][field]/base[k][field]-1) for k in order)
Bmax=max(r['C_11_vs_41_relative_l2'] for r in x['B']['refinement_per_od'].values())
Bpp=max(r['spread_11_vs_41_abs_pp'] for r in x['B']['refinement_per_od'].values())
main=(P/'results_main_v82.tex').read_text()
main=main.replace('study demands. Appendix','study demands, selected as the eight largest published demands. All involve\nnode $10$, so the comparison is local rather than a representative OD sample. Appendix')
# Replace each B entry in the main table, preserving all other fields.
lines=[]
for line in main.splitlines():
 if line.startswith('$(') and len(line.split(' & '))==6:
  cells=line.split(' & ');key=cells[0].strip('$()').replace(',','->')
  cells[3]=f"${x['B']['per_od'][key]['spread_pct']:.2f}$";line=' & '.join(cells)
 lines.append(line)
main='\n'.join(lines)+'\n'
main=main.replace('$34.18\\%$',f"${x['B']['per_od']['10->16']['spread_pct']:.2f}\\%$")
main=main.replace('$1.26\\%$',f"${x['B']['per_od']['10->11']['spread_pct']:.2f}\\%$")
main=main.replace('C uses $41$-node implicit sensitivities.','B and C use $41$ quadrature points and implicit sensitivities.')
main=main.replace('the originally low-range OD', 'the zero-background low-range OD')
main=main.replace('query ODs. Values have the outcome\'s numerical units.','query ODs. Values use the TSTT units and one-hour reporting convention\ndefined in Appendix~\\ref{supp:networks}.')
main=main.replace('($41$ nodes)','($41$ quadrature points)')
insert='''
For OD $(17,10)$, the normalized unrestricted range decreases from
$59.95\\%$ in A to $54.77\\%$ in C, while its absolute range increases
from $1.76\\times10^4$ to $1.23\\times10^5$ TSTT units, about sevenfold.
The origin-cut denominator also increases; a lower percentage therefore
does not imply a smaller absolute boundary difference.
'''
main=main.replace('\\subsection{Separating the background route response}',insert+'\n\\subsection{Separating the background route response}')
anchor='For two specified separating sets, Eq.~\\eqref{eq:cutdiff} makes their'
bound=r'''For a concrete boundary comparison, retain the connected regions
$U=\{10\}$ and $U'=\{10,17\}$ from the zero-background illustration.
Their complements are also connected, and both separate the two queries
in Table~\ref{tab:bg_boundary}. These are specified local regions, rather
than regions selected to attain the range. With
$g\in\{\mathrm{study},\mathrm{bg},\mathrm{tot}\}$,
\[
\Delta\phi_j^{(g)}
=-\bigl(\mathbf1_{U'}-\mathbf1_U\bigr)^\top B\widetilde C_j^{(g)},
\qquad
\Delta\phi_j^{\mathrm{tot}}=\Delta\phi_j^{\mathrm{study}}+\Delta\phi_j^{\mathrm{bg}}.
\]
\begin{table}[pos=tb]
\caption{The same connected boundary enlargement in setting C, $\theta=1$,
$41$ quadrature points. Signed accounts use the TSTT numerical units of
Appendix~\ref{supp:networks}. Study includes all eight query ODs.}
\label{tab:bg_boundary}
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrr}
\toprule
OD & $\phi_j(U)$ & $\phi_j(U')$ & $\Delta\phi^{\mathrm{study}}$ & $\Delta\phi^{\mathrm{bg}}$ & $\Delta\phi^{\mathrm{tot}}$\\
\midrule
'''
for k in ['10->16','10->11']:
 r=base[k]['boundary_10_to_10_17']
 bound+=od(k)+' & '+' & '.join('$'+fmt(r[f],1)+'$' for f in ['phi_U','phi_Uprime','delta_study','delta_background','delta_total'])+r' \\'+'\n'
bound+=r'''\bottomrule
\end{tabular*}
\end{table}
For $(10,16)$, the enlargement changes the account by $-3{,}361.1$,
comprising $-1{,}305.0$ from study response and $-2{,}056.1$ from
background response. This signed decomposition concerns the same pair of
boundaries at one joint equilibrium along the demand path.

'''
main=main.replace(anchor,bound+anchor)
main=main.replace('The background $21\\to41$ refinement',f'''For frozen background, $11\\to41$ quadrature points change full
attribution columns by at most ${100*Bmax:.4f}\\%$ and normalized ranges
by at most ${Bpp:.4f}$ percentage points. The main table uses the refined
$41$-point results for both background settings.
The re-routing $21\\to41$ refinement''')
main=main.replace('At all $41$ nodes','At all $41$ quadrature points')
main=main.replace('same $41$ zero-background\nnodes','same $41$ zero-background\nquadrature points')
main+=r'''
\subsection{Full-background path-set and dispersion sensitivity}
We expand every truncated OD path set in setting C from ten to twenty
candidates, keeping the demand path, all background demands, costs and
$41$ quadrature points fixed. All $528$ ODs are truncated in this network,
so the total candidate count increases from $5{,}280$ to $10{,}560$.
This compares different fixed-path logit models, rather than establishing
convergence to unrestricted SUE. Across eight queries, origin-cut accounts
change by at most $'''+f'{100*maxrel("phi_s"):.3f}'+r'''\%$; unrestricted and connected
range percentages change by at most $'''+f'{maxchange("spread_pct"):.3f}'+r'''$ and $'''+f'{maxchange("connected_range_pct"):.3f}'+r'''$ percentage
points. The three largest internal-node terms have the same ordered node
IDs for every query; their signed values and absolute ranges are reported
in Appendix~\ref{supp:coverage}. The broad background pattern persists.

With ten paths per OD and $\theta\in\{0.5,1,2\}$, unrestricted ranges
across the eight queries span $30.91$--$181.81\%$ of the corresponding
origin-cut accounts, and connected ranges span $20.05$--$145.27\%$.
Background response remains substantial at the largest internal-node terms,
but individual magnitudes and rankings can change. In particular, the
$(10,16)$ difference for $\{10\}\to\{10,17\}$ is $-9{,}776.4$,
$-3{,}361.1$ and $+351.0$ at the three respective $\theta$ values.
Thus large boundary variation survives these checks, while the direction
of a particular local comparison need not do so. Appendix~\ref{supp:coverage}
gives the per-query results; these checks do not cover other networks or
endogenous route sets.
'''
(P/'results_main_v83.tex').write_text(main)
app=(P/'results_appendix_v82.tex').read_text()
app=app.replace('the analogous $11$-node comparison','the analogous $11$-point comparison')
app=app.replace('is $2.25\\times10^{-5}$', 'is $2.24\\times10^{-5}$')
app=app.replace('original zero-background checks','zero-background checks').replace('The new and original background solvers','The link Newton and successive-averages background solvers').replace('new background solves','background solves').replace('original path fixed-point','path fixed-point')
app=app.replace('their eleven common nodes','their eleven common quadrature points')
app=app.replace('For all $328$ background node--OD Jacobian columns','For all $328$ re-routing point--OD Jacobian columns at ten paths and $\\theta=1$')
app=app.replace('Absolute origin-cut accounts in outcome units, divided by $10^5$.','Absolute origin-cut accounts in TSTT units (Appendix~\\ref{supp:networks}),\ndivided by $10^5$. All three SUE settings use $41$ quadrature points.')
lines=[]
for line in app.splitlines():
 if line.startswith('$(') and len(line.split(' & '))==4:
  cells=line.split(' & ');key=cells[0].strip('$()').replace(',','->')
  cells[2]=f"${abs(x['B']['per_od'][key]['phi_s'])/1e5:.4f}$";line=' & '.join(cells)
 lines.append(line)
app='\n'.join(lines)+'\n'
start=app.index('Zero-background route-choice dispersion checks')
app=app[:start]+r'''Zero-background route-choice dispersion checks at $\theta\in\{0.5,1,2\}$
retain the large/small separation: large ranges are $27$--$64\%$ and the
other ranges remain below $9.4\%$ (eleven-point direct-SUE calculations).
Full-background checks are reported separately below.
'''
app+=r'''
\subsection{Frozen-background integration coverage}\label{supp:frozenaccuracy}
The background flow is frozen at the joint equilibrium with study demand
$q^0=0.2q_{\mathrm{base}}$. Table~\ref{tab:frozenaccuracy} compares nested
$11$- and $41$-point trapezoids computed with the same implicit sensitivities.
Each range percentage uses its own grid's origin account.
\begin{table}[pos=H]
\caption{Frozen-background quadrature refinement; column differences are
relative link-vector norms, and range changes are percentage points.}
\label{tab:frozenaccuracy}
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrr}
\toprule
OD & Full-column difference (\%) & Range change (pp)\\
\midrule
'''
for k in order:
 r=x['B']['refinement_per_od'][k]
 app+=f"{od(k)} & ${100*r['C_11_vs_41_relative_l2']:.5f}$ & ${r['spread_11_vs_41_abs_pp']:.5f}$ \\\\\n"
app+=r'''\bottomrule
\end{tabular*}
\end{table}
At the same eleven quadrature points, the implicit full columns differ
from the central-finite-difference reference by at most $'''+f"{100*x['B']['old_11_vs_new_11_C_max_relative_l2']:.5f}"+r'''\%$.
That comparison changes the derivative evaluation, whereas
Table~\ref{tab:frozenaccuracy} isolates the integration grid.
The refined B calculation's relative scalar completeness gap is
$'''+f"{x['B']['gates']['completeness_relative']:.2e}".replace('e-05',r'\times10^{-5}')+r'''$.

\subsection{Full-background path-set and dispersion coverage}\label{supp:coverage}
The path-set comparison expands all $528$ truncated OD sets, including
all $520$ background ODs; any fully enumerated set would be retained,
but none occurs here. Ten and twenty candidates per OD give $5{,}280$
and $10{,}560$ total paths. The first ten candidates of every expanded set
are identical to its ten-path set. Demands, BPR functions, $\theta=1$ and
$41$ quadrature points remain fixed.
\begin{table}[pos=H]
\caption{Setting C, ten versus twenty paths per OD. All values are absolute
TSTT accounts or ranges divided by $10^5$, using Appendix~\ref{supp:networks}'s
one-hour convention.}
\label{tab:pathcoverage}
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrr}
\toprule
&\multicolumn{2}{c}{$\phi_j(\{s_j\})$}&\multicolumn{2}{c}{$S_j$}&\multicolumn{2}{c}{$S_j^{\mathrm{conn}}$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
OD & 10 & 20 & 10 & 20 & 10 & 20\\
\midrule
'''
for k in order:
 vals=[r[f]/1e5 for f in ('phi_s','spread_abs','connected_range_abs') for r in (base[k],exp[k])]
 app+=od(k)+' & '+' & '.join('$'+fmt(v,5)+'$' for v in vals)+r' \\'+'\n'
app+=r'''\bottomrule
\end{tabular*}
\end{table}
Absolute unrestricted and connected ranges change by at most
$'''+f'{100*maxrel("spread_abs"):.3f}'+r'''\%$ and $'''+f'{100*maxrel("connected_range_abs"):.3f}'+r'''\%$, respectively.
Table~\ref{tab:pathnodes} reports the signed values at the three largest
internal nodes in the ten-path model; their ordered IDs are unchanged
with twenty paths. Every node's signed study and background contributions
are retained in the numerical result arrays.
\begin{table}[pos=H]
\caption{Principal internal-node divergence in setting C. Each tuple follows
the listed node order; values are signed TSTT accounts divided by $10^5$.}
\label{tab:pathnodes}
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llrr}
\toprule
OD & Nodes & Ten paths & Twenty paths\\
\midrule
'''
for k in order:
 ids=base[k]['major_nodes']
 vals=['$('+','.join(fmt(r['node_divergence_total'][i-1]/1e5,4) for i in ids)+')$' for r in (base[k],exp[k])]
 app+=od(k)+' & $('+','.join(map(str,ids))+')$ & '+' & '.join(vals)+r' \\'+'\n'
app+=r'''\bottomrule
\end{tabular*}
\end{table}
\begin{table}[pos=H]
\caption{Setting C dispersion sensitivity, ten paths per OD and $41$
quadrature points. Ranges are percentages of each model's own absolute
origin-cut account; $S$ and $S^{\mathrm{conn}}$ abbreviate the two ranges.}
\label{tab:thetacoverage}
\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrr}
\toprule
&\multicolumn{2}{c}{$\theta=0.5$}&\multicolumn{2}{c}{$\theta=1$}&\multicolumn{2}{c}{$\theta=2$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
OD & $S$ & $S^{\mathrm{conn}}$ & $S$ & $S^{\mathrm{conn}}$ & $S$ & $S^{\mathrm{conn}}$\\
\midrule
'''
for k in order:
 vals=[x['C_k10_theta'+th]['per_od'][k][f] for th in ['0.5','1','2'] for f in ['spread_pct','connected_range_pct']]
 app+=od(k)+' & '+' & '.join('$'+fmt(v,3)+'$' for v in vals)+r' \\'+'\n'
app+=r'''\bottomrule
\end{tabular*}
\end{table}
The $\theta$ comparisons change the fixed-path logit model, and do not
assert numerical invariance of individual accounts. In particular, the
specified $(10,16)$ boundary difference reverses sign at $\theta=2$.
Its study/background components at $\theta=0.5$ are $-7{,}404.9$ and
$-2{,}371.5$; at $\theta=2$ they are $+115.3$ and $+235.7$.
For the $(10,16)$ largest node terms at nodes $15$, $8$ and $20$, the
background magnitudes exceed the study magnitudes at each tested $\theta$.

For the expanded-path and two additional $\theta$ models, all eight
implicit derivative columns are checked against independently re-solved
half-step central differences at the demand-path midpoint. The maximum
relative column difference across these checks is $'''+f"{100*max(x[k]['gates']['midpoint_FD_halfstep_max_column_relative_l2'] for k in ['C_k20_theta1','C_k10_theta0.5','C_k10_theta2']):.5f}"+r'''\%$.
These midpoint checks supplement the ten-path $\theta=1$ validation;
they do not imply finite-difference coverage of all quadrature points in
the additional models. The largest relative completeness gap among these
$41$-point comparisons is $'''+f"{max(x[k]['gates']['completeness_relative'] for k in ['C_k20_theta1','C_k10_theta0.5','C_k10_theta2']):.2e}".replace('e-05',r'\times10^{-5}')+r'''$.
'''
# Unambiguous integration vocabulary throughout both result sections.
for oldword,newword in [('$41$-node','$41$-point'),('$21$-node','$21$-point'),('$11$-node','$11$-point'),('$201$-node','$201$-point'),('$801$-node','$801$-point'),('direct-SUE nodes','direct-SUE quadrature points'),('learned nodes','learned quadrature points')]:
 main=main.replace(oldword,newword);app=app.replace(oldword,newword)
(P/'results_main_v83.tex').write_text(main)
(P/'results_appendix_v83.tex').write_text(app)
print('Generated v83 tables from e84 raw JSON/NPZ and preserved v82 reference sections.')
