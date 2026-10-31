"""Generate manuscript tables from saved spatial results, no new equilibria."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from run_e113_v103_spatial import report,ROOT,OLD,OUT

def main():
 a=np.load(OLD/'arrays.npz'); rows=[]
 for line in (ROOT/'experiments/data/Anaheim/Anaheim_net.tntp').read_text().splitlines():
  s=line.split()
  if s and s[0].isdigit():rows.append(s)
 net=SimpleNamespace(B=a['B'],free_flow_time=np.array([float(s[4]) for s in rows]));report(a,net,None,a['q'])
 refined=(OUT/'results_41.json').exists();r=json.loads((OUT/('results_41.json' if refined else 'results_21.json')).read_text());points=r['points'];old=json.loads((OLD/'results.json').read_text())
 tex=(ROOT/'paper/results_anaheim_v102.tex').read_text().replace('Connected range','Selected connected range')
 if refined:
  C=np.load(OUT/'columns_41.npz')['C'];BC=net.B@C
  for j,row in enumerate(old['rows']):
   s0,t0=[v-1 for v in row['query']]; interior=[v for v in range(len(BC)) if v not in (s0,t0)]
   lo=-BC[s0,j]-np.maximum(BC[interior,j],0).sum();hi=-BC[s0,j]+np.maximum(-BC[interior,j],0).sum()
   values=[float(-BC[U,j].sum()) for U in row['connected_bfs']['regions']]
   line=f"({row['query'][0]},{row['query'][1]}) & {C[:,j].sum():.1f} & [{lo:.1f},{hi:.1f}] & [{min(values):.1f},{max(values):.1f}] & {len(values)}"+r'\\'
   pattern=r'^\('+str(row['query'][0])+','+str(row['query'][1])+r'\).*?\\\\$'
   tex=re.sub(pattern,lambda m:line,tex,flags=re.M)
  tex=tex.replace('Ranges refer to the integrated columns,', 'Ranges use 41-point integrated columns,')
 # Preserve v102 overall comparison, add full-path mechanism and direct precision.
 tex=tex.replace('These calculations do not establish cross-network learning gains.','')
 tex+='\n'+r'''For full-path mechanism analysis, order each query's pre-existing BFS regions lexicographically by its sorted node list and select the first, middle and last regions, denoted $U_0,U_1,U_2$. This rule uses topology and the OD only, without inspecting attribution. Each fixed comparison is decomposed using the same equilibrium states and total-flow marginal costs into free-flow/study, free-flow/background, congestion/study and congestion/background terms. Table~\ref{tab:anaheimmechanism} reports $\varphi_j(U_2)-\varphi_j(U_0)$; the additional $U_1-U_0$ comparisons and exact region memberships are archived with the experiment. These are differences for identical fixed regions, not sums of componentwise ranges.
\begin{table}[t]\centering\small
\caption{Anaheim full-path differences between two preselected connected regions (vehicle-minutes per hour). FF denotes free-flow costs, C congestion marginal costs, S the six study demands and B all other demands.}\label{tab:anaheimmechanism}
\begin{tabular}{lrrrrr}\toprule OD & FF/S & FF/B & C/S & C/B & Total\\\midrule
'''
 for row in r['rows']:
  d=row['differences'][-1];v=d['components'];tex+=f"({row['query'][0]},{row['query'][1]}) & "+' & '.join(f'{v[k]:.2f}' for k in ['ff_study','ff_bg','cong_study','cong_bg'])+f" & {d['total']:.2f}"+r'\\'+'\n'
 tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
 tex+=f"These mechanism columns use {points} equally spaced trapezoidal states. The maximum unrounded decomposition discrepancy is {r['column_decomposition_max']:.2g} per edge. "
 if refined:
  p=r['precision'];tex+=f"Refining from 21 to 41 states changes the node representation by relative Euclidean errors between {min(p['node_relative_l2_per_query']):.2g} and {max(p['node_relative_l2_per_query']):.2g} across queries; the largest selected-account change is {p['selected_account_max_absolute']:.3g} vehicle-minutes per hour ({100*p['selected_account_relative_max']:.3g}\\% relative). This evaluates spatial accounts directly rather than relying on cancellation in scalar completeness. "
 vals=[(row['query'],row['differences'][-1]['components']['cong_bg'],row['differences'][-1]['total']) for row in r['rows']]
 tex+='For $(1,2)$, the background free-flow term is positive whereas its congestion term is negative, and these contributions largely cancel. For most other selected comparisons the background terms are small relative to the study-response congestion term. The mechanism decomposition identifies both cancellation and relative magnitude under common total-flow cost weights.\n'
 (ROOT/'paper/results_anaheim_v103.tex').write_text(tex)
 settings=r'''\subsection{Anaheim network, equilibrium and predefined regions}\label{app:anaheimsettings}
Both network and demand files are the public Transportation Networks for Research Core Team Anaheim TNTP data \citep{tnfrc2024transportation}. The network header supplies free-flow times in minutes and capacities in vehicles per hour; all roads use the published BPR coefficient 0.15 and power 4. We use $t_e(F_e)=t_e^0[1+0.15(F_e/c_e)^4]$ and TSTT marginal weight $t_e^0[1+0.75(F_e/c_e)^4]$. The six study OD demands, ranked by descending base demand and then lexicographic OD, are '''
 settings+=', '.join(f"$({row['query'][0]},{row['query'][1]})$: {q:g}" for row,q in zip(r['rows'],r['qbase_six']))+' vehicles per hour. '
 settings+=r'''They move jointly from 0.8 to 1.2 times base demand; all remaining positive demands stay at their published base values and retain endogenous route choice. Every OD uses ten loopless free-flow shortest routes from Yen's algorithm. Ties within $10^{-12}$ minutes are resolved by lexicographic edge-index sequence in the TNTP road order. Nodes 1--38 are centroids and cannot be intermediate route nodes; the first through node is 39. Logit scale $\theta=1$ minute$^{-1}$ applies to route travel time in minutes.
The edge Newton solver targets maximum loading residual below $10^{-8}$ vehicles per hour, uses at most 80 iterations and backtracking up to 30 halvings per step, and checks the original path loading residual below $10^{-7}$. Implicit derivatives solve the same fixed-point equation. Trapezoidal integration uses equally spaced states, with warm starts from a neighboring saved state. The 41-state refinement reuses all 21 old states and computes only the twenty new midpoints.
For each study OD, construct undirected BFS spanning trees rooted at each endpoint. At every visited node, traverse neighbors in increasing node-index order. Deleting each edge on the tree's endpoint path creates two connected parts; orient the region to contain the OD origin, and remove duplicate regions. This gives the selected connected family, not all connected separating regions. Sort the resulting node lists lexicographically to select first, middle and last regions for the mechanism comparisons. Free-flow weights are $t^0$ and congestion weights are the TSTT marginal weight minus $t^0$, at the same total-flow state for both study and background responses.
'''
 (ROOT/'paper/anaheim_settings_v103.tex').write_text(settings)
 md=['# v103 空间实验结果','',f'Anaheim机制分解积分点数：{points}。BFS区域按节点列表字典序预先选首、中、末三个，分解固定区域差，未把分项极值相加。','', '## 机制结果','']
 for row in r['rows']:md.append(str({'OD':row['query'],'区域差':row['differences']}))
 if refined:md+=['','## 21与41点精度',json.dumps(r['precision'],ensure_ascii=False,indent=2)]
 scan=ROOT/'experiments/results/e115_v103/results.json'
 if scan.exists():md+=['','## 预设直线截面扫描',scan.read_text(),'','扫描使用稀疏关联矩阵；同投影节点成组。对相同1642个查询-位置账户逐边求和，差异为浮点累加误差。多列计时重复六列，不算新增科学样本。']
 (ROOT/'revision-stage/v103/SPATIAL_RESULTS.md').write_text('\n'.join(md)+'\n');print('reported',points)
if __name__=='__main__':main()
