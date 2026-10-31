"""Compute predefined topology-only BFS-tree bonds from saved Anaheim columns."""
from pathlib import Path
from collections import deque
import json,numpy as np
O=Path('experiments/results/e112_v102');a=np.load(O/'arrays.npz');r=json.loads((O/'results.json').read_text());edges=a['edges'];n=a['B'].shape[0];adj=[set() for _ in range(n)]
for u,v in edges:adj[u].add(v);adj[v].add(u)
for j,row in enumerate(r['rows']):
 s,t=[v-1 for v in row['query']];cuts=set()
 for root in(s,t):
  parent={root:None};queue=deque([root])
  while queue:
   v=queue.popleft()
   for w in sorted(adj[v]):
    if w not in parent:parent[w]=v;queue.append(w)
  v=t if root==s else s
  while parent[v] is not None:
   # removing this edge of the fixed BFS tree creates two connected sides
   subtree={v};pending=[v]
   while pending:
    w=pending.pop()
    children=[u for u in adj[w] if parent.get(u)==w]
    subtree.update(children);pending.extend(children)
   U=subtree if s in subtree else set(range(n))-subtree
   cuts.add(tuple(sorted(U)));v=parent[v]
 d=a['B']@a['C'][:,j];values=[float(-d[list(U)].sum()) for U in sorted(cuts)];row['connected_bfs']={'count':len(cuts),'min':min(values),'max':max(values),'regions':[list(U) for U in sorted(cuts)]}
(O/'results.json').write_text(json.dumps(r,indent=2,default=int))
lines=[r'\subsection{External spatial calculation on Anaheim}\label{sec:anaheim}',r'The public Anaheim network has 416 nodes, 914 directed links and 1,406 positive OD demands \citep{tnfrc2024transportation}. We select the six largest demands, vary them jointly from 0.8 to 1.2 times their published values, and hold all other demands fixed while their route choices remain endogenous. Each OD uses ten free-flow shortest paths; candidate routes exclude other zone centroids as intermediate nodes (the first through node is 39). This reference calculation evaluates spatial computation and background mechanisms; no Anaheim surrogate is trained.',r'\begin{table}[t]\centering\small',r'\caption{Anaheim reference attribution (absolute accounts in vehicle-minutes per hour; divide by 60 for vehicle-hours per hour): unrestricted accounts and connected regions formed by removing edges on the OD path in two deterministic breadth-first spanning trees rooted at the origin and destination. These topology-only region rules use no attribution values. Ranges refer to the integrated columns, with no connected-region enumeration.}\label{tab:anaheim}',r'\begin{tabular}{lrrrr}\toprule',r'OD & System & Unrestricted range & Connected range & Regions\\\midrule']
for row in r['rows']:
 b=row['connected_bfs'];lines.append(f"({row['query'][0]},{row['query'][1]}) & {row['system']:.1f} & [{row['min']:.1f},{row['max']:.1f}] & [{b['min']:.1f},{b['max']:.1f}] & {b['count']}"+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}',f"The reference chain with 21 quadrature states took {r['total_seconds']:.1f} seconds on the local CPU, including route construction and implicit sensitivities; the six unrestricted extrema took {r['six_extrema_seconds']:.4f} seconds after the columns were available. This separates the saved boundary-enumeration cost from reference attribution cost. Maximum path fixed-point residual was {r['solver_max_residual']:.2g}, flow-balance residual {r['conservation_max']:.2g}, and relative quadrature completeness discrepancy {r['completeness_relative']:.2g}.",r'At the common middle state, the six background-response norms range from 0.115 to 0.487, while their maximum node divergence is below $3.1\times10^{-15}$. Weighting those same responses by the common total-flow marginal costs gives divergence norms 0.116--0.796. Background rerouting therefore remains a nonzero spatial attribution mechanism on this larger network. These calculations do not establish cross-network learning gains.']
Path('paper/results_anaheim_v102.tex').write_text('\n'.join(lines)+'\n')
report=['# Anaheim 外部空间计算','',r['protocol'],'',f"节点{n}，边{len(edges)}，正需求{r['network']['positive_ods']}，候选路径{r['network']['paths']}。真实总耗时{r['total_seconds']:.1f}秒；已有归因列上的六查询闭式极值{r['six_extrema_seconds']:.4f}秒。",'', '全部六查询的非受限范围均跨零；BFS树切分得到两侧连通区域，规则只依赖拓扑和OD、不依赖归因值。没有在Anaheim训练代理，不能据此声称跨网络学习优势。', '',f"路径固定点残差{r['solver_max_residual']:.3g}；守恒误差{r['conservation_max']:.3g}；分解闭合误差{r['decomposition_max']:.3g}；积分完备性相对差{r['completeness_relative']:.3g}。"]
Path('revision-stage/v102/ANAHEIM_RESULTS.md').write_text('\n'.join(report)+'\n')
