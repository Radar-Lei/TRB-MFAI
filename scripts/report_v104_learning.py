"""Produce matched-path learned-assignment tables and protocol summaries."""
import json
from pathlib import Path
import numpy as np
O=Path('experiments/results/v104_learning');r=json.loads((O/'results.json').read_text());s=r['summary'];names={'flow':'Flow','edge':'Flow + endpoint edge','cut':'Flow + endpoint edge + boundary','response':'Flow + response'}
tex=[r'\begin{table}[t]\centering\small',r'\caption{Prediction and spatial explanation on the same 24 test paths. Entries average queries and seeds within paths and report path medians; flow additionally averages the 41 states within each path.}',r'\label{tab:path-explanations}',r'\begin{tabular}{lrrr}\toprule',r'Model & Flow (\%) & Column (\%) & Geometric (\%)\\\midrule']
for m in ['flow','response']:tex.append(names[m]+' & '+' & '.join(f'{s[m][k]:.4f}' if k=='flow_pct' else f'{s[m][k]:.2f}' for k in ['flow_pct','column_pct','geometry_pct'])+r'\\')
tex+=[r'\bottomrule\end{tabular}\end{table}']
body=r'''The same-path comparison in Table~\ref{tab:path-explanations} evaluates both
flow predictions and spatial explanations on 24 prescribed paths, with six
paths each containing one, two, three and six active queries. Five conservative
models share the same initial states, data and 1,200 continuation updates.
Validation includes update zero and every 100th update, selecting minimum
boundary error subject to endpoint flow error at most 110\% of its initial
value. Statistics average seeds and active queries within paths; the 24 paths
are the independent evaluation units. Full objectives, denominators and the
four-method comparison appear in Appendix~\ref{app:pairs-v102}.
'''+ '\n'.join(tex)+r'''
Ordinary response supervision gives 14.22\% median column error and 5.81\%
geometric error, compared with 21.61\% and 7.91\% for flow supervision.
The corresponding flow errors are 0.0291\% and 0.0325\% on precisely the same
states. Thus spatial criteria quantify differences in explanation quality even
when both models reproduce flow levels accurately. The endpoint-edge and
endpoint-edge-plus-boundary objectives give no overall improvement; both
select the common initial model for three of five seeds.

The product decomposition in Eq.~\eqref{eq:coldecomp} separates marginal-cost
weight error from demand-response error along these same paths. For both models
in Table~\ref{tab:path-explanations}, the response component has the larger
norm on all 24 paths after averaging active queries and seeds. Their path-median
response and weight-component norms are respectively 256.1 and 4.1 for flow
supervision, and 158.1 and 3.8 for response supervision. Component norms need
not add because signed components may cancel. This identifies response
accuracy as the main explanation-error mechanism in this comparison.

Sign reliability is boundary specific. On the separate frozen six-query
baseline segment of Appendix~\ref{supp:screenlinelearned}, all five models
retain negative accounts at the reference minimizing screenlines for $(1,19)$
and $(7,12)$. Their learned accounts lie in $[-3886.6,-3365.1]$ and
$[-2810.4,-1916.6]$, respectively, against reference values $-2766.6$ and
$-2967.3$. The largest errors over the prescribed screenline families,
1430.8 and 2499.2, are below the corresponding reference sign margins.
These account-specific bounds establish the reported signs while retaining
quantitative magnitude errors. This frozen-segment evidence is separate from
the 24-path comparison above.
'''
Path('paper/results_pairs_v104.tex').write_text(body)
full=[r'\begin{table}[t]\centering\small',r'\caption{Four objectives with update zero included in validation selection. All prediction and explanation entries use the same 24 paths and report path medians after averaging seeds and active queries.}',r'\label{tab:pairs-v102}',r'\begin{tabular}{lrrrrr}\toprule Method & Flow (\%) & Column (\%) & System (\%) & Boundary (\%) & Geometric (\%)\\\midrule']
for m in names:full.append(names[m]+' & '+' & '.join(f'{s[m][k]:.4f}' if k=='flow_pct' else f'{s[m][k]:.2f}' for k in ['flow_pct','column_pct','system_pct','boundary_pct','geometry_pct'])+r'\\')
full += [r'\bottomrule\end{tabular}\end{table}']
full.append('Selected updates in seed order 0--4 are '+ '; '.join(names[m]+': '+', '.join(str(x['step']) for x in r['selection'] if x['method']==m) for m in names)+'. All selected models meet the endpoint flow constraint. For each endpoint objective, three seeds retain update zero.')
if (O/'heldout.json').exists():
 h=json.loads((O/'heldout.json').read_text())['seed_mean'];full.append('On the original held-out demand-state split, seed-mean relative flow errors are '+', '.join(names[m]+f": {h[m]:.4f}\\%" for m in names)+'. These are distinct from the matched-path flow errors in the table.')
Path('paper/learning_full_table_v104.tex').write_text('\n'.join(full)+'\n')
# Cost table is generated after completion of the matched inference measurements.
if (O/'selected_inference_cost.json').exists():
 c=json.loads((O/'label_costs.json').read_text());ic=json.loads((O/'selected_inference_cost.json').read_text())['rows'];old=Path('experiments/results/e111_v102')
 t=[r'\begin{table}[t]\centering\small',r'\caption{Incremental computation costs (seconds). Training-label and common-validation costs are measured separately on the same CPU protocol; training times are archived continuation medians, and explanation times are newly measured sequential medians for the selected models.}',r'\label{tab:learning-costs}',r'\begin{tabular}{lrrrr}\toprule Method & Training labels & Common validation & Continuation & Explanation\\\midrule']
 for m in names:
  label=0 if m=='flow' else c['response_train' if m=='response' else 'endpoint_train']['seconds'];train=np.median([json.loads((old/f'{m}_{seed}_cost.json').read_text())['seconds'] for seed in range(5)]);infer=np.median([x['seconds'] for x in ic if x['method']==m]);t.append(names[m]+f" & {label:.2f} & {c['common_endpoint_validation']['seconds']:.2f} & {train:.1f} & {infer:.2f}"+r'\\')
 t +=[r'\bottomrule\end{tabular}\end{table}',r'Training labels comprise 768 endpoint states for either endpoint objective, or 384 center states and their sensitivities for response supervision. Every method additionally uses the same 192 endpoint states for validation; unused center-validation derivatives are excluded. Endpoint objectives share their labels and do not incur duplicate reference solves. Existing flow data, pretrained models and the common test reference are excluded from these incremental costs. The 984 test states and sensitivities took 74.4\,s in the archived reference run. Explanation timing covers all 24 paths, forward evaluation, six directional derivatives per state, quadrature and node aggregation, excluding model loading. Archived continuation times include initialization, validation and saving under concurrent seed jobs; they describe the actual training expense and do not establish a method speed advantage.']
 Path('paper/learning_cost_table_v104.tex').write_text('\n'.join(t)+'\n')
lines=['# v104 学习模型空间解释比较','','20个模型均以第0步及每100步为候选，验证端点流量误差约束全部满足；测试路径不参与选择。三个位于共同初始模型的种子在两个端点目标下未取得可接受的继续训练改进。','','|目标|同路径流量%|列误差%|几何误差%|任意边界误差%|','|---|---:|---:|---:|---:|']
for m in names:lines.append('|'+names[m]+'|'+'|'.join(f'{s[m][k]:.5f}' for k in ['flow_pct','column_pct','geometry_pct','boundary_pct'])+'|')
lines +=['','指标先平均路径内活跃查询和种子，再在24条路径上取中位数；流量先平均路径内41状态。种子不增加独立路径数。响应误差分解使用与附录相同的乘积展开，两项重构归因误差的逐项检查小于1e-7。两种代表模型均在全部24路径上由响应误差项主导。','','全部原始结果位于experiments/results/v104_learning/，无需重训。run_v104_learning.py复用已有模型、日志和数组，run_v104_learning_cost.py对最终选定模型进行相同流程的顺序CPU解释计时。']
Path('revision-stage/v104').mkdir(exist_ok=True,parents=True);Path('revision-stage/v104/LEARNING_RESULTS.md').write_text('\n'.join(lines)+'\n')
