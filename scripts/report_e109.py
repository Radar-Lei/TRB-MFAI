import json,time
from pathlib import Path
import numpy as np
R=Path('experiments/results/e109_v101');z=json.loads((R/'results.json').read_text());rows=z['rows'];summ=z['summary']; costs={m:[json.loads((R/f'{m}_{s}_cost.json').read_text())['seconds'] for s in range(5)] for m in ['flow','response','structure']}; paired={}
for base in ['original','flow','response']:
 a={(r['seed'],r['path'],r['query']):r for r in rows if r['method']==base};b={(r['seed'],r['path'],r['query']):r for r in rows if r['method']=='structure'}
 paired[base]={}
 for q in ['ALL']+sorted(set(k[2] for k in a)):
  keys=[k for k in a if q=='ALL' or k[2]==q];entry={}
  for metric in ['column_pct','system_pct','boundary_abs']:
   dif=[b[k][metric]-a[k][metric] for k in keys];ratio=[b[k][metric]/a[k][metric] for k in keys];entry[metric]={'improved':sum(d<0 for d in dif),'n':len(dif),'median_ratio':float(np.median(ratio)),'ratio_range':[float(min(ratio)),float(max(ratio))]}
  paired[base][q]=entry
checks=z['checks'];proof={'max_conservation_abs':max(c['conservation_abs'] for c in checks),'max_completeness_relative':max(c['completeness_relative'] for c in checks),'training_seconds':costs,'paired':paired};(R/'paired_summary.json').write_text(json.dumps(proof,indent=2))
refcost=json.loads((R/'reference_cost.json').read_text());lines=['# v101 建设性学习比较结果','', '实际任务：三个监督目标、五种子、共同原e93检查点延续；固定600步末步，不从测试挑选。四条独立测试路径，60个活跃查询×路径×种子组合。流量5个唯一种子，有限增量20个唯一组合。','',f'参考标签与路径耗时：{refcost["seconds"]:.1f}秒（320导数状态+164测试状态）；其中仅256状态参与训练，64验证状态未用于选择。',f'模型最大守恒残差：{proof["max_conservation_abs"]:.3g}；41点积分完整性最大相对误差：{proof["max_completeness_relative"]:.3g}。','', '## 方法总体中位数','', '|方法|流量%|增量%|归因列%|系统%|边界绝对量|边界%|','|---|---:|---:|---:|---:|---:|---:|']
for m in summ:lines.append('|'+m+'|'+'|'.join(f'{summ[m][k]:.3f}' for k in ['flow_test_pct','increment_pct','column_pct','system_pct','boundary_abs','boundary_pct'])+'|')
lines+=['','## 配对结果（结构监督相对一般响应监督）','', '|查询|边界改善/总数|边界误差比中位数|范围|归因列改善/总数|','|---|---:|---:|---|---:|']
for q,e in paired['response'].items():
 p=e['boundary_abs'];c=e['column_pct'];lines.append(f'|{q}|{p["improved"]}/{p["n"]}|{p["median_ratio"]:.3f}|{p["ratio_range"]}|{c["improved"]}/{c["n"]}|')
lines+=['','## 计算成本','']
for m,v in costs.items():lines.append(f'- {m}五种子训练秒数：{v}；中位数{np.median(v):.1f}秒。')
lines+=['','限制：单网络、六输入需求、固定路径集合、原模型延续训练；等优化步数不等计算耗时，未证明未知需求域的一致保证。一般响应监督使用额外导数信息，结构目标不使用额外独立标签。全部逐路径逐种子数值见 results.json，原始归因列/流量/J见 evaluation.npz。','']
lines+=['','## 逐路径配对结果（结构监督相对一般响应监督）','','|路径|查询|边界改善/5种子|误差比中位数|','|---|---|---:|---:|']
a={(r['seed'],r['path'],r['query']):r for r in rows if r['method']=='response'};b={(r['seed'],r['path'],r['query']):r for r in rows if r['method']=='structure'}
pathpairs=[]
for pi,q in sorted(set((k[1],k[2]) for k in a)):
 keys=[k for k in a if k[1]==pi and k[2]==q];ratios=[b[k]['boundary_abs']/a[k]['boundary_abs'] for k in keys];improved=sum(x<1 for x in ratios);pathpairs.append({'path':pi,'query':q,'improved':improved,'n':len(keys),'median_ratio':float(np.median(ratios))});lines.append(f'|{pi+1}|{q}|{improved}/5|{np.median(ratios):.3f}|')
proof['path_pairs']=pathpairs;(R/'paired_summary.json').write_text(json.dumps(proof,indent=2))
Path('revision-stage/v101/LEARNING_RESULTS.md').write_text('\n'.join(lines))
tex=r'''\subsection{Structured-response continuation protocol}\label{app:learning-protocol}
All three continuation methods start from each of the same five full-background flow-trained checkpoints. The original data split and three-layer 128-unit tanh architecture with 5,280 path logits are retained. We use 600 Adam updates at learning rate $3\times10^{-5}$, flow batches of 32 and response batches of eight. One query column is cycled at each update, yielding 100 updates per query. Implicit reference Jacobians are computed at 256 states sampled without replacement from the original 2,500 training states (sampling seed 20261029); 64 additional validation derivatives are saved but used neither for selection nor tuning. Exact automatic-differentiation Jacobian--vector products supply trainable model responses.

For edge $e$, the flow scale is $0.01\sqrt{N_{\rm tr}^{-1}\sum_q F_e(q)^2}$, using the original training flows. For query $j$, the response scale is $\sqrt{256^{-1}\sum_q\|J_j^S(q)\|_2^2}$; the boundary scale is $\sqrt{256^{-1}\sum_q\mathcal K_j(D^S(q)J_j^S(q))^2}$. All scales use training data only. The flow term averages squared edgewise normalized residuals; the response term averages squared column-norm residuals divided by its squared scale; the boundary term averages squared $\mathcal K_j(h_j)$ divided by its squared scale. Both added coefficients are one. There is no hyperparameter search or test-based checkpoint selection: all methods retain the fixed final update, share the same seed-specific flow minibatch sequence, and receive the same update count. The structural term uses the same derivative labels as general response supervision. Its additional cost is differentiation through the outcome weights and boundary norm.

Four independent paths are fixed before training using seed 101029. Their baselines are sampled independently within $[0.65,1.3]$ times published query demands; respectively one, two, three, and six query components change, including mixed signs. Each path uses 41 uniformly spaced integration states, none used to tune the methods. Metrics retain the path as an evaluation unit. Flow medians use five seeds; finite-increment medians use 20 path--seed pairs; attribution-column, system-contribution and boundary medians use 60 active query--path--seed combinations. The boundary metric is the exact unrestricted separating-cut error; its reference scale is the corresponding maximum absolute reference account, including the endpoint term. Raw results retain every path, seed and active query.
'''
tex+=f'\nReference derivative labels and the independent test paths together required {refcost["seconds"]:.1f} CPU seconds. Median training times over five seeds were '+', '.join(f'{m}: {np.median(v):.1f} s' for m,v in costs.items())+'. Equal update counts therefore do not imply equal compute. '
tex+=f'The maximum learned flow-conservation residual over the path states was ${proof["max_conservation_abs"]:.2g}$; the largest relative completeness discrepancy under the 41-point quadrature was ${proof["max_completeness_relative"]:.2g}$.\n'
tex+=r'\begin{table}[ht]\centering\small\caption{Paired structural-versus-general-response boundary errors on independent paths. Ratios below one indicate improvement; each row retains the active path--seed units.}\label{tab:paired-structure}\begin{tabular}{lrrr}\toprule Query & Improved/total & Median ratio & Ratio range\\\midrule'+'\n'
for q,e in paired['response'].items():
 p=e['boundary_abs'];lo,hi=p['ratio_range'];label=q.replace('->',r'$\to$');tex+=f'{label} & {p["improved"]}/{p["n"]} & {p["median_ratio"]:.2f} & [{lo:.2f},{hi:.2f}]'+r'\\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
tex+=r'\begin{table}[ht]\centering\small\caption{Path-specific paired boundary errors, structural versus general response supervision. Each entry retains five seeds.}\label{tab:path-pairs}\begin{tabular}{llrr}\toprule Path & Query & Improved/5 & Median ratio\\\midrule'+'\n'
for entry in pathpairs:
 label=entry['query'].replace('->',r'$\to$');tex+=f"{entry['path']+1} & {label} & {entry['improved']}/5 & {entry['median_ratio']:.2f}"+r'\\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
Path('paper/learning_protocol_v101.tex').write_text(tex)
