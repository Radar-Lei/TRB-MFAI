import json
from pathlib import Path
import numpy as np
R=Path('experiments/results/e110_v101');decision=json.loads((R/'selection.json').read_text());w=decision['selected_weight'];old=json.loads((R/'old_results.json').read_text());new=json.loads((R/'new_results.json').read_text())
def pairs(z):
 rows=z['rows'];a={(r['seed'],r['path'],r['query']):r for r in rows if r['method']=='response'};b={(r['seed'],r['path'],r['query']):r for r in rows if r['method']=='structure'};out=[]
 for p,q in sorted(set((k[1],k[2]) for k in a)):
  keys=[k for k in a if k[1]==p and k[2]==q];br=[b[k]['boundary_abs']/a[k]['boundary_abs'] for k in keys];cr=[b[k]['column_pct']/a[k]['column_pct'] for k in keys];out.append({'path':p,'query':q,'boundary_improved':sum(x<1 for x in br),'column_improved':sum(x<1 for x in cr),'boundary_ratio_median':float(np.median(br)),'boundary_ratio_range':[float(min(br)),float(max(br))]})
 return out
po,pn=pairs(old),pairs(new)
def totals(entries):
 out={}
 for q in ['ALL']+sorted(set(e['query'] for e in entries)):
  es=[e for e in entries if q=='ALL' or e['query']==q];out[q]={'boundary_improved':sum(e['boundary_improved'] for e in es),'column_improved':sum(e['column_improved'] for e in es),'n':5*len(es)}
 return out
(R/'paired_summary.json').write_text(json.dumps({'old_paths':po,'new_paths':pn,'old_totals':totals(po),'new_totals':totals(pn)},indent=2))
# One new-path main table, keep the old first-round table in place.
s=(R/'calibration_new_table.tex').read_text().replace('Equal-update continuation from five flow-trained checkpoints on four independent demand paths.',f'Exploratory validation-calibrated continuation ($\\lambda_{{\\mathrm{{cut}}}}={w:g}$), tested on four new independent demand paths (seed 101030).').replace('tab:structured-learning','tab:calibration-learning').replace('Flow + response + boundary','Calibrated boundary')
Path('paper/calibration_results_v101.tex').write_text(s)
tex=r'''\subsection{Exploratory validation calibration}\label{app:calibration-protocol}
Following the first experiment's negative aggregate result and oscillating structural training loss, we fixed a second, exploratory protocol that calibrates the boundary coefficient. All first-experiment results are retained. Candidate coefficients $\{0.001,0.01,0.1\}$ are each trained for the same 600 updates from the original seed-zero checkpoint. The previously unused 64 validation derivative states select the smallest average six-query normalized $\mathcal K_j(h_j)^2$ among candidates whose validation flow mean relative error is at most 110\% of the general-response checkpoint's error. No old or new test path is used in this choice. The selected coefficient is then applied to all five seeds, reusing its seed-zero candidate. This adds a calibration budget unavailable to the comparison methods; the experiment does not establish equal-search-budget superiority.
'''
tex+=f'The validation-selected coefficient was ${w:g}$. The sum of elapsed durations of the three candidate jobs was {decision["candidate_total_process_seconds"]:.1f} seconds, including validation evaluation; these overlapping job durations are not CPU time. '
traincost=[json.loads((R/'selected'/f'structure_{s}_cost.json').read_text())['seconds'] for s in range(5)];tex+=f'The selected five seed runs had median training time {np.median(traincost):.1f} s; seed zero is reused and is not an additional run. All reported durations are elapsed wall-clock measurements. '
np_cost=json.loads((R/'newpaths_cost.json').read_text());tex+=f'After selection, four new paths were generated with seed 101030 using the same one-, two-, three-, and six-query signed-change templates; their 164 reference states required {np_cost["seconds"]:.1f} s. New path endpoints and integration states were evaluated only after selection. The original four paths remain a separately reported diagnostic set.\n'
checks=new['checks'];tex+=f"On the new paths the maximum learned conservation residual was ${max(c['conservation_abs'] for c in checks):.2g}$ and the largest relative 41-point completeness discrepancy was ${max(c['completeness_relative'] for c in checks):.2g}$. "
if (R/'wall_seconds.json').exists():tex+=f"The complete calibration, selected training and two-set evaluation driver required {json.loads((R/'wall_seconds.json').read_text())['total_wall_seconds']:.1f} s elapsed. "
tex+=r'\begin{table}[ht]\centering\small\caption{Validation-only coefficient selection. Flow is the mean relative flow error on 64 validation states; the K column is the mean squared training-scale-normalized boundary-response discrepancy.}\label{tab:calibration-selection}\begin{tabular}{lrrl}\toprule Coefficient & Flow (\%) & K mean square & Eligible\\\midrule'+'\n'
for key,v in decision['results'].items():
 eligible=str(v.get('eligible','reference'));tex+=f'{key} & {100*v["flow_relative_mean"]:.4f} & {v["normalized_K_mse"]:.4f} & {eligible}'+r'\\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
tex+=r'\begin{table}[ht]\centering\small\caption{Paired calibrated-boundary versus general-response errors on each path; each row retains five seeds. Old and new paths are reported separately.}\label{tab:calibration-pairs}\begin{tabular}{llrrr}\toprule Set/path & Query & Cut improved/5 & Column improved/5 & Cut ratio\\\midrule'+'\n'
for name,entries in [('Old',po),('New',pn)]:
 for e in entries:
  q=e['query'].replace('->',r'$\to$');tex+=f'{name} {e["path"]+1} & {q} & {e["boundary_improved"]}/5 & {e["column_improved"]}/5 & {e["boundary_ratio_median"]:.2f}'+r'\\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
# Old path aggregate table has a different label.
tex+=(R/'calibration_old_table.tex').read_text().replace('Equal-update continuation from five flow-trained checkpoints on four independent demand paths.',f'Exploratory calibrated continuation on the original four paths ($\\lambda_{{\\mathrm{{cut}}}}={w:g}$); first-round fixed-weight results remain in Table~\\ref{{tab:structured-learning}}.').replace(r'\label{tab:structured-learning}', r'\label{tab:calibration-old}').replace('Flow + response + boundary','Calibrated boundary')
Path('paper/calibration_protocol_v101.tex').write_text(tex)
lines=['# 第二协议：验证校准结果','','首轮负面结果完整保留；以下为已观察负面后制定的探索性修复。只使用此前未用的64验证状态选择权重，新路径seed101030在选择后才生成。',f'选择权重：{w}；三候选各作业墙钟时长之和{decision["candidate_total_process_seconds"]:.1f}秒。','', '## 全新四路径中位数','',json.dumps(new['summary'],ensure_ascii=False,indent=2),'','## 新路径逐种子配对计数（相对一般响应）','',json.dumps(pn,ensure_ascii=False,indent=2),'','## 原四路径中位数','',json.dumps(old['summary'],ensure_ascii=False,indent=2)]
Path('revision-stage/v101/LEARNING_CALIBRATION_RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'selected':w,'new_summary':new['summary'],'new_pairs':pn,'old_summary':old['summary']},indent=2))
