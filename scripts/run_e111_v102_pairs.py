"""Frozen v102 endpoint supervision protocol. All reference arrays and costs persisted.
Training pairs and paths are fixed before training/evaluation; no test selection.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,argparse
from pathlib import Path
import numpy as np
import torch
import run_e109_v101_structured as base
from run_e83_v82_validation import attribution
O=base.R/'e111_v102';O.mkdir(exist_ok=True)
def screenline_masks(ods):
 from run_e107_v100_screenline import load_network
 import networkx as nx
 nodes,coords,g,arcs=load_network();out=[]
 for s,t in ods:
  seen=set()
  for deg in np.arange(0,180,5):
   normal=np.array([np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg))]);proj={v:np.array(coords[v])@normal for v in nodes};p=sorted(proj.values())
   for a,b in zip(p[:-1],p[1:]):
    if b<=a:continue
    below=frozenset(v for v in nodes if proj[v]<(a+b)/2)
    for U in (below,frozenset(set(nodes)-below)):
     if s+1 in U and t+1 not in U and nx.is_connected(g.subgraph(U)) and nx.is_connected(g.subgraph(set(nodes)-U)):seen.add(U)
  out.append(np.array([[v in U for v in range(1,25)] for U in sorted(seen,key=lambda x:tuple(sorted(x)))],dtype=float))
 return out

METHODS=['flow','edge','cut','response']; STEPS=1200; BUDGETS=[300,600,1200]; EVERY=100
LR=3e-5; BS=32; BJ=8; CUT_WEIGHT=.01

def cost(f,n):return f*n.free_flow_time*(1+.15*(f/n.capacity)**4)
def kt(z,B,s,t):
 d=z@B.T;inter=[i for i in range(B.shape[0]) if i not in(s,t)]
 return .5*((d[...,t]-d[...,s]).abs()+d[...,inter].abs().sum(-1))
def kn(z,B,s,t):
 d=z@B.T;inter=[i for i in range(B.shape[0]) if i not in(s,t)]
 return .5*(np.abs(d[...,t]-d[...,s])+np.abs(d[...,inter]).sum(-1))
def data():
 n,ods,qb,qbg,ps,sv,T=base.setup(); rng=np.random.default_rng(102111);pairs=[];elapsed=time.time();warm=None
 # The same 384 training and 96 validation endpoint pairs for edge and cut.
 for j in range(6):
  for i in range(80):
   q=rng.uniform(.65,1.3,6)*qb;h=qb[j]*(.025 if i%2==0 else .15);a=q.copy();b=q.copy();a[j]-=h;b[j]+=h
   f0,_,_,_=sv.solve(np.r_[a,qbg],warm);warm=f0;f1,_,_,_=sv.solve(np.r_[b,qbg],warm);warm=f1
   pairs.append((a,b,f0,f1,j,i<64))
 pair_seconds=time.time()-elapsed
 qa,qz,fa,fz,jid,train=map(np.array,zip(*pairs));z=cost(fz,n)-cost(fa,n);es=[];ks=[]
 for j,(s,t) in enumerate(ods):
  ix=(jid==j)&train;es.append(max(np.sqrt(np.mean(np.sum(z[ix]**2,axis=1))),1e-6));ks.append(max(np.sqrt(np.mean(kn(z[ix],n.B,s,t)**2)),1e-6))
 # Solve all 480 pair centers and sensitivities: 384 enter response training;
 # the 96 validation-center derivative labels are unused but their cost is included.
 elapsed=time.time();qm=.5*(qa+qz);fm=[];jm=[];warm=None
 for i,q in enumerate(qm):
  f,h,p,_=sv.solve(np.r_[q,qbg],warm);warm=f;jac,*_=sv.sensitivity(f,h,p,np.r_[q,qbg]);fm.append(f);jm.append(jac[:,jid[i]])
 response_seconds=time.time()-elapsed; jm=np.array(jm)
 ds=np.load(base.old.DATASET);fs=np.sqrt(np.mean(ds['F'][:2500]**2,axis=0))*.01
 js=[max(np.sqrt(np.mean(np.sum(jm[(jid==j)&train]**2,axis=1))),1e-6) for j in range(6)]
 np.savez_compressed(O/'labels.npz',a=qa,b=qz,F0=fa,F1=fz,z=z,j=jid,train=train,qm=qm,Fm=fm,J=jm,escale=es,kscale=ks,jscale=js,fscale=fs)
 # 24 paths, balanced active-query counts within each group. Mixed and same signs.
 elapsed=time.time();rng=np.random.default_rng(102112);qs=[];ff=[];jj=[];groups=[];ts=np.linspace(0,1,41)
 for k in [1,2,3,6]:
  for r in range(6):
   a=rng.uniform(.65,1.3,6)*qb;b=a.copy();active=[(r+i)%6 for i in range(k)]
   delta=rng.uniform(.15,.3,k)*qb[active]
   if r%2:delta[::2]*=-1
   b[active]+=delta;Q=a+ts[:,None]*(b-a);F=[];J=[];warm=None
   for q in Q:
    f,h,p,_=sv.solve(np.r_[q,qbg],warm);warm=f;jac,*_=sv.sensitivity(f,h,p,np.r_[q,qbg]);F.append(f);J.append(jac)
   qs.append(Q);ff.append(F);jj.append(J);groups.append(k)
   print('path reference',k,r,flush=True)
 np.savez_compressed(O/'paths.npz',Q=qs,F=ff,J=jj,ts=ts,group=groups)
 (O/'protocol.json').write_text(json.dumps({'pair_seed':102111,'test_seed':102112,'train_pairs_per_query':64,'validation_pairs_per_query':16,'pair_halfwidth_fraction':[.025,.15],'train_domain_centers':[.65,1.3],'steps':STEPS,'budget_points':BUDGETS,'validation_every':EVERY,'methods':METHODS,'lambda_edge':1.,'lambda_cut':CUT_WEIGHT,'selection':'Minimum validation K MSE among states with validation flow error <= 1.1 times original checkpoint; fallback minimum validation flow error, common for all four methods. Test paths never used.','pair_reference_seconds':pair_seconds,'response_reference_seconds':response_seconds,'test_reference_seconds':time.time()-elapsed,'pair_endpoint_solves':960,'response_center_solves_and_sensitivities':480,'test_solves_and_sensitivities':984,'max_solver_residual':sv.max_residual},indent=2))

def train(seed,method):
 st=time.time();n,ods,qb,qbg,ps,sv,T=base.setup();z=np.load(base.old.DATASET);lab=np.load(O/'labels.npz');m=base.old.build_fullnet_model(ps,qbg,seed);m.load_state_dict(torch.load(base.old.CKPT_DIR/f'seed{seed}.pt',weights_only=False)['best']);opt=torch.optim.Adam(m.parameters(),lr=LR)
 x=torch.tensor(z['Q'][:2500]);f=torch.tensor(z['F'][:2500]);a=torch.tensor(lab['a']);b=torch.tensor(lab['b']);zt=torch.tensor(lab['z']);qm=torch.tensor(lab['qm']);jm=torch.tensor(lab['J']);fs=torch.tensor(lab['fscale']);es=torch.tensor(lab['escale']);ks=torch.tensor(lab['kscale']);js=torch.tensor(lab['jscale']);B=torch.tensor(n.B);t0=torch.tensor(n.free_flow_time);cap=torch.tensor(n.capacity)
 train_ids=[np.where((lab['j']==j)&lab['train'])[0] for j in range(6)];vi=np.where(~lab['train'])[0];rng=np.random.default_rng(7102+seed)
 def g(ft):return ft*t0*(1+.15*(ft/cap)**4)
 def validate():
  with torch.no_grad():
   fa=m(a[vi]);fb=m(b[vi]);err=g(fb)-g(fa)-zt[vi];values=[]
   for j,(s,t) in enumerate(ods):values.extend((kt(err[lab['j'][vi]==j],B,s,t)/ks[j]).square().tolist())
   flow=torch.cat([fa-torch.tensor(lab['F0'][vi]),fb-torch.tensor(lab['F1'][vi])]);ref=torch.tensor(np.concatenate([lab['F0'][vi],lab['F1'][vi]]));fr=float((torch.linalg.vector_norm(flow,dim=1)/torch.linalg.vector_norm(ref,dim=1)).mean())
  return {'K_mse':float(np.mean(values)),'flow_relative':fr}
 original=validate();logs=[];best=None;eligible_best=float('inf');fallback=float('inf');beststep=0
 for step in range(1,STEPS+1):
  ix=rng.integers(0,2500,BS);j=(step-1)%6;ij=rng.choice(train_ids[j],BJ);opt.zero_grad();lf=(((m(x[ix])-f[ix])/fs)**2).mean();loss=lf;le=lc=lr=torch.tensor(0.)
  if method in ['edge','cut']:
   err=g(m(b[ij]))-g(m(a[ij]))-zt[ij];le=err.square().sum(1).mean()/es[j]**2;loss=loss+le
   if method=='cut':
    s,t=ods[j];lc=(kt(err,B,s,t)/ks[j]).square().mean();loss=loss+CUT_WEIGHT*lc
  elif method=='response':
   v=torch.zeros_like(qm[ij]);v[:,j]=1;_,jp=torch.func.jvp(m,(qm[ij],),(v,));lr=(jp-jm[ij]).square().sum(1).mean()/js[j]**2;loss=loss+lr
  loss.backward();opt.step()
  if step%EVERY==0:
   val=validate();eligible=val['flow_relative']<=1.1*original['flow_relative'];take=(eligible and val['K_mse']<eligible_best) or (eligible_best==float('inf') and val['flow_relative']<fallback)
   if take:
    best={k:v.detach().clone() for k,v in m.state_dict().items()};beststep=step;fallback=val['flow_relative']
    if eligible:eligible_best=val['K_mse']
   logs.append({'step':step,'flow_loss':float(lf.detach()),'edge_loss':float(le.detach()),'cut_loss':float(lc.detach()),'response_loss':float(lr.detach()),'validation':val,'elapsed_seconds':time.time()-st})
   print(seed,method,step,val,flush=True)
  if step in BUDGETS:torch.save({'best':m.state_dict(),'step':step},O/f'{method}_{seed}_budget{step}.pt')
 torch.save({'best':best,'selected_step':beststep},O/f'{method}_{seed}.pt');(O/f'{method}_{seed}_cost.json').write_text(json.dumps({'seconds':time.time()-st,'selected_step':beststep,'original_validation':original,'logs':logs},indent=2))

def evaluate():
 st=time.time();n,ods,qb,qbg,ps,sv,T=base.setup();paths=np.load(O/'paths.npz');Q=paths['Q'];F=paths['F'];J=paths['J'];ts=paths['ts'];ds=np.load(base.old.DATASET);ref=np.array([attribution(np.array([T.grad(f) for f in F[pi]]),J[pi],ts,Q[pi,-1]-Q[pi,0]) for pi in range(24)]);rows=[];arr={'C_reference':ref};floor=1e-6;masks=screenline_masks(ods)
 for method in METHODS:
  for seed in range(5):
   m=base.old.build_fullnet_model(ps,qbg,seed);m.load_state_dict(torch.load(O/f'{method}_{seed}.pt',weights_only=False)['best'])
   with torch.no_grad():pred=m(torch.tensor(ds['Q'][3357:])).numpy();flow=float(np.mean(np.linalg.norm(pred-ds['F'][3357:],axis=1)/np.linalg.norm(ds['F'][3357:],axis=1))*100)
   infer_st=time.time();Cs=[];Fs=[];Js=[]
   for pi in range(24):
    q=torch.tensor(Q[pi]);f=m(q).detach().numpy();js=[]
    for j in range(6):
     v=torch.zeros_like(q);v[:,j]=1;_,jv=torch.func.jvp(m,(q,),(v,));js.append(jv.detach().numpy())
    jac=np.stack(js,axis=2);C=attribution(np.array([T.grad(f0) for f0 in f]),jac,ts,Q[pi,-1]-Q[pi,0]);Cs.append(C);Fs.append(f);Js.append(jac)
    for j,(s,t) in enumerate(ods):
     if abs(Q[pi,-1,j]-Q[pi,0,j])<1e-10:continue
     err=C[:,j]-ref[pi,:,j];ka=float(kn(err,n.B,s,t));kr=float(kn(ref[pi,:,j],n.B,s,t));cr=np.linalg.norm(ref[pi,:,j]);sr=abs(ref[pi,:,j].sum());dt=T.value(f[-1])-T.value(f[0]);complete=abs(C.sum()-dt)/max(abs(dt),floor)
     rows.append({'method':method,'seed':seed,'path':pi,'group':int(paths['group'][pi]),'query':j,'flow_test_pct':flow,'increment_pct':float(100*np.linalg.norm((f[-1]-f[0])-(F[pi,-1]-F[pi,0]))/max(np.linalg.norm(F[pi,-1]-F[pi,0]),floor)),'screenline_abs':float(np.max(np.abs(masks[j]@n.B@err))),'screenline_pct':float(100*np.max(np.abs(masks[j]@n.B@err))/max(np.max(np.abs(masks[j]@n.B@ref[pi,:,j])),floor)),'boundary_abs':ka,'reference_boundary_scale':kr,'boundary_pct':100*ka/max(kr,floor),'column_pct':float(100*np.linalg.norm(err)/max(cr,floor)),'system_pct':float(100*abs(err.sum())/max(sr,floor)),'completeness_relative':float(complete)})
   (O/f'{method}_{seed}_inference.json').write_text(json.dumps({'seconds':time.time()-infer_st,'paths':24,'states_per_path':41,'jvp_per_state':6},indent=2));arr[f'C_{method}_{seed}']=Cs;arr[f'F_{method}_{seed}']=Fs;arr[f'J_{method}_{seed}']=Js;print('evaluated',method,seed,flush=True)
 np.savez_compressed(O/'evaluation.npz',**arr)
 ref_checks=[]
 for pi in range(24):
  dt=T.value(F[pi,-1])-T.value(F[pi,0]);item={'path':pi,'completeness_relative':float(abs(ref[pi].sum()-dt)/max(abs(dt),floor))}
  if pi<6:
   exact=cost(F[pi,-1],n)-cost(F[pi,0],n);item['single_od_endpoint_column_relative']=float(np.linalg.norm(ref[pi,:,pi]-exact)/max(np.linalg.norm(exact),floor))
  ref_checks.append(item)
 (O/'reference_integration_checks.json').write_text(json.dumps(ref_checks,indent=2))
 # Path is the independent unit: average active queries then seeds, then summarize 24 paths.
 pathrows=[]
 for method in METHODS:
  for pi in range(24):
   rr=[r for r in rows if r['method']==method and r['path']==pi];pathrows.append({'method':method,'path':pi,'group':rr[0]['group'],**{k:float(np.mean([r[k] for r in rr])) for k in ['flow_test_pct','boundary_abs','boundary_pct','column_pct','system_pct','screenline_pct','increment_pct']}})
 summary={}
 for method in METHODS:
  rr=[r for r in pathrows if r['method']==method];summary[method]={k:float(np.median([r[k] for r in rr])) for k in ['flow_test_pct','boundary_abs','boundary_pct','column_pct','system_pct','screenline_pct','increment_pct']};summary[method]['boundary_p90_pct']=float(np.quantile([r['boundary_pct'] for r in rr],.9))
 (O/'results.json').write_text(json.dumps({'rows':rows,'path_means':pathrows,'summary':summary,'denominator_absolute_floor':floor,'evaluation_seconds':time.time()-st},indent=2));print(summary)
 budget_evaluate();report(summary,rows,pathrows)

def budget_evaluate():
 """Exact single-OD endpoint attribution on six frozen test paths; no quadrature."""
 n,ods,qb,qbg,ps,sv,T=base.setup();p=np.load(O/'paths.npz');rows=[]
 for method in METHODS:
  for seed in range(5):
   m=base.old.build_fullnet_model(ps,qbg,seed)
   for budget in BUDGETS:
    m.load_state_dict(torch.load(O/f'{method}_{seed}_budget{budget}.pt',weights_only=False)['best'])
    with torch.no_grad():a=m(torch.tensor(p['Q'][:6,0])).numpy();b=m(torch.tensor(p['Q'][:6,-1])).numpy()
    z=cost(b,n)-cost(a,n);zr=cost(p['F'][:6,-1],n)-cost(p['F'][:6,0],n)
    for pi,(s,t) in enumerate(ods):
     scale=float(kn(zr[pi],n.B,s,t));rows.append({'method':method,'seed':seed,'budget':budget,'path':pi,'boundary_pct':float(100*kn(z[pi]-zr[pi],n.B,s,t)/max(scale,1e-6))})
 (O/'budget_results.json').write_text(json.dumps({'scope':'Six independent single-OD test paths, exact endpoint columns; checkpoints at fixed budgets without validation selection','rows':rows},indent=2))

def report(summary,rows,pathrows):
 protocol=json.loads((O/'protocol.json').read_text());p=Path('revision-stage/v102');p.mkdir(parents=True,exist_ok=True);lines=['# v102 有限变化端点监督实验','', '六查询各64训练对、16验证对；小/中幅度对半。四方法从原五种子相同检查点继续训练1200步，学习率3e-5，共同验证规则每100步选择一次。边级与边界方法使用相同端点标签；普通响应使用相同中心状态但需另算导数。测试24条独立路径，每类6条；每一类中查询出现次数均衡。统计先平均同路径查询及五种子，再以24路径为单位汇总。','', '固定权重：边级1、边界0.01；本轮没有调参搜索。原始数据及全部日志保存在 experiments/results/e111_v102/。','', '| 方法 | 流量% | 列误差% | 系统误差% | 边界绝对误差 | 边界% | 边界90%分位 |','|---|---:|---:|---:|---:|---:|---:|']
 names={'flow':'流量','edge':'流量＋边级端点','cut':'流量＋边级端点＋边界','response':'流量＋普通响应'}
 for meth in METHODS:
  v=summary[meth];lines.append('|'+names[meth]+'|'+'|'.join(f'{v[k]:.4f}' for k in ['flow_test_pct','column_pct','system_pct','boundary_abs','boundary_pct','boundary_p90_pct'])+'|')
 lines+=['','所有量均为实际计算结果；是否整体改善须联合流量、系统和边界误差判断，不能根据局部查询或单一分位数宣布优势。','',f"端点标签求解 {protocol['pair_reference_seconds']:.1f}秒；响应中心求解与导数 {protocol['response_reference_seconds']:.1f}秒；测试参考 {protocol['test_reference_seconds']:.1f}秒。",'', '## 各组及各查询边界误差（均值，%）']
 for method in METHODS:
  lines.append(names[method]+': '+', '.join(f"{g} OD={np.mean([r['boundary_pct'] for r in pathrows if r['method']==method and r['group']==g]):.3f}" for g in [1,2,3,6]))
  lines.append('逐查询: '+', '.join(f"{protocol['query_ods_1based'][j]}={np.mean([r['boundary_pct'] for r in rows if r['method']==method and r['query']==j]):.3f}" for j in range(6)))
 costs={method:[json.loads((O/f'{method}_{seed}_cost.json').read_text()) for seed in range(5)] for method in METHODS}
 lines+=['','## 计算成本、种子离散及预算曲线','', '| 方法 | 训练秒中位数 | 24路径归因评估秒中位数 | 选择步数（五种子） | 每种子24路径边界均值的范围% |','|---|---:|---:|---|---|']
 for method in METHODS:
  vals=[np.mean([np.mean([r['boundary_pct'] for r in rows if r['method']==method and r['seed']==seed and r['path']==pi]) for pi in range(24)]) for seed in range(5)]
  infer=[json.loads((O/f'{method}_{seed}_inference.json').read_text())['seconds'] for seed in range(5)]
  lines.append(f"|{names[method]}|{np.median([v['seconds'] for v in costs[method]]):.1f}|{np.median(infer):.1f}|{[v['selected_step'] for v in costs[method]]}|[{min(vals):.3f},{max(vals):.3f}]|")
 budget=json.loads((O/'budget_results.json').read_text())['rows']
 lines+=['','预算曲线固定取六条单OD测试路径，以两个端点直接计算精确归因列；每条路径先平均种子再对六条路径取中位数。这里只比较固定步数，不使用验证选优后的检查点。','', '| 方法 | 300步 | 600步 | 1200步 |','|---|---:|---:|---:|']
 for method in METHODS:
  vals=[np.median([np.mean([r['boundary_pct'] for r in budget if r['method']==method and r['budget']==b and r['path']==pi]) for pi in range(6)]) for b in BUDGETS]
  lines.append('|'+names[method]+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|')
 lines+=['','直线截面和流量增量误差（24路径中位数，%）：']
 for method in METHODS:lines.append(f"{names[method]}：直线截面{summary[method]['screenline_pct']:.3f}%，流量增量{summary[method]['increment_pct']:.3f}%。")
 lines+=['','训练计时包含初始化、验证与检查点写入，五种子并发且方法依次运行；共享机器资源争抢使分方法墙钟时长不能用作速度优势证据。24路径归因评估计时包含前向、六个方向JVP、积分与指标组装。所有成本仅为本轮新增成本，不含旧流量标签及预训练。','响应实际求解480个端点对中心及其导数，384个中心导数用于训练，96个验证中心导数未被使用；全部53.1秒均计入。','边级和边界方法各有种子0、2、3没有满足验证流量阈值的检查点，按冻结规则退回流量验证误差最小点；不能说所有最终模型均满足该阈值。']
 lines+=['',f"参考边界分母最小值为{min(r['reference_boundary_scale'] for r in rows):.6g}；预设绝对下限为1e-6。",f"积分完整性相对残差最大值{max(r['completeness_relative'] for r in rows):.6g}。"]
 (p/'LEARNING_RESULTS.md').write_text('\n'.join(lines)+'\n')
 tex=[r'\begin{table}[t]\centering\small',r'\caption{Endpoint supervision on 24 independent paths (six each with one, two, three and six active OD queries), five seeds. Flow is the five-seed mean relative error on the original held-out demand states. Other entries first average active queries and seeds within each path, then report medians over paths; the last column gives the 90th percentile. All methods share 1,200 continuation updates and a common validation selection rule.}',r'\label{tab:pairs-v102}',r'\begin{tabular}{lrrrrrr}\toprule',r'Method & Flow (\%) & Column (\%) & System (\%) & Cut abs. & Cut (\%) & Cut P90 (\%)\\\midrule']
 for meth,label in [('flow','Flow'),('edge','Flow + endpoint edge'),('cut','Flow + endpoint edge + cut'),('response','Flow + response')]:tex.append(label+' & '+' & '.join((f'{summary[meth][k]:.4f}' if k=='flow_test_pct' else f'{summary[meth][k]:.2f}') for k in ['flow_test_pct','column_pct','system_pct','boundary_abs','boundary_pct','boundary_p90_pct'])+r'\\')
 tex += [r'\bottomrule\end{tabular}\end{table}'];Path('paper/results_pairs_v102.tex').write_text('\n'.join(tex)+'\n')
 extra=[r'\begin{table}[t]\centering\small',r'\caption{Fixed-budget endpoint boundary error (\%) on six independent single-OD test paths. Seeds are averaged within each path before taking the path median.}',r'\label{tab:pair-budget-v102}',r'\begin{tabular}{lrrr}\toprule Method & 300 updates & 600 updates & 1,200 updates\\\midrule']
 for method in METHODS:
  vals=[np.median([np.mean([r['boundary_pct'] for r in budget if r['method']==method and r['budget']==b and r['path']==pi]) for pi in range(6)]) for b in BUDGETS]
  label={'flow':'Flow','edge':'Flow + endpoint edge','cut':'Flow + endpoint edge + cut','response':'Flow + response'}[method]
  extra.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
 extra +=[r'\bottomrule\end{tabular}\end{table}'];Path('paper/results_pair_budget_v102.tex').write_text('\n'.join(extra)+'\n')
 detail=[r'\begin{table}[t]\centering\small',r'\caption{Mean unrestricted boundary error (\%) by number of active queries. Each group contains six independent paths, each balanced across queries.}',r'\label{tab:pair-groups-v102}',r'\begin{tabular}{lrrrr}\toprule Method & One OD & Two ODs & Three ODs & Six ODs\\\midrule']
 for method in METHODS:
  label={'flow':'Flow','edge':'Flow + endpoint edge','cut':'Flow + endpoint edge + cut','response':'Flow + response'}[method]
  detail.append(label+' & '+' & '.join(f"{np.mean([r['boundary_pct'] for r in pathrows if r['method']==method and r['group']==g]):.2f}" for g in [1,2,3,6])+r'\\')
 detail +=[r'\bottomrule\end{tabular}\end{table}'];Path('paper/results_pair_groups_v102.tex').write_text('\n'.join(detail)+'\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['data','train','eval']);a.add_argument('--seed',type=int,default=0);a.add_argument('--method',choices=METHODS,default='flow');v=a.parse_args()
 if v.phase=='data':data()
 elif v.phase=='train':train(v.seed,v.method)
 else:evaluate()
