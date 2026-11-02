"""Re-select continuation models including update zero and compare matched paths."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
import run_e111_v102_pairs as p
from run_e83_v82_validation import attribution
O=Path('experiments/results/v104_learning');O.mkdir(exist_ok=True)
OLD=p.O
def main():
 n,ods,qb,qbg,ps,sv,T=p.base.setup();lab=np.load(OLD/'labels.npz');paths=np.load(OLD/'paths.npz');old=np.load(OLD/'evaluation.npz');Q=paths['Q'];F=paths['F'];Jref=paths['J'];ref=old['C_reference'];masks=p.screenline_masks(ods); vi=np.where(~lab['train'])[0]
 initial={}; selection=[]; costrows=[]
 if (O/'initial_evaluation.npz').exists():
  costrows=json.loads((O/'initial_inference_cost.json').read_text()) if (O/'initial_inference_cost.json').exists() else []
  cache=np.load(O/'initial_evaluation.npz');initial={seed:tuple(cache[f'{kind}_{seed}'] for kind in ['F','C','J']) for seed in range(5)}
 else:
  for seed in range(5):
   m=p.base.old.build_fullnet_model(ps,qbg,seed);m.load_state_dict(torch.load(p.base.old.CKPT_DIR/f'seed{seed}.pt',weights_only=False)['best'])
   st=time.perf_counter(); ff=[];cc=[];jj=[]
   for pi in range(24):
    q=torch.tensor(Q[pi]);f=m(q).detach().numpy();J=[]
    for j in range(6):
     v=torch.zeros_like(q);v[:,j]=1;_,jv=torch.func.jvp(m,(q,),(v,));J.append(jv.detach().numpy())
    J=np.stack(J,axis=2);ff.append(f);jj.append(J);cc.append(attribution(np.array([T.grad(f0) for f0 in f]),J,paths['ts'],Q[pi,-1]-Q[pi,0]))
   initial[seed]=(np.array(ff),np.array(cc),np.array(jj));costrows.append({'seed':seed,'initial_path_explanation_seconds':time.perf_counter()-st});print('initial',seed,flush=True)
  np.savez_compressed(O/'initial_evaluation.npz',**{f'{kind}_{seed}':initial[seed][idx] for seed in range(5) for idx,kind in enumerate(['F','C','J'])})
 if costrows: (O/'initial_inference_cost.json').write_text(json.dumps(costrows,indent=2))
 rows=[]
 for method in p.METHODS:
  for seed in range(5):
   hist=json.loads((OLD/f'{method}_{seed}_cost.json').read_text()); original=hist['original_validation'];eligible=[l for l in hist['logs'] if l['validation']['flow_relative']<=1.1*original['flow_relative']];winner=min([{'step':0,'validation':original}]+eligible,key=lambda l:l['validation']['K_mse']);step=winner['step']
   if step==0:fp,cp,jp=initial[seed]
   else:
    assert step==hist['selected_step'],(method,seed,step,hist['selected_step'])
    fp=old[f'F_{method}_{seed}'];cp=old[f'C_{method}_{seed}'];jp=old[f'J_{method}_{seed}']
   selection.append({'method':method,'seed':seed,'step':step,'initial':original,'selected':winner['validation'],'eligible':bool(winner['validation']['flow_relative']<=1.1*original['flow_relative'])})
   for pi in range(24):
    flow=100*np.mean(np.linalg.norm(fp[pi]-F[pi],axis=1)/np.linalg.norm(F[pi],axis=1))
    for j,(s,t) in enumerate(ods):
     if abs(Q[pi,-1,j]-Q[pi,0,j])<1e-10: continue
     e=cp[pi,:,j]-ref[pi,:,j];r=ref[pi,:,j];geo=masks[j]@n.B
     # Exact product decomposition at common reference states along this path.
     gl=np.array([T.grad(v) for v in fp[pi]]);gr=np.array([T.grad(v) for v in F[pi]])
     response=attribution(gl,jp[pi]-Jref[pi],paths['ts'],Q[pi,-1]-Q[pi,0])[:,j]
     weight=attribution(gl-gr,Jref[pi],paths['ts'],Q[pi,-1]-Q[pi,0])[:,j]
     assert np.max(np.abs(e-response-weight))<1e-7
     rows.append({'method':method,'seed':seed,'path':pi,'query':j,'flow_pct':float(flow),'column_pct':float(100*np.linalg.norm(e)/max(np.linalg.norm(r),1e-6)),'system_pct':float(100*abs(e.sum())/max(abs(r.sum()),1e-6)),'boundary_abs':float(p.kn(e,n.B,s,t)),'boundary_pct':float(100*p.kn(e,n.B,s,t)/max(p.kn(r,n.B,s,t),1e-6)),'geometry_pct':float(100*np.max(abs(geo@e))/max(np.max(abs(geo@r)),1e-6)),'response_norm':float(np.linalg.norm(response)),'weight_norm':float(np.linalg.norm(weight))})
 summary={};pathrows=[]
 metrics=['flow_pct','column_pct','system_pct','boundary_abs','boundary_pct','geometry_pct','response_norm','weight_norm']
 for method in p.METHODS:
  mr=[]
  for pi in range(24):
   rr=[r for r in rows if r['method']==method and r['path']==pi];v={'method':method,'path':pi,**{k:float(np.mean([r[k] for r in rr])) for k in metrics}};pathrows.append(v);mr.append(v)
  summary[method]={k:float(np.median([r[k] for r in mr])) for k in metrics};summary[method]['response_dominant_paths']=sum(r['response_norm']>r['weight_norm'] for r in mr)
 result={'selection':selection,'rows':rows,'path_means':pathrows,'summary':summary,'initial_inference_cost':costrows};(O/'results.json').write_text(json.dumps(result,indent=2));print(summary,flush=True)
 # Label costs use the same fixed labels, solver and warm-start rule within each task.
 if (O/'label_costs.json').exists(): return
 costs={}
 for name,ids,centers in [('endpoint_train',np.where(lab['train'])[0],False),('common_endpoint_validation',vi,False),('response_train',np.where(lab['train'])[0],True)]:
  st=time.perf_counter();warm=None
  for i in ids:
   if centers:
    q=lab['qm'][i];f,h,prob,_=sv.solve(np.r_[q,qbg],warm);warm=f;sv.sensitivity(f,h,prob,np.r_[q,qbg])
   else:
    for key in ['a','b']:
     f,*_=sv.solve(np.r_[lab[key][i],qbg],warm);warm=f
  costs[name]={'seconds':time.perf_counter()-st,'states':len(ids)*(1 if centers else 2),'sensitivities':len(ids) if centers else 0};print(name,costs[name],flush=True)
 costs['scope']='New CPU measurements of method-specific labels and shared validation labels; existing flow data and pretraining excluded. Each task begins without a warm state; subsequent states reuse the preceding flow.'
 (O/'label_costs.json').write_text(json.dumps(costs,indent=2))
if __name__=='__main__':main()
