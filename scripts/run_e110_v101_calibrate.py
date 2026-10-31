"""Exploratory validation-only calibration after the frozen negative e109 run."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,sys,shutil,concurrent.futures
from pathlib import Path
import numpy as np
import torch
import run_e109_v101_structured as base
ROOT=base.R/'e110_v101';ROOT.mkdir(exist_ok=True)
ORIGINAL=base.O

def configure(folder,weight):
 base.O=Path(folder);base.O.mkdir(exist_ok=True);base.CUT_WEIGHT=weight
 if not (base.O/'labels.npz').exists():shutil.copyfile(ORIGINAL/'labels.npz',base.O/'labels.npz')

def validation(checkpoint):
 n,ods,qb,qbg,ps,sv,T=base.setup();lab=np.load(ORIGINAL/'labels.npz');ck=torch.load(checkpoint,weights_only=False);m=base.old.build_fullnet_model(ps,qbg,0);m.load_state_dict(ck['best']);q=torch.tensor(lab['Q'][256:]);ft=lab['F'][256:];jt=lab['J'][256:];fp=m(q).detach().numpy();js=[]
 for j in range(6):
  v=torch.zeros_like(q);v[:,j]=1.;_,jp=torch.func.jvp(m,(q,),(v,));js.append(jp.detach().numpy())
 jl=np.stack(js,axis=2);gp=np.array([T.grad(f) for f in fp]);gt=lab['g'][256:];h=gp[:,:,None]*jl-gt[:,:,None]*jt;d=np.einsum('ve,bej->bvj',n.B,h);loss=[]
 for j,(s,t) in enumerate(ods):
  inter=[v for v in range(24) if v not in(s,t)];k=.5*(np.abs(d[:,t,j]-d[:,s,j])+np.abs(d[:,inter,j]).sum(1));loss.append(np.mean((k/lab['kscale'][j])**2))
 return {'flow_relative_mean':float(np.mean(np.linalg.norm(fp-ft,axis=1)/np.linalg.norm(ft,axis=1))),'normalized_K_mse':float(np.mean(loss)),'K_mse_per_query':loss}

def candidate(w):
 st=time.time();folder=ROOT/f'candidate_{w}';configure(folder,w)
 if not(folder/'structure_0.pt').exists():base.train(0,'structure')
 v=validation(folder/'structure_0.pt');v['candidate_seconds']=time.time()-st
 (folder/'validation.json').write_text(json.dumps(v,indent=2));print('candidate',w,v,flush=True)
def select():
 results={};ref=validation(ORIGINAL/'response_0.pt');results['response']=ref;results['lambda1']=validation(ORIGINAL/'structure_0.pt')
 for w in [.001,.01,.1]:
  v=json.loads((ROOT/f'candidate_{w}'/'validation.json').read_text());v['eligible']=v['flow_relative_mean']<=1.1*ref['flow_relative_mean'];results[str(w)]=v
 eligible=[w for w in [.001,.01,.1] if results[str(w)]['eligible']];selected=min(eligible,key=lambda w:results[str(w)]['normalized_K_mse']) if eligible else None
 decision={'selection':'64 held-out derivative validation states only; no test paths read','selected_weight':selected,'results':results,'candidate_total_process_seconds':sum(results[str(w)]['candidate_seconds'] for w in [.001,.01,.1])};(ROOT/'selection.json').write_text(json.dumps(decision,indent=2));print('SELECTION',decision,flush=True)

def final_train(seed):
 dec=json.loads((ROOT/'selection.json').read_text());w=dec['selected_weight']
 if w is None:return
 folder=ROOT/'selected';configure(folder,w)
 if seed==0:shutil.copyfile(ROOT/f'candidate_{w}'/'structure_0.pt',folder/'structure_0.pt');shutil.copyfile(ROOT/f'candidate_{w}'/'structure_0_cost.json',folder/'structure_0_cost.json')
 elif not(folder/f'structure_{seed}.pt').exists():base.train(seed,'structure')

def newpaths():
 assert (ROOT/'selection.json').exists()
 n,ods,qb,qbg,ps,sv,T=base.setup();rng=np.random.default_rng(101030);qs=[];fs=[];js=[];ts=np.linspace(0,1,41);st=time.time()
 for pi in range(4):
  a=rng.uniform(.65,1.3,6)*qb;b=a.copy();active=[[0],[2,4],[0,1,5],list(range(6))][pi];changes=[[.35],[-.22,.3],[.25,-.3,.2],[.2,-.25,.3,-.2,.18,-.22]][pi];b[active]+=qb[active]*changes;Q=a[None,:]+ts[:,None]*(b-a)[None,:];F=[];J=[];warm=None
  for q in Q:
   f,h,p,_=sv.solve(np.r_[q,qbg],warm);warm=f;j,*_=sv.sensitivity(f,h,p,np.r_[q,qbg]);F.append(f);J.append(j)
  qs.append(Q);fs.append(F);js.append(J)
 np.savez_compressed(ROOT/'selected'/'paths.npz',Q=qs,F=fs,J=js,ts=ts)
 shutil.copyfile(ROOT/'selected'/'paths.npz',ROOT/'new_paths.npz')
 (ROOT/'newpaths_cost.json').write_text(json.dumps({'seed':101030,'seconds':time.time()-st,'states':164,'max_solver_residual':sv.max_residual},indent=2))

def evaluate():
 dec=json.loads((ROOT/'selection.json').read_text());configure(ROOT/'selected',dec['selected_weight'])
 for m in ['flow','response']:
  for seed in range(5):shutil.copyfile(ORIGINAL/f'{m}_{seed}.pt',base.O/f'{m}_{seed}.pt')
 # New, unselected paths first. Preserve all results; first protocol remains untouched.
 base.TABLE_OUTPUT=ROOT/'calibration_new_table.tex'
 base.evaluate()
 for filename in ['results.json','evaluation.npz']:shutil.copyfile(base.O/filename,ROOT/f'new_{filename}')
 shutil.copyfile(ORIGINAL/'paths.npz',base.O/'paths.npz');base.TABLE_OUTPUT=ROOT/'calibration_old_table.tex';base.evaluate()
 for filename in ['results.json','evaluation.npz']:shutil.copyfile(base.O/filename,ROOT/f'old_{filename}')
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='candidate':candidate(float(sys.argv[2]))
 elif phase=='select':select()
 elif phase=='train':final_train(int(sys.argv[2]))
 elif phase=='paths':newpaths()
 elif phase=='eval':evaluate()
