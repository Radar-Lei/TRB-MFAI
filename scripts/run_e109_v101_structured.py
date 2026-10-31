"""Equal-budget checkpoint continuation: flow, response, structural response.
No test path participates in objective/step selection. Float64 CPU, exact JVP.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,time,json,argparse
from pathlib import Path
import numpy as np
import torch
import run_e93_v89_fullbg_surrogate as old
from run_e83_v82_validation import attribution
R=Path(__file__).resolve().parent/'results'; O=R/'e109_v101';O.mkdir(exist_ok=True)
torch.set_num_threads(1)
STEPS=600; BS=32; BJ=8; LR=3e-5; CUT_WEIGHT=1.0; TABLE_OUTPUT=Path('paper/learning_results_v101.tex')

def setup():
 n=old.load_sioux_falls();ods,trips,_=old.select_ods(n);bg=[o for o,d in trips.items() if d>0 and o not in ods];qb=np.array([trips[o] for o in ods]);qbg=np.array([trips[o] for o in bg]);ps=old.PathStructure(n,ods+bg)
 return n,ods,qb,qbg,ps,old.EdgeSUE(ps,study_k=6),old.TSTTT(n.free_flow_time,n.capacity,n.m)

def data():
 st=time.time();n,ods,qb,qbg,ps,sv,T=setup();z=np.load(old.DATASET); rng=np.random.default_rng(20261029)
 ids=rng.choice(2500,256,replace=False);vids=2500+rng.choice(857,64,replace=False); Q=z['Q'][np.r_[ids,vids]];Fs=[];Js=[];warm=None
 for i,q in enumerate(Q):
  f,h,p,_=sv.solve(np.r_[q,qbg],warm);warm=f; J,*_=sv.sensitivity(f,h,p,np.r_[q,qbg]);Fs.append(f);Js.append(J)
  if i%80==0:print('reference derivative',i,flush=True)
 F=np.array(Fs);J=np.array(Js);g=np.array([T.grad(f) for f in F]); w=g[:,:,None]*J;d=np.einsum('ve,bej->bvj',n.B,w)
 scales=[]
 for j,(s,t) in enumerate(ods):
  ii=[v for v in range(24) if v not in (s,t)];k=.5*(np.abs(d[:,t,j]-d[:,s,j])+np.abs(d[:,ii,j]).sum(1));scales.append(max(np.sqrt(np.mean(k[:256]**2)),1e-6))
 np.savez_compressed(O/'labels.npz',Q=Q,F=F,J=J,g=g,ids=ids,vids=vids,jscale=np.sqrt(np.mean(np.sum(J[:256]**2,axis=1),axis=0)),kscale=scales, fscale=np.sqrt(np.mean(z['F'][:2500]**2,axis=0))*.01)
 # Fixed paths, separate RNG, varied baseline and signed changes; 4 independent paths.
 rng=np.random.default_rng(101029);paths=[]
 for pth in range(4):
  a=rng.uniform(.65,1.3,6)*qb;b=a.copy()
  active=[[0],[2,4],[0,1,5],list(range(6))][pth];changes=[ [.35],[-.22,.3],[.25,-.3,.2],[.2,-.25,.3,-.2,.18,-.22] ][pth]
  b[active]+=qb[active]*changes;paths.append((a,b))
 Fs=[];Js=[];Qs=[];ts=np.linspace(0,1,41)
 for pth,(a,b) in enumerate(paths):
  fq=[];jq=[];qq=[];warm=None
  for t in ts:
   q=a+t*(b-a);f,h,p,_=sv.solve(np.r_[q,qbg],warm);warm=f;J,*_=sv.sensitivity(f,h,p,np.r_[q,qbg]);fq.append(f);jq.append(J);qq.append(q)
  Fs.append(fq);Js.append(jq);Qs.append(qq)
 np.savez_compressed(O/'paths.npz',Q=Qs,F=Fs,J=Js,ts=ts)
 (O/'reference_cost.json').write_text(json.dumps({'seconds':time.time()-st,'derivative_labels':320,'test_states':164,'max_solver_residual':sv.max_residual,'seed':101029},indent=2))
 print('reference data done',time.time()-st,flush=True)

def train(seed,method):
 st=time.time();n,ods,qb,qbg,ps,sv,T=setup();z=np.load(old.DATASET);lab=np.load(O/'labels.npz'); ck=torch.load(old.CKPT_DIR/f'seed{seed}.pt',weights_only=False);m=old.build_fullnet_model(ps,qbg,seed);m.load_state_dict(ck['best']);opt=torch.optim.Adam(m.parameters(),lr=LR)
 x=torch.tensor(z['Q'][:2500],dtype=torch.float64);f=torch.tensor(z['F'][:2500],dtype=torch.float64);xj=torch.tensor(lab['Q'][:256],dtype=torch.float64);fj=torch.tensor(lab['F'][:256],dtype=torch.float64);jj=torch.tensor(lab['J'][:256],dtype=torch.float64)
 fs=torch.tensor(lab['fscale']);js=torch.tensor(lab['jscale']);ks=torch.tensor(lab['kscale']);B=torch.tensor(n.B); t0=torch.tensor(n.free_flow_time);cap=torch.tensor(n.capacity)
 rng=np.random.default_rng(7000+seed);logs=[]
 for step in range(STEPS):
  ix=rng.integers(0,2500,BS); ij=rng.integers(0,256,BJ);j=step%6;opt.zero_grad();lossf=(((m(x[ix])-f[ix])/fs)**2).mean();loss=lossf
  lj=lk=torch.tensor(0.)
  if method!='flow':
   v=torch.zeros_like(xj[ij]);v[:,j]=1.;fp,jp=torch.func.jvp(m,(xj[ij],),(v,));lj=((jp-jj[ij,:,j])**2).sum(1).mean()/js[j]**2;loss=loss+lj
   if method=='structure':
    gp=t0*(1+.75*(fp/cap)**4);gr=t0*(1+.75*(fj[ij]/cap)**4);d=(gp*jp-gr*jj[ij,:,j])@B.T;s,t=ods[j];inter=[v for v in range(24) if v not in (s,t)];k=.5*((d[:,t]-d[:,s]).abs()+d[:,inter].abs().sum(1));lk=(k/ks[j]).square().mean();loss=loss+CUT_WEIGHT*lk
  loss.backward();opt.step()
  if step%100==0: print(seed,method,step,float(lossf.detach()),float(lj.detach()),float(lk.detach()),time.time()-st,flush=True);logs.append([step,float(lossf.detach()),float(lj.detach()),float(lk.detach())])
 torch.save({'best':m.state_dict(),'seed':seed,'method':method},O/f'{method}_{seed}.pt')
 (O/f'{method}_{seed}_cost.json').write_text(json.dumps({'seconds':time.time()-st,'steps':STEPS,'lr':LR,'batch_flow':BS,'batch_response':BJ,'logs':logs},indent=2))

def evaluate():
 st=time.time();n,ods,qb,qbg,ps,sv,T=setup();p=np.load(O/'paths.npz');Q=p['Q'];F=p['F'];J=p['J'];ts=p['ts'];ds=np.load(old.DATASET);allrows=[];outarr={};ref=[];checks=[]
 for pi in range(4):ref.append(attribution(np.array([T.grad(f) for f in F[pi]]),J[pi],ts,Q[pi,-1]-Q[pi,0]))
 ref=np.array(ref)
 for method in ['original','flow','response','structure']:
  for seed in range(5):
   path=old.CKPT_DIR/f'seed{seed}.pt' if method=='original' else O/f'{method}_{seed}.pt';ck=torch.load(path,weights_only=False);m=old.build_fullnet_model(ps,qbg,seed);m.load_state_dict(ck['best']);Cs=[];fl=[];jls=[]
   with torch.no_grad():
    xt=torch.tensor(ds['Q'][3357:],dtype=torch.float64);pred=m(xt).numpy();testflow=np.mean(np.linalg.norm(pred-ds['F'][3357:],axis=1)/np.linalg.norm(ds['F'][3357:],axis=1))
   for pi in range(4):
    q=torch.tensor(Q[pi]);f=m(q).detach().numpy();jac=[]
    for j in range(6):
     v=torch.zeros_like(q);v[:,j]=1;_,jv=torch.func.jvp(m,(q,),(v,));jac.append(jv.detach().numpy())
    jl=np.stack(jac,axis=2);C=attribution(np.array([T.grad(ff) for ff in f]),jl,ts,Q[pi,-1]-Q[pi,0]);Cs.append(C);fl.append(f);jls.append(jl);d=n.B@(C-ref[pi]);dq=Q[pi,-1]-Q[pi,0]
    cons=np.max(np.abs(f@n.B.T-np.concatenate([Q[pi],np.broadcast_to(qbg,(41,len(qbg)))],axis=1)@n.od_M(ods+[o for o,d0 in old.select_ods(n)[1].items() if d0>0 and o not in ods]).T));dt=T.value(f[-1])-T.value(f[0]);compl=abs(C.sum()-dt)/max(abs(dt),1e-12);checks.append({'method':method,'seed':seed,'path':pi,'conservation_abs':float(cons),'completeness_relative':float(compl)})
    for j,(s,t) in enumerate(ods):
     if abs(dq[j])<1e-10:continue
     inter=[v for v in range(24) if v not in (s,t)];K=.5*(abs(d[t,j]-d[s,j])+np.abs(d[inter,j]).sum());dc=n.B@ref[pi,:,j];scale=.5*(abs(dc[t]-dc[s])+np.abs(dc[inter]).sum());cj=ref[pi,:,j].sum();inc=np.linalg.norm(f[-1]-f[0]-(F[pi,-1]-F[pi,0]))/np.linalg.norm(F[pi,-1]-F[pi,0]);allrows.append({'method':method,'seed':seed,'path':pi,'query':f'{s+1}->{t+1}','flow_test_pct':float(testflow*100),'increment_pct':float(inc*100),'column_pct':float(100*np.linalg.norm(C[:,j]-ref[pi,:,j])/np.linalg.norm(ref[pi,:,j])),'system_pct':float(100*abs(C[:,j].sum()-cj)/abs(cj)),'boundary_abs':float(K),'boundary_pct':float(100*K/scale),'reference_boundary_scale':float(scale)})
   outarr[f'C_{method}_{seed}']=Cs;outarr[f'F_{method}_{seed}']=fl;outarr[f'J_{method}_{seed}']=jls
   print('eval',method,seed,flush=True)
 np.savez_compressed(O/'evaluation.npz',C_reference=ref,**outarr)
 summary={}
 for method in ['original','flow','response','structure']:
  rows=[r for r in allrows if r['method']==method];summary[method]={k:float(np.median([r[k] for r in rows])) for k in ['column_pct','system_pct','boundary_abs','boundary_pct']}
  summary[method]['flow_test_pct']=float(np.median(list({r['seed']:r['flow_test_pct'] for r in rows}.values())))
  summary[method]['increment_pct']=float(np.median(list({(r['seed'],r['path']):r['increment_pct'] for r in rows}.values())))
 (O/'results.json').write_text(json.dumps({'rows':allrows,'summary':summary,'checks':checks,'evaluation_seconds':time.time()-st},indent=2));print(summary)
 lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{Equal-update continuation from five flow-trained checkpoints on four independent demand paths. Entries are medians over 60 active query--path--seed combinations, except flow (five seeds, held-out split) and increment error (20 path--seed pairs). Boundary error uses all separating cuts and is normalized by the corresponding maximum absolute reference account.}',r'\label{tab:structured-learning}',r'\begin{tabular}{lrrrrrr}\toprule',r'Method & Flow (\%) & Increment (\%) & Column (\%) & System (\%) & Cut abs. & Cut (\%)\\\midrule']
 for meth,label in [('original','Original'),('flow','Flow continuation'),('response','Flow + response'),('structure','Flow + response + boundary')]:
  v=summary[meth];lines.append(label+' & '+' & '.join(f'{v[k]:.2f}' for k in ['flow_test_pct','increment_pct','column_pct','system_pct','boundary_abs','boundary_pct'])+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\end{table}']
 TABLE_OUTPUT.write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['data','train','eval']);a.add_argument('--seed',type=int,default=0);a.add_argument('--method',default='flow');v=a.parse_args()
 if v.phase=='data':data()
 elif v.phase=='train':train(v.seed,v.method)
 else:evaluate()
