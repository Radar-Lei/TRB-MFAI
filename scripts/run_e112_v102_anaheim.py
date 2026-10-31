"""Anaheim external spatial calculation: six largest OD queries, remaining positive
published OD demands respond endogenously on ten fixed free-flow routes each.
No learning claim is made by this external reference experiment.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,re,json,time
from pathlib import Path
import numpy as np
from nets import Network
from sue import PathStructure
from run_e83_v82_validation import EdgeSUE,attribution
from run_e9_transport import TSTTT
HERE=Path(__file__).resolve().parent; OUT=HERE/'results/e112_v102';OUT.mkdir(exist_ok=True)
def setup():
 rows=[]
 for line in (HERE/'data/Anaheim/Anaheim_net.tntp').read_text().splitlines():
  a=line.strip().split()
  if a and a[0].isdigit(): rows.append(a)
 net=Network('Anaheim',[(int(a[0])-1,int(a[1])-1) for a in rows],capacity=np.array([float(a[2]) for a in rows]),free_flow_time=np.array([float(a[4]) for a in rows]))
 trips={};origin=None
 for line in (HERE/'data/Anaheim/Anaheim_trips.tntp').read_text().splitlines():
  if line.strip().startswith('Origin'):origin=int(line.split()[1])-1
  elif origin is not None:
   for d,q in re.findall(r'(\d+)\s*:\s*([\d.eE+-]+)\s*;',line):
    if float(q)>0 and int(d)-1!=origin:trips[origin,int(d)-1]=float(q)
 ods=sorted(trips,key=lambda o:(-trips[o],o));qs=np.array([trips[o] for o in ods])
 # Build fixed paths directly: avoid enumerating simple paths on the larger graph.
 ps=PathStructure.__new__(PathStructure);ps.net=net;ps.ods=ods;ps.m=net.m;ps.K=len(ods);ps.paths=[];ps.path_od=[];ps.od_slices=[];ps.od_path_idx=[]
 for j,(s,t) in enumerate(ods):
  allowed=[e for e,(u,v) in enumerate(net.edges) if (u>=38 or u==s) and (v>=38 or v==t)]
  route_net=Network('centroid_restricted',[net.edges[e] for e in allowed])
  paths=[[allowed[e] for e in path] for path in route_net.k_shortest_paths(s,t,10,net.free_flow_time[allowed])];start=len(ps.paths)
  assert paths
  for path in paths:
   nodes=[s]+[net.edges[e][1] for e in path];assert not any(v<38 for v in nodes[1:-1]),'centroid transit'
   ps.paths.append(path);ps.path_od.append(j)
  stop=len(ps.paths);ps.od_slices.append((start,stop));ps.od_path_idx.append(np.arange(start,stop))
 ps.P=len(ps.paths);ps.Delta=np.zeros((net.m,ps.P))
 for i,path in enumerate(ps.paths):ps.Delta[path,i]=1
 return net,ps,qs

def main():
 st=time.time();n,ps,q=setup();print('setup',n.n,n.m,ps.K,ps.P,time.time()-st,flush=True);sv=EdgeSUE(ps,study_k=6);T=TSTTT(n.free_flow_time,n.capacity,n.m);ts=np.linspace(0,1,21);F=[];J=[];Js=[];Jb=[];warm=None;cons=0.;q0=.8*q[:6];dq=.4*q[:6]
 for t in ts:
  qq=q.copy();qq[:6]=q0+t*dq;f,h,p,r=sv.solve(qq,warm);warm=f;j,js,jb=sv.sensitivity(f,h,p,qq);F.append(f);J.append(j);Js.append(js);Jb.append(jb);cons=max(cons,float(np.abs(n.B@f-n.od_M(ps.ods)@qq).max()));print('state',t,'elapsed',time.time()-st,flush=True)
 F=np.array(F);J=np.array(J);Js=np.array(Js);Jb=np.array(Jb);g=np.array([T.grad(f) for f in F]);C=attribution(g,J,ts,dq);Cs=attribution(g,Js,ts,dq);Cb=attribution(g,Jb,ts,dq)
 rows=[];timer=time.perf_counter()
 for k,(s,t) in enumerate(ps.ods[:6]):
  d=n.B@C[:,k];inter=[v for v in range(n.n) if v not in(s,t)];lo=-d[s]-np.maximum(d[inter],0).sum();hi=-d[s]+np.maximum(-d[inter],0).sum();K=.5*(abs(d[t]-d[s])+np.abs(d[inter]).sum());rows.append({'query':[s+1,t+1],'origin_account':float(-d[s]),'min':float(lo),'max':float(hi),'K':float(K),'system':float(C[:,k].sum()),'background_column_norm':float(np.linalg.norm(Cb[:,k])),'total_column_norm':float(np.linalg.norm(C[:,k]))})
 elapsed_extrema=time.perf_counter()-timer
 # Same-state cost-response mechanism at the middle of the path.
 mid=len(ts)//2;mechanism=[]
 for k,(s,t) in enumerate(ps.ods[:6]):
  mechanism.append({'query':[s+1,t+1],'background_response_norm':float(np.linalg.norm(Jb[mid,:,k])),'background_divergence_max':float(np.abs(n.B@Jb[mid,:,k]).max()),'background_weighted_divergence_norm':float(np.linalg.norm(n.B@(g[mid]*Jb[mid,:,k])))})
 np.savez_compressed(OUT/'arrays.npz',F=F,J=J,Js=Js,Jb=Jb,g=g,C=C,Cs=Cs,Cb=Cb,ts=ts,q=q,edges=n.edges,B=n.B)
 result={'network':{'nodes':n.n,'links':n.m,'positive_ods':ps.K,'paths':ps.P},'protocol':'Full positive published demand; six largest OD demands 0.8 to 1.2 base, all other demands fixed but routes endogenous; 10 free-flow paths per OD; theta=1; 21-point trapezoid. No surrogate trained on Anaheim.','rows':rows,'same_state_mechanism':mechanism,'solver_max_residual':sv.max_residual,'conservation_max':cons,'response_conservation_max':float(np.max(np.abs(np.einsum('ve,tej->tvj',n.B,J)-n.od_M(ps.ods[:6])))),'decomposition_max':float(np.max(np.abs(C-Cs-Cb))),'completeness_relative':float(abs(C.sum()-(T.value(F[-1])-T.value(F[0])))/abs(T.value(F[-1])-T.value(F[0]))),'total_seconds':time.time()-st,'six_extrema_seconds':elapsed_extrema}
 (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
