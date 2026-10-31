"""Exact scan on 36 predefined directions, grouped ties and DSU connectivity.
Benchmark same oriented, connected, OD-separating candidate cuts against direct
crossing-edge summation; both methods receive already generated columns.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from run_e107_v100_screenline import load_network,ODS_1IDX
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results/e115_v103';OUT.mkdir(exist_ok=True)
def connectivity(order,ends,adj):
 n=len(order); parent=np.arange(n); active=np.zeros(n,bool);components=0; out=[];prev=0
 def find(v):
  while parent[v]!=v: parent[v]=parent[parent[v]];v=parent[v]
  return v
 for end in ends:
  for v in order[prev:end]:
   active[v]=True;components+=1
   for u in adj[v]:
    if active[u]:
     a,b=find(v),find(u)
     if a!=b:parent[a]=b;components-=1
  out.append(components==1);prev=end
 return np.array(out)
def preprocess(xy,edges,ods):
 n=len(xy);adj=[set() for _ in range(n)]
 for u,v in edges:adj[u].add(v);adj[v].add(u)
 directions=[];candidates=0
 for deg in range(0,180,5):
  normal=np.array([np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg))]);normal[np.abs(normal)<1e-14]=0
  proj=xy@normal;order=np.argsort(proj,kind='stable');p=proj[order]
  ends=np.r_[np.flatnonzero(np.diff(p)>1e-12)+1,n];starts=np.r_[0,ends[:-1]]
  forward=connectivity(order,ends,adj);reverse=connectivity(order[::-1],n-starts[::-1],adj)[::-1]
  good=forward[:-1]&reverse[1:];rank=np.empty(n,int);rank[order]=np.arange(n)
  byquery=[]
  for s,t in ods:
   sep=(rank[s]<ends[:-1])!=(rank[t]<ends[:-1]);ix=np.flatnonzero(good&sep);e=ends[:-1][ix];sign=np.where(rank[s]<e,-1.,1.);byquery.append((e,sign));candidates+=len(ix)
  directions.append((order,byquery))
 return directions,candidates

def scan(B,C,directions,query_ids):
 b=B@C;output=[]
 for order,queries in directions:
  P=np.cumsum(b[order],axis=0)
  for col,j in enumerate(query_ids):
   ends,sign=queries[j];output.append(P[ends-1,col]*sign)
 return np.concatenate(output)
def direct(edges,C,directions,query_ids):
 n=max(edges.ravel())+1;output=[]
 for order,queries in directions:
  for col,j in enumerate(query_ids):
   ends,sign=queries[j]
   for end,sg in zip(ends,sign):
    inside=np.zeros(n,bool);inside[order[:end]]=True
    # Prefix account is outgoing minus incoming; orient with origin inside.
    coeff=inside[edges[:,0]].astype(float)-inside[edges[:,1]].astype(float)
    output.append((-sg)*np.dot(coeff,C[:,col]))
 return np.array(output)
def time_call(fn,reps):
 for _ in range(3):fn()
 times=[]
 for _ in range(5):
  start=time.perf_counter()
  for _ in range(reps):fn()
  times.append((time.perf_counter()-start)/reps)
 return float(np.median(times))
def main():
 nodes,coords,g,arcs=load_network();xy=np.array([coords[v] for v in nodes]);edges=np.array(arcs)-1;n=len(nodes);B=np.zeros((n,len(edges)))
 for e,(u,v) in enumerate(edges):B[u,e]=-1;B[v,e]=1
 B=csr_matrix(B)
 ods=np.array(ODS_1IDX)-1;start=time.perf_counter();directions,count=preprocess(xy,edges,ods);prep=time.perf_counter()-start
 C=np.load(ROOT/'experiments/results/e87_v85_newod_refine.npz')['C_41'];rows=[]
 for copies in [1,20]:
  cols=np.tile(C,(1,copies));ids=np.tile(np.arange(6),copies);a=scan(B,cols,directions,ids);b=direct(edges,cols,directions,ids);err=float(np.max(np.abs(a-b)));assert np.allclose(a,b,rtol=1e-11,atol=1e-8)
  rows.append({'columns':cols.shape[1],'accounts':len(a),'max_absolute_error':err,'scan_seconds':time_call(lambda:scan(B,cols,directions,ids),20),'direct_seconds':time_call(lambda:direct(edges,cols,directions,ids),2)})
 result={'network':'Sioux Falls','directions':36,'preprocess_seconds':prep,'preprocess_repeated_seconds':time_call(lambda:preprocess(xy,edges,ods),10),'candidate_query_accounts':count,'unique_regions_per_query':[len(set(tuple(sorted(order[:e] if sign<0 else order[e:])) for order,qs in directions for e,sign in zip(*qs[j]))) for j in range(6)],'rows':rows,'timing':'median of five batches, three warmups; scan 20 repetitions per batch, direct two. 120 columns repeat six reference columns 20 times: workload timing only.'}
 (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 tex=r'''\subsection{Exact scanning of predefined straight screenlines}\label{sec:scan}
Fix $D$ directions before evaluating attribution. For each direction, sort nodes by their coordinate projection and group equal projection values; no screenline splits such a group. A candidate position is a gap between adjacent groups. Forward and reverse union--find sweeps record whether each prefix and suffix induces a connected subgraph. This geometry and topology preprocessing is reusable across queries, demand paths and models.
For one attribution column, form $b_j=B\widetilde C_{:,j}$ and prefix sums $P_{j,k}=\sum_{i\leq k}(b_j)_{\pi(i)}$. If $s_j$ lies in the prefix and $t_j$ outside, its account is $-P_{j,k}$; for the reverse orientation it is $P_{j,k}$ because $\mathbf1^\top b_j=0$. Retain only endpoint-separating gaps with both sides connected. Scanning these values gives exact accounts and extrema over this prescribed finite screenline family. It does not optimize over every possible direction in the plane. Preprocessing takes $O(D(|V|\log|V|+|E|\alpha(|V|)))$; after preprocessing, a column takes $O(|E|+D|V|)$ using sparse incidence multiplication. Duplicate regions may be removed without changing extrema.
'''
 tex+='\nOn Sioux Falls, 36 directions at five-degree intervals give '+str(count)+' admissible query--position accounts across six queries (duplicates across directions retained for both algorithms). Direct summation recomputes outgoing minus incoming crossing-edge contributions on exactly these same oriented regions. '
 tex+=f"The largest account discrepancy was {max(r['max_absolute_error'] for r in rows):.2g} TSTT units. Reusable preprocessing took {result['preprocess_repeated_seconds']*1000:.2f} ms. "
 tex+=r'\begin{table}[t]\centering\small\caption{Same-task spatial aggregation after reference columns are available. Median batch times; reference generation is excluded. The 120-column workload repeats the six columns twenty times and measures repeated-query cost, not additional empirical evidence.}\label{tab:scan}\begin{tabular}{rrrr}\toprule Columns & Accounts & Prefix scan (ms) & Direct crossings (ms)\\\midrule'+'\n'
 for r in rows:tex+=f"{r['columns']} & {r['accounts']} & {r['scan_seconds']*1000:.2f} & {r['direct_seconds']*1000:.2f}"+r'\\'+'\n'
 tex+=r'\bottomrule\end{tabular}\end{table}'+'\n';(ROOT/'paper/spatial_scan_v103.tex').write_text(tex)
if __name__=='__main__':main()
