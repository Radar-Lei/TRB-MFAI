"""v103: full-path Anaheim mechanism and 21/41 node-account refinement.
Reuse every saved state, evaluate only missing midpoints; checkpoints per state.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json,time,argparse
from pathlib import Path
import numpy as np
from run_e112_v102_anaheim import setup
from run_e83_v82_validation import EdgeSUE,attribution
ROOT=Path(__file__).resolve().parents[1]; OLD=ROOT/'experiments/results/e112_v102'; OUT=ROOT/'experiments/results/e113_v103'; OUT.mkdir(exist_ok=True)
def report(a,net,ps,q):
 ts=a['ts']; dq=.4*q[:6]; t0=net.free_flow_time
 weights={'ff':np.broadcast_to(t0,a['F'].shape),'cong':a['g']-t0}
 parts={f'{w}_{p}':attribution(v,a[J],ts,dq) for w,v in weights.items() for p,J in [('study','Js'),('bg','Jb')]}
 C=attribution(a['g'],a['J'],ts,dq); BC=net.B@C
 old=json.loads((OLD/'results.json').read_text()); rows=[]
 for j,r in enumerate(old['rows']):
  # Predeclared without accounts: lexicographic first, middle, last BFS bonds.
  regions=sorted(r['connected_bfs']['regions']); chosen=[regions[i] for i in sorted(set([0,len(regions)//2,len(regions)-1]))]
  vals=[float(-BC[U,j].sum()) for U in chosen]; diffs=[]
  for k in range(1,len(chosen)):
   component={key:float(-(net.B@v)[chosen[k],j].sum()+(net.B@v)[chosen[0],j].sum()) for key,v in parts.items()}
   diffs.append({'comparison':[0,k],'total':vals[k]-vals[0],'components':component,'closure':sum(component.values())-(vals[k]-vals[0])})
  rows.append({'query':r['query'],'regions_0based':chosen,'accounts':vals,'differences':diffs})
 result={'points':len(ts),'rows':rows,'column_decomposition_max':float(np.max(np.abs(C-sum(parts.values())))),'qbase_six':q[:6].tolist()}
 oldC=np.load(OLD/'arrays.npz')['C']; oldB=net.B@oldC
 if len(ts)==41:
  result['precision']={'node_relative_l2_per_query':(np.linalg.norm(BC-oldB,axis=0)/np.linalg.norm(BC,axis=0)).tolist(),'node_max_absolute':float(np.max(np.abs(BC-oldB))),'selected_account_max_absolute':max(abs(float(-(BC-oldB)[U,j].sum())) for j,row in enumerate(rows) for U in row['regions_0based']),'selected_account_relative_max':max(abs(float(-(BC-oldB)[U,j].sum()))/max(abs(float(-BC[U,j].sum())),1e-12) for j,row in enumerate(rows) for U in row['regions_0based'])}
 np.savez_compressed(OUT/f'columns_{len(ts)}.npz',C=C,BC=BC,**parts)
 (OUT/f'results_{len(ts)}.json').write_text(json.dumps(result,indent=2));return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--refine',action='store_true');args=ap.parse_args();start=time.perf_counter();net,ps,q=setup();print('setup',time.perf_counter()-start,flush=True);old=np.load(OLD/'arrays.npz');report(old,net,ps,q)
 if not args.refine:return
 sv=EdgeSUE(ps,study_k=6); ts=np.linspace(0,1,41); data={k:[] for k in ['F','J','Js','Jb','g']}
 for i,t in enumerate(ts):
  if i%2==0: row={k:old[k][i//2] for k in data}
  else:
   checkpoint=OUT/f'state_{i:02d}.npz'
   if checkpoint.exists():row=dict(np.load(checkpoint))
   else:
    qq=q.copy();qq[:6]=(.8+.4*t)*q[:6];f,h,p,res=sv.solve(qq,old['F'][i//2]);j,js,jb=sv.sensitivity(f,h,p,qq);g=net.free_flow_time*(1+.75*(f/net.capacity)**4);row={'F':f,'J':j,'Js':js,'Jb':jb,'g':g};np.savez_compressed(checkpoint,**row)
  for k in data:data[k].append(row[k])
  print('state',i,'seconds',time.perf_counter()-start,flush=True)
 a={k:np.array(v) for k,v in data.items()};a['ts']=ts;np.savez_compressed(OUT/'arrays_41.npz',**a);r=report(a,net,ps,q);r['additional_seconds']=time.perf_counter()-start;r['solver_max_residual_new_states']=sv.max_residual;(OUT/'results_41.json').write_text(json.dumps(r,indent=2));print('done',flush=True)
if __name__=='__main__':main()
