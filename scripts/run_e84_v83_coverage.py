"""v83: frozen-background quadrature and full-background path/theta coverage.

Every truncated OD, including all background ODs, uses k=20 in the path-set
comparison. Fully enumerated ODs retain their paths. These are different
fixed-path logit models, not convergence toward an unrestricted SUE.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.linalg import solve
from sue import PathStructure
from nets import load_sioux_falls_trips
from run_e9_transport import _sioux_setup, TSTTT
from run_e83_v82_validation import EdgeSUE, attribution, cut_metrics, column_error
from run_e80_connected_cuts import build_adj_bits, load_c_enum, enum_c, connected

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'


class FrozenSUE(EdgeSUE):
    """Newton variable is TOTAL link flow; background loading stays fixed."""
    def __init__(self, ps, background):
        super().__init__(ps)
        self.background = background

    def loading(self, F, q):
        L, h, p = super().loading(F, q)
        return L+self.background, h, p

    def solve(self, q, warm=None):
        F = self.loading(self.background, q)[0] if warm is None else warm.copy()
        for _ in range(80):
            L,h,p = self.loading(F,q)
            residual = L-F
            err = np.max(np.abs(residual))
            if err < 1e-8:
                break
            S = self.covariance(h,q)
            cp = .6*self.t0*(F/self.cap)**3/self.cap
            step = solve(np.eye(self.ps.m)+S*cp[None,:],residual,
                         assume_a='gen',check_finite=False)
            alpha = 1.
            for _ in range(30):
                Ft = F+alpha*step
                if np.max(np.abs(self.loading(Ft,q)[0]-Ft)) < err:
                    F=Ft
                    break
                alpha *= .5
            else:
                raise RuntimeError(f'Frozen Newton line search failed: {err}')
        else:
            raise RuntimeError(f'Frozen Newton failed: {err}')
        F = self.background+self.D@h
        _, h_again, _ = self.loading(F,q)
        path_res = float(np.max(np.abs(h_again-h)))
        if path_res > 1e-7:
            raise RuntimeError(f'Frozen original path residual {path_res}')
        self.solves += 1
        self.max_residual=max(self.max_residual,path_res)
        return F,h,p,path_res


def path_info(ps):
    counts=[len(idx) for idx in ps.od_path_idx]
    return {'total_paths':ps.P,'min_paths_per_od':min(counts),
            'max_paths_per_od':max(counts),
            'truncated_ods':sum(n.startswith('kshortest') for n in ps.notes),
            'enumerated_ods':sum(n.startswith('enumerate') for n in ps.notes),
            'per_od_counts':counts,'per_od_rules':ps.notes}


def swap_dimension(ps):
    # Delta ker A: for each OD subtract one reference path from all others.
    pieces=[ps.Delta[:,idx[1:]]-ps.Delta[:,idx[0],None]
            for idx in ps.od_path_idx if len(idx)>1]
    singular=np.linalg.svd(np.concatenate(pieces,axis=1),compute_uv=False)
    tol=max(ps.m,sum(len(idx)-1 for idx in ps.od_path_idx))*np.finfo(float).eps*singular[0]
    return {'dimension':int(np.sum(singular>tol)), 'rank_tolerance':float(tol),
            'singular_values':singular.tolist()}


def evaluate(C,Cs,Cb,B,ods,adj,n,lib):
    BC=B@C;BCs=B@Cs;BCb=B@Cb
    metrics=cut_metrics(B,C,ods);rows={}
    for j,(s,t) in enumerate(ods):
        row=metrics[j]
        result=enum_c(lib,adj,n,s,t,BC[:,j:j+1])
        lo,hi=float(result['phi_min'][0]),float(result['phi_max'][0])
        row.update(connected_range_abs=hi-lo,
                   connected_range_pct=100*(hi-lo)/abs(row['phi_s']))
        assert hi-lo<=row['spread_abs']*(1+1e-9)
        interior=[v for v in range(n) if v not in (s,t)]
        row['node_divergence_total']=BC[:,j].tolist()
        row['node_divergence_study']=BCs[:,j].tolist()
        row['node_divergence_background']=BCb[:,j].tolist()
        row['major_nodes']=[v+1 for v in sorted(interior,key=lambda v:abs(BC[v,j]),reverse=True)[:3]]
        if s==9 and t not in (9,16):
            U=1<<9;Up=U|(1<<16);full=(1<<n)-1
            assert connected(adj,U) and connected(adj,Up)
            assert connected(adj,full^U) and connected(adj,full^Up)
            row['boundary_10_to_10_17']={
                'U':[10],'Uprime':[10,17],
                'phi_U':float(-BC[9,j]),'phi_Uprime':float(-BC[9,j]-BC[16,j]),
                'delta_total':float(-BC[16,j]),
                'delta_study':float(-BCs[16,j]),
                'delta_background':float(-BCb[16,j])}
        rows[f'{s+1}->{t+1}']=row
    return rows


def chain(solver,qs,ts,dQ,T,B,M,tag,fd_midpoint=True):
    Fs=[];gs=[];Jall=[];Jss=[];Jbs=[];warm=None
    max_jcons=0.;mid_fd_error=0.
    for i,(t,q) in enumerate(zip(ts,qs)):
        F,h,p,_=solver.solve(q,warm);warm=F
        J,Js,Jb=solver.sensitivity(F,h,p,q)
        max_jcons=max(max_jcons,float(np.max(np.abs(B@J-M))))
        assert np.max(np.abs(J-Js-Jb))<1e-9
        if fd_midpoint and i==len(ts)//2:
            fd=np.empty_like(J)
            for j in range(8):
                step=.0025*(dQ[j]/1.4);qp=q.copy();qm=q.copy()
                qp[j]+=step;qm[j]-=step
                fd[:,j]=(solver.solve(qp,F)[0]-solver.solve(qm,F)[0])/(2*step)
            mid_fd_error=float(np.max(column_error(fd,J)))
            assert mid_fd_error<1e-3, f'{tag}: implicit/FD mismatch {mid_fd_error}'
        Fs.append(F);gs.append(T.grad(F));Jall.append(J);Jss.append(Js);Jbs.append(Jb)
    assert max_jcons<1e-8, f'{tag}: sensitivity conservation {max_jcons}'
    F=np.stack(Fs);g=np.stack(gs);J=np.stack(Jall);Js=np.stack(Jss);Jb=np.stack(Jbs)
    arrays={'F':F,'gT':g,'J':J,'J_study':Js,'J_background':Jb,
            'C':attribution(g,J,ts,dQ),'C_study':attribution(g,Js,ts,dQ),
            'C_background':attribution(g,Jb,ts,dQ)}
    gate={'path_residual':solver.max_residual,'J_conservation':max_jcons,
          'midpoint_FD_halfstep_max_column_relative_l2':mid_fd_error,
          'C_additivity':float(np.max(np.abs(arrays['C']-arrays['C_study']-arrays['C_background']))),
          'completeness_relative':float(abs(arrays['C'].sum()-(T.value(F[-1])-T.value(F[0])))/abs(T.value(F[-1])-T.value(F[0]))),
          'solves':solver.solves}
    print(f'{tag}: {gate}',flush=True)
    return arrays,gate


def main():
    started=time.time();net,ods,qbase,ps,classes,ec=_sioux_setup();B,M=ec.B,ec.M
    trips=load_sioux_falls_trips();study=set(ods)
    bgods=[od for od,d in trips.items() if d>0 and od not in study]
    allods=list(ods)+bgods;qbg=np.array([trips[od] for od in bgods])
    q0=.2*qbase;dQ=1.6*qbase-q0;ts=np.linspace(0,1,41)
    qstudy=q0+ts[:,None]*dQ
    qfull=np.concatenate([qstudy,np.broadcast_to(qbg,(41,len(qbg)))],axis=1)
    ps10=PathStructure(net,allods,k_sioux=10)
    ps20=PathStructure(net,allods,k_sioux=20)
    changed=0
    for j,(note,p10,p20) in enumerate(zip(ps10.notes,ps10.od_path_idx,ps20.od_path_idx)):
        paths10=[ps10.paths[k] for k in p10];paths20=[ps20.paths[k] for k in p20]
        if note.startswith('enumerate'):
            assert paths10==paths20
        else:
            assert len(paths10)==10 and len(paths20)==20 and paths20[:10]==paths10
            changed+=1
    adj,n=build_adj_bits(net);lib=load_c_enum();T=TSTTT(net.free_flow_time,net.capacity,net.m)
    result={'protocol':{'integration_points':41,'endpoints':'q0=0.2 qbase, q1=1.6 qbase',
             'background_freeze_query_vector':q0.tolist(),'background_ods':len(bgods),
             'all_ods':len(allods),'queries':[[s+1,t+1] for s,t in ods],
             'path_comparison':'different fixed candidate-path models; no unrestricted-SUE convergence claim'},
            'paths_k10':path_info(ps10),'paths_k20':path_info(ps20),
            'expanded_truncated_ods':changed,
            'swap_space':{'A_k10':swap_dimension(ps),'C_k10':swap_dimension(ps10),
                          'C_k20':swap_dimension(ps20),'full_cycle_dimension':net.m-net.n+1}}
    saved={'ts':ts,'dQ':dQ}
    # Anchor frozen background at q0, not the published query demands.
    joint=EdgeSUE(ps10);F0,h0,_,_=joint.solve(qfull[0])
    background=ps10.Delta[:,ps.P:]@h0[ps.P:]
    assert ps10.paths[:ps.P]==ps.paths
    frozen=FrozenSUE(ps,background)
    arrays,gate=chain(frozen,qstudy,ts,dQ,T,B,M,'B_k10_theta1')
    assert np.max(np.abs(arrays['F'][0]-F0))<1e-6
    old=np.load(OUT/'e79_chain_arrays.npz')
    oldC=attribution(old['gT_B'],old['Js_B'],old['ts_B'],dQ)
    C11=attribution(arrays['gT'][::4],arrays['J'][::4],ts[::4],dQ)
    m11=cut_metrics(B,C11,ods);m41=cut_metrics(B,arrays['C'],ods)
    result['B']={'gates':gate,'per_od':evaluate(arrays['C'],arrays['C_study'],arrays['C_background'],B,ods,adj,n,lib),
        'refinement_per_od':{f'{s+1}->{t+1}':{
            'C_11_vs_41_relative_l2':float(column_error(C11,arrays['C'])[j]),
            'spread_11_vs_41_abs_pp':abs(m11[j]['spread_pct']-m41[j]['spread_pct'])}
            for j,(s,t) in enumerate(ods)},
        'old_11_vs_new_11_C_max_relative_l2':float(np.max(column_error(C11,oldC))),
        'background_vector':background.tolist()}
    for key,val in arrays.items():saved['B_'+key]=val
    for k,theta,psfull in ((10,1.,ps10),(20,1.,ps20),(10,.5,ps10),(10,2.,ps10)):
        tag=f'C_k{k}_theta{theta:g}';solver=EdgeSUE(psfull);solver.theta=theta
        arrays,gate=chain(solver,qfull,ts,dQ,T,B,M,tag)
        # Check total flow conservation with all constant background demand.
        cons=np.max(np.abs(arrays['F']@B.T-qfull@net.od_M(allods).T))
        assert cons<1e-6
        gate['flow_conservation']=float(cons)
        result[tag]={'gates':gate,'per_od':evaluate(arrays['C'],arrays['C_study'],arrays['C_background'],B,ods,adj,n,lib)}
        for key,val in arrays.items():saved[tag+'_'+key]=val
    legacy=np.load(OUT/'e83_v82_validation.npz')
    result['C_k10_theta1']['v82_C_max_relative_l2']=float(np.max(column_error(saved['C_k10_theta1_C'],legacy['C_41'])))
    np.savez_compressed(OUT/'e84_v83_coverage.npz',**saved)
    result['elapsed_seconds']=time.time()-started
    (OUT/'e84_v83_coverage.json').write_text(json.dumps(result,indent=2))
    print(f'Completed in {result["elapsed_seconds"]:.2f}s',flush=True)

if __name__=='__main__':main()
