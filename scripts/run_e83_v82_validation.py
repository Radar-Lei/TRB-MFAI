"""v82 targeted validation on the unchanged Sioux Falls path set.

The 76-dimensional edge Newton equation solves the SAME logit fixed point
as sue.solve_sue. Its Jacobian is the Schur complement of the path KKT
system, avoiding a dense 5280-path Hessian. All FD comparisons re-solve
that fixed point independently. Historical e79 arrays are only references.
Use --sanity for one full-background node; default executes all 41 nodes.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

# Must be set before BLAS/torch imports when launched directly.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.linalg import solve
import torch
from sue import PathStructure, logit_loading
from nets import load_sioux_falls_trips
from run_e9_transport import _sioux_setup, TSTTT
from run_e9b_validation import SUE_THETA, FD_HREL
from run_e80_connected_cuts import (build_adj_bits, load_c_enum, enum_c,
                                  mask_to_nodes, connected)
from models import build_model
from attribution import jacobians_along_path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def stats(values):
    v = np.asarray(values, dtype=float)
    return {'median': float(np.median(v)), 'max': float(np.max(v))}


def column_error(a, b):
    """Relative 2-norm over links; every node/OD is retained separately."""
    return np.linalg.norm(a-b, axis=-2)/np.linalg.norm(b, axis=-2)


class EdgeSUE:
    """F=L(F,q); S=theta sum_k Delta Cov(h_k) Delta^T.

    dF/dq_j = (I+S diag(c'))^-1 Delta p_j.
    For a subset r of ODs, dF_r/dq_j = Delta p_j 1[j in r]
    - S_r diag(c') dF/dq_j. Thus bg responses circulate exactly.
    """
    def __init__(self, ps, study_k=8):
        self.ps = ps
        self.D = ps.Delta
        self.starts = np.array([a for a,b in ps.od_slices])
        self.path_od = np.array(ps.path_od)
        self.study_k = study_k
        self.study_p = ps.od_slices[study_k-1][1]
        self.theta = SUE_THETA
        self.t0 = np.asarray(ps.net.free_flow_time)
        self.cap = np.asarray(ps.net.capacity)
        self.solves = 0
        self.max_residual = 0.

    def loading(self, F, q):
        cost = self.D.T @ (self.t0*(1+.15*(F/self.cap)**4))
        logits = -self.theta*cost
        maxima = np.maximum.reduceat(logits, self.starts)
        ex = np.exp(logits-maxima[self.path_od])
        denom = np.add.reduceat(ex, self.starts)
        p = ex/denom[self.path_od]
        h = q[self.path_od]*p
        return self.D@h, h, p

    def covariance(self, h, q, stop=None):
        if stop is None:
            D, starts, qq = self.D, self.starts, q
        else:
            D, starts, qq = self.D[:, :self.study_p], self.starts[:stop], q[:stop]
            h = h[:self.study_p]
        Dh = D*h[None,:]
        by_od = np.add.reduceat(Dh, starts, axis=1)
        return self.theta*(Dh@D.T-(by_od/qq[None,:])@by_od.T)

    def solve(self, q, warm=None):
        F = self.loading(np.zeros(self.ps.m), q)[0] if warm is None else warm.copy()
        for iteration in range(80):
            L,h,p = self.loading(F,q)
            residual = L-F
            err = np.max(np.abs(residual))
            if err < 1e-8:
                break
            S = self.covariance(h,q)
            cp = .6*self.t0*(F/self.cap)**3/self.cap
            step = solve(np.eye(self.ps.m)+S*cp[None,:], residual,
                         assume_a='gen', check_finite=False)
            alpha = 1.
            for _ in range(30):
                Ft = F+alpha*step
                Lt,_,_ = self.loading(Ft,q)
                if np.max(np.abs(Lt-Ft)) < err:
                    F=Ft
                    break
                alpha *= .5
            else:
                raise RuntimeError(f'Newton line search failed: {err}')
        else:
            raise RuntimeError(f'Newton did not converge: {err}')
        # Store actual path construction, verify the original path fixed point.
        F = self.D@h
        h_again = logit_loading(self.ps,h,q,self.theta)
        path_res = float(np.max(np.abs(h_again-h)))
        if path_res > 1e-7:
            raise RuntimeError(f'Original path fixed-point residual {path_res}')
        self.solves += 1
        self.max_residual = max(self.max_residual,path_res)
        return F,h,p,path_res

    def sensitivity(self,F,h,p,q):
        S = self.covariance(h,q)
        Ss = self.covariance(h,q,self.study_k)
        cp = .6*self.t0*(F/self.cap)**3/self.cap
        direct = np.stack([self.D[:,idx]@p[idx]
                           for idx in self.ps.od_path_idx[:self.study_k]], axis=1)
        J = solve(np.eye(self.ps.m)+S*cp[None,:],direct,
                  assume_a='gen',check_finite=False)
        Js = direct-Ss@(cp[:,None]*J)
        Jb = -(S-Ss)@(cp[:,None]*J)
        return J,Js,Jb


def attribution(g,J,ts,dQ):
    return np.trapezoid(g[:,:,None]*J,ts,axis=0)*dQ[None,:]


def cut_metrics(B,C,ods):
    BC = B@C
    out=[]
    for j,(s,t) in enumerate(ods):
        interior=np.ones(B.shape[0],dtype=bool);interior[[s,t]]=False
        phi=-BC[s,j]
        spread=np.abs(BC[interior,j]).sum()
        out.append({'phi_s':float(phi),'spread_abs':float(spread),
                    'spread_pct':float(100*spread/abs(phi))})
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sanity',action='store_true')
    args=parser.parse_args();started=time.time();torch.set_num_threads(1)
    net,ods,qbase,ps,classes,ec=_sioux_setup();B,M=ec.B,ec.M
    trips=load_sioux_falls_trips();study=set(ods)
    bgods=[od for od,d in trips.items() if d>0 and od not in study]
    psfull=PathStructure(net,list(ods)+bgods)
    # Same first paths and path order are essential for both the historical
    # comparison and the study/background decomposition.
    assert psfull.paths[:ps.P]==ps.paths
    qbg=np.array([trips[od] for od in bgods]);q0=.2*qbase;dQ=1.6*qbase-q0
    solver=EdgeSUE(psfull);T=TSTTT(net.free_flow_time,net.capacity,net.m)
    old=np.load(RESULTS/'e79_chain_arrays.npz')
    ts=np.linspace(0,1,41);ts=ts[:1] if args.sanity else ts
    arrays={};Fs=[];Hs=[];gs=[];Jall=[];Jstudy=[];Jbg=[];Jfd=[];Jhalf=[]
    max_cons=0.;max_jcons=0.;max_studycons=0.;max_bgcons=0.;warm=None
    for ti,t in enumerate(ts):
        q=np.concatenate([q0+t*dQ,qbg]);F,h,p,res=solver.solve(q,warm)
        warm=F;J,Js,Jb=solver.sensitivity(F,h,p,q)
        fds=[]
        for factor in (1.,.5):
            Jf=np.zeros_like(J)
            for j in range(8):
                step=FD_HREL*qbase[j]*factor;qp=q.copy();qm=q.copy()
                qp[j]+=step;qm[j]-=step
                Fp,*_=solver.solve(qp,F);Fm,*_=solver.solve(qm,F)
                Jf[:,j]=(Fp-Fm)/(2*step)
            fds.append(Jf)
        Fs.append(F);Hs.append(h);gs.append(T.grad(F));Jall.append(J)
        Jstudy.append(Js);Jbg.append(Jb);Jfd.append(fds[0]);Jhalf.append(fds[1])
        max_cons=max(max_cons,float(np.max(np.abs(B@F-net.od_M(psfull.ods)@q))))
        max_jcons=max(max_jcons,float(np.max(np.abs(B@J-M))))
        max_studycons=max(max_studycons,float(np.max(np.abs(B@Js-M))))
        max_bgcons=max(max_bgcons,float(np.max(np.abs(B@Jb))))
        print(f'node {ti+1}/{len(ts)} t={t:.3f} path_res={res:.2e} '
              f'solves={solver.solves} elapsed={time.time()-started:.1f}s',flush=True)
    F=np.stack(Fs);g=np.stack(gs);J=np.stack(Jall);Js=np.stack(Jstudy);Jb=np.stack(Jbg)
    fd=np.stack(Jfd);half=np.stack(Jhalf)
    old_idx=np.arange(0,len(ts),4)
    legacy_flow_rel=float(np.max(np.linalg.norm(F[old_idx]-old['F_C'][:len(old_idx)],axis=1)
                                 /np.linalg.norm(old['F_C'][:len(old_idx)],axis=1)))
    legacy_Jrel=stats(column_error(J[old_idx],old['Js_C'][:len(old_idx)]))
    out={'protocol':{'theta':SUE_THETA,'all_ods':len(psfull.ods),'study_ods':8,
         'paths':psfull.P,'nodes':len(ts),'fd_hrel':FD_HREL,
         'integration':'uniform composite trapezoid','solver':'edge Newton, same logit fixed point',
         'study_definition':'all eight queried ODs; background is remaining 520'},
         'sanity_only':args.sanity,'legacy_11n_flow_max_relative_l2':legacy_flow_rel,
         'legacy_11n_FD_vs_implicit_J':legacy_Jrel,
         'gates':{'path_fixed_point_max_abs':solver.max_residual,
         'flow_conservation_max_abs':max_cons,'total_J_conservation_max_abs':max_jcons,
         'study_J_conservation_max_abs':max_studycons,'background_J_conservation_max_abs':max_bgcons,
         'J_additivity_max_abs':float(np.max(np.abs(J-Js-Jb)))},
         'FD_node_OD_column_errors':{'h_vs_half_relative_to_half':stats(column_error(fd,half)),
          'h_vs_implicit':stats(column_error(fd,J)),
          'half_vs_implicit':stats(column_error(half,J))}}
    if legacy_flow_rel>1e-8 or max_cons>1e-6 or max_studycons>1e-8 or max_bgcons>1e-8:
        raise RuntimeError(f'Same-equilibrium/conservation gate failed: {out}')
    if args.sanity:
        out['elapsed_seconds']=time.time()-started
        (RESULTS/'e83_v82_sanity.json').write_text(json.dumps(out,indent=2))
        print(json.dumps(out,indent=2));return
    metrics={};Cs={}
    for nodes,stride in ((11,4),(21,2),(41,1)):
        idx=np.arange(0,41,stride);C=attribution(g[idx],J[idx],ts[idx],dQ)
        Cs[nodes]=C;metrics[nodes]=cut_metrics(B,C,ods)
    C=Cs[41];Cstudy=attribution(g,Js,ts,dQ);Cbg=attribution(g,Jb,ts,dQ)
    BC=B@C;BCs=B@Cstudy;BCb=B@Cbg
    out['per_od']={};adj,n=build_adj_bits(net);lib=load_c_enum();fullmask=(1<<n)-1
    for j,(s,t) in enumerate(ods):
        key=f'{s+1}->{t+1}';row={'integration':{str(k):v[j] for k,v in metrics.items()}}
        row['C_21_vs_41_relative_l2']=float(column_error(Cs[21],C)[j])
        row['C_11_vs_41_relative_l2']=float(column_error(Cs[11],C)[j])
        row['spread_21_vs_41_abs_pp']=abs(metrics[21][j]['spread_pct']-metrics[41][j]['spread_pct'])
        row['spread_11_vs_41_abs_pp']=abs(metrics[11][j]['spread_pct']-metrics[41][j]['spread_pct'])
        connected_result=enum_c(lib,adj,n,s,t,BC[:,j:j+1])
        lo,hi=float(connected_result['phi_min'][0]),float(connected_result['phi_max'][0])
        for mask,val in ((connected_result['min_mask'][0],lo),(connected_result['max_mask'][0],hi)):
            assert connected(adj,mask) and connected(adj,fullmask&~mask)
            nd=np.array(mask_to_nodes(mask))-1
            assert s in nd and t not in nd
            assert abs(-float(BC[nd,j].sum())-val)<1e-6
        span=hi-lo;unres=metrics[41][j]['spread_abs'];assert span<=unres*(1+1e-9)
        row['connected']={'range':span,'range_pct':100*span/abs(metrics[41][j]['phi_s']),
             'unrestricted_fraction':span/unres,'n_cuts':connected_result['n_sets'],
             'min_nodes':mask_to_nodes(connected_result['min_mask'][0]),
             'max_nodes':mask_to_nodes(connected_result['max_mask'][0])}
        interior=[v for v in range(n) if v not in (s,t)]
        major=sorted(interior,key=lambda v:abs(BC[v,j]),reverse=True)[:5]
        if 16 not in major:major.append(16)
        row['signed_node_decomposition']=[{'node':v+1,'total':float(BC[v,j]),
                  'study':float(BCs[v,j]),'background':float(BCb[v,j])} for v in major]
        out['per_od'][key]=row
    out['quadrature']={'C_21_vs_41_relative_l2':stats(column_error(Cs[21],C)),
           'C_11_vs_41_relative_l2':stats(column_error(Cs[11],C)),
           'spread_21_vs_41_max_abs_pp':max(v['spread_21_vs_41_abs_pp'] for v in out['per_od'].values()),
           'spread_11_vs_41_max_abs_pp':max(v['spread_11_vs_41_abs_pp'] for v in out['per_od'].values())}
    Cfd=attribution(g,fd,ts,dQ);Chalf=attribution(g,half,ts,dQ)
    mfd=cut_metrics(B,Cfd,ods);mhalf=cut_metrics(B,Chalf,ods)
    out['FD_integrated']={'C_h_vs_half_relative_l2':stats(column_error(Cfd,Chalf)),
           'C_half_vs_implicit_relative_l2':stats(column_error(Chalf,C)),
           'spread_h_vs_half_max_abs_pp':max(abs(a['spread_pct']-b['spread_pct']) for a,b in zip(mfd,mhalf)),
           'spread_half_vs_implicit_max_abs_pp':max(abs(a['spread_pct']-b['spread_pct']) for a,b in zip(mhalf,metrics[41]))}
    out['gates']['C_additivity_max_abs']=float(np.max(np.abs(C-Cstudy-Cbg)))
    out['completeness_relative']=float(abs(C.sum()-(T.value(F[-1])-T.value(F[0])))/abs(T.value(F[-1])-T.value(F[0])))
    # The learned model has eight ODs and no background: compare on its
    # corresponding zero-background target, at the same 41 demand nodes.
    zero=EdgeSUE(ps);Fzero=[];Jzero=[];warm=None
    for t in ts:
        q=q0+t*dQ;fz,hz,pz,_=zero.solve(q,warm);warm=fz
        jz,*_=zero.sensitivity(fz,hz,pz,q);Fzero.append(fz);Jzero.append(jz)
    Fzero=np.stack(Fzero);Jzero=np.stack(Jzero)
    Czero=attribution(T.grad(Fzero),Jzero,ts,dQ)
    seed_rows=[];learned_J=[];learned_C=[]
    for seed in range(5):
        ckpt=torch.load(RESULTS/'e5_ckpt'/f'M3|norm|raw|{seed}.pt',weights_only=False,map_location='cpu')
        model=build_model('M3',ps,seed,norm={'mu_q':np.zeros(8),'sd_q':np.ones(8)})
        model.load_state_dict(ckpt['best']);model.eval()
        jl,fl,tlearn,dql=jacobians_along_path(model,q0,q0+dQ,steps=40)
        assert np.allclose(ts,tlearn) and np.allclose(dQ,dql)
        cl=attribution(T.grad(fl),jl,ts,dQ)
        seed_rows.append({'seed':seed,'J_column_relative_l2':stats(column_error(jl,Jzero)),
                          'C_column_relative_l2':stats(column_error(cl,Czero))})
        learned_J.append(jl);learned_C.append(cl)
    out['surrogate_zero_background']={'per_seed':seed_rows,
          'J_column_relative_l2_all_5_seeds':stats(column_error(np.stack(learned_J),Jzero)),
          'C_column_relative_l2_all_5_seeds':stats(column_error(np.stack(learned_C),Czero)),
          'reference':'same path-set exact implicit SUE, identical 41 integration nodes',
          'zero_SUE_fixed_point_max_abs':zero.max_residual}
    arrays={'ts':ts,'F':F,'h':np.stack(Hs),'gT':g,'J':J,'J_study':Js,'J_background':Jb,
            'J_FD_h':fd,'J_FD_half':half,'C_11':Cs[11],'C_21':Cs[21],'C_41':C,
            'C_study':Cstudy,'C_background':Cbg,'BC':BC,'BC_study':BCs,'BC_background':BCb,
            'J_zero':Jzero,'F_zero':Fzero,'C_zero':Czero,
            'J_learned':np.stack(learned_J),'C_learned':np.stack(learned_C),'dQ':dQ}
    out['elapsed_seconds']=time.time()-started
    np.savez_compressed(RESULTS/'e83_v82_validation.npz',**arrays)
    (RESULTS/'e83_v82_validation.json').write_text(json.dumps(out,indent=2))
    write_report(out)
    print(f'Completed e83 in {out["elapsed_seconds"]:.1f}s',flush=True)


def write_report(o):
    q=o['quadrature'];fd=o['FD_node_OD_column_errors'];fi=o['FD_integrated']
    lines=['# v82 定向补算报告','',
      '本次保持 v81 的网络、528 个 OD、固定候选路径、需求端点与 θ=1；只补关键数值证据。',
      '76 维链路 Newton 法求解同一 logit 均衡，隐式敏感性是路径最优性系统的等价消元。',
      f'与旧 11 节点全背景流量最大相对差：{o["legacy_11n_flow_max_relative_l2"]:.3e}。',
      f'21→41 节点完整归因列相对误差，中位数 {q["C_21_vs_41_relative_l2"]["median"]:.3e}，最大 {q["C_21_vs_41_relative_l2"]["max"]:.3e}；断面范围最大变化 {q["spread_21_vs_41_max_abs_pp"]:.5f} 个百分点。',
      f'全部 41 节点、八 OD 的差分步长减半，导数列相对误差中位数 {fd["h_vs_half_relative_to_half"]["median"]:.3e}，最大 {fd["h_vs_half_relative_to_half"]["max"]:.3e}；积分后范围最大变化 {fi["spread_h_vs_half_max_abs_pp"]:.5f} 个百分点。','',
      '| OD | 无约束范围/% | 连通范围/% | 连通/无约束 |','|---|---:|---:|---:|']
    for key,v in o['per_od'].items():
        lines.append(f'| {key} | {v["integration"]["41"]["spread_pct"]:.4f} | {v["connected"]["range_pct"]:.4f} | {v["connected"]["unrestricted_fraction"]:.4f} |')
    lines+=['','连通限制明显缩小背景重路由的范围；不能沿用零背景“几乎不缩小”的结论。',
      'study 表示八个研究 OD 的合计响应，background 表示另外 520 个 OD；两者在同一完整背景均衡和同一总流量结果梯度下计算。',
      '各节点有符号归因精确相加；两个断面范围不应当作可加的百分比。',
      '主要节点分解、全部逐 OD 精度、有限差分与隐式导数对照见 JSON；完整数组见 NPZ。','']
    sur=o['surrogate_zero_background']
    lines += [f'已有五个种子的零背景代理模型，对直接 SUE 的 Jacobian 列相对误差（所有节点和 OD 汇总）中位数 {sur["J_column_relative_l2_all_5_seeds"]["median"]:.4f}，最大 {sur["J_column_relative_l2_all_5_seeds"]["max"]:.4f}；完整归因列相对误差中位数 {sur["C_column_relative_l2_all_5_seeds"]["median"]:.4f}，最大 {sur["C_column_relative_l2_all_5_seeds"]["max"]:.4f}。',
      '该对照直接检验敏感性，不将小流量预测误差等同于小导数误差。','',
      f'总计算耗时：{o["elapsed_seconds"]:.1f} 秒；未重新训练模型；本任务计划内补算均已完成。']
    (RESULTS/'e83_v82_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
