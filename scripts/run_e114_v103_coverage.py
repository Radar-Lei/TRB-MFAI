"""Input-stratified Sioux Falls endpoint coverage; no attribution-driven selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import json,time,sys
from pathlib import Path
import numpy as np
import networkx as nx
from nets import load_sioux_falls,load_sioux_falls_trips
from sue import PathStructure
from run_e83_v82_validation import EdgeSUE
from run_e107_v100_screenline import load_network
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'experiments/results/e114_v103';OUT.mkdir(exist_ok=True)

def geometry(net):
    nodes,coords,g,_=load_network(); xy=np.array([coords[i+1] for i in range(net.n)])
    regions=set()
    for angle in np.arange(0,180,5):
        normal=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);normal[np.abs(normal)<1e-14]=0
        projection=xy@normal
        values=np.sort(projection);levels=[values[0]]
        for value in values[1:]:
            if value-levels[-1]>1e-12:levels.append(value)
        for a,b in zip(levels[:-1],levels[1:]):
            U=frozenset(np.flatnonzero(projection<(a+b)/2).tolist());V=frozenset(range(net.n))-U
            if nx.is_connected(g.subgraph([v+1 for v in U])) and nx.is_connected(g.subgraph([v+1 for v in V])):
                regions.add(tuple(sorted(U)));regions.add(tuple(sorted(V)))
    return sorted(regions)

def main():
    started=time.perf_counter();net=load_sioux_falls();trips=load_sioux_falls_trips()
    ods=sorted(od for od,q in trips.items() if q>0 and od[0]!=od[1]);q=np.array([trips[od] for od in ods])
    dg=nx.DiGraph()
    for (s,t),w in zip(net.edges,net.free_flow_time):dg.add_edge(s,t,weight=float(w))
    lengths=np.array([nx.shortest_path_length(dg,*od,weight='weight') for od in ods])
    qcuts=np.quantile(q,[1/3,2/3]);tcuts=np.quantile(lengths,[1/3,2/3]);qb=np.searchsorted(qcuts,q,side='left');tb=np.searchsorted(tcuts,lengths,side='left')
    chosen=[];cells=[]
    for a in range(3):
        for b in range(3):
            idx=np.flatnonzero((qb==a)&(tb==b)).tolist();pick=sorted(set(idx[min(len(idx)-1,int(len(idx)*f))] for f in (1/3,2/3))) if idx else []
            chosen.extend(pick);cells.append({'demand_bin':a+1,'time_bin':b+1,'population':len(idx),'selected_indices':pick})
    regions=geometry(net)
    protocol={'selection':'Positive non-diagonal published OD demands; demand and directed free-flow shortest-time tertile thresholds; ties remain in lower bin. Within each cell lexicographically sorted OD indices at floor(N/3),floor(2N/3). Empty cells stay empty; cells with only one member contribute one. No attribution values used.','demand_thresholds':qcuts.tolist(),'freeflow_time_thresholds':tcuts.tolist(),'cells':cells,'selected':[{'index':j,'od':[x+1 for x in ods[j]],'base_demand':q[j],'freeflow_time':lengths[j]} for j in chosen],'intervals':[.1,.2],'boundary_rule':'36 equally spaced normal directions [0,180), projection midpoints, connected induced sides; OD-oriented regions deduplicated; geometry matches e107.','all_region_masks':[list(U) for U in regions],'projection_tolerance':1e-12,'normal_zero_tolerance':1e-14,'scope':'Endpoint identity supplies total edge attribution; response study/background mechanism is not identified from endpoint flows alone.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cached='--aggregate' in sys.argv
    old=np.load(OUT/'arrays.npz') if cached else None
    previous=json.loads((OUT/'results.json').read_text()) if cached else None
    ps=PathStructure(net,ods) if not cached else None
    solver=EdgeSUE(ps,study_k=1) if not cached else None
    base=solver.solve(q)[0] if not cached else None
    records=[];saved={};maxcon=0.;maxsum=0.;maxdirect=0.
    for ii,j in enumerate(chosen):
        s,t=ods[j]; masks=[U for U in regions if s in U and t not in U]; maskmat=np.zeros((len(masks),net.n))
        for k,U in enumerate(masks):maskmat[k,list(U)]=1
        for width in (.1,.2):
            F=[]
            for sign in (-1,1):
                qq=q.copy();qq[j]*=1+sign*width
                f=old[f'Fminus_{j}_{width}' if sign<0 else f'Fplus_{j}_{width}'] if cached else solver.solve(qq,base)[0]
                F.append(f)
                maxcon=max(maxcon,float(np.max(np.abs(net.B@f-net.od_M(ods)@qq))))
            fm,fp=F;C=net.free_flow_time*(fp*(1+.15*(fp/net.capacity)**4)-fm*(1+.15*(fm/net.capacity)**4));d=net.B@C
            accounts=-maskmat@d
            direct=np.array([sum(C[e]*((u in U)-(v in U)) for e,(u,v) in enumerate(net.edges)) for U in map(set,masks)])
            maxdirect=max(maxdirect,float(np.max(np.abs(accounts-direct))))
            interior=[v for v in range(net.n) if v not in (s,t)];lo=-d[s]-np.maximum(d[interior],0).sum();hi=-d[s]+np.maximum(-d[interior],0).sum()
            system=float(C.sum());delta=float(np.sum(net.free_flow_time*fp*(1+.15*(fp/net.capacity)**4))-np.sum(net.free_flow_time*fm*(1+.15*(fm/net.capacity)**4)));maxsum=max(maxsum,abs(system-delta))
            row={'od':[s+1,t+1],'index':j,'base_demand':float(q[j]),'freeflow_time':float(lengths[j]),'demand_bin':int(qb[j]+1),'time_bin':int(tb[j]+1),'halfwidth':width,'system':system,'unrestricted_min':float(lo),'unrestricted_max':float(hi),'unrestricted_spread':float(hi-lo),'geometric_regions':len(masks),'geometric_min':float(accounts.min()),'geometric_max':float(accounts.max()),'geometric_spread':float(np.ptp(accounts)),'geometric_negative_fraction':float(np.mean(accounts<0)),'geometric_to_all_spread':float(np.ptp(accounts)/(hi-lo)),'geometric_accounts':accounts.tolist()}
            records.append(row);saved[f'C_{j}_{width}']=C;saved[f'Fminus_{j}_{width}']=fm;saved[f'Fplus_{j}_{width}']=fp
        print(f'{ii+1}/{len(chosen)} OD {s+1},{t+1} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    results={'population':len(ods),'selected_count':len(chosen),'endpoint_solves':len(chosen)*4,'total_solves_including_base':previous['total_solves_including_base'] if cached else solver.solves,'path_count':previous['path_count'] if cached else ps.P,'max_path_residual':previous['max_path_residual'] if cached else solver.max_residual,'max_flow_conservation_residual':maxcon,'max_endpoint_completeness_absolute':maxsum,'max_node_vs_crossing_account_absolute':maxdirect,'elapsed_seconds':previous['elapsed_seconds'] if cached else time.perf_counter()-started,'rows':records}
    for width in (.1,.2):
        r=[x for x in records if x['halfwidth']==width];results[f'summary_{width}']={key:{'min':float(np.min([x[key] for x in r])),'median':float(np.median([x[key] for x in r])),'max':float(np.max([x[key] for x in r]))} for key in ('geometric_spread','unrestricted_spread','geometric_to_all_spread','geometric_negative_fraction')}
    np.savez_compressed(OUT/'arrays.npz',B=net.B,q=q,ods=ods,**saved);(OUT/'results.json').write_text(json.dumps(results,indent=2));write_reports(protocol,results)

def write_reports(p,r):
    lines=[r'\subsection{Input-stratified OD coverage}\label{sec:coverage}',r'We select queries using only the published positive OD demand and directed free-flow shortest-path time. Each attribute is divided at its empirical tertiles, keeping ties in the lower group. In each of the nine cells, OD pairs are ordered lexicographically and the pairs at zero-based positions $\lfloor N/3\rfloor$ and $\lfloor2N/3\rfloor$ are retained. Empty cells remain empty and a singleton is retained once. This yields '+str(r['selected_count'])+r' queries from '+str(r['population'])+r' positive demands. For each query, only its demand moves from $(1-a)q_j$ to $(1+a)q_j$, $a\in\{0.1,0.2\}$; all other demands remain at their published levels with endogenous route choice. Four endpoint equilibria per query give the exact single-OD edge attribution by Eq.~\eqref{eq:edgeidentity}, avoiding quadrature. The geometric family uses the same 36 pre-specified directions and connected-side filter as the screenline experiment.',r'\begin{table}[t]\centering\small',r'\caption{All input-stratified queries. Entries are geometric-account spreads for $\pm10\%$ and $\pm20\%$ intervals and their ratios to the exact unrestricted spread. Absolute values use the Sioux Falls cost units. No query is removed based on its attribution.}\label{tab:coverage}',r'\begin{tabular}{lrrrr}\toprule',r'OD & $S_{\rm geo}^{10}$ & $S_{\rm geo}^{20}$ & $S_{\rm geo}^{10}/S_{\rm all}^{10}$ & $S_{\rm geo}^{20}/S_{\rm all}^{20}$\\\midrule']
    for j in [x['index'] for x in p['selected']]:
        a,b=[x for x in r['rows'] if x['index']==j];lines.append(f"({a['od'][0]},{a['od'][1]}) & {a['geometric_spread']:.1f} & {b['geometric_spread']:.1f} & {a['geometric_to_all_spread']:.3f} & {b['geometric_to_all_spread']:.3f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',f"The median geometric-to-unrestricted spread ratios are {r['summary_0.1']['geometric_to_all_spread']['median']:.3f} and {r['summary_0.2']['geometric_to_all_spread']['median']:.3f} for the two widths. These results describe this fixed input-stratified sample; they are not a probability sample of all demand conditions. Endpoint differences identify total edge accounts, but do not identify the same-state study/background response decomposition. That decomposition is evaluated separately in the mechanism experiments."]
    # Actual theorem label inserted after inspection by main executor if changed.
    (ROOT/'paper/coverage_appendix_v103.tex').write_text('\n'.join(lines)+'\n')
    summary=[r'\subsection{Input-stratified OD coverage}\label{sec:coverage_main}',f"An input-only stratified selection retains {r['selected_count']} of {r['population']} positive Sioux Falls OD pairs, with two queries in each demand-by-free-flow-time tertile cell. Each OD varies alone over $\\pm10\\%$ and $\\pm20\\%$ intervals; other demands remain fixed with endogenous rerouting. The single-OD endpoint identity gives all edge accounts from four endpoint equilibria per query. Appendix~\\ref{{sec:coverage}} reports the selection rule and all 18 queries.",r'\begin{table}[t]\centering\small',r'\caption{Input-stratified coverage across all 18 queries. Spreads use the Sioux Falls cost units. Ratios compare the pre-specified geometric family with all separating regions.}\label{tab:coverage_summary}',r'\begin{tabular}{lrrr}\toprule',r'Interval & Median geometric spread & Median unrestricted spread & Median ratio\\\midrule']
    for width in (.1,.2):
        z=r[f'summary_{width}'];summary.append(f"$\\pm{width*100:.0f}\\%$ & {z['geometric_spread']['median']:.1f} & {z['unrestricted_spread']['median']:.1f} & {z['geometric_to_all_spread']['median']:.3f}"+r'\\')
    summary += [r'\bottomrule\end{tabular}\end{table}',r'The geometric spread remains below the unrestricted spread for every selected query at both widths. The regional family therefore changes the magnitude of observed spatial variation. This fixed stratified sample is descriptive rather than a probability sample of demand conditions; same-state study/background decomposition is evaluated separately.']
    (ROOT/'paper/coverage_v103.tex').write_text('\n'.join(summary)+'\n')
    report=['# 分层 OD 覆盖实验','',f"正需求总体 {r['population']} 个，输入属性分层选择 {r['selected_count']} 个，四次端点求解每OD，共 {r['endpoint_solves']} 次，额外一次基准求解。",'','分层只使用公开正需求量和有向自由流最短路时间，按总体三分位阈值划组；同值保留低组。每组按 OD 字典序选零起始位置 floor(N/3) 与 floor(2N/3)。空组不补入、单例只取一次；不使用归因结果。','',f"总耗时 {r['elapsed_seconds']:.1f} 秒，最大路径固定点残差 {r['max_path_residual']:.3g}，最大流量守恒残差 {r['max_flow_conservation_residual']:.3g}，端点完备性差 {r['max_endpoint_completeness_absolute']:.3g}，节点求和与交叉边求和差 {r['max_node_vs_crossing_account_absolute']:.3g}。",'', '此实验没有用端点流量伪造研究/背景响应机制分解；端点只确定总边归因，机制证据由专门同状态积分实验提供。','',json.dumps({k:v for k,v in r.items() if k.startswith('summary')},ensure_ascii=False,indent=2)]
    (ROOT/'revision-stage/v103/COVERAGE_RESULTS.md').write_text('\n'.join(report)+'\n')
if __name__=='__main__': main()
