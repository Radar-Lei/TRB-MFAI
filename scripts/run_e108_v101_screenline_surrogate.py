"""Evaluate five frozen e93 columns on the geometry-only e107 screenlines."""
import json
from pathlib import Path
import numpy as np
from run_e107_v100_screenline import load_network, ODS_1IDX
R=Path(__file__).resolve().parent/'results'
def main():
    nodes,coords,g,arcs=load_network()
    import networkx as nx
    B=np.zeros((24,len(arcs)))
    for e,(a,b) in enumerate(arcs): B[a-1,e]=-1; B[b-1,e]=1
    ref=np.load(R/'e87_v85_newod_refine.npz')['C_41']
    learned=np.load(R/'e93_v89_fullbg_surrogate.npz')['C_L']
    BC=B@ref; BL=np.einsum('ve,kej->kvj',B,learned)
    old=json.loads((R/'e107_v100_screenline.json').read_text())
    out={}
    for j,(s,t) in enumerate(ODS_1IDX):
        seen=set()
        for deg in np.arange(0,180,5):
            normal=np.array([np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg))])
            proj={v:np.array(coords[v])@normal for v in nodes}
            p=sorted(proj.values())
            for a,b in zip(p[:-1],p[1:]):
                if b<=a: continue
                below=frozenset(v for v in nodes if proj[v]<(a+b)/2)
                for U in (below,frozenset(set(nodes)-below)):
                    if s in U and t not in U and nx.is_connected(g.subgraph(U)) and nx.is_connected(g.subgraph(set(nodes)-U)): seen.add(U)
        sets=sorted(seen,key=lambda x:tuple(sorted(x)))
        masks=np.array([[v in U for v in range(1,25)] for U in sets],dtype=float)
        ph=-masks@BC[:,j]; pl=-BL[:,:,j]@masks.T
        key=f'{s}->{t}'; prev=old['per_od'][key]
        # Detect an OD ordering or boundary-generation mismatch before reporting.
        assert len(sets)==prev['n_admissible_dedup']
        assert abs(ph.min()-prev['phi_min'])<0.05
        idx=int(ph.argmin())
        d=BL[:,:,j]-BC[:,j]
        interior=[v for v in range(24) if v not in (s-1,t-1)]
        K=.5*(abs(d[:,t-1]-d[:,s-1])+abs(d[:,interior]).sum(1))
        eps=abs(pl-ph).max(1)
        assert np.all(eps<=K+1e-7)
        out[key]={'n':len(sets),'reference_min':float(ph.min()),'reference_max':float(ph.max()),'fixed_reference_min_nodes':sorted(sets[idx]),'learned_at_fixed_min_per_seed':pl[:,idx].tolist(),'learned_min_per_seed':pl.min(1).tolist(),'learned_max_per_seed':pl.max(1).tolist(),'screenline_error_per_seed':eps.tolist(),'all_boundary_error_K_per_seed':K.tolist(),'reference_min_sign_preserved_per_seed':(np.sign(pl[:,idx])==np.sign(ph[idx])).tolist()}
    (R/'e108_v101_screenline_surrogate.json').write_text(json.dumps({'protocol':'Fixed e107 5-degree connected screenlines; e87 C_41 reference; frozen e93 seed0..4; no training or additional solves','per_od':out},indent=2))
    tex=[r'\begin{table}[pos=htbp]',r'\caption{Fixed straight screenlines: reference minimum and the five frozen flow-supervised surrogates at that same screenline (TSTT units). The last column gives the largest error over the screenline family, as an across-seed range.}',r'\label{tab:screenlinelearned}',r'\small\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrr}\toprule',r'OD & Reference minimum & Learned account: seed range & Maximum screenline error: seed range\\\midrule']
    for key,v in out.items():
        a=np.array(v['learned_at_fixed_min_per_seed']);e=np.array(v['screenline_error_per_seed'])
        tex.append('$(%s)$ & %.1f & $[%.1f,\\ %.1f]$ & $[%.1f,\\ %.1f]$ \\\\'%(key.replace('->',','),v['reference_min'],a.min(),a.max(),e.min(),e.max()))
    tex += [r'\bottomrule\end{tabular*}\end{table}']
    (R.parent.parent/'paper'/'screenline_results_v101.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
