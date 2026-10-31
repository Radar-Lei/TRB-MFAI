"""v98：仅使用已有数组，重汇总绝对尺度与固定边界四项分解；不求解均衡。"""
from pathlib import Path
import json
import numpy as np
from nets import load_sioux_falls
from run_e83_v82_validation import attribution
from run_e86_v84_newod import select_ods
R=Path(__file__).resolve().parent/'results'
def read(n): return json.loads((R/(n+'.json')).read_text())
net=load_sioux_falls(); z=np.load(R/'e87_v85_newod_refine.npz'); idx=np.arange(0,81,2)
ods,_,_=select_ods(net); keys=[f'{s+1}->{t+1}' for s,t in ods]
gff=np.broadcast_to(net.free_flow_time,z['gT'][idx].shape)
gc=z['gT'][idx]-gff
cols={f'{a}_{b}': attribution(g,z[j][idx],z['ts'][idx],z['dQ']) for a,g in [('ff',gff),('cong',gc)] for b,j in [('study','J_study'),('bg','J_background')]}
# 检测求和或积分网格错误；失败则不得将分解数值写入论文。
err=float(np.max(np.abs(sum(cols.values())-z['C_41']))); assert err<1e-8
mech=read('e91_v87_negative_mechanism')['mechanism']; cross={}
for k in ['1->19','1->15','7->12']:
 j=keys.index(k); u=np.array(mech[k]['U_minus_nodes_1idx'])-1
 row={a:-float((net.B@c[:,j])[u].sum()) for a,c in cols.items()}
 row['total']=mech[k]['phi_U_minus']; assert abs(sum(row[a] for a in cols)-row['total'])<1e-8
 cross[k]=row
A=read('e85_v84_precision')['precision']['grids']['41']['per_od']; bc=read('e84_v83_coverage')
order=['17->10','10->17','16->10','10->16','15->10','10->15','10->11','11->10']
den={k:{'A':abs(A[k]['phi_s']),'B':abs(bc['B']['per_od'][k]['phi_s']),'C':abs(bc['C_k10_theta1']['per_od'][k]['phi_s'])} for k in order}
seed=read('e93_v89_fullbg_surrogate')['per_seed']; errors={k:float(np.median([s['per_od'][k]['E_cut_abs'] for s in seed])) for k in keys}
out={'protocol':'冻结数组重汇总；固定原 U_minus；study=六个研究OD，bg=其余522个OD；不重选边界、不重新求解、不训练','column_additivity_max_abs':err,'fixed_boundary_cross_split':cross,'origin_denominators':den,'median_absolute_connected_error':errors,'17->10_absolute_span':{'A':A['17->10']['spread_abs'],'C':bc['C_k10_theta1']['per_od']['17->10']['spread_abs']}}
(R/'e106_v98_summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
