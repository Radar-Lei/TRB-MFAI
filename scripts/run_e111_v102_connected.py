"""Fixed structural outcomes: connected versus unrestricted separating sets.
Checks all affine response directions (a particular solution + cycle basis).
This is not a TSTT experiment or a sampled-response certificate.
"""
from pathlib import Path
import json
import numpy as np
from nets import two_bridges


def main():
    net = two_bridges()
    B = net.B
    n, m = B.shape
    # Incidence columns preserve reciprocal/parallel edges without simplification.
    adj = [set() for _ in range(n)]
    for e in range(m):
        u, v = np.flatnonzero(B[:, e])
        adj[u].add(int(v)); adj[v].add(int(u))
    def connected(nodes):
        unseen = set(nodes)
        seen = {unseen.pop()}
        stack = list(seen)
        while stack:
            v = stack.pop()
            fresh = adj[v] & unseen
            unseen -= fresh
            stack.extend(fresh)
        return not unseen
    Z = net.nullspace_basis()
    records = []
    scenarios = {
        'first_query': np.array([1.]*7+[2.]*4+[3.,4.,5.]),
        'merge_new_query': np.array([1.]*11+[3.,4.,5.]),
    }
    for scenario, w in scenarios.items():
        for s,t in [(0,4),(4,8)]:
            mu = np.eye(n)[t]-np.eye(n)[s]
            x = np.linalg.pinv(B) @ mu
            interior = [v for v in range(n) if v not in (s,t)]
            masks=[]; cmasks=[]
            for bits in range(2**len(interior)):
                U={s}|{v for i,v in enumerate(interior) if bits>>i&1}
                mask = np.array([v in U for v in range(n)],float)
                masks.append(mask)
                if connected(U) and connected(set(range(n))-U):
                    cmasks.append(mask)
            rec={'scenario':scenario,'query_0based':[s,t],'weights':w.tolist()}
            for name, rows in [('all',masks),('connected',cmasks)]:
                # Every signed response is x+Z c; equality for all c is linear.
                A = -np.array(rows) @ (B*w[None,:])
                offset=A@x
                witness = np.zeros(m); witness[11:14] = [1.,-1.,1.]
                witness_accounts = A@(x+witness)
                cycle=A@Z
                rec[name]={'boundary_count':len(rows),
                    'particular_min':float(offset.min()),
                    'particular_max':float(offset.max()),
                    'witness_min':float(witness_accounts.min()),
                    'witness_max':float(witness_accounts.max()),
                    'max_cycle_coefficient':float(np.abs(cycle).max()),
                    'universal_common_account':bool(np.ptp(offset)<1e-10 and np.abs(cycle).max()<1e-10)}
            records.append(rec)
    assert [r['connected']['universal_common_account'] for r in records]==[True,False,True,True]
    assert not any(r['all']['universal_common_account'] for r in records)
    out=Path(__file__).parent/'results'/'e111_v102_connected.json'
    out.write_text(json.dumps({'purpose':'fixed structural outcome, not TSTT',
        'response_check':'particular solution and complete circulation basis',
        'records':records},indent=2)+'\n')
    print(json.dumps(records,indent=2))
if __name__=='__main__': main()
