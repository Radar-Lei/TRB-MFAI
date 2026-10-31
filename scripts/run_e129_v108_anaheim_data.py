"""E129 (v108): Anaheim learning reference data (flows, derivative centers,
test paths) generated in parallel single-thread workers.

External review of v107 (strongly recommended item): repeat the
conservative learned-assignment comparison (flow supervision vs response
supervision) on a second network.  Anaheim uses the same conservative
parameterization as Sioux Falls (per-OD softmax path shares over ten
centroid-restricted free-flow paths per OD, e112 protocol), the six
largest published OD demands as study queries, all remaining positive
demands fixed with endogenous route choice.

Data budget (documented, frozen before training):
  - 1000 flow states: q_study ~ U(0.65, 1.3) * q_base (800 train / 200
    validation), equilibrium flows only;
  - 480 derivative centers (384 train / 96 validation): equilibrium +
    sensitivity columns of the six study demands;
  - 48 test demand paths x 21 states: active-query groups k in {1,2,3,6}
    x 12 paths, mixed and same signs, generation rule identical to
    e111/e119 (fresh seed), equilibrium + sensitivity at every state.

Workers solve disjoint shards; --collect merges, runs conservation and
completeness gates, and stores the ordered nested-prefix boundary family
per study query (the same distance-based family as the Anaheim profile)
for the evaluation stage.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/e129_v108_anaheim_data'
OUT.mkdir(exist_ok=True)

N_FLOW = 1000
N_FLOW_VAL = 200
N_CENTER = 480
N_CENTER_VAL = 96
FLOW_SEED = 108001
CENTER_SEED = 108002
PATH_SEED = 108003
N_PATHS = 48
N_STATES = 21
GROUPS = (1, 2, 3, 6)


SETUP_CACHE = OUT / 'setup.pkl'


def cached_setup():
    """e112 setup takes ~7 min (10 shortest paths x 1406 ODs); cache it."""
    import pickle
    import run_e112_v102_anaheim as e112
    if SETUP_CACHE.exists():
        with SETUP_CACHE.open('rb') as fh:
            return pickle.load(fh)
    built = e112.setup()
    tmp = SETUP_CACHE.with_suffix('.tmp')
    with tmp.open('wb') as fh:
        pickle.dump(built, fh, protocol=4)
    tmp.rename(SETUP_CACHE)
    return built


def generate_demands():
    """Deterministic demand plan shared by all workers (no solving)."""
    net, ps, q = cached_setup()
    qb = q[:6]
    rng = np.random.default_rng(FLOW_SEED)
    QF = rng.uniform(.65, 1.3, (N_FLOW, 6)) * qb
    rng = np.random.default_rng(CENTER_SEED)
    QC = rng.uniform(.65, 1.3, (N_CENTER, 6)) * qb
    rng = np.random.default_rng(PATH_SEED)
    QP, dQP, groups = [], [], []
    ts = np.linspace(0, 1, N_STATES)
    for k in GROUPS:
        for r in range(N_PATHS // len(GROUPS)):
            a = rng.uniform(.65, 1.3, 6) * qb
            b = a.copy()
            active = [(r + i) % 6 for i in range(k)]
            delta = rng.uniform(.15, .3, k) * qb[active]
            if r % 2:
                delta[::2] *= -1
            b[active] += delta
            QP.append(a + ts[:, None] * (b - a))
            dQP.append(b - a)
            groups.append(k)
    np.savez_compressed(
        OUT / 'demands.npz', QF=QF, QC=QC, QP=np.array(QP), dQP=np.array(dQP),
        group=np.array(groups), ts=ts, qb=qb, q=q,
        n_flow_val=N_FLOW_VAL, n_center_val=N_CENTER_VAL)
    meta = {'nodes': int(net.n), 'links': int(net.m), 'positive_ods': int(ps.K),
            'paths': int(ps.P), 'study_ods': [[int(s) + 1, int(t) + 1] for s, t in ps.ods[:6]]}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('demands written', meta, flush=True)


def solve_shard(which, wid, nw):
    from run_e83_v82_validation import EdgeSUE
    net, ps, q = cached_setup()
    sv = EdgeSUE(ps, study_k=6)
    d = np.load(OUT / 'demands.npz')
    qbg = d['q'][6:]
    if which == 'flows':
        idx = np.array_split(np.arange(N_FLOW), nw)[wid]
        Q = d['QF'][idx]
        F, warm = [], None
        for i, qs in enumerate(Q):
            f, h, p, _ = sv.solve(np.r_[qs, qbg], warm)
            warm = f
            F.append(f)
            if i % 20 == 0:
                print(f'flow w{wid} {i}/{len(Q)}', flush=True)
        np.savez_compressed(OUT / f'flows_{wid}.npz', idx=idx, F=np.array(F))
    elif which == 'centers':
        idx = np.array_split(np.arange(N_CENTER), nw)[wid]
        Q = d['QC'][idx]
        F, J, warm = [], [], None
        for i, qs in enumerate(Q):
            f, h, p, _ = sv.solve(np.r_[qs, qbg], warm)
            warm = f
            j, _, _ = sv.sensitivity(f, h, p, np.r_[qs, qbg])
            F.append(f)
            J.append(j)
            if i % 10 == 0:
                print(f'center w{wid} {i}/{len(Q)}', flush=True)
        np.savez_compressed(OUT / f'centers_{wid}.npz', idx=idx,
                            F=np.array(F), J=np.array(J))
    elif which == 'paths':
        # whole paths per worker (warm chaining along the path)
        pids = np.array_split(np.arange(N_PATHS), nw)[wid]
        QP, dQP, group = d['QP'], d['dQP'], d['group']
        F, J, kept = [], [], []
        for pi in pids:
            fq, jq, warm = [], [], None
            for qs in QP[pi]:
                f, h, p, _ = sv.solve(np.r_[qs, qbg], warm)
                warm = f
                j, _, _ = sv.sensitivity(f, h, p, np.r_[qs, qbg])
                fq.append(f)
                jq.append(j)
            F.append(fq)
            J.append(jq)
            kept.append(pi)
            print(f'path w{wid} {pi} done', flush=True)
        np.savez_compressed(OUT / f'paths_{wid}.npz', pids=np.array(kept),
                            F=np.array(F), J=np.array(J))
    print(f'worker {which} {wid}/{nw} complete '
          f'residual={sv.max_residual:.2e} solves={sv.solves}', flush=True)


def collect():
    from run_e83_v82_validation import attribution, TSTTT
    net, ps, q = cached_setup()
    d = np.load(OUT / 'demands.npz')
    qbg = d['q'][6:]
    ts = d['ts']
    # merge shards
    Ff = [None] * N_FLOW
    for f in sorted(OUT.glob('flows_*.npz')):
        z = np.load(f)
        for row, i in zip(z['F'], z['idx']):
            Ff[int(i)] = row
    Fc = [None] * N_CENTER
    Jc = [None] * N_CENTER
    for f in sorted(OUT.glob('centers_*.npz')):
        z = np.load(f)
        for fr, jr, i in zip(z['F'], z['J'], z['idx']):
            Fc[int(i)] = fr
            Jc[int(i)] = jr
    Fp = [None] * N_PATHS
    Jp = [None] * N_PATHS
    for f in sorted(OUT.glob('paths_*.npz')):
        z = np.load(f)
        for fr, jr, i in zip(z['F'], z['J'], z['pids']):
            Fp[int(i)] = fr
            Jp[int(i)] = jr
    assert all(x is not None for x in Ff + Fc + Jc + Fp + Jp)
    QF, QC, QP, dQP = d['QF'], d['QC'], d['QP'], d['dQP']
    # conservation gate on every reference state (study + background supply)
    ods6 = ps.ods[:6]
    Ms = net.od_M(ods6)
    Mb = net.od_M(ps.ods[6:])
    def cons(F, Q):
        return max(float(np.max(np.abs(net.B @ np.atleast_2d(F[i]).T
                                       - (Ms @ np.atleast_2d(Q[i]).T
                                          + (Mb @ qbg)[:, None]))))
                   for i in range(len(Q)))
    c_flow = cons(Ff, QF)
    c_center = cons(Fc, QC)
    c_path = cons([np.asarray(x) for x in Fp], [np.asarray(x) for x in QP])
    # reference attribution columns on the 48 test paths
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    Cref = np.array([attribution(np.array([T.grad(f) for f in Fp[pi]]), Jp[pi], ts, dQP[pi])
                     for pi in range(N_PATHS)])
    dt = np.array([T.value(Fp[pi][-1]) - T.value(Fp[pi][0]) for pi in range(N_PATHS)])
    completeness = float(max(abs(Cref[pi].sum() - dt[pi]) / max(abs(dt[pi]), 1e-9)
                             for pi in range(N_PATHS)))
    # ordered nested-prefix boundary family per study query (v105 construction)
    import networkx as nx
    from scipy.sparse import csr_matrix
    from run_e115_v103_scan import connectivity
    g = nx.Graph()
    for line in (HERE / 'data/Anaheim/Anaheim_net.tntp').read_text().splitlines():
        p = line.split()
        if p and p[0].isdigit():
            u, v, w = int(p[0]) - 1, int(p[1]) - 1, float(p[4])
            g.add_edge(u, v, weight=min(w, g[u][v]['weight']) if g.has_edge(u, v) else w)
    adj = [set(g.neighbors(v)) for v in range(net.n)]
    B = csr_matrix(net.B)
    fams = []
    for s, t in ods6:
        ds = nx.single_source_dijkstra_path_length(g, s, weight='weight')
        dt2 = nx.single_source_dijkstra_path_length(g, t, weight='weight')
        progress = np.array([(ds[v] - dt2[v] + ds[t]) / (2 * ds[t]) for v in range(net.n)])
        order = np.argsort(progress, kind='stable')
        p = progress[order]
        ends = np.r_[np.flatnonzero(np.diff(p) > 1e-12) + 1, net.n]
        starts = np.r_[0, ends[:-1]]
        good = connectivity(order, ends, adj)[:-1] & connectivity(order[::-1], net.n - starts[::-1], adj)[::-1][1:]
        rank = np.argsort(order)
        good &= (rank[s] < ends[:-1]) & (rank[t] >= ends[:-1])
        e = ends[:-1][good]
        levels = ((p[e - 1] + p[e]) / 2).tolist()
        fams.append({'masks': [order[:k].tolist() for k in e], 'levels': levels})
    np.savez_compressed(
        OUT / 'reference.npz', F=np.array(Ff), FC=np.array(Fc), JC=np.array(Jc),
        Cref=Cref, dT=dt)
    (OUT / 'families.json').write_text(json.dumps(fams, indent=2))
    (OUT / 'collect.json').write_text(json.dumps({
        'conservation_max': {'flows': c_flow, 'centers': c_center, 'paths': c_path},
        'reference_completeness_relative': completeness,
        'families': [{'query': [int(s) + 1, int(t) + 1], 'regions': len(f['masks'])}
                     for f, (s, t) in zip(fams, ods6)],
    }, indent=2))
    print(json.dumps({'conservation': [c_flow, c_center, c_path],
                      'completeness': completeness,
                      'family_regions': [len(f['masks']) for f in fams]}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['demands', 'solve', 'collect'])
    ap.add_argument('--task', choices=['flows', 'centers', 'paths'])
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--nworkers', type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == 'demands':
        generate_demands()
    elif a.cmd == 'solve':
        solve_shard(a.task, a.worker, a.nworkers)
    else:
        collect()
    print(f'[{a.cmd}] {time.time() - t0:.1f}s', flush=True)
