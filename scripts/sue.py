"""Logit-SUE data generation under the frozen protocol.

Frozen choices (recorded in refine-logs/protocol_appendix_od.md):
  * Path sets: small networks -> ALL directed simple paths (cap 200 per OD);
    Sioux Falls -> Yen k-shortest by free-flow time, k=10 per OD, ties by
    lexicographic edge sequence.
  * Stochastic user equilibrium: path-size-free multinomial logit on the frozen
    path set, path utility -theta * BPR path cost, BPR t0*(1+0.15*(f/c)^4).
  * Solver: MSA (step 1/(k+1)) down to mid tolerance, then damped fixed-point
    refinement (beta halving on divergence) to the target tolerance.  Both phases
    target the same fixed point h* = D(c(h*)).  Warm starts across samples are
    allowed (solution unique by strict monotonicity; only iteration counts change).
  * Demand sampling: q_j = q_base_j * s_j, s_j ~ LogNormal(0, 0.5^2) iid,
    clipped to [0.2, 2.0]; split seed 20260820 (permutation over the full sample
    set); default n_train=4000 / n_val=857 / n_test=857 -- under a budget
    downgrade n_train may shrink (2500) while val/test stay at 857 each.
  * In-domain paired IG baseline q0 = 0.2 * q_base (all ODs simultaneously at
    the lower domain bound); q0 never enters training.
"""
from __future__ import annotations

import numpy as np

from nets import Network

SPLIT_SEED = 20260820
N_TRAIN, N_VAL, N_TEST = 4000, 857, 857
THETAS = (0.5, 1.0)
BPR_B, BPR_P = 0.15, 4.0


# ----------------------------------------------------------------------------
# Path sets
# ----------------------------------------------------------------------------
class PathStructure:
    """Frozen path set for a network + OD list."""

    def __init__(self, net: Network, ods, k_sioux: int | None = 10, cap: int = 200):
        self.net = net
        self.ods = list(ods)
        self.paths: list[list[int]] = []      # global list of edge-id lists
        self.path_od: list[int] = []          # OD index of each path
        od_paths = []
        for j, (s, t) in enumerate(self.ods):
            ps = net.all_simple_paths(s, t, cap=cap + 1)
            if len(ps) > cap:
                # too many: fall back to k-shortest by free-flow time
                ps = net.k_shortest_paths(s, t, k_sioux, net.free_flow_time)
                if len(ps) > cap:
                    ps = ps[:cap]
                note = f"kshortest{len(ps)}"
            else:
                note = f"enumerate{len(ps)}"
            od_paths.append((ps, note))
            for p in ps:
                self.paths.append(p)
                self.path_od.append(j)
        self.notes = [n for _, n in od_paths]
        self.P = len(self.paths)
        self.m = net.m
        self.K = len(self.ods)
        # link-path incidence Delta (m x P)
        Delta = np.zeros((self.m, self.P))
        for p_idx, p in enumerate(self.paths):
            for e in p:
                Delta[e, p_idx] = 1.0
        self.Delta = Delta
        # per-OD path index lists
        self.od_path_idx = [np.array([p for p in range(self.P) if self.path_od[p] == j],
                                      dtype=int) for j in range(self.K)]
        # free-flow path costs
        self.free_flow_cost = np.array([net.free_flow_time[np.array(p)].sum()
                                        for p in self.paths])
        # per-OD slices for fast softmax
        self.od_slices = [(self.od_path_idx[j].min(), self.od_path_idx[j].max() + 1)
                          for j in range(self.K)]

    def describe(self):
        return {"P": self.P, "per_od": self.notes}


# ----------------------------------------------------------------------------
# SUE fixed point
# ----------------------------------------------------------------------------
def bpr(f, t0, cap):
    return t0 * (1.0 + BPR_B * (f / cap) ** BPR_P)


def path_costs(ps: PathStructure, f):
    link_c = bpr(f, ps.net.free_flow_time, ps.net.capacity)
    return ps.Delta.T @ link_c


def logit_loading(ps: PathStructure, h, q, theta):
    """D(h): path flows from current link flows f = Delta h."""
    f = ps.Delta @ h
    c = path_costs(ps, f)
    out = np.empty_like(h)
    for j in range(ps.K):
        idx = ps.od_path_idx[j]
        u = -theta * c[idx]
        u = u - u.max()
        w = np.exp(u)
        out[idx] = q[j] * w / w.sum()
    return out


def solve_sue(ps: PathStructure, q, theta, tol=1e-8, h0=None,
              msa_iters=3000, damp_max=10000, verbose=False):
    """MSA + damped refinement to fixed-point residual max_p|h_p - D(h)_p| <= tol.

    Phase 1 MSA exits at max(1e-4, 100*tol) or stagnation (the damped phase
    converges linearly and is faster in the tail on stiff BPR regimes).
    Returns (h, residual, info dict)."""
    if h0 is None:
        h = logit_loading(ps, np.zeros(ps.P), q, theta)  # free-flow loading
    else:
        # rescale warm start to the new demands (keeps each OD's path-flow sum
        # exactly q_j; speeds up the damped phase and preserves normalization)
        h = h0.copy()
        for j in range(ps.K):
            idx = ps.od_path_idx[j]
            s = h[idx].sum()
            if s > 0:
                h[idx] *= q[j] / s
            else:
                h[idx] = q[j] / len(idx)
    d = logit_loading(ps, h, q, theta)
    res = np.max(np.abs(d - h))
    info = {"msa_iters": 0, "damp_iters": 0, "min_res": res, "beta": None}
    # Phase 1: MSA only for cold starts -- a warm start is already near the fixed
    # point and MSA's large early steps (1/2, 1/3, ...) would destroy it; damped
    # refinement alone converges linearly from warm solutions.
    # Early exit on stagnation: <1% improvement over 300 iters.
    best_res = res
    stag = 0
    skip_msa = h0 is not None
    k = 0
    for k in range(1, 0 if skip_msa else msa_iters + 1):
        if res <= max(1e-4, 100 * tol):
            break
        h = h + (d - h) / (k + 1)
        d = logit_loading(ps, h, q, theta)
        res = np.max(np.abs(d - h))
        if res < 0.99 * best_res:
            best_res, stag = res, 0
        else:
            stag += 1
            if stag >= 300:
                break
    info["msa_iters"] = min(k, msa_iters)
    info["min_res"] = res
    # Phase 2: damped refinement
    if res > tol:
        beta = 0.05
        h_best, res_best = h.copy(), res
        for it in range(damp_max):
            if res_best <= tol:
                break
            h_new = h + beta * (d - h)
            d_new = logit_loading(ps, h_new, q, theta)
            res_new = np.max(np.abs(d_new - h_new))
            if res_new < res:                       # accept
                h, d, res = h_new, d_new, res_new
                if res < res_best:
                    h_best, res_best = h.copy(), res
            else:                                   # diverging: shrink beta, restart from best
                beta *= 0.5
                h, d = h_best.copy(), logit_loading(ps, h_best, q, theta)
                res = res_best
                if beta < 1e-6:
                    break
        h, res = h_best, res_best
        info["damp_iters"] = it + 1
        info["beta"] = beta
    f = ps.Delta @ h
    return h, res, {"residual": float(res), **info}


# ----------------------------------------------------------------------------
# Dataset generation (frozen protocol)
# ----------------------------------------------------------------------------
def sample_demands(rng: np.random.Generator, q_base: np.ndarray, n: int):
    """q = q_base * s, s ~ LogNormal(0, 0.5^2) iid per (sample, OD), clipped [0.2, 2]."""
    s = rng.lognormal(mean=0.0, sigma=0.5, size=(n, len(q_base)))
    s = np.clip(s, 0.2, 2.0)
    return s * q_base[None, :]


def generate_dataset(ps: PathStructure, q_base: np.ndarray, theta: float,
                     n_total: int, seed: int, tol: float = 1e-8,
                     warm_start: bool = True, order_by_norm: bool = True):
    """Generate (q, F) samples.  Returns dict with q, F, residuals, solver stats."""
    rng = np.random.default_rng(seed)
    q = sample_demands(rng, q_base, n_total)
    order = np.argsort(np.linalg.norm(q, axis=1)) if order_by_norm else np.arange(n_total)
    F = np.zeros((n_total, ps.m))
    residuals = np.zeros(n_total)
    iters = []
    h_warm = None
    for rank, i in enumerate(order):
        h, res, info = solve_sue(ps, q[i], theta, tol=tol,
                                 h0=h_warm if warm_start else None)
        if res > tol and warm_start:
            # warm-start damped refinement failed (stiff regime): retry once
            # from a cold free-flow start with the full MSA phase
            h, res, info = solve_sue(ps, q[i], theta, tol=tol, h0=None)
        if res > tol:
            print(f"    WARNING: SUE did not reach tol at sample {i} "
                  f"(res={res:.2e} > {tol:.0e})", flush=True)
        F[i] = ps.Delta @ h
        residuals[i] = res
        iters.append(info["msa_iters"] + info.get("damp_iters", 0))
        if warm_start:
            h_warm = h
        if (rank + 1) % 500 == 0:
            print(f"    SUE {rank+1}/{n_total} theta={theta} "
                  f"max_res={residuals.max():.2e}", flush=True)
    n_fail = int(np.sum(residuals > tol))
    # conservation must hold to machine precision by construction (path flows)
    cons_err = float(np.max(np.abs(ps.net.B @ F.T - ps.net.od_M(ps.ods) @ q.T)))
    return {"q": q, "F": F, "residuals": residuals,
            "cons_err": cons_err, "theta": theta, "tol": tol,
            "n_fail": n_fail,
            "mean_iters": float(np.mean(iters)), "max_iters": int(np.max(iters))}


def split_indices(n_total: int, n_train: int = None, seed: int = SPLIT_SEED):
    """Frozen 70/15/15 permutation split with DYNAMIC boundaries -- the fixed
    boundaries (4000/857/857) only apply when n_total = 5714; a budget-downgraded
    n_train=2500 must still produce non-empty val/test sets."""
    if n_train is None:
        n_train = N_TRAIN
    assert n_total >= n_train + N_VAL + N_TEST, \
        f"n_total={n_total} too small for train={n_train}+val={N_VAL}+test={N_TEST}"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    return (perm[:n_train],
            perm[n_train:n_train + N_VAL],
            perm[n_train + N_VAL:n_train + N_VAL + N_TEST])


def in_domain_baseline(q_base: np.ndarray):
    """Frozen in-domain paired IG baseline q0 = 0.2 * q_base."""
    return 0.2 * q_base.copy()


__all__ = ["PathStructure", "solve_sue", "logit_loading", "path_costs", "bpr",
           "generate_dataset", "sample_demands", "split_indices",
           "in_domain_baseline", "SPLIT_SEED", "N_TRAIN", "N_VAL", "N_TEST",
           "THETAS"]
