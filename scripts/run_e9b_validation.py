"""E9b: learned-surrogate and assignment-map validation of the responsibility
law on the Sioux Falls topology, plus the representation-space comparison.

Extends E9 (analytic conservative field) along three axes:

  (a) Trained M3 surrogate.  The E5 grid M3|norm|raw best-validation
      checkpoints (results/e5_ckpt/M3|norm|raw|{seed}.pt, seeds 0-4, trained
      on the Sioux Falls top-8 logit-SUE dataset -- the same checkpoint
      source as run_v5_metrics.py) run through the full E9 outcome suite on the
      same endpoints q0=0.2 q_base, q1=1.6 q_base -- the law verdicts on a
      *learned* exactly conservative surrogate rather than an analytic
      construction.

  (b) Logit-SUE assignment map.  The fixed-point map F(q) itself (theta=1.0)
      is demand-conserving (BF(q)=Mq); its central finite-difference Jacobian
      along the path provides the model-specific representation for Level-I
      pointwise diagnostics and the endpoint IG quantities (P1/P3), with an
      FD step-halving convergence check at the midpoint.

  (c) Representation spaces.  The cycle-space residual restricted to
        Z_full   = ker B (all signed circulations; the paper's quantifier),
        Z_active = circulations supported on the strictly positive active
                   support at the realized midpoint flow (basis FROZEN from
                   the midpoint realization and reused along the whole
                   attribution path), and, since v7, the PATHWISE variant
                   that rebuilds Z_act(t) from the realized flow F(t) at
                   every path node (reported under "pathwise_active", with
                   min/median/max active-space dimension along the path),
        T_swap   = frozen-path route-swap tangent space
                   span{Delta(e_p - e_p') : same OD},
      for every E9 outcome and every field (analytic / M3 / SUE), plus a
      per-OD route-swap same-OD residual; and realized-flow statistics
      (min flow, negative count, max F/c, active-support size) for all three
      fields.

Output: experiments/results/e9b_validation.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import load_sioux_falls, sioux_falls_top8_od, load_sioux_falls_trips  # noqa: E402
from sue import PathStructure, solve_sue  # noqa: E402
from models import build_model  # noqa: E402
from attribution import (EndpointCalculator, LinearT, path_points,  # noqa: E402
                         endpoint_scales)
from run_e9_transport import (_sioux_setup, LinkFlowT, BeckmannT, TSTTT,  # noqa: E402
                              EmissionsT)
from run_e6_pointwise import chain_summary, ZERO_TOL  # noqa: E402
from run_e1 import analytic_M3_field  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RESULTS, "e9b_validation.json")

SUE_THETA = 1.0
SUE_TOL = 1e-7          # fixed-point residual on path flows (veh/h)
SUE_NODES = 11          # path grid for the SUE-map diagnostics
FD_HREL = 5e-3          # FD step as a fraction of q_base[j]
TAU_REL = 1e-6          # active-support threshold, relative to max flow


def _outcomes(net, ps):
    # v7: corridor = LONGEST frozen path of OD 0 (paper OD (10,16)), 6 arcs
    # (the first frozen path is the direct single arc and would duplicate
    # the single-link control)
    i_corr = max(ps.od_path_idx[0],
                 key=lambda i: (len(ps.paths[i]), -ps.free_flow_cost[i]))
    corridor = np.zeros(net.m)
    corridor[ps.paths[i_corr]] = 1.0
    # v17 active-link control, mirroring run_e9_transport: middle arc of
    # the first frozen path of OD 0 -- frozen-path-active, so the M3 and
    # SUE fields have nonzero flow and nonzero Jacobian row on it (the
    # non-hollow counterpart of the inactive 1->2 single link)
    p0 = ps.paths[ps.od_path_idx[0][0]]
    e_act = p0[len(p0) // 2]
    return {
        "link_flow": LinkFlowT(0, net.m),
        "link_flow_active": LinkFlowT(e_act, net.m),
        "beckmann": BeckmannT(net.free_flow_time, net.capacity, net.m),
        "tstt": TSTTT(net.free_flow_time, net.capacity, net.m),
        "emissions": EmissionsT(net.free_flow_time, net.capacity, net.m),
        "corridor_fixed_weights": LinearT(corridor),
        "uniform_aggregate_CONTROL": LinearT(np.ones(net.m)),
    }


# ----------------------------------------------------------------------------
# (a) trained M3 surrogate through the E9 outcome suite
# ----------------------------------------------------------------------------
def e9b_m3(net, ods, q_base, ps, classes, ec, Ts):
    # v6: unified protocol -- the E5 grid M3|norm|raw FREELY SELECTED BEST
    # checkpoints (the ones the paper's E9b table reports), replacing the
    # E4 validation-selected checkpoints used before, so that this file and
    # run_v5_metrics.py share one checkpoint source.
    ckpt_dir = os.path.join(RESULTS, "e5_ckpt")
    per_seed = {}
    for seed in range(5):
        path = os.path.join(ckpt_dir, f"M3|norm|raw|{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        full = torch.load(path, weights_only=False, map_location="cpu")
        # mu_q/sd_q are registered buffers, hence part of the state dict --
        # dummy values in the constructor are overwritten by load_state_dict.
        norm = {"mu_q": np.zeros(ps.K), "sd_q": np.ones(ps.K)}
        model = build_model("M3", ps, seed, norm=norm)
        model.load_state_dict(full["best"])
        model.eval()
        row = {"_meta": {"best_step": int(full["best_step"]),
                         "best_val_err": float(full["best_val_err"])}}
        for name, T in Ts.items():
            s = chain_summary(ec, model, T, 0.2 * q_base, 1.6 * q_base, classes)
            s["law_verdict"] = bool(s["sup_r_class"] < ZERO_TOL)
            row[name] = s
        per_seed[str(seed)] = row
        print(f"  E9b M3 seed {seed}: "
              f"tstt eta={row['tstt']['sup_r_class']:.3e} "
              f"r_od={row['tstt']['sup_r_od']:.3e} "
              f"P3={row['tstt']['endpoint_p3_median']:.3e}", flush=True)
    # seed aggregation of the headline scalars
    agg = {}
    for name in Ts:
        keys = ["sup_r_class", "sup_r_cycle", "sup_r_od",
                "endpoint_p3_median", "endpoint_p3_abs_median",
                "beta_abs_median", "E_od_end_median"]
        agg[name] = {}
        for k in keys:
            vals = np.array([per_seed[s][name][k] for s in per_seed])
            vals = vals[np.isfinite(vals)]
            agg[name][k] = {"median": float(np.median(vals)),
                            "min": float(vals.min()) if vals.size else None,
                            "max": float(vals.max()) if vals.size else None}
        verdicts = [per_seed[s][name]["law_verdict"] for s in per_seed]
        agg[name]["law_verdict_all_seeds"] = bool(all(verdicts))
    return {"per_seed": per_seed, "seed_aggregates": agg}


# ----------------------------------------------------------------------------
# (b) logit-SUE assignment map: FD Jacobian diagnostics
# ----------------------------------------------------------------------------
class SUEMap:
    """Demand-conserving assignment map F(q) = Delta h*(q) at SUE tolerance.

    solve_sue warm starts are chained; per-solve fixed-point residuals are
    recorded so the conservation quality of the map itself is reportable."""

    def __init__(self, ps, theta, tol):
        self.ps = ps
        self.theta = theta
        self.tol = tol
        self.max_res = 0.0
        self.n_solves = 0
        self._warm = None

    def F(self, q, warm=None):
        h, res, _ = solve_sue(self.ps, q, self.theta, tol=self.tol,
                              h0=self._warm if warm is None else warm)
        self.max_res = max(self.max_res, float(res))
        self.n_solves += 1
        return self.ps.Delta @ h, h

    def reset_warm(self, h):
        self._warm = h


def _p_mu(mu):
    return np.outer(mu, mu) / (mu @ mu)


def e9b_sue(net, ods, q_base, ps, classes, ec, Ts):
    K = len(ods)
    B, M, N = ec.B, ec.M, ec.N
    Mnorm = np.sqrt((M * M).sum(axis=0))
    BpM = ec.Bpinv @ M                       # (n, K) min-norm witnesses
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    dQ = q1 - q0
    ts = np.linspace(0.0, 1.0, SUE_NODES)
    smap = SUEMap(ps, SUE_THETA, SUE_TOL)

    # --- path nodes: F_t and central-difference Jacobian columns ---
    Fts, Js = [], []
    for ti, t in enumerate(ts):
        qc = q0 + t * dQ
        Fc, hc = smap.F(qc)
        smap.reset_warm(hc)
        Fts.append(Fc)
        J = np.zeros((net.m, K))
        for j in range(K):
            hj = FD_HREL * q_base[j]
            Fp, _ = smap.F(qc + hj * np.eye(K)[j], warm=hc)
            Fm, _ = smap.F(qc - hj * np.eye(K)[j], warm=hc)
            J[:, j] = (Fp - Fm) / (2.0 * hj)
        Js.append(J)
        print(f"  E9b SUE node {ti + 1}/{SUE_NODES} (t={t:.2f}) "
              f"solves={smap.n_solves} max_res={smap.max_res:.2e}", flush=True)
    Fts = np.stack(Fts)                       # (T, m)
    Js = np.stack(Js)                         # (T, m, K)

    # --- FD step-halving check at the midpoint ---
    tmid = ts[(len(ts) - 1) // 2]
    qc = q0 + tmid * dQ
    Fc, hc = smap.F(qc)
    J2 = np.zeros((net.m, K))
    for j in range(K):
        hj = 0.5 * FD_HREL * q_base[j]
        Fp, _ = smap.F(qc + hj * np.eye(K)[j], warm=hc)
        Fm, _ = smap.F(qc - hj * np.eye(K)[j], warm=hc)
        J2[:, j] = (Fp - Fm) / (2.0 * hj)
    mid = (len(ts) - 1) // 2
    fd_check = {
        "hrel": FD_HREL, "hrel_check": 0.5 * FD_HREL,
        "jacobian_max_rel_diff": float(
            np.abs(J2 - Js[mid]).max() / np.abs(Js[mid]).max()),
        "jacobian_rel_l2": float(
            np.linalg.norm(J2 - Js[mid]) / np.linalg.norm(Js[mid])),
    }

    # --- conservation quality of the map itself ---
    cons = [np.linalg.norm(B @ Fts[i] - M @ (q0 + ts[i] * dQ))
            for i in range(len(ts))]
    conserv = {"max_abs_residual": float(max(cons)),
               "scale_reference": float(np.linalg.norm(M @ q1))}

    # --- per-outcome diagnostics ---
    out = {"theta": SUE_THETA, "sue_tol": SUE_TOL, "n_nodes": SUE_NODES,
           "fd_check": fd_check, "map_conservation": conserv,
           "solver_max_residual": float(smap.max_res),
           "n_solves": int(smap.n_solves), "outcomes": {}}
    _swap_per_od, swap_merged = route_swap_basis(ps)
    for name, T in Ts.items():
        gT = np.stack([T.grad(F) for F in Fts])          # (T, m)
        # v7: pathwise active-support diagnostics on this block's own path
        # grid (Z_act(t) rebuilt at every node from the realized SUE flow),
        # plus spec-norm restricted residuals on the same grid so the SUE
        # column of the paper's restricted-space table is a path supremum
        # computed exactly like the analytic/M3 columns (v6 used the single
        # realized midpoint point here)
        pathwise_act = pathwise_active_residuals(B, Fts, gT)
        restricted = restricted_residuals(B, M, gT, {
            "Z_full_signed_cycle_space": N,
            "T_route_swap_space": swap_merged,
        })
        # class dispersion (single 76-edge class on Sioux Falls)
        eta = [float(np.abs(g - g.mean()).max()) for g in gT]
        # cycle residual on the full signed cycle space
        r_cycle = [float(np.linalg.norm(B @ (g[:, None] * N), "fro")
                         / max(np.linalg.norm(g), 1e-300)) for g in gT]
        # Level-I model-specific directional residual: J_fe_j as the
        # representation (this is the SUE map's own sensitivity field)
        r_od1 = np.zeros((len(ts), K))
        for i, g in enumerate(gT):
            for j in range(K):
                mu = M[:, j]
                v = B @ (g * Js[i][:, j])
                r_od1[i, j] = np.linalg.norm(v - _p_mu(mu) @ v) / Mnorm[j]
        # Level-II same-OD residual (min-norm witness) for reference
        r_od2 = np.zeros((len(ts), K))
        for i, g in enumerate(gT):
            LtM = B @ (g[:, None] * BpM)
            for j in range(K):
                mu = M[:, j]
                r_od2[i, j] = np.linalg.norm(
                    LtM[:, j] - _p_mu(mu) @ LtM[:, j]) / Mnorm[j]
        # endpoint IG quantities from the FD Jacobians (trapezoid); v18: the
        # scale family comes from the shared attribution.endpoint_scales so
        # beta/beta_end/E_od_end/P3 match E6/E9/E12 definitions exactly.
        G = dQ[None, :] * np.trapezoid(Js, ts, axis=0)       # (m, K)
        es = endpoint_scales(B, M, gT, Js, ts, dQ)
        beta = es["beta"]
        p1 = []
        for j in range(K):
            denom = abs(dQ[j]) * Mnorm[j]
            p1.append(np.linalg.norm(B @ G[:, j] - M[:, j] * dQ[j]) / denom
                      if denom > 0 else np.nan)
        p3 = list(es["p3"])
        p3_abs = [float(x) for x in es["p3_abs"]]
        out["outcomes"][name] = {
            "sup_eta": float(max(eta)),
            "sup_r_cycle": float(max(r_cycle)),
            "pathwise_active_support": pathwise_act,
            "restricted_sup_spec": restricted,
            "sup_r_od_levelI_model": float(r_od1.max()),
            "sup_r_od_levelI_model_per_od": [float(x) for x in r_od1.max(axis=0)],
            "sup_r_od_minnorm_witness": float(r_od2.max()),
            "flow_p1_median": float(np.nanmedian(p1)),
            "flow_p1_per_od": [float(x) for x in p1],
            "endpoint_p3_median": float(np.nanmedian(p3)),
            "endpoint_p3_per_od": [float(x) for x in p3],
            "endpoint_p3_abs_median": float(np.median(p3_abs)),
            "beta_abs_median": float(np.median(np.abs(beta))),
            "beta": [float(b) for b in beta],
            "beta_end": [float(b) for b in es["beta_end"]],
            "E_od_end_median": float(np.median(es["E_od_end"])),
            "E_od_end_per_od": [float(x) for x in es["E_od_end"]],
            "sup_r_od_realized": float(max(es["r_od_realized_sup"])),
            "endpoint_split_deviation": es["split_deviation"],
            "law_verdict": bool(max(eta) < ZERO_TOL),
        }
        r = out["outcomes"][name]
        print(f"  E9b SUE {name:28s} eta={r['sup_eta']:.3e} "
              f"r_cyc={r['sup_r_cycle']:.3e} "
              f"r_odI={r['sup_r_od_levelI_model']:.3e} "
              f"P1={r['flow_p1_median']:.3e} P3={r['endpoint_p3_median']:.3e}",
              flush=True)
    out["F_midpoint"] = [float(x) for x in Fts[mid]]
    out["F_endpoint"] = [float(x) for x in Fts[-1]]
    return out


# ----------------------------------------------------------------------------
# (c) representation spaces and realized-flow statistics
# ----------------------------------------------------------------------------
def _orthonormal_range(A):
    """Orthonormal basis of the column space of A (m x c)."""
    if A.size == 0 or A.shape[1] == 0:
        return np.zeros((A.shape[0] if A.ndim == 2 else 0, 0))
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(s > 1e-12 * (s[0] if s.size else 1.0)))
    return u[:, :r]


def active_support_basis(B, F, tau):
    """Orthonormal basis of circulations supported on {e : F_e > tau}."""
    act = np.where(np.asarray(F) > tau)[0]
    Bsub = B[:, act]
    # full_matrices=True: vt must be (|act| x |act|) so that vt[r:] spans
    # the FULL nullspace even when |act| > n (thin SVD would silently
    # truncate it).
    u, s, vt = np.linalg.svd(Bsub, full_matrices=True)
    r = int(np.sum(s > 1e-10 * (s[0] if s.size else 1.0)))
    Nb = vt[r:].T                                     # (|act|, dim)
    Z = np.zeros((B.shape[1], Nb.shape[1]))
    Z[act, :] = Nb
    return Z, act


def active_support_quadruple(B, act):
    """(m_act, n_act, c_act, dim formula m-n+c) of the active subgraph.

    m_act = active edges; n_act = distinct nodes touched by them; c_act =
    connected components of the undirected active subgraph (union-find);
    the cyclomatic identity dim Z_act = m_act - n_act + c_act is what the
    SVD nullspace dimension must equal (checked by callers)."""
    m_act = int(len(act))
    touched = np.where((np.abs(B[:, act]) > 0).any(axis=1))[0]
    n_act = int(len(touched))
    parent = list(range(B.shape[0]))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in act:
        u, v = np.where(np.abs(B[:, e]) > 0)[0]
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    c_act = len({find(int(x)) for x in touched})
    return m_act, n_act, c_act, m_act - n_act + c_act


def pathwise_active_residuals(B, Fts, gT, tau_rel=TAU_REL):
    """v7 pathwise active-support diagnostics. At each path node t the
    active-support cycle basis Z_act(t) is REBUILT from the realized flow
    F(t) (threshold tau_rel * max_e |F_e(t)|), and the local residual
    ||B (g_t * Z_act(t))||_F / ||g_t|| is evaluated with that same node's
    basis and gradient; the reported value is sup_t. This strengthens (and
    must be reported alongside) the frozen-midpoint variant, which freezes
    Z_act(t=1/2) once and reuses it along the whole path. Also reports the
    min/median/max active-space dimension along the path: the support, and
    hence dim Z_act(t), can change along the attribution path. The
    cyclomatic identity dim = m_act - n_act + c_act is asserted at every
    node."""
    fro, spec = 0.0, 0.0
    dims, n_act_edges = [], []
    for F, g in zip(Fts, gT):
        tau = tau_rel * float(np.abs(F).max())
        Z, act = active_support_basis(B, F, tau)
        m_a, n_a, c_a, dim_f = active_support_quadruple(B, act)
        assert dim_f == Z.shape[1], (dim_f, Z.shape[1])
        dims.append(int(Z.shape[1]))
        n_act_edges.append(int(len(act)))
        if Z.shape[1] == 0:
            continue
        BZ = B @ (g[:, None] * Z)
        den = max(np.linalg.norm(g), 1e-300)
        fro = max(fro, float(np.linalg.norm(BZ, "fro") / den))
        spec = max(spec, float(np.linalg.norm(BZ, 2) / den))
    return {"sup_fro": fro, "sup_spec": spec,
            "dim_min": int(min(dims)), "dim_median": float(np.median(dims)),
            "dim_max": int(max(dims)),
            "active_edges_min": int(min(n_act_edges)),
            "active_edges_max": int(max(n_act_edges)),
            "n_nodes": len(dims)}


def route_swap_basis(ps):
    """Per-OD orthonormal bases of the frozen-path route-swap tangent space
    span{Delta(e_p - e_p0) : p in paths(j)}, and the merged space."""
    per_od = {}
    cols = []
    for j, idx in enumerate(ps.od_path_idx):
        p0 = ps.Delta[:, idx[0]]
        dj = [ps.Delta[:, p] - p0 for p in idx[1:]]
        per_od[j] = _orthonormal_range(np.stack(dj, axis=1))
        cols.extend(dj)
    merged = _orthonormal_range(np.stack(cols, axis=1))
    return per_od, merged


def restricted_residuals(B, M, gT, bases):
    """sup_t ||B D_t Z||_F / ||w(t)|| and the spectral (worst-direction)
    version, for each named orthonormal basis Z."""
    out = {}
    for label, Z in bases.items():
        if Z.shape[1] == 0:
            out[label] = {"dim": 0, "sup_fro": 0.0, "sup_spec": 0.0}
            continue
        fro, spec = 0.0, 0.0
        for g in gT:
            BZ = B @ (g[:, None] * Z)
            den = max(np.linalg.norm(g), 1e-300)
            fro = max(fro, float(np.linalg.norm(BZ, "fro") / den))
            spec = max(spec, float(np.linalg.norm(BZ, 2) / den))
        out[label] = {"dim": int(Z.shape[1]),
                      "sup_fro": fro, "sup_spec": spec}
    return out


def swap_od_residual(B, M, gT, swap_per_od):
    """Feasible-swap same-OD residual: sup over unit route-swap directions d
    of OD j of ||(I-P_{mu_j}) B D_t d|| / ||mu_j|| -- the directional law
    restricted to feasible route swaps of the frozen path set."""
    Mnorm = np.sqrt((M * M).sum(axis=0))
    per_od = {}
    for j, Tj in swap_per_od.items():
        if Tj.shape[1] == 0:
            per_od[str(j)] = 0.0
            continue
        mu = M[:, j]
        P = _p_mu(mu)
        best = 0.0
        for g in gT:
            R = (np.eye(B.shape[0]) - P) @ B @ (g[:, None] * Tj)
            best = max(best, float(np.linalg.norm(R, 2) / Mnorm[j]))
        per_od[str(j)] = best
    return {"per_od": per_od, "sup": float(max(per_od.values()))}


def flow_stats(net, F, tag):
    cap = net.capacity
    F = np.asarray(F)
    fmax = float(np.abs(F).max())
    taus = {"abs_1e-9": 1e-9, "rel_1e-6": TAU_REL * fmax, "abs_1.0_veh_h": 1.0}
    act = {k: int((F > v).sum()) for k, v in taus.items()}
    return {"field": tag, "min_flow": float(F.min()),
            "n_negative": int((F < 0).sum()),
            "max_voc": float((F / cap).max()),
            "median_voc": float(np.median(F / cap)),
            "active_support_size": act, "m": int(net.m)}


def gradient_path(model_or_F, T, q0, q1, steps):
    """w(t) = grad T(F_t) along the straight path, either from a torch model
    or from a precomputed (T+1, m) array of realized flows."""
    if isinstance(model_or_F, np.ndarray):
        Fs = model_or_F
    else:
        X, _, _ = path_points(q0, q1, steps)
        with torch.no_grad():
            Fs = model_or_F(torch.tensor(X, dtype=torch.float64)).numpy()
    return np.stack([T.grad(F) for F in Fs]), Fs


def e9b_spaces(net, ods, q_base, ps, classes, ec, Ts):
    B, M = ec.B, ec.M
    q0, q1 = 0.2 * q_base, 1.6 * q_base

    # realized midpoint flows per field (analytic, M3 seed 0 at its E5-best
    # checkpoint, SUE)
    fields = {}
    analytic = analytic_M3_field(ps, 11)
    _, Fa = gradient_path(analytic, LinearT(np.ones(net.m)), q0, q1, 200)
    fields["analytic"] = Fa[len(Fa) // 2]
    ck = torch.load(os.path.join(RESULTS, "e5_ckpt", "M3|norm|raw|0.pt"),
                    weights_only=False, map_location="cpu")
    norm = {"mu_q": np.zeros(ps.K), "sd_q": np.ones(ps.K)}
    m3 = build_model("M3", ps, 0, norm=norm)
    m3.load_state_dict(ck["best"])
    m3.eval()
    _, Fm = gradient_path(m3, LinearT(np.ones(net.m)), q0, q1, 200)
    fields["m3_seed0"] = Fm[len(Fm) // 2]

    smap = SUEMap(ps, SUE_THETA, SUE_TOL)
    Fs_mid, hm = smap.F(0.5 * (q0 + q1))
    fields["sue_theta1"] = Fs_mid
    print(f"  E9b spaces: SUE midpoint solve res={smap.max_res:.2e}",
          flush=True)

    stats = {tag: flow_stats(net, F, tag) for tag, F in fields.items()}

    # representation-space bases (active support at each field's midpoint)
    swap_per_od, swap_merged = route_swap_basis(ps)
    spaces_report = {}
    for tag, Fmid in fields.items():
        tau = TAU_REL * float(np.abs(Fmid).max())
        Zact, act = active_support_basis(B, Fmid, tau)
        m_act, n_act, c_act, dim_formula = active_support_quadruple(B, act)
        assert dim_formula == Zact.shape[1], (tag, dim_formula,
                                              Zact.shape[1])
        bases = {
            "Z_full_signed_cycle_space": ec.N,
            "Z_active_support_cycle_space": Zact,
            "T_route_swap_space": swap_merged,
        }
        row = {"active_edges": int(len(act)),
               "m_act": int(len(act)), "n_act": int(n_act),
               "c_act": int(c_act),
               "dim_formula_m_minus_n_plus_c": int(dim_formula),
               "dim_Z_act_basis": int(Zact.shape[1]),
               "spaces": {}, "swap_od_residual": {}}
        if tag != "sue_theta1":
            gTs = {}
            steps = 200
            model = analytic if tag == "analytic" else m3
            row["pathwise_active"] = {}
            for name, T in Ts.items():
                gT, FTs = gradient_path(model, T, q0, q1, steps)
                gTs[name] = gT
                row["pathwise_active"][name] = pathwise_active_residuals(
                    B, FTs, gT)
        else:
            gTs = {name: np.stack([T.grad(Fs_mid)])  # single realized point
                   for name, T in Ts.items()}
            row["note"] = ("single realized midpoint point (the SUE-map "
                           "restricted residuals are reported per outcome "
                           "in the sue_map block via its own path grid, "
                           "which also carries the pathwise active-support "
                           "diagnostics)")
        for name, gT in gTs.items():
            row["spaces"][name] = restricted_residuals(B, M, gT, bases)
            row["swap_od_residual"][name] = swap_od_residual(B, M, gT,
                                                             swap_per_od)
        spaces_report[tag] = row

    return {"flow_stats": stats,
            "spaces": spaces_report,
            "dims": {"Z_full": int(ec.N.shape[1]),
                     "T_swap_merged": int(swap_merged.shape[1]),
                     "T_swap_per_od": {str(j): int(Tj.shape[1])
                                       for j, Tj in swap_per_od.items()}}}


def m3_five_seed_spaces(net, ods, q_base, ps, classes, ec, Ts):
    """Restricted-space residuals for all five M3 seeds: supports the
    'five-seed medians agree' claim in the paper's E9b table note."""
    B, M = ec.B, ec.M
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    swap_per_od, swap_merged = route_swap_basis(ps)
    per_seed = {}
    for seed in range(5):
        ck = torch.load(os.path.join(RESULTS, "e5_ckpt",
                                     f"M3|norm|raw|{seed}.pt"),
                        weights_only=False, map_location="cpu")
        norm = {"mu_q": np.zeros(ps.K), "sd_q": np.ones(ps.K)}
        model = build_model("M3", ps, seed, norm=norm)
        model.load_state_dict(ck["best"])
        model.eval()
        _, Fm = gradient_path(model, LinearT(np.ones(net.m)), q0, q1, 200)
        Fmid = Fm[len(Fm) // 2]
        tau = TAU_REL * float(np.abs(Fmid).max())
        Zact, act = active_support_basis(B, Fmid, tau)
        m_act, n_act, c_act, dim_formula = active_support_quadruple(B, act)
        assert dim_formula == Zact.shape[1], (seed, dim_formula,
                                              Zact.shape[1])
        bases = {"Z_full_signed_cycle_space": ec.N,
                 "Z_active_support_cycle_space": Zact,
                 "T_route_swap_space": swap_merged}
        row = {"active_edges": int(len(act)), "m_act": int(m_act),
               "n_act": int(n_act), "c_act": int(c_act),
               "dim_formula_m_minus_n_plus_c": int(dim_formula),
               "dim_Z_act": int(Zact.shape[1]), "spaces": {},
               "pathwise_active": {}}
        for name, T in Ts.items():
            gT, FTs = gradient_path(model, T, q0, q1, 200)
            row["spaces"][name] = restricted_residuals(B, M, gT, bases)
            row["pathwise_active"][name] = pathwise_active_residuals(
                B, FTs, gT)
        per_seed[str(seed)] = row
        print(f"  E9b five-seed spaces: seed {seed} "
              f"(active {len(act)})", flush=True)
    med = {}
    for name in Ts:
        med[name] = {}
        for sp in next(iter(per_seed.values()))["spaces"][name]:
            vals = [per_seed[s]["spaces"][name][sp]["sup_spec"]
                    for s in per_seed]
            med[name][sp] = float(np.median(vals))
        med[name]["pathwise_active"] = float(np.median(
            [per_seed[s]["pathwise_active"][name]["sup_spec"]
             for s in per_seed]))
    return {"per_seed": per_seed, "median_sup_spec": med}


def _sanitize(o):
    """NaN/inf -> None so the exported JSON is strict (undefined fields
    become JSON null instead of the non-standard NaN literal)."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main():
    t0 = time.time()
    torch.set_num_threads(1)
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    Ts = _outcomes(net, ps)
    report = {
        "config": {
            "network": "Sioux Falls (TNTP)", "n": int(net.B.shape[0]),
            "m": int(net.m), "K": len(ods),
            "endpoints": "q0=0.2 q_base, q1=1.6 q_base",
            "grid_steps": 200, "sue_theta": SUE_THETA, "sue_tol": SUE_TOL,
            "sue_nodes": SUE_NODES, "fd_hrel": FD_HREL, "tau_rel": TAU_REL,
        },
    }
    print("[E9b] (a) trained M3 surrogate", flush=True)
    report["m3_trained"] = e9b_m3(net, ods, q_base, ps, classes, ec, Ts)
    print("[E9b] (b) logit-SUE assignment map", flush=True)
    report["sue_map"] = e9b_sue(net, ods, q_base, ps, classes, ec, Ts)
    print("[E9b] (c) representation spaces + flow stats", flush=True)
    report["spaces_and_stats"] = e9b_spaces(net, ods, q_base, ps, classes,
                                            ec, Ts)
    print("[E9b] (d) five-seed restricted-space sweep", flush=True)
    report["m3_five_seed_spaces"] = m3_five_seed_spaces(
        net, ods, q_base, ps, classes, ec, Ts)
    report["elapsed_seconds"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(_sanitize(report), fh, indent=1, allow_nan=False)
    print(f"[E9b] wrote {OUT} ({report['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
