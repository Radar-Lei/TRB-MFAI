"""E82 (v81, external review round 52, review item 5): logit
dispersion sensitivity -- theta in {0.5, 1, 2} on the direct SUE map.

All published SUE results use theta = 1 (TNTP time units; Appendix
supp:sue).  The review asks for a small route-choice robustness
check: same demand path (q0 = 0.2 q_base -> q1 = 1.6 q_base), same
solver/FD protocol, same 11-node chain, direct SUE only, zero
background, all eight study ODs -- reporting the all-cut spread and
the dominant interior node at each theta.  No surrogate retraining.

Gates:
  (a) the theta = 1 chain reproduces the frozen e78 Part A 11-node
      per-OD rows to <1e-12 relative (byte-level replica of the
      published protocol);
  (b) solver residuals stay within tol on every chain.

Output: experiments/results/e82_theta_sensitivity.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution import endpoint_scales  # noqa: E402
from run_e9_transport import _sioux_setup, TSTTT  # noqa: E402
from run_e9b_validation import SUEMap, SUE_TOL, FD_HREL  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RESULTS, "e82_theta_sensitivity.json")
E78_JSON = os.path.join(RESULTS, "e78_sue_allod.json")

THETAS = (0.5, 1.0, 2.0)


def chain_theta(net, ods, q_base, ps, theta, nodes=11):
    K = len(ods)
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    ts = np.linspace(0.0, 1.0, nodes)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    smap = SUEMap(ps, theta, SUE_TOL)
    Fts, Js = [], []
    for ti, t in enumerate(ts):
        qc = q0 + t * (q1 - q0)
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
        print(f"  th={theta} node {ti + 1}/{nodes} solves={smap.n_solves} "
              f"max_res={smap.max_res:.2e}", flush=True)
    Fts, Js = np.stack(Fts), np.stack(Js)
    gT = np.stack([T.grad(F) for F in Fts])
    return ts, gT, Js, smap


def main():
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    B, M = ec.B, ec.M
    n = B.shape[0]
    dQ = (1.6 - 0.2) * q_base   # byte-matches e78 part_a's increment
    e78 = json.load(open(E78_JSON))

    out = {"protocol": {
        "thetas": list(THETAS),
        "chain": ("q0=0.2 q_base -> q1=1.6 q_base, 11 nodes, "
                  "theta as listed, tol=1e-7, FD_HREL=5e-3 central "
                  "differences, warm starts; zero background"),
    }, "per_theta": {}}

    for theta in THETAS:
        print(f"== theta = {theta} ==", flush=True)
        t_start = time.time()
        ts, gT, Js, smap = chain_theta(net, ods, q_base, ps, theta)
        print(f"theta={theta} chain time: {time.time() - t_start:.0f}s",
              flush=True)
        # gate (b): every solve on the chain converged within tolerance
        assert smap.max_res <= SUE_TOL, \
            f"solver tolerance exceeded at theta={theta}: {smap.max_res}"
        es = endpoint_scales(B, M, gT, Js, ts, dQ)
        BC = es["BC"]
        rows = {}
        for j, (s, t) in enumerate(ods):
            col = BC[:, j]
            interior = np.ones(n, dtype=bool)
            interior[[s, t]] = False
            spread = float(np.abs(col[interior]).sum())
            phi_s = float(-col[s])
            inter_idx = np.where(interior)[0]
            v_star = int(inter_idx[np.argmax(np.abs(col[inter_idx]))])
            rows[f"{s + 1}->{t + 1}"] = {
                "phi_s": phi_s,
                "spread_abs": spread,
                "ratio_cutspread": spread / abs(phi_s),
                "dominant_node_1idx": v_star + 1,
                "dominant_ratio_pct": 100.0 * abs(col[v_star]) / abs(phi_s),
                "dominant_share_of_spread": abs(col[v_star]) / spread,
                "beta_end": float(es["beta_end"][j]),
                "E_od_end": float(es["E_od_end"][j]),
                "delta_dir": float(es["E_od_end"][j]
                                   / np.hypot(es["beta_end"][j],
                                              es["E_od_end"][j])),
            }
        out["per_theta"][str(theta)] = {
            "per_od": rows,
            "n_solves": int(smap.n_solves),
            "max_res": float(smap.max_res),
        }

    # gate (a): theta=1 byte-replica of e78 Part A 11n
    ref = e78["sue_all_od"]["11"]["per_od"]
    for key, r in out["per_theta"]["1.0"]["per_od"].items():
        rb = ref[key]
        for k in ("ratio_cutspread", "delta_dir", "beta_end", "E_od_end"):
            assert abs(r[k] / rb[k] - 1) < 1e-12, \
                f"gate a failed ({key}/{k}): {r[k]} vs {rb[k]}"
    print("gate (a) passed: theta=1 chain reproduces e78 11n rows",
          flush=True)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"\n[E82] wrote {OUT}")

    # console digest
    print("\n== theta sensitivity (spread % / dominant node) ==")
    for key in out["per_theta"]["1.0"]["per_od"]:
        parts = [f"{th}: {100 * out['per_theta'][str(th)]['per_od'][key]['ratio_cutspread']:6.2f}%"
                 f"/n{out['per_theta'][str(th)]['per_od'][key]['dominant_node_1idx']}"
                 for th in THETAS]
        print(f"{key:>9}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
