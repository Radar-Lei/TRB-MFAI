"""E78 (v78, external review round 49): two direct-logit-SUE extensions.

Review item 1 (the round's only new experiment): the learned-vs-direct
cut-spread comparison currently rests on OD (10,16) alone.  Part A
re-runs the EXACT e74 SUE FD chains (11 path nodes = published
protocol, 41 nodes = convergence gate; theta=1, tol=1e-7, FD_HREL=5e-3
central differences, warm starts) and extracts, for the FIRST time,
the per-OD all-cuts cut-value spread (Eq. cutspread) and direction
ratio delta^dir for ALL EIGHT study ODs.  The learned side of the
comparison (five-seed median + range per OD) is read from the frozen
e73_cut_span_summary.json records -- no new training.

Review item 3.1 (optional background-demand check, one representative
result only): Part B adds a FIXED-BACKGROUND direct-SUE replication
for OD (10,16).  All non-study published demands are assigned once at
their joint logit-SUE equilibrium with the study ODs at q0, then the
background link flows are frozen and only the eight study ODs vary
(the affine balance law Bf(q)=Mq+b of Sec. 3.2 with constant b --
differentiation still gives BJ_f=M, so every attribution-balance
statement applies unchanged).  TSTT and its gradient are evaluated on
TOTAL flows (study + background); the demand increment dQ still
belongs to the study ODs alone, so completeness sum_j C_j = dT holds
with respect to the background-shifted outcome.

Gates (all must pass before any number enters the paper):
  (a) Part A (10,16) rows reproduce e74_cutspread_table.json sue_rows
      11/41 (ratio_cutspread, delta_dir, beta_end, E_od_end) to
      <1e-12 relative -- the chain is a byte-level replica;
  (b) Part A 11-node beta_end / E_od_end per-OD arrays reproduce
      e9b_validation.json sue_map.outcomes.tstt to <1e-10 relative;
  (c) Part A 41-node (10,16) delta_dir matches
      e13_sue_endpoint_per_od.json (9->15 entry) to <1e-8;
  (d) Part B: the study-only solve with the frozen-background cost
      patch reproduces the joint solution's study link flows to
      <=1e-5 absolute (same fixed point; solver tol 1e-7);
  (e) Part B conservation: B F_study(q) - M q = 0 at machine precision
      on every chain node (path-flow construction), and the patch is
      removed before Part A results are finalized (Part A runs first,
      unpatched).

Output: experiments/results/e78_sue_allod.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sue as sue_mod  # noqa: E402
from nets import (load_sioux_falls, sioux_falls_top8_od,  # noqa: E402
                  load_sioux_falls_trips)
from sue import PathStructure  # noqa: E402
from attribution import endpoint_scales  # noqa: E402
from run_e9_transport import _sioux_setup, TSTTT  # noqa: E402
from run_e9b_validation import SUEMap, SUE_THETA, SUE_TOL, FD_HREL  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RESULTS, "e78_sue_allod.json")
E73_JSON = os.path.join(RESULTS, "e73_cut_span_summary.json")
E74_JSON = os.path.join(RESULTS, "e74_cutspread_table.json")
E9B_JSON = os.path.join(RESULTS, "e9b_validation.json")
E13_JSON = os.path.join(RESULTS, "e13_sue_endpoint_per_od.json")


def spread_row(BC, j, s, t, n):
    """Eq.-(cutspread) all-cuts spread over the origin-cut value (e74 form)."""
    col = BC[:, j]
    interior = np.ones(n, dtype=bool)
    interior[[s, t]] = False
    spread = float(np.abs(col[interior]).sum())
    phi_s = float(-col[s])
    return {
        "phi_s": phi_s,
        "spread_abs": spread,
        "ratio_cutspread": spread / abs(phi_s) if phi_s != 0 else None,
    }


def sue_chain(net, ods, q_base, ps, nodes):
    """Byte-level replica of run_e74_cutspread_table.sue_chain."""
    K = len(ods)
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    ts = np.linspace(0.0, 1.0, nodes)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    smap = SUEMap(ps, SUE_THETA, SUE_TOL)
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
        print(f"  SUE {nodes}n node {ti + 1}/{nodes} solves={smap.n_solves} "
              f"max_res={smap.max_res:.2e}", flush=True)
    Fts = np.stack(Fts)
    Js = np.stack(Js)
    gT = np.stack([T.grad(F) for F in Fts])
    return ts, gT, Js, smap, (q0, q1)


def part_a(net, ods, q_base, ps, ec):
    """Eight-OD direct-SUE cut spreads on the e74 chains."""
    B, M = ec.B, ec.M
    n = B.shape[0]
    dQ = (1.6 - 0.2) * q_base
    out = {}
    for nodes in (11, 41):
        print(f"== Part A: SUE chain, {nodes} nodes ==", flush=True)
        ts, gT, Js, smap, _ = sue_chain(net, ods, q_base, ps, nodes)
        es = endpoint_scales(B, M, gT, Js, ts, dQ)
        BC = es["BC"]
        rows = {}
        for j, (s, t) in enumerate(ods):
            r = spread_row(BC, j, s, t, n)
            r["delta_dir"] = float(es["E_od_end"][j]
                                   / np.hypot(es["beta_end"][j],
                                              es["E_od_end"][j]))
            r["beta_end"] = float(es["beta_end"][j])
            r["E_od_end"] = float(es["E_od_end"][j])
            rows[f"{s + 1}->{t + 1}"] = r
        out[str(nodes)] = {
            "per_od": rows,
            "n_solves": int(smap.n_solves),
            "solver_max_residual": float(smap.max_res),
        }
    return out


def check_gates_a(sue_a):
    e74 = json.load(open(E74_JSON))["sue_rows"]
    for nodes in ("11", "41"):
        a = sue_a[nodes]["per_od"]["10->16"]
        b = e74[nodes]
        for k in ("ratio_cutspread", "delta_dir", "beta_end", "E_od_end"):
            assert abs(a[k] / b[k] - 1) < 1e-12, \
                f"gate a failed ({nodes}/{k}): {a[k]} vs {b[k]}"
    e9b = json.load(open(E9B_JSON))["sue_map"]["outcomes"]["tstt"]
    beta_arr = np.asarray(e9b["beta_end"], dtype=float)
    e_arr = np.asarray(e9b["E_od_end_per_od"], dtype=float)
    for j, key in enumerate(sue_a["11"]["per_od"]):
        a = sue_a["11"]["per_od"][key]
        assert abs(a["beta_end"] / beta_arr[j] - 1) < 1e-10, \
            f"gate b beta failed ({key}): {a['beta_end']} vs {beta_arr[j]}"
        assert abs(a["E_od_end"] / e_arr[j] - 1) < 1e-10, \
            f"gate b E failed ({key}): {a['E_od_end']} vs {e_arr[j]}"
    e13 = json.load(open(E13_JSON))["sue"]["41"]["per_od"]["9->15"]
    e13_delta = e13["E_od_end"] / np.hypot(e13["beta_end"], e13["E_od_end"])
    a41 = sue_a["41"]["per_od"]["10->16"]["delta_dir"]
    assert abs(a41 - e13_delta) < 1e-8, f"gate c failed: {a41} vs {e13_delta}"
    print("Part A gates (a)/(c) passed", flush=True)


def learned_side():
    e73 = json.load(open(E73_JSON))
    per_od = {}
    for rec in e73["records"]:
        key = f"{rec['od'][0]}->{rec['od'][1]}"
        per_od.setdefault(key, []).append(rec["ratio_cutspread"])
    return {k: {"median": float(np.median(v)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "n_seeds": len(v)}
            for k, v in per_od.items()}


def part_b(net, ods, q_base, ps, ec):
    """Fixed-background direct-SUE replication for OD (10,16).

    Background = all non-study published demands, assigned once at the
    joint logit-SUE equilibrium with the study ODs at q0 = 0.2 q_base,
    then frozen.  Only the study ODs vary; costs see study + background
    flows (cost patch on sue.path_costs).
    """
    B, M = ec.B, ec.M
    n, K = B.shape[0], len(ods)
    study = set(ods)
    trips = load_sioux_falls_trips()
    bg_ods = [(o, t) for (o, t), d in trips.items()
              if d > 0 and (o, t) not in study]
    print(f"Part B: {len(bg_ods)} background OD pairs "
          f"(+{K} study)", flush=True)
    full_ods = list(ods) + bg_ods
    ps_full = PathStructure(net, full_ods)
    q_bg = np.array([trips[od] for od in bg_ods])
    q0 = 0.2 * q_base
    q1 = 1.6 * q_base
    q_full0 = np.concatenate([q0, q_bg])

    print("solving joint equilibrium (background at published demand, "
          "study at q0)...", flush=True)
    h_star, res, info = sue_mod.solve_sue(
        ps_full, q_full0, SUE_THETA, tol=SUE_TOL)
    print(f"joint solve: res={res:.2e} msa={info['msa_iters']} "
          f"damp={info['damp_iters']}", flush=True)
    assert res <= max(SUE_TOL, 1e-7), f"background solve too loose: {res}"
    h_study, h_bg = h_star[:ps.P], h_star[ps.P:]
    F_BG = ps_full.Delta[:, ps.P:] @ h_bg
    F_study_joint = ps_full.Delta[:, :ps.P] @ h_study

    # freeze the background: patch the cost function to add F_BG
    orig_pc = sue_mod.path_costs

    def patched_pc(ps_, f):
        return orig_pc(ps_, f + F_BG)
    sue_mod.path_costs = patched_pc
    try:
        # gate (d): study-only solve at q0 reproduces the joint study flows
        smap = SUEMap(ps, SUE_THETA, SUE_TOL)
        F_check, _ = smap.F(q0)
        gate_d = float(np.max(np.abs(F_check - F_study_joint)))
        print(f"gate (d): max|F_study_re - F_joint_study| = {gate_d:.2e}",
              flush=True)
        assert gate_d <= 1e-5, f"gate d failed: {gate_d}"

        # the 11-node chain on the patched study map; TSTT on TOTAL flows
        ts = np.linspace(0.0, 1.0, 11)
        T = TSTTT(net.free_flow_time, net.capacity, net.m)
        Fts, Js, Tts = [], [], []
        for ti, t in enumerate(ts):
            qc = q0 + t * (q1 - q0)
            Fc, hc = smap.F(qc)
            smap.reset_warm(hc)
            Fts.append(Fc)
            Tts.append(float(T.value(Fc + F_BG)))
            J = np.zeros((net.m, K))
            for j in range(K):
                hj = FD_HREL * q_base[j]
                Fp, _ = smap.F(qc + hj * np.eye(K)[j], warm=hc)
                Fm, _ = smap.F(qc - hj * np.eye(K)[j], warm=hc)
                J[:, j] = (Fp - Fm) / (2.0 * hj)
            Js.append(J)
            print(f"  bg-SUE node {ti + 1}/11 solves={smap.n_solves} "
                  f"max_res={smap.max_res:.2e}", flush=True)
        Fts = np.stack(Fts)
        Js = np.stack(Js)
        Tts = np.array(Tts)
        gT = np.stack([T.grad(F + F_BG) for F in Fts])
        dQ = q1 - q0

        # gate (e): study conservation on every node
        cons = max(float(np.max(np.abs(B @ F - M @ qc_i)))
                   for F, qc_i in zip(Fts, q0 + ts[:, None] * dQ))
        print(f"gate (e): study conservation max = {cons:.2e}", flush=True)
        assert cons < 1e-8, f"gate e failed: {cons}"

        es = endpoint_scales(B, M, gT, Js, ts, dQ)
        BC = es["BC"]
        idx = {tuple(od): j for j, od in enumerate(ods)}
        j16 = idx[(9, 15)]  # paper OD (10,16)
        s, t = ods[j16]
        col = BC[:, j16]
        r = spread_row(BC, j16, s, t, n)
        r["delta_dir"] = float(es["E_od_end"][j16]
                               / np.hypot(es["beta_end"][j16],
                                          es["E_od_end"][j16]))
        r["beta_end"] = float(es["beta_end"][j16])
        r["E_od_end"] = float(es["E_od_end"][j16])
        # single-node expansion U={10} -> U'={10,17}: node 17 = idx 16
        col17 = float(col[16])
        phi_s = float(-col[s])
        r["node17_imbalance"] = col17
        r["cut_change_ratio_node17"] = col17 / phi_s
        r["node17_share_of_spread"] = abs(col17) / r["spread_abs"]
        # free-flow vs congestion split of the node-17 imbalance
        G = dQ[None, :] * np.trapezoid(Js, ts, axis=0)
        ff17 = float((B @ (net.free_flow_time * G[:, j16]))[16])
        r["node17_free_flow_part"] = ff17
        r["node17_congestion_part"] = col17 - ff17
        r["dT_total"] = float(Tts[-1] - Tts[0])
        r["completeness_gap"] = float(
            abs(np.sum(dQ[None, :] * np.trapezoid(
                gT[:, :, None] * Js, ts, axis=0)) - (Tts[-1] - Tts[0])))
        r["n_solves"] = int(smap.n_solves)
        r["solver_max_residual"] = float(smap.max_res)
        r["background"] = {
            "n_bg_ods": len(bg_ods),
            "f_bg_l1": float(np.abs(F_BG).sum()),
            "joint_residual": float(res),
            "joint_msa_iters": int(info["msa_iters"]),
            "joint_damp_iters": int(info["damp_iters"]),
            "gate_d_reproduction": gate_d,
            "gate_e_conservation": cons,
        }
        return r
    finally:
        sue_mod.path_costs = orig_pc


def main():
    net, ods, q_base, ps, classes, ec = _sioux_setup()

    sue_a = part_a(net, ods, q_base, ps, ec)
    check_gates_a(sue_a)

    learned = learned_side()
    comparison = {}
    for key, lr in learned.items():
        s11 = sue_a["11"]["per_od"][key]
        s41 = sue_a["41"]["per_od"][key]
        med = lr["median"] * 100.0
        d11 = abs(s11["ratio_cutspread"] * 100.0 - med)
        d41 = abs(s41["ratio_cutspread"] * 100.0 - med)
        comparison[key] = {
            "learned_median_pct": med,
            "learned_range_pct": [lr["min"] * 100.0, lr["max"] * 100.0],
            "sue11_pct": s11["ratio_cutspread"] * 100.0,
            "sue41_pct": s41["ratio_cutspread"] * 100.0,
            "abs_diff_pp_11": d11,
            "abs_diff_pp_41": d41,
            "sue_delta_dir_41": s41["delta_dir"],
        }
    max_diff = max(c["abs_diff_pp_41"] for c in comparison.values())
    print(f"\nmax |learned median - SUE(41n)| = {max_diff:.3f} pp",
          flush=True)

    print("\n== Part B: fixed-background replication ==", flush=True)
    bg = part_b(net, ods, q_base, ps, ec)

    summary = {
        "protocol": {
            # round-49 external review P3-5: the replay order for the
            # sort_keys-flattened per_od dicts (Table B.2's frozen OD list)
            "ods_order": [f"{s + 1}->{t + 1}" for s, t in ods],
            "part_a": ("e74 SUE FD chain replica (theta=1, tol=1e-7, "
                       "FD_HREL=5e-3 central differences, warm starts; "
                       "11 path nodes published, 41 convergence gate); "
                       "learned side read from e73_cut_span_summary.json "
                       "(five seeds per OD, no new training)"),
            "part_b": ("fixed background: all non-study published demands "
                       "assigned once at the joint logit-SUE equilibrium "
                       "with study ODs at q0=0.2 q_base, then frozen "
                       "(affine balance law Bf=Mq+b); TSTT on total "
                       "flows; 11-node chain, same FD protocol"),
        },
        "sue_all_od": sue_a,
        "learned_summary": learned,
        "comparison": comparison,
        "max_abs_diff_pp_41": max_diff,
        "fixed_background_od10_16": bg,
    }
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(f"\n[E78] wrote {OUT}")
    print(json.dumps(comparison, indent=1, sort_keys=True))
    print(json.dumps(bg, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
