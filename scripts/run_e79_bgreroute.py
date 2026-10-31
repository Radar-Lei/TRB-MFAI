"""E79 (v81, external review round 52, review item 1): background
re-routing -- the third demand setting.

The v80 frozen background-flow check assigns all 520 non-study
published demands once at their joint logit-SUE equilibrium with the
study demands at q0 and then FREEZES the background link flows; only
the eight study ODs re-solve.  This experiment adds the complementary
setting the review asks for: background DEMANDS stay at their
published values, but at every integration node and every
finite-difference perturbation point ALL 528 ODs jointly re-equilibrate.
The differentiated object is then the TOTAL edge flow, so the Jacobian
columns include the background re-routing derivative dF_bg/dq_j --
a circulation component (B dF_bg/dq_j = 0 by background demand
constancy) that carries no vehicle-OD net demand but can produce
attribution imbalance through non-uniform outcome weights.  TSTT and
its gradient are evaluated on total flows.  Attribution still targets
the eight study ODs only.

Three settings, eight ODs each (the three-setting comparison the
review requests):
  A  no background (all non-study demands zero) -- 41-node chain,
     byte-replica of the e78 Part A 41-node protocol, now with the
     node-space columns BC and the chain arrays stored for the
     connected-cut (E80) and integral-cancellation (E81) analyses;
  B  frozen background flows (e78 Part B protocol extended from
     OD (10,16) to all eight study ODs; 11-node chain);
  C  background demands fixed, all ODs re-equilibrate (11-node chain,
     joint 528-OD solves at every node and FD point).

Per-OD rows report: origin-cut value phi_s, all-cuts spread (Eq.
cutspread), spread ratio %, dominant interior node (id, imbalance,
ratio to phi_s, share of spread, free-flow/congestion split), realized
endpoint scale beta_end, direction ratio delta_dir, E_od_end; OD
(10,16) additionally reports the node-17 expansion record in every
setting.

Gates (all must pass before any number enters the paper):
  (a) Part A 41-node per-OD rows reproduce e78_sue_allod.json
      sue_all_od["41"] to <1e-12 relative (byte-level chain replica);
  (b) Part B OD (10,16) row reproduces e78_sue_allod.json
      fixed_background_od10_16 scalars to <1e-12 relative;
  (c) Part C's independent cold joint solve at q0 reproduces the e78
      joint equilibrium's background l1 flow to <1e-9 relative and its
      residual stays <= max(SUE_TOL, 1e-7);
  (d) Part C total conservation max |B F_total - M_full q_full| over
      all chain nodes <= 1e-8 (path-flow construction);
  (e) completeness: |sum_j C_j - (T(q1) - T(q0))| reported relative to
      dT for settings B and C (Part B reference value 5.1e-4).

Output: experiments/results/e79_bgreroute.json
        experiments/results/e79_chain_arrays.npz
"""
from __future__ import annotations

import json
import os
import sys
import time

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
OUT = os.path.join(RESULTS, "e79_bgreroute.json")
NPZ = os.path.join(RESULTS, "e79_chain_arrays.npz")
E78_JSON = os.path.join(RESULTS, "e78_sue_allod.json")


def od_row(B, BC, es, j, s, t, n, G=None, t0=None):
    """Per-OD row: spread, dominant interior node, node-17 record."""
    col = BC[:, j]
    interior = np.ones(n, dtype=bool)
    interior[[s, t]] = False
    spread = float(np.abs(col[interior]).sum())
    phi_s = float(-col[s])
    row = {
        "phi_s": phi_s,
        "spread_abs": spread,
        "ratio_cutspread": spread / abs(phi_s) if phi_s != 0 else None,
        "beta_end": float(es["beta_end"][j]),
        "E_od_end": float(es["E_od_end"][j]),
        "delta_dir": float(es["E_od_end"][j]
                           / np.hypot(es["beta_end"][j],
                                      es["E_od_end"][j])),
    }
    inter_idx = np.where(interior)[0]
    v_star = int(inter_idx[np.argmax(np.abs(col[inter_idx]))])
    row["dominant_node_1idx"] = v_star + 1
    row["dominant_imbalance"] = float(col[v_star])
    row["dominant_ratio_pct"] = 100.0 * abs(col[v_star]) / abs(phi_s)
    row["dominant_share_of_spread"] = abs(col[v_star]) / spread
    if G is not None and t0 is not None:
        ff = float((B @ (t0 * G[:, j]))[v_star])
        row["dominant_free_flow_part"] = ff
        row["dominant_congestion_part"] = float(col[v_star]) - ff
    # node-17 (idx 16) record for the OD (10,16) column
    if (s, t) == (9, 15):
        row["node17_imbalance"] = float(col[16])
        row["cut_change_ratio_node17"] = float(col[16] / phi_s)
        row["node17_share_of_spread"] = abs(col[16]) / spread
        if G is not None and t0 is not None:
            ff17 = float((B @ (t0 * G[:, j]))[16])
            row["node17_free_flow_part"] = ff17
            row["node17_congestion_part"] = float(col[16]) - ff17
    return row


def sue_chain_allod(net, ods, q_base, ps, nodes, tag):
    """e78 sue_chain replica (41 or 11 nodes); returns Js, gT, F."""
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
        print(f"  {tag} {nodes}n node {ti + 1}/{nodes} "
              f"solves={smap.n_solves} max_res={smap.max_res:.2e} "
              f"({time.strftime('%H:%M:%S')})", flush=True)
    Fts = np.stack(Fts)
    Js = np.stack(Js)
    gT = np.stack([T.grad(F) for F in Fts])
    Tts = np.array([float(T.value(F)) for F in Fts])
    return ts, gT, Js, Fts, Tts, smap


def check_gate_a(rows_a):
    e78 = json.load(open(E78_JSON))
    ref = e78["sue_all_od"]["41"]["per_od"]
    for key, ra in rows_a.items():
        rb = ref[key]
        for k in ("ratio_cutspread", "delta_dir", "beta_end", "E_od_end"):
            assert abs(ra[k] / rb[k] - 1) < 1e-12, \
                f"gate a failed ({key}/{k}): {ra[k]} vs {rb[k]}"
    print("gate (a) passed: 41-node rows reproduce e78 Part A", flush=True)


def frozen_chain(net, ods, q_base, ps, ec, nodes=11):
    """Setting B chain: frozen background link flows (e78 Part B replica,
    solve sequence identical), all eight study ODs, TSTT on total flows."""
    B, M = ec.B, ec.M
    K = len(ods)
    study = set(ods)
    trips = load_sioux_falls_trips()
    bg_ods = [(o, t) for (o, t), d in trips.items()
              if d > 0 and (o, t) not in study]
    full_ods = list(ods) + bg_ods
    ps_full = PathStructure(net, full_ods)
    q_bg = np.array([trips[od] for od in bg_ods])
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    q_full0 = np.concatenate([q0, q_bg])

    print(f"Part B: solving joint equilibrium at q0 ({len(bg_ods)} bg ODs)",
          flush=True)
    t_start = time.time()
    h_star, res, info = sue_mod.solve_sue(ps_full, q_full0, SUE_THETA,
                                          tol=SUE_TOL)
    print(f"joint solve: res={res:.2e} msa={info['msa_iters']} "
          f"damp={info['damp_iters']} ({time.time() - t_start:.0f}s)",
          flush=True)
    assert res <= max(SUE_TOL, 1e-7), f"background solve too loose: {res}"
    h_bg = h_star[ps.P:]
    F_BG = ps_full.Delta[:, ps.P:] @ h_bg
    f_bg_l1 = float(np.abs(F_BG).sum())

    orig_pc = sue_mod.path_costs

    def patched_pc(ps_, f):
        return orig_pc(ps_, f + F_BG)
    sue_mod.path_costs = patched_pc
    try:
        smap = SUEMap(ps, SUE_THETA, SUE_TOL)
        # e78 gate (d): study-only solve at q0 under the frozen background
        # reproduces the joint equilibrium's study link flows
        F_study_joint = ps_full.Delta[:, :ps.P] @ h_star[:ps.P]
        F_check, _ = smap.F(q0)
        gate_d = float(np.max(np.abs(F_check - F_study_joint)))
        print(f"gate (d): max|F_study_re - F_joint_study| = {gate_d:.2e}",
              flush=True)
        assert gate_d <= 1e-5, f"gate d failed: {gate_d}"
        ts = np.linspace(0.0, 1.0, nodes)
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
            print(f"  B node {ti + 1}/{nodes} solves={smap.n_solves} "
                  f"max_res={smap.max_res:.2e}", flush=True)
        Fts, Js, Tts = np.stack(Fts), np.stack(Js), np.array(Tts)
        gT = np.stack([T.grad(F + F_BG) for F in Fts])
        dQ = q1 - q0
        # study conservation on every node (e78 gate e)
        cons = max(float(np.max(np.abs(B @ F - M @ qc_i)))
                   for F, qc_i in zip(Fts, q0 + ts[:, None] * dQ))
        assert cons < 1e-8, f"study conservation gate failed: {cons}"
        assert smap.max_res <= SUE_TOL, \
            f"solver tolerance exceeded (frozen bg): {smap.max_res}"
        es = endpoint_scales(B, M, gT, Js, ts, dQ)
        return (es, F_BG, f_bg_l1, res, cons, Tts, smap, ts, gT, Js,
                gate_d)
    finally:
        sue_mod.path_costs = orig_pc


class JointSUEMap:
    """528-OD joint logit-SUE map with the frozen warm-fail retry protocol.

    sue.generate_dataset's sanctioned pattern: solve warm-started first
    (damped refinement only); if the residual misses tolerance -- the
    joint map is stiff at FD perturbation points and the damped phase
    can stall (observed res up to 2.6) -- retry once from a cold
    free-flow start with the full MSA phase; a second cold retry with
    doubled MSA budget precedes a hard failure."""

    def __init__(self, ps_full, theta, tol):
        self.ps = ps_full
        self.theta = theta
        self.tol = tol
        self.max_res = 0.0
        self.n_solves = 0
        self.n_warm_fail = 0
        self.n_cold2 = 0
        self._warm = None

    def F(self, q, warm=None):
        h0 = self._warm if warm is None else warm
        h, res, info = sue_mod.solve_sue(self.ps, q, self.theta,
                                         tol=self.tol, h0=h0)
        self.n_solves += 1
        if res > self.tol and h0 is not None:
            self.n_warm_fail += 1
            print(f"    warm solve stalled (res={res:.2e}); cold retry",
                  flush=True)
            h, res, info = sue_mod.solve_sue(self.ps, q, self.theta,
                                             tol=self.tol, h0=None)
            self.n_solves += 1
        if res > self.tol:
            self.n_cold2 += 1
            print(f"    cold solve missed tol (res={res:.2e}); extended "
                  f"MSA retry", flush=True)
            h, res, info = sue_mod.solve_sue(self.ps, q, self.theta,
                                             tol=self.tol, h0=None,
                                             msa_iters=6000)
            self.n_solves += 1
        if res > self.tol:
            raise RuntimeError(
                f"joint SUE solve failed to converge: res={res:.2e} "
                f"> {self.tol:.0e}")
        self.max_res = max(self.max_res, float(res))
        return self.ps.Delta @ h, h

    def reset_warm(self, h):
        self._warm = h


def reroute_chain(net, ods, q_base, ps, ec, nodes=11):
    """Setting C chain: background demands fixed, ALL ODs re-equilibrate at
    every node and FD point; Jacobian of TOTAL flow wrt study demands."""
    B, M = ec.B, ec.M
    K = len(ods)
    study = set(ods)
    trips = load_sioux_falls_trips()
    bg_ods = [(o, t) for (o, t), d in trips.items()
              if d > 0 and (o, t) not in study]
    full_ods = list(ods) + bg_ods
    ps_full = PathStructure(net, full_ods)
    P_study = ps.P
    q_bg = np.array([trips[od] for od in bg_ods])
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    dQ = q1 - q0
    M_full = net.od_M(full_ods)  # (n, K_full)

    # independent cold joint solve at q0 -- gate (c) anchor
    print(f"Part C: cold joint solve at q0 ({len(bg_ods)} bg ODs)",
          flush=True)
    t_start = time.time()
    h_star, res0, info0 = sue_mod.solve_sue(
        ps_full, np.concatenate([q0, q_bg]), SUE_THETA, tol=SUE_TOL)
    print(f"q0 joint solve: res={res0:.2e} msa={info0['msa_iters']} "
          f"damp={info0['damp_iters']} ({time.time() - t_start:.0f}s)",
          flush=True)
    assert res0 <= max(SUE_TOL, 1e-7), f"q0 joint solve too loose: {res0}"
    F_bg_q0 = ps_full.Delta[:, P_study:] @ h_star[P_study:]
    fbg_l1_q0 = float(np.abs(F_bg_q0).sum())

    smap = JointSUEMap(ps_full, SUE_THETA, SUE_TOL)
    smap.reset_warm(h_star)  # node 0 continues from the q0 solution
    ts = np.linspace(0.0, 1.0, nodes)
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    Fts, Js, Tts = [], [], []
    for ti, t in enumerate(ts):
        qc = q0 + t * dQ
        q_full = np.concatenate([qc, q_bg])
        Fc, hc = smap.F(q_full)
        smap.reset_warm(hc)
        Fts.append(Fc)
        Tts.append(float(T.value(Fc)))
        J = np.zeros((net.m, K))
        for j in range(K):
            hj = FD_HREL * q_base[j]
            qp = q_full.copy()
            qp[j] += hj
            qm = q_full.copy()
            qm[j] -= hj
            Fp, _ = smap.F(qp, warm=hc)
            Fm, _ = smap.F(qm, warm=hc)
            J[:, j] = (Fp - Fm) / (2.0 * hj)
        Js.append(J)
        print(f"  C node {ti + 1}/{nodes} solves={smap.n_solves} "
              f"max_res={smap.max_res:.2e} ({time.strftime('%H:%M:%S')})",
              flush=True)
    Fts, Js, Tts = np.stack(Fts), np.stack(Js), np.array(Tts)
    gT = np.stack([T.grad(F) for F in Fts])

    # gate (d): total conservation on every node
    cons = 0.0
    for ti, t in enumerate(ts):
        qc = q0 + t * dQ
        q_full = np.concatenate([qc, q_bg])
        cons = max(cons, float(np.max(np.abs(B @ Fts[ti]
                                             - M_full @ q_full))))
    assert smap.max_res <= SUE_TOL, \
        f"solver tolerance exceeded (re-routing): {smap.max_res}"
    es = endpoint_scales(B, M, gT, Js, ts, dQ)
    return es, fbg_l1_q0, res0, cons, Tts, smap, ts, gT, Js, Fts


def main():
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    B, n = ec.B, ec.B.shape[0]
    t0v = np.asarray(net.free_flow_time, dtype=float)
    # e78's exact increment expressions: part_a used (1.6-0.2)*q_base,
    # part_b used q1 - q0; keep both bitwise as in the frozen protocol
    dQ_a = (1.6 - 0.2) * q_base
    dQ_b = 1.6 * q_base - 0.2 * q_base      # = q1 - q0
    e78 = json.load(open(E78_JSON))
    # frozen e78 joint-equilibrium background record (gate c reference)
    ref_bg_l1 = e78["fixed_background_od10_16"]["background"]["f_bg_l1"]

    # ---------------- Part A: no background, 41 nodes ----------------
    print("== Part A: no-background SUE chain, 41 nodes ==", flush=True)
    t_start = time.time()
    ts_a, gT_a, Js_a, Fts_a, Tts_a, smap_a = sue_chain_allod(
        net, ods, q_base, ps, 41, "A")
    print(f"Part A chain time: {time.time() - t_start:.0f}s", flush=True)
    assert smap_a.max_res <= SUE_TOL, \
        f"solver tolerance exceeded (41n chain): {smap_a.max_res}"
    es_a = endpoint_scales(B, ec.M, gT_a, Js_a, ts_a, dQ_a)
    BC_a = es_a["BC"]
    G_a = dQ_a[None, :] * np.trapezoid(Js_a, ts_a, axis=0)
    rows_a = {f"{s + 1}->{t + 1}": od_row(B, BC_a, es_a, j, s, t, n,
                                          G=G_a, t0=t0v)
              for j, (s, t) in enumerate(ods)}
    check_gate_a(rows_a)
    comp_a = abs(float(np.sum(dQ_a[None, :] * np.trapezoid(
        gT_a[:, :, None] * Js_a, ts_a, axis=0))) - (Tts_a[-1] - Tts_a[0]))

    # ---------------- Part B: frozen background flows ----------------
    print("\n== Part B: frozen background, 11 nodes, all 8 ODs ==", flush=True)
    (es_b, F_BG, f_bg_l1, joint_res, cons_b, Tts_b, smap_b, ts_b, gT_b,
     Js_b, gate_d) = frozen_chain(net, ods, q_base, ps, ec)
    # the re-solved joint equilibrium itself matches the frozen record
    assert abs(f_bg_l1 / ref_bg_l1 - 1) < 1e-9, \
        f"Part B joint solve drifted from e78: {f_bg_l1} vs {ref_bg_l1}"
    BC_b = es_b["BC"]
    G_b = dQ_b[None, :] * np.trapezoid(Js_b, ts_b, axis=0)
    rows_b = {f"{s + 1}->{t + 1}": od_row(B, BC_b, es_b, j, s, t, n,
                                          G=G_b, t0=t0v)
              for j, (s, t) in enumerate(ods)}
    comp_b = abs(float(np.sum(dQ_b[None, :] * np.trapezoid(
        gT_b[:, :, None] * Js_b, ts_b, axis=0))) - (Tts_b[-1] - Tts_b[0]))

    # gate (b): OD (10,16) reproduces e78 Part B
    ref = e78["fixed_background_od10_16"]
    rb = rows_b["10->16"]
    for k in ("ratio_cutspread", "delta_dir", "beta_end", "E_od_end"):
        assert abs(rb[k] / ref[k] - 1) < 1e-12, \
            f"gate b failed ({k}): {rb[k]} vs {ref[k]}"
    for k in ("node17_imbalance", "node17_share_of_spread",
              "node17_free_flow_part"):
        assert abs(rb[k] / ref[k] - 1) < 1e-12, \
            f"gate b failed ({k}): {rb[k]} vs {ref[k]}"
    print("gate (b) passed: frozen-bg (10,16) reproduces e78 Part B",
          flush=True)

    # ---------------- Part C: background re-routing ----------------
    print("\n== Part C: background re-routing (joint 528-OD), 11 nodes ==",
          flush=True)
    t_start = time.time()
    (es_c, fbg_l1_q0, res0, cons_c, Tts_c, smap_c, ts_c, gT_c, Js_c,
     Fts_c) = reroute_chain(net, ods, q_base, ps, ec)
    print(f"Part C chain time: {time.time() - t_start:.0f}s", flush=True)
    BC_c = es_c["BC"]
    G_c = dQ_b[None, :] * np.trapezoid(Js_c, ts_c, axis=0)
    rows_c = {f"{s + 1}->{t + 1}": od_row(B, BC_c, es_c, j, s, t, n,
                                          G=G_c, t0=t0v)
              for j, (s, t) in enumerate(ods)}
    comp_c = abs(float(np.sum(dQ_b[None, :] * np.trapezoid(
        gT_c[:, :, None] * Js_c, ts_c, axis=0))) - (Tts_c[-1] - Tts_c[0]))

    # gates (c)/(d): the re-routing q0 joint solve reproduces the FROZEN
    # e78 joint-equilibrium background record
    assert abs(fbg_l1_q0 / ref_bg_l1 - 1) < 1e-9, \
        f"gate c failed: q0 joint background l1 {fbg_l1_q0} vs e78 {ref_bg_l1}"
    assert cons_c <= 1e-8, f"gate d failed: total conservation {cons_c}"
    print(f"gate (c) passed: q0 joint background l1 matches frozen e78 "
          f"({fbg_l1_q0:.6f} vs {ref_bg_l1:.6f})", flush=True)
    print(f"gate (d) passed: total conservation max {cons_c:.2e}",
          flush=True)

    # ---------------- 11-node zero-background chain ----------------
    # (e78 Part A 11n replica; Js needed for the Jacobian-shift statistic
    #  and downstream E81/E82 analyses)
    print("\n== 11-node zero-background chain (protocol replica) ==",
          flush=True)
    ts_11, gT_11, Js_11, Fts_11, Tts_11, smap_11 = sue_chain_allod(
        net, ods, q_base, ps, 11, "A")
    es_11 = endpoint_scales(B, ec.M, gT_11, Js_11, ts_11, dQ_a)
    ref11 = e78["sue_all_od"]["11"]["per_od"]
    for j, (s, t) in enumerate(ods):
        key = f"{s + 1}->{t + 1}"
        phi_s = float(-es_11["BC"][s, j])
        spread = float(np.abs(np.delete(es_11["BC"][:, j], [s, t])).sum())
        ratio = spread / abs(phi_s)
        assert abs(ratio / ref11[key]["ratio_cutspread"] - 1) < 1e-12, \
            f"11n replica gate failed ({key})"
    print("11-node zero-background chain reproduces e78 11n rows",
          flush=True)
    # Jacobian shift setting C vs zero background (descriptive; mixes the
    # congestion-level change with re-routing per se)
    jshift = np.abs(Js_c - Js_11).sum(axis=1)          # (nodes, K) L1/edge
    jshift_rel = jshift / np.abs(Js_11).sum(axis=1)    # relative to ||J||_1

    dT_c = float(Tts_c[-1] - Tts_c[0])
    summary = {
        "protocol": {
            "ods_order": [f"{s + 1}->{t + 1}" for s, t in ods],
            "setting_a": ("no background (non-study demands zero); direct "
                          "logit-SUE map, theta=1, tol=1e-7, 41-node chain, "
                          "FD_HREL=5e-3 central differences (e78 Part A "
                          "replica)"),
            "setting_b": ("background link flows frozen at the q0 joint "
                          "equilibrium (e78 Part B protocol, extended to "
                          "all eight study ODs); TSTT on total flows"),
            "setting_c": ("background demands fixed at published values, "
                          "all 528 ODs jointly re-equilibrate at every "
                          "node and FD point; Jacobian of TOTAL flow; "
                          "TSTT on total flows"),
        },
        "setting_a_nobg_41n": rows_a,
        "setting_b_frozen_11n": rows_b,
        "setting_c_reroute_11n": rows_c,
        "completeness": {
            "a_gap_abs": comp_a,
            "a_gap_rel": comp_a / abs(Tts_a[-1] - Tts_a[0]),
            "b_gap_abs": comp_b,
            "b_gap_rel": comp_b / abs(Tts_b[-1] - Tts_b[0]),
            "c_gap_abs": comp_c,
            "c_gap_rel": comp_c / abs(dT_c),
            "c_dT": dT_c,
        },
        "background": {
            "f_bg_l1": f_bg_l1,
            "joint_residual_q0": float(joint_res),
            "f_bg_l1_reroute_q0": fbg_l1_q0,
            "reroute_q0_residual": float(res0),
            "cons_b_study": float(cons_b),
            "cons_c_total": float(cons_c),
            "jacobian_shift_l1_mean": {f"{s + 1}->{t + 1}":
                                       float(jshift[:, j].mean())
                                       for j, (s, t) in enumerate(ods)},
            "jacobian_shift_l1_rel_mean": {f"{s + 1}->{t + 1}":
                                           float(jshift_rel[:, j].mean())
                                           for j, (s, t) in enumerate(ods)},
        },
        "solver": {
            "a_n_solves": int(smap_a.n_solves),
            "a_max_res": float(smap_a.max_res),
            "b_n_solves": int(smap_b.n_solves),
            "b_max_res": float(smap_b.max_res),
            "c_n_solves": int(smap_c.n_solves),
            "c_max_res": float(smap_c.max_res),
            "c_warm_fail_retries": int(smap_c.n_warm_fail),
            "c_extended_msa_retries": int(smap_c.n_cold2),
        },
    }
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    np.savez_compressed(
        NPZ,
        BC_A=BC_a, gT_A=gT_a, Js_A=Js_a, F_A=Fts_a, ts_A=ts_a,
        BC_B=BC_b, gT_B=gT_b, Js_B=Js_b, ts_B=ts_b, F_BG=F_BG,
        BC_C=BC_c, gT_C=gT_c, Js_C=Js_c, F_C=Fts_c, ts_C=ts_c,
        Js_11=Js_11, gT_11=gT_11, ts_11=ts_11,
        dQ=dQ_a)
    print(f"\n[E79] wrote {OUT}\n[E79] wrote {NPZ}")

    # console digest: the three-setting comparison
    print("\n== three-setting comparison (spread % of origin cut) ==")
    for key in rows_a:
        print(f"{key:>9}: A {100 * rows_a[key]['ratio_cutspread']:6.2f}%"
              f"  B {100 * rows_b[key]['ratio_cutspread']:6.2f}%"
              f"  C {100 * rows_c[key]['ratio_cutspread']:6.2f}%"
              f"  dom {rows_a[key]['dominant_node_1idx']}"
              f"/{rows_b[key]['dominant_node_1idx']}"
              f"/{rows_c[key]['dominant_node_1idx']}")


if __name__ == "__main__":
    main()
