"""E81 (v81, external review round 52, review item 3): is a low
all-cut spread weak response or integral cancellation?

The integrated columns B C~_{:,j} alone cannot distinguish (a) a weak
pointwise response along the path from (b) pointwise imbalances that
cancel within the integral.  Following the review's definitions, with

    a_j(lambda) = [B D_lambda J_f(gamma(lambda)) e_j]_interior,
    A_j = |dq_j| * int_0^1 ||a_j(lambda)||_1 dlambda,
    S_j = sum_{v != s_j, t_j} |(B C~_{:,j})_v|  (= the Eq.-(cutspread)
                                                 all-cut spread),

the triangle inequality gives S_j <= A_j, and

    kappa_j = 1 - S_j / A_j

measures how much of the pointwise imbalance cancels within the
integral: small A_j supports the weak-response reading; large A_j
with small S_j supports cancellation.

Fields (per OD pair):
  * learned surrogate, five seeds (exact forward-mode Jacobians on the
    frozen 200-step e9b/e73 chain) -- the primary evidence (analytic
    Jacobians, no FD noise);
  * direct logit-SUE setting A (no background, 41-node FD chain),
    B (frozen background, 11 nodes), C (background re-routing,
    11 nodes) from e79_chain_arrays.npz -- FD Jacobians; the A_j
    columns carry the FD noise floor of the published protocol, and
    low-spread ODs are read with that caveat.

Gates:
  (a) S_j <= A_j * (1 + 1e-9) for every column (triangle inequality);
  (b) S_j equals the recorded Eq.-(cutspread) spread of the same
      chain (learned: e73_cut_span_summary.json; SUE: e79 rows);
  (c) SUE A_j decreases monotonically in nothing in particular -- no
      gate; descriptive.

Output: experiments/results/e81_cancellation.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model  # noqa: E402
from attribution import outcome_field_C, endpoint_scales  # noqa: E402
from run_e9_transport import _sioux_setup, TSTTT  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RESULTS, "e81_cancellation.json")
E73_JSON = os.path.join(RESULTS, "e73_cut_span_summary.json")
E79_NPZ = os.path.join(RESULTS, "e79_chain_arrays.npz")
E79_JSON = os.path.join(RESULTS, "e79_bgreroute.json")


def cancellation_record(B, gT, Js, ts, dQ, BC, j, s, t):
    """A_j, S_j, kappa_j and the normalized A_j for one column."""
    n = B.shape[0]
    interior = np.ones(n, dtype=bool)
    interior[[s, t]] = False
    # pointwise weighted divergence on interior nodes, per path node
    a_l1 = np.empty(len(ts))
    for li in range(len(ts)):
        a = B @ (gT[li] * Js[li][:, j])
        a_l1[li] = np.abs(a[interior]).sum()
    A = abs(dQ[j]) * float(np.trapezoid(a_l1, ts))
    col = BC[:, j]
    S = float(np.abs(col[interior]).sum())
    phi_s = float(-col[s])
    assert S <= A * (1 + 1e-9), \
        f"triangle-inequality gate failed: S={S} > A={A}"
    return {
        "A_abs": A,
        "A_norm_pct": 100.0 * A / abs(phi_s),
        "S_abs": S,
        "S_norm_pct": 100.0 * S / abs(phi_s),
        "kappa": 1.0 - S / A if A > 0 else None,
        "phi_s": phi_s,
    }


def main():
    torch.set_num_threads(1)
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    B, M = ec.B, ec.M
    n = B.shape[0]
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    dQ = q1 - q0
    e73 = json.load(open(E73_JSON))
    e79 = json.load(open(E79_JSON))
    npz = np.load(E79_NPZ)

    out = {"gates": {}, "per_od": {}}
    gates = out["gates"]

    # ---- learned side: five seeds, exact Jacobians, 200-step chain ----
    learned = []
    for seed in range(5):
        path = os.path.join(RESULTS, "e5_ckpt", f"M3|norm|raw|{seed}.pt")
        full = torch.load(path, weights_only=False, map_location="cpu")
        norm = {"mu_q": np.zeros(len(ods)), "sd_q": np.ones(len(ods))}
        model = build_model("M3", ps, seed, norm=norm)
        model.load_state_dict(full["best"])
        model.eval()
        C, info = outcome_field_C(model, T, q0, q1, steps=200)
        es = endpoint_scales(B, M, info["gT_path"], info["J_path"],
                             info["ts"], info["dQ"])
        learned.append((info["gT_path"], info["J_path"], info["ts"],
                        info["dQ"], es["BC"]))

    sue_fields = {
        "sue_A_nobg_41n": (npz["gT_A"], npz["Js_A"], npz["ts_A"],
                           npz["BC_A"]),
        "sue_B_frozen_11n": (npz["gT_B"], npz["Js_B"], npz["ts_B"],
                             npz["BC_B"]),
        "sue_C_reroute_11n": (npz["gT_C"], npz["Js_C"], npz["ts_C"],
                              npz["BC_C"]),
    }
    e79_src = {"sue_A_nobg_41n": "setting_a_nobg_41n",
               "sue_B_frozen_11n": "setting_b_frozen_11n",
               "sue_C_reroute_11n": "setting_c_reroute_11n"}
    rec73 = {}
    for r in e73["records"]:
        rec73.setdefault((r["seed"], tuple(r["od"])), r)

    for j, (s, t) in enumerate(ods):
        key = f"{s + 1}->{t + 1}"
        row = {}
        # learned seeds
        kap, anorm = [], []
        for seed in range(5):
            gT, Js, ts, dQl, BC = learned[seed]
            r = cancellation_record(B, gT, Js, ts, dQl, BC, j, s, t)
            row[f"learned_seed{seed}"] = r
            kap.append(r["kappa"])
            anorm.append(r["A_norm_pct"])
            # gate (b): S equals the e73 spread of the same chain
            ref = rec73[(seed, (s + 1, t + 1))]["spread_abs"]
            rel = abs(r["S_abs"] - ref) / max(ref, 1e-300)
            gates[f"{key}_seed{seed}_S_vs_e73_rel"] = rel
            assert rel < 1e-10, f"e73 S gate failed {key} seed {seed}"
        row["learned_median"] = {
            "A_norm_pct": float(np.median(anorm)),
            "A_norm_pct_range": [float(np.min(anorm)),
                                 float(np.max(anorm))],
            "kappa_median": float(np.median(kap)),
            "kappa_range": [float(np.min(kap)), float(np.max(kap))],
        }
        # SUE fields
        for name, (gT, Js, ts, BC) in sue_fields.items():
            r = cancellation_record(B, gT, Js, ts, dQ, BC, j, s, t)
            row[name] = r
            ref = e79[e79_src[name]][key]["spread_abs"]
            rel = abs(r["S_abs"] - ref) / max(ref, 1e-300)
            gates[f"{key}_{name}_S_vs_e79_rel"] = rel
            assert rel < 1e-12, f"e79 S gate failed {key} {name}"
        out["per_od"][key] = row
        print(f"{key}: learned A {row['learned_median']['A_norm_pct']:7.2f}%"
              f" kappa {row['learned_median']['kappa_median']:.3f}"
              f" | SUE-A A {row['sue_A_nobg_41n']['A_norm_pct']:7.2f}%"
              f" kappa {row['sue_A_nobg_41n']['kappa']:.3f}", flush=True)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"\n[E81] wrote {OUT}")


if __name__ == "__main__":
    main()
