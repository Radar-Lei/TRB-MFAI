"""E80 (v81, external review round 52, review item 2): connected-region
cut analysis -- how much of the unrestricted all-cut spread survives
on transportation-meaningful (connected) regional boundaries.

The Eq.-(cutspread) identity ranges over EVERY endpoint-separating
node subset, with no connectivity restriction on the cut region; the
transportation reading (screenlines, cordons, regional boundaries)
lives on the restricted family
    U_j^conn = {U : s_j in U, t_j notin U, G'[U] connected and
                G'[V\\U] connected},
with G' the underlying undirected graph (24 nodes, so the family is
enumerated EXACTLY -- a C bitmask enumerator compiled from
e80_enum.c, include/exclude frontier recursion; the connected-subset
tree exceeds 3e6 nodes, so the pure-Python reference is used only on
restricted neighborhoods as a cross-check).  phi_f(U) is carried
incrementally; the exact min/max of the cut value over the family is
reported side by side with the unrestricted Eq.-(cutspread) value.

Columns analyzed per OD pair:
  * learned surrogate, five seeds (TSTT outcome, frozen 200-step
    e9b/e73 chain);
  * direct logit-SUE, settings A (no background, 41-node chain),
    B (frozen background flows, 11 nodes), C (background re-routing,
    11 nodes) -- read from e79_chain_arrays.npz.

A small range on the connected family is NOT structural compatibility:
it says the account is approximately cut-independent on the examined
regional boundaries only; Theorems 1-2 are untouched.

Gates (all must pass before any number enters the paper):
  (a) connected range <= unrestricted spread for every column
      (the connected family is a subset of the separating family);
  (b) C enumerator == pure-Python enumerator on the distance-2 and
      distance-3 restricted families of every OD (count and per-field
      min/max to round-off);
  (c) the attaining min/max masks are in-family (both sides connected,
      in Python) and their phi values match direct evaluation;
  (d) learned per-seed unrestricted spreads reproduce the frozen e73
      records (<1e-10 rel); direct-SUE unrestricted spreads reproduce
      the e79 rows (<1e-12 rel);
  (e) for OD (10,16), the paper's {10} -> {10,17} expansion lies in
      the family and its phi lies within the connected range.

Output: experiments/results/e80_connected_cuts.json
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model  # noqa: E402
from attribution import outcome_field_C, endpoint_scales  # noqa: E402
from run_e9_transport import _sioux_setup, TSTTT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "e80_connected_cuts.json")
E73_JSON = os.path.join(RESULTS, "e73_cut_span_summary.json")
E79_NPZ = os.path.join(RESULTS, "e79_chain_arrays.npz")
E79_JSON = os.path.join(RESULTS, "e79_bgreroute.json")
C_SRC = os.path.join(HERE, "e80_enum.c")
C_SO = os.path.join(HERE, "e80_enum.so")


# ----------------------------------------------------------------------------
# Graph/bitmask helpers (Python reference)
# ----------------------------------------------------------------------------
def build_adj_bits(net):
    n = net.n
    adj = [0] * n
    for u, v in net.edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj, n


def reachable(adj, start_mask, allowed):
    reach = start_mask & allowed
    frontier = reach
    while frontier:
        nxt = 0
        m = frontier
        while m:
            b = m & (-m)
            v = b.bit_length() - 1
            nxt |= adj[v]
            m ^= b
        nxt &= allowed & ~reach
        reach |= nxt
        frontier = nxt
    return reach


def connected(adj, node_mask):
    if node_mask == 0:
        return False
    low = node_mask & (-node_mask)
    return reachable(adj, low, node_mask) == node_mask


def enum_python(adj, n, s, t, cols, allowed=None):
    """Pure-Python reference enumerator (restricted families only).

    ESU discipline: a vertex popped from the candidate list is banned
    along the whole subtree (excl mask), so adjacency growth cannot
    re-introduce it; each connected subset containing s (t excluded)
    is generated exactly once."""
    F = cols.shape[1]
    full = (1 << n) - 1
    if allowed is None:
        allowed = full
    best_min = np.full(F, np.inf)
    best_max = np.full(F, -np.inf)
    n_sets = 0

    def process(S, acc):
        nonlocal n_sets
        n_sets += 1
        np.minimum(best_min, acc, out=best_min)
        np.maximum(best_max, acc, out=best_max)

    def expand(S, cand, excl, acc):
        popped = 0
        while cand:
            b = cand & (-cand)
            v = b.bit_length() - 1
            cand ^= b
            if v == t:
                popped |= b
                continue
            S2 = S | b
            acc2 = acc - cols[v, :]
            if connected(adj, full & ~S2):
                process(S2, acc2)
            expand(S2, (cand | adj[v]) & ~S2 & ~(excl | popped) & allowed,
                   excl | popped, acc2)
            popped |= b

    s_ok = connected(adj, full & ~(1 << s))
    if s_ok:
        process(1 << s, -cols[s, :].copy())
    expand(1 << s, adj[s] & ~((1 << s) | (1 << t)) & allowed,
           1 << t, -cols[s, :].copy())
    return {"n_sets": n_sets, "phi_min": best_min, "phi_max": best_max,
            "singleton_admissible": s_ok}


def enum_bruteforce(adj, n, s, t, cols, allowed=None):
    """Independent brute force over all subsets of the allowed vertices
    containing s (restricted families only; |allowed| <= ~20).  A subset
    is in-family iff it avoids t, G'[U] and G'[V\\U] are connected."""
    F = cols.shape[1]
    full = (1 << n) - 1
    if allowed is None:
        allowed = full
    others = [v for v in range(n)
              if (allowed >> v) & 1 and v != s]
    best_min = np.full(F, np.inf)
    best_max = np.full(F, -np.inf)
    n_sets = 0
    for code in range(1 << len(others)):
        U = 1 << s
        c = code
        for v in others:
            if c & 1:
                U |= 1 << v
            c >>= 1
        if (U >> t) & 1:
            continue
        if not connected(adj, U):
            continue
        if not connected(adj, full & ~U):
            continue
        acc = -np.array([sum(cols[v2, f] for v2 in range(n)
                             if (U >> v2) & 1)
                         for f in range(F)])
        n_sets += 1
        np.minimum(best_min, acc, out=best_min)
        np.maximum(best_max, acc, out=best_max)
    s_ok = connected(adj, full & ~(1 << s))
    return {"n_sets": n_sets, "phi_min": best_min, "phi_max": best_max,
            "singleton_admissible": s_ok}


# ----------------------------------------------------------------------------
# C enumerator (ctypes)
# ----------------------------------------------------------------------------
def load_c_enum():
    if not os.path.exists(C_SO):
        subprocess.run(
            ["gcc", "-O2", "-shared", "-fPIC", "-o", C_SO, C_SRC],
            check=True)
    lib = ctypes.CDLL(C_SO)
    lib.enumerate.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64)]
    return lib


def enum_c(lib, adj, n, s, t, cols, allowed=None):
    F = cols.shape[1]
    assert F <= 8
    full = (1 << n) - 1
    if allowed is None:
        allowed = full
    adj_arr = (ctypes.c_uint32 * n)(*adj)
    cols_arr = (ctypes.c_double * (n * F))(
        *[float(x) for x in cols.ravel()])
    bmin = (ctypes.c_double * F)()
    bmax = (ctypes.c_double * F)()
    mmask = (ctypes.c_uint32 * F)()
    xmask = (ctypes.c_uint32 * F)()
    counts = (ctypes.c_uint64 * 2)()
    lib.enumerate(n, adj_arr, s, t, F, cols_arr, allowed,
                  bmin, bmax, mmask, xmask, counts)
    return {"n_sets": int(counts[1]),
            "singleton_admissible": bool(counts[0]),
            "phi_min": np.array(bmin[:F]),
            "phi_max": np.array(bmax[:F]),
            "min_mask": [int(mmask[f]) for f in range(F)],
            "max_mask": [int(xmask[f]) for f in range(F)]}


def mask_to_nodes(mask):
    out = []
    while mask:
        b = mask & (-mask)
        out.append(b.bit_length() - 1)
        mask ^= b
    return [v + 1 for v in out]  # 1-indexed


def neighborhood_mask(adj, n, s, depth):
    """All nodes within undirected distance <= depth of s (plus s)."""
    mask = 1 << s
    frontier = mask
    for _ in range(depth):
        nxt = 0
        m = frontier
        while m:
            b = m & (-m)
            nxt |= adj[b.bit_length() - 1]
            m ^= b
        nxt &= ~mask
        mask |= nxt
        frontier = nxt
    return mask


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    torch.set_num_threads(1)
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    B, M = ec.B, ec.M
    n = net.n
    T = TSTTT(net.free_flow_time, net.capacity, net.m)
    q0, q1 = 0.2 * q_base, 1.6 * q_base
    e73 = json.load(open(E73_JSON))
    e79 = json.load(open(E79_JSON))
    npz = np.load(E79_NPZ)
    adj, _ = build_adj_bits(net)
    lib = load_c_enum()

    # ---- learned columns (five seeds, frozen 200-step chain) ----
    learned_BC = []
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
        learned_BC.append(es["BC"])

    sue_cols = {"A_nobg_41n": npz["BC_A"], "B_frozen_11n": npz["BC_B"],
                "C_reroute_11n": npz["BC_C"]}

    out = {"gates": {}, "per_od": {}, "family_sizes": {}}
    gates = out["gates"]
    rng = np.random.default_rng(20261017)

    for j, (s, t) in enumerate(ods):
        key = f"{s + 1}->{t + 1}"
        fields = {}
        for seed in range(5):
            fields[f"learned_seed{seed}"] = learned_BC[seed][:, j]
        for name, BC in sue_cols.items():
            fields[f"sue_{name}"] = BC[:, j]
        cols = np.stack(list(fields.values()), axis=1)   # (n, F)

        # gate (d1): learned unrestricted spreads reproduce e73
        rec73 = {}
        for r in e73["records"]:
            if tuple(r["od"]) == (s + 1, t + 1):
                rec73[r["seed"]] = r["spread_abs"]
        for seed in range(5):
            interior = np.ones(n, dtype=bool)
            interior[[s, t]] = False
            spread = float(np.abs(learned_BC[seed][interior, j]).sum())
            rel = abs(spread - rec73[seed]) / max(rec73[seed], 1e-300)
            gates[f"{key}_seed{seed}_e73_rel"] = rel
            assert rel < 1e-10, f"e73 gate failed {key} seed {seed}"
        # gate (d2): direct unrestricted spreads reproduce e79 rows
        for name, src in (("A_nobg_41n", "setting_a_nobg_41n"),
                          ("B_frozen_11n", "setting_b_frozen_11n"),
                          ("C_reroute_11n", "setting_c_reroute_11n")):
            interior = np.ones(n, dtype=bool)
            interior[[s, t]] = False
            spread = float(np.abs(sue_cols[name][interior, j]).sum())
            ref = e79[src][key]["spread_abs"]
            rel = abs(spread - ref) / max(ref, 1e-300)
            gates[f"{key}_{name}_e79_rel"] = rel
            assert rel < 1e-12, f"e79 gate failed {key} {name}"

        # gate (b): C == Python == independent brute force on restricted
        # neighborhoods (random columns exercise the min/max paths)
        for depth in (2, 3):
            allowed = neighborhood_mask(adj, n, s, depth)
            rand_cols = rng.normal(size=(n, 3))
            for tag, cc in (("rand", rand_cols), ("real", cols)):
                rp = enum_python(adj, n, s, t, cc, allowed=allowed)
                rc = enum_c(lib, adj, n, s, t, cc, allowed=allowed)
                assert rp["n_sets"] == rc["n_sets"], \
                    (f"gate b count failed {key} d{depth} {tag}: "
                     f"{rp['n_sets']} vs {rc['n_sets']}")
                dmin = np.max(np.abs(rp["phi_min"] - rc["phi_min"])
                              / np.maximum(np.abs(rp["phi_min"]), 1e-12))
                dmax = np.max(np.abs(rp["phi_max"] - rc["phi_max"])
                              / np.maximum(np.abs(rp["phi_max"]), 1e-12))
                gates[f"{key}_d{depth}_{tag}_py_vs_c"] = \
                    float(max(dmin, dmax))
                assert max(dmin, dmax) < 1e-9, \
                    f"gate b failed {key} d{depth} {tag}"
            # independent brute force (exact, all subsets of the allowed
            # set; bounded to <=20 allowed vertices)
            n_allowed = bin(allowed).count("1")
            if n_allowed <= 20:
                rb = enum_bruteforce(adj, n, s, t, rand_cols,
                                     allowed=allowed)
                rc = enum_c(lib, adj, n, s, t, rand_cols, allowed=allowed)
                assert rb["n_sets"] == rc["n_sets"], \
                    (f"gate b brute-force count failed {key} d{depth}: "
                     f"{rb['n_sets']} vs {rc['n_sets']}")
                dev = float(np.max(np.abs(
                    np.concatenate([rb["phi_min"] - rc["phi_min"],
                                    rb["phi_max"] - rc["phi_max"]])
                ) / np.maximum(np.abs(np.concatenate(
                    [rb["phi_min"], rb["phi_max"]])), 1e-12)))
                gates[f"{key}_d{depth}_bruteforce_vs_c"] = dev
                assert dev < 1e-9, \
                    f"gate b brute-force failed {key} d{depth}"

        # full exact enumeration
        t_start = time.time()
        res = enum_c(lib, adj, n, s, t, cols)
        out["family_sizes"][key] = res["n_sets"]

        row = {"n_connected_cuts": res["n_sets"],
               "singleton_admissible": bool(res["singleton_admissible"])}
        fld_names = list(fields.keys())
        full_mask = (1 << n) - 1
        for f, name in enumerate(fld_names):
            col = cols[:, f]
            interior = np.ones(n, dtype=bool)
            interior[[s, t]] = False
            unrestricted = float(np.abs(col[interior]).sum())
            phi_s = float(-col[s])
            conn_range = float(res["phi_max"][f] - res["phi_min"][f])
            # gate (a): connected range <= unrestricted spread
            assert conn_range <= unrestricted * (1 + 1e-9), \
                f"gate a failed {key}/{name}: {conn_range} > {unrestricted}"
            # gate (c): attaining masks are in-family with matching phi
            for msk, val in ((res["min_mask"][f], res["phi_min"][f]),
                             (res["max_mask"][f], res["phi_max"][f])):
                nodes = mask_to_nodes(msk)
                assert (s + 1) in nodes and (t + 1) not in nodes
                assert connected(adj, msk) and \
                    connected(adj, full_mask & ~msk), \
                    f"gate c failed {key}/{name}: mask not in family"
                phi_direct = -float(col[np.array(nodes) - 1].sum())
                assert abs(phi_direct - val) <= 1e-6 * max(1.0, abs(val)), \
                    f"gate c failed {key}/{name}: phi mismatch"
            entry = {
                "phi_s": phi_s,
                "unrestricted_pct": 100.0 * unrestricted / abs(phi_s),
                "conn_min": float(res["phi_min"][f]),
                "conn_max": float(res["phi_max"][f]),
                "conn_range": conn_range,
                "conn_range_pct": 100.0 * conn_range / abs(phi_s),
                "shrink_factor": conn_range / unrestricted
                if unrestricted > 0 else None,
            }
            if name == "sue_A_nobg_41n":
                entry["min_cut_nodes"] = mask_to_nodes(res["min_mask"][f])
                entry["max_cut_nodes"] = mask_to_nodes(res["max_mask"][f])
            row[name] = entry
        # gate (e): the {10} -> {10,17} expansion for OD (10,16)
        if (s, t) == (9, 15):
            U = (1 << 9) | (1 << 16)
            comp = full_mask & ~U
            ok = connected(adj, U) and connected(adj, comp)
            row["cut_s10_n17_in_family"] = bool(ok)
            fA = fld_names.index("sue_A_nobg_41n")
            if ok:
                colA = sue_cols["A_nobg_41n"][:, j]
                phi_direct = -float(colA[9] + colA[16])
                assert res["phi_min"][fA] - 1e-6 <= phi_direct \
                    <= res["phi_max"][fA] + 1e-6, \
                    "gate e failed: {10,17} phi outside connected range"
                row["cut_s10_n17_phi_sueA"] = phi_direct
                row["cut_s10_n17_change_pct"] = \
                    100.0 * abs(phi_direct - (-colA[9])) / abs(-colA[9])
        out["per_od"][key] = row
        print(f"{key}: family {res['n_sets']} sets "
              f"({time.time() - t_start:.0f}s), SUE-A conn "
              f"{row['sue_A_nobg_41n']['conn_range_pct']:.2f}% vs unres "
              f"{row['sue_A_nobg_41n']['unrestricted_pct']:.2f}%",
              flush=True)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"\n[E80] wrote {OUT}")


if __name__ == "__main__":
    main()
