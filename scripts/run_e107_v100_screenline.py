"""e107 (v100): pre-specified straight-line screenline sections.

Motivation (v99 external review, optional supplement): the paper's two
prescribed boundary families are hop balls and free-flow-time balls;
straight geometric screenlines are a different, practically legible
class.  Design fixed by the review:

  * public node coordinates (SiouxFalls_node.tntp);
  * directions pre-fixed at every 5 degrees (36 lines over [0,180));
  * section positions at midpoints between adjacent node projections
    onto each direction's normal;
  * keep partitions that separate the OD endpoints with BOTH induced
    subgraphs connected (the paper's connected-family admissibility);
  * deduplicate node sets;
  * evaluate phi_j(U) = -1_U^T B C~_{:,j} on the FROZEN published
    41-point columns of e86 (no new solves, no training);
  * report per query: boundary count, min, max, negative share.

Direction selection uses geometry only: no attribution value enters
the boundary-generation process.  Longitude is rescaled by
cos(mean latitude) before projection so that 5-degree direction
increments are isotropic in the local plane.

Outputs: results/e107_v100_screenline.json (+ console log).
"""
import json
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments" / "data"
RESULTS = ROOT / "experiments" / "results"
SRC_NPZ = RESULTS / "e86_v84_newod.npz"

ODS_1IDX = [(7, 18), (1, 19), (4, 5), (1, 15), (6, 8), (7, 12)]
DEG_STEP = 5.0


def load_network():
    """Nodes (1-based) with plane coordinates and the undirected graph."""
    nodes, xs, ys = [], [], []
    with open(DATA / "SiouxFalls_node.tntp") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().rstrip(";").split("\t")]
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            v = int(parts[0])
            lon, lat = float(parts[1]), float(parts[2])
            nodes.append(v)
            xs.append(lon)
            ys.append(lat)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # plane projection: equirectangular local frame, isotropic directions
    lat0 = math.radians(ys.mean())
    kx = math.cos(lat0)
    px = (xs - xs.mean()) * kx
    py = ys - ys.mean()
    coords = {v: (px[i], py[i]) for i, v in enumerate(nodes)}

    g = nx.Graph()
    g.add_nodes_from(nodes)
    arcs = []
    with open(DATA / "SiouxFalls_net.tntp") as fh:
        for line in fh:
            if line.startswith("<") or line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.strip().rstrip(";").split("\t")]
            try:
                a, b = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue
            g.add_edge(a, b)  # parallel arcs collapse; connectivity only
            arcs.append((a, b))
    return nodes, coords, g, arcs


def main():
    t0 = time.time()
    nodes, coords, g, arcs = load_network()
    data = np.load(SRC_NPZ, allow_pickle=True)
    BC = data["BC"]  # (24, 6): B C~_{:,j}, frozen published columns
    C = data["C"]
    assert BC.shape == (24, 6) and C.shape == (76, 6)

    # independent recomputation of the node divergence from the directed
    # incidence matrix and the attribution columns (carrier cross-check)
    Bm = np.zeros((24, 76))
    for e, (a, b) in enumerate(arcs):
        Bm[a - 1, e] = -1.0
        Bm[b - 1, e] = +1.0
    BC_recomp = Bm @ C
    bc_gap = float(np.abs(BC_recomp - BC).max())

    idx = {v: i for i, v in enumerate(nodes)}
    # BC rows are node-ID ordered (verified: -BC[s-1,k] reproduces Table 3)
    row = {v: v - 1 for v in nodes}
    xy = np.array([coords[v] for v in nodes])  # (24, 2)

    # carrier check: reproduce Table 3 origin-cut accounts from BC
    phi_s_repro = {f"{s}->{t}": -float(BC[row[s], k])
                   for k, (s, t) in enumerate(ODS_1IDX)}

    degrees = np.arange(0.0, 180.0, DEG_STEP)
    all_nodes = set(nodes)

    def run_family(step_deg):
        """Enumerate straight-line partitions at the given angle grid."""
        degs = np.arange(0.0, 180.0, step_deg)
        out = {}
        for k, (s, t) in enumerate(ODS_1IDX):
            seen = {}
            n_cand = 0
            for deg in degs:
                th = math.radians(deg)
                nvec = np.array([math.cos(th), math.sin(th)])
                proj = {v: float(np.array(coords[v]) @ nvec) for v in nodes}
                sp = np.sort(np.array(list(proj.values())))
                for i in range(len(sp) - 1):
                    if sp[i + 1] <= sp[i]:
                        continue
                    thresh = 0.5 * (sp[i] + sp[i + 1])
                    below = frozenset(v for v in nodes if proj[v] < thresh)
                    # one line, two admissible sides: keep the side containing s
                    for U in (below, frozenset(all_nodes - below)):
                        n_cand += 1
                        if s not in U or t in U:
                            continue
                        if U in seen:
                            continue
                        if not (nx.is_connected(g.subgraph(U))
                                and nx.is_connected(g.subgraph(all_nodes - U))):
                            continue
                        phi_bc = -float(sum(BC[row[v], k] for v in U))
                        phi_rec = -float(BC_recomp[[row[v] for v in U], k].sum())
                        assert abs(phi_bc - phi_rec) < 1e-9, (s, t, deg, phi_bc, phi_rec)
                        seen[U] = phi_bc
            phis = np.array(list(seen.values()))
            n_neg = int((phis < 0).sum())
            out[f"{s}->{t}"] = {
                "origin_s_1idx": s,
                "destination_t_1idx": t,
                "phi_s_published_reproduced": round(phi_s_repro[f"{s}->{t}"], 4),
                "n_directions": int(len(degs)),
                "n_candidates_evaluated": n_cand,
                "n_admissible_dedup": int(len(seen)),
                "phi_min": float(phis.min()),
                "phi_max": float(phis.max()),
                "n_negative": n_neg,
                "negative_share": n_neg / len(phis),
                "argmin_nodes_1idx": sorted(min(seen, key=lambda U: seen[U])),
                "argmax_nodes_1idx": sorted(max(seen, key=lambda U: seen[U])),
            }
        return out

    per_od = run_family(DEG_STEP)
    fine = run_family(1.0)  # angle-grid sensitivity of the negative findings

    out = {
        "protocol": {
            "purpose": ("pre-specified straight-line screenline sections on "
                        "public node coordinates; accounts evaluated on the "
                        "frozen published e86 41-point columns; no boundary "
                        "was selected using attribution values"),
            "source_columns": "e86_v84_newod.npz (setting C, six-query protocol)",
            "coordinates": "SiouxFalls_node.tntp, equirectangular local frame "
                           "(longitude rescaled by cos(mean lat)) so 5-degree "
                           "direction increments are isotropic",
            "directions": "every 5 degrees over [0,180), 36 directions; both "
                          "sides of each line kept when admissible",
            "sections": "thresholds at midpoints between adjacent node "
                        "projections onto the direction normal; partition U = "
                        "{v : v.n < threshold}",
            "admissibility": "s in U, t not in U, U and complement both "
                             "connected on the undirected network (the "
                             "paper's connected family); node sets "
                             "deduplicated across directions and positions",
            "independent_check": ("phi recomputed from the directed incidence "
                                 "matrix times the attribution columns; "
                                 f"max |B C - BC_frozen| = {bc_gap:.3e}, "
                                 "asserted 1e-9 on every retained boundary"),
            "angle_grid_sensitivity": "per_od_fine_1deg: same protocol at a "
                                      "1-degree grid (180 directions)",
            "claims_supported": "which signs and magnitudes straight "
                                "screenline partitions produce; NOT "
                                "representativeness of any planning practice",
        },
        "per_od": per_od,
        "per_od_fine_1deg": {
            od: {kk: vv for kk, vv in rec.items()
                 if kk in ("n_admissible_dedup", "phi_min", "phi_max",
                           "n_negative", "negative_share")}
            for od, rec in fine.items()},
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_path = RESULTS / "e107_v100_screenline.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"written {out_path}")
    for od, rec in per_od.items():
        print(f"{od}: n={rec['n_admissible_dedup']} "
              f"min={rec['phi_min']:.1f} max={rec['phi_max']:.1f} "
              f"neg={rec['n_negative']} ({rec['negative_share']:.1%}) "
              f"phi_s={rec['phi_s_published_reproduced']}")


if __name__ == "__main__":
    main()
