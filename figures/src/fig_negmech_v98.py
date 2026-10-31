"""fig_negmech (v91): the connected minimizing boundary U_- for OD (1,19)
under the six-query full-background protocol (setting C), with the
cross-boundary attribution decomposition.

v91 recoding (external review round 63, item V-2): the map colour and
style now encode the signed CONTRIBUTION TO phi_j,
a_{e,j}(U) = sigma_e * C~_{e,j}  (sigma_e = +1 leaving, -1 entering),
which is exactly the quantity listed in the side panel, so map and
panel speak the same quantity and no on-the-fly sign conversion is
needed.  Arrows keep the true link direction; a POSITIVE attribution
on an ENTERING link now correctly appears red (it lowers the account)
-- the mechanism note moves to the caption.  The map box is enlarged
(0.415 -> 0.50 of figure width) and the side panel prose reduced.

Encoding (house conventions from fig_resp_field / figstyle):
- all 76 arcs drawn; context arcs light grey, internal U_- arcs light
  blue (the induced subgraph is connected -- membership is carried by
  discs and internal arcs, NOT by a filled hull: eight non-member nodes
  fall inside the convex hull of U_-, so a fill would misstate the set);
- the 26 crossing links are arrows (direction = link direction, so
  entering links visibly point into the region); opposite directions of
  the same physical road are separated by a small perpendicular offset
  so both remain visible; width follows the shared square-root scale of
  |C~_{e,j}|, solid blue for a_{e,j} > 0, dashed red for a_{e,j} < 0;
- the five largest negative and three largest positive contributions
  carry badges L1..L8, resolved in the side panel (values in TSTT
  units, signed contribution to phi_j over ALL 26 crossing links);
- origin 1 = solid blue bold disc; destination 19 = hollow bold disc.

Data: experiments/results/e94_v89_divsplit_links.json (part B, frozen
e87 published columns; gates re-verified there).  Node coordinates:
published TNTP file.

v93 (external review round 65, item V-5): the map grows in effective
area, not only in box share -- the figure is taller (3.95 -> 4.30 in)
and the map box aspect now matches the data aspect (0.44 share,
0.965 height), so the network itself is ~10% larger in linear size;
all in-figure text is raised to print at the 8-9 pt level after the
0.85\\linewidth include (origin/destination 10.6 pt source, U_- nodes
9.6, context nodes 8.5, badges 8.8, panel 9.4-10.1, title 10.4), with
discs, arrow heads, and panel spacing scaled to match.
"""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from figstyle import C_BLUE, C_BLUE_L, C_DARK, C_RED, FULL_W, save_pub  # noqa: E402,F401

# v94 fix (visual check): the local Agg rasterizer (matplotlib 3.10.9)
# corrupts a trailing cross-font subscript — a bare "$U_-$" at the end of a
# legend string loses the subscript minus and breaks the U glyph in the PNG
# output only (PDF/SVG render correctly; size, dpi, fontset and hinting are
# all irrelevant). Wrapping the math as "$(U_-)$" — verified to rasterize
# correctly — is used for the two legend strings below.

DATA_DIR = os.path.join(ROOT, "experiments", "data")
E94 = os.path.join(ROOT, "experiments", "results",
                   "e94_v89_divsplit_links.json")

# perpendicular split for opposite directions, in data units
# v97: 0.0022 -> 0.0032 (visual QA round 69: the two opposite-direction
# arrowheads on the 10-15 corridor fused into one blob at the larger
# network scale; a wider split plus longer endpoint shrink separates
# every opposite-direction pair)
SPLIT = 0.0032


def load_nodes():
    xy = {}
    with open(os.path.join(DATA_DIR, "SiouxFalls_node.tntp")) as fh:
        for line in fh:
            parts = [p for p in re.split(r"[\s\t]+", line.strip()) if p]
            if not parts or parts[0] == "Node":
                continue
            xy[int(parts[0]) - 1] = (float(parts[1]), float(parts[2]))
    assert len(xy) == 24
    return xy


def load_edges():
    """76 directed arcs, 0-based (same parser as nets.load_sioux_falls)."""
    edges = []
    with open(os.path.join(DATA_DIR, "SiouxFalls_net.tntp")) as fh:
        started = False
        for line in fh:
            if line.startswith("~"):
                started = True
                continue
            if not started or not line.strip():
                continue
            parts = [p for p in re.split(r"[\s\t]+", line.strip()) if p]
            if parts[-1] == ";":
                parts = parts[:-1]
            edges.append((int(parts[0]) - 1, int(parts[1]) - 1))
    assert len(edges) == 76
    return edges


def lw_of(x, xmax):
    return 1.0 + 2.9 * np.sqrt(x / xmax)


def main() -> None:
    e94 = json.load(open(E94))
    part = e94["part_b_cross_links"]
    xy = load_nodes()
    # legibility nudges: pairs of nodes whose published coordinates sit
    # closer than the sum of their disc radii get a small DRAWN-position
    # displacement so every label pair stays separated (all links follow
    # the drawn position; geometry note kept here, not in the caption)
    xy[8] = (xy[8][0] + 0.0015, xy[8][1] + 0.0031)    # 9 vs 10 corridor
    xy[16] = (xy[16][0] + 0.005, xy[16][1])           # 17 vs 16 / 20
    xy[20] = (xy[20][0] - 0.0012, xy[20][1] - 0.0024) # 21 vs 22
    edges = load_edges()
    U = set(v - 1 for v in part["U_minus_1idx"])
    s_od, t_od = 0, 18                     # OD (1,19), 0-based
    assert s_od in U and t_od not in U

    links = {tuple(r["link_1idx"]): r for r in part["all_links"]}
    neg = part["most_negative_links"]
    pos = part["most_positive_links"]
    badges = ([(tuple(r["link_1idx"]), r) for r in neg] +
              [(tuple(r["link_1idx"]), r) for r in pos])
    badge_ids = {lk: f"L{i + 1}" for i, (lk, _) in enumerate(badges)}
    cmax = max(abs(r["C_total"]) for r in links.values())

    # pairwise clearance gate: every drawn node pair must be at least a
    # disc-pair width apart (r = sqrt(s/pi) points converted to data
    # units via the axis ranges), so enlarged labels can never collide
    def _disc_s(v_):
        if v_ in (s_od, t_od):
            return 230
        return 100 if v_ in U else 54

    DATA_PER_PT = 0.162 / (4.10 * 72.0)   # axes ~4.10 in tall
    for a in sorted(xy):
        for b in sorted(xy):
            if b <= a:
                continue
            ra = np.sqrt(_disc_s(a) / np.pi)
            rb = np.sqrt(_disc_s(b) / np.pi)
            d = np.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
            need = (ra + rb + 2.0) * DATA_PER_PT   # 2 pt daylight
            if d < need:
                print(f"  warn: nodes {a + 1},{b + 1} gap {d:.5f} "
                      f"< {need:.5f} (halo carries the labels)")

    fig = plt.figure(figsize=(FULL_W, 4.30))
    # v93 (external review round 65): map box aspect matches the data
    # aspect (0.162/0.130 = 1.246) so the network fills the box; the
    # side panel keeps the principal-link values and headline numbers
    ax = fig.add_axes([0.005, 0.015, 0.440, 0.965])
    leg = fig.add_axes([0.465, 0.015, 0.530, 0.965])
    leg.axis("off")
    # freeze limits BEFORE any artist: mixed data-coord artists (lines,
    # scatter autoscale) otherwise collapse the limits around the scatter
    # points and render the unclipped text far outside the canvas
    leg.set_xlim(0.0, 1.0)
    leg.set_ylim(0.0, 1.0)

    # --- base network ---
    for (u, v) in edges:
        if (u + 1, v + 1) in links:
            continue
        x1, y1 = xy[u]
        x2, y2 = xy[v]
        inside = u in U and v in U
        ax.plot([x1, x2], [y1, y2],
                color=C_BLUE_L if inside else "#E4E6E8",
                lw=1.9 if inside else 0.65,
                zorder=1.6 if inside else 1.2,
                solid_capstyle="round")

    # --- crossing links: arrows, width by |C~|, colour/style by the
    #     signed contribution a = sigma * C~ (same sign as the panel) ---
    for (u1, v1), r in links.items():
        u, v = u1 - 1, v1 - 1
        cval = r["C_total"]
        acon = r["signed_contribution"]       # sigma * C~: panel quantity
        entering = (v in U) and (u not in U)
        x1, y1 = xy[u]
        x2, y2 = xy[v]
        # unit perpendicular on the UNDIRECTED endpoint order, so the
        # two directions of the same road receive opposite offsets
        lo, hi = (u, v) if u < v else (v, u)
        dxu, dyu = xy[hi][0] - xy[lo][0], xy[hi][1] - xy[lo][1]
        Lu = np.hypot(dxu, dyu)
        px, py = -dyu / Lu, dxu / Lu
        sgn = 1.0 if entering else -1.0
        ox, oy = sgn * SPLIT * px, sgn * SPLIT * py
        # v97: 9/11 -> 12.5/14.5 endpoint shrink so no arrowhead enters
        # the endpoint disc (visual QA: node 16's ring was pierced);
        # capped at 28% of the drawn arc length so short corridors keep
        # a visible shaft instead of degenerating to a bare teardrop
        # head (visual QA recheck round 69)
        base_shrink = 14.5 if (u1, v1) in badge_ids else 12.5
        p1d = ax.transData.transform((x1 + ox, y1 + oy))
        p2d = ax.transData.transform((x2 + ox, y2 + oy))
        lpt = np.hypot(p2d[0] - p1d[0], p2d[1] - p1d[1]) * 72.0 / fig.dpi
        shrink = min(base_shrink, 0.28 * lpt)
        arr = FancyArrowPatch(
            (x1 + ox, y1 + oy), (x2 + ox, y2 + oy),
            arrowstyle="-|>", mutation_scale=11.0,
            lw=lw_of(abs(cval), cmax),
            color=C_BLUE if acon > 0 else C_RED,
            linestyle="-" if acon > 0 else (0, (4.0, 2.2)),
            shrinkA=shrink, shrinkB=shrink,
            zorder=2.8 if (u1, v1) in badge_ids else 2.2,
            capstyle="round")
        ax.add_patch(arr)

    # --- badges on the eight principal links (offsets tuned off-discs) ---
    # v97: (10,16) raised to (5, 20) so the badge no longer cuts its own
    # arrow into two pieces; (10,15) nudged further off the fused
    # arrowhead pair above node 15 (visual QA round 69)
    dxs = {(10, 15): (-20, 3), (5, 9): (-15, 8), (13, 24): (1, -12),
           (14, 15): (2, -12.5), (10, 16): (5, 20), (9, 10): (-21, 2),
           (15, 19): (4, -13), (1, 3): (-13.5, 5)}
    for (lk, r) in badges:
        u, v = lk[0] - 1, lk[1] - 1
        mx, my = 0.5 * (xy[u][0] + xy[v][0]), 0.5 * (xy[u][1] + xy[v][1])
        dxdy = dxs.get(lk, (0, 0))
        ax.annotate(badge_ids[lk], (mx, my),
                    textcoords="offset points", xytext=dxdy,
                    fontsize=9.6, fontweight="bold", color="white",
                    ha="center", va="center", zorder=5.5,
                    bbox=dict(boxstyle="round,pad=0.18",
                              fc=C_DARK, ec="none"))

    # --- nodes ---
    for v_ in sorted(xy):
        x, y = xy[v_]
        if v_ == s_od:
            ax.scatter([x], [y], s=230, c=C_BLUE, edgecolors=C_DARK,
                       linewidths=2.0, zorder=6)
            ax.text(x, y, str(v_ + 1), fontsize=10.2, fontweight="bold",
                    color="white", ha="center", va="center", zorder=6.5)
        elif v_ == t_od:
            ax.scatter([x], [y], s=230, c="white", edgecolors=C_DARK,
                       linewidths=2.0, zorder=6)
            ax.text(x, y, str(v_ + 1), fontsize=10.2, fontweight="bold",
                    color=C_DARK, ha="center", va="center", zorder=6.5)
        elif v_ in U:
            ax.scatter([x], [y], s=100, c=C_BLUE_L, edgecolors=C_BLUE,
                       linewidths=1.2, zorder=4.5)
            ax.text(x, y, str(v_ + 1), fontsize=10.2, color=C_DARK,
                    ha="center", va="center", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.4,
                                                foreground="white")])
        else:
            ax.scatter([x], [y], s=54, c="#ECEDEE", edgecolors="#B9BCC0",
                       linewidths=0.8, zorder=3.5)
            ax.text(x, y, str(v_ + 1), fontsize=9.2, color="#73777B",
                    ha="center", va="center", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")])

    ax.set_aspect("equal", adjustable="box")
    # v97 (external review round 69, item VII-3): the network itself is
    # enlarged by trimming whitespace, not by raising figure height --
    # the axis spans were 0.130 x 0.162 against a 0.100 x 0.122 data
    # extent (~23% padding); the reduced spans keep ~0.005 x and ~0.007 y
    # clearance for badges/halos, so no figure or page-budget change.
    ax.set_xlim(-96.8020, -96.6850)
    ax.set_ylim(43.4830, 43.6210)
    ax.axis("off")
    ax.set_title("OD $(1,19)$: $U_-$ and its crossing contributions",
                 fontsize=10.4, pad=3, color=C_DARK)

    # --- side panel (v94, external review round 66: the long header
    #     prose -- signed contribution to phi_j, top-five/three
    #     selection, in/out marks -- moved to the LaTeX caption; the
    #     panel keeps one header line, the principal-link values, the
    #     scales, and the two headline numbers, all slightly larger) ---
    X0, Y0 = 0.0, 0.985
    leg.text(X0, Y0, "Principal crossing links (TSTT units):",
             fontsize=10.0, color=C_DARK, va="top")
    yy = Y0 - 0.058          # below the one-line header
    for (lk, r) in badges:
        neg_flag = r["signed_contribution"] < 0
        dir_txt = " (in)" if r["direction"] == "in" else " (out)"
        val_txt = format(r["signed_contribution"], "+,.0f")
        leg.text(X0 + 0.010, yy,
                 f"{badge_ids[lk]}  {lk[0]}$\\to${lk[1]}{dir_txt}"
                 f"   {val_txt}",
                 fontsize=9.8,
                 color=C_RED if neg_flag else C_BLUE, va="center",
                 family="DejaVu Sans")
        yy -= 0.050
    yy -= 0.034
    for ref in (2000, 9000):
        leg.add_line(Line2D([X0, X0 + 0.17], [yy, yy], color=C_DARK,
                            lw=lw_of(ref, cmax), solid_capstyle="round"))
        leg.text(X0 + 0.20, yy, f"width $|\\widetilde C| = {ref:,}$",
                 fontsize=9.8, color=C_DARK, va="center")
        yy -= 0.050
    yy -= 0.018
    leg.scatter([X0 + 0.010], [yy], s=118, c=C_BLUE_L, edgecolors=C_BLUE,
                linewidths=1.2, clip_on=False)
    leg.text(X0 + 0.045, yy, "node in $(U_-)$", fontsize=9.8,
             color=C_DARK, va="center")
    yy -= 0.050
    leg.scatter([X0 + 0.010], [yy], s=62, c="#ECEDEE", edgecolors="#B9BCC0",
                linewidths=0.8, clip_on=False)
    leg.text(X0 + 0.045, yy, "node outside $(U_-)$", fontsize=9.8,
             color=C_DARK, va="center")

    yy -= 0.070
    phi_txt = format(part["phi_U_minus"], ",.1f")
    leg.text(X0, yy,
             f"$\\phi_j(U_-) = {phi_txt}$ = entering $-21,368.8$"
             " + leaving $+7,867.3$",
             fontsize=10.4, color=C_RED, va="center")
    yy -= 0.054
    leg.text(X0, yy, "system contribution $C_j = +50{,}097.7 > 0$",
             fontsize=10.4, color=C_BLUE, va="center")

    save_pub(fig, "fig_negmech_v98")
    print("fig_negmech written")


if __name__ == "__main__":
    main()
