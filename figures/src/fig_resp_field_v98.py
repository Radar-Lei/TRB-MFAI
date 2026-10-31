"""v31 redesign of the endpoint-responsibility-field figure (main-text Fig. 6).

Why redesign (v31 external review): the v30 encoding spent a full-network
map on a field whose node channel is exactly zero in panels (a)/(b)
(0/24 and 0/24 nonzero nodes) while the active panel carries a single
nonzero edge -- three near-identical grey maps that failed to carry the
balanced / direction-balanced / unbalanced contrast. The v31 encoding
keeps every number from fig_resp_field_v30.npz unchanged but makes the
sparsity itself legible:
- shared edge-encoding legend (line-width reference ticks 0.5 / 1.0 / 2.5
  of |C_e|/|beta_can| on the square-root scale; solid blue = positive,
  dashed red = negative) -- the v30 figure never legend-defined the edge
  channels;
- per-panel footer names the two core readouts (direction error,
  reference-scale mismatch) so a zero channel reads as an explicit
  zero, not as an empty map (v34: the max node imbalance row moved
  out -- panel (c) annotates its largest node imbalances in place);
- panel (c) annotates its five nonzero nodes directly (node number and
  signed value); panel (b) gives the single active edge a halo so the
  one-edge field is unmissable;
- OD endpoints switch from square/triangle to labelled discs, matching
  the node-number convention of Fig. 4 (cross-figure unification);
- colorbar label disambiguated (v30 read as an unbalanced parenthesis).

Data: experiments/results/fig_resp_field_v30.npz (deterministic inference
from the frozen M3|norm|raw|0 checkpoint; values reproduce main-text
Table 5). Node coordinates: published TNTP file.

v35 (external review, 88/100): footer readouts enlarged 7.5 -> 9.3 pt
(~8 pt at the 0.858 TeX include scale) and the canvas grown 3.80 ->
4.15 in so the map bodies enlarge with them; the accounting-cut
markers switch from red rings to BLACK solid (U) / dashed (U') ellipse
rings -- red previously did triple duty (negative attributions,
negative node imbalance, cut membership); the inset zoom frame and
the U' label drop their red as well.

v36 (external review, 91/100): the cut labels spell out their sets in
figure -- U -> U={10}, U' -> U'={10,17} (review: the U' tag next to
node 17 could read as {17} alone); endpoint numbers 8.6 -> 9.8 pt and
panel-(c) node annotations 7.4 -> 8.4 pt with the endpoint discs and
zoom padding tuned so the key nodes 10/16/17 read larger (review:
enlarge the local network and the key node marks, esp. in the TSTT
panel).

v37 (external review, 88/100): canvas 4.15 -> 4.60 in and the map band
widened (bottom 0.27 -> 0.25, top 0.90 -> 0.91) so the response-region
maps enlarge ~20-30% into the p16 whitespace (review: enlarge the
TSTT panel); the full-network context inset shrinks to 2/3 area and
moves higher (0.30x0.375 @ y.585 -> 0.20x0.26 @ y.66) -- Fig. 4
already carries the topology, the inset only locates the zoom (review:
weaken the tiny inset); the U' label gains a third line stating the
net cut-flux difference directly (net +0.29|beta_can|), so the
U -> U' comparison no longer requires the caption (review: mark the
net difference between the two accounting cuts in panel).

v39 (external review, 90/100): panel (c) widened to ~half the figure
width (width ratios 0.85 : 0.85 : 2.0, same data scale and vertical
extent, (c) showing a wider x-window of the same response region);
the full-network inset REMOVED entirely (Fig. 4 carries the topology;
review: the tiny inset was unreadable and added visual load); panel
(c) keeps numeric annotations on nodes 10/16/17 only -- nodes 15/19
stay colour-coded without printed values (review: visual attention to
the OD endpoints and the cut-changing node 17); the U' block enlarged
(8.2 -> 9.6 pt) and its value line restated in reference-scale units
(+0.29 |beta_can^j|) as the panel's most prominent reading.
v40 (external review, 89/100): panel (c) decluttered -- the ONLY
in-panel numeric annotation is node 17's ("17 +0.29", the cut-changing
value); the OD endpoints 10/16 keep their bold endpoint numbers and
drop the value texts (the -0.16 reading duplicated the colour/size
channels); the rings stay as pure visual marks and the U/U'/cut-flux
statement moves OUT of the panel into a third footer line under (c)
(an outside annotation strip). The v37/v38 in-panel collision-scan
machinery for the U' block is deleted with it.
v41 (external review, 89/100): node 17's annotation keeps the NUMBER
only (the +0.29 value lives solely in the footer's Delta-cut-flux
line, its former in-panel duplicate removed); panel (b)'s "target
arc" leader line removed (the haloed edge is the only strongly
coloured object there); U/U' ring radii re-spaced outward from the
node discs and responsibility edges.
v42 (external review, 89/100, closing round): (1) the colorbar label
renames the channel "OD-direction residual" (the old "off-OD
imbalance" read as excluding the OD endpoints spatially -- nodes
10/16 legitimately carry the residual); (2) the footer's
U/U' set-definition line is dropped (the sets stay defined in the
TeX caption; on-panel only the compact Delta-cut-flux reading
remains); (3) the colorbar gains explicit ticks at the range
endpoints (+-dmax ~ +-0.29) so node 17's annotated value maps onto
the scale without extrapolating the 0.2 major ticks.
v43 (external review, 90/100, convergence round): the response-region
zoom is TIGHTENED to the informative cluster itself -- the key set
drops the one-hop neighbours of the OD endpoints (they tripled the
window for grey-silhouette context), and the per-side clearance is
sized to the actual marks (node-10 rings 13.4 pt left, node-17 label
+ ring right, endpoint discs top/bottom). Width ratios
0.85:0.85:2.0 -> 1.15:1.15:1.5 so panel (c) keeps its dominant width
while its x-widening factor drops 2.35x -> 1.30x (the v42 geometry
spent the widening on grey periphery: the 10/15/16/17/19 cluster
filled 27%/16% of (a) and 11%/16% of (c); now 57%/45% and
44%/45% -- a ~3.9x scale-up in (c), review: "one should not need to
zoom the PDF to tell U from U' and see why node 17 changes the cut
flux"). All v41/v42 frozen conventions (ring radii, occlusion zorder
3.4, footer rows, colorbar ticks, node-17-only annotation) are
unchanged.

v49 (external review, 90/100, closing round): KEY-ANNOTATION ENLARGE-
MENT, geometry untouched (same canvas, bands, ratios, zoom windows --
the collision asserts and ring placements stay valid). Panel titles
9.5 -> 10.5 pt; endpoint numbers 9.8 -> 11.0 with discs 125 -> 145
pt^2; node-17 label 8.4 -> 9.6 pt; U/U' ring-side set tags 8.6 ->
10.0 pt; ring radii +0.8 pt (11.2/14.2/12.4, still zorder 3.4 behind
the discs); footer ladder now starts at 11.0 pt (floor 7.8 unchanged)
and the cut row drops its "U"/"U'" prefixes (the v48 in-panel ring
tags already name the sets), which buys ~10.4 pt for the row;
colorbar label 8.0 -> 9.0, ticks 7.6 -> 8.4; legend 7.8 -> 8.6 /
7.4 -> 8.2 (review: enlarge nodes 10/16/17 and the cut net-value
statements -- Fig. 6 carries the paper's core transportation result).

v50 (external review, 89/100): BOTH set tags re-placed INSIDE panel
(c) with short leader lines (review: U={10} ran left into panel (b)'s
side of the figure, U'={10,17} ran right into the colorbar and its
tail was clipped). U={10} sits in the blank lower-left corner below
node 10 (leader up to its rings); U'={10,17} sits in the blank top
band over the inter-column corridor (leader down to node 17's dashed
ring; a straight approach from above node 17 is impossible -- node 16
sits directly above node 17, and the right approach is blocked by the
"17" number, the lower-right by the "19" number). Both placements are
gated by assertions: the label boxes must lie fully inside the panel
(c) axes box, clear of every placed text box, and their leader
segments must miss every placed box. Panel (a)/(b) footer label
"ref-scale mismatch" renamed "diff. from reference" and (c)'s
"reference-scale mismatch" renamed "relative difference from
reference" (review: "mismatch" suggests an estimation error; the
reference scale is not a true scale) -- values unchanged.

v51 (external review, 89/100): the numeric footer rows are replaced
by per-panel VERDICTS -- (a) "OD-balanced", (b) "OD-balanced," +
"circulation-dependent scale" (two ladder rows), (c)
"boundary-dependent account" + the cut values + the 24.8%
difference row. The direction-error / diff.-from-reference numbers
move to the TeX caption and Table 4 (review: the panels' first-layer
reading should be the balance verdict; the reference scale is
auxiliary and was headlining two of the three panels). Geometry,
channels, rings, tags, and legend unchanged.
v53 (external review, 88/100, twentieth round): print legibility --
node 10/16/17 numbers and the two cut set-tags enlarged one notch,
cut rings and the background silhouette drawn heavier (review: the
dashed cuts and the light-grey network may fade in print); the beta^can
glyphs in the shared legend and the colorbar label renamed beta^ref
with the main text.

v57 (external review, 91/100, minor-revision round): PRINT-SAFE NODE
SIGN -- the node residual's sign now also encodes in marker FILL
(filled = positive, hollow with coloured rim = negative; disc size
keeps |residual| in both cases), so the channel survives greyscale
printing (review: edge signs had solid/dashed, node signs only had
red/blue). The shared legend row legend-defines the fill channel
(v31 convention: every channel is legend-defined). Geometry,
collision asserts, rings, tags, footers unchanged.

v61 (external review, 88/100, twenty-seventh round): NORMALIZATION
SWITCHED to the initial cut value -- every panel normalizes its
channels by |phi_j({s_j})| (the oriented cut value at the origin
singleton, computed from the same frozen C records: uniform 6.160e3,
active 3.834e3, TSTT 5.767e4), replacing |beta^ref_j|. The node
values therefore read directly as fractions of the initial cut:
node 17's residual is +0.248, exactly the 24.8% cut change the
footer states (review: the auxiliary reference scale should not
carry the figure's normalization; the interior-node values should
read as cut-change proportions). Legend ticks 0.5/1.0/2.5 ->
0.25/0.5/1.0 (the new global edge-scale max is the active panel's
1.0); colorbar endpoints +-0.29 -> +-0.25. All frozen record values
unchanged; geometry, collision machinery, rings, tags, footers
untouched.

v65 (external review, 90/100, thirty-first round): U'={10,17} now
ONE common dashed outline -- the convex hull of the 14.2-pt circle
at node 10 and the 12.4-pt circle at node 17, replacing the two
separate dashed rings -- so the cut reads as a moved network
boundary rather than two highlighted nodes; node 16 stays outside
by construction (61 pt off the 10-17 axis vs. a ~14-pt hull
half-width). The U solid ring, radii, window sizing, set-tag
placement gates, and all frozen record values unchanged.

v69 (external review, 91/100, thirty-fifth round): (1) negative OD
endpoints gain an inner residual COLOUR ring inside the dark endpoint
contour -- the dark rim alone hid the residual-magnitude colour on
the endpoints (panel (c): both OD endpoints read -0.134 of the
initial cut, not near-zero); the inner ring carries the interior
nodes' coloured-rim channel while the dark contour keeps the
endpoint identity. (2) The footer's middle cut-value row drops;
the two cut values move to the TeX caption. Geometry, rings, set
tags, collision gates, and all frozen record values unchanged.

v71 (external review, 91/100, thirty-seventh round): the two cut
values move from the TeX caption INTO the panel as second lines of
the U/U' set tags -- the reader sees phi_j(U)=+5.77e4 and
phi_j(U')=+4.34e4 directly beside the rings they belong to (review:
the key numbers lived only in the caption). The footer keeps the
24.8% difference row; the caption drops the numeric duplication.
BCn/netU/netUp are computed once, before the ring block, so the tag
strings carry in-script provenance from the frozen record. All
placement gates (inside-panel, box clearance, leader-miss) still
assert on the enlarged two-line boxes.

v77 (external review, 89/100, forty-eighth round): (1) canvas height
4.60 -> 5.05 in at the same band fractions, so the figure occupies
more vertical space at its width=linewidth TeX include (fonts are
absolute pt, so the include scale and all pt sizes are unchanged);
(2) key annotations one notch larger -- endpoint numbers 11.6 -> 12.6,
node-17 label 10.4 -> 11.2, U/U' set tags 10.8 -> 11.6 (review:
enlarge node numbers 10/16/17 and the cut-value labels); (3) the two
cut MARKS join the halo as legend-defined -- a third legend line
under the shared band names the solid ring (U) and the dashed outline
(U') in-figure, so the ring styles no longer live only in the TeX
caption. Geometry, channels, rings, footers, and all frozen record
values unchanged; the collision gates re-assert on the new sizes.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.transforms import Bbox
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
from figstyle import C_BLUE, C_RED, C_DARK, save_pub, FULL_W

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from nets import DATA_DIR  # noqa: E402

GRAY_EDGE = "#C9CED4"   # v31: lighter silhouette than v30's #B9BEC4
# v77: taller canvas so the figure occupies more vertical space at the
# width=linewidth TeX include (band fractions unchanged; pt sizes are
# absolute, so nothing rescales)
HEIGHT = 5.05

PANELS = [
    # v57 (external review, 91/100): "Uniform control" follows the
    # outcome rename -> "uniform-weight outcome" (panel-title short
    # form "Uniform weight", as in Tables 4/S7)
    ("uniform", "(a) Uniform weight"),
    ("active", "(b) Active link"),
    ("tstt", "(c) TSTT"),
]

# candidate label offsets for panel (c), generated as a direction fan
# (preferred angles first) at three push-off distances, tried in order
# until one clears every placed text box AND its leader segment
# misses every box (QA rounds 3-5: pairwise overlap, then a leader
# crossing the bold OD endpoint number -- both failure classes are
# now gated by assertion, not by eyeballing). 0-based node keys.
import math as _math


def _fan(angles_deg, dists=(18, 27, 36)):
    out = []
    for d in dists:
        for a in angles_deg:
            out.append((round(d * _math.cos(_math.radians(a))),
                        round(d * _math.sin(_math.radians(a)))))
    return out


CAND_OFFS = {
    16: _fan([0, -25, 25]),                                  # node 17: right
    15: _fan([90, 65, 115, 200, 160, 250, 180, 290, 220]),   # node 16: up first
    9:  _fan([180, 155, 205, 135]),                          # node 10: left
    14: _fan([200, 250, 220, 160]),                          # node 15: left-down
    18: _fan([305, 335, 270, 250]),                          # node 19: right-down
}


def _seg_hits_bbox(p0, p1, bb, pad=6.0):
    """Liang-Barsky: does segment p0->p1 cross bbox (padded)? Display px."""
    x0, y0 = p0
    x1, y1 = p1
    bx0, by0, bx1, by1 = (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad,
                          bb.y1 + pad)
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - bx0), (dx, bx1 - x0),
                 (-dy, y0 - by0), (dy, by1 - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


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


def lw_of(x, xmax):
    """Shared square-root width scale, capped at 3.1 pt (v31: 3.7 -> 3.1)."""
    return 0.8 + 2.3 * np.sqrt(x / xmax)


def main() -> None:
    data = np.load(os.path.join(ROOT, "experiments", "results",
                                "fig_resp_field_v30.npz"))
    xy = load_nodes()
    edges = [tuple(e) for e in data["edges"]]
    edge_set = set(edges)
    s, t = int(data["ods"][data["od_j"]][0]), int(data["ods"][data["od_j"]][1])
    e_act = int(data["e_act"])

    # v61 (external review, 88/100): per-panel INITIAL CUT VALUE
    # |phi_j({s_j})| = |1_{s}^T(-B C~_{:,j})| -- the oriented cut value
    # at the origin singleton, computed from the same frozen C records
    # (uniform 6.160e3, active 3.834e3, TSTT 5.767e4). It replaces
    # |beta^ref_j| as the figure's normalization denominator, so node
    # values read directly as fractions of the initial cut (node 17:
    # +0.248 = the 24.8% cut change of the footer).
    phi0 = {}
    for n, _ in PANELS:
        bcn = np.zeros(len(xy))
        for (u, v), c in zip(edges, data[f"{n}_C"].astype(float)):
            bcn[u] += c
            bcn[v] -= c
        phi0[n] = abs(float(bcn[s]))

    div_cmap = LinearSegmentedColormap.from_list(
        "offod", [C_RED, "#EFCFC9", "#E8E8E4", "#C9D9EC", C_BLUE])
    dmax = max(float(np.abs(data[f"{n}_d"]).max() / phi0[n])
               for n, _ in PANELS)
    norm = Normalize(vmin=-dmax, vmax=dmax)

    # shared geographic extent from the node file (TNTP coordinates are
    # longitude/latitude -- do NOT hand-pick axis limits), 6% padding
    xs_all = [p[0] for p in xy.values()]
    ys_all = [p[1] for p in xy.values()]
    padx, pady = 0.06 * (max(xs_all) - min(xs_all)), 0.06 * (max(ys_all) - min(ys_all))
    EXTENT = (min(xs_all) - padx, max(xs_all) + padx,
              min(ys_all) - pady, max(ys_all) + pady)

    # v32: the response region -- the OD endpoints, the five largest
    # off-OD imbalances, and their one-hop neighbours -- becomes the
    # panel body (external review: the full-network panels spent most
    # of their area on a light-grey silhouette away from nodes
    # 10/16/17); the full network survives as a small context inset.
    # v43: the neighbour shell is DROPPED -- the informative cluster
    # (OD endpoints + the nonzero-residual nodes, paper nodes
    # 10/15/16/17/9..19) is the panel body, and the per-side
    # clearance is sized to the marks that must clear the window
    # edge: left = node-10's outer U' ring (13.4 pt), right = node
    # 17's number label and its ring, top/bottom = endpoint discs
    # plus a thin silhouette margin.
    AXY = 1.0 / np.cos(np.deg2rad(43.55))
    # v61: zoom-key threshold follows the phi normalization
    bkey = phi0["tstt"]
    key = ({s, t}
           | {v for v in xy
              if abs(float(data["tstt_d"][v])) / bkey >= 5e-3})
    kx = [xy[v][0] for v in key]
    ky = [xy[v][1] for v in key]
    cdx, cdy = max(kx) - min(kx), max(ky) - min(ky)
    zpxl, zpxr = 0.27 * cdx, 0.47 * cdx
    zpyt, zpyb = 0.26 * cdy, 0.20 * cdy
    ZOOM = (min(kx) - zpxl, max(kx) + zpxr,
            min(ky) - zpyb, max(ky) + zpyt)
    # v39: UNEQUAL panels -- the aspect target is set for the narrow
    # (a)/(b) boxes; panel (c) then widens its x-window by the box
    # width ratio (same data scale, same vertical extent, so the three
    # panels stay directly comparable). y is compressed by AXY through
    # set_aspect below. QA round 1 of v39: because the boxes now match
    # the data aspect exactly (no apply_aspect shrink), the transAxes
    # footers hang from the FULL band bottom -- the map band is lifted
    # (bottom 0.26 -> 0.33) and the legend band lowered and slimmed
    # (0.155 h0.13 -> 0.09 h0.10) so footers, legend, and maps occupy
    # disjoint bands; wspace 0.02 -> 0.05 separates the (a)/(b) titles.
    # v43: ratios 0.85:0.85:2.0 -> 1.15:1.15:1.5 -- (c) stays the
    # dominant panel but its x-widening factor drops 2.35x -> 1.30x,
    # so the widened window no longer re-admits the grey periphery the
    # tightened zoom just removed.
    RATIOS = (1.15, 1.15, 1.5)
    BAND = (0.33, 0.91)                            # map band, fig frac
    HG = HEIGHT * (BAND[1] - BAND[0])             # map band height, in
    total_w = FULL_W * (0.873 - 0.01)
    w_mean = total_w / (sum(RATIOS) + 2 * 0.05)    # wspace=0.05 of mean
    WA = w_mean * RATIOS[0]                        # (a)/(b) box width
    target = WA / HG                               # panel width / height
    dx = ZOOM[1] - ZOOM[0]
    dyv = (ZOOM[3] - ZOOM[2]) * AXY
    if dx / dyv < target:
        g = (dyv * target - dx) / 2.0
        ZOOM = (ZOOM[0] - g, ZOOM[1] + g, ZOOM[2], ZOOM[3])
    else:
        g = (dx / target - dyv) / 2.0 / AXY
        ZOOM = (ZOOM[0], ZOOM[1], ZOOM[2] - g, ZOOM[3] + g)
    XC = 0.5 * (ZOOM[0] + ZOOM[1])
    HWX = 0.5 * (ZOOM[1] - ZOOM[0])

    fig, axes = plt.subplots(
        1, 3, figsize=(FULL_W, HEIGHT),
        gridspec_kw=dict(width_ratios=list(RATIOS), wspace=0.05,
                         bottom=BAND[0], top=BAND[1], left=0.01,
                         right=0.873))
    cax = fig.add_axes([0.907, 0.42, 0.013, 0.44])

    # shared square-root width scale: v61 global max of |C|/phi0 = 1.0
    # (the active panel's selected arc; the 2.774 beta^ref reading
    # moves to the supplement's records with the main-text trim)
    xmax = max(float(np.abs(data[f"{n}_C"]).max() / phi0[n])
               for n, _ in PANELS)

    for ax, (name, title) in zip(axes, PANELS):
        phi = phi0[name]
        Cn = data[f"{name}_C"] / phi
        dn = data[f"{name}_d"] / phi
        # v32 QA2: fix the data limits BEFORE any drawing -- the
        # annotation collision tests transform node positions to
        # pixels, and a stale auto-limit frame made every collision
        # box land in the wrong place (latent since v31; surfaced by
        # the zoomed layout)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(43.55)))
        if name == "tstt":
            # v39: widen (c) by the box ratio -- same scale/height, a
            # wider x-window of the same response region
            f = RATIOS[2] / RATIOS[0]
            ax.set_xlim(XC - f * HWX, XC + f * HWX)
        else:
            ax.set_xlim(ZOOM[0], ZOOM[1])
        ax.set_ylim(ZOOM[2], ZOOM[3])
        ax.axis("off")
        # v37 QA round 3: finalize the aspect-adjusted axes box NOW, so
        # every later window-extent measurement (annotation collision
        # set, inset, net-label scan) shares ONE transform -- applying
        # it mid-panel left the already-recorded boxes ~26 display px
        # stale and let the net label land on node 19's annotation
        ax.apply_aspect()

        for k, (u, v) in enumerate(edges):
            x = abs(float(Cn[k]))
            if x < 1e-3:
                # keep the unused-network silhouette visible in every
                # panel (v30 QA round 3)
                ax.add_patch(FancyArrowPatch(
                    xy[u], xy[v], arrowstyle="-", mutation_scale=5.0,
                    connectionstyle=(f"arc3,rad={0.12 if (v, u) in edge_set else 0.0}"),
                    color=GRAY_EDGE, lw=0.85, alpha=0.95, zorder=1.5))
                continue
            # v31: halo under the single active edge so the one-hot field
            # of panel (b) is unmissable (v30 review: one thick arc in a
            # grey map read as an artefact, not as the message)
            if name == "active" and k == e_act:
                # v31 QA2: shrink so both ends clear the large endpoint
                # discs (they used to cover the halo entirely)
                ax.add_patch(FancyArrowPatch(
                    xy[u], xy[v], arrowstyle="-", mutation_scale=9.0,
                    connectionstyle=(f"arc3,rad={0.12 if (v, u) in edge_set else 0.0}"),
                    shrinkA=10.0, shrinkB=10.0,
                    color="#F3D9A4", lw=lw_of(x, xmax) + 3.2, alpha=0.9,
                    zorder=2.2))
            lw = lw_of(x, xmax)
            ax.add_patch(FancyArrowPatch(
                xy[u], xy[v], arrowstyle="-|>", mutation_scale=6.5,
                connectionstyle=f"arc3,rad={0.12 if (v, u) in edge_set else 0.0}",
                shrinkA=1.5, shrinkB=1.5,
                color=C_BLUE if Cn[k] > 0 else C_RED,
                lw=lw, zorder=2.5,
                linestyle="-" if Cn[k] > 0 else (0, (3.2, 1.6))))

        # v31 QA2: sized so panel (c)'s field-on-nodes reads against (a)'s
        # bare skeleton without reading the caption. v32: cap 500 -> 360
        # (vision QA: at 500 the node-17 and node-16 discs physically
        # overlapped -- the two nodes are 0.0054 deg apart)
        scal = 360.0 / max(dmax, 1e-12)
        # interior nodes: colour = imbalance, disc size = |imbalance|
        # (thin grey rim keeps zero-imbalance nodes visible as skeleton)
        # v57: negative residuals draw HOLLOW (coloured rim) so the
        # sign survives greyscale printing; |residual| stays in the
        # disc size for both fills
        for v, (px, py) in xy.items():
            if v in (s, t):
                continue
            val = float(dn[v])
            if val < -1e-6:
                ax.scatter(px, py,
                           s=min(abs(val), 0.6 * dmax) * scal + 9.0,
                           facecolors="none",
                           edgecolors=[div_cmap(norm(val))],
                           linewidths=1.5, zorder=3.5)
            else:
                ax.scatter(px, py, s=min(abs(val), 0.6 * dmax) * scal + 9.0,
                           c=[val], cmap=div_cmap, norm=norm,
                           edgecolors="#9AA0A6", linewidths=0.35, zorder=3.5)
        # OD endpoints: labelled discs (node numbers, Fig. 4 convention),
        # colour channel identical to interior nodes (in panel (c) the
        # most negative residuals sit exactly on the endpoints).
        # v31 QA5: white-stroke halo replaces the opaque bbox (it used
        # to erase background arcs behind the number), and in panel (c)
        # the rendered label boxes seed the annotation collision set.
        if name == "tstt":
            renderer = fig.canvas.get_renderer()
            ep_boxes = []
            # v32 QA2: seed the collision set with the big discs themselves
            # (the node-16 label's halo used to bite the node-17 disc);
            # disc radius in px from the scatter area (pt^2). Boxes carry
            # an owner so a node's own disc never blocks its own label
            # or leader (round-2 build failure).
            for v in xy:
                sv = min(abs(float(dn[v])), 0.6 * dmax) * scal + 9.0 \
                    if v not in (s, t) else 160.0
                if abs(float(dn[v])) < 0.5 * dmax and v not in (s, t):
                    continue
                cx_, cy_ = ax.transData.transform(xy[v])
                r_px = (sv / np.pi) ** 0.5 * fig.dpi / 72.0 + 2.0
                ep_boxes.append((Bbox([[cx_ - r_px, cy_ - r_px],
                                       [cx_ + r_px, cy_ + r_px]]), v))
        for v in (s, t):
            px, py = xy[v]
            # v57: endpoint discs take the same fill channel (hollow
            # rim = negative residual) as the interior nodes
            # v69 (external review, 91/100, thirty-fifth round):
            # negative endpoints gain an inner residual COLOUR ring
            # inside the dark contour -- the dark rim alone hid the
            # residual-magnitude colour on the OD endpoints (both
            # read -0.134 in panel (c), not near-zero); the dark
            # outer contour keeps the endpoint identity, the inner
            # coloured rim carries the same negative-value channel
            # as the interior nodes' coloured rims (s=160 -> r 7.1 pt;
            # s=64 -> r 4.5 pt with a 3.0 pt band = 3.0-6.0 pt,
            # clear of the dark contour's 6.2 pt inner edge)
            if float(dn[v]) < -1e-6:
                ax.scatter([px], [py], s=160, facecolors="none",
                           edgecolors=C_DARK, linewidths=1.9, zorder=5)
                ax.scatter([px], [py], s=64, facecolors="none",
                           edgecolors=[div_cmap(norm(float(dn[v])))],
                           linewidths=3.0, zorder=5)
            else:
                ax.scatter([px], [py], s=160, c=[float(dn[v])],
                           cmap=div_cmap, norm=norm,
                           edgecolors=C_DARK, linewidths=1.3, zorder=5)
            ann_ep = ax.annotate(str(v + 1), (px, py), xytext=(10, 8),
                                 textcoords="offset points", fontsize=12.6,
                                 fontweight="bold", color=C_DARK, zorder=6,
                                 ha="left",
                                 path_effects=[pe.withStroke(
                                     linewidth=2.8, foreground="white")])
            if name == "tstt":
                ep_boxes.append((ann_ep.get_window_extent(renderer), v))
        # v31: panel (c) states its five nonzero nodes in place -- the
        # node channel is nonzero only here (0/24 and 0/24 in (a)/(b)),
        # so a legend-scale colour alone cannot carry it
        if name == "tstt":
            # v31 QA4/QA5: collision-aware placement -- a candidate
            # stands only if its text bbox clears every placed box
            # (including the bold OD endpoint numbers) AND its leader
            # segment misses every other box; assert (not eyeball) is
            # the gate
            placed = list(ep_boxes)
            # v40 (external review, 89/100): the cut-changing node 17
            # only -- the OD endpoints 10/16 keep their bold endpoint
            # numbers (their printed values duplicated the colour/size
            # channels) and nodes 15/19 stay colour-coded
            order = sorted((v for v in xy
                            if abs(float(dn[v])) >= 5e-3 and v in (16,)),
                           key=lambda v: -abs(float(dn[v])))
            for v in order:
                node_px = ax.transData.transform(xy[v])
                # v41 (external review, 89/100): the node keeps its
                # NUMBER only -- the +0.29 value is stated once, in the
                # cut-flux line of the footer strip below panel (c)
                lbl = f"{v + 1}"
                placed_ok = False
                for dx, dy in CAND_OFFS.get(v, [(10, 6), (12, -6)]):
                    ha = "left" if dx >= 0 else "right"
                    # text-only box, hand-built: Annotation's own
                    # get_window_extent unions in the leader arrow
                    # (v31 QA6: the inflated box rejected every
                    # candidate). A shadow Text in axes coords gives
                    # the glyph size; anchor = node + offset points;
                    # vertical range approximates the baseline va.
                    probe = ax.text(0.5, 0.5, lbl, fontsize=11.2,
                                    transform=ax.transAxes, ha=ha,
                                    va="baseline")
                    pb = probe.get_window_extent(renderer)
                    probe.remove()
                    w, h = pb.width, pb.height
                    ox = dx * fig.dpi / 72.0
                    oy = dy * fig.dpi / 72.0
                    x0 = (node_px[0] + ox) if ha == "left" \
                        else (node_px[0] + ox - w)
                    y0 = node_px[1] + oy - 0.25 * h
                    bb = Bbox([[x0, y0], [x0 + w, y0 + 1.1 * h]])
                    if any(bb.overlaps(pb2)
                           for pb2, ow in placed if ow != v):
                        continue
                    # leader proxy: bbox centre -> node segment must
                    # miss every placed box (6 px padding). The
                    # segment stops 12 px short of the node centre:
                    # the real arrow ends at the disc rim, and the
                    # padded endpoint-number box sits right at the
                    # rim, so testing to the centre would reject
                    # every geometrically fine slot.
                    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
                    vx, vy = node_px[0] - cx, node_px[1] - cy
                    L = _math.hypot(vx, vy) or 1.0
                    end = (node_px[0] - vx / L * 12.0,
                           node_px[1] - vy / L * 12.0)
                    if any(_seg_hits_bbox((cx, cy), end, pb2)
                           for pb2, ow in placed if ow != v):
                        continue
                    ax.annotate(lbl, xy[v], xytext=(dx, dy),
                                textcoords="offset points",
                                fontsize=11.2, color=C_DARK, zorder=6,
                                ha=ha,
                                arrowprops=dict(
                                    arrowstyle="-", lw=0.6,
                                    color=C_DARK, shrinkA=0.5,
                                    shrinkB=2.5),
                                path_effects=[
                                    pe.withStroke(linewidth=2.5,
                                                  foreground="white")])
                    placed.append((bb, v))
                    placed_ok = True
                    break
                assert placed_ok, \
                    ("no collision-free label slot for node %d; placed=%s"
                     % (v + 1, [(round(p.x0), round(p.y0),
                                 round(p.x1), round(p.y1))
                                for p, _ in placed]))
        # v33 (external review): cross-section cut markers -- node 10
        # carries the solid ring of U={10} inside the dashed ring of
        # U'={10,17}, node 17 the matching dashed ring; the node-17
        # imbalance (+0.29) IS the U->U' net cut-responsibility change
        # (main-text identity). v35: both channels black, distinguished
        # by solid/dashed (external review: red triple-loaded with the
        # negative attribution and node-imbalance channels). Ellipses in
        # data coordinates with height = width/AXY render as visual
        # circles under the fixed panel aspect; radii are converted from
        # points through the same transData the collision tests use
        # (limits are already fixed at this point).
        if name == "tstt":
            from matplotlib.patches import Ellipse, Polygon

            # v71: the cut values for the set tags -- computed ONCE
            # here (before the ring block) from the frozen record,
            # reused by the footer rows below. BCn[v] = -(B C~)_v
            # (tail +1 / head -1), so phi(U) for U={10} is BCn[s] and
            # phi(U') = BCn[s] + BCn[16] (v58 sign convention).
            BCn = np.zeros(len(xy))
            for (u, v), c in zip(edges, data["tstt_C"].astype(float)):
                BCn[u] += c
                BCn[v] -= c
            netU = BCn[s]                  # phi(U), U = {10} (0-based s)
            netUp = BCn[s] + BCn[16]       # phi(U'), U' = {10, 17}

            def _ring(ax_, fig_, center, r_pt, dashed, lw):
                pa = ax_.transData.transform(center)
                px = r_pt * fig_.dpi / 72.0
                inv = ax_.transData.inverted()
                b = inv.transform((pa[0] + px, pa[1]))
                rx = abs(b[0] - center[0])
                ax_.add_patch(Ellipse(
                    center, width=2.0 * rx, height=2.0 * rx / AXY,
                    fill=False, ec=C_DARK, lw=lw,
                    linestyle=(0, (3.4, 1.6)) if dashed else "-",
                    zorder=3.4))

            def _hull_outline(ax_, fig_, c1, r1_pt, c2, r2_pt, lw):
                # v65 (external review, 90/100, thirty-first round):
                # ONE common dashed outline for U'={10,17} -- the
                # convex hull of the two clear-of-disc circles
                # (support-function sampling in display px; the tangent
                # segments between the sampled arcs come out exactly as
                # the chords at the support switch). Geometry check
                # (TNTP coords, display units): node 16 sits 61 pt
                # off the 10-17 axis while the hull's boundary stays
                # within ~14 pt of the axis -- node 16's disc stays
                # OUTSIDE the outline by a wide margin.
                pa1 = np.array(ax_.transData.transform(c1), dtype=float)
                pa2 = np.array(ax_.transData.transform(c2), dtype=float)
                px1 = r1_pt * fig_.dpi / 72.0
                px2 = r2_pt * fig_.dpi / 72.0
                th = np.linspace(0.0, 2.0 * np.pi, 721)
                u = np.stack([np.cos(th), np.sin(th)], axis=1)
                s1 = u @ pa1 + px1
                s2 = u @ pa2 + px2
                pts = np.where((s1 >= s2)[:, None],
                               pa1[None, :] + px1 * u,
                               pa2[None, :] + px2 * u)
                inv_ = ax_.transData.inverted()
                ax_.add_patch(Polygon(
                    inv_.transform(pts), closed=True, fill=False,
                    ec=C_DARK, lw=lw,
                    linestyle=(0, (3.4, 1.6)), zorder=3.4))

            r10x, r10y = xy[s]                      # node 10 = OD origin
            r17x, r17y = xy[16]                     # node 17 (0-based)
            # v35 vision-audit tweak: radii pushed outward so the solid
            # U ring clears node 10's own disc edge and the U' ring
            # clears the tightly packed node 16/17 pair.
            # v41 (external review, 89/100): radii pushed outward again
            # (8.6/11.4/10.3 -> 10.4/13.4/11.6) so the ring contours
            # keep clear spacing from the node discs and the
            # responsibility edges they used to graze (review: rings
            # and edges stacked in the 10/16/17 cluster).
            # v41 figure QA: zorder 5.6 -> 3.4 -- OCCLUSION CONVENTION.
            # The 16/17 pair sits 88 px apart with ~45 px discs, so any
            # ring radius that clears node 17's own disc (>= ~70 px)
            # geometrically must cross node 16's disc; drawing the
            # rings BELOW the discs (3.4: still above the 2.5
            # responsibility edges) makes the ring pass behind the disc
            # instead of dashed-line-over-coloured-disc, preserving the
            # node colour channel
            # v65 (external review, 90/100, thirty-first round): U'
            # switches from TWO separate dashed rings (one per member
            # node) to ONE common dashed outline enclosing both -- the
            # set reads as a moved network boundary (which edges become
            # internal), not as two highlighted nodes (review §5). The
            # U solid ring and the radii convention (14.2 around node
            # 10, 12.4 around node 17 -- the v41 clearances) are
            # unchanged, so the window extents and the v43 sizing stay
            # valid.
            _ring(ax, fig, (r10x, r10y), 11.2, dashed=False, lw=1.6)   # U
            _hull_outline(ax, fig, (r10x, r10y), 14.2,
                          (r17x, r17y), 12.4, lw=1.6)                  # U'
            # v48 (external review, 88/100): ring-side set tags inside
            # the panel, so the two cuts are identified without
            # re-reading the caption.
            # v50 (external review, 89/100): BOTH tags re-placed fully
            # INSIDE panel (c) with short leader lines (the v49
            # left/right ring-side offsets ran U={10} into panel (b)'s
            # side of the figure and U'={10,17} into the colorbar with
            # its tail clipped). U={10} takes the blank lower-left
            # corner (leader up to node 10's rings); U'={10,17} takes
            # the blank top band over the inter-column corridor --
            # node 16 sits directly above node 17 and the "17"/"19"
            # numbers block the right/bottom-right approaches, so the
            # leader comes down from the top-left onto node 17's
            # dashed ring. Both placements are assertion-gated below.
            def _pt_data(center, dx_pt, dy_pt):
                pa = ax.transData.transform(center)
                inv = ax.transData.inverted()
                return tuple(inv.transform(
                    (pa[0] + dx_pt * fig.dpi / 72.0,
                     pa[1] + dy_pt * fig.dpi / 72.0)))

            # leader targets ON the rings (px offsets on the v49 radii):
            # v65: U -> node 10's SOLID ring at 215 deg (the dashed
            # companion ring is gone with the common-outline switch;
            # 14.2 -> 11.2 pt anchor radius); U' -> the common outline's
            # node-17 arc at 160 deg, approached from the top-left
            # across the empty inter-column corridor
            u_anchor = _pt_data((r10x, r10y), -9.2, -6.4)
            up_anchor = _pt_data((r17x, r17y), -11.7, 4.3)
            # v71 (external review, 91/100, thirty-seventh round): each
            # set tag gains a SECOND line with the panel's own cut value
            # (from the frozen record above), so the two numbers the
            # figure exists to deliver read directly beside their cuts;
            # the TeX caption drops its numeric copy.
            def _sci(val):
                # raw mathtext body (no $ delimiters) for embedding
                m, e = f"{val:.2e}".split("e")
                return rf"+{m}{{\times}}10^{{{int(e)}}}"

            # v77: the two set tags place through a CANDIDATE-FAN of
            # offsets (the node-label pattern above) -- the v49/v50
            # fixed offsets were tuned for the 10.8 pt tags on the
            # 4.60 in canvas and the enlarged 11.6 pt tags on the
            # taller canvas need a different approach angle; the first
            # candidate that passes ALL v50 gates (inside panel, box
            # clearance, leader-miss) wins, and exhaustion is a hard
            # error as before.
            _settag_specs = [
                (r"$U=\{10\}$" + "\n" + rf"$\phi_j(U)={_sci(netU)}$",
                 u_anchor,
                 [(-27, -86), (-40, -86), (-14, -86),
                  (-27, -101), (-40, -101), (-27, -71),
                  (-14, -101), (-40, -71), (-14, -71)],
                 "left", "top", s),
                (r"$U'=\{10,17\}$" + "\n"
                 + rf"$\phi_j(U')={_sci(netUp)}$",
                 up_anchor,
                 [(-60, 43), (-72, 43), (-48, 43),
                  (-60, 58), (-72, 58), (-84, 58),
                  (-72, 73), (-84, 73), (-96, 58), (-84, 43), (-96, 43),
                  (-48, 58), (-36, 43), (-24, 43)],
                 "left", "bottom", 16),
            ]
            ren_ = fig.canvas.get_renderer()
            for lbl, anchor, cand_offs, ha, va, owner in _settag_specs:
                placed_ok = False
                for offx, offy in cand_offs:
                    tag = ax.annotate(
                        lbl, (anchor[0], anchor[1]), xytext=(offx, offy),
                        textcoords="offset points", fontsize=11.6,
                        color=C_DARK, ha=ha, va=va, zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.6,
                                                    foreground="white")])
                    # v50 gates, on the REAL text box (no arrow yet, so
                    # get_window_extent is the pure text extent): (1) the
                    # label box lies fully inside the panel (c) axes box
                    # (4 px margin); (2) it clears every placed text box;
                    # (3) its leader segment -- drawn below from the box
                    # edge to the ring-edge anchor -- misses every placed
                    # box except the owner node's own.
                    tb = tag.get_window_extent(ren_)
                    ax_bb = ax.bbox
                    if not (tb.x0 >= ax_bb.x0 + 4
                            and tb.x1 <= ax_bb.x1 - 4
                            and tb.y0 >= ax_bb.y0 + 4
                            and tb.y1 <= ax_bb.y1 - 4):
                        tag.remove()
                        continue
                    if any(tb.overlaps(pb2) for pb2, _ in placed):
                        tag.remove()
                        continue
                    axp = ax.transData.transform((anchor[0], anchor[1]))
                    # leader start: where the box-centre -> anchor ray
                    # exits the label box, then a 3 px gap
                    ccx, ccy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
                    vx, vy = axp[0] - ccx, axp[1] - ccy
                    t_exit = min(((tb.x1 - ccx) / vx if vx > 0 else
                                  (tb.x0 - ccx) / vx if vx < 0 else 1e9),
                                 ((tb.y1 - ccy) / vy if vy > 0 else
                                  (tb.y0 - ccy) / vy if vy < 0 else 1e9))
                    ex, ey = ccx + t_exit * vx, ccy + t_exit * vy
                    exn, eyn = (
                        ex + 3.0 * vx / (hypot := _math.hypot(vx, vy)),
                        ey + 3.0 * vy / hypot)
                    if any(_seg_hits_bbox((exn, eyn), axp, pb2)
                           for pb2, ow in placed if ow != owner):
                        tag.remove()
                        continue
                    inv = ax.transData.inverted()
                    (sx, sy), = inv.transform([(exn, eyn)])
                    ax.plot([sx, anchor[0]], [sy, anchor[1]], "-",
                            lw=0.7, color=C_DARK, zorder=4.6)
                    placed.append((tb, "settag"))
                    placed_ok = True
                    break
                assert placed_ok, (
                    "no collision-free slot for set tag %r; placed=%s"
                    % (lbl, [(round(p.x0), round(p.y0),
                              round(p.x1), round(p.y1), ow)
                             for p, ow in placed]))
            # v40 (external review, 89/100): the rings stay as pure
            # visual marks; the U/U'/Delta-cut-flux statement moved to
            # the outside annotation strip under panel (c) (footer
            # rows below). The v37/v38 in-panel collision-scan
            # machinery (U' label block, disc/edge/leader gates,
            # free-rectangle fallback) is retired with it -- the only
            # in-panel annotation left is node 17's number.

        # v41 (external review, 89/100): panel (b)'s "target arc"
        # leader removed -- the haloed edge is the panel's only
        # strongly coloured object and the title already names it

        # v61: the reference-scale direction/scale footer quantities are
        # deleted with the beta^ref normalization (the records live in
        # the supplement's Table S7; the v51 verdict rows remain)
        ax.set_title(title, fontsize=10.5, pad=5)
        # v32 QA2 wording per external review: "direction error" /
        # "ref-scale difference" (the bare "direction"/"scale" read
        # ambiguous). v33: "norm." prefix -- the footer quantities are
        # divided by |beta^can| (the split identity of main-text Eq. 7)
        # and must not be read against the absolute values of
        # main-text Table 4. v34: footer reduced to the two core
        # readings; the max node imbalance is annotated in place in
        # panel (c). v35: 7.5 -> 9.3 pt so the footer values land at
        # ~8 pt after the TeX include scale. v39: the narrow (a)/(b)
        # panels cannot carry the full phrase -- "norm." is dropped
        # (the split equation fixes the normalization) and the font
        # auto-fits the panel width (9.3 -> down to 7.6 pt), with the
        # wide (c) panel keeping 9.3 pt.
        renderer = fig.canvas.get_renderer()
        # v44 (external review, 89/100): "ref-scale diff" renamed to
        # "reference-scale mismatch" (the reviewer's wording)
        # v50 (external review, 89/100): renamed again to "diff. from
        # reference" (review: "mismatch" suggests an estimation error;
        # the reference scale is not a true scale -- the TeX caption
        # defines the abbreviation). Only the narrow (a)/(b) panels
        # print this row (panel (c) states the cut readings instead),
        # and they cannot carry the full phrase above the fit floor.
        # v51 (external review, 89/100): the numeric footer rows are
        # REPLACED by per-panel verdicts -- (a) OD-balanced, (b)
        # OD-balanced with a circulation-dependent scale, (c)
        # cut-dependent responsibility with the cut readings. The
        # direction-error / diff.-from-reference numbers move to the
        # TeX caption and Table 4 (review: the auxiliary reference
        # scale must not headline two of the three panels; the
        # direction/scale values above stay computed for the caption's
        # stated record). Width-check new rows with FIG_DEBUG=1.
        if name != "tstt":
            _verdict = {"uniform": "OD-balanced",
                        "active": "OD-balanced,"}[name]
            rows = [(-0.062, _verdict)]
            if name == "active":
                rows.append((-0.116, "circulation-dependent scale"))
        else:
            rows = []
        if name == "tstt":
            # v46/v58 history: the footer keeps the magnitude reading
            # |phi' - phi| / |phi| under the v58 sign convention
            # (netU/netUp computed above the ring block since v71).
            pct = abs((netUp - netU) / netU)

            # v69 (external review, 91/100, thirty-fifth round): the
            # middle cut-value row ("cut: +5.77e4 -> +4.34e4") is
            # DROPPED -- the two values move into the TeX caption
            # (review: keep "boundary-dependent account" and the 24.8%
            # reading in the strip; the two magnitudes crowd the three
            # lines). netU/netUp stay computed so pct keeps its
            # in-script provenance from the frozen record.
            rows = [(-0.062, "boundary-dependent account"),
                    # v47 (external review, 90/100): denominator stated
                    # explicitly -- the difference is 24.8% of the
                    # initial cut's MAGNITUDE (the caption names the
                    # referent; node-17 attribution stays with the
                    # caption).
                    (-0.116, f"difference: {pct * 100:.1f}%"
                             " of the initial cut")]
        for yy, lbl in rows:
            # v48 (external review, 88/100): ladder floor lifted
            # 7.4 -> 7.8 and the start 9.3 -> 10.0 (review: slightly
            # larger bottom cut-value type)
            # v49 (external review, 90/100): start lifted to 11.0 --
            # every row that fit at 10.0 now tries 11.0/10.5 first
            for fs in (11.6, 11.0, 10.4, 9.8, 9.3, 8.8, 8.3, 7.8):
                probe = ax.text(0.5, yy, lbl, transform=ax.transAxes,
                                fontsize=fs, ha="center", va="top",
                                color=C_DARK)
                pb = probe.get_window_extent(renderer)
                if pb.width <= ax.bbox.width - 8.0:
                    break
                probe.remove()
            # v40 lesson: a line too wide at 7.4 pt is REMOVED by this
            # ladder (every failed probe is removed) -- width-check new
            # footer lines with FIG_DEBUG=1 before shipping
            import os as _os
            if _os.environ.get("FIG_DEBUG"):
                kept = any(t.get_text() == lbl and t.get_position()[1] == yy
                           for t in ax.texts)
                print(f"[footer] {name!r} fs={fs} w={pb.width:.1f} "
                      f"floor={ax.bbox.width - 8.0:.1f} "
                      f"kept={kept} :: {lbl}")
        # v39: the per-panel full-network context inset is REMOVED
        # (external review, 90/100: unreadable at this size; Fig. 4
        # carries the topology)

    # ---- v31: shared edge-encoding legend row (the v30 figure never
    # legend-defined the edge channels; external review flagged it).
    # Line-width ticks at |C|/|beta| = 0.5 / 1.0 / 2.5 on the SAME
    # square-root scale as the maps; sign channels as colour samples.
    # The node-colour channel is NOT repeated here -- the colorbar
    # beside panel (c) already defines it (QA round 2: the duplicated
    # entry overflowed the legend axis into the colorbar).
    # v37 QA round 2: legend band lifted so the gap between the footer
    # lines and the legend closes; v39: lowered and slimmed
    # (0.155 h0.13 -> 0.09 h0.10) -- the unshrunk v39 map boxes hang
    # their footers from the full band bottom, and the v37 overlap of
    # the two bands put (a)/(b) footers inside the legend (vision QA)
    leg = fig.add_axes([0.075, 0.09, 0.86, 0.10])
    leg.set_xlim(0, 9.0)
    leg.set_ylim(0, 1)
    leg.axis("off")
    leg.text(0.0, 0.55, r"edge attribution $|\widetilde C_{e,j}|"
             r"/|\phi_j(\{s_j\})|$"
             r"  (width $\propto\sqrt{\cdot}$):",
             fontsize=9.2, color=C_DARK, va="center", ha="left")
    # v57: the node-sign fill channel, legend-defined in the free
    # bottom-left corner of the band (the tick value labels start at
    # x=4.05 on the same row; compact wording, the TeX caption states
    # the precise reading)
    leg.text(0.0, 0.10, "node sign:", fontsize=8.8, color=C_DARK,
             va="center", ha="left")
    leg.scatter([1.05], [0.10], s=44,
                c=[div_cmap(norm(dmax))],
                edgecolors="#9AA0A6", linewidths=0.35, zorder=3.5)
    leg.text(1.22, 0.10, "$+$ filled", fontsize=8.8, color=C_DARK,
             va="center", ha="left")
    leg.scatter([2.35], [0.10], s=44, facecolors="none",
                edgecolors=[div_cmap(norm(-dmax))],
                linewidths=1.4, zorder=3.5)
    leg.text(2.52, 0.10, "$-$ hollow", fontsize=8.8, color=C_DARK,
             va="center", ha="left")
    # v61: ticks refit to the phi normalization (global max 1.0)
    # v64: ticks shifted +0.25 -- the 9.2 pt edge-attribution label now ends
    # at x=4.04 and the 8.6 pt one cleared the 4.05 arrow start by only ~0.01
    # data units
    for xpos, val in ((4.30, 0.25), (5.10, 0.5), (6.00, 1.0)):
        leg.add_patch(FancyArrowPatch(
            (xpos, 0.55), (xpos + 0.55, 0.55), arrowstyle="-|>",
            mutation_scale=6.5, color=C_BLUE, lw=lw_of(val, xmax),
            zorder=2.5))
        # v61 fixup (codex re-review): :.1f truncated 0.25 -> "0.2"
        # while the line drew lw_of(0.25) -- :g prints 0.25 verbatim
        leg.text(xpos + 0.27, 0.08, f"{val:g}", fontsize=8.8,
                 color=C_DARK, ha="center", va="center")
    leg.add_patch(FancyArrowPatch(
        (6.85, 0.55), (7.45, 0.55), arrowstyle="-|>", mutation_scale=6.5,
        color=C_BLUE, lw=lw_of(1.0, xmax), zorder=2.5))
    leg.text(7.60, 0.55, "positive", fontsize=9.2, color=C_DARK,
             va="center", ha="left")
    leg.add_patch(FancyArrowPatch(
        (6.85, 0.10), (7.45, 0.10), arrowstyle="-|>", mutation_scale=6.5,
        color=C_RED, lw=lw_of(0.6, xmax), zorder=2.5,
        linestyle=(0, (3.2, 1.6))))
    leg.text(7.60, 0.10, "negative", fontsize=9.2, color=C_DARK,
             va="center", ha="left")

    # v77 (external review, 89/100, forty-eighth round): the two cut
    # MARKS are legend-defined in-figure -- a third legend line under
    # the shared band pairs a solid-ring sample with U and a dashed-
    # outline sample with U' (review: the solid-ring / dashed-outline
    # reading lived only in the TeX caption). Circles are drawn in
    # figure coordinates; the x/y radius ratio W/H keeps them round.
    # Full set definitions stay on the in-panel U/U' set tags.
    from matplotlib.patches import Ellipse as _Ellipse
    _r_x, _r_y = 0.0102, 0.0102 * FULL_W / HEIGHT   # ~5.5 pt radius
    _ly = 0.048
    fig.text(0.075, _ly, "cuts (panel c):", fontsize=8.4, color=C_DARK,
             va="center", ha="left")
    fig.add_artist(_Ellipse((0.207, _ly), width=2 * _r_x, height=2 * _r_y,
                            transform=fig.transFigure, fill=False,
                            ec=C_DARK, lw=1.6, zorder=5))
    fig.text(0.224, _ly, "$U$: solid ring", fontsize=8.2, color=C_DARK,
             va="center", ha="left")
    fig.add_artist(_Ellipse((0.342, _ly), width=2 * _r_x, height=2 * _r_y,
                            transform=fig.transFigure, fill=False,
                            ec=C_DARK, lw=1.6, zorder=5,
                            linestyle=(0, (3.4, 1.6))))
    fig.text(0.359, _ly, "$U'$: dashed outline", fontsize=8.2, color=C_DARK,
             va="center", ha="left")

    sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax)
    # v42 (external review, 89/100): channel renamed -- "off-OD
    # imbalance" read as excluding the OD endpoints spatially; the
    # residual is orthogonal to the OD *direction*, not off the OD
    # *nodes*
    # v61: the label follows the phi normalization
    # v68 (external review, 89/100, thirty-fourth round): the rotated
    # colorbar title is the plain channel name only -- the
    # normalization formula $(d_j)_v/|phi_j({s_j})|$ moves to the TeX
    # caption (review: the long vertical title crowded the right
    # margin; node 17 and the two cut outlines stay the visual
    # anchors)
    cb.set_label("OD-direction residual", fontsize=10.2)
    # v42 (external review, 89/100): explicit ticks at the range
    # endpoints (+-dmax ~ +-0.29) so the annotated node-17 value maps
    # onto the scale; the default major ticks stopped at 0.2 while
    # the color band kept running
    ticks = sorted({round(-dmax, 2), -0.2, -0.1, 0.0, 0.1, 0.2,
                    round(dmax, 2)})
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.ax.tick_params(labelsize=9.0)
    cb.outline.set_linewidth(0.6)

    # v60 (external review, 89/100, twenty-sixth round): the panel-(b)
    # tan halo (#F3D9A4 underlay beneath the selected active arc,
    # added v31) is now legend-defined -- the shared legend band is
    # full, so the sample line + two-line label live in the free
    # right margin below the colorbar; the TeX caption names the halo
    # as well (review: every visible channel must be explained)
    halo_y = 0.390
    fig.add_artist(plt.Line2D([0.885, 0.955], [halo_y, halo_y],
                              color="#F3D9A4",
                              lw=lw_of(1.0, xmax) + 3.2, alpha=0.9,
                              solid_capstyle="round"))
    fig.text(0.920, halo_y - 0.026, "selected arc",
             fontsize=8.6, color=C_DARK, ha="center", va="top")
    fig.text(0.920, halo_y - 0.054, "(halo, panel b)",
             fontsize=8.0, color=C_DARK, ha="center", va="top")

    save_pub(fig, "fig_resp_field_v98")


if __name__ == "__main__":
    main()
