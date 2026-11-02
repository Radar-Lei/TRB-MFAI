"""Fig 1 (v112) — pipeline overview: from an OD demand change to a certified,
error-controlled spatial explanation, for reference and learned assignment.

Contract (external review of v111, P0-2/P1-2): one figure replaces
Table tab:objects and Table tab:threequestions in the Introduction and makes
the AI line explicit: conservative learned assignment -> edge attribution ->
topology certificate -> adaptive summary -> response supervision. Four panels,
each answering one narrative question: (a) the explanation object (the signed
edge-attribution column), (b) where it lives (the boundary-attribution
profile over nested connected regions), (c) how it is reported (one value or
the minimum-segment summary), (d) how a learned assignment is certified,
validated and trained.

Archetype: schematic-led composite, single column, 2x2 panels with the
pipeline spine on top. All elements share the single data-coordinate system
(no inset axes); mini-profile axes are drawn by hand inside each panel so
positions cannot drift. No experiment numbers appear; symbols only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

from figstyle import C_BLUE, C_BLUE_L, C_RED, C_RED_L, C_NEUTRAL, C_DARK, save_pub, FULL_W

GRAY_TXT = "#5F6368"
ATOM_FC = "#EAF1F9"
GAIN_FC = "#E7F0E7"
GAIN_EC = "#4C8A54"
GAIN_TX = "#2E5F35"
BOX_FC = "#F5F5F4"


def rbox(ax, x, y, w, h, fc, ec, lw=0.9, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=C_DARK, lw=1.0, z=3, head=6.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=head, lw=lw, color=color, zorder=z))


def txt(ax, x, y, s, size=7.2, color=C_DARK, ha="center", va="center",
        weight="normal", style="normal", z=5):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color,
            fontweight=weight, style=style, zorder=z)


def mini_axes(ax, x0, y0, x1, y1, ylim=None):
    """Hand-drawn axes frame; returns a value mapper into the frame.

    ylim: fixed (lo, hi) value range.  When omitted it is derived from
    the first call's data; later calls (e.g. single-point queries for
    segment representatives) then reuse it instead of re-scaling.
    """
    ax.plot([x0, x1], [y0, y0], lw=0.8, color=C_NEUTRAL, zorder=2)
    ax.plot([x0, x0], [y0, y1], lw=0.8, color=C_NEUTRAL, zorder=2)
    ax.plot([x0, x1], [0.5 * (y0 + y1)] * 2, lw=0.5, color=C_NEUTRAL, zorder=2)
    state = {'ylim': ylim}

    def mapper(xs, ys):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        ux = x0 + (xs - xs.min()) / max(xs.max() - xs.min(), 1e-9) * (x1 - x0 - 0.10)
        if state['ylim'] is None:
            state['ylim'] = (ys.min() - 0.15 * (ys.max() - ys.min() + 1e-9),
                             ys.max() + 0.15 * (ys.max() - ys.min() + 1e-9))
        lo, hi = state['ylim']
        uy = y0 + 0.08 + (ys - lo) / (hi - lo) * (y1 - y0 - 0.20)
        return ux, uy
    return mapper


def main():
    fig, ax = plt.subplots(figsize=(FULL_W, 6.15))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 10.75)
    ax.axis("off")

    # ---------------- top pipeline spine ----------------
    ysp = 10.10
    txt(ax, 5.0, ysp,
        "learned assignment $\\rightarrow$ edge attribution $\\rightarrow$ "
        "topology certificate $\\rightarrow$ adaptive summary $\\rightarrow$ response supervision",
        size=6.6, color=C_BLUE, weight="bold")
    ax.plot([0.35, 9.65], [ysp - 0.20, ysp - 0.20], lw=0.7, color=C_BLUE_L, zorder=1)
    txt(ax, 5.0, ysp + 0.36, "the same pipeline serves reference and learned assignment maps",
        size=7.0, color=GRAY_TXT, style="italic")

    # ---------------- panel frames ----------------
    L0, L1 = 0.18, 4.86
    R0, R1 = 5.14, 9.82
    Tt, Tb = 9.55, 5.00
    Bt, Bb = 4.62, 0.22
    for (x0, x1, yt, yb) in ((L0, L1, Tt, Tb), (R0, R1, Tt, Tb),
                             (L0, L1, Bt, Bb), (R0, R1, Bt, Bb)):
        ax.add_patch(Rectangle((x0, yb), x1 - x0, yt - yb, fc="none",
                               ec=C_NEUTRAL, lw=0.7, zorder=1))

    # ================ (a) explanation object ================
    txt(ax, L0 + 0.12, Tt - 0.24, "(a)", size=10, ha="left", weight="bold")
    txt(ax, L0 + 0.52, Tt - 0.24, "The explanation object",
        size=8.2, ha="left", weight="bold")
    cy = 8.42
    chips = [(0.42, 1.14, r"$\Delta q_j$", BOX_FC, C_DARK),
             (1.80, 1.44, "assignment $f$", ATOM_FC, C_BLUE),
             (3.50, 1.16, "outcome $T$", BOX_FC, C_DARK)]
    for (cxx, w, lab, fc, ec) in chips:
        rbox(ax, cxx, cy - 0.20, w, 0.40, fc, ec)
        txt(ax, cxx + w / 2, cy, lab, size=8.2, color=ec if fc == ATOM_FC else C_DARK)
    arrow(ax, 1.58, cy, 1.78, cy)
    arrow(ax, 3.26, cy, 3.48, cy)
    nox, noy = 1.10, 7.30
    path = [(0, 0), (0.85, 0.26), (1.85, 0.38), (2.80, 0.14)]
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        ax.plot([nox + x0, nox + x1], [noy + y0, noy + y1], lw=2.0, color=C_BLUE, zorder=3)
    ax.plot([nox + 0.85, nox + 1.33, nox + 1.85], [noy + 0.26, noy - 0.16, noy + 0.38],
            lw=1.0, color=C_NEUTRAL, zorder=2)
    for (px, py) in path:
        ax.add_patch(Circle((nox + px, noy + py), 0.075, fc=C_NEUTRAL, ec=C_DARK, lw=0.6, zorder=4))
    ax.add_patch(Circle((nox, noy), 0.095, fc=C_RED, ec=C_DARK, lw=0.6, zorder=5))
    ax.add_patch(Circle((nox + 2.80, noy + 0.14), 0.095, fc=C_RED, ec=C_DARK, lw=0.6, zorder=5))
    txt(ax, nox - 0.03, noy - 0.24, "$s_j$", size=7.8, color=C_RED)
    txt(ax, nox + 2.82, noy - 0.26, "$t_j$", size=7.8, color=C_RED)
    txt(ax, nox + 0.42, noy + 0.24, "+", size=9, color=C_BLUE)
    txt(ax, nox + 1.35, noy + 0.56, "$-$", size=9, color=C_RED)
    txt(ax, nox + 2.32, noy + 0.40, "+", size=9, color=C_BLUE)
    txt(ax, nox + 1.33, noy - 0.28, "$-$", size=8.4, color=C_RED)
    txt(ax, 2.52, noy - 0.62, "signed entries; other ODs' rerouting enters too",
        size=6.4, color=GRAY_TXT, style="italic")
    txt(ax, 2.52, 5.92,
        r"$c_j=\Delta q_j\int_0^1 \mathrm{diag}(\nabla T(f))\,J_f\,\mathbf{e}_j\,d\lambda$"
        "\n" r"$C_j=\mathbf{1}^{\top}c_j$ is the system-wide contribution",
        size=7.6)

    # ================ (b) boundary profile ================
    txt(ax, R0 + 0.12, Tt - 0.24, "(b)", size=10, ha="left", weight="bold")
    txt(ax, R0 + 0.52, Tt - 0.24, "Nested boundaries and the profile",
        size=8.2, ha="left", weight="bold")
    rx, ry = 5.52, 8.68
    rbox(ax, rx, ry - 1.62, 2.60, 1.62, C_BLUE_L, C_BLUE, lw=0.9)
    rbox(ax, rx + 0.10, ry - 1.52, 1.42, 1.42, "#FFFFFF", C_BLUE, lw=0.9)
    txt(ax, rx + 0.20, ry - 0.16, "$U_1$", size=7.6, ha="left", va="top", color=C_BLUE)
    txt(ax, rx + 1.66, ry - 0.16, "$U_2$", size=7.6, ha="left", va="top", color=C_BLUE)
    ax.add_patch(Circle((rx + 0.28, ry - 1.16), 0.08, fc=C_RED, ec=C_DARK, lw=0.6, zorder=5))
    txt(ax, rx + 0.28, ry - 1.42, "$s_j$", size=7.6, color=C_RED)
    ax.add_patch(Circle((rx + 2.83, ry - 0.58), 0.08, fc=C_NEUTRAL, ec=C_DARK, lw=0.6, zorder=5))
    txt(ax, rx + 2.83, ry - 0.86, "$t_j$", size=7.6, color=GRAY_TXT)
    ax.plot([rx + 0.62, rx + 1.02, rx + 1.46, rx + 2.83],
            [ry - 1.22, ry - 0.86, ry - 0.80, ry - 0.58], lw=1.8, color=C_BLUE, zorder=4)
    txt(ax, 7.05, 6.82, "two-sided connected regions\n(screens, cordons, corridors)",
        size=6.4, color=GRAY_TXT)
    mp = mini_axes(ax, 5.42, 5.62, 8.90, 6.62)
    xg = np.linspace(0, 1, 80)
    prof = 0.55 * np.exp(-((xg - 0.35) ** 2) / 0.06) - 0.9 * np.exp(-((xg - 0.78) ** 2) / 0.03) + 0.22
    ux, uy = mp(xg, prof)
    ax.plot(ux, uy, lw=1.3, color=C_BLUE, zorder=3)
    txt(ax, 5.34, 6.66, r"$\phi_j$", size=7.4, ha="left")
    txt(ax, 7.16, 5.26, r"$\phi_j(U)=-\mathbf{1}_U^{\top}Bc_j$, signed sum on crossing links",
        size=7.0)

    # ================ (c) adaptive summary ================
    txt(ax, L0 + 0.12, Bt - 0.24, "(c)", size=10, ha="left", weight="bold")
    txt(ax, L0 + 0.52, Bt - 0.24, "Adaptive summary",
        size=8.2, ha="left", weight="bold")
    rbox(ax, 0.40, 3.10, 1.94, 1.08, "#FFFFFF", C_DARK)
    txt(ax, 1.37, 4.02, "one value suffices", size=7.4, weight="bold")
    txt(ax, 1.37, 3.42, r"$S_j^{\mathcal{F}}\leq 2\epsilon$:" "\n"
        r"report $\beta_j^{\mathcal{F}}\pm\varepsilon_j^{\mathcal{F}}$", size=7.2)
    rbox(ax, 2.62, 3.10, 2.04, 1.08, "#FFFFFF", C_DARK)
    txt(ax, 3.64, 4.02, "profile needed", size=7.4, weight="bold")
    txt(ax, 3.64, 3.42, "minimum segments,\nprovably fewest", size=7.2)
    mp2 = mini_axes(ax, 0.50, 1.82, 4.60, 2.92)
    seg_b = 0.42
    ux, uy = mp2(xg, prof)
    ax.plot(ux, uy, lw=1.1, color=C_BLUE, zorder=3)
    for (a, b) in ((0.0, 0.30), (0.30, 0.62), (0.62, 1.0)):
        ia, ib = int(a * 79), int(b * 79)
        seg = prof[ia:ib + 1]
        rep = 0.5 * (seg.max() + seg.min())
        xa_, xb_ = ux[ia], ux[ib]
        ya_ = mp2([0.5], [rep])[1][0]
        band = (mp2([0.5], [rep + seg_b])[1][0] - mp2([0.5], [rep - seg_b])[1][0]) / 2
        ax.fill_between([xa_, xb_], [ya_ - band] * 2, [ya_ + band] * 2,
                        color=C_RED_L, alpha=0.5, lw=0, zorder=2)
        ax.plot([xa_, xb_], [ya_, ya_], lw=2.0, color=C_RED, zorder=4)
    txt(ax, 2.55, 1.66, "tolerance band $t_i$; segment representatives", size=6.4, color=GRAY_TXT)
    txt(ax, 2.55, 1.14, "Anaheim: 83$-$211 boundary values $\\to$ 4$-$14 segments\nat the working tolerance",
        size=6.6, color=GRAY_TXT, style="italic")

    # ================ (d) certify, validate, train ================
    txt(ax, R0 + 0.12, Bt - 0.24, "(d)", size=10, ha="left", weight="bold")
    txt(ax, R0 + 0.52, Bt - 0.24, "Learned assignment",
        size=8.2, ha="left", weight="bold")
    rbox(ax, 5.34, 3.14, 2.04, 1.04, ATOM_FC, C_BLUE)
    txt(ax, 6.36, 4.00, "structural certificate", size=7.1, color=C_BLUE, weight="bold")
    txt(ax, 6.36, 3.50, "weights constant on the\ntraversed block atoms\n(Theorem, per state)", size=6.6)
    rbox(ax, 7.58, 3.14, 2.10, 1.04, BOX_FC, C_DARK)
    txt(ax, 8.63, 4.00, "exact validation", size=7.1, weight="bold")
    txt(ax, 8.63, 3.52, r"node divergence gives" "\n"
        r"$\max_U|\phi^L_j-\phi^S_j|$ exactly", size=6.6)
    rbox(ax, 6.22, 1.96, 2.56, 0.82, GAIN_FC, GAIN_EC)
    txt(ax, 7.50, 2.58, "response supervision", size=7.6, color=GAIN_TX, weight="bold")
    txt(ax, 7.50, 2.22, "trains the flow Jacobian\nidentified by the error bound",
        size=6.6)
    arrow(ax, 6.36, 3.18, 7.02, 2.82, color=C_NEUTRAL)
    arrow(ax, 8.63, 3.18, 7.98, 2.82, color=C_NEUTRAL)
    txt(ax, 7.48, 1.58,
        r"summary error $\leq\varepsilon^{\mathrm{sum}}_j+\delta_j$" "\n"
        "(compression $+$ model error)", size=7.0)
    txt(ax, 7.50, 0.84,
        "compatibility and compression are certified\nfrom the learned model alone; learned-reference\nfidelity is an offline validation guarantee",
        size=6.4, color=GRAY_TXT, style="italic")

    save_pub(fig, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "fig0_pipeline_v112"), formats=("pdf", "png", "svg"))
    print("saved fig0_pipeline_v112")


if __name__ == "__main__":
    main()
