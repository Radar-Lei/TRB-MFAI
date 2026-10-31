"""Fig 1 (v107): block structure and query-driven class merging, two panels.

Panel (a): the two-bridges network with its block-cut tree (non-bridge
block atoms, bridge atoms, shared cut vertices, off-path pendant atom).
Panel (b): single- and joint-query class merging -- the OD block paths
merge the traversed atoms transitively; the off-path atom stays separate.
All facts derive from nets.two_bridges() at build time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Polygon
from figstyle import C_BLUE, C_RED, C_NEUTRAL, C_DARK, save_pub, FULL_W

GRAY_TXT = "#5F6368"
CUT_C = C_RED
CLASS_C = C_BLUE
OFF_C = GRAY_TXT
ATOM_FC = "#EAF1F9"
OFF_FC = "#F2F3F4"
BRIDGE_FC = "#FBEF9E"
SHARED_C = "#7B5EA7"


def network_facts():
    from nets import two_bridges
    net = two_bridges()
    eq = net.equivalence_classes([(0, 8)])
    bnm = net.block_node_map()
    cuts = sorted(v for v, ks in bnm.items() if len(ks) >= 2)
    atoms = eq["atoms"]
    classes = eq["classes"]
    D_edges = {11, 12, 13}
    A_edges = {0, 1, 2}
    D_nodes = {u for e in D_edges for u in net.edges[e]}
    A_nodes = {u for e in A_edges for u in net.edges[e]}
    assert D_nodes & A_nodes == {1}
    assert cuts == [1, 2, 3, 5, 6]
    assert len(atoms) == 6 and len(classes) == 2
    return net, eq, cuts


def tri(ax, pts, color=CLASS_C, lw=1.6, fc=ATOM_FC):
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                         edgecolor=color, lw=lw, joinstyle="round", zorder=2))


def node(ax, p, label="", cut=False, fs=11.2, s=28, dx=0.13, dy=0.16):
    ax.scatter(*p, s=s * (1.6 if cut else 1.0),
               color=(CUT_C if cut else C_DARK), zorder=4)
    if label:
        ax.text(p[0] + dx, p[1] + dy, label, fontsize=fs,
                color=(CUT_C if cut else C_DARK))


def atom_box(ax, x, y, w, h, label, fc=ATOM_FC, fs=12.0, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec or C_DARK, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=ec or C_DARK, zorder=4)


def panel_a(ax, net, eq):
    """Network with OD path + block-cut tree."""
    ax.set_xlim(0, 12.55)
    ax.set_ylim(-0.4, 5.3)
    ax.text(0.18, 5.02, "(a)  Network and block-cut tree",
            fontsize=12.2, fontweight="bold", color=C_DARK)
    yN = 3.2
    A = [(1.30, yN), (1.77, yN + 0.65), (2.24, yN)]
    B = [(4.00, yN), (4.47, yN + 0.65), (4.94, yN)]
    C = [(6.70, yN), (7.17, yN + 0.65), (7.64, yN)]
    for T in (A, B, C):
        tri(ax, T)
    ax.plot([2.24, 4.00], [yN, yN], color=CUT_C, lw=2.4, zorder=2)
    ax.plot([4.94, 6.70], [yN, yN], color=CUT_C, lw=2.4, zorder=2)
    D = [A[1], (1.37, yN + 1.30), (2.17, yN + 1.30)]
    tri(ax, D, color=OFF_C, lw=1.4, fc=OFF_FC)
    node(ax, D[1], "9", fs=9.8, dx=-0.32, dy=0.04)
    node(ax, D[2], "10", fs=9.8, dx=0.16, dy=0.04)
    ax.text(1.77, yN - 0.36, "$A$", fontsize=13.6, color=CLASS_C, ha="center")
    ax.text(4.47, yN - 0.36, "$B$", fontsize=13.6, color=CLASS_C, ha="center")
    ax.text(7.17, yN - 0.36, "$C$", fontsize=13.6, color=CLASS_C, ha="center")
    ax.text(1.77, yN + 1.54, "$D$ (off path)", fontsize=11.2, color=OFF_C,
            ha="center")
    node(ax, A[0], "$s{=}0$", fs=14.8, s=30, dx=-1.06, dy=-0.10)
    node(ax, A[1], cut=True)
    ax.text(A[1][0] - 0.28, A[1][1] + 0.06, "1", fontsize=9.8, color=CUT_C)
    node(ax, A[2], cut=True)
    ax.text(A[2][0] + 0.10, A[2][1] - 0.38, "2", fontsize=9.8, color=CUT_C,
            ha="center")
    node(ax, B[1], "$u{=}4$", fs=13.2, s=26, dx=-0.72, dy=0.10)
    node(ax, B[0], cut=True)
    ax.text(B[0][0] + 0.10, B[0][1] - 0.38, "3", fontsize=9.8, color=CUT_C,
            ha="center")
    node(ax, B[2], cut=True)
    ax.text(B[2][0] + 0.10, B[2][1] - 0.38, "5", fontsize=9.8, color=CUT_C,
            ha="center")
    node(ax, C[0], cut=True)
    ax.text(C[0][0] + 0.10, C[0][1] - 0.38, "6", fontsize=9.8, color=CUT_C,
            ha="center")
    node(ax, C[1], "7", fs=9.8, dx=-0.34, dy=0.04)
    node(ax, C[2], "$t{=}8$", fs=14.8, s=30, dx=0.26, dy=-0.08)
    ax.annotate("", xy=(7.64, yN + 0.98), xytext=(1.30, yN + 0.98),
                arrowprops=dict(arrowstyle="->", color="#7A8087", lw=1.2,
                                ls=(0, (4, 2))))
    ax.text(4.47, yN + 1.18, "OD block path  $s\\to t$", fontsize=11.2,
            color=GRAY_TXT, ha="center")
    ax.text(3.12, yN + 0.16, "$b_1$", fontsize=12.9, color=CUT_C, ha="center")
    ax.text(5.82, yN + 0.16, "$b_2$", fontsize=12.9, color=CUT_C, ha="center")

    lx = 8.75
    ly0, dly = 4.74, 0.40
    ax.add_patch(FancyBboxPatch((lx, ly0 - 0.08), 0.34, 0.20,
                                boxstyle="round,pad=0.02", fc=ATOM_FC,
                                ec=CLASS_C, lw=0.9, zorder=3))
    ax.text(lx + 0.44, ly0, "non-bridge block atom", fontsize=11.6,
            color=GRAY_TXT, ha="left", va="center")
    ax.add_patch(FancyBboxPatch((lx, ly0 - dly - 0.08), 0.34, 0.20,
                                boxstyle="round,pad=0.02", fc=BRIDGE_FC,
                                ec=C_DARK, lw=0.9, zorder=3))
    ax.text(lx + 0.44, ly0 - dly, "bridge atom $b_i$", fontsize=11.6,
            color=GRAY_TXT, ha="left", va="center")
    ax.scatter(lx + 0.17, ly0 - 2 * dly, s=26, color=CUT_C, zorder=4)
    ax.text(lx + 0.44, ly0 - 2 * dly, "cut vertex", fontsize=11.6,
            color=GRAY_TXT, ha="left", va="center")
    ax.plot([lx, lx + 0.34], [ly0 - 3 * dly] * 2, color=CUT_C, lw=2.6, zorder=2)
    ax.text(lx + 0.44, ly0 - 3 * dly, "bridge (network)", fontsize=11.6,
            color=GRAY_TXT, ha="left", va="center")

    # block-cut tree below the network
    yT = 0.78
    BW, GW = 0.72, 0.50
    xs = {}
    x = 0.55
    for nm in ("A", "b1", "B", "b2", "C"):
        w = BW if len(nm) == 1 else GW
        xs[nm] = (x, x + w)
        x += w + 0.46
    for nm in ("A", "B", "C"):
        a, b = xs[nm]
        atom_box(ax, a, yT, b - a, 0.72, f"${nm}$", fs=13.6, ec=CLASS_C)
    for nm in ("b1", "b2"):
        a, b = xs[nm]
        atom_box(ax, a, yT, b - a, 0.72, f"${nm[0]}_{{{nm[1]}}}$",
                 fc=BRIDGE_FC, fs=12.9)
    ymid = yT + 0.36
    joints = [xs["A"][1], xs["b1"][1], xs["B"][1], xs["b2"][1]]
    joint_num = ["2", "3", "5", "6"]
    for jx, jn in zip(joints, joint_num):
        ax.plot([jx, jx + 0.46], [ymid, ymid], color=CUT_C, lw=1.6, zorder=2)
        ax.scatter(jx + 0.23, ymid, s=14, color=CUT_C, zorder=4)
        ax.text(jx + 0.23, ymid + 0.17, jn, fontsize=9.8, color=CUT_C,
                ha="center", va="bottom", zorder=5)
    xA = sum(xs["A"]) / 2
    atom_box(ax, xA - 0.42, yT + 1.04, 0.84, 0.40, "$D$",
             fc=OFF_FC, fs=12.9, ec=OFF_C)
    ax.plot([xA, xA], [yT + 0.77, yT + 1.04], color=CUT_C, lw=1.6, zorder=2)
    ax.scatter(xA, yT + 0.905, s=14, color=CUT_C, zorder=4)
    ax.text(xA + 0.16, yT + 0.88, "1", fontsize=9.6, color=CUT_C,
            ha="left", va="center", zorder=5)
    return xs


def panel_b(ax, xs):
    """Single- and joint-query class merging."""
    ax.set_xlim(0, 12.55)
    ax.set_ylim(-2.6, 1.05)
    ax.text(0.18, 0.82, "(b)  Queries merge the atoms they traverse",
            fontsize=12.2, fontweight="bold", color=C_DARK)

    def chip_row(yC, shared=False):
        for nm in ("A", "b1", "B", "b2", "C"):
            a, b = xs[nm]
            w = b - a
            sh = shared and nm == "B"
            atom_box(ax, a, yC, w, 0.56,
                     f"${nm}$" if len(nm) == 1 else f"${nm[0]}_{{{nm[1]}}}$",
                     fc="#DDD2EE" if sh else (ATOM_FC if len(nm) == 1 else BRIDGE_FC),
                     fs=12.9 if len(nm) == 1 else 12.0,
                     ec=SHARED_C if sh else None)
            if sh:
                ax.add_patch(FancyBboxPatch((a, yC), w, 0.56,
                                            boxstyle="round,pad=0.05",
                                            fc="none", ec=SHARED_C, lw=1.8,
                                            zorder=3.5))

    def span(y, x0, x1, color, lab):
        ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, 0.15,
                                    boxstyle="round,pad=0.015",
                                    fc=color, ec="none", alpha=0.75, zorder=2))
        ax.text(x1 + 0.10, y + 0.075, lab, fontsize=11.2, color=color,
                ha="left", va="center")

    # single query (s,t)
    yC1 = -0.30
    chip_row(yC1)
    ax.text(sum(xs["A"]) / 2, yC1 + 0.64, "$s$", fontsize=13.8, color=C_DARK,
            ha="center", va="bottom")
    ax.text(sum(xs["C"]) / 2, yC1 + 0.64, "$t$", fontsize=13.8, color=C_DARK,
            ha="center", va="bottom")
    span(yC1 - 0.30, xs["A"][0], xs["C"][1], CLASS_C, r"$(s,t)$")
    ax.annotate("", xy=(7.05, yC1 + 0.28), xytext=(5.85, yC1 + 0.28),
                arrowprops=dict(arrowstyle="-|>", color=CUT_C, lw=1.4))
    ax.add_patch(FancyBboxPatch((7.15, yC1 - 0.18), 5.15, 0.82,
                                boxstyle="round,pad=0.06", fc=ATOM_FC,
                                ec=CLASS_C, lw=1.6, zorder=3))
    ax.text(9.72, yC1 + 0.26, r"one final class $\bar E_1$",
            ha="center", va="center", fontsize=12.2, color=CLASS_C, zorder=4)
    ax.text(9.72, yC1 - 0.06, r"$A\cup b_1\cup B\cup b_2\cup C$",
            ha="center", va="center", fontsize=10.4, color=C_DARK, zorder=4)
    ax.add_patch(FancyBboxPatch((8.60, yC1 - 1.10), 2.20, 0.56,
                                boxstyle="round,pad=0.06", fc=OFF_FC,
                                ec=OFF_C, lw=0.9, linestyle=(0, (3, 2)),
                                zorder=3))
    ax.text(9.70, yC1 - 0.82, r"$\bar E_2=D$ (off path)",
            ha="center", va="center", fontsize=10.8, color=GRAY_TXT, zorder=4,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.plot([9.72, 9.72], [yC1 - 0.18, yC1 - 0.54], color=OFF_C, lw=0.9,
            zorder=2)

    # joint query (s,u) + (u,t)
    yC2 = -1.98
    chip_row(yC2, shared=True)
    ax.text(sum(xs["A"]) / 2, yC2 + 0.64, "$s$", fontsize=13.8, color=C_DARK,
            ha="center", va="bottom")
    ax.text(sum(xs["B"]) / 2, yC2 + 0.64, "$u$", fontsize=13.8,
            color=SHARED_C, ha="center", va="bottom")
    ax.text(sum(xs["C"]) / 2, yC2 + 0.64, "$t$", fontsize=13.8, color=C_DARK,
            ha="center", va="bottom")
    span(yC2 - 0.28, xs["A"][0], xs["B"][1], CLASS_C, r"$(s,u)$")
    span(yC2 - 0.55, xs["B"][0], xs["C"][1], SHARED_C, r"$(u,t)$")
    ax.annotate("shared atom", xy=(sum(xs["B"]) / 2, yC2),
                xytext=(3.05, yC2 - 0.84), fontsize=11.0, color=SHARED_C,
                ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=SHARED_C, lw=0.8,
                                shrinkA=1, shrinkB=4))
    ax.annotate("", xy=(7.05, yC2 + 0.28), xytext=(5.85, yC2 + 0.28),
                arrowprops=dict(arrowstyle="-|>", color=CUT_C, lw=1.4))
    ax.add_patch(FancyBboxPatch((7.15, yC2 - 0.28), 5.15, 0.72,
                                boxstyle="round,pad=0.06", fc=ATOM_FC,
                                ec=CLASS_C, lw=1.6, zorder=3))
    ax.text(9.72, yC2 + 0.14, r"joint query $\to$ one final class",
            ha="center", va="center", fontsize=11.2, color=CLASS_C, zorder=4)
    ax.text(9.72, yC2 - 0.14, r"$A\cup b_1\cup B\cup b_2\cup C$",
            ha="center", va="center", fontsize=10.8, color=C_DARK, zorder=4)


def main():
    net, eq, cuts = network_facts()
    fig, axs = plt.subplots(2, 1, figsize=(FULL_W, 9.4))
    for ax in axs:
        ax.axis("off")
    xs = panel_a(axs[0], net, eq)
    panel_b(axs[1], xs)
    fig.subplots_adjust(hspace=0.06)
    save_pub(fig, "fig1_block_cut_v107")


if __name__ == "__main__":
    main()
