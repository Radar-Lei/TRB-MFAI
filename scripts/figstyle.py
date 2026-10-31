"""Shared publication style for TRB-MFAI figures (cas-sc single column, Elsevier TRB-B).

Per the internal figure contract: white background, restrained palette
(neutral + signal + accent), direct labels over legends where spatially fixed,
5 pt glyph floor, editable fonts (fonttype 42), no undisclosed smoothing or
re-aggregation. All numbers come from experiments/results/*.json via figdata.py.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# cas-sc single-column full text width ~190 mm (7.48 in).
FULL_W = 7.48   # in
HALF_W = 3.66   # in

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,  # final-size floor: axis text >= 9 pt, ticks >= 8.5 pt
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.6,
})

# Restrained palette: neutral gray family, one blue signal family, one red accent.
# Unified verdict semantics across all figures (v3): blue circles = law holds /
# conserving; purple diamonds = projected / neutral; red squares = violating;
# light grey band = machine precision (drawn with axhspan, never a dashed text).
M_LAW, M_PROJ, M_VIOL = "o", "D", "s"
C_NEUTRAL = "#9AA0A6"
C_DARK    = "#3C4043"
C_BLUE    = "#3B6FB6"   # signal (conserving / M3)
C_BLUE_L  = "#A8C4E8"   # light signal
C_RED     = "#C4473A"   # accent (breaking / violation)
C_RED_L   = "#E8B3AC"
C_GREEN   = "#4C8A54"   # reserved for gains only
C_AMBER   = "#C9932B"   # secondary accent (M2 / penalties)
C_PURPLE  = "#7B5EA7"   # tertiary (manifold / M4)

METHOD_COLORS = {
    "M1": C_RED,
    "M2": C_AMBER,
    "M3": C_BLUE,
    "M4": C_PURPLE,
    "conserving": C_BLUE,
    "breaking": C_RED,
    "manifold": C_PURPLE,
}


def save_pub(fig, name, formats=("pdf", "svg", "png"), dpi=600):
    """Export to figures/ with tight bbox; PDF is the LaTeX target, PNG for QA."""
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for ext in formats:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
