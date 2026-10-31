"""Direction-grouped boundary families for Sioux Falls (v108).

External review of v107: ordering all separating regions of a query into
one sequence by OD progress mixes screenlines with different normal
directions, so adjacent array positions need not be adjacent traffic
cross-sections.  This module rebuilds the e107/e114 geometry with
provenance: for each of the 36 projection normals the separating regions
form a nested chain ordered by the projection level, and each direction
is kept as its own ordered profile.  Construction is identical to
e114_v103 (angle grid 0:5:175, distinct projection values, midpoint
levels, both induced sides connected); only the bookkeeping changes.
"""
import numpy as np
import networkx as nx

from nets import load_sioux_falls


def _graph(net):
    g = nx.Graph()
    for (u, v) in net.edges:
        g.add_edge(u, v)
    return g


def direction_levels(net=None, coords=None):
    """Per-angle nested chains with connectivity filter.

    Returns a list of (angle, levels, masks) where masks[k] is the
    0-based node tuple of the below-side set at levels[k]; both induced
    sides are connected at every retained level.
    """
    if net is None:
        net = load_sioux_falls()
    if coords is None:
        xy = _node_coords(net)
    else:
        xy = np.asarray(coords)
    g = _graph(net)
    out = []
    for angle in np.arange(0, 180, 5):
        normal = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        normal[np.abs(normal) < 1e-14] = 0
        proj = xy @ normal
        values = np.sort(proj)
        lv = [values[0]]
        for v in values[1:]:
            if v - lv[-1] > 1e-12:
                lv.append(v)
        levels, masks = [], []
        for a, b in zip(lv[:-1], lv[1:]):
            U = frozenset(np.flatnonzero(proj < (a + b) / 2).tolist())
            V = frozenset(range(net.n)) - U
            if nx.is_connected(g.subgraph(U)) and nx.is_connected(g.subgraph(V)):
                levels.append((a + b) / 2)
                masks.append(tuple(sorted(U)))
        out.append((float(angle), np.array(levels), masks))
    return out


def _node_coords(net):
    """Sioux Falls coordinates via the e107 loader (1-based keys)."""
    from run_e107_v100_screenline import load_network
    _, coords, _, _ = load_network()
    return np.array([coords[i + 1] for i in range(net.n)])


def query_families(s, t, chains=None, net=None):
    """Ordered separating chains per direction for query (s, t), 0-based.

    Returns a list of dicts {angle, levels, masks} with only directions
    that separate s from t; each chain is ordered by projection level
    (below-side sets when proj_s < proj_t, complements otherwise), so
    consecutive entries are nested along that direction.
    """
    if chains is None:
        chains = direction_levels(net=net)
    if net is None:
        net = load_sioux_falls()
    allset = frozenset(range(net.n))
    out = []
    for angle, levels, masks in chains:
        # separating below-side sets: s inside, t outside
        sel = [(lv, m) for lv, m in zip(levels, masks) if s in m and t not in m]
        if sel:
            sel.sort(key=lambda x: x[0])
            out.append({'angle': angle, 'levels': [x[0] for x in sel],
                        'masks': [x[1] for x in sel], 'side': 'below'})
            continue
        # else s lies above t for this normal: separating sets are
        # complements of below-side sets with proj_t < level <= proj_s
        sel = [(lv, m) for lv, m in zip(levels, masks)
               if t in m and s not in m]
        if sel:
            sel.sort(key=lambda x: x[0])
            out.append({'angle': angle, 'levels': [x[0] for x in sel],
                        'masks': [tuple(sorted(allset - frozenset(m))) for _, m in sel],
                        'side': 'complement'})
    return out


def family_masks_union(fams):
    s = set()
    for f in fams:
        s.update(f['masks'])
    return s
