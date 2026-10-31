/* E80: exact enumeration of connected separating cuts (bitmask ops).
 *
 * Family for OD (s, t):  U contains s, excludes t, G'[U] and G'[V\U]
 * both connected in the underlying undirected graph.  Each connected
 * subset containing s (and avoiding t) is generated EXACTLY ONCE by
 * the include/exclude frontier recursion (ESU discipline): a vertex
 * popped from the candidate list is banned along the whole subtree,
 * so adjacency growth may not re-introduce it.  phi_f(U) =
 * -sum_{v in U} COLS[v][f] is carried incrementally; per-field min/max
 * and the attaining masks are returned.  Sets whose complement is
 * disconnected are not recorded, but the recursion continues
 * (complements of supersets are subsets, which can re-connect).
 *
 * An optional allowed-mask restricts the vertices that may be added
 * (used only by the Python<->C cross-validation on restricted
 * families; the full run passes all vertices).
 */
#include <stddef.h>
#include <stdint.h>

static int T, F;
static uint32_t ADJ[32], FULL, ALLOWED;
static double COLS[32][8];
static double best_min[8], best_max[8];
static uint32_t min_mask[8], max_mask[8];
static uint64_t n_sets;

static int connected(uint32_t mask) {
    if (!mask) return 0;
    uint32_t low = mask & (~mask + 1);
    uint32_t reach = low, frontier = low;
    while (frontier) {
        uint32_t nxt = 0, m = frontier;
        while (m) {
            nxt |= ADJ[__builtin_ctz(m)];
            m &= m - 1;
        }
        nxt &= mask & ~reach;
        reach |= nxt;
        frontier = nxt;
    }
    return reach == mask;
}

static void process(uint32_t S, const double *acc) {
    int f;
    n_sets++;
    for (f = 0; f < F; f++) {
        if (acc[f] < best_min[f]) { best_min[f] = acc[f]; min_mask[f] = S; }
        if (acc[f] > best_max[f]) { best_max[f] = acc[f]; max_mask[f] = S; }
    }
}

/* S: current set; cand: addable vertices; excl: banned vertices
 * (popped earlier along the branch, or t); acc: phi per field. */
static void expand(uint32_t S, uint32_t cand, uint32_t excl,
                   const double *acc) {
    double acc2[8];
    uint32_t popped = 0;
    while (cand) {
        uint32_t b = cand & (~cand + 1);
        int v = __builtin_ctz(cand);
        uint32_t S2;
        cand ^= b;
        if (v == T) { popped |= b; continue; }
        S2 = S | b;
        for (int f = 0; f < F; f++) acc2[f] = acc[f] - COLS[v][f];
        if (connected(FULL & ~S2)) process(S2, acc2);
        expand(S2, (cand | ADJ[v]) & ~S2 & ~(excl | popped) & ALLOWED,
               excl | popped, acc2);
        popped |= b;
    }
}

/* called via ctypes */
void enumerate(int n, uint32_t *adj, int s, int t, int f, double *cols,
               uint32_t allowed, double *bmin, double *bmax,
               uint32_t *mmask, uint32_t *xmask, uint64_t *counts) {
    int v, j;
    double acc0[8];
    FULL = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1);
    for (v = 0; v < n; v++) ADJ[v] = adj[v];
    T = t;
    F = f;
    ALLOWED = allowed;
    for (v = 0; v < n; v++)
        for (j = 0; j < f; j++) COLS[v][j] = cols[(size_t)v * f + j];
    n_sets = 0;
    for (j = 0; j < f; j++) {
        best_min[j] = 1e308;
        best_max[j] = -1e308;
        min_mask[j] = 0;
        max_mask[j] = 0;
    }
    for (j = 0; j < f; j++) acc0[j] = -COLS[s][j];
    counts[0] = 0;                      /* singleton_admissible flag */
    if (connected(FULL & ~(1u << s))) {
        counts[0] = 1;
        process(1u << s, acc0);
    }
    expand(1u << s, ADJ[s] & ~((1u << s) | (1u << t)) & ALLOWED,
           1u << t, acc0);
    counts[1] = n_sets;
    for (j = 0; j < f; j++) {
        bmin[j] = best_min[j];
        bmax[j] = best_max[j];
        mmask[j] = min_mask[j];
        xmask[j] = max_mask[j];
    }
}
