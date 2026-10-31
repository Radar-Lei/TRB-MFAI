"""Network constructors and topology analysis for conservation-consistent attribution.

Sign convention (inherited from pilot_c1.py, frozen):
    B[u, e] = -1 (tail), B[v, e] = +1 (head)  ->  node balance (B f)[v] = inflow - outflow
    Conservation:  B f(q) = M q   with  M[s_j, j] = -q_j, M[t_j, j] = +q_j.

Topology (T3c contract):
    circuit block = biconnected component (Hopcroft-Tarjan block) containing a cycle;
    each bridge forms a single-edge atom.  Atoms are merged along block-cut-tree paths
    of every OD pair, then transitively closed -> final equivalence classes.

NOTE: circuit blocks are *biconnected components*, NOT 2-edge-connected components.
Two loops sharing only a cut vertex share no edge and hence no cycle -> they are
distinct matroid components (Lemma M forces constant weights inside each, but no
relation between them).  This distinction is exactly the T3c correction tested in E1.4.
"""
from __future__ import annotations

import heapq
import os
import re
from dataclasses import dataclass, field

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ----------------------------------------------------------------------------
# Network container
# ----------------------------------------------------------------------------
@dataclass
class Network:
    name: str
    edges: list[tuple[int, int]]                    # directed (u, v), 0-based
    capacity: np.ndarray = None                     # (m,) BPR capacity c_e
    free_flow_time: np.ndarray = None               # (m,) BPR t0_e
    _B: np.ndarray = field(default=None, repr=False)
    _blocks: list = field(default=None, repr=False)
    # v17 lazy caches for the block-cut machinery (one O(n+m) build, reused)
    _bc_tree: dict = field(default=None, repr=False)
    _bc_rooted: tuple = field(default=None, repr=False)

    # ---- basic ----
    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def n(self) -> int:
        return (max(max(u, v) for u, v in self.edges) + 1) if self.edges else 0

    @property
    def B(self) -> np.ndarray:
        if self._B is None:
            B = np.zeros((self.n, self.m))
            for e, (u, v) in enumerate(self.edges):
                B[u, e] = -1.0
                B[v, e] = +1.0
            self._B = B
        return self._B

    def od_M(self, ods: list[tuple[int, int]]) -> np.ndarray:
        """Node-OD incidence matrix M (n x K)."""
        M = np.zeros((self.n, len(ods)))
        for j, (s, t) in enumerate(ods):
            M[s, j] = -1.0
            M[t, j] = +1.0
        return M

    def nullspace_basis(self) -> np.ndarray:
        """Orthonormal basis N (m x (m - rank B)) of ker B (= undirected cycle
        orientation absorbed by signs).

        v17: each column is canonically oriented (sign fixed so the entry of
        largest absolute value is positive).  LAPACK/SVD results may differ
        across versions otherwise only by column signs, which changes the
        realized analytic field N z(q) under a fixed seed; the canonical
        orientation removes that drift.  (Degenerate singular values can in
        principle rotate the basis within a subspace; for reproducibility
        across environments the analytic-field checkpoints freeze the full
        state_dict, N included, see models.load_or_freeze_field.)"""
        U, S, Vt = np.linalg.svd(self.B)
        r = int(np.sum(S > 1e-10 * max(S[0], 1.0)))
        N = Vt[r:].T.copy()
        # canonical column orientation: pivot on the max-abs entry per column
        piv = np.argmax(np.abs(N), axis=0)
        signs = np.sign(N[piv, np.arange(N.shape[1])])
        signs[signs == 0] = 1.0
        return N * signs[None, :]

    # ---- topology: biconnected blocks (Hopcroft-Tarjan) ----
    def _undirected_adj(self):
        adj = [[] for _ in range(self.n)]
        for e, (u, v) in enumerate(self.edges):
            if u == v:
                continue
            adj[u].append((v, e))
            adj[v].append((u, e))
        return adj

    def blocks(self) -> list[list[int]]:
        """Biconnected components (each = list of edge ids), via iterative
        Hopcroft-Tarjan edge-stack algorithm.  A bridge appears as a single-edge
        block."""
        if self._blocks is not None:
            return self._blocks
        adj = self._undirected_adj()
        n = self.n
        disc = [-1] * n
        low = [0] * n
        counter = 0
        edge_stack = []
        blocks = []

        for root in range(n):
            if disc[root] != -1:
                continue
            disc[root] = low[root] = counter
            counter += 1
            stack = [(root, -1, 0)]          # (node, parent tree-edge, adj index)
            while stack:
                node, pedge, idx = stack[-1]
                if idx < len(adj[node]):
                    stack[-1] = (node, pedge, idx + 1)
                    nb, e = adj[node][idx]
                    if e == pedge:
                        continue              # never traverse the tree edge back
                    if disc[nb] == -1:        # tree edge
                        edge_stack.append(e)
                        disc[nb] = low[nb] = counter
                        counter += 1
                        stack.append((nb, e, 0))
                    elif disc[nb] < disc[node]:   # back edge (to an ancestor)
                        edge_stack.append(e)
                        low[node] = min(low[node], disc[nb])
                    # else: edge already handled from the descendant side -> skip
                else:
                    stack.pop()
                    if stack:
                        parent = stack[-1][0]
                        low[parent] = min(low[parent], low[node])
                        if low[node] >= disc[parent]:   # parent is a cut vertex
                            blk = []
                            while edge_stack:
                                e = edge_stack.pop()
                                blk.append(e)
                                if e == pedge:
                                    break
                            blocks.append(sorted(blk))
        # any edges never popped (cannot happen for simple graphs; safety net)
        if edge_stack:
            blocks.append(sorted(edge_stack))
        # sanity: every edge in exactly one block
        flat = [e for b in blocks for e in b]
        assert len(flat) == self.m and len(set(flat)) == self.m, \
            f"block decomposition lost edges: {len(flat)} vs {self.m}"
        self._blocks = blocks
        return blocks

    def atoms(self) -> list[dict]:
        """T3c atomic partition: one atom per circuit block (biconnected block with
        a cycle, i.e. >= 3 edges or a multigraph 2-cycle) and one per bridge.

        Self-loops (u == u) also carry a cycle; our networks have none.
        Returns list of {'type': 'circuit'|'bridge', 'edges': [...]}."""
        atoms = []
        for blk in self.blocks():
            # bridge iff block is a single edge whose endpoints form a K2
            if len(blk) == 1:
                u, v = self.edges[blk[0]]
                atom_type = "bridge"
            else:
                atom_type = "circuit"
            atoms.append({"type": atom_type, "edges": blk})
        return atoms

    def block_node_map(self) -> dict[int, list[int]]:
        """node -> atom ids containing it."""
        out = {}
        for k, atom in enumerate(self.atoms()):
            nodes = set()
            for e in atom["edges"]:
                u, v = self.edges[e]
                nodes.add(u)
                nodes.add(v)
            for v in nodes:
                out.setdefault(v, []).append(k)
        return out

    def block_cut_tree(self) -> "nx_like_tree":
        """Bipartite tree: atom nodes 'A{k}' and shared cut-vertex nodes 'V{v}'.
        Returns (adj_dict, atom_count).  Built once per network and cached
        (v17: the previous implementation rebuilt it on every od_path_atoms
        call, which does not support the reported offline preprocessing
        complexity)."""
        if self._bc_tree is None:
            bnm = self.block_node_map()
            adj: dict[str, set[str]] = {}

            def add(a, b):
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

            for v, ks in bnm.items():
                for k in ks:
                    add(f"V{v}", f"A{k}")
            self._bc_tree = adj
        return self._bc_tree

    def _rooted_bc_tree(self):
        """One BFS rooting of the (forest of the) block-cut tree: returns
        (parent, depth) dicts used by od_path_atoms.  Cached.  The s-t atom
        path is the unique tree path, recovered by walking both endpoints up
        to a common ancestor -- O(path length) per query after O(V)
        preprocessing."""
        if self._bc_rooted is None:
            from collections import deque
            adj = self.block_cut_tree()
            parent: dict[str, str | None] = {}
            depth: dict[str, int] = {}
            for start in adj:            # forest-safe: root every component
                if start in parent:
                    continue
                parent[start] = None
                depth[start] = 0
                dq = deque([start])
                while dq:
                    cur = dq.popleft()
                    for nb in adj.get(cur, ()):
                        if nb not in parent:
                            parent[nb] = cur
                            depth[nb] = depth[cur] + 1
                            dq.append(nb)
            self._bc_rooted = (parent, depth)
        return self._bc_rooted

    def od_path_atoms(self, s: int, t: int) -> list[int]:
        """Atoms crossed by every s-t path (the unique s-t path of the block-cut
        tree).  Requires s, t in the same connected component.  v17: uses the
        cached rooted tree (depth-guided ascent) instead of a fresh BFS per
        query; returns the same sorted atom ids as before."""
        parent, depth = self._rooted_bc_tree()
        a, b = f"V{s}", f"V{t}"
        if a not in parent or b not in parent:
            raise ValueError(f"nodes {s},{t} not both in block-cut tree")
        # walk both endpoints up to a common ancestor
        path_a: list[str] = []
        path_b: list[str] = []
        xa, xb = a, b
        while depth[xa] > depth[xb]:
            path_a.append(xa)
            xa = parent[xa]
        while depth[xb] > depth[xa]:
            path_b.append(xb)
            xb = parent[xb]
        while xa != xb:
            path_a.append(xa)
            path_b.append(xb)
            xa = parent[xa]
            xb = parent[xb]
        node_path = path_a + [xa] + list(reversed(path_b))
        if not (depth[a] >= 0 and depth[b] >= 0):
            raise ValueError  # unreachable
        atom_ids = sorted(int(x[1:]) for x in node_path if x.startswith("A"))
        return atom_ids

    def equivalence_classes(self, ods: list[tuple[int, int]]) -> dict:
        """T3c predicted final equivalence classes via transitive closure of
        atoms merged along each OD's block-cut-tree path.  v17: union-find
        with union by rank and path halving (inverse-Ackermann amortized);
        final classes are output in the canonical order of their smallest
        atom id (independent of union internals)."""
        atoms = self.atoms()
        parent = list(range(len(atoms)))
        rank = [0] * len(atoms)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        od_atoms = {}
        for (s, t) in ods:
            ks = self.od_path_atoms(s, t)
            od_atoms[f"{s}->{t}"] = ks
            for a, b in zip(ks, ks[1:]):
                union(a, b)

        classes: dict[int, list[int]] = {}
        for k in range(len(atoms)):
            classes.setdefault(find(k), []).append(k)
        # canonical output order: ascending smallest atom id of each class
        final = []
        for _, members in sorted(classes.items(), key=lambda kv: min(kv[1])):
            edges = sorted(e for k in members for e in atoms[k]["edges"])
            atom_members = sorted(members)
            final.append({"atoms": atom_members, "edges": edges,
                          "types": [atoms[k]["type"] for k in atom_members]})
        return {"atoms": atoms, "od_atoms": od_atoms, "classes": final}

    # ---- paths ----
    def all_simple_paths(self, s: int, t: int, cap: int = 1000) -> list[list[int]]:
        """All directed simple paths s->t as edge lists (DFS, bounded by cap)."""
        adj = [[] for _ in range(self.n)]
        for e, (u, v) in enumerate(self.edges):
            adj[u].append((v, e))
        results = []
        visited = [False] * self.n
        path = []

        def dfs(node):
            if len(results) >= cap:
                return
            visited[node] = True
            if node == t:
                results.append(list(path))
            else:
                for nb, e in adj[node]:
                    if not visited[nb]:
                        path.append(e)
                        dfs(nb)
                        path.pop()
            visited[node] = False

        dfs(s)
        return results

    def k_shortest_paths(self, s: int, t: int, k: int, weight: np.ndarray) -> list[list[int]]:
        """Yen's K shortest loopless directed paths, ties broken by
        lexicographic edge sequence (frozen rule).  weight: (m,) edge cost."""
        adj = [[] for _ in range(self.n)]
        for e, (u, v) in enumerate(self.edges):
            adj[u].append((v, e))

        def dijkstra(src, banned_nodes, banned_edges):
            """Returns (dist, prev).  Ties in dist are broken by keeping the
            lexicographically smallest edge sequence (frozen tie-break rule).
            No permanent 'done' marks: a path within the 1e-12 cost tolerance
            with a lexicographically smaller sequence must be allowed to
            re-expand downstream labels."""
            dist = [float("inf")] * self.n
            seq = [None] * self.n            # best edge-tuple to each node
            prev = [None] * self.n
            dist[src] = 0.0
            seq[src] = ()
            pq = [(0.0, (), src)]
            while pq:
                d, sq, u = heapq.heappop(pq)
                if (d, sq) != (dist[u], seq[u]):
                    continue                 # stale heap entry
                for v, e in adj[u]:
                    if v in banned_nodes or e in banned_edges:
                        continue
                    nd = d + weight[e]
                    nsq = sq + (e,)
                    better = nd < dist[v] - 1e-12
                    equal = abs(nd - dist[v]) <= 1e-12 and (
                        seq[v] is None or nsq < seq[v])
                    if better or equal:
                        dist[v] = nd
                        seq[v] = nsq
                        prev[v] = (u, e)
                        heapq.heappush(pq, (nd, nsq, v))
            return dist, prev

        def path_from(src, t, prev):
            if prev[t] is None and t != src:
                return None
            edges = []
            cur = t
            while cur != src:
                u, e = prev[cur]
                edges.append(e)
                cur = u
            return list(reversed(edges))

        dist0, prev0 = dijkstra(s, set(), set())
        if not np.isfinite(dist0[t]):
            return []
        A = [path_from(s, t, prev0)]
        candidates = []
        seen = {tuple(A[0])}
        for it in range(k - 1):
            for i in range(len(A[it])):
                spur_edge = A[it][i]
                root_edges = A[it][:i]
                root_nodes = {s}
                for e in root_edges:
                    root_nodes.add(self.edges[e][1])
                banned_edges = set()
                banned_nodes = set(root_nodes)
                for p in A:
                    if len(p) > i and p[:i] == root_edges:
                        banned_edges.add(p[i])
                d, pr = dijkstra(self.edges[spur_edge][0], banned_nodes, banned_edges)
                sub = path_from(self.edges[spur_edge][0], t, pr)
                if sub is None:
                    continue
                total_cost = float(sum(weight[e] for e in root_edges) + d[t])
                cand = root_edges + sub
                tc = tuple(cand)
                if tc not in seen:
                    seen.add(tc)
                    heapq.heappush(candidates, (total_cost, tc))
            if not candidates:
                break
            _, best = heapq.heappop(candidates)
            A.append(list(best))
            if len(A) >= k:
                break
        return A

    # ---- E1.6: spanning forest ----
    def spanning_forest(self) -> list[int]:
        """Edge set of a BFS spanning forest (undirected semantics, matching the
        cycle-space dimension m - n + c)."""
        adj = self._undirected_adj()
        seen_node = [False] * self.n
        forest = []
        for root in range(self.n):
            if seen_node[root]:
                continue
            seen_node[root] = True
            dq = [root]
            while dq:
                u = dq.pop()
                for v, e in adj[u]:
                    if not seen_node[v]:
                        seen_node[v] = True
                        forest.append(e)
                        dq.append(v)
        return sorted(forest)

    def connected_components(self) -> int:
        adj = self._undirected_adj()
        seen = [False] * self.n
        c = 0
        for r in range(self.n):
            if seen[r]:
                continue
            c += 1
            seen[r] = True
            dq = [r]
            while dq:
                u = dq.pop()
                for v, _ in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        dq.append(v)
        return c


# ----------------------------------------------------------------------------
# Small theorem-verification networks (E1.4 / E1.6 / E2)
# ----------------------------------------------------------------------------
def braess() -> Network:
    """Braess diamond: 0(s)-1(a), 0-2(b), 1-2, 1-3(t), 2-3.
    Single biconnected block {e0..e4} (no bridges).  OD (0,3)."""
    E = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    return Network("braess", E,
                   capacity=np.ones(5),
                   free_flow_time=np.array([1.0, 1.0, 0.1, 1.0, 1.0]))


def two_bridges() -> Network:
    """Two bridges in series + off-path pendant triangle.
    Tri A on {0,1,2}; bridge e3=(2,3); Tri B on {3,4,5}; bridge e7=(5,6);
    Tri C on {6,7,8}; pendant Tri D on {1,9,10} (off the s-t path).
    OD (0,8): atoms {A, e3, B, e7, C} merge; D stays its own class."""
    E = [(0, 1), (0, 2), (1, 2),          # 0,1,2  Tri A
         (2, 3),                           # bridge 1
         (3, 4), (3, 5), (4, 5),          # Tri B
         (5, 6),                           # bridge 2
         (6, 7), (6, 8), (7, 8),          # Tri C
         (1, 9), (1, 10), (9, 10)]        # pendant Tri D (off path)
    t0 = np.array([1, 1, .5, .5, 1, 1, .5, .5, 1, 1, .5, 1, 1, .5])
    return Network("two_bridges", E, capacity=np.ones(len(E)), free_flow_time=t0)


def shared_cut_loops() -> Network:
    """Shared-cut-vertex loops: Tri1 {0,1,2}, Tri2 shares cut vertex 2 {2,3,4},
    Tri3 shares cut vertex 4 (off main path) {4,5,6}.  OD (0,4) merges Tri1+Tri2;
    Tri3 is an independent class.  Multi-OD variant also uses OD (5,6)."""
    E = [(0, 1), (0, 2), (1, 2),
         (2, 3), (2, 4), (3, 4),
         (4, 5), (4, 6), (5, 6)]
    t0 = np.array([1, 1, .5, 1, 1, .5, 1, 1, .5])
    return Network("shared_cut_loops", E, capacity=np.ones(len(E)), free_flow_time=t0)


def triangular() -> Network:
    """Triangular mesh: Tri {0,1,2} and Tri {1,2,3} share edge e2=(1,2) -> one
    biconnected block {e0..e4}; bridge e5=(3,4) on the OD path; pendant bridge
    e6=(4,5) off the path (free single-edge class).  OD (0,4)."""
    E = [(0, 1), (0, 2), (1, 2),
         (1, 3), (2, 3),
         (3, 4),
         (4, 5)]
    t0 = np.array([1, 1, .5, 1, 1, .5, .3])
    return Network("triangular", E, capacity=np.ones(len(E)), free_flow_time=t0)


def synthetic_mid() -> Network:
    """Synthetic mid-size layered network for E2 (23 arcs, path counts controlled).
    Layers: s=0 -> {1,2} -> {3,4} -> {5,6} -> t=7 with intra-layer edges creating
    cycles, two shortcuts and two back-edges."""
    E = [(0, 1), (0, 2),                  # s -> L1
         (1, 3), (1, 4), (2, 3), (2, 4),  # L1 -> L2
         (3, 5), (3, 6), (4, 5), (4, 6),  # L2 -> L3
         (5, 7), (6, 7),                  # L3 -> t
         (1, 2), (3, 4), (5, 6),          # intra-layer (cycles)
         (0, 3), (4, 7),                  # shortcuts
         (3, 1), (6, 4),                  # back-edges
         (0, 4), (2, 6)]                  # diagonals
    t0 = np.array([1.] * 12 + [.4] * 3 + [1.5] * 2 + [.8] * 2 + [2.0] * 2)
    return Network("synthetic_mid", E, capacity=np.ones(len(E)), free_flow_time=t0)


# ----------------------------------------------------------------------------
# Sioux Falls loader (tntp)
# ----------------------------------------------------------------------------
def load_sioux_falls() -> Network:
    net_path = os.path.join(DATA_DIR, "SiouxFalls_net.tntp")
    edges, cap, t0 = [], [], []
    with open(net_path) as fh:
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
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            c = float(parts[2])
            tt = float(parts[4])
            edges.append((u, v))
            cap.append(c)
            t0.append(tt)
    assert len(edges) == 76, f"Sioux Falls expects 76 links, got {len(edges)}"
    return Network("sioux_falls", edges, capacity=np.array(cap), free_flow_time=np.array(t0))


def load_sioux_falls_trips() -> dict[tuple[int, int], float]:
    """Full OD demand matrix as {(0-based o, 0-based d): flow}."""
    path = os.path.join(DATA_DIR, "SiouxFalls_trips.tntp")
    demands = {}
    origin = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.lower().startswith("origin"):
                origin = int(line.split()[1]) - 1
                continue
            if origin is None:
                continue
            for mm in re.finditer(r"(\d+)\s*:\s*([\d.eE+-]+)\s*;", line):
                d = int(mm.group(1)) - 1
                demands[(origin, d)] = float(mm.group(2))
    assert len(demands) == 24 * 24, f"expected 576 OD cells, got {len(demands)}"
    return demands


def sioux_falls_top8_od() -> list[tuple[int, int]]:
    """Frozen selection rule (E0.1): top-8 OD pairs by demand, ties broken by
    ascending (origin, destination) node id."""
    demands = load_sioux_falls_trips()
    pairs = [(d, o, t) for (o, t), d in demands.items() if d > 0 and o != t]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(o, t) for _, o, t in pairs[:8]]


__all__ = ["Network", "braess", "two_bridges", "shared_cut_loops", "triangular",
           "synthetic_mid", "load_sioux_falls", "load_sioux_falls_trips",
           "sioux_falls_top8_od", "DATA_DIR"]
