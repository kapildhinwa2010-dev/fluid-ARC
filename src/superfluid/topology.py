"""Graph topology: top-k edge construction, Kruskal barcode, Betti
numbers, and the O(n + m) Hodge projection.

Pure NumPy. Every block selects its edges here, and the harmonic
projection that closes each block lives here too: it subtracts the
per-component mean, which is exactly the kernel of the sheaf
Laplacian."""

import numpy as np


class UnionFind:
    """Disjoint sets with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_components -= 1
        return True


def topk_edges(scores, k):
    """Per-row top-k selection collapsed to unique undirected edges.

    Self-loops are dropped; an edge (i, j) with i < j survives when
    j is among the top-k of row i or i is among the top-k of row j.
    Returns a sorted list of (i, j) pairs."""
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n < 2:
        return []
    k = max(1, min(int(k), n - 1))
    edges = set()
    for i in range(n):
        row = scores[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(-row, k - 1)[:k]
        for j in idx:
            j = int(j)
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def edge_mask(n, edges):
    """Dense symmetric 0/1 mask of an undirected edge list."""
    M = np.zeros((n, n))
    for i, j in edges:
        M[i, j] = 1.0
        M[j, i] = 1.0
    return M


def kruskal_barcode(n, edges, weights):
    """0-dimensional persistence barcode via Kruskal filtration.

    Edges enter in weight order; every merge kills one component
    born at 0. Returns (barcode, survivors): barcode is the list of
    (birth, death) pairs in death order, survivors is the number of
    components that never die."""
    edges = [(int(i), int(j)) for i, j in edges]
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(weights, kind="stable")
    uf = UnionFind(n)
    barcode = []
    for oi in order:
        if uf.union(*edges[int(oi)]):
            barcode.append((0.0, float(weights[oi])))
    return barcode, uf.n_components


def betti_numbers(n, edges):
    """Graph ranks: beta0 = components, beta1 = m - n + beta0."""
    uf = UnionFind(n)
    for i, j in edges:
        uf.union(int(i), int(j))
    beta0 = uf.n_components
    return beta0, len(edges) - n + beta0


def persistence_descriptor(barcode, survivors=1):
    """Fixed-length summary of a 0-D barcode."""
    if barcode:
        deaths = np.array([d for _, d in barcode], dtype=float)
        return {
            "count": int(deaths.size),
            "total_persistence": float(deaths.sum()),
            "max_death": float(deaths.max()),
            "mean_death": float(deaths.mean()),
            "survivors": int(survivors),
        }
    return {"count": 0, "total_persistence": 0.0, "max_death": 0.0,
            "mean_death": 0.0, "survivors": int(survivors)}


def component_ids(n, edges):
    """Union-find component label per node, roots remapped to 0..C-1."""
    uf = UnionFind(n)
    for i, j in edges:
        uf.union(int(i), int(j))
    remap = {}
    cid = np.empty(n, dtype=int)
    for i in range(n):
        r = uf.find(i)
        if r not in remap:
            remap[r] = len(remap)
        cid[i] = remap[r]
    return cid, len(remap)


def harmonic_projection(x, edges, n):
    """Remove the harmonic part: subtract the per-component mean.

    The kernel of the sheaf Laplacian is exactly the piecewise-constant
    vectors, so this is the Hodge projection onto the orthogonal
    complement of the kernel. O(n + m): two union-find passes and one
    scatter, no eigensolver. Returns (projected, component_ids)."""
    x = np.asarray(x, dtype=float)
    cid, C = component_ids(n, edges)
    sums = np.zeros(C)
    counts = np.zeros(C)
    np.add.at(sums, cid, x)
    np.add.at(counts, cid, 1.0)
    h = (sums / counts)[cid]
    return x - h, cid

