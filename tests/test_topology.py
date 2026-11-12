import numpy as np
import pytest

from superfluid.diffusion import spectral_gap
from superfluid.sheaf import laplacian_from_edges
from superfluid.topology import (UnionFind, betti_numbers, component_ids,
                                 edge_mask, harmonic_projection,
                                 kruskal_barcode, persistence_descriptor,
                                 topk_edges)


def test_topk_edges_structure():
    """Unique, undirected, no self-loops, sorted, at most n*k edges."""
    rng = np.random.default_rng(20)
    scores = rng.normal(size=(8, 8))
    scores = 0.5 * (scores + scores.T)
    edges = topk_edges(scores, 2)
    assert edges == sorted(set(edges))
    for i, j in edges:
        assert i < j
    assert len(edges) <= 8 * 2


def test_topk_edges_respects_row_winners():
    """Every chosen edge is a masked row top-k winner in one direction."""
    rng = np.random.default_rng(21)
    scores = rng.normal(size=(6, 6))
    edges = topk_edges(scores, 1)

    def top1(i):
        row = scores[i].copy()
        row[i] = -np.inf
        return int(np.argmax(row))

    for i, j in edges:
        assert j == top1(i) or i == top1(j)


def test_topk_edges_single_node():
    assert topk_edges(np.zeros((1, 1)), 3) == []


def test_kruskal_barcode_path_graph():
    """Path 0-1-2-3, weights shuffled: deaths come in weight order."""
    edges = [(0, 1), (1, 2), (2, 3)]
    barcode, survivors = kruskal_barcode(4, edges, [3.0, 1.0, 2.0])
    assert survivors == 1
    assert [d for _, d in barcode] == [1.0, 2.0, 3.0]
    assert all(b == 0.0 for b, _ in barcode)


def test_kruskal_barcode_disconnected():
    """Isolated node: two components survive, one merge fewer."""
    barcode, survivors = kruskal_barcode(4, [(0, 1), (1, 2)], [5.0, 1.0])
    assert survivors == 2
    assert [d for _, d in barcode] == [1.0, 5.0]


def test_betti_numbers():
    """Tree: (1, 0). Triangle: (1, 1). Two triangles: (2, 2)."""
    assert betti_numbers(4, [(0, 1), (1, 2), (2, 3)]) == (1, 0)
    assert betti_numbers(3, [(0, 1), (1, 2), (0, 2)]) == (1, 1)
    tri = [(0, 1), (1, 2), (0, 2)]
    both = tri + [(a + 3, b + 3) for a, b in tri]
    assert betti_numbers(6, both) == (2, 2)


def test_persistence_descriptor():
    barcode, survivors = kruskal_barcode(4, [(0, 1), (1, 2), (2, 3)], [1.0, 2.0, 3.0])
    desc = persistence_descriptor(barcode, survivors)
    assert desc["count"] == 3
    assert desc["total_persistence"] == pytest.approx(6.0)
    assert desc["max_death"] == pytest.approx(3.0)
    assert desc["survivors"] == 1
    empty = persistence_descriptor([], survivors=2)
    assert empty["count"] == 0 and empty["survivors"] == 2


def test_harmonic_projection_removes_component_means():
    """Connected: result sums to zero. Disconnected: each component does."""
    rng = np.random.default_rng(22)
    x = rng.normal(size=7)
    out, cid = harmonic_projection(x, [(i, i + 1) for i in range(6)], 7)
    assert out.sum() == pytest.approx(0.0, abs=1e-12)
    x2 = rng.normal(size=6)
    out2, cid2 = harmonic_projection(x2, [(0, 1), (1, 2), (3, 4)], 6)
    for c in range(cid2.max() + 1):
        assert out2[cid2 == c].sum() == pytest.approx(0.0, abs=1e-12)


def test_harmonic_part_in_laplacian_kernel():
    """The removed part h is piecewise constant, so L h = 0 exactly."""
    rng = np.random.default_rng(23)
    x = rng.normal(size=5)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    Lap = laplacian_from_edges(5, edges, np.ones(4))
    _, cid = harmonic_projection(x, edges, 5)
    sums = np.zeros(cid.max() + 1)
    counts = np.zeros(cid.max() + 1)
    np.add.at(sums, cid, x)
    np.add.at(counts, cid, 1.0)
    h = (sums / counts)[cid]
    assert np.allclose(Lap @ h, 0.0, atol=1e-12)


def test_harmonic_projection_idempotent():
    rng = np.random.default_rng(24)
    x = rng.normal(size=6)
    edges = [(i, (i + 1) % 6) for i in range(6)]
    out, _ = harmonic_projection(x, edges, 6)
    out2, _ = harmonic_projection(out, edges, 6)
    assert np.allclose(out, out2, atol=1e-12)


def test_topk_then_laplacian_connected():
    """End-to-end: top-k selection plus the temporal chain keeps the
    ring connected, so the spectral gap stays positive."""
    rng = np.random.default_rng(25)
    n = 8
    scores = rng.normal(size=(n, n))
    chain = [(i, (i + 1) % n) for i in range(n)]
    edges = sorted(set(topk_edges(scores, 2)) | set(chain))
    Lap = laplacian_from_edges(n, edges, np.ones(len(edges)))
    assert spectral_gap(Lap) > 0.0


def test_union_find_components():
    uf = UnionFind(5)
    assert uf.union(0, 1) and uf.union(3, 4)
    assert not uf.union(1, 0)
    assert uf.n_components == 3
    assert uf.find(0) == uf.find(1)
    assert uf.find(2) != uf.find(3)

