import numpy as np
import pytest

from superfluid.autodiff import Tensor, check_grads
from superfluid.diffusion import (band_filter_bank, chebyshev_basis,
                                  chebyshev_coefficients_exp, expm_multiply,
                                  gershgorin_bound, jam_time,
                                  power_iteration_lmax, spectral_gap)
from superfluid.sheaf import (balance_residual, laplacian_from_edges,
                              laplacian_tensor, total_variation)


def _edges_and_alphas(n, seed=0, signed=True):
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]
    edges += [(i, (i + 3) % n) for i in range(n)]
    lo, hi = (-1.0, 1.0) if signed else (0.05, 1.0)
    alphas = rng.uniform(lo, hi, size=len(edges))
    return edges, alphas


def test_laplacian_psd_signed():
    """Gershgorin guarantee: signed sheaf Laplacian is PSD by construction."""
    for seed in range(5):
        edges, alphas = _edges_and_alphas(8, seed=seed, signed=True)
        Lap = laplacian_from_edges(8, edges, alphas)
        assert np.allclose(Lap, Lap.T)
        assert np.linalg.eigvalsh(Lap).min() > -1e-10


def test_tv_identity_unsigned():
    """For unsigned weights: x^T L x equals the total variation energy."""
    edges, alphas = _edges_and_alphas(7, seed=1, signed=False)
    Lap = laplacian_from_edges(7, edges, alphas)
    rng = np.random.default_rng(2)
    x = rng.normal(size=7)
    assert float(x @ Lap @ x) == pytest.approx(total_variation(x, edges, alphas))


def test_balance_residual_phase():
    """All-positive maps: constants in the kernel (residual 0).
    Signed mass: residual grows - the G-balance phase transition."""
    edges, pos = _edges_and_alphas(8, seed=3, signed=False)
    assert balance_residual(laplacian_from_edges(8, edges, pos)) == pytest.approx(0.0, abs=1e-12)
    neg = pos.copy()
    neg[:8] = -neg[:8]
    assert balance_residual(laplacian_from_edges(8, edges, neg)) > 0.5


def test_tensor_laplacian_matches_numpy():
    edges, alphas = _edges_and_alphas(8, seed=4)
    Lt = laplacian_tensor(Tensor(alphas), edges, 8)
    Ln = laplacian_from_edges(8, edges, alphas)
    assert np.allclose(Lt.data, Ln)


def test_tensor_laplacian_grad_reaches_alphas():
    """Regression: restriction maps once received no gradient at all."""
    edges, alphas = _edges_and_alphas(6, seed=5)
    at = Tensor(alphas)
    xt = Tensor(np.array([1.0, -0.5, 0.3, 2.0, -1.2, 0.7]))
    def build():
        Lt = laplacian_tensor(at, edges, 6)
        return (Lt.matmul(xt) * xt).sum()
    ok, worst = check_grads(build, [at], rtol=1e-4, atol=1e-8)
    assert ok and worst < 1e-4


def test_gershgorin_bounds_lmax():
    Lap = np.diag([2.0, 3.0, 5.0])
    assert gershgorin_bound(Lap) == pytest.approx(5.0)
    assert power_iteration_lmax(Lap) == pytest.approx(5.0, rel=1e-6)
    edges, alphas = _edges_and_alphas(8, seed=6)
    Lap = laplacian_from_edges(8, edges, alphas)
    assert power_iteration_lmax(Lap) <= gershgorin_bound(Lap) + 1e-9


def test_chebyshev_recurrence_factor_two():
    """Regression: the missing factor 2 plateaued the expansion."""
    edges, alphas = _edges_and_alphas(8, seed=7, signed=False)
    Lap = laplacian_from_edges(8, edges, alphas)
    lam = gershgorin_bound(Lap)
    rng = np.random.default_rng(8)
    x = rng.normal(size=8)
    Y = chebyshev_basis(Lap, lam, x, 6)
    Lt = (2.0 / lam) * Lap - np.eye(8)
    for k in range(2, 6):
        assert np.allclose(Y[k], 2.0 * (Lt @ Y[k - 1]) - Y[k - 2])


def test_expm_matches_eigendecomposition():
    edges, alphas = _edges_and_alphas(8, seed=9, signed=True)
    Lap = laplacian_from_edges(8, edges, alphas)
    lam = gershgorin_bound(Lap)
    w, Q = np.linalg.eigh(Lap)
    rng = np.random.default_rng(10)
    x = rng.normal(size=8)
    for t in [0.0, 0.05, 0.3, 1.0]:
        got = expm_multiply(Lap, t, x, degree=40, lam=lam)
        want = Q @ (np.exp(-t * w) * (Q.T @ x))
        assert np.allclose(got, want, atol=1e-10)


def test_band_telescoping_exact():
    """Sum of bands = exp(-t_0 L) x - exp(-t_B L) x, coefficient-exact."""
    edges, alphas = _edges_and_alphas(6, seed=11, signed=False)
    Lap = laplacian_from_edges(6, edges, alphas)
    lam = gershgorin_bound(Lap)
    ts = 2.0 * np.geomspace(0.05, 20.0, 6) / lam
    rng = np.random.default_rng(12)
    x = rng.normal(size=6)
    bands = band_filter_bank(Lap, lam, x, ts, degree=40)
    total = np.zeros_like(x)
    for b in bands:
        total = total + b
    want = expm_multiply(Lap, ts[0], x, degree=40, lam=lam) - expm_multiply(Lap, ts[-1], x, degree=40, lam=lam)
    assert np.allclose(total, want, atol=1e-12)


def test_band_stability_contraction():
    """Regression: exp(+alpha) once multiplied bands by e^2alpha ~ 1e6.
    L is PSD, so exp(-tL) is a contraction and every band obeys
    ||band|| <= 2 ||x||."""
    edges, alphas = _edges_and_alphas(8, seed=13, signed=True)
    Lap = laplacian_from_edges(8, edges, alphas)
    lam = gershgorin_bound(Lap)
    ts = 2.0 * np.geomspace(0.05, 10.0, 6) / lam
    rng = np.random.default_rng(14)
    x = rng.normal(size=8)
    bands = band_filter_bank(Lap, lam, x, ts, degree=40)
    limit = 2.0 * np.linalg.norm(x) * 1.01
    for b in bands:
        assert np.linalg.norm(b) <= limit


def test_jam_time_bound():
    """The jamming lemma: after t = ln(1/tol)/lambda_2 the transient
    has decayed by exactly tol (complete graph keeps s small enough
    for the Chebyshev expansion)."""
    n = 8
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    alphas = np.ones(len(edges))
    Lap = laplacian_from_edges(n, edges, alphas)
    lam = gershgorin_bound(Lap)
    tol = 1e-3
    t = jam_time(Lap, tol=tol)
    rng = np.random.default_rng(15)
    x = rng.normal(size=n)
    h = np.full(n, x.mean())
    got = expm_multiply(Lap, t, x, degree=40, lam=lam)
    err = np.linalg.norm(got - h)
    base = np.linalg.norm(x - h)
    assert err <= tol * base + 1e-8


def test_spectral_gap_connected_vs_disconnected():
    edges, alphas = _edges_and_alphas(8, seed=16, signed=False)
    Lap = laplacian_from_edges(8, edges, alphas)
    assert spectral_gap(Lap) > 0.0
    lonely = laplacian_from_edges(8, [(0, 1)], [1.0])
    assert spectral_gap(lonely) == 0.0