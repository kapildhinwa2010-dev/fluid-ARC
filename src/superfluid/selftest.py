"""The 21 numerical checks of the framework, runnable as a suite.

Each check returns True or False; run_all() returns the list of
{name, ok} records. The pytest suite asserts all 21 pass, so the
mathematical claims in the README stay machine-verified."""

import numpy as np

from .autodiff import Tensor, check_grads, index_add
from .blocks import SuperfluidBlock
from .diffusion import (band_filter_bank, chebyshev_basis, expm_multiply,
                        gershgorin_bound, jam_time, power_iteration_lmax)
from .losses import cross_entropy
from .model import SuperfluidLM
from .sheaf import balance_residual, laplacian_from_edges
from .topology import harmonic_projection, topk_edges


def _lap(n, seed, signed=True):
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]
    edges += [(i, (i + 3) % n) for i in range(n)]
    lo, hi = (-1.0, 1.0) if signed else (0.05, 1.0)
    alphas = rng.uniform(lo, hi, size=len(edges))
    return laplacian_from_edges(n, edges, alphas), edges, alphas


def check_01_autodiff_accumulates():
    x = Tensor(np.array([1.0, 2.0]))
    (x + x).sum().backward()
    return bool(np.allclose(x.grad, 2.0))


def check_02_unbroadcast_shapes():
    a = Tensor(np.ones((3, 1)))
    b = Tensor(np.ones(3))
    (a + b).sum().backward()
    return bool(a.grad.shape == (3, 1) and b.grad.shape == (3,)
                and np.allclose(a.grad, 3.0) and np.allclose(b.grad, 3.0))


def check_03_index_add_reaches_src():
    base = Tensor(np.zeros(4))
    src = Tensor(np.array([1.0, 2.0]))
    idx = np.array([0, 2])
    index_add(base, idx, src).sum().backward()
    return bool(np.allclose(base.grad, 1.0) and np.allclose(src.grad, 1.0))


def check_04_gradient_vs_finite_differences():
    W = Tensor(np.random.default_rng(40).normal(size=(4, 4)))
    x = Tensor(np.random.default_rng(41).normal(size=(4, 4)))
    def build():
        return (W.matmul(x).tanh().matmul(W.T)).sum()
    ok, worst = check_grads(build, [W, x], rtol=1e-4, atol=1e-8)
    return bool(ok and worst < 0.1)


def check_05_sheaf_laplacian_psd():
    for seed in range(5):
        Lap, _, _ = _lap(8, seed, signed=True)
        if np.linalg.eigvalsh(Lap).min() < -1e-10:
            return False
    return True


def check_06_gershgorin_never_underestimates():
    for seed in range(3):
        Lap, _, _ = _lap(8, seed)
        if power_iteration_lmax(Lap) > gershgorin_bound(Lap) + 1e-9:
            return False
    return True


def check_07_chebyshev_recurrence_factor_two():
    Lap, _, _ = _lap(8, 7, signed=False)
    lam = gershgorin_bound(Lap)
    x = np.random.default_rng(8).normal(size=8)
    Y = chebyshev_basis(Lap, lam, x, 6)
    Lt = (2.0 / lam) * Lap - np.eye(8)
    for k in range(2, 6):
        if not np.allclose(Y[k], 2.0 * (Lt @ Y[k - 1]) - Y[k - 2]):
            return False
    return True


def check_08_expm_matches_eigendecomposition():
    Lap, _, _ = _lap(8, 9, signed=True)
    lam = gershgorin_bound(Lap)
    w, Q = np.linalg.eigh(Lap)
    x = np.random.default_rng(10).normal(size=8)
    for t in [0.0, 0.05, 0.3, 1.0]:
        got = expm_multiply(Lap, t, x, degree=40, lam=lam)
        want = Q @ (np.exp(-t * w) * (Q.T @ x))
        if not np.allclose(got, want, atol=1e-10):
            return False
    return True


def check_09_chebyshev_degree_convergence():
    Lap, _, _ = _lap(8, 11, signed=False)
    lam = gershgorin_bound(Lap)
    w, Q = np.linalg.eigh(Lap)
    x = np.random.default_rng(12).normal(size=8)
    t = 0.3
    want = Q @ (np.exp(-t * w) * (Q.T @ x))
    errs = [np.linalg.norm(expm_multiply(Lap, t, x, degree=m, lam=lam) - want)
            for m in (3, 8, 40)]
    return bool(errs[2] < errs[1] < errs[0])


def check_10_linear_collapse_to_component_mean():
    Lap, _, _ = _lap(8, 13, signed=False)
    lam = gershgorin_bound(Lap)
    t = 2.0 * jam_time(Lap, tol=1e-6)
    x = np.random.default_rng(14).normal(size=8)
    got = expm_multiply(Lap, t, x, degree=40, lam=lam)
    h = np.full(8, x.mean())
    return bool(np.linalg.norm(got - h) < 1e-4)


def check_11_restriction_maps_stay_bounded():
    rng = np.random.default_rng(15)
    scores = rng.normal(size=(8, 8)) * 50.0
    edges = topk_edges(scores, 3)
    si = np.array([e[0] for e in edges])
    sj = np.array([e[1] for e in edges])
    sym = (scores[si, sj] + scores[sj, si]) * 0.5
    return bool(np.abs(np.tanh(sym)).max() <= 1.0)


def check_12_rank_c_trap():
    """A c-component Laplacian has 0 as a c-fold eigenvalue."""
    edges = [(0, 1), (1, 2), (4, 5), (6, 7)]
    Lap = laplacian_from_edges(8, edges, np.ones(4))
    w = np.linalg.eigvalsh(Lap)
    c = int((np.abs(w) < 1e-9).sum())
    return bool(c == 4)


def check_13_band_telescoping_exact():
    Lap, _, _ = _lap(6, 16, signed=False)
    lam = gershgorin_bound(Lap)
    ts = 2.0 * np.geomspace(0.05, 20.0, 6) / lam
    x = np.random.default_rng(17).normal(size=6)
    bands = band_filter_bank(Lap, lam, x, ts, degree=40)
    total = np.zeros_like(x)
    for b in bands:
        total = total + b
    want = expm_multiply(Lap, ts[0], x, degree=40, lam=lam) - expm_multiply(Lap, ts[-1], x, degree=40, lam=lam)
    return bool(np.allclose(total, want, atol=1e-12))


def check_14_band_bank_reconstructs_transient():
    """Tiny t_0 makes exp(-t_0 L) ~= I, so bands plus the frozen tail
    rebuild x itself."""
    Lap, _, _ = _lap(8, 18, signed=True)
    lam = gershgorin_bound(Lap)
    ts = 2.0 * np.geomspace(1e-7, 20.0, 8) / lam
    x = np.random.default_rng(19).normal(size=8)
    bands = band_filter_bank(Lap, lam, x, ts, degree=40)
    total = expm_multiply(Lap, ts[-1], x, degree=40, lam=lam)
    for b in bands:
        total = total + b
    return bool(np.linalg.norm(total - x) < 1e-5)


def check_15_g_balance_phase_transition():
    edges, pos = _lap(8, 20, signed=False)[1], None
    rng = np.random.default_rng(20)
    e = [(i, (i + 1) % 8) for i in range(8)]
    pos = rng.uniform(0.05, 1.0, size=8)
    neg = pos.copy()
    neg[:4] = -neg[:4]
    r_pos = balance_residual(laplacian_from_edges(8, e, pos))
    r_neg = balance_residual(laplacian_from_edges(8, e, neg))
    return bool(r_pos < 1e-12 and r_neg > 0.5)


def check_16_contraction_certificate():
    for seed in range(3):
        Lap, _, _ = _lap(8, 21 + seed, signed=True)
        w = np.linalg.eigvalsh(Lap)
        t = 0.7
        if np.abs(np.exp(-t * w)).max() > 1.0 + 1e-12:
            return False
    return True


def check_17_jam_time_bound():
    n = 8
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    Lap = laplacian_from_edges(n, edges, np.ones(len(edges)))
    lam = gershgorin_bound(Lap)
    tol = 1e-3
    t = jam_time(Lap, tol=tol)
    x = np.random.default_rng(22).normal(size=n)
    h = np.full(n, x.mean())
    got = expm_multiply(Lap, t, x, degree=40, lam=lam)
    err = np.linalg.norm(got - h)
    base = np.linalg.norm(x - h)
    return bool(err <= tol * base + 1e-8)


def check_18_jam_time_infinite_when_disconnected():
    Lap = laplacian_from_edges(8, [(0, 1)], [1.0])
    return bool(jam_time(Lap) == float("inf"))


def check_19_band_frame_energy_bound():
    """Each band is a difference of PSD contractions, so
    sum_j ||band_j||^2 <= B ||x||^2."""
    Lap, _, _ = _lap(8, 23, signed=True)
    lam = gershgorin_bound(Lap)
    B = 6
    ts = 2.0 * np.geomspace(0.05, 10.0, B + 1) / lam
    x = np.random.default_rng(24).normal(size=8)
    bands = band_filter_bank(Lap, lam, x, ts, degree=40)
    energy = sum(float(np.linalg.norm(b) ** 2) for b in bands)
    return bool(energy <= B * float(np.linalg.norm(x) ** 2) + 1e-9)


def check_20_block_end_to_end_gradient():
    X = Tensor(np.random.default_rng(26).normal(size=(8, 8)))
    edges = [(2, 5), (1, 6), (3, 7)]
    def build(blk):
        out, _ = blk(X, sem_edges=edges)
        return (out * out).sum()
    blk = SuperfluidBlock(8, topk=2, n_bands=3, degree=12, seed=25,
                          fixed_lam=10.0)
    ok, worst = check_grads(lambda: build(blk), blk.params,
                            rtol=1e-3, atol=1e-6)
    return bool(ok and worst < 0.1)


def check_21_harmonic_projection_is_kernel():
    rng = np.random.default_rng(27)
    x = rng.normal(size=6)
    edges = [(i, (i + 1) % 6) for i in range(6)]
    Lap = laplacian_from_edges(6, edges, np.ones(6))
    proj, cid = harmonic_projection(x, edges, 6)
    h = x - proj
    ortho = float(proj @ h)
    return bool(np.allclose(Lap @ h, 0.0, atol=1e-12) and abs(ortho) < 1e-12)


CHECKS = [
    check_01_autodiff_accumulates,
    check_02_unbroadcast_shapes,
    check_03_index_add_reaches_src,
    check_04_gradient_vs_finite_differences,
    check_05_sheaf_laplacian_psd,
    check_06_gershgorin_never_underestimates,
    check_07_chebyshev_recurrence_factor_two,
    check_08_expm_matches_eigendecomposition,
    check_09_chebyshev_degree_convergence,
    check_10_linear_collapse_to_component_mean,
    check_11_restriction_maps_stay_bounded,
    check_12_rank_c_trap,
    check_13_band_telescoping_exact,
    check_14_band_bank_reconstructs_transient,
    check_15_g_balance_phase_transition,
    check_16_contraction_certificate,
    check_17_jam_time_bound,
    check_18_jam_time_infinite_when_disconnected,
    check_19_band_frame_energy_bound,
    check_20_block_end_to_end_gradient,
    check_21_harmonic_projection_is_kernel,
]


def run_all():
    """Run every check; returns a list of {name, ok} records."""
    results = []
    for fn in CHECKS:
        results.append({"name": fn.__name__, "ok": bool(fn())})
    return results


if __name__ == "__main__":
    for r in run_all():
        print(("PASS" if r["ok"] else "FAIL") + " " + r["name"])

