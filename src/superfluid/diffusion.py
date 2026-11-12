"""Spectral scale, Chebyshev diffusion, band filter bank, jam time.

Pure NumPy - no torch, no scipy. lambda_max comes from the
Gershgorin bound (differentiable, never underestimates - the power
iteration once did, pushing the Chebyshev recurrence outside its
stable region). The Chebyshev recurrence carries the factor 2:
T_k = 2 L~ T_{k-1} - T_{k-2}. Band coefficients use exp(-alpha),
never exp(+alpha) - both regressions are tested."""

import numpy as np


def gershgorin_bound(L):
    """Upper bound on lambda_max: max row sum of |L|. Never underestimates."""
    return float(np.max(np.sum(np.abs(L), axis=1)))


def power_iteration_lmax(L, iters=300, tol=1e-13, seed=0):
    """Reference lambda_max estimate via power iteration + Rayleigh quotient."""
    v = np.random.default_rng(seed).normal(size=L.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = L @ v
        lam = float(v @ (L @ v))
        nw = np.linalg.norm(w)
        if nw < 1e-300:
            return 0.0
        v = w / nw
    return lam


def chebyshev_scaled(L, lam):
    """L~ = (2 / lam) L - I, spectrum mapped into [-1, 1]."""
    return (2.0 / lam) * L - np.eye(L.shape[0])


def chebyshev_basis(L, lam, x, m):
    """Y[k] = T_k(L~) x for k = 0..m-1, shared by every band.

    T_0 = x, T_1 = L~ x, T_k = 2 L~ T_{k-1} - T_{k-2}. The factor 2
    is load-bearing: without it the expansion plateaued at 78% error."""
    Lt = chebyshev_scaled(L, lam)
    Y = [np.asarray(x, dtype=float).copy()]
    if m > 1:
        Y.append(Lt @ Y[0])
    for _ in range(2, m):
        Y.append(2.0 * (Lt @ Y[-1]) - Y[-2])
    return Y


def chebyshev_coefficients_exp(t, lam, m):
    """Coefficients of exp(-tL) in the Chebyshev basis of L~.

    With s = t * lam / 2: exp(-tL) = e^{-s} sum_k (-s)^k / k! T_k(L~).
    c_0 = e^{-s}; c_k = e^{-s} * 2 * (-s)^k / k! for k >= 1."""
    s = 0.5 * float(t) * float(lam)
    c = np.zeros(m)
    if s <= 0.0:
        c[0] = 1.0
        return c
    ks = np.arange(m)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, m)))])
    c = np.exp(-s + ks * np.log(s) + np.log(2.0) - log_fact) * ((-1.0) ** ks)
    c[0] = np.exp(-s)
    return c


def expm_multiply(L, t, x, degree=40, lam=None):
    """exp(-t L) x via the Chebyshev expansion. No scipy."""
    if lam is None:
        lam = max(gershgorin_bound(L), 1e-12)
    Y = chebyshev_basis(L, lam, x, degree)
    c = chebyshev_coefficients_exp(t, lam, degree)
    out = np.zeros_like(Y[0])
    for ck, y in zip(c, Y):
        out = out + ck * y
    return out


def band_filter_bank(L, lam, x, ts, degree=40):
    """band_j = (exp(-t_j L) - exp(-t_{j+1} L)) x for j = 0..B-1.

    All bands share one Chebyshev basis: m matvecs total, not m per
    band. Coefficients subtract as (c_j - c_{j+1}) - the sign is
    exp(-alpha), and the stability test asserts it stays that way."""
    x = np.asarray(x, dtype=float)
    Y = chebyshev_basis(L, lam, x, degree)
    cs = [chebyshev_coefficients_exp(t, lam, degree) for t in ts]
    bands = []
    for j in range(len(ts) - 1):
        dc = cs[j] - cs[j + 1]
        b = np.zeros_like(Y[0])
        for k in range(degree):
            b = b + dc[k] * Y[k]
        bands.append(b)
    return bands


def spectral_gap(L):
    """lambda_2: the smallest strictly positive eigenvalue."""
    w = np.sort(np.linalg.eigvalsh(L))
    nz = w[w > 1e-10]
    return float(nz[0]) if nz.size else 0.0


def jam_time(L, tol=1e-3):
    """Spectral time to jam within tol: t = ln(1/tol) / lambda_2.

    Bound: ||exp(-tL)(x - mean)|| <= e^{-t lambda_2} ||x - mean||,
    so after t the transient has decayed by exactly tol."""
    lam2 = spectral_gap(L)
    if lam2 <= 0.0:
        return float("inf")
    return float(np.log(1.0 / tol) / lam2)