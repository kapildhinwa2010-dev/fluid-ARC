"""Signed sheaf Laplacian, G-balance, and total variation energy.

The Laplacian is L = D_abs - A: signed restriction maps alpha_uv in
[-1, 1] with absolute-value degree. Gershgorin makes it PSD by
construction (regression: the original build used A - D, which is
non-PSD and overflows exp(-tL)). The differentiable path scatters
alphas with index_add, so restriction maps receive gradients.
"""

import numpy as np

from .autodiff import Tensor, index_add


def laplacian_from_edges(n, edges, alphas):
    """Dense signed sheaf Laplacian L = D_abs - A (NumPy path)."""
    L = np.zeros((n, n))
    for (u, v), a in zip(edges, np.asarray(alphas, dtype=float)):
        L[u, v] += -a
        L[v, u] += -a
        L[u, u] += abs(a)
        L[v, v] += abs(a)
    return L


def total_variation(x, edges, alphas):
    """TV energy: sum over edges of |alpha| * (x_u - x_v)^2.

    For unsigned weights this equals the quadratic form x^T L x."""
    tv = 0.0
    for (u, v), a in zip(edges, np.asarray(alphas, dtype=float)):
        tv += abs(a) * (x[u] - x[v]) ** 2
    return float(tv)


def balance_residual(L):
    """G-balance indicator: the infinity norm of L @ 1.

    Zero when all restriction maps are positive (constants lie in the
    kernel); grows with the signed mass - the phase transition signal."""
    return float(np.max(np.abs(L @ np.ones(L.shape[0]))))


def laplacian_tensor(alphas, edges, n):
    """Differentiable Laplacian: scatter signed weights and abs-degrees.

    Gradients reach the restriction maps because index_add backward
    reaches src (the regression that once left the whole fluid inert)."""
    m = len(edges)
    if m == 0:
        return Tensor(np.zeros((n, n)))
    rows = np.array([e[0] for e in edges])
    cols = np.array([e[1] for e in edges])
    abs_a = alphas.abs()
    A = index_add(Tensor(np.zeros((n, n))), (rows, cols), alphas)
    A = index_add(A, (cols, rows), alphas)
    D = index_add(Tensor(np.zeros((n, n))), (rows, rows), abs_a)
    D = index_add(D, (cols, cols), abs_a)
    return D + (-A)