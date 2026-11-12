"""The SuperfluidBlock: the six-stage fluid stack.

Router scores -> top-k semantic edges plus the temporal chain ->
signed tanh-bounded restriction maps -> sheaf Laplacian -> shared-
basis Chebyshev band bank in spectral time -> gated sum with the
harmonic branch -> gelu mixing -> residual. The Gershgorin scale
that sets the spectral time grid is detached: gradients flow
through the scaled operator, the band gates, and the mixer."""

import numpy as np

from .autodiff import Tensor, concat, index_add
from .diffusion import chebyshev_coefficients_exp, gershgorin_bound
from .sheaf import laplacian_tensor
from .topology import component_ids, topk_edges


def rmsnorm(x, weight, eps=1e-5):
    """Row-wise RMS normalization with a learned scale."""
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x * weight / (ms + eps).sqrt()


class SuperfluidBlock:
    def __init__(self, d_model, topk=4, n_bands=6, chain_beta=1.0,
                 a_lo=0.05, a_hi=20.0, degree=24, seed=0):
        rng = np.random.default_rng(seed)
        self.d = int(d_model)
        self.topk = int(topk)
        self.n_bands = int(n_bands)
        self.chain_beta = float(chain_beta)
        self.a_lo = float(a_lo)
        self.a_hi = float(a_hi)
        self.degree = int(degree)
        self.Wq = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        self.Wk = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        self.g_bands = Tensor(np.full(n_bands, -3.0))
        self.g_harmonic = Tensor(-3.0)
        self.Wmix = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        self.norm_scale = Tensor(np.ones(d_model))
        self.params = [self.Wq, self.Wk, self.g_bands, self.g_harmonic,
                       self.Wmix, self.norm_scale]

    def fluid(self, X, sem_edges=None):
        """Stages 1-3: scores, edge set, signed sheaf Laplacian.

        Semantic edges come from masked per-row top-k on the router
        scores; the temporal chain (i, i+1) always joins with the
        constant chain_beta weight. Restriction maps are tanh of the
        symmetrized score - signed, bounded, differentiable."""
        n = X.shape[0]
        Xr = rmsnorm(X, self.norm_scale)
        Q = Xr.matmul(self.Wq)
        K = Xr.matmul(self.Wk)
        scores = Q.matmul(K.T) * (1.0 / np.sqrt(self.d))
        if sem_edges is None:
            sem_edges = topk_edges(scores.data, self.topk)
        chain = [(i, i + 1) for i in range(n - 1)]
        chain_set = set(chain)
        sem = [(i, j) for (i, j) in sem_edges if (i, j) not in chain_set]
        si = np.array([e[0] for e in sem], dtype=int)
        sj = np.array([e[1] for e in sem], dtype=int)
        a_sem = ((scores[si, sj] + scores[sj, si]) * 0.5).tanh()
        a_chain = Tensor(np.full(len(chain), self.chain_beta))
        alphas = concat([a_sem, a_chain])
        edges = sem + chain
        L = laplacian_tensor(alphas, edges, n)
        return Xr, scores, L, edges

    def __call__(self, X, sem_edges=None):
        """Stages 4-6: freeze, harmonic, read. Returns (out, aux)."""
        n = X.shape[0]
        Xr, scores, L, edges = self.fluid(X, sem_edges)
        lam = max(gershgorin_bound(L.data), 1e-12)
        ts = 2.0 * np.geomspace(self.a_lo, self.a_hi, self.n_bands + 1) / lam
        Lsc = L * (2.0 / lam) - Tensor(np.eye(n))
        Y = [Xr]
        if self.degree > 1:
            Y.append(Lsc.matmul(Xr))
        for k in range(2, self.degree):
            Y.append(Lsc.matmul(Y[-1]) * 2.0 - Y[k - 2])
        cs = [chebyshev_coefficients_exp(t, lam, self.degree) for t in ts]
        z = None
        for j in range(self.n_bands):
            dc = cs[j] - cs[j + 1]
            b = None
            for k in range(self.degree):
                term = Y[k] * float(dc[k])
                b = term if b is None else b + term
            zb = self.g_bands[j].softplus() * b
            z = zb if z is None else z + zb
        cid, C = component_ids(n, edges)
        counts = np.bincount(cid, minlength=C).astype(float)
        sums = index_add(Tensor(np.zeros(C)), cid, Xr)
        h = sums[cid] * (1.0 / counts[cid])
        z = z + self.g_harmonic.softplus() * h
        out = X + z.matmul(self.Wmix).gelu()
        aux = {"edges": edges, "lam": lam, "scores": scores, "ts": ts}
        return out, aux

