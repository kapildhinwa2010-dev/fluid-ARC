"""Losses: cross-entropy, MSE, L2 - all on the Tensor engine."""

import numpy as np

from .autodiff import Tensor


def cross_entropy(logits, targets):
    """Mean negative log-likelihood under the row-wise softmax."""
    x = logits.data
    shifted = x - x.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    logp = shifted - logsumexp
    n = x.shape[0]
    idx = np.asarray(targets, dtype=np.int64)
    picked = logp[np.arange(n), idx]
    out = Tensor(-picked.mean(), (logits,))
    def _bw():
        s = np.exp(logp)
        d = s.copy()
        d[np.arange(n), idx] -= 1.0
        logits._accumulate((out.grad / n) * d)
    out._backward = _bw
    return out


def mse(pred, target):
    """Mean squared error against a constant target."""
    diff = pred - target
    return (diff * diff).mean()


def l2_penalty(params):
    """Differentiable sum of squares over a parameter list."""
    acc = Tensor(0.0)
    for p in params:
        acc = acc + (p * p).sum()
    return acc

