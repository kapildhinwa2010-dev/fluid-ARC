import numpy as np
import pytest

from superfluid.autodiff import Tensor, check_grads, index_add, numerical_grad


def _rng():
    return np.random.default_rng(0)


def test_arithmetic_chain():
    rng = _rng()
    a = Tensor(rng.normal(size=(3, 4)))
    b = Tensor(rng.normal(size=(3, 4)))
    ok, worst = check_grads(lambda: ((a * b + a / 2.0 - b) * 1.7).sum(), [a, b])
    assert ok and worst < 0.1


def test_matmul():
    rng = _rng()
    a = Tensor(rng.normal(size=(4, 5)))
    b = Tensor(rng.normal(size=(5, 3)))
    ok, worst = check_grads(lambda: a.matmul(b).sum(), [a, b])
    assert ok and worst < 0.1


def test_nonlinearities():
    rng = _rng()
    x = Tensor(rng.normal(size=(5, 6)) * 0.5)
    def build():
        return (x.tanh() * x.gelu() + x.softplus() - x.sigmoid()).sum()
    ok, worst = check_grads(build, [x])
    assert ok and worst < 0.1


def test_softmax():
    rng = _rng()
    x = Tensor(rng.normal(size=(4, 7)))
    w = Tensor(rng.normal(size=(4, 7)))
    ok, worst = check_grads(lambda: (x.softmax() * w).sum(), [x, w])
    assert ok and worst < 0.1


def test_sum_axis_and_mean_do_not_collide():
    """Regression: mean() used to overwrite the sum() backward."""
    rng = _rng()
    x = Tensor(rng.normal(size=(3, 4)))
    def build():
        return x.sum(axis=0).mean() + x.mean(axis=1).sum()
    ok, worst = check_grads(build, [x])
    assert ok and worst < 0.1


def test_broadcast_and_newaxis():
    """Regression: newaxis indexing broke the scatter backward."""
    rng = _rng()
    a = Tensor(rng.normal(size=(3, 1)))
    b = Tensor(rng.normal(size=(1, 4)))
    ok, worst = check_grads(lambda: (a + b * 2.0).sum(), [a, b])
    assert ok and worst < 0.1


def test_index_add_reaches_both_inputs():
    """Regression: the scatter backward never reached src."""
    rng = _rng()
    base = Tensor(rng.normal(size=(5, 3)))
    src = Tensor(rng.normal(size=(4, 3)))
    idx = np.array([0, 2, 2, 4])
    ok, worst = check_grads(lambda: index_add(base, idx, src).sum(), [base, src])
    assert ok and worst < 0.1


def test_index_add_duplicate_indices_accumulate():
    rng = _rng()
    base = Tensor(np.zeros((4, 2)))
    src = Tensor(np.ones((3, 2)))
    idx = np.array([1, 1, 1])
    out = index_add(base, idx, src)
    assert np.allclose(out.data[1], 3.0)


def test_getitem_with_duplicates():
    rng = _rng()
    x = Tensor(rng.normal(size=(6, 3)))
    idx = np.array([2, 2, 5])
    ok, worst = check_grads(lambda: (x[idx] * 2.0).sum(), [x])
    assert ok and worst < 0.1


def test_full_composite_against_finite_differences():
    """End-to-end style check on a small block-like computation."""
    rng = _rng()
    x = Tensor(rng.normal(size=(4, 6)) * 0.3)
    w1 = Tensor(rng.normal(size=(6, 6)) * 0.3)
    w2 = Tensor(rng.normal(size=(6, 2)) * 0.3)
    def build():
        h = x.matmul(w1).gelu()
        return (h.matmul(w2).tanh()).sum()
    ok, worst = check_grads(build, [x, w1, w2])
    assert ok and worst < 0.1


def test_numerical_grad_helper_matches_closed_form():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    (g,) = numerical_grad(lambda: (x * x).sum(), [x])
    assert np.allclose(g, 2.0 * x.data)