import numpy as np
import pytest

from superfluid.autodiff import Tensor, check_grads, concat
from superfluid.data import batch, char_dataset, synthetic_corpus
from superfluid.losses import cross_entropy, l2_penalty, mse
from superfluid.optim import Adam


def test_sqrt_max_concat_grads():
    a = Tensor(np.array([4.0, 9.0]))
    b = Tensor(np.array([1.0, 16.0]))
    def build():
        return concat([a, b]).sqrt().max()
    ok, worst = check_grads(build, [a, b], rtol=1e-4, atol=1e-8)
    assert ok and worst < 0.1


def test_cross_entropy_value_and_grad():
    rng = np.random.default_rng(30)
    logits = Tensor(rng.normal(size=(5, 4)))
    targets = np.array([0, 1, 2, 3, 1])
    def build():
        return cross_entropy(logits, targets)
    x = logits.data
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    want = -np.mean(np.log(p[np.arange(5), targets]))
    assert float(build().data) == pytest.approx(want, rel=1e-12)
    ok, worst = check_grads(build, [logits], rtol=1e-4, atol=1e-8)
    assert ok and worst < 0.1


def test_cross_entropy_confident_prediction_near_zero():
    logits = Tensor(np.array([[100.0, 0.0, 0.0]]))
    assert float(cross_entropy(logits, np.array([0])).data) < 1e-6


def test_mse_value_and_grad():
    pred = Tensor(np.array([1.0, 2.0, 3.0]))
    ok, worst = check_grads(lambda: mse(pred, np.array([1.5, 1.0, 2.0])), [pred])
    assert ok and worst < 0.1
    assert float(mse(pred, np.array([1.0, 2.0, 3.0])).data) == pytest.approx(0.0)


def test_l2_penalty_reaches_all_params():
    a = Tensor(np.ones(3))
    b = Tensor(np.full(2, 2.0))
    out = l2_penalty([a, b])
    out.backward()
    assert float(out.data) == pytest.approx(11.0)
    assert np.allclose(a.grad, 2.0)
    assert np.allclose(b.grad, 4.0)


def test_adam_minimizes_quadratic():
    x = Tensor(np.array([5.0, -3.0]))
    opt = Adam([x], lr=0.1)
    first = None
    for _ in range(200):
        opt.zero_grad()
        d = x - np.array([2.0, 1.0])
        loss = (d * d).sum()
        loss.backward()
        if first is None:
            first = float(loss.data)
        opt.step()
    assert float(loss.data) < first * 1e-4
    assert np.allclose(x.data, [2.0, 1.0], atol=1e-2)


def test_adam_first_step_is_sign_like():
    x = Tensor(np.array([1.0, -1.0]))
    opt = Adam([x], lr=0.1)
    x.sum().backward()
    opt.step()
    assert np.allclose(x.data, [0.9, -1.1], atol=1e-3)


def test_adam_zero_grad():
    x = Tensor(np.ones(2))
    opt = Adam([x])
    (x * x).sum().backward()
    assert not np.allclose(x.grad, 0.0)
    opt.zero_grad()
    assert np.allclose(x.grad, 0.0)


def test_char_dataset_roundtrip():
    text = "abca cab"
    ds = char_dataset(text, seq_len=3)
    assert ds["ids"].dtype == np.int64
    assert len(ds["vocab"]) == 4
    inv = {i: c for c, i in ds["stoi"].items()}
    assert "".join(inv[i] for i in ds["ids"]) == text


def test_batch_shapes_and_shift():
    ds = char_dataset("abcdefgh" * 10, seq_len=4)
    rng = np.random.default_rng(31)
    x, y = batch(ds, batch_size=6, rng=rng)
    assert x.shape == (6, 4) and y.shape == (6, 4)
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_synthetic_corpus_deterministic():
    a = synthetic_corpus(seed=7)
    b = synthetic_corpus(seed=7)
    c = synthetic_corpus(seed=8)
    assert a == b and a != c
    assert len(a) > 200

