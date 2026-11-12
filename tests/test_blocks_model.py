import numpy as np
import pytest

from superfluid.autodiff import Tensor, check_grads
from superfluid.blocks import SuperfluidBlock, rmsnorm
from superfluid.data import char_dataset, synthetic_corpus
from superfluid.diffusion import spectral_gap
from superfluid.losses import cross_entropy
from superfluid.model import (SuperfluidLM, TinyTransformer,
                              TinyTransformerBlock, train_steps)
from superfluid.sheaf import laplacian_from_edges


def test_rmsnorm_value_and_grad():
    x = Tensor(np.array([[3.0, 4.0]]))
    w = Tensor(np.array([1.0, 2.0]))
    out = rmsnorm(x, w)
    want = np.array([[3.0, 4.0]]) / np.sqrt(12.5 + 1e-5) * np.array([1.0, 2.0])
    assert np.allclose(out.data, want)
    ok, worst = check_grads(lambda: rmsnorm(x, w).sum(), [x, w])
    assert ok and worst < 1e-4


def test_block_forward_shape_and_finiteness():
    blk = SuperfluidBlock(8, topk=2, n_bands=3, degree=12, seed=1)
    X = Tensor(np.random.default_rng(2).normal(size=(8, 8)))
    out, aux = blk(X)
    assert out.shape == (8, 8)
    assert np.isfinite(out.data).all()
    assert len(aux["edges"]) >= 7


def test_block_edges_connected_via_chain():
    """The temporal chain alone connects the ring: spectral gap > 0."""
    blk = SuperfluidBlock(6, topk=1, n_bands=2, degree=8, seed=3)
    X = Tensor(np.random.default_rng(4).normal(size=(6, 6)))
    _, aux = blk(X)
    Lap = laplacian_from_edges(6, aux["edges"], np.ones(len(aux["edges"])))
    assert spectral_gap(Lap) > 0.0


def test_block_gradient_check_fixed_edges():
    """With the edge set fixed, every parameter gradient is exact.

    The top-k selection is discrete and sits outside the checked graph -
    exactly where the original build hid a dead gradient."""
    X = Tensor(np.random.default_rng(10).normal(size=(8, 8)))
    edges = [(2, 5), (1, 6), (3, 7)]
    def build(blk):
        out, _ = blk(X, sem_edges=edges)
        return (out * out).sum()
    # adaptive scale: the gate and mixer path is exact
    blk = SuperfluidBlock(8, topk=2, n_bands=3, degree=12, seed=9)
    ok, worst = check_grads(lambda: build(blk),
                            [blk.g_bands, blk.g_harmonic, blk.Wmix],
                            rtol=1e-3, atol=1e-6)
    assert ok and worst < 1e-3
    # fixed scale: every parameter gradient is exact end to end
    blk2 = SuperfluidBlock(8, topk=2, n_bands=3, degree=12, seed=9,
                           fixed_lam=10.0)
    ok2, worst2 = check_grads(lambda: build(blk2), blk2.params,
                              rtol=1e-3, atol=1e-6)
    assert ok2 and worst2 < 1e-3


def test_band_gates_start_near_closed():
    blk = SuperfluidBlock(8, n_bands=4, seed=11)
    opened = 1.0 / (1.0 + np.exp(3.0))
    assert np.allclose(1.0 / (1.0 + np.exp(-blk.g_bands.data)), opened, rtol=1e-9)


def test_superfluid_lm_forward_shapes():
    corpus = synthetic_corpus(seed=1)
    ds = char_dataset(corpus, seq_len=16)
    model = SuperfluidLM(len(ds["vocab"]), d_model=16, n_layers=2, seed=3)
    tokens = ds["ids"][:16]
    lg = model.logits(tokens)
    assert lg.shape == (16, len(ds["vocab"]))
    assert np.isfinite(lg.data).all()
    r = model.route(tokens)
    assert r.shape == (len(ds["vocab"]),)


def test_superfluid_lm_grads_reach_fluid_path():
    """Regression: the fluid path once trained inert - gates and Wq
    must both receive gradient."""
    corpus = synthetic_corpus(seed=1)
    ds = char_dataset(corpus, seq_len=16)
    model = SuperfluidLM(len(ds["vocab"]), d_model=16, n_layers=1, seed=4)
    tokens = ds["ids"][:16]
    loss = cross_entropy(model.logits(tokens), ds["ids"][1:17])
    loss.backward()
    for p in model.params:
        assert np.isfinite(p.grad).all()
    blk = model.blocks[0]
    assert np.abs(blk.g_bands.grad).max() > 0.0
    assert np.abs(blk.Wq.grad).max() > 0.0


def test_superfluid_lm_learns():
    corpus = synthetic_corpus(seed=1)
    ds = char_dataset(corpus, seq_len=16)
    model = SuperfluidLM(len(ds["vocab"]), d_model=16, n_layers=1, seed=5)
    curve = train_steps(model, corpus, steps=25, lr=3e-3, seed=5)
    assert curve[-1] < curve[0]


def test_transformer_baseline_learns():
    corpus = synthetic_corpus(seed=1)
    ds = char_dataset(corpus, seq_len=16)
    model = TinyTransformer(len(ds["vocab"]), d_model=16, n_layers=1, seed=6)
    curve = train_steps(model, corpus, steps=25, lr=3e-3, seed=6)
    assert curve[-1] < curve[0]


def test_transformer_block_gradient_check():
    blk = TinyTransformerBlock(8, n_mlp=16, seed=7)
    X = Tensor(np.random.default_rng(8).normal(size=(6, 8)))
    def build():
        return (blk(X) * blk(X)).sum()
    ok, worst = check_grads(build, blk.params, rtol=1e-3, atol=1e-6)
    assert ok and worst < 1e-3

