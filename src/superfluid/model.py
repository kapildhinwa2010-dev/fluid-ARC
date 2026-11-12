"""SuperfluidLM and a TinyTransformer baseline - honest A/B siblings.

Both are non-causal char-level models over one flattened stream;
the spec is explicit that at tiny scale the transformer may win,
and the A/B test keeps that honest."""

import numpy as np

from .autodiff import Tensor
from .blocks import SuperfluidBlock, rmsnorm


class SuperfluidLM:
    """Embedding -> fluid blocks -> (train: per-position head |
    decode: top-k route head)."""

    def __init__(self, vocab, d_model=32, n_layers=2, topk=4, n_bands=6,
                 degree=24, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab = int(vocab)
        self.route_k = int(topk)
        self.emb = Tensor(rng.normal(0.0, 0.02, (vocab, d_model)))
        self.blocks = [SuperfluidBlock(d_model, topk=topk, n_bands=n_bands,
                                       degree=degree, seed=seed + 1 + i)
                       for i in range(n_layers)]
        self.norm_out = Tensor(np.ones(d_model))
        self.W_out = Tensor(rng.normal(0.0, 0.02, (d_model, vocab)))
        self.params = [self.emb, self.norm_out, self.W_out]
        for b in self.blocks:
            self.params = self.params + b.params

    def _encode(self, tokens):
        X = self.emb[tokens]
        for blk in self.blocks:
            X, _ = blk(X)
        return rmsnorm(X, self.norm_out)

    def logits(self, tokens):
        """Per-position vocabulary logits - the training path."""
        return self._encode(tokens).matmul(self.W_out)

    def route(self, tokens):
        """Decode-time route head: softmax over the top-k positions
        gathered by the last-row scores, mixing their vectors."""
        Xr = self._encode(tokens)
        S = Xr.matmul(Xr.T) * (1.0 / np.sqrt(Xr.shape[1]))
        last = S[-1]
        k = min(self.route_k, Xr.shape[0])
        idx = np.argsort(-last.data)[:k]
        w = last[idx].softmax()
        h = (Xr[idx] * w.reshape(-1, 1)).sum(axis=0)
        return h.matmul(self.W_out)


class TinyTransformerBlock:
    """Pre-norm self-attention + MLP - the baseline block."""

    def __init__(self, d_model, n_mlp=None, seed=0):
        rng = np.random.default_rng(seed)
        h = int(n_mlp or 4 * d_model)
        self.Wq = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        self.Wk = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        self.Wv = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        self.Wo = Tensor(rng.normal(0.0, 0.02, (d_model, d_model)))
        self.W1 = Tensor(rng.normal(0.0, 0.02, (d_model, h)))
        self.W2 = Tensor(rng.normal(0.0, 0.02, (h, d_model)))
        self.n1 = Tensor(np.ones(d_model))
        self.n2 = Tensor(np.ones(d_model))
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo, self.W1,
                       self.W2, self.n1, self.n2]

    def __call__(self, X):
        Xr = rmsnorm(X, self.n1)
        Q = Xr.matmul(self.Wq)
        K = Xr.matmul(self.Wk)
        V = Xr.matmul(self.Wv)
        A = Q.matmul(K.T) * (1.0 / np.sqrt(X.shape[1]))
        X = X + A.softmax().matmul(V).matmul(self.Wo)
        Xr = rmsnorm(X, self.n2)
        H = Xr.matmul(self.W1).gelu()
        return X + H.matmul(self.W2)


class TinyTransformer:
    def __init__(self, vocab, d_model=32, n_layers=2, seed=0):
        rng = np.random.default_rng(seed)
        self.emb = Tensor(rng.normal(0.0, 0.02, (vocab, d_model)))
        self.blocks = [TinyTransformerBlock(d_model, seed=seed + 1 + i)
                       for i in range(n_layers)]
        self.norm_out = Tensor(np.ones(d_model))
        self.W_out = Tensor(rng.normal(0.0, 0.02, (d_model, vocab)))
        self.params = [self.emb, self.norm_out, self.W_out]
        for b in self.blocks:
            self.params = self.params + b.params

    def logits(self, tokens):
        X = self.emb[tokens]
        for blk in self.blocks:
            X = blk(X)
        return rmsnorm(X, self.norm_out).matmul(self.W_out)


def train_steps(model, text, steps=30, lr=3e-3, batch_size=4,
                seq_len=16, seed=0):
    """A short Adam run on the char corpus; returns the loss curve.

    Batches are flattened into one stream, so the chain links across
    rows - fine for a smoke test, documented here so nobody is fooled."""
    from .data import batch, char_dataset
    from .losses import cross_entropy
    from .optim import Adam
    ds = char_dataset(text, seq_len=seq_len)
    opt = Adam(model.params, lr=lr)
    rng = np.random.default_rng(seed)
    curve = []
    for _ in range(steps):
        x, y = batch(ds, batch_size, rng)
        lg = model.logits(x.reshape(-1))
        loss = cross_entropy(lg, y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(loss.data))
    return curve

