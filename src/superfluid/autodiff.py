"""Reverse-mode autodiff on NumPy - the engine of the framework.

Every operation ACCUMULATES into its inputs (the original build lost
gradients when mean() overwrote the sum backward). Broadcasting and
newaxis are handled by unbroadcast, and backward() seeds the output
gradient with ones. All regression-tested against finite differences.
"""

import numpy as np


def _unbroadcast(grad, shape):
    """Reduce grad down to shape: handles broadcasting and newaxis."""
    if shape == ():
        return grad.sum()
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """A NumPy array with a gradient and a backward closure."""

    def __init__(self, data, _prev=()):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = None
        self._prev = tuple(_prev)

    @property
    def shape(self):
        return self.data.shape

    def _accumulate(self, grad):
        self.grad = self.grad + grad

    def backward(self):
        """Seed the output gradient with ones, then propagate in
        reverse topological order. Gradients accumulate, never overwrite."""
        topo, visited = [], set()
        def build(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)
        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()

    # ---- arithmetic ----------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        def _bw():
            self._accumulate(_unbroadcast(out.grad, self.data.shape))
            other._accumulate(_unbroadcast(out.grad, other.data.shape))
        out._backward = _bw
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, (self,))
        def _bw():
            self._accumulate(-out.grad)
        out._backward = _bw
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        def _bw():
            self._accumulate(_unbroadcast(out.grad * other.data, self.data.shape))
            other._accumulate(_unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _bw
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other))
        def _bw():
            self._accumulate(_unbroadcast(out.grad / other.data, self.data.shape))
            other._accumulate(_unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape))
        out._backward = _bw
        return out

    # ---- reductions ------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(a % self.data.ndim for a in axes):
                    g = np.expand_dims(g, ax)
            self._accumulate(np.broadcast_to(g, self.data.shape))
        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = 1
            for a in axes:
                n *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- matmul and shapes ------------------------------------------------
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self.data, other.data
        if a.ndim == 2 and b.ndim == 2:
            out = Tensor(a @ b, (self, other))
            def _bw():
                self._accumulate(out.grad @ b.T)
                other._accumulate(a.T @ out.grad)
        elif a.ndim == 2 and b.ndim == 1:
            out = Tensor(a @ b, (self, other))
            def _bw():
                self._accumulate(np.outer(out.grad, b))
                other._accumulate(a.T @ out.grad)
        elif a.ndim == 1 and b.ndim == 2:
            out = Tensor(a @ b, (self, other))
            def _bw():
                self._accumulate(out.grad @ b.T)
                other._accumulate(np.outer(a, out.grad))
        else:
            raise NotImplementedError("matmul gradients implemented for 1-D and 2-D only")
        out._backward = _bw
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))
        def _bw():
            self._accumulate(out.grad.reshape(self.data.shape))
        out._backward = _bw
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, (self,))
        def _bw():
            self._accumulate(out.grad.T)
        out._backward = _bw
        return out

    # ---- elementwise nonlinearities ----------------------------------------
    def exp(self):
        out = Tensor(np.exp(self.data), (self,))
        def _bw():
            self._accumulate(out.grad * out.data)
        out._backward = _bw
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,))
        def _bw():
            self._accumulate(out.grad * (1.0 - out.data ** 2))
        out._backward = _bw
        return out

    def gelu(self):
        """GELU, tanh approximation (no scipy erf needed)."""
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = Tensor(0.5 * x * (1.0 + t), (self,))
        def _bw():
            dt = (1.0 - t ** 2) * c * (1.0 + 3.0 * 0.044715 * x ** 2)
            self._accumulate(out.grad * (0.5 * (1.0 + t) + 0.5 * x * dt))
        out._backward = _bw
        return out

    def softplus(self):
        """log(1 + exp(x)), computed stably."""
        x = self.data
        out = Tensor(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0), (self,))
        def _bw():
            self._accumulate(out.grad * (1.0 / (1.0 + np.exp(-x))))
        out._backward = _bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,))
        def _bw():
            self._accumulate(out.grad * s * (1.0 - s))
        out._backward = _bw
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), (self,))
        def _bw():
            self._accumulate(out.grad * np.sign(self.data))
        out._backward = _bw
        return out

    def sqrt(self):
        """sqrt with a guarded derivative at 0."""
        out = Tensor(np.sqrt(np.maximum(self.data, 0.0)), (self,))
        def _bw():
            self._accumulate(out.grad * 0.5 / np.maximum(out.data, 1e-12))
        out._backward = _bw
        return out

    def max(self):
        """Full-array max; ties share the gradient."""
        out = Tensor(self.data.max(), (self,))
        def _bw():
            g = np.zeros_like(self.data)
            g[self.data == out.data] = out.grad
            self._accumulate(g)
        out._backward = _bw
        return out

    def softmax(self):
        """Row-wise softmax over the last axis."""
        x = self.data
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        s = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(s, (self,))
        def _bw():
            g = out.grad
            self._accumulate(s * (g - (g * s).sum(axis=-1, keepdims=True)))
        out._backward = _bw
        return out

    # ---- indexing and scatter ----------------------------------------------
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,))
        def _bw():
            g = np.zeros_like(self.data)
            try:
                np.add.at(g, idx, out.grad)
            except TypeError:
                g[idx] += out.grad
            self._accumulate(g)
        out._backward = _bw
        return out


def index_add(base, idx, src):
    """out = base with out[idx] += src.

    Gradients reach BOTH base and src (regression: the original build
    had a scatter whose backward never reached src, leaving the whole
    fluid path inert)."""
    out_data = base.data.copy()
    np.add.at(out_data, idx, src.data)
    out = Tensor(out_data, (base, src))
    def _bw():
        base._accumulate(out.grad)
        src._accumulate(out.grad[idx])
    out._backward = _bw
    return out


def concat(tensors, axis=0):
    """np.concatenate with gradients routed back to each part."""
    tensors = list(tensors)
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors))
    def _bw():
        off = 0
        for t in tensors:
            sl = [slice(None)] * out.data.ndim
            sl[axis] = slice(off, off + t.data.shape[axis])
            t._accumulate(out.grad[tuple(sl)])
            off += t.data.shape[axis]
    out._backward = _bw
    return out


def numerical_grad(build, tensors, eps=1e-6):
    """Central finite differences for build() -> scalar Tensor.

    build must construct a fresh graph from the tensors each call; the
    tensors are perturbed in place."""
    grads = []
    for x in tensors:
        g = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            old = x.data[i]
            x.data[i] = old + eps
            fp = float(build().data.sum())
            x.data[i] = old - eps
            fm = float(build().data.sum())
            x.data[i] = old
            g[i] = (fp - fm) / (2.0 * eps)
            it.iternext()
        grads.append(g)
    return grads


def check_grads(build, tensors, rtol=1e-5, atol=1e-7):
    """Compare analytic gradients against finite differences.

    worst is the largest fraction of the combined tolerance
    atol + rtol * |grad| that any entry consumes, so worst < 1.0
    exactly when allclose passes; callers assert a safety margin
    (worst well below 1.0) instead of a raw relative error that
    near-zero entries blow up with round-off noise."""
    for t in tensors:
        t.grad = np.zeros_like(t.data)
    out = build()
    out.backward()
    num = numerical_grad(build, tensors)
    worst = 0.0
    for t, g in zip(tensors, num):
        denom = atol + rtol * np.maximum(np.abs(t.grad), np.abs(g))
        rel = np.abs(t.grad - g) / np.maximum(denom, 1e-12)
        worst = max(worst, float(np.max(rel)))
        if not np.allclose(t.grad, g, rtol=rtol, atol=atol):
            return False, worst
    return True, worst