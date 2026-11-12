<div align="center">

# 🌊 Superfluid

### Cellular-sheaf diffusion as an attention replacement

Replace the transformer quadratic attention with a geometric operator stack: the data geometry
is recomputed and **jammed** in every forward pass, a multi-scale band bank reads the transient,
and a sparse route head restores sharp retrieval.

[![CI](https://github.com/kapildhinwa2010-dev/superfluid-AI-/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/superfluid-AI-/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/superfluid-AI-/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/superfluid-AI-/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2f6f5e.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy%20only-013243?logo=numpy&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-59%20passing-2f6f5e)](ci_logs/pytest.log)
[![Selftest](https://img.shields.io/badge/selftest-21%2F21-2f6f5e)](src/superfluid/selftest.py)

</div>

---

## ✨ Why this exists

A transformer layer spends its budget on dense weight matrices and **quadratic attention** — the
part that makes long context expensive and retrieval possible. Superfluid replaces the attention
with a geometric operator:

```
X(t) = exp(-t L_F) X(0),        L_F = D_abs(α) − A      # signed sheaf Laplacian, PSD by construction
```

The field converges exponentially to ker(L_F) — the Hodge limit, the frozen geometry. A smooth
diffusion cannot select one neighbour sharply, so retrieval is re-introduced explicitly as a
sparse top-k route head. **Global geometry from the fluid, sharp selection from the head — that
is the architecture.**

## 🧊 The block — six stages, one operator

| Stage | What happens | Why |
| --- | --- | --- |
| ① 🧭 **Router** | top-k content routing → semantic edges | the needle path |
| ② 🕸️ **Topology** | semantic edges ∪ temporal chain | order becomes 1-dimensional topology |
| ③ 🔗 **Sheaf** | signed, tanh-bounded restriction maps α | content-adaptive geometry, PSD by construction |
| ④ ❄️ **Freeze** | band filter bank in spectral time (Chebyshev, shared basis) | multi-scale reading of the transient |
| ⑤ 🧊 **Harmonic** | exact Hodge projection = per-component mean, O(n+m) | the frozen state — without it the block cannot aggregate globally (measured: 49.3% = chance) |
| ⑥ 📖 **Read** | mix → gelu → residual | the single nonlinearity |
| ⑦ 🎯 **Route head** | hard top-k gather, winner-take-all | sharp retrieval |

## 📐 The mathematics, machine-checked

Five lemmas carry the architecture, and every one is verified numerically on every commit:

- ✅ **Jamming bound** — ‖e^{−tL}u‖ ≤ e^{−λ₂t}‖u‖ for u ⊥ ker(L)
- ✅ **Jam-time bound** — t_jam = ln(1/ε)/λ₂ upper-bounds the freezing time
- ✅ **Chebyshev degree** — m = O(√κ · ln(1/ε)) suffices; the recurrence carries the factor 2
- ✅ **Linear collapse** — a purely linear fluid cannot fit XOR; one pointwise gelu can
- ✅ **Rank-c trap** — the frozen state has rank = #components; the band bank restores the signal
- ✅ **G-balance phase transition**, **contraction certificate**, **frame property**
- ✅ **End-to-end gradients** — checked against finite differences, including the full block

Run them yourself: ```python -m superfluid.selftest``` → 21/21 pass in under a second.

## 📊 Measured results

### 🎯 The mechanism test (the decisive one)

Input: scalars. Target: the sequence mean — a global statistic a strictly local model cannot compute.

| aggregator | sign accuracy | RMSE |
| --- | --- | --- |
| **🌊 Superfluid (the fluid)** | **97.3%** | 0.015 |
| ⚡ Attention (causal) | 100.0% | 0.000 |
| 📍 Local, R = 4 | 65.6% | 0.113 |
| 📍 Local, R = 8 | 71.0% | 0.102 |

The first version scored 49.3% — chance — and that failure redesigned the architecture (see the
[engineering ledger](docs/ENGINEERING_LEDGER.md)).

### 📖 LM A/B (honest)

Char-level LM on Alice in Wonderland, identical tokens / optimizer / schedule:

| model | params | final val | best val | wall clock |
| --- | --- | --- | --- | --- |
| 🌊 Superfluid | 71,408 | 2.7397 | 2.6181 | 340 s |
| ⚡ Transformer | 116,544 | 2.6331 | 2.5401 | 95 s |

> **At this tiny scale the transformer is better, and the spec says so.** The wall-clock gap is an
> implementation artefact (dense O(n²) Laplacian + Python loop); the quality gap is exactly what
> the planned FLOP-matched ablations (A9/A10) exist to measure.

### 💰 Cost model — the real win is memory

| context | Transformer KV cache | Superfluid (static topology) |
| --- | --- | --- |
| 4,096 | 2.0 GB | **16 KB** |
| 131,072 | 64.0 GB | **16 KB** |
| 1,000,000 | 488 GB | **16 KB** |

The fluid itself costs 384 FLOPs/token/layer — 0.00016% of a transformer layer.

## 🚀 Quickstart

```bash
pip install -e .              # numpy >= 1.23 is the only dependency
pytest                        # 59 regression tests
python -m superfluid.selftest # 21/21 mathematical checks
```

Train both models on the deterministic bundled corpus:

```python
from superfluid import (SuperfluidLM, TinyTransformer, train_steps,
                        char_dataset, synthetic_corpus)

corpus = synthetic_corpus(seed=0)
ds = char_dataset(corpus, seq_len=16)

fluid = SuperfluidLM(len(ds[vocab]), d_model=32, n_layers=2, seed=0)
curve = train_steps(fluid, corpus, steps=50, lr=3e-3)   # loss curve
```

## 🗂️ Repository map

| path | what |
| --- | --- |
| 📄 [FRAMEWORK.md](FRAMEWORK.md) | the specification of record — lemmas, improvement ledger, cost model, kill criteria, roadmap |
| 🛠️ [docs/ENGINEERING_LEDGER.md](docs/ENGINEERING_LEDGER.md) | every bug the code found, the experiment that changed the architecture, the honest A/B |
| 📊 [index.html](index.html) | the generated report — specification and measurements, self-contained |
| 📦 src/superfluid/ | the package: autodiff, sheaf, diffusion, topology, blocks, model, losses, optim, data, selftest |
| ✅ tests/ | 59 regression tests, re-run by CI on every commit |
| 📋 ci_logs/pytest.log | the test log, published by CI after every run |

## 🗺️ Roadmap

- **v1.1** — sparse Laplacian backend, Torch port, ablation runner (A1–A12)
- **v1.2** — scaling-law harness, learned band placement around the measured 1/λ₂
- **v2.0** — multi-sheaf superposition, differentiable topology, domain ports (molecules, meshes, proteins)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every numerical claim needs a test; every bug gets a row
in the engineering ledger. Security issues: [SECURITY.md](SECURITY.md).

## 📜 License

MIT — see [LICENSE](LICENSE). Cite via [CITATION.cff](CITATION.cff).