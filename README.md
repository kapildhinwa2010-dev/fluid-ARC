# Superfluid Framework v1.0

Replace the transformer's quadratic attention with a **cellular-sheaf diffusion**: the data's geometry is recomputed and *jammed* in every forward pass, a multi-scale band bank reads the transient, and a sparse route head restores sharp retrieval.

```
X(t) = exp(-t L_F) X(0),        L_F = delta-dagger delta      # sheaf Laplacian
```

The field converges exponentially to ker(L_F) — the Hodge limit, the frozen geometry. A smooth diffusion cannot select one neighbour sharply, so retrieval is re-introduced explicitly as a sparse top-k route head. **Global geometry from the fluid, sharp selection from the head — that is the architecture.**

## The block, in one breath

1. **Router** — top-k content routing (the needle path)
2. **Topology** — semantic edges + temporal chain (order as 1-D topology)
3. **Sheaf** — signed, content-adaptive restriction maps (tanh-bounded)
4. **Freeze** — band filter bank in spectral time (Chebyshev, shared basis)
5. **Harmonic** — exact Hodge projection, the frozen state (O(n+m), union-find)
6. **Read** — mix, gelu, residual (the single nonlinearity)
7. **Route head** — hard top-k gather (winner-take-all retrieval)

## What is measured

- **Mechanism test (the decisive one):** the fluid computes a global statistic in one pass with **97.3% sign accuracy**, vs 65.6% / 71.0% for strictly-local aggregators and 100.0% for attention. The first version scored 49.3% (chance) and exposed two real gaps — the missing harmonic branch and open band gates — both fixed and both now regression-tested.
- **LM A/B (honest):** char-level LM on Project Gutenberg — Superfluid 71.4k params reaches 2.740 val nats/char; a param-heavier transformer reaches 2.633. **At this tiny scale the transformer is better, and the spec says so.**
- **Cost model:** the fluid itself is 384 FLOPs/token/layer — 0.00016% of a transformer layer. The real win is decode state: ~16 KB instead of a KV cache that grows to 488 GB at 1M-token context.
- **21/21 mathematical checks pass**, including end-to-end gradient checks against finite differences (max relative error ~3e-7).

## What is here

| path | what |
| --- | --- |
| [FRAMEWORK.md](FRAMEWORK.md) | the specification of record — module map, five machine-verified lemmas, improvement ledger, loss design, instrumentation, ablation plan A1-A12, scaling-law protocol, kill criteria |
| [index.html](index.html) | the generated report — specification and measurements, self-contained |

## Honest status

- The specification and the measured results come from the original v1.0 build.
- The **reference implementation is being restored** into this repository (the `superfluid/` NumPy package, `selftest.py`, `train.py`) — see the roadmap in FRAMEWORK.md.
- Known limits, stated up front: the dense O(n^2) Laplacian is the current scaling limit; nothing here has been trained at scale; the kill criteria in FRAMEWORK.md section 12 are the point — each is cheap to test.

## License

MIT