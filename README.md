# Superfluid Framework v1.0

Replace the transformer's quadratic attention with a **cellular-sheaf diffusion**: the data's geometry is recomputed and *jammed* in every forward pass, a multi-scale band bank reads the transient, and a sparse route head restores sharp retrieval.

```
X(t) = exp(-t L_F) X(0),        L_F = delta-dagger delta      # sheaf Laplacian
```

The field converges exponentially to ker(L_F) - the Hodge limit, the frozen geometry. A smooth diffusion cannot select one neighbour sharply, so retrieval is re-introduced explicitly as a sparse top-k route head. **Global geometry from the fluid, sharp selection from the head - that is the architecture.**

## The block, in one breath

1. **Router** - top-k content routing (the needle path)
2. **Topology** - semantic edges + temporal chain (order as 1-D topology)
3. **Sheaf** - signed, content-adaptive restriction maps (tanh-bounded)
4. **Freeze** - band filter bank in spectral time (Chebyshev, shared basis)
5. **Harmonic** - exact Hodge projection, the frozen state (O(n+m), union-find)
6. **Read** - mix, gelu, residual (the single nonlinearity)
7. **Route head** - hard top-k gather (winner-take-all retrieval)

## What is measured

- **Mechanism test (the decisive one):** the fluid computes a global statistic in one pass with **97.3% sign accuracy**, vs 65.6% / 71.0% for strictly-local aggregators and 100.0% for attention. The first version scored 49.3% (chance) and exposed two real gaps - the missing harmonic branch and open band gates - both fixed and both now regression-tested.
- **LM A/B (honest):** char-level LM on Alice in Wonderland - Superfluid 71.4k params reaches 2.740 val nats/char; a param-heavier transformer reaches 2.633. **At this tiny scale the transformer is better, and the spec says so.**
- **Cost model:** the fluid itself is 384 FLOPs/token/layer - 0.00016% of a transformer layer. The real win is decode state: ~16 KB instead of a KV cache that grows to 488 GB at 1M-token context.
- **21/21 mathematical checks passed** in the original build, including end-to-end gradient checks against finite differences (max relative error ~3e-7).

## What is here

| path | what |
| --- | --- |
| [FRAMEWORK.md](FRAMEWORK.md) | the specification of record - module map, five machine-verified lemmas, improvement ledger, loss design, instrumentation, ablation plan A1-A12, scaling-law protocol, kill criteria |
| [index.html](index.html) | the generated report - specification and measurements, self-contained |
| `src/superfluid/` | the reference implementation, being restored module by module |
| `recovered_transcript.py` | the original v1.0 build log - the engineering ledger lives here |

## Restoration status

- [x] `autodiff.py` - reverse-mode engine; every op accumulates, broadcasting and newaxis handled, gradient-checked against finite differences
- [x] `sheaf.py` - signed sheaf Laplacian (PSD by construction), G-balance residual, TV energy; differentiable scatter path gradient-checked
- [x] `diffusion.py` - Gershgorin lambda-max, Chebyshev exp(-tL) (eigen-verified), shared-basis band bank, spectral gap, jam-time bound
- [ ] `topology.py` - top-k routing, Kruskal 0-D barcode, Betti numbers, persistence descriptor
- [ ] `blocks.py`, `model.py`, `losses.py`, `optim.py`, `data.py`
- [ ] `selftest.py` - the 21 numerical checks

Every restored module lands with its own regression tests, and CI publishes the test log after every run.

## License

MIT