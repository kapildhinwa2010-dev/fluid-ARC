# 🛠️ Engineering Ledger

> The bugs that only running the code could find — and the experiment that changed the architecture.
> Every entry below was caught by a test, not by reading the code.

---

## 1. The bug ledger

Each of these bugs silently turned a core claim into a no-op. Each one had a passing-looking
model behind it — and each was caught by a numerical check rather than by inspection.

### Original build (v1.0)

| # | bug | symptom | fix |
| --- | --- | --- | --- |
| B1 | Laplacian built as A − D instead of D − A | non-PSD operator, overflow in exp(−tL) | correct construction: D_abs − A |
| B2 | Chebyshev recurrence missing the factor 2 | expansion plateaued at 78% error, independent of degree | T_k = 2L̃T_{k−1} − T_{k−2} |
| B3 | index_add never scattered in the forward pass | the entire fluid was inert — the block Laplacian was identically zero | scatter in the forward pass + regression test |
| B4 | index_add backward never reached src | restriction maps received no gradient at all | gather the incoming gradient at the scattered rows |
| B5 | gradients zeroed after backward() | both models frozen at ln(vocab) for 400 steps | zero before the forward pass |
| B6 | exp(+α) instead of exp(−α) in the coefficients | bands multiplied by e^(2α) ≈ 10⁶ → divergence | sign fix + stability regression test |
| B7 | mean() overwrote the sum backward | gradients silently never reached the inputs | chain the backward, never replace it |
| B8 | no gradient seeding in backward() | every gradient exactly zero | seed the output gradient with ones |
| B9 | newaxis unhandled in the backward | scatter with mismatched index arrays | _unbroadcast strips and squeezes newaxis axes |
| B10 | λmax underestimated by power iteration | Chebyshev recurrence outside its stable region → band energies of 10¹⁵ | Gershgorin upper bound (never underestimates) |

### Restoration (this repository)

| # | bug | symptom | fix |
| --- | --- | --- | --- |
| B11 | matmul rejected 1-D operands | tensor-Laplacian gradient test crashed | 1-D × 2-D matmul with outer-product gradients |
| B12 | Chebyshev coefficients built from a power series in (−s)^k / k! | exp(−tL) missed the eigendecomposition by 2e-2 — T_k are not monomials | Bessel expansion: c_k = e^{−s} · 2 · (−1)^k · I_k(s), machine-verified to 4e-16 |
| B13 | spectral gap returned the smallest positive eigenvalue | disconnected graphs reported λ₂ = 2 instead of 0 | second-smallest eigenvalue, zeroed below round-off |
| B14 | harmonic branch scattered scalars instead of mean vectors | block forward crashed on multi-channel fields | per-component mean vectors via index_add |
| B15 | gradient comparator divided by a 1e-8 floor | finite-difference round-off on near-zero entries read as a 1e-2 relative error | tolerance-fraction metric: worst < 1.0 exactly when allclose passes |

Regression tests now exist for every class of failure above — 59 tests re-run on every commit,
plus the 21 mathematical checks in src/superfluid/selftest.py.

---

## 2. The experiment that changed the architecture

The decisive mechanism test: the input is a sequence of scalars in fixed features, the target is
the sequence mean — a global, order-invariant statistic. Only the aggregation mechanism differs
between the arms; nothing else is trainable.

The first version scored **49.3% — pure chance**. That failure was the most valuable result of the
whole build, because it exposed two real gaps in the architecture:

1. **The band bank cannot carry the frozen state.** The bands carry the transient; the harmonic
   component is exactly what they cancel by construction. Fix: the harmonic branch — the exact
   Hodge projection (per-component mean), computed in O(n + m) with union-find.
2. **Open band gates swamp the frozen state.** The transient is orders of magnitude larger than
   the harmonic term. Fix: gates initialized at softplus(−3) ≈ 0.05 — 74.1% → 97.3%.

### Final mechanism results

| aggregator | sign accuracy | RMSE |
| --- | --- | --- |
| **Superfluid (the fluid)** | **97.3%** | 0.015 |
| Attention (causal) | 100.0% | 0.000 |
| Local, R = 4 | 65.6% | 0.113 |
| Local, R = 8 | 71.0% | 0.102 |

The claim is no longer global geometry in one pass — it is a measurement: the fluid beats both
strictly-local aggregators by 26+ points and near-matches attention.

---

## 3. The honest LM A/B

Character-level language modelling on real text (Project Gutenberg, Alice in Wonderland),
identical tokens, optimizer and schedule:

| model | params | final val (nats/char) | best val | wall clock |
| --- | --- | --- | --- | --- |
| Superfluid | 71,408 | 2.7397 | 2.6181 | 340 s |
| Transformer | 116,544 | 2.6331 | 2.5401 | 95 s |

Read honestly: at this scale the transformer is ~0.09 nats better despite the Superfluid having
39% fewer parameters, and the wall-clock gap (3.6×) is an implementation artefact — dense
O(n²) Laplacians and a Python loop, not an architectural property. 71k parameters and 400 steps
are not a place where a topological prior can show its value. The specification says so; nothing
is hidden. The mechanism test above is the more informative result.

---

## 4. Why this document exists

Every numeric claim in the README and FRAMEWORK.md traces back to a check in
src/superfluid/selftest.py, re-run by CI on every commit. This ledger records what it cost to
get there — because a from-scratch implementation of a new architecture is mostly the story of
the bugs, and the bugs are the part that never makes it into the paper.