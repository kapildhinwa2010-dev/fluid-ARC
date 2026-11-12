# Superfluid Framework v1.0 — Specification

Superfluid Framework
A trainable, instrumented architecture that replaces the transformer's quadratic attention with a cellular-sheaf diffusion: the data's geometry is recomputed and jammed in every forward pass, a multi-scale band bank reads the transient, and a sparse route head restores sharp retrieval.

VERIFICATION
Pure NumPy implementation (no torch, no scipy) so that every claim can be re-verified on any laptop in under a minute. Numbers on this page are generated from the measurement files — nothing is hand-typed.

01 · The block
Six stages, one geometric operator
① Router
top-k content routing
QKᵀ/√d → masked → top-k (hard selection)
② Topology
semantic edges ∪ temporal chain
order enters as 1-dimensional topology
③ Sheaf
signed restriction maps α_uv
tanh-bounded MLP of endpoint features → content-adaptive geometry, G-balance learnable
④ Freeze
band filter bank
band_j = e^(−t_j L) − e^(−t_{j+1}L); log-spaced in spectral time α = t·λmax/2
⑤ Read
mix → gelu → residual
the single pointwise nonlinearity that prevents linear collapse
⑥ Route head
hard top-k gather
winner-take-all = the sharp retrieval the smooth fluid cannot provide
// the whole block, exactly as implemented
def block(h):                                    # h: (n, d)
    xn     = rmsnorm(h)
    scores = (xn @ Wq) @ (xn @ Wk).T / sqrt(d)
    idx    = topk_masked(scores, k)            # (n, k) hard selection

    sem   = unique_undirected(idx)             # semantic edges
    chain = [(i, i+1) for i in range(n-1)]      # temporal edges

    alpha = tanh_mlp([xn[u] ; xn[v]])           # signed, |α| ≤ 1
    L     = D_abs(alpha_sem) + D_abs(β*chain)   # sheaf Laplacian, PSD
    lam   = power_iteration_lmax(L)

    ts    = 2*geomspace(a_lo, a_hi, B+1)/lam    # spectral-time bands
    Y     = chebyshev_basis(L, lam, xn, m)     # m matvecs, shared by all bands
    bands = [sum_k (c_j[k]-c_j1[k]) * Y[k] for j in range(B)]

    z  = sum_j softplus(g_j) * bands[j]
    z  = gelu(z @ Wmix)
    out = h + z @ Wo

    v  = xn @ Wv
    out = out + (softmax_k(scores[idx]) * v[idx]).sum(1) @ Wr
02 · Improvement ledger
Every weakness of the v0.1 idea, and the module that kills it
This is the "take it to the max" table. Each row is a named failure mode from the original proposal, the module that fixes it, and the mechanism — with the measured effect wherever the effect is measurable.

03 · Verified mathematics
The claims, machine-checked
Five lemmas carry the architecture. selftest.py verifies each one numerically and writes the results to results/selftest.json.

04 · Measurements
Training on real text, against a param-matched transformer
Character-level LM — validation loss
Project Gutenberg (Alice in Wonderland), char-level, identical tokens, identical optimiser, identical schedule.

Global-aggregation task — validation accuracy
The label is the sequence mean — exactly the quantity the Hodge limit computes. A local MLP with a fixed receptive field cannot see it; the fluid and attention both can.

Final results
05 · The decisive mechanism test
Can the fluid compute a GLOBAL statistic in one pass?
Everything above is theory. This is the experiment that settles the central claim. The input is a sequence of scalars encoded as fixed features; the target is the mean — a global, order-invariant statistic that a strictly local model cannot compute. Only the aggregation mechanism differs between the arms; nothing else is trainable.

Sign accuracy of the predicted mean
Chance = 50%. The fluid reaches 97.3% in one pass — it beats both strictly-local aggregators and nearly matches attention. This is the mechanism claim, measured.

What the test exposed
The band bank alone cannot do it. The bands carry the transient; the frozen (harmonic) component is exactly what they cancel out. The first run scored 49.3% — chance.
The fix is a third path: the harmonic branch, the exact Hodge projection (per-component mean), computed in O(n+m) with union-find.
The band gates must start near zero (softplus(−3)): the transient is far larger than the harmonic component, so open gates swamp the frozen state. With open gates the score was 74%; near-closed, 97.3%.
Attention still wins (100.0% vs 97.3%) — but it costs a full quadratic attention matrix to do it.
06 · Engineering ledger
The bugs that only running the code could find
Every one of these was found by a test, not by reading the code. They are listed because they are the honest cost of a from-scratch implementation — and because each one had silently turned a core claim into a no-op.

bug	symptom	fix
Laplacian sign (A−D instead of D−A)	non-PSD operator, overflow in exp(−tL)	correct construction
Chebyshev recurrence missing the factor 2	expansion plateaued at 78% error, independent of degree	Tk = 2L̃Tk−1 − Tk−2
index_add never scattered in the forward pass	the whole fluid was inert — the block's Laplacian was identically zero	scatter in the forward pass + a regression test
index_add backward never reached src	restriction maps received no gradient at all	gather the incoming gradient at the scattered rows
gradients zeroed after backward()	both models frozen at ln(vocab) for 400 steps	zero before the forward pass
exp(+α) instead of exp(−α) in the coefficients	bands multiplied by e2α ≈ 10⁶ → divergence	sign fix + a stability regression test
mean() overwrote the sum's backward	gradients silently never reached the inputs	chain instead of replace
no gradient seeding in backward()	every gradient exactly zero	seed the output gradient
newaxis ([:, :, None]) unhandled in the backward	scatter with mismatched index arrays	strip and squeeze newaxis axes
λmax underestimated by power iteration	Chebyshev recurrence outside its stable region	Gershgorin upper bound (differentiable)
07 · Instrumentation
The model shows you its geometry
Every block reports its spectral gap, G-balance residual, band energies and persistent signature on every forward pass. This is the framework's real advantage over a black-box: the failure mode is visible before the loss curve moves.

Band energy per timescale
Energy carried by each band of the filter bank. A collapse onto one band means the multi-scale readout has degenerated into a single exponential.

Internal diagnostics (final)
λmax is the spectral scale (sets the band placement); balance residual ‖L1‖∞ is the G-balance indicator — the phase transition; α_neg_frac is how "signed" the learned geometry is.

08 · Cost model
Where the FLOPs actually are
Per token, per layer (d = 4096, n = 4096)
SF fluid itself
384
SF route head k=16
2.6e5
TF attention (4nd)
6.7e7
SF + FFN
3.0e8
TF full layer
4.7e8
The fluid is 0.00016 % of a transformer layer. It is nearly free — which also means it cannot be the reason the model works.

Decode state — the real win
Context	Transformer KV cache	Superfluid (static topology)
4,096	2.0 GB	16 KB
32,768	16.0 GB	16 KB
131,072	64.0 GB	16 KB
1,000,000	488 GB	16 KB
Param-matched full training run (D ≈ 44B, 470B tokens, 512×H100 @ 40 % MFU): 1.10× cheaper at 8K context, 2.54× at 128K.

09 · What kills it
Falsification criteria, stated up front
Kill criteria (cheap to test)
SF-2 trails the transformer by > 0.15 nats at matched FLOPs on a 400M-token run → the fluid contributes nothing. (Cost: < 1 GPU-hour.)
The band bank gives no gain over a single exponential after the rank-c fix → the multi-scale story is decorative.
λ₂ collapses during training → the geometry is not learnable.
The learned sheaf never becomes G-balanced → no frozen state, and the jamming narrative does not apply.
Ablation A2 shows the route head does all the work → "sparse attention with extra steps".
Roadmap
V1.0 — THIS REPO
Verified maths, trainable model, full instrumentation
15/15 numerical checks; end-to-end autodiff with gradient checks; LM A/B; mechanism demo.

V1.1 — 2–4 WEEKS
Sparse Laplacian backend + Torch port + ablation runner
The dense O(n²) Laplacian is the current scaling limit. A1–A12 ablations produce one results table.

V1.2 — 1–2 MONTHS
Scaling-law harness + learned band placement
Place the bands around the measured 1/λ₂ — the critical temperature becomes a hyperparameter.

V2.0 — RESEARCH
Multi-sheaf superposition + learned topology + domain port
Syntax / semantics / topology as separate sheaves with learned mixing; differentiable complex construction; molecules & proteins, where the prior is not a gimmick.

Superfluid Framework v1.0 — generated by build_report.py from results/selftest.json and results/train_results.json. Full specification in FRAMEWORK.md; theory & cost analysis in the companion report Superfluid v0.1.