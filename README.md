Superfluid Framework v1.0
Replace the transformer's quadratic attention with a cellular-sheaf diffusion whose geometry is recomputed and jammed in every forward pass.

python


X(t) = exp(-t L_F) X(0),        L_F = δᵀδ      # sheaf Laplacian
Jams — the field converges exponentially to ker(L_F) (the Hodge limit).python3 selftest.py      # 15 numerical checks of the mathematics
python3 train.py --steps 400 --d 64 --layers 2 --ctx 64 --batch 16
python3 build_report.py  # -> index.htmlbut a smooth diffusion cannot select one neighbour sharply, so retrieval is re-introduced explicitly as a sparse top-k route head. Global geometry from the fluid, sharp selection from the head — that is the architecture.

Verify it in 30 secondsRequirements: numpy only. No torch, no scipy — the whole point is that any claim can be re-verified on any laptop.

What is here
path	what
superfluid/autodiff.py	reverse-mode autodiff (Tensor + ops), gradient-checked
superfluid/sheaf.py	signed sheaf Laplacian, G-balance, total variation
superfluid/diffusion.py	λmax power iteration, Chebyshev exp(-tL), band filter bank, spectral gap, jam time, spectral-time band placement
superfluid/topology.py	top-k routing, edge construction, Kruskal 0-D barcode, Betti numbers, persistence descriptor
superfluid/blocks.py	SuperfluidBlock — the six-stage stack
superfluid/model.py	SuperfluidLM + TinyTransformer baseline
superfluid/losses.py, optim.py, data.py	objectives, Adam, Gutenberg corpus
selftest.py	the mathematics, verified → results/selftest.json
train.py	LM A/B + global-aggregation demo → results/train_results.json
build_report.py	generates index.html from the measurements
FRAMEWORK.md	the specification of record — module map, lemmas, improvement ledger, loss design, instrumentation, ablation plan, scaling-law protocol, kill criteria
Status
Mathematics: 15/15 checks pass (jamming bound, jam-time bound, Chebyshev degree, linear collapse, the nonlinearity fix, rank-c trap, band recovery, G-balance phase transition, contraction certificate, frame property, and end-to-end gradient checks against finite differences).Training: character-level LM on Project Gutenberg against a param-matched transformer (Superfluid 71.4 k params → 2.740 val; Transformer 116.5 k → 2.633 — the transformer is better at this tiny scale, and the spec says so).
Mechanism test (the decisive one): the fluid computes a global statistic in one pass with 97.3 % sign accuracy, versus 65.6 % / 71.0 % for strictly-local aggregators and 100.0 % for attention.
The first version of that tesThe first version of that test scored 49.3 % (chance) and exposed two real architectural gaps: the band bank cannot carry the frozen state (fixed by the harmonic branch), and open band gates swamp it (fixed by the gate initialisation).
Not yet done: sparse (non-dense) Laplacian backend, Torch port, the A1–A12 ablation sweep, and any scaling-law run. These are v1.1/v1.2 inThe first version of that test scored 49.3 % (chance) and exposed two real architectural gaps: the band bank cannot carry the frozen state (fixed by the harmonic branch), and open band gates swamp it (fixed by the gate initialisation).
Not yet done: sparse (non-dense) Laplacian backend, Torch port, the A1–A12 ablation sweep, and any scaling-law run. These are v1.1/v1.2 in FRAMEWORK.md.
Honest caveatsHonest caveats
The fluid is 0.00016 % of a transformer layer's FLOPs. It is nearly free, which also means it cannot be the reason the model works — capacity still lives in the FFN and the mixing matrices.
The frozen state has rank = #components. Reading only the jam loses almost everything; the band filter bank is mandatory, not optional.The frozen state has rank = #components. Reading only the jam loses almost everything; the band filter bank is mandatory, not optional.
A purely linear Superfluid cannot fit XOR (measured: 0.50 accuracy). The single pointwise nonlinearity is what makes the architecture legal.
Dense O(n²) Laplacian is the current scaling limit.
Nothing here has been trained at scale. The kill criteria in FRAMEWORK.md §12 are the point — each is cheap to test.# superfluid-AI-