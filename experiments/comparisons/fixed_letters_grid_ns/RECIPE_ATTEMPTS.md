# Fixgrid Dale recipe attempts

Goal: **48/48** ixed_letters_grid_ns runs at **<=3% word error**, **seed 1 only**, one coherent recipe.

Checkpoint tree: checkpoints/r*/rnn_dale/. Kill broken cluster arrays before each new full rerun.

## v1 - baseline (cluster, done)

- Job 55234894: **30/48 at 3%**, median 0.027, worst r42 @ 0.953
- Recipe: H=200, LR=0.025, dropout=0, init=0.005, e_fraction=0.8

## v2 - init 0.015 + dropout 0.2 (killed)

- r41 8k: WE 96.9% — not submitted

## v2b - init 0.025, no dropout (partial)

- r41 40k: 29.3%; r42 40k dead; r13 regressed

## v3 - H=400 for L>=6 (killed)

- r42 dead; r13 regressed

## v4 - tiered init by length (done, failed)

- Job 55275982: **28/48 at 3%**, median 0.028, worst 0.961
- By L hits: L3 8/12, L4 7/12, L5 9/12, L6 4/12
- Dead: r41, r42, r47, r25

## v5 - architecture fix: input reaches inhibitory cells (smoke passed)

**v1-v4 were all crippled by an architecture bug, not by hyperparameters.**
`enforce_dale_input_exc_only` hard-zeroed every `W_xh` row onto an inhibitory
unit, so the 40 I cells received no sensory drive and could only be driven
recurrently. That is not Dale's law: Dale's law constrains the sign of a
neuron's *outgoing* synapses, not which afferents it receives. Feedforward
inhibition (thalamic drive onto interneurons) is a canonical cortical motif.

Fix: `W_xh >= 0` on **all** rows, E and I alike. `W_hh` / `W_ho` columns are
still signed by source type, unchanged. The flag was removed - there is no
E-only mode to fall back to.

- Recipe: H=200, init=0.01, LR=0.025, dropout=0, e_fraction=0.8, seed 1
- Verified: 40/40 I units receive input, `min(W_xh) >= 0`, Dale violation 0.0

Local smoke (20k-step cap; production runs get 80k-150k):

| cell | v1-v4 | v5 @ 20k | note |
|---|---|---|---|
| r13 (L3, easy) | 3.15% @ 11650 | **1.53% @ 6200** | target hit ~2x faster |
| r42 (L6, hardest) | dead, CE 1.349 (chance) | 18.5%, CE 0.464 | still falling at cap |
| r41 (L6) | dead, 96.9% | 40.0%, CE 0.473 | still falling at cap |

r41/r42 had never left chance under any prior recipe. Both now descend
steadily; neither had converged when the smoke cap truncated them.

Verdict: promote to full 48-cell rerun. All prior Dale checkpoints are invalid
and must be discarded (this changes the model, not just the optimizer).

## v5 cluster submit

- Job **55728939** array 0-47 submitted 2026-08-16 ~12:45 IDT
- Old Dale ckpts wiped under checkpoints/r*/rnn_dale/
- Code on cluster: W_xh >= 0 onto E+I (no exc_only)
- Recipe: H=200, init=0.01, LR=0.025, dropout=0, e_fraction=0.8, seed 1
