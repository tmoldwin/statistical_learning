# Plan: decoding over the course of learning

Follow-up for mixed-vocab readout dynamics (paper Fig 16 family). Goal: see **when** features become linearly available (1 PC / 5 PCs / full H), not only at the final checkpoint.

## Decision summary

| Question | Choice |
|---|---|
| Need a new run? | **Yes** — current `model.npz` keep final/best weights only (no usable intermediate `W`). |
| One run vs all 50? | **One mid-DFA run first**; optional ~6–10 run stratum later. |
| Aggregate how? | Prefer **within-run feature ranks** on a **normalized training axis** — not pooled raw accuracy. |

## Primary pilot run

- **Task:** `mixeddfa_r26_ns` (DFA = 30, n_words = 14, seed 1)
- Backups if needed: `r25` (DFA 28), `r31` (DFA 32)

## Checkpointing (must do before decode analysis)

Existing `--save-snapshots` on `rnn/min_char_rnn.py` is too dense / huge for H=100 and unsupported on GPU torch trainer.

**Add sparse learning checkpoints** (CPU or torch — prefer whichever already trains these tasks):

1. Save a small weight file at:
   - iter 0 (init)
   - every metric eval after `min_checkpoint_iter`, **or** at word-error crossings ≈ {0.5, 0.2, 0.1, 0.05, 0.03} plus early-stop
2. Store under e.g.  
   `experiments/comparisons/mixed_vocab_dfa_ns/checkpoints/r26/rnn/learning/iter_XXXXXX.npz`  
   (or one stacked archive with `iterations` + weight arrays — keep total ≤ ~20–40 snaps)
3. Record alongside each snap: `iteration`, `word_err`, `smooth_loss` (from existing metric loop)

Do **not** invent a parallel “regen” plot script that regenerates paper figures; extend the canonical mixed-dfa viz path.

## Phase A — single-run analysis (tomorrow, minimum viable)

1. Retrain `mixeddfa_r26_ns` seed 1 with sparse snaps (force new run; don’t reuse final-only ckpt).
2. For each snap, run the same probes as Fig 16 bottom:
   - features: char, DFA, position, position-from-end
   - bases: top-1 PC, top-5 PCs, full hidden
   - metric: chance-corrected linear-probe accuracy
3. Plot:
   - **x** = normalized progress = `iter / iter_stop` (and/or word-error on a twin axis)
   - **y** = chance-corr. acc.
   - small multiples: rows = {1 PC, 5 PCs, full H}, columns = features  
     (or one panel per basis with 4 feature curves)
4. Caption question to answer: does DFA/state readout rise earlier / on fewer PCs than position-from-end?

Deliverable path (suggested):  
`experiments/comparisons/mixed_vocab_dfa_ns/decoding/learning_decode_r26.png`

## Phase B — aggregation (only if Phase A is interesting)

Do **not** average raw accuracy across all 50 DFAs (harder automata → lower absolute DFA-probe accuracy even with chance correction).

Unbiased-ish aggregates:

1. **Within-run feature rank** at each progress quantile  
   - Rank the 4 features by chance-corr. acc (higher = better).  
   - Average ranks across runs → “mean emergence order.”
2. **Normalized emergence time**  
   - For each (run, feature, basis): earliest progress where chance-corr. acc ≥ τ (e.g. 0.5 or 0.8).  
   - Report median ± IQR across runs; stratify by DFA tercile.
3. **Stratified curves**  
   - Mean ± band of chance-corr. acc vs progress within low / mid / high DFA bins (not one pool).

Optional scale-up set (~8 runs): mid 4 + 2 low + 2 high DFA from the existing manifest.

## Implementation notes

- Reuse `viz.compare.decoding.compute_panel_decoding` / `chance_corrected` — load weights from snap into the same model wrapper used by `load_task_decoding_context`.
- GPU trainer currently refuses `--save-snapshots`; either wire sparse saves into torch path or retrain pilot on CPU with the new flag.
- Keep paper collect mapping off this until the figure is intentional for `paper/draft.md`.

## Tomorrow checklist

- [ ] Implement sparse learning-checkpoint save + load
- [ ] Retrain `mixeddfa_r26_ns` seed 1
- [ ] Compute decode curves over snaps
- [ ] Plot Phase A figure; stare at feature timing
- [ ] Decide yes/no on Phase B stratum
- [ ] If yes: pick ≤10 runs, aggregate **ranks / emergence times** only

## Out of scope (for now)

- Full 50-run resave
- Dense `--save-snapshots` history
- Replacing Fig 16 (final-state) — this is a companion learning figure
