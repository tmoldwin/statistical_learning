# Tomorrow plan — word identity + learning decode in the paper

Two deliverables: (1) add **word identity** as a standard analysis feature, (2) put **one** readout-over-learning figure into [draft.md](draft.md).

Related prior work: [LEARNING_DECODE_PLAN.md](../experiments/comparisons/mixed_vocab_dfa_ns/LEARNING_DECODE_PLAN.md) (Phase A done; learning snaps + curves already exist).

---

## 1. Add `word` (word identity) as a feature

**Definition.** At each timestep (or condensed prefix), the label is the **vocabulary word of the current corpus segment** — i.e. which lexical item is being streamed, not the raw prefix string and not DFA state. Ambiguous shared prefixes still belong to one segment word in the constructed stream.

Reuse existing segment logic already used for trajectories (`segment_word_label` / word-at-step style), not a new ad-hoc labeling path.

### Code touch points (canonical only — no regen scripts)

| Where | Change |
|---|---|
| `unit_selectivity.py` | Add `"word"` to `ANALYSIS_FEATURES` / decoding feature lists; `FEATURE_DISPLAY` / `FEATURE_COLORS`; field + branch in `TimestepLabels.feature_values`; compute in `build_timestep_labels` (sequential + condensed). |
| `viz/compare/decoding.py` | Append `"word"` to `DECODING_FEATURES` (palette follows `FEATURE_COLORS`). |
| Downstream that hardcode the 4-feat tuple | Mixed DFA viz / learning-decode / within-corr / selectivity example-unit panels — inherit from shared constants where possible; bump any local copies. |
| Demo `visualize.py` paths | Final-state decoding / feature-separation / selectivity that use `ANALYSIS_FEATURES` should pick up `word` automatically once labels exist. |

### Interpretation checklist (before trusting plots)

- Word identity ≠ prefix (many prefixes per word).
- Word identity ≠ DFA state (shared states across words).
- On the six-word demo, expect strong decode when full \(H\) is available; compare rank vs DFA / char on few PCs.
- Condensed view: one label per prefix group — use the segment word of the representative timestep (same index convention as `position_from_end`).

### Recompute (after labels land)

1. Demo seed-mean decoding / separation if those figures should show five features (Figs 7, 9).
2. Mixed final decode curves (Fig 13 family) and cosine-within (Fig 14) if word is in those feature sets.
3. Learning-decode curves (below) with the new feature line.

Prefer extending CLI / viz entrypoints; do not invent a parallel plot script.

---

## 2. Put readout-over-learning into the paper

Existing artifacts (no paper slot yet):

| Candidate | Path | Fit |
|---|---|---|
| **Preferred for main text** | `experiments/comparisons/mixed_vocab_dfa_ns/decoding/learning_decode_r26.png` | Clean 3-panel (1 PC / 5 PCs / full \(H\)) vs progress; one mid-DFA run. |
| Alt (if seed robustness wanted) | `.../learning_decode_r41_seed_mean.png` | Seed-mean; busier. |
| Too dense for main | `.../learning_decode_by_dfa.png` | Keep supplementary / omit unless pruned hard. |

### Paper integration steps

1. Replot the chosen figure **with word identity** included (same snap JSON path if possible; recompute if feature list changed).
2. Sync via `scripts/paper_collect_figures.py` → e.g. `paper/figures/compare/fig_mixed_learning_decode.jpg`.
3. Insert in draft **after Fig 13** (final-state decoding), before cosine-within:
   - New **Figure 14** = learning decode  
   - Renumber current 14 → 15 (cosine), 15 → 16 (weights)
4. Short caption/story: when features become linearly available relative to word-error collapse; whether word identity tracks DFA or rises later / needs more dimensions.

### Caption question to answer

Does **word identity** emerge with DFA (structure) or only once the network can hold full lexical routing — and is that visible on 5 PCs vs full \(H\)?

---

## Done (demo)

- [x] Implement `word` labels in `build_timestep_labels` + display/colors
- [x] Opt-in `WORD_DECODING_FEATURES` (kept default `DECODING_FEATURES` unchanged)
- [x] Demo figure: [decoding_with_word_seed_mean.png](../experiments/six_word_mixed_demo_ns/rnn/plots/decoding_with_word_seed_mean.png)
- [x] Position×length figure: [decoding_by_position_word_length.png](../experiments/six_word_mixed_demo_ns/rnn/plots/decoding_by_position_word_length.png)

## Remaining checklist

- [ ] Recompute learning-decode for r26 (or r41 seed-mean) including `word`
- [ ] Collect JPG into `paper/figures/compare/`
- [ ] Insert figure + caption into draft; renumber 14–16
- [ ] Visual check: no overlapping text; colorbar/legend readable with 5 feature curves

## Out of scope tomorrow

- Full Phase B multi-run rank aggregation (unless the single figure is ambiguous)
- Replacing final-state Fig 13 — learning figure is a companion
- Dense early-window by-DFA grid in main text
