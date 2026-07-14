# Pow2 sweep (h100, L1–6, max 32w)

New experiment alongside the existing `word_count_pow2_sweep_ns` run. Same seeds and
pipeline; smaller grid, fixed hidden size, no L7 or w64.

## Grid

| Axis | Values |
|---|---|
| Word counts | 1, 2, 4, 8, 16, 32 |
| Lengths | 1, 2, 3, 4, 5, 6, mixed |
| Hidden | 100 (all cells) |
| Train seeds | 1–15 |
| Seed-comparison seeds | 1–7 |
| Cells | 6 × 7 = 42 |
| Total training jobs | 42 × 15 = 630 |

**Mixed vocab:** split only over lengths 3–6 (four pools). L1/L2 cells remain
fixed-length conditions only.

## Comparison folder

```
experiments/comparisons/word_count_pow2_sweep_h100_ns/
```

Task prefix (avoids collision with original sweep):

- `pow2sweep_h100_w{N}_l{L}_ns`
- `pow2sweep_h100_w{N}_lmix_ns`

Checkpoint leaves: `checkpoints/w{N}/l{L}_ns` or `…/lmix_ns`.

## Code changes (before training)

1. **New sweep config module** (`vocab_sweep_pow2_h100.py`)
   - Constants: word counts, lengths, seeds, comparison name
   - `hidden_size = 100` always
   - Mixed: `_mixed_vocab` over lengths 3–6 only (not 7)
   - Drop `n_words >= 64` branches from step/stall scaling
   - Register regimes + tasks in `task.py` / `experiment.py`

2. **Checkpoint routing** — extend `experiment_subpath()` so `pow2sweep_h100_*` maps
   to this comparison folder.

3. **New driver script** (`scripts/pow2_sweep_h100.py`) — same `plan | train | plot`
   interface as `scripts/pow2_sweep.py`.

4. **Parameterize viz modules** — comparison name + axis constants are currently
   hardcoded in four `viz/compare/pow2_sweep_*.py` files; point them at this sweep
   (or pass comparison name through).

## Training hyperparameters

Keep existing pow2 scaling formulas for steps, LR, stall patience, rollout length, etc.
Only change:

- `hidden_size → 100`
- Remove `n_words >= 64` branches
- Mixed split over lengths 3–6

Revisit if w32/L6 cells stall or overfit with fixed-100 hidden on small vocabs.

## Output tree

```
word_count_pow2_sweep_h100_ns/
├── checkpoints/
│   └── w{1,2,4,8,16,32}/
│       └── {l1_ns … l6_ns, lmix_ns}/     # 42 leaves
│           ├── input.txt
│           ├── rnn/model_seed{S}.{npz,progress}
│           └── shared/vocabulary_{trie,min_dfa}.svg
├── data/
│   ├── sweep_training.json
│   ├── sweep_geometry.json
│   ├── sweep_spectra.json
│   └── sweep_decoding.json
├── learning_curves/
│   └── overview_seed{S}.png               # 7 rows × 6 cols
├── trajectories/
│   ├── sweep_heatmaps.png
│   ├── sweep_pc_spectra.png              # PCA variance
│   └── sweep_closed_loop_2d_seed{S}.png  # PCA trajectories
├── seed_comparison/
│   ├── by_length/
│   │   └── closed_loop_2d_{l1…l6|lmix}.png   # 7 figs; rows=w, cols=seed
│   └── by_word_count/
│       └── closed_loop_2d_w{N}.png           # 6 figs; rows=L, cols=seed
├── sequences/
│   └── demo_sequences_seed{S}.png
└── decoding/
    ├── sweep_decode_curves.png           # PCA-basis decoding
    └── sweep_decode_neuron_curves.png
```

## Figures and plot pipeline

### Phase 1 — Train

```bash
python scripts/pow2_sweep_h100.py plan
python scripts/pow2_sweep_h100.py train --jobs 4
# optional pilot:
python scripts/pow2_sweep_h100.py train --seeds 1 2 3
```

`--jobs N` trains up to N cells in parallel. `--device auto` uses PyTorch GPU when available.

### Phase 2 — Full plot (PCA included)

Default `plot` should run everything below, including decoding (not in the
current original-sweep default):

| Step | Outputs | PCA? |
|---|---|---|
| Heatmaps + geometry | `data/sweep_*.json`, `sweep_heatmaps.png` | geometry uses top-2 PC var |
| PC spectra | `sweep_spectra.json`, `sweep_pc_spectra.png` | yes |
| Learning curves | `learning_curves/overview_seed*.png` | no |
| Closed-loop trajectories | `sweep_closed_loop_2d_seed*.png` | yes |
| Demo sequences | `sequences/demo_sequences_seed*.png` | no |
| Seed comparison | `seed_comparison/**/closed_loop_2d_*.png` | yes |
| PCA decoding | `decoding/sweep_decode_curves.png` | yes |

Default `plot` precomputes hidden-state caches under `data/viz_cache/` (corpus +
decoding rollouts) so replots and follow-on analyses skip redundant forward passes.
Use `--refresh-cache` to rebuild; `--skip-cache` to bypass.

```bash
python scripts/pow2_sweep_h100.py plot
```

Incremental reruns:

```bash
python scripts/pow2_sweep_h100.py plot --trajectories-only
python scripts/pow2_sweep_h100.py plot --spectrum-only
python scripts/pow2_sweep_h100.py plot --decoding-only
python scripts/pow2_sweep_h100.py plot --seed-comparison-only
```

**Not in default plot pass:** jPCA variants (`*_jpca.png`). Add later with
`embed_methods=("pca", "jpca")` if needed.

## Grid layout (overview figures)

- **Rows:** lengths 1, 2, 3, 4, 5, 6, mixed (7)
- **Cols:** word counts 1, 2, 4, 8, 16, 32 (6)

Seed-comparison panels flip axes: by-length → rows = word count, cols = seed;
by-word-count → rows = length, cols = seed.

## Implementation order

1. Wire config + task registration + checkpoint routing
2. Parameterize viz comparison name
3. `plan` → confirm 42 cells / 630 jobs
4. Pilot train seeds 1–3 on a few cells
5. Full train
6. Full plot (with decoding in default)
7. Spot-check PCA panels vs learning curves
