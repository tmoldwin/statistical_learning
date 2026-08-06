# Dale branch — home resume plan

Capacity: demo H=100, mixed H=200. Models under `*/rnn_dale/`. Soft Dale + ReLU.

## Already done (safe on disk / this commit)

| Artifact | Status |
|----------|--------|
| Demo Dale finals (seeds 1,2,3,5,7,8) | done |
| Mixed Dale finals `checkpoints/rXX/rnn_dale/model_seed1.npz` | **50/50** |
| Mixed Dale learning snaps `*_learning/iter_*.npz` | **50/50** |
| Dale `mixed_dfa_panels.json` | **`rnn_dale` x 50** (flushed) |
| Median best WE (mixed H=200) | ~2.5% (92% <=5%) |
| Full plot suite / learning-decode JSON / paper collect | **NOT done** (plot job stopped after panels) |

Training does **not** need to be redone. Panel decode does **not** need to be redone.

## Start here at home

### 0. Health check

```powershell
(Get-ChildItem experiments\comparisons\mixed_vocab_dfa_ns\checkpoints\*\rnn_dale\model_seed1.npz).Count
# -> 50

python -c "import json; d=json.load(open(r'experiments/comparisons/mixed_vocab_dfa_ns/data/mixed_dfa_panels.json')); print(d.get('model_type'), len(d.get('panels',[])))"
# -> want: rnn_dale 50
```

### 1. Finish mixed analysis plots

Panels are already Dale. Prefer avoiding a full panel recompute:

```powershell
python scripts/mixed_dfa_sweep.py plot --model-type rnn_dale --seeds 1 --replot-only
```

Note: `--replot-only` sets `recompute=False`. If learning-decode / metrics / within-corr JSONs are still from old `rnn`, delete those stale JSONs first or run without `--replot-only`:

```powershell
# nuclear: recomputes everything including panels (~45-90+ min)
python scripts/mixed_dfa_sweep.py plot --model-type rnn_dale --seeds 1
```

Recommended middle path (keep Dale panels, redo downstream that still needs Dale):

```powershell
# optional: remove stale non-panel analysis caches if they predate Dale
Remove-Item -ErrorAction SilentlyContinue `
  experiments\comparisons\mixed_vocab_dfa_ns\data\mixed_dfa_metric_board.json, `
  experiments\comparisons\mixed_vocab_dfa_ns\data\within_corr_vs_dfa.json, `
  experiments\comparisons\mixed_vocab_dfa_ns\decoding\learning_decode_by_dfa.json

python scripts/mixed_dfa_sweep.py plot --model-type rnn_dale --seeds 1 --replot-only
```

If that still skips needed work, drop `--replot-only` (will redo panels too — slow).

Expect learning-decode over snaps to be the slow remaining step.

### 2. Sync paper figures

```powershell
python scripts/paper_collect_figures.py
```

(`MODEL_DIR` should already be `rnn_dale`.)

### 3. Commit remaining analysis + paper figs

After plots finish, commit regenerated JSON/PNGs under
`experiments/comparisons/mixed_vocab_dfa_ns/` and `paper/figures/**`.

### 4. Sanity

Spot-check decoding-vs-DFA, learning-decode, trajectory grid.

## Retrain only if ckpts missing

```powershell
python scripts/train_dale_paper.py --demo --force
python scripts/train_dale_paper.py --mixed --seeds 1 --force
```

## Notes

- `plot --model-type rnn_dale` is required; default is still `rnn`.
- Panel collection writes `mixed_dfa_panels.json` only at end of the 50-run loop (already done).
- Old unconstrained `rnn/` ckpts can stay for comparison; paper collect uses `rnn_dale`.
