# Word-count × length sweep (h100, L1–6, 5–25 words)

Replaces the archived powers-of-two grid
(`experiments/comparisons/old/word_count_pow2_sweep_h100_ns_pow2`).

## Grid

| Axis | Values |
|---|---|
| Word counts | 5, 10, 15, 20, 25 |
| Lengths | 1, 2, 3, 4, 5, 6, mixed |
| Hidden | 100 (all cells) |
| Train seeds | 1–15 |
| Seed-comparison seeds | 1–5 |
| Cells | 5 × 7 = 35 |
| Total training jobs | 35 × 15 = 525 |

**Mixed vocab:** split over lengths 3–6 (four pools).

## Comparison folder

```
experiments/comparisons/word_count_pow2_sweep_h100_ns/
```

Task prefix: `pow2sweep_h100_w{N}_l{L}_ns` / `…_lmix_ns`

## CLI

```bash
python scripts/pow2_sweep_h100.py plan --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
python scripts/pow2_sweep_h100.py train --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --jobs 4
python scripts/pow2_sweep_h100.py plot --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```
