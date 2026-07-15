"""Sparse learning-time weight checkpoints (lightweight alternative to --save-snapshots)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# First crossing of each word-error level triggers a snap (high → low).
WORD_ERR_CROSSINGS: tuple[float, ...] = (0.50, 0.20, 0.10, 0.05, 0.03)

# Also snap on a coarse iteration grid so curves aren't only 5–6 points.
DEFAULT_ITER_EVERY = 500


def learning_snap_dir(model_path: str | Path) -> Path:
    """``…/model_seed1.npz`` → ``…/model_seed1_learning/``."""
    p = Path(model_path)
    return p.parent / f"{p.stem}_learning"


def learning_snap_path(model_path: str | Path, iteration: int) -> Path:
    return learning_snap_dir(model_path) / f"iter_{int(iteration):07d}.npz"


def list_learning_snaps(model_path: str | Path) -> list[Path]:
    d = learning_snap_dir(model_path)
    if not d.is_dir():
        return []
    return sorted(d.glob("iter_*.npz"))


def should_save_learning_snap(
    *,
    iteration: int,
    word_err: float,
    crossed: set[float],
    already_saved: set[int],
    iter_every: int = DEFAULT_ITER_EVERY,
    force: bool = False,
) -> bool:
    if force:
        return iteration not in already_saved
    if iteration in already_saved:
        return False
    if iteration == 0:
        return True
    if iter_every > 0 and iteration % iter_every == 0:
        return True
    if np.isfinite(word_err):
        for thr in WORD_ERR_CROSSINGS:
            if thr not in crossed and word_err <= thr:
                return True
    return False


def mark_crossings(word_err: float, crossed: set[float]) -> None:
    if not np.isfinite(word_err):
        return
    for thr in WORD_ERR_CROSSINGS:
        if word_err <= thr:
            crossed.add(thr)


def save_learning_snap(
    model_path: str | Path,
    *,
    iteration: int,
    weights: dict[str, np.ndarray],
    chars: list[str],
    hidden_size: int,
    vocab_size: int,
    word_err: float,
    smooth_loss: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one loadable mini-checkpoint with weights + probe metadata."""
    out = learning_snap_path(model_path, iteration)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        **{k: np.asarray(v) for k, v in weights.items()},
        "chars": np.array(chars),
        "hidden_size": np.array(hidden_size),
        "vocab_size": np.array(vocab_size),
        "learning_snap_iteration": np.array(iteration, dtype=np.int32),
        "learning_snap_word_err": np.array(word_err, dtype=np.float64),
        "learning_snap_smooth_loss": np.array(smooth_loss, dtype=np.float64),
    }
    if extra:
        for k, v in extra.items():
            payload[k] = np.asarray(v)
    np.savez_compressed(out, **payload)
    print(f"learning snap -> {out}  (iter={iteration}, word_err={100.0 * word_err:.2f}%)", flush=True)
    return out
