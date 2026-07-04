"""Scalar readouts of hidden-state dimensionality and correlation structure."""

from __future__ import annotations

from typing import Any

import numpy as np


def _cov_eigenvalues(points: np.ndarray) -> np.ndarray:
    """Variance eigenvalues of centered ``points`` (rows = observations)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 1:
        return np.empty(0)
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    dof = max(points.shape[0] - 1, 1)
    return (s * s) / dof


def variance_top_k_frac(points: np.ndarray, k: int = 2) -> float:
    """Fraction of total variance in the top ``k`` principal components."""
    var = _cov_eigenvalues(points)
    if len(var) == 0:
        return float("nan")
    total = float(var.sum())
    if total <= 1e-12:
        return float("nan")
    n = min(max(int(k), 1), len(var))
    return float(var[:n].sum() / total)


def participation_ratio(points: np.ndarray) -> float:
    """Effective dimensionality: (sum λ)² / sum λ² on covariance eigenvalues."""
    var = _cov_eigenvalues(points)
    if len(var) == 0:
        return float("nan")
    num = float(var.sum()) ** 2
    den = float((var * var).sum())
    return num / den if den > 1e-12 else float("nan")


def dims_for_variance_frac(points: np.ndarray, threshold: float = 0.9) -> float:
    """Smallest number of PCs whose cumulative variance reaches ``threshold``."""
    var = _cov_eigenvalues(points)
    if len(var) == 0:
        return float("nan")
    total = float(var.sum())
    if total <= 1e-12:
        return float("nan")
    cum = np.cumsum(var) / total
    idx = int(np.searchsorted(cum, threshold, side="left"))
    return float(min(idx + 1, len(var)))


def mean_offdiag_abs_corr(points: np.ndarray) -> float:
    """Mean |Pearson r| over off-diagonal pairs of row vectors."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 3:
        return float("nan")
    corr = np.corrcoef(points)
    np.fill_diagonal(corr, np.nan)
    vals = np.abs(corr)
    finite = vals[np.isfinite(vals)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def state_space_metrics(points: np.ndarray) -> dict[str, float]:
    """Dimensionality + correlation summary for one cloud of hidden vectors."""
    points = np.asarray(points, dtype=float)
    return {
        "effective_dim": participation_ratio(points),
        "top2_variance_frac": variance_top_k_frac(points, 2),
        "dims_90pct": dims_for_variance_frac(points, 0.9),
        "mean_abs_corr": mean_offdiag_abs_corr(points),
        "n_points": float(len(points)),
        "n_dims": float(points.shape[1]) if points.ndim == 2 and len(points) else float("nan"),
    }


def paired_state_space_metrics(
    corpus_states: np.ndarray,
    loop_states: np.ndarray,
) -> dict[str, Any]:
    """Teacher-forced corpus cloud vs closed-loop trajectory in ℝᴴ."""
    return {
        "corpus": state_space_metrics(corpus_states),
        "loop": state_space_metrics(loop_states),
    }
