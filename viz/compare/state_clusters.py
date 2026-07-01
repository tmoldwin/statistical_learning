"""Cluster-based vertex proxy: count groups in state clouds (ℝᴴ and PCA 2D)."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def _prepare_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2:
        return points
    if points.shape[1] > 2:
        centered = points - points.mean(axis=0)
        std = centered.std(axis=0)
        std[std < 1e-12] = 1.0
        return centered / std
    return points


def count_state_clusters(
    points: np.ndarray,
    *,
    link_threshold_frac: float = 0.12,
) -> int:
    """Agglomerative cluster count at a fraction of the max pairwise distance."""
    points = _prepare_points(points)
    n = len(points)
    if n <= 1:
        return max(n, 0)
    dists = pdist(points, metric="euclidean")
    if len(dists) == 0:
        return 1
    max_d = float(np.max(dists))
    if max_d <= 1e-12:
        return 1
    thr = link_threshold_frac * max_d
    z = linkage(dists, method="average")
    labels = fcluster(z, t=thr, criterion="distance")
    return int(len(set(labels)))


def cluster_counts_hidden_and_pca(
    hidden_states: np.ndarray,
    pca_xy: np.ndarray,
    *,
    link_threshold_frac_hidden: float = 0.58,
    link_threshold_frac_pca: float = 0.42,
) -> tuple[int, int]:
    """Return (n_clusters_ℝᴴ, n_clusters_PCA2)."""
    k_h = count_state_clusters(
        hidden_states, link_threshold_frac=link_threshold_frac_hidden,
    )
    k_pca = count_state_clusters(pca_xy, link_threshold_frac=link_threshold_frac_pca)
    return k_h, k_pca


def format_cluster_counts(k_h: int, k_pca: int) -> str:
    return f"kH={k_h} k2={k_pca}"
