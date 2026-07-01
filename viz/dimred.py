"""PCA and JPCA embedding utilities shared across visualization code."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EMBED_METHODS: tuple[str, ...] = ("pca", "jpca")


def embed_save_path(save_path: str, method: str) -> str:
    """Derive sibling output path for an embedding method (e.g. *_pca.png -> *_jpca.png)."""
    if method not in EMBED_METHODS:
        raise ValueError(f"unknown embed method {method!r}; choose from {EMBED_METHODS}")
    p = Path(save_path)
    stem = p.stem
    if method == "pca":
        return save_path
    if stem.endswith("_pca"):
        return str(p.with_name(f"{stem[:-4]}_{method}{p.suffix}"))
    return str(p.with_name(f"{stem}_{method}{p.suffix}"))


def embed_dim_label(method: str) -> str:
    return "JPCA" if method == "jpca" else "PCA"


def fit_pca_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA fit: return 2D coords, mean, and (2, D) principal axes for reconstruction."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    return coords, mean, components


def fit_pca_2d_with_evr(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA fit + explained variance ratio for PC1/PC2."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    denom = float(np.sum(s * s)) if len(s) else 1.0
    evr = (s[:2] * s[:2]) / denom if denom > 0 else np.array([0.0, 0.0])
    return coords, mean, components, evr


def fit_pca_3d_with_evr(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA fit + explained variance ratio for PC1/PC2/PC3."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(3, points.shape[0], points.shape[1])
    components = vh[:n_comp]
    coords = centered @ components.T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    denom = float(np.sum(s * s)) if len(s) else 1.0
    evr = np.zeros(3, dtype=float)
    if denom > 0 and n_comp:
        evr[:n_comp] = (s[:n_comp] * s[:n_comp]) / denom
    return coords, mean, components, evr


def fit_jpca_components(
    trajectories: list[np.ndarray],
    *,
    num_jpcs: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Churchland-style jPCA axes from trajectory segments (each T×D, T >= 3).

    Returns mean (D,), components (num_jpcs, D), and plane rotation rates.
    """
    valid = [np.asarray(t, dtype=float) for t in trajectories if t.shape[0] >= 3]
    if not valid:
        raise ValueError("need at least one trajectory with >= 3 timepoints")

    dim = valid[0].shape[1]
    cross_cov = np.zeros((dim, dim))
    for traj in valid:
        centered = traj - traj.mean(axis=0, keepdims=True)
        deriv = np.diff(centered, axis=0)
        states = centered[:-1]
        cross_cov += states.T @ deriv
    cross_cov /= len(valid)

    skew = cross_cov - cross_cov.T
    eigvals, eigvecs = np.linalg.eig(skew)

    planes: list[tuple[float, np.ndarray, np.ndarray]] = []
    used: set[int] = set()
    for i, eigval in enumerate(eigvals):
        if i in used:
            continue
        imag = float(np.imag(eigval))
        if imag <= 1e-10:
            continue
        vec = eigvecs[:, i]
        jpc1 = np.real(vec)
        jpc2 = np.imag(vec)
        n1, n2 = float(np.linalg.norm(jpc1)), float(np.linalg.norm(jpc2))
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        jpc1 /= n1
        jpc2 /= n2
        planes.append((imag, jpc1, jpc2))
        used.add(i)
        for j in range(i + 1, len(eigvals)):
            if abs(eigvals[j] - np.conj(eigval)) < 1e-6:
                used.add(j)
                break

    planes.sort(key=lambda item: -item[0])
    if not planes:
        raise ValueError("no rotational jPCA planes found")

    components: list[np.ndarray] = []
    rates: list[float] = []
    for rate, jpc1, jpc2 in planes:
        rates.append(rate)
        components.extend([jpc1, jpc2])
        if len(components) >= num_jpcs:
            break

    while len(components) < num_jpcs:
        components.append(np.zeros(dim, dtype=float))

    mean = np.vstack(valid).mean(axis=0)
    return mean, np.stack(components[:num_jpcs]), np.asarray(rates, dtype=float)


def trajectories_from_segments(
    hidden_states: np.ndarray,
    segments: list[tuple[int, int, str]],
    *,
    min_len: int = 3,
) -> list[np.ndarray]:
    """Extract trajectory arrays from inclusive corpus segment index ranges."""
    out: list[np.ndarray] = []
    for start, end, _word in segments:
        if end < start:
            continue
        traj = hidden_states[start : end + 1]
        if traj.shape[0] >= min_len:
            out.append(traj)
    return out


def trajectories_for_embed(
    hidden_states: np.ndarray,
    *,
    segments: list[tuple[int, int, str]] | None = None,
    word_path_indices: list[list[int]] | None = None,
) -> list[np.ndarray]:
    """Build trajectory list for JPCA fitting."""
    if word_path_indices is not None:
        trajs = [hidden_states[idxs] for idxs in word_path_indices if len(idxs) >= 3]
        if trajs:
            return trajs
    if segments:
        trajs = trajectories_from_segments(hidden_states, segments)
        if trajs:
            return trajs
    if hidden_states.shape[0] >= 3:
        return [hidden_states]
    return []


def fit_embed_2d_with_evr(
    points: np.ndarray,
    *,
    method: str = "pca",
    trajectories: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2D linear embedding (PCA or JPCA) and project ``points``."""
    points = np.asarray(points, dtype=float)
    if method == "pca":
        return fit_pca_2d_with_evr(points)
    trajs = trajectories or trajectories_for_embed(points)
    try:
        mean, components, rates = fit_jpca_components(trajs, num_jpcs=2)
    except ValueError:
        return fit_pca_2d_with_evr(points)
    coords = (points - mean) @ components.T
    rate = float(rates[0]) if len(rates) else 0.0
    evr = np.array([rate, rate], dtype=float)
    return coords, mean, components, evr


def fit_embed_3d_with_evr(
    points: np.ndarray,
    *,
    method: str = "pca",
    trajectories: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 3D linear embedding (PCA or JPCA) and project ``points``."""
    points = np.asarray(points, dtype=float)
    if method == "pca":
        return fit_pca_3d_with_evr(points)
    trajs = trajectories or trajectories_for_embed(points)
    try:
        mean, components, rates = fit_jpca_components(trajs, num_jpcs=3)
    except ValueError:
        return fit_pca_3d_with_evr(points)
    if components.shape[0] < 3:
        _, pca_components, _, pca_evr = fit_pca_3d_with_evr(points)
        components = np.vstack([components, pca_components[components.shape[0] : 3]])
        rates = np.concatenate([rates, pca_evr[components.shape[0] - 1 : 2]])
    coords = (points - mean) @ components[:3].T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    evr = np.zeros(3, dtype=float)
    if len(rates) >= 1:
        evr[0] = evr[1] = float(rates[0])
    if len(rates) >= 2:
        evr[2] = float(rates[1])
    elif len(rates) == 1:
        evr[2] = float(rates[0])
    return coords, mean, components[:3], evr


def embed_axis_labels_2d(evr: np.ndarray, method: str) -> tuple[str, str]:
    if method == "jpca":
        rate = float(evr[0]) if len(evr) else 0.0
        return f"jPC1 (ω={rate:.3f})", f"jPC2 (ω={rate:.3f})"
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    return f"PC1 ({pc1:.1f}%)", f"PC2 ({pc2:.1f}%)"


def embed_axis_labels_3d(evr: np.ndarray, method: str) -> tuple[str, str, str]:
    if method == "jpca":
        r1 = float(evr[0]) if len(evr) > 0 else 0.0
        r2 = float(evr[2]) if len(evr) > 2 else (float(evr[1]) if len(evr) > 1 else r1)
        return (
            f"jPC1 (ω={r1:.3f})",
            f"jPC2 (ω={r1:.3f})",
            f"jPC3 (ω={r2:.3f})",
        )
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    pc3 = 100.0 * float(evr[2]) if len(evr) > 2 else 0.0
    return f"PC1 ({pc1:.1f}%)", f"PC2 ({pc2:.1f}%)", f"PC3 ({pc3:.1f}%)"
