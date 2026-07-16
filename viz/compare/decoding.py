"""Linear-probe decoding from hidden states and top-k PCA projections."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment import TASKS
from unit_selectivity import FEATURE_COLORS, FEATURE_DISPLAY, build_timestep_labels
from vocab_diagrams import build_minimized_vocabulary_automaton

from viz.compare._data import TaskVizContext

DECODING_FEATURES: tuple[str, ...] = (
    "char", "dfa", "position", "position_from_end", "word",
)
# Alias kept for callers that explicitly opt into word-inclusive readouts.
WORD_DECODING_FEATURES: tuple[str, ...] = DECODING_FEATURES
# Alias of the shared feature-type palette (keep in sync via FEATURE_COLORS).
DECODE_FEATURE_COLORS: dict[str, str] = {
    f: FEATURE_COLORS[f] for f in WORD_DECODING_FEATURES
}
_DEFAULT_MAX_PCS = 20
_DEFAULT_MAX_K = _DEFAULT_MAX_PCS
_DEFAULT_TEST_SIZE = 0.2
_DEFAULT_RANDOM_STATE = 0
_DEFAULT_NEURON_RANDOM_TRIALS = 30


def fit_pca_k(
    points: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA fit to ``k`` components. Returns projected coords, mean, and (k, D) axes."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    k_eff = min(int(k), points.shape[0], points.shape[1])
    if k_eff < 1:
        raise ValueError("need at least one PC")
    components = vh[:k_eff]
    coords = centered @ components.T
    return coords, mean, components


def project_pca_k(
    points: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project ``points`` onto fitted PCA axes."""
    return (points - mean) @ components.T


def select_random_neurons(hidden_dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random subset of ``k`` hidden units (no replacement)."""
    k_eff = min(int(k), hidden_dim)
    if k_eff < 1:
        raise ValueError("need at least one neuron")
    return rng.choice(hidden_dim, size=k_eff, replace=False)


def select_top_variance_neurons(x_train: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest-variance hidden units (fit on train only)."""
    var = np.var(x_train, axis=0)
    k_eff = min(int(k), x_train.shape[1])
    if k_eff < 1:
        raise ValueError("need at least one neuron")
    return np.argsort(var)[-k_eff:][::-1]


def project_neurons(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(points[:, indices], dtype=float)


def _fit_probe_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
    k: int | None = None,
) -> float:
    from sklearn.linear_model import LogisticRegression

    n_train = len(y_train)
    ratio = n_classes / max(n_train, 1)
    C = 1.0
    if ratio > 0.1:
        C = 0.3
    if ratio > 0.2:
        C = 0.1
    if ratio > 0.35:
        C = 0.03
    if k is not None and k > max(2, n_train // 4):
        C *= 0.5

    clf = LogisticRegression(max_iter=2000, C=C, random_state=_DEFAULT_RANDOM_STATE)
    try:
        clf.fit(x_train, y_train)
        return float(clf.score(x_test, y_test))
    except ValueError:
        return float("nan")


def _k_values_for_split(
    *,
    hidden_dim: int,
    n_train: int,
    n_classes: int,
    max_k: int,
) -> list[int]:
    # Cap by geometry / sample size. Do not use n_train//n_classes — that
    # truncates many-state DFA curves early even though L2 logistic probes
    # remain well-defined (C is already scaled by class/sample ratio).
    del n_classes
    upper = min(max_k, hidden_dim, max(1, n_train - 1))
    if upper < 1:
        return []
    return list(range(1, upper + 1))


def empirical_null_chance(feat: str, y: np.ndarray) -> float | None:
    """Null = uniform over label values observed in the condensed corpus."""
    labels = np.asarray(y)
    if len(labels) == 0:
        return None
    n = len(np.unique(labels))
    return 1.0 / n if n > 0 else None


def chance_corrected(acc: float, chance: float) -> float:
    """Map chance→0 and perfect→1: (acc − chance) / (1 − chance)."""
    if not np.isfinite(acc) or not np.isfinite(chance) or chance >= 1.0:
        return float("nan")
    return float((acc - chance) / (1.0 - chance))


def theoretical_chance(
    feat: str,
    *,
    words: list[str],
    length: int,
    automaton: Any | None = None,
) -> float | None:
    """Uniform-guess null for each label type given vocabulary structure."""
    if feat == "char":
        letters = set("".join(words))
        return 1.0 / len(letters) if letters else None
    if feat == "dfa":
        if automaton is None:
            return None
        n_states = int(automaton.dfa._n)
        return 1.0 / n_states if n_states > 0 else None
    if feat in ("position", "position_from_end"):
        return 1.0 / length if length > 0 else None
    if feat == "word":
        return 1.0 / len(words) if words else None
    return None


def null_chances_for_vocab(
    words: list[str],
    length: int,
    *,
    features: tuple[str, ...] | None = None,
) -> dict[str, float]:
    automaton = build_minimized_vocabulary_automaton(words)
    feats = features if features is not None else DECODING_FEATURES
    out: dict[str, float] = {}
    for feat in feats:
        ch = theoretical_chance(feat, words=words, length=length, automaton=automaton)
        if ch is not None and np.isfinite(ch):
            out[feat] = float(ch)
    return out


def prepare_decoding_data(
    ctx: TaskVizContext,
    *,
    features: tuple[str, ...] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-feature (X, y) on all timesteps."""
    feats = features if features is not None else DECODING_FEATURES
    automaton = build_minimized_vocabulary_automaton(ctx.words)
    x = np.asarray(ctx.hidden_states, dtype=float)
    labels = build_timestep_labels(
        ctx.text,
        automaton,
        spaced=ctx.spaced,
        words=ctx.words,
        model=ctx.model,
        activations=x,
    )

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feat in feats:
        vals, mask = labels.feature_values(feat)
        if mask is not None:
            idxs = [i for i, ok in enumerate(mask) if ok]
            out[feat] = (x[idxs], np.asarray([vals[i] for i in idxs]))
        else:
            out[feat] = (x, np.asarray(vals))
    return out


def decode_feature_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_k: int = _DEFAULT_MAX_K,
    basis: str = "pca",
    test_size: float = _DEFAULT_TEST_SIZE,
    random_state: int = _DEFAULT_RANDOM_STATE,
    neuron_sampling: str = "variance",
    n_random_trials: int = _DEFAULT_NEURON_RANDOM_TRIALS,
    neuron_rng_seed: int = 0,
) -> dict[str, Any] | None:
    """Train/test linear probe on full hidden state and top-k PCA or neuron subspace."""
    from sklearn.model_selection import train_test_split

    if basis not in ("pca", "neuron"):
        raise ValueError(f"unknown basis {basis!r}")
    if neuron_sampling not in ("variance", "random"):
        raise ValueError(f"unknown neuron_sampling {neuron_sampling!r}")

    y_arr = np.asarray(y)
    if len(y_arr) < 4:
        return None

    classes, counts = np.unique(y_arr, return_counts=True)
    supported = set(classes[counts >= 2])
    if len(supported) < 2:
        return None
    keep = np.array([yi in supported for yi in y_arr])
    x = x[keep]
    y_arr = y_arr[keep]
    if len(y_arr) < 4:
        return None

    n_classes = len(supported)
    chance = float(1.0 / n_classes)

    n = len(y_arr)
    abs_test = max(int(np.ceil(n * test_size)), 1)
    if abs_test < n_classes:
        abs_test = n_classes
    if n - abs_test < n_classes:
        return None
    frac = abs_test / n

    try:
        train_idx, test_idx = train_test_split(
            np.arange(n),
            test_size=frac,
            random_state=random_state,
            stratify=y_arr,
        )
    except ValueError:
        return None

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]

    k_values = _k_values_for_split(
        hidden_dim=x.shape[1],
        n_train=len(y_train),
        n_classes=n_classes,
        max_k=max_k,
    )
    by_k: dict[int, float] = {}
    by_k_std: dict[int, float] = {}
    hidden_dim = x.shape[1]
    neuron_rng = np.random.default_rng(neuron_rng_seed)
    for k in k_values:
        if basis == "pca":
            _, mean, components = fit_pca_k(x_train, k)
            xtr = project_pca_k(x_train, mean, components)
            xte = project_pca_k(x_test, mean, components)
            by_k[k] = _fit_probe_accuracy(
                xtr, y_train, xte, y_test, n_classes=n_classes, k=k,
            )
        elif neuron_sampling == "variance":
            idx = select_top_variance_neurons(x_train, k)
            xtr = project_neurons(x_train, idx)
            xte = project_neurons(x_test, idx)
            by_k[k] = _fit_probe_accuracy(
                xtr, y_train, xte, y_test, n_classes=n_classes, k=k,
            )
        else:
            accs = []
            for _ in range(n_random_trials):
                idx = select_random_neurons(hidden_dim, k, neuron_rng)
                xtr = project_neurons(x_train, idx)
                xte = project_neurons(x_test, idx)
                accs.append(
                    _fit_probe_accuracy(
                        xtr, y_train, xte, y_test, n_classes=n_classes, k=k,
                    )
                )
            acc_arr = np.asarray(accs, dtype=float)
            by_k[k] = float(np.nanmean(acc_arr))
            by_k_std[k] = (
                float(np.nanstd(acc_arr, ddof=1)) if np.sum(np.isfinite(acc_arr)) > 1 else 0.0
            )

    full_hidden = _fit_probe_accuracy(
        x_train, y_train, x_test, y_test, n_classes=n_classes, k=None,
    )
    out: dict[str, Any] = {
        "chance": chance,
        "basis": basis,
        "n_classes": int(n_classes),
        "n_label_values": int(len(np.unique(y))),
        "n_samples": int(len(y_arr)),
        "full_hidden": full_hidden,
        "by_k": by_k,
    }
    if basis == "neuron" and neuron_sampling == "random":
        out["neuron_sampling"] = "random"
        out["n_random_trials"] = int(n_random_trials)
        out["by_k_std"] = by_k_std
    return out


def _curve_to_list(curve: dict[str, Any], *, max_k: int, field: str = "by_k") -> list[float]:
    data = curve.get(field) or {}
    return [data.get(k, float("nan")) for k in range(1, max_k + 1)]


def compute_panel_decoding(
    ctx: TaskVizContext,
    *,
    max_k: int = _DEFAULT_MAX_K,
    neuron_sampling: str = "variance",
    n_random_trials: int = _DEFAULT_NEURON_RANDOM_TRIALS,
    neuron_rng_seed: int = 0,
    features: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Decode all features for one (task, seed) checkpoint."""
    feats = features if features is not None else DECODING_FEATURES
    feature_data = prepare_decoding_data(ctx, features=feats)
    hidden_size = int(ctx.hidden_states.shape[1])
    cfg = TASKS.get(ctx.task, {})
    if "hidden_size" in cfg:
        hidden_size = int(cfg["hidden_size"])

    features_out: dict[str, Any] = {}
    n_samples = 0
    vocab_null = null_chances_for_vocab(ctx.words, _infer_length(ctx), features=feats)
    for feat in feats:
        x_feat, y_feat = feature_data[feat]
        n_samples = max(n_samples, len(y_feat))
        label_null = empirical_null_chance(feat, y_feat)
        curve_pca = decode_feature_curve(x_feat, y_feat, max_k=max_k, basis="pca")
        curve_neu = decode_feature_curve(
            x_feat, y_feat, max_k=max_k, basis="neuron",
            neuron_sampling=neuron_sampling,
            n_random_trials=n_random_trials,
            neuron_rng_seed=neuron_rng_seed,
        )
        if curve_pca is None and curve_neu is None:
            features_out[feat] = {"error": "insufficient samples or classes"}
            continue
        curve = curve_pca or curve_neu
        if label_null is not None:
            null_ch = label_null
        else:
            null_ch = vocab_null.get(feat)
        entry: dict[str, Any] = {
            "chance": curve["chance"],
            "null_chance": null_ch,
            "n_classes": curve["n_classes"],
            "n_label_values": curve.get("n_label_values"),
            "n_samples": curve["n_samples"],
        }
        if curve_pca is not None:
            entry["full_hidden"] = curve_pca["full_hidden"]
            entry["by_k"] = _curve_to_list(curve_pca, max_k=max_k)
        if curve_neu is not None:
            entry["full_hidden_neurons"] = curve_neu["full_hidden"]
            entry["by_k_neurons"] = _curve_to_list(curve_neu, max_k=max_k)
            if curve_neu.get("neuron_sampling") == "random":
                entry["neuron_sampling"] = "random"
                entry["n_neuron_trials"] = curve_neu.get("n_random_trials")
                entry["by_k_neurons_std"] = _curve_to_list(
                    curve_neu, max_k=max_k, field="by_k_std",
                )
        features_out[feat] = entry

    null_chance = {
        feat: features_out[feat].get("null_chance")
        for feat in feats
        if feat in features_out and features_out[feat].get("null_chance") is not None
    }
    for feat in feats:
        if feat not in null_chance and feat in vocab_null:
            null_chance[feat] = vocab_null[feat]

    return {
        "task": ctx.task,
        "seed": ctx.seed,
        "hidden_size": hidden_size,
        "n_samples": n_samples,
        "feature_order": list(feats),
        "null_chance": null_chance,
        "features": features_out,
    }


def _infer_length(ctx: TaskVizContext) -> int:
    cfg = TASKS.get(ctx.task, {})
    if "sweep_length" in cfg:
        sweep_length = cfg["sweep_length"]
        if isinstance(sweep_length, (int, float)):
            return int(sweep_length)
    if ctx.words:
        return max(len(w) for w in ctx.words)
    return 1


def feature_display_name(feature: str) -> str:
    return FEATURE_DISPLAY.get(feature, feature)
