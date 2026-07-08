"""
Visualize a trained min-char-rnn after training is done.

Loads the saved model from `model.npz`, runs a forward pass over the first
`--length` characters of `input.txt`, and plots:

  1) activation_heatmap.png
       Heatmap of every hidden unit's activation at every timestep.
       Y-axis: hidden units (h0, h1, ...). X-axis: input character.
       Works for any `hidden_size`.

  2) next_char_prob_sequence_heatmap.png
       Heatmap of the model's next-char probability distribution at every
       position, with the actual next character marked.

  3) state_trajectory_by_input.png   (only when hidden_size == 2)
       2D scatter of every hidden state visited, colored by the input
       character that produced it, with grey arrows showing the temporal
       trajectory through state space.

  4) state_trajectory_by_target.png    (only when hidden_size == 2)
       Same scatter, colored by the *next* (target) character.

  5) learning_curve.png
       Per-window training loss vs iteration (from model.npz).

  6) embedding_panels_context.png
       2D PCA of vectors with context labels.

  7) next_char_regions_pca.png
       Two PCA panels: argmax next-char regions and prediction entropy (2D h).

  8) next_char_prob_panels_pca.png
       One panel per vocab char: P(next = char) over the PCA plane (softmax).

  9) activation_clustered_heatmap.png
       Heatmap of timesteps × hidden units with row/column dendrograms
       (average linkage). Row labels: two preceding chars + current char.

  10) state_correlation_clustered_heatmap.png
       Timestep × timestep Pearson correlation of hidden states, hierarchically
       clustered; row/column labels = prefix since last space; tick colors = min DFA state.

  10b) state_correlation_by_dfa_state.png
       Timesteps grouped by min DFA state; Pearson r within and between state blocks.

  11) dfa_state_distance_comparison.png
       Pairwise Euclidean distances between hidden states; within vs between
       minimized DFA state, same input character, same position in word (all timestep pairs).

  12) weights.png
       Side-by-side heatmaps of final input weights (char columns × hidden rows)
       and recurrent hidden→hidden weights (h0..h{n-1} in index order).

  13) weight_dynamics_over_training.png
       Eight E/I-block heatmaps of W_xh and W_hh weights over training snapshots.

  14) learning_dynamics/hidden_state_pca.mp4 (optional; pass --learning-dynamics-video)
       Hidden states on a fixed final-model PCA basis across weight snapshots.

Usage:
    python visualize.py --exp ten_word_overlap_s
    python visualize.py --exp ten_word_overlap --length 100
    python visualize.py --model path/to/model.npz --input path/to/input.txt --out-dir path/to/plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage

from experiment import (
    ensure_experiment_dirs,
    experiment_uses_word_space,
    input_path,
    learning_dynamics_dir,
    model_dir,
    model_path,
    plots_dir,
    shared_dir,
)
from transformer.adapter import forward_pass as transformer_forward_pass
from transformer.adapter import load_model as load_transformer_model
from transformer.viz_repr import run_transformer_visualization
from viz_timing import VizTimer
from readme_figures import (
    numbered_plot_path,
    remove_flat_plot_files,
    remove_legacy_readme_plot_names,
    remove_shared_figures_from_model_plots,
)
from task import REGIMES
from rnn.rnn_dyn import activation_label, inject_timestep_noise, no_input_hidden_step, rnn_hidden_step
from vocab_diagrams import (
    MinimizedVocabAutomaton,
    build_minimized_vocabulary_automaton,
    build_vocabulary_coverage_text,
    dfa_state_at_position,
    dfa_state_for_prefix,
    dfa_state_label,
    draw_minimized_dfa_on_axes,
    in_word_prefix_at_position,
    in_word_prefix_before_current,
    position_in_word_at_index,
    position_in_word_for_prefix_label,
    prefix_before_from_string_label,
    segment_corpus_by_words,
    trie_prefix_display_order,
    vocabulary_for_experiment,
    write_vocabulary_diagrams,
)


def load_model(path: str = "model.npz"):
    data = np.load(path, allow_pickle=True)
    model = {
        "weights_input_to_hidden":  data["weights_input_to_hidden"],
        "weights_hidden_to_hidden": data["weights_hidden_to_hidden"],
        "weights_hidden_to_output": data["weights_hidden_to_output"],
        "bias_hidden":              data["bias_hidden"],
        "bias_output":              data["bias_output"],
        "chars":                    [str(c) for c in data["chars"]],
        "hidden_size":              int(data["hidden_size"]),
        "vocab_size":               int(data["vocab_size"]),
    }
    if "sequence_length" in data.files:
        model["sequence_length"] = int(data["sequence_length"])
    if "loss_iterations" in data.files:
        model["loss_iterations"] = data["loss_iterations"]
        model["loss_smooth"] = data["loss_smooth"]
        model["loss_window"] = data["loss_window"]
    if "metric_iterations" in data.files:
        model["metric_iterations"] = data["metric_iterations"]
        model["metric_valid_vocab_letter_frac"] = data["metric_valid_vocab_letter_frac"]
    if "metric_word_error_frac" in data.files:
        model["metric_word_error_frac"] = data["metric_word_error_frac"]
    if "metric_val_ce" in data.files:
        model["metric_val_ce"] = data["metric_val_ce"]
    if "metric_rollout_samples" in data.files:
        model["metric_rollout_samples"] = [str(s) for s in data["metric_rollout_samples"]]
    if "vocab_words" in data.files:
        model["vocab_words"] = [str(w) for w in data["vocab_words"]]
    if "sample_before" in data.files:
        model["sample_before"] = str(data["sample_before"])
    if "sample_after" in data.files:
        model["sample_after"] = str(data["sample_after"])
    if "demo_snippet" in data.files:
        model["demo_snippet"] = str(data["demo_snippet"])
    elif "demo_prompt" in data.files or "demo_target" in data.files:
        model["demo_snippet"] = (
            str(data["demo_prompt"]) if "demo_prompt" in data.files else ""
        ) + (
            str(data["demo_target"]) if "demo_target" in data.files else ""
        )
    if "demo_before" in data.files:
        model["demo_before"] = str(data["demo_before"])
    if "demo_after" in data.files:
        model["demo_after"] = str(data["demo_after"])
    if "demo_word_error_frac" in data.files:
        model["demo_word_error_frac"] = float(data["demo_word_error_frac"])
    if "demo_rng_seed" in data.files:
        model["demo_rng_seed"] = int(data["demo_rng_seed"])
    if "demo_seed_char" in data.files:
        model["demo_seed_char"] = str(data["demo_seed_char"])
    if "dale_law" in data.files:
        model["dale_law"] = bool(data["dale_law"])
    if "use_relu" in data.files:
        model["use_relu"] = bool(data["use_relu"])
    elif "dale_law" in model:
        model["use_relu"] = model["dale_law"]
    else:
        model["use_relu"] = False
    if "e_fraction" in data.files:
        model["e_fraction"] = float(data["e_fraction"])
    if "dale_sign" in data.files:
        ds = data["dale_sign"]
        model["dale_sign"] = ds if len(ds) else None
    if "weight_snap_iterations" in data.files and "weight_snap_outgoing" in data.files:
        model["weight_snap_iterations"] = data["weight_snap_iterations"]
        model["weight_snap_outgoing"] = data["weight_snap_outgoing"]
        if "weight_snap_violation_frac" in data.files:
            model["weight_snap_violation_frac"] = data["weight_snap_violation_frac"]
    if "weight_snap_bias_hidden" in data.files:
        model["weight_snap_bias_hidden"] = data["weight_snap_bias_hidden"]
    if "weight_snap_bias_output" in data.files:
        model["weight_snap_bias_output"] = data["weight_snap_bias_output"]
    if "metric_word_error_frac" in data.files:
        model["metric_word_error_frac"] = data["metric_word_error_frac"]
    if "timestep_noise_std" in data.files:
        model["timestep_noise_std"] = float(data["timestep_noise_std"])
    else:
        model["timestep_noise_std"] = 0.0
    return model


def forward_pass(model, text: str):
    """Run the trained RNN over `text` and return per-timestep states + probs."""
    hidden_size = model["hidden_size"]
    vocab_size  = model["vocab_size"]
    chars       = model["chars"]
    char_to_index = {c: i for i, c in enumerate(chars)}

    weights_input_to_hidden  = model["weights_input_to_hidden"]
    weights_hidden_to_hidden = model["weights_hidden_to_hidden"]
    weights_hidden_to_output = model["weights_hidden_to_output"]
    bias_hidden              = model["bias_hidden"]
    bias_output              = model["bias_output"]
    noise_std = float(model.get("timestep_noise_std", 0.0))
    noise_rng = np.random.default_rng()

    hidden_state = np.zeros((hidden_size, 1))
    hidden_states = np.zeros((len(text), hidden_size))
    output_probs  = np.zeros((len(text), vocab_size))

    for t, char in enumerate(text):
        input_one_hot = np.zeros((vocab_size, 1))
        input_one_hot[char_to_index[char]] = 1
        hidden_state, _ = rnn_hidden_step(
            hidden_state,
            input_one_hot,
            weights_input_to_hidden,
            weights_hidden_to_hidden,
            bias_hidden,
            use_relu=model.get("use_relu", False),
            timestep_noise_std=noise_std,
            noise_rng=noise_rng,
        )
        logits = weights_hidden_to_output @ hidden_state + bias_output
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        hidden_states[t] = hidden_state.ravel()
        output_probs[t]  = probs.ravel()

    return hidden_states, output_probs


def plot_state_trajectory(
    hidden_states,
    color_by_chars,
    chars,
    title,
    save_path,
    *,
    condensed: CondensedView | None = None,
):
    """2D scatter of hidden states colored by some categorical char per timestep."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        title = _condensed_plot_title(title, condensed)
    if hidden_states.shape[1] != 2:
        raise ValueError(
            f"This plot expects hidden_size == 2, got {hidden_states.shape[1]}. "
            f"Re-train with hidden_size = 2 (already the default in min-char-rnn.py)."
        )

    cmap = plt.get_cmap("tab10")
    char_to_color = {c: cmap(i) for i, c in enumerate(chars)}

    fig, ax = plt.subplots(figsize=(8, 7))

    xs, ys = hidden_states[:, 0], hidden_states[:, 1]
    ax.plot(xs, ys, color="lightgrey", linewidth=0.5, zorder=1)
    ax.quiver(
        xs[:-1], ys[:-1],
        xs[1:] - xs[:-1], ys[1:] - ys[:-1],
        angles="xy", scale_units="xy", scale=1,
        color="lightgrey", width=0.002, headwidth=4, alpha=0.6, zorder=1,
    )

    for c in chars:
        mask = np.array([ch == c for ch in color_by_chars])
        if not mask.any():
            continue
        ax.scatter(
            xs[mask], ys[mask],
            color=char_to_color[c], label=repr(c), s=30,
            edgecolor="black", linewidth=0.3, zorder=3,
        )

    ax.set_xlabel("hidden unit 0")
    ax.set_ylabel("hidden unit 1")
    ax.set_title(title)
    ax.legend(title="char", loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_hidden_states_heatmap(
    text,
    hidden_states,
    save_path,
    *,
    act_label: str = "tanh",
    y_label: str = "hidden unit",
    title: str | None = None,
    colorbar_label: str | None = None,
    condensed: CondensedView | None = None,
    exp_name: str | None = None,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
    words: list[str] | None = None,
    cluster_units: bool = False,
):
    """Heatmap of a per-timestep vector representation over the sequence."""
    prefix_keys: list[str] | None = None
    if condensed is not None:
        hidden_states = condensed.hidden_states
        prefix_keys = condensed.labels
        x_labels = [_display_prefix_label(l) for l in prefix_keys]
        x_axis = prefix_axis_label(
            spaced=condensed.spaced, text=text, words=condensed.words,
        )
        spaced = condensed.spaced
        words = condensed.words
    else:
        x_labels = list(text)
        x_axis = "timestep / input character"
    length, hidden_size = hidden_states.shape
    use_relu = act_label == "relu"
    use_raw = act_label == "raw"
    cmap = "magma" if use_relu else "RdBu_r"
    if use_raw:
        cmap = "RdBu_r"
        vmin = None
        vmax = None
    else:
        vmin = 0.0 if use_relu else -1.0
        vmax = None if use_relu else 1.0

    default_title = (
        f"Hidden state activations ({act_label} output) over the input sequence"
        if y_label == "hidden unit"
        else f"{y_label} over the input sequence"
    )
    if cluster_units:
        default_title += " · units hierarchically clustered"

    if cluster_units:
        unit_labels = [
            f"h{i}" if y_label == "hidden unit" else f"{y_label}{i}"
            for i in range(hidden_size)
        ]
        data = pd.DataFrame(
            hidden_states.T,
            index=unit_labels,
            columns=x_labels,
        )
        grid = sns.clustermap(
            data,
            row_cluster=True,
            col_cluster=False,
            method="average",
            metric="euclidean",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=None if use_relu or use_raw else 0.0,
            figsize=(max(12, length * 0.15), max(2.5, hidden_size * 0.35)),
            dendrogram_ratio=(0.08, 0.0),
            cbar_kws={"label": colorbar_label or (f"activation ({act_label})" if not use_raw else "value")},
            xticklabels=True,
            yticklabels=True,
        )
        if automaton is not None:
            if prefix_keys is not None:
                state_ids = _dfa_state_ids_for_prefixes(
                    prefix_keys, automaton, spaced=spaced,
                )
            else:
                state_ids = _dfa_state_ids_at_timesteps(
                    text, automaton, spaced=spaced, words=words,
                )
            _color_tick_labels_by_state_ids(grid.ax_heatmap.get_xticklabels(), state_ids)
            x_axis += " · tick color = min DFA state"
        grid.ax_heatmap.set_xlabel(x_axis)
        grid.ax_heatmap.set_ylabel(y_label)
        grid.ax_heatmap.tick_params(axis="x", labelsize=7)
        grid.ax_heatmap.tick_params(axis="y", labelsize=8)
        grid.fig.suptitle(
            _condensed_plot_title(title or default_title, condensed),
            y=1.02,
            fontsize=11,
        )
        grid.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(grid.fig)
        print(f"wrote {save_path}")
        return

    fig, ax = plt.subplots(figsize=(max(12, length * 0.15),
                                    max(2.5, hidden_size * 0.35)))
    im = ax.imshow(
        hidden_states.T,
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", origin="lower",
    )

    ax.set_yticks(range(hidden_size))
    ax.set_yticklabels([f"h{i}" for i in range(hidden_size)] if y_label == "hidden unit"
                       else [f"{y_label}{i}" for i in range(hidden_size)])
    ax.set_xticks(range(length))
    ax.set_xticklabels(x_labels, fontsize=7)
    if automaton is not None:
        if prefix_keys is not None:
            state_ids = _dfa_state_ids_for_prefixes(
                prefix_keys, automaton, spaced=spaced,
            )
        else:
            state_ids = _dfa_state_ids_at_timesteps(
                text, automaton, spaced=spaced, words=words,
            )
        _color_tick_labels_by_state_ids(ax.get_xticklabels(), state_ids)
        x_axis += " · tick color = min DFA state"
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_label)
    ax.set_title(
        _condensed_plot_title(title or default_title, condensed)
    )

    cb_label = colorbar_label or (f"activation ({act_label})" if not use_raw else "value")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=cb_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def average_linkage_hierarchy(rows):
    """Average-linkage clustering; returns (linkage, leaf_order).

    linkage has shape (n-1, 4) with columns [left_id, right_id, distance, count]
    in the same convention as scipy.cluster.hierarchy.linkage.
    """
    n_rows = rows.shape[0]
    if n_rows == 0:
        return np.zeros((0, 4)), []
    if n_rows == 1:
        return np.zeros((0, 4)), [0]

    distances = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=2)
    clusters = [
        {"indices": [i], "members": [i], "cluster_id": i, "size": 1}
        for i in range(n_rows)
    ]
    linkage = []
    next_cluster_id = n_rows

    while len(clusters) > 1:
        best_pair = None
        best_distance = np.inf

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                member_distances = distances[np.ix_(
                    clusters[i]["members"],
                    clusters[j]["members"],
                )]
                distance = float(np.mean(member_distances))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        left, right = best_pair
        left_cluster, right_cluster = clusters[left], clusters[right]
        linkage.append([
            left_cluster["cluster_id"],
            right_cluster["cluster_id"],
            best_distance,
            left_cluster["size"] + right_cluster["size"],
        ])
        merged = {
            "indices": left_cluster["indices"] + right_cluster["indices"],
            "members": left_cluster["members"] + right_cluster["members"],
            "cluster_id": next_cluster_id,
            "size": left_cluster["size"] + right_cluster["size"],
        }
        next_cluster_id += 1
        clusters[left] = merged
        del clusters[right]

    return np.array(linkage), clusters[0]["indices"]


def average_linkage_cluster_order(rows):
    """Return row indices ordered by a small average-linkage clustering pass."""
    _, order = average_linkage_hierarchy(rows)
    return order


def display_char(char):
    """Format a character so labels stay readable for whitespace too."""
    if char == "\n":
        return "\\n"
    if char == "\t":
        return "\\t"
    if char == " ":
        return "␣"
    return char


def argmax_region_glyph(char: str) -> str | None:
    """Single visible glyph for an argmax region label (None = skip region)."""
    if char == " ":
        return "␣"
    if len(char) == 1:
        return char
    return None


def corpus_uses_word_spacing(text: str, exp_name: str | None = None) -> bool:
    if exp_name is not None and experiment_uses_word_space(exp_name):
        return True
    return " " in text


_VIS_WORDS: list[str] | None = None
_VIS_LABEL_WORDS: list[str] | None = None


def _resolve_words(text: str, words: list[str] | None = None) -> list[str] | None:
    if words is not None:
        return words
    if _VIS_WORDS:
        return _VIS_WORDS
    return infer_task_words(text)


def _corpus_vocab(text: str, words: list[str] | None = None) -> set[str] | None:
    if _VIS_LABEL_WORDS:
        return set(_VIS_LABEL_WORDS)
    w = _resolve_words(text, words)
    return set(w) if w else None


def _trajectory_vocabulary_words(text: str, words: list[str] | None = None) -> list[str]:
    """Word list used for trajectory rollout length and seed letters."""
    if words:
        return list(words)
    if _VIS_LABEL_WORDS:
        return list(_VIS_LABEL_WORDS)
    resolved = _resolve_words(text)
    if resolved:
        return list(resolved)
    vocab = _corpus_vocab(text)
    return sorted(vocab) if vocab else []


def _longest_vocabulary_word_length(words: list[str]) -> int:
    return max((len(w) for w in words), default=3)


def _trajectory_seed_letters(model: dict, words: list[str]) -> list[str]:
    """Unique vocabulary letters (excluding space), in sorted order."""
    model_chars = set(model["chars"])
    letters = sorted({c for w in words for c in w if c != " " and c in model_chars})
    return letters or [c for c in sorted(model_chars) if c != " "][:1]


def _accumulated_output_labels(generated: list[str], n_states: int) -> list[str]:
    """Cumulative sampled string at each hidden-state index."""
    return ["".join(generated[: i + 1]) for i in range(n_states)]


def _seed_letter_colors(seed_letters: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20", max(len(seed_letters), 1))
    return {letter: cmap(i) for i, letter in enumerate(seed_letters)}


def _hidden_state_after_char(
    model: dict,
    seed_char: str,
    *,
    hidden_size: int,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Hidden state after one forward step from ``h0`` (default zeros) with ``seed_char``."""
    chars = list(model["chars"])
    char_to_index = {c: i for i, c in enumerate(chars)}
    if seed_char not in char_to_index:
        seed_char = chars[0]
    W_xh = np.asarray(model["weights_input_to_hidden"])
    W_hh = np.asarray(model["weights_hidden_to_hidden"])
    b_h_col = np.asarray(model["bias_hidden"])
    use_relu = bool(model.get("use_relu", False))
    h = np.zeros((hidden_size, 1), dtype=float) if h0 is None else np.asarray(h0, dtype=float).reshape(-1, 1)
    x = np.zeros((len(chars), 1), dtype=float)
    x[char_to_index[seed_char], 0] = 1.0
    h, _ = rnn_hidden_step(
        h, x, W_xh, W_hh, b_h_col, use_relu=use_relu,
        timestep_noise_std=float(model.get("timestep_noise_std", 0.0)),
    )
    return h.ravel().copy()


def _letter_seed_no_input_trajectory_pca(
    model: dict,
    *,
    seed_char: str,
    steps: int,
    mean: np.ndarray,
    components: np.ndarray,
    hidden_size: int,
    normalize_activity: bool = False,
) -> np.ndarray:
    """PCA path from zero init: feed ``seed_char``, then ``steps`` no-input steps."""
    W_hh = np.asarray(model["weights_hidden_to_hidden"])
    b_h = np.asarray(model["bias_hidden"]).ravel()
    use_relu = bool(model.get("use_relu", False))
    h = _hidden_state_after_char(model, seed_char, hidden_size=hidden_size)
    noise_std = float(model.get("timestep_noise_std", 0.0))
    noise_rng = np.random.default_rng()
    zs = [
        _project_hidden_to_pca(
            h, mean, components, normalize_activity=normalize_activity,
        )[0],
    ]
    for _ in range(max(0, int(steps) - 1)):
        h = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)
        if noise_std > 0:
            h = inject_timestep_noise(h, noise_std, noise_rng)
        zs.append(
            _project_hidden_to_pca(
                h, mean, components, normalize_activity=normalize_activity,
            )[0],
        )
    return np.asarray(zs, dtype=float)


def _random_hidden_no_input_trajectory_pca(
    model: dict,
    *,
    h0: np.ndarray,
    steps: int,
    mean: np.ndarray,
    components: np.ndarray,
    normalize_activity: bool = False,
) -> np.ndarray:
    """PCA path from arbitrary hidden seed, then ``steps - 1`` no-input recurrent steps."""
    W_hh = np.asarray(model["weights_hidden_to_hidden"])
    b_h = np.asarray(model["bias_hidden"]).ravel()
    use_relu = bool(model.get("use_relu", False))
    h = np.asarray(h0, dtype=float).ravel().copy()
    noise_std = float(model.get("timestep_noise_std", 0.0))
    noise_rng = np.random.default_rng()
    zs = [
        _project_hidden_to_pca(
            h, mean, components, normalize_activity=normalize_activity,
        )[0],
    ]
    for _ in range(max(0, int(steps) - 1)):
        h = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)
        if noise_std > 0:
            h = inject_timestep_noise(h, noise_std, noise_rng)
        zs.append(
            _project_hidden_to_pca(
                h, mean, components, normalize_activity=normalize_activity,
            )[0],
        )
    return np.asarray(zs, dtype=float)


def _random_hidden_seeds(
    hidden_size: int,
    n_seeds: int,
    *,
    rng_seed: int = 0,
    reference_states: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Wide uniform random hidden vectors spanning (and exceeding) observed state magnitudes."""
    rng = np.random.default_rng(rng_seed)
    if reference_states is not None and len(reference_states):
        ref = np.asarray(reference_states, dtype=float)
        bound = float(np.percentile(np.abs(ref), 99.0))
        bound = max(bound * 3.0, 1.0)
    else:
        bound = 5.0
    return [rng.uniform(-bound, bound, hidden_size) for _ in range(n_seeds)]


def _random_seed_colors(n_seeds: int) -> list[tuple]:
    return [
        plt.matplotlib.colors.to_rgba(_DIVERGENT_WORD_PALETTE[i % len(_DIVERGENT_WORD_PALETTE)])
        for i in range(n_seeds)
    ]


def _boundary_prefix_labels(text: str, *, spaced: bool) -> list[str]:
    """In-word prefix at each timestep (characters since last word boundary)."""
    vocab = _corpus_vocab(text)
    return [
        in_word_prefix_at_position(text, i, spaced=spaced, vocab=vocab)
        for i in range(len(text))
    ]


def _label_condense_radius(coords: np.ndarray, frac: float = 0.06) -> float:
    if len(coords) == 0:
        return 0.1
    span = float(np.max(coords.max(axis=0) - coords.min(axis=0)))
    return max(span * frac, 1e-3)


def _clusters_for_nearby_labels(
    coords: np.ndarray,
    labels: list[str],
    eps: float,
) -> list[tuple[str, list[int]]]:
    """Merge repeated labels whose PCA positions fall within ``eps``."""
    clusters: list[tuple[str, list[int]]] = []
    for i, label in enumerate(labels):
        if not label:
            continue
        placed = False
        for cl, idxs in clusters:
            if cl != label:
                continue
            centroid = coords[idxs].mean(axis=0)
            if float(np.linalg.norm(coords[i] - centroid)) <= eps:
                idxs.append(i)
                placed = True
                break
        if not placed:
            clusters.append((label, [i]))
    clusters.sort(key=lambda item: item[1][0])
    return clusters


def _estimate_label_radius(display: str, fontsize: float, span: float) -> float:
    """Approximate label footprint radius in data coordinates."""
    char_w = fontsize * 0.55
    width = max(char_w, len(display) * char_w * 0.68)
    return max(span * 0.038, width * span * 0.0022)


def _point_segment_distance_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _minimal_leader_label_position(
    centroid2: np.ndarray,
    outward: np.ndarray,
    display: str,
    *,
    span: float,
    base_offset_frac: float,
    fontsize: float,
    placed: list[tuple[np.ndarray, float]],
    avoid_points: np.ndarray | None = None,
    path_plane: np.ndarray | None = None,
    line_clearance_frac: float = 0.03,
) -> np.ndarray:
    """Place a label as close as possible to ``centroid2`` without overlap."""
    def _rot2(v: np.ndarray, deg: float) -> np.ndarray:
        t = np.deg2rad(deg)
        c, s = float(np.cos(t)), float(np.sin(t))
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def _clear_of_path(cand2: np.ndarray, radius: float) -> bool:
        if path_plane is None or len(path_plane) < 2:
            return True
        clearance = span * line_clearance_frac + radius * 0.45
        for j in range(len(path_plane) - 1):
            if _point_segment_distance_2d(cand2, path_plane[j], path_plane[j + 1]) < clearance:
                return False
        return True

    base_offset = span * base_offset_frac
    radius = _estimate_label_radius(display, fontsize, span)
    label_pad = radius * 1.02 + span * 0.012
    angle_candidates_deg = (
        0, 12, -12, 24, -24, 36, -36, 48, -48, 60, -60,
        75, -75, 90, -90, 105, -105, 120, -120, 150, -150, 180,
    )
    distance_scales = (1.0, 1.05, 1.1, 1.16, 1.23, 1.31, 1.4, 1.5, 1.62, 1.75, 1.9, 2.1)

    norm = float(np.linalg.norm(outward))
    if norm < 1e-9:
        outward = np.array([0.0, 1.0])
    else:
        outward = outward / norm

    best2: np.ndarray | None = None
    best_dist = float("inf")
    for dist_scale in distance_scales:
        for deg in angle_candidates_deg:
            cand2 = centroid2 + _rot2(outward, deg) * (base_offset * dist_scale)
            if any(
                float(np.linalg.norm(cand2 - pos2)) < (label_pad + r2)
                for pos2, r2 in placed
            ):
                continue
            if avoid_points is not None and len(avoid_points):
                if any(
                    float(np.linalg.norm(cand2 - pt)) < label_pad
                    for pt in avoid_points
                ):
                    continue
            if not _clear_of_path(cand2, radius):
                continue
            dist = float(np.linalg.norm(cand2 - centroid2))
            if dist < best_dist:
                best_dist = dist
                best2 = cand2
        if best2 is not None:
            break

    if best2 is None:
        best2 = centroid2 + outward * base_offset * distance_scales[-1]
    placed.append((best2, radius))
    return best2


def _layout_trajectory_label_positions(
    vertices: np.ndarray,
    keys: list[str],
    displays: list[str],
    *,
    offset_frac: float = 0.04,
    fontsize: float = 9.0,
    path_coords: np.ndarray | None = None,
    line_clearance_frac: float = 0.03,
) -> dict[str, np.ndarray]:
    """Place offset labels on PC1–PC2 with no mutual overlap (2D or 3D paths)."""
    plane = vertices[:, :2]
    center = plane.mean(axis=0)
    span = max(float(np.max(np.ptp(plane, axis=0))), 1e-3)
    path_plane = path_coords[:, :2] if path_coords is not None and len(path_coords) else None

    placed: list[tuple[np.ndarray, float]] = []
    positions: dict[str, np.ndarray] = {}

    order = sorted(range(len(keys)), key=lambda i: float(np.linalg.norm(plane[i] - center)))
    for i in order:
        key = keys[i]
        centroid2 = plane[i]
        outward = centroid2 - center
        best2 = _minimal_leader_label_position(
            centroid2, outward, displays[i],
            span=span,
            base_offset_frac=offset_frac,
            fontsize=fontsize,
            placed=placed,
            avoid_points=plane,
            path_plane=path_plane,
            line_clearance_frac=line_clearance_frac,
        )
        full = vertices[i].copy()
        full[0], full[1] = float(best2[0]), float(best2[1])
        positions[key] = full

    return positions


def _layout_cluster_label_positions(
    vertices: np.ndarray,
    keys: list[str],
    *,
    offset_frac: float = 0.10,
) -> dict[str, np.ndarray]:
    """Push labels outward from the trajectory centroid (2D or 3D)."""
    ndim = vertices.shape[1]
    center = vertices.mean(axis=0)
    span = max(float(np.max(np.ptp(vertices, axis=0))), 1e-3)
    offset_dist = span * offset_frac
    positions: dict[str, np.ndarray] = {}
    for i, key in enumerate(keys):
        centroid = vertices[i]
        outward = centroid - center
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            outward = np.zeros(ndim, dtype=float)
            outward[min(1, ndim - 1)] = 1.0
        else:
            outward = outward / norm
        positions[key] = centroid + outward * offset_dist
    return positions


def _annotate_trajectory_labels(
    ax,
    coords: np.ndarray,
    labels: list[str],
    *,
    colors: list | None = None,
    label_colors: list | None = None,
    text_color: str = "#000000",
    fontsize: float = 9.0,
    fontweight: str = "bold",
    dedupe: bool = True,
    word_keys: list[str] | None = None,
    condense_nearby: bool = False,
    condense_radius_frac: float = 0.06,
    use_leaders: bool = True,
    leader_linewidth: float = 0.3,
    leader_alpha: float = 0.42,
    label_offset_frac: float = 0.04,
    line_clearance_frac: float = 0.03,
) -> None:
    """Text labels at PCA coordinates."""
    n = min(len(coords), len(labels))
    if n == 0:
        return
    fs = max(7, int(fontsize))

    def _cluster_text(bucket_key) -> str:
        return bucket_key[0] if isinstance(bucket_key, tuple) else bucket_key

    def _layout_key(bucket_key, idxs: list[int]) -> str:
        if isinstance(bucket_key, tuple):
            return f"{bucket_key[0]}\0{bucket_key[1]}"
        if dedupe:
            return str(bucket_key)
        return f"{bucket_key}:{idxs[0]}"

    clusters: list[tuple[object, list[int]]]
    if condense_nearby:
        eps = _label_condense_radius(coords[:n], condense_radius_frac)
        clusters = _clusters_for_nearby_labels(coords[:n], labels[:n], eps)
    elif dedupe:
        from collections import defaultdict

        buckets: dict[object, list[int]] = defaultdict(list)
        for i in range(n):
            label = labels[i]
            if label:
                key = (label, word_keys[i]) if word_keys is not None else label
                buckets[key].append(i)
        clusters = sorted(buckets.items(), key=lambda item: item[1][0])
    else:
        clusters = [(labels[i], [i]) for i in range(n) if labels[i]]

    if not clusters:
        return

    keys = [_layout_key(bucket_key, idxs) for bucket_key, idxs in clusters]
    anchor_vertices = np.array([coords[idxs].mean(axis=0) for _, idxs in clusters])
    displays = [
        "␣" if _cluster_text(clusters[i][0]) == " " else _cluster_text(clusters[i][0])
        for i in range(len(clusters))
    ]
    is_3d = anchor_vertices.shape[1] >= 3

    if use_leaders:
        label_positions = _layout_trajectory_label_positions(
            anchor_vertices, keys, displays,
            offset_frac=label_offset_frac,
            fontsize=fs,
            path_coords=coords[:n],
            line_clearance_frac=line_clearance_frac,
        )
    else:
        label_positions = {key: anchor_vertices[i] for i, key in enumerate(keys)}

    leader_color = "0.55"
    for i, key in enumerate(keys):
        bucket_key, idxs = clusters[i]
        label = _cluster_text(bucket_key)
        if not label:
            continue
        display = displays[i]
        if label_colors is not None and idxs:
            color = label_colors[idxs[0]]
        elif colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = text_color
        text_pos = label_positions[key]
        leader_targets = [coords[j] for j in idxs] if dedupe else [anchor_vertices[i]]
        if use_leaders:
            for vertex in leader_targets:
                if is_3d:
                    ax.plot(
                        [text_pos[0], vertex[0]],
                        [text_pos[1], vertex[1]],
                        [text_pos[2], vertex[2]],
                        color=leader_color,
                        linewidth=leader_linewidth,
                        alpha=leader_alpha,
                        solid_capstyle="round",
                        zorder=9,
                    )
                else:
                    ax.plot(
                        [text_pos[0], vertex[0]],
                        [text_pos[1], vertex[1]],
                        color=leader_color,
                        linewidth=leader_linewidth,
                        alpha=leader_alpha,
                        solid_capstyle="round",
                        zorder=9,
                    )
        if is_3d:
            ax.text(
                text_pos[0], text_pos[1], text_pos[2],
                display, fontsize=fs, color=color, fontweight=fontweight,
                ha="center", va="center", zorder=10,
            )
        else:
            ax.text(
                text_pos[0], text_pos[1],
                display, fontsize=fs, color=color, fontweight=fontweight,
                ha="center", va="center", zorder=10,
            )


def _annotate_boundary_prefixes(
    ax,
    coords: np.ndarray,
    prefix_labels: list[str],
    *,
    spaced: bool = False,
    fontsize: float = 7.0,
) -> None:
    """Prefix-since-boundary labels at PCA coordinates (no DFA coloring)."""
    _annotate_trajectory_labels(ax, coords, prefix_labels, fontsize=fontsize)


_CLOSED_LOOP_LOOPS_PER_WORD = 3
_CLOSED_LOOP_AVERAGE_TRIALS = 24
_INTERNAL_RANDOM_HIDDEN_SEED_COUNT = 512
_MEAN_TRAJECTORY_COLOR = "#1a1a1a"


def _one_vocab_cycle_steps(words: list[str], *, spaced: bool) -> int:
    if not words:
        return 16
    max_len = max(len(w) for w in words)
    if spaced:
        return len(words) * (max_len + 1) + 1
    return len(words) * max_len + 1


def _default_closed_loop_rollout_steps(words: list[str], *, spaced: bool) -> int:
    """Autoregressive steps for closed-loop: several full passes over the vocabulary."""
    return _CLOSED_LOOP_LOOPS_PER_WORD * _one_vocab_cycle_steps(words, spaced=spaced)


def _closed_loop_rollout_steps(
    words: list[str],
    closed_loop_steps: int | None,
    *,
    spaced: bool,
) -> int:
    if closed_loop_steps is not None:
        return max(1, int(closed_loop_steps))
    return _default_closed_loop_rollout_steps(words, spaced=spaced)


def _unspaced_word_at_each_index(
    text: str,
    words: list[str],
    vocab: set[str],
) -> list[str]:
    out = [""] * len(text)
    pos = 0
    while pos < len(text):
        end_word: int | None = None
        word: str | None = None
        for end in range(pos, len(text)):
            sub = text[pos : end + 1]
            if sub in vocab:
                end_word = end
                word = sub
                break
            if not any(w.startswith(sub) for w in words):
                break
        if word is None or end_word is None:
            word = words[0]
            end_word = pos
        for i in range(pos, end_word + 1):
            out[i] = word
        pos = end_word + 1
    return out


def _rollout_word_at_positions(
    generated: list[str],
    seed_len: int,
    n_states: int,
    *,
    spaced: bool,
    words: list[str],
) -> list[str]:
    """Greedy vocabulary word at each rollout timestep (not raw prefix string)."""
    vocab = set(words)
    full_text = "".join(generated[: seed_len + n_states])
    if not spaced:
        char_words = _unspaced_word_at_each_index(full_text, words, vocab)
        return [
            char_words[seed_len + i] if seed_len + i < len(char_words) else words[0]
            for i in range(n_states)
        ]

    labels: list[str] = []
    for i in range(n_states):
        text = "".join(generated[: seed_len + i + 1])
        pos = len(text) - 1
        prefix = in_word_prefix_at_position(text, pos, spaced=spaced, vocab=vocab)
        if prefix == " ":
            labels.append("␣")
            continue
        if prefix in vocab:
            labels.append(prefix)
            continue
        candidates = [w for w in words if w.startswith(prefix)]
        if len(candidates) == 1:
            labels.append(candidates[0])
            continue
        if len(candidates) > 1:
            resolved = None
            for start, end, seg in corpus_segments(text, words, spaced=spaced):
                if start <= pos <= end:
                    w = segment_word_label(seg)
                    if w in vocab:
                        resolved = w
                        break
            labels.append(resolved if resolved is not None else candidates[0])
            continue
        labels.append(words[0])
    return labels


def _rollout_prefix_labels(
    generated: list[str],
    seed_len: int,
    n_states: int,
    *,
    spaced: bool,
    vocab: set[str] | None,
) -> list[str]:
    labels: list[str] = []
    for i in range(n_states):
        text = "".join(generated[: seed_len + i + 1])
        if spaced:
            if " " in text:
                prefix = text[text.rfind(" ") + 1 :]
            else:
                prefix = text
            labels.append(prefix if prefix else "␣")
        else:
            idx = len(text) - 1
            labels.append(in_word_prefix_at_position(text, idx, spaced=False, vocab=vocab))
    return labels


def _completed_word_end_indices(word_at_step: list[str], vocab: set[str]) -> list[int]:
    ends: list[int] = []
    for i, w in enumerate(word_at_step):
        if w not in vocab:
            continue
        if i + 1 >= len(word_at_step) or word_at_step[i + 1] != w:
            ends.append(i)
    return ends


def _sparse_word_end_labels(word_at_step: list[str], vocab: set[str], n: int) -> list[str]:
    labels = [""] * n
    for i in _completed_word_end_indices(word_at_step, vocab):
        if i < n:
            labels[i] = word_at_step[i]
    return labels


def _sparse_unique_word_end_labels(
    word_at_step: list[str], vocab: set[str], n: int,
) -> list[str]:
    """Word-boundary labels, keeping only the first occurrence of each vocabulary word."""
    labels = [""] * n
    seen: set[str] = set()
    for i in _completed_word_end_indices(word_at_step, vocab):
        if i >= n:
            continue
        w = word_at_step[i]
        if w in seen:
            continue
        seen.add(w)
        labels[i] = w
    return labels


def _closed_loop_summary_seed(vocab_words: list[str], seed_letters: list[str], *, spaced: bool) -> str:
    if spaced:
        return " "
    if vocab_words:
        from collections import Counter

        word_starts = sorted({w[0] for w in vocab_words if w})
        vowels = set("aeiou")
        consonant_starts = [c for c in word_starts if c not in vowels]
        if consonant_starts:
            start_counts = Counter(w[0] for w in vocab_words if w)
            return max(consonant_starts, key=lambda c: (start_counts[c], -ord(c)))
        return word_starts[0]
    return seed_letters[0] if seed_letters else "b"


def _majority_labels_per_step(label_lists: list[list[str]]) -> list[str]:
    from collections import Counter

    if not label_lists:
        return []
    n = len(label_lists[0])
    out: list[str] = []
    for t in range(n):
        votes = Counter(row[t] for row in label_lists if t < len(row))
        out.append(votes.most_common(1)[0][0] if votes else "")
    return out


def _same_length_average_trajectory(paths: list[np.ndarray]) -> np.ndarray | None:
    if not paths:
        return None
    n = min(len(p) for p in paths)
    if n < 2:
        return None
    return np.mean(np.stack([p[:n] for p in paths], axis=0), axis=0)


def _position_aligned_average_trajectory(paths: list[np.ndarray]) -> np.ndarray | None:
    """Mean PCA coordinate at each in-word index (variable-length paths)."""
    if not paths:
        return None
    max_len = max(len(p) for p in paths)
    if max_len < 2:
        return None
    dims = paths[0].shape[1]
    accum = np.zeros((max_len, dims), dtype=float)
    counts = np.zeros(max_len, dtype=float)
    for path in paths:
        for i in range(len(path)):
            accum[i] += path[i]
            counts[i] += 1.0
    counts = np.maximum(counts, 1.0)
    return accum / counts[:, None]


def _plot_mean_trajectory_overlay(
    ax,
    path: np.ndarray | None,
    *,
    is_3d: bool,
    linewidth: float = 1.1,
    alpha: float = 0.9,
    zorder: int = 8,
) -> None:
    """Thin plain line for trial-averaged trajectories (no per-step arrows)."""
    if path is None or len(path) < 2:
        return
    if is_3d:
        ax.plot(
            path[:, 0], path[:, 1], path[:, 2],
            color=_MEAN_TRAJECTORY_COLOR, linewidth=linewidth, alpha=alpha, zorder=zorder,
        )
    else:
        ax.plot(
            path[:, 0], path[:, 1],
            color=_MEAN_TRAJECTORY_COLOR, linewidth=linewidth, alpha=alpha,
            solid_capstyle="round", zorder=zorder,
        )


def _word_start_segment_flags(
    prefix_labels: list[str],
    word_at_step: list[str],
) -> list[bool]:
    flags: list[bool] = []
    for i in range(len(prefix_labels) - 1):
        nxt = prefix_labels[i + 1]
        prev = prefix_labels[i]
        if nxt in ("", "␣") or len(nxt) != 1:
            flags.append(False)
            continue
        if prev in ("", "␣"):
            flags.append(True)
            continue
        flags.append(word_at_step[i] != word_at_step[i + 1])
    return flags


def _trim_first_word_from_labeled_path(
    coords: np.ndarray,
    prefix_labels: list[str],
    word_at_step: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop the seed-primed first word (incomplete vs regular word trajectories)."""
    ranges = _word_segment_ranges(word_at_step)
    if len(ranges) < 2:
        return coords, prefix_labels, word_at_step
    start = ranges[1][0]
    return coords[start:], prefix_labels[start:], word_at_step[start:]


def _trim_first_word_from_hidden(
    hidden: np.ndarray,
    word_at_step: list[str],
) -> np.ndarray:
    """Drop hidden states belonging to the first word segment."""
    ranges = _word_segment_ranges(word_at_step)
    if len(ranges) < 2:
        return hidden
    start = ranges[1][0]
    return np.asarray(hidden[start:], dtype=float)


def _states_after_first_word(
    text: str,
    hidden_states: np.ndarray,
    *,
    spaced: bool,
    words: list[str] | None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Corpus hidden states / word trajectories excluding the first word segment."""
    segments = corpus_segments(text, list(_corpus_vocab(text, words) or []), spaced=spaced)
    if len(segments) < 2:
        trajs = _embed_trajectories_for_text(
            text, hidden_states, spaced=spaced, words=words,
        )
        return np.asarray(hidden_states, dtype=float), trajs
    start = int(segments[1][0])
    hs = np.asarray(hidden_states[start:], dtype=float)
    remapped = [
        (int(s) - start, int(e) - start, seg)
        for s, e, seg in segments[1:]
        if int(e) >= start
    ]
    from viz.dimred import trajectories_for_embed

    trajs = trajectories_for_embed(hs, segments=remapped) if remapped else []
    return hs, trajs


def _closed_loop_rollout_pca(
    model: dict,
    *,
    seed_text: str,
    steps: int,
    rng: np.random.Generator,
    mean: np.ndarray,
    components: np.ndarray,
    vocab_words: list[str],
    spaced: bool,
    normalize_activity: bool = False,
    skip_first_word: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """One stochastic closed-loop rollout projected into PCA space."""
    hidden, generated = rnn_closed_loop_rollout(
        model, seed_text=seed_text, steps=steps, rng=rng,
    )
    gen_z = _project_hidden_to_pca(
        hidden, mean, components, normalize_activity=normalize_activity,
    )
    seed_len = len(seed_text)
    n_states = len(gen_z)
    vocab = set(vocab_words)
    prefix_labels = _rollout_prefix_labels(
        generated, seed_len, n_states, spaced=spaced, vocab=vocab,
    )
    word_at_step = _rollout_word_at_positions(
        generated, seed_len, n_states, spaced=spaced, words=vocab_words,
    )
    if skip_first_word:
        gen_z, prefix_labels, word_at_step = _trim_first_word_from_labeled_path(
            gen_z, prefix_labels, word_at_step,
        )
    return gen_z, prefix_labels, word_at_step


def _segmented_rollout_styles(
    prefix_labels: list[str],
    word_at_step: list[str],
    word_colors: dict[str, tuple],
    n_segments: int,
    *,
    color_mode: str = "word",
    max_word_len: int = 4,
) -> tuple[list, list[str]]:
    word_start = _word_start_segment_flags(prefix_labels, word_at_step)
    gray = word_colors["␣"]
    segment_colors: list = []
    for i in range(n_segments):
        if _is_return_to_baseline_segment(
            prefix_labels[i], prefix_labels[i + 1], word_start=word_start[i],
        ):
            segment_colors.append(gray)
        elif color_mode == "step":
            plen = len(prefix_labels[i + 1]) if prefix_labels[i + 1] not in ("", "␣") else 1
            segment_colors.append(_step_palette_rgba(min(plen, max_word_len)))
        else:
            segment_colors.append(word_colors.get(word_at_step[i + 1], word_colors["?"]))
    segment_linestyles = [":" if is_start else "-" for is_start in word_start]
    return segment_colors, segment_linestyles


def _plot_return_to_start_segment(
    ax,
    path: np.ndarray,
    *,
    is_3d: bool = False,
    linewidth: float = 1.2,
    alpha: float = 0.75,
    zorder: int = 2,
) -> None:
    """Gray dashed arrow from trajectory end back to its start."""
    if len(path) < 2:
        return
    gray = _TRAJECTORY_RETURN_COLOR
    p0, p1 = np.asarray(path[-1]), np.asarray(path[0])
    if is_3d:
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=gray, linestyle=":", linewidth=linewidth, alpha=alpha,
        )
        _midsegment_arrowhead_3d(ax, p0, p1, color=gray, alpha=alpha)
    else:
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=gray, linestyle=":", linewidth=linewidth, alpha=alpha, zorder=zorder,
        )
        _midsegment_arrowhead_2d(ax, p0, p1, color=gray, alpha=alpha, zorder=zorder + 1)


def _plot_segmented_vocab_rollout(
    ax,
    gen_z: np.ndarray,
    prefix_labels: list[str],
    word_at_step: list[str],
    *,
    word_colors: dict[str, tuple],
    vocab_words: list[str],
    annotate: bool = False,
    annotate_fontsize: float = 9.0,
    is_3d: bool = False,
    linewidth: float = 1.5,
    alpha: float = 0.78,
    unique_word_labels: bool = False,
    color_mode: str = "word",
    max_word_len: int = 4,
) -> None:
    if len(gen_z) < 2:
        return
    vocab = set(vocab_words)
    segment_colors, segment_linestyles = _segmented_rollout_styles(
        prefix_labels, word_at_step, word_colors, len(gen_z) - 1,
        color_mode=color_mode, max_word_len=max_word_len,
    )
    plot_path = _plot_step_colored_path_arrows_3d if is_3d else _plot_step_colored_path_arrows
    if is_3d:
        plot_path(
            ax, gen_z,
            linewidth=linewidth, alpha=alpha,
            segment_colors=segment_colors,
            segment_linestyles=segment_linestyles,
            arrow_mutation_scale=12.0,
        )
    else:
        plot_path(
            ax, gen_z,
            linewidth=linewidth, alpha=alpha, zorder=2,
            segment_colors=segment_colors,
            segment_linestyles=segment_linestyles,
            arrow_mutation_scale=12.0,
        )
    if annotate:
        label_fn = _sparse_unique_word_end_labels if unique_word_labels else _sparse_word_end_labels
        end_labels = label_fn(word_at_step, vocab, len(gen_z))
        _annotate_trajectory_labels(
            ax, gen_z, end_labels,
            fontsize=annotate_fontsize,
            dedupe=True,
            word_keys=word_at_step,
            label_colors=[
                word_colors.get(word_at_step[i], word_colors["?"])
                for i in range(len(end_labels))
            ],
            use_leaders=True,
            leader_linewidth=0.4,
        )


def _plot_segmented_closed_loop_rollout(
    ax,
    gen_z: np.ndarray,
    prefix_labels: list[str],
    word_at_step: list[str],
    *,
    word_colors: dict[str, tuple],
    vocab_words: list[str],
    annotate: bool,
    annotate_fontsize: float,
    is_3d: bool,
    linewidth: float = 1.5,
    alpha: float = 0.78,
    unique_word_labels: bool = False,
) -> None:
    _plot_segmented_vocab_rollout(
        ax, gen_z, prefix_labels, word_at_step,
        word_colors=word_colors,
        vocab_words=vocab_words,
        annotate=annotate,
        annotate_fontsize=annotate_fontsize,
        is_3d=is_3d,
        linewidth=linewidth,
        alpha=alpha,
        unique_word_labels=unique_word_labels,
        color_mode="word",
    )


def rnn_closed_loop_rollout(
    model: dict,
    *,
    seed_text: str,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Autoregressive RNN rollout; returns hidden states (T, hidden_size) and char list."""
    chars = list(model["chars"])
    char_to_index = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    hidden_size = int(model["hidden_size"])

    W_xh = np.asarray(model["weights_input_to_hidden"])
    W_hh = np.asarray(model["weights_hidden_to_hidden"])
    W_ho = np.asarray(model["weights_hidden_to_output"])
    b_h_col = np.asarray(model["bias_hidden"])
    b_o = np.asarray(model["bias_output"]).ravel()
    use_relu = bool(model.get("use_relu", False))
    noise_std = float(model.get("timestep_noise_std", 0.0))

    h = np.zeros((hidden_size, 1), dtype=float)
    generated = list(seed_text) if seed_text else []
    if not generated:
        generated = [chars[0]]

    for ch in seed_text:
        if ch not in char_to_index:
            continue
        x = np.zeros((vocab_size, 1), dtype=float)
        x[char_to_index[ch], 0] = 1.0
        h, _ = rnn_hidden_step(
            h, x, W_xh, W_hh, b_h_col, use_relu=use_relu,
            timestep_noise_std=noise_std, noise_rng=rng,
        )

    hidden_rows: list[np.ndarray] = []
    for _ in range(max(1, int(steps))):
        hidden_rows.append(h.ravel().copy())
        logits = W_ho @ h.ravel() + b_o
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        next_ix = int(rng.choice(vocab_size, p=probs))
        generated.append(chars[next_ix])
        x = np.zeros((vocab_size, 1), dtype=float)
        x[next_ix, 0] = 1.0
        h, _ = rnn_hidden_step(
            h, x, W_xh, W_hh, b_h_col, use_relu=use_relu,
            timestep_noise_std=noise_std, noise_rng=rng,
        )

    return np.asarray(hidden_rows, dtype=np.float64), generated


def prefix_axis_label(
    *, spaced: bool, text: str = "", words: list[str] | None = None,
) -> str:
    if spaced:
        return "prefix since last space"
    if _corpus_vocab(text, words):
        return "in-word prefix"
    return "prefix (≤3 chars)"


def word_subsequent_label(
    text: str,
    index: int,
    *,
    spaced: bool = False,
    words: list[str] | None = None,
) -> str:
    """In-word prefix label at this timestep (space, vocab boundary, or ≤3 chars)."""
    return in_word_prefix_at_position(
        text, index, spaced=spaced, vocab=_corpus_vocab(text, words),
    )


@dataclass
class CondensedView:
    """Hidden states averaged over equivalent in-word prefixes (trie positions)."""

    hidden_states: np.ndarray
    labels: list[str]
    input_chars: list[str]
    timestep_indices: list[int]
    spaced: bool
    words: list[str] | None = None
    output_probs: np.ndarray | None = None
    counts: list[int] = field(default_factory=list)
    next_chars: list[str] = field(default_factory=list)
    label_to_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}


def _prefix_condense_order(
    prefixes: set[str],
    words: list[str] | None,
    *,
    spaced: bool,
) -> list[str]:
    """Order condensed prefixes: space (if spaced), then trie BFS, then leftovers."""
    remaining = set(prefixes)
    ordered: list[str] = []
    if spaced and " " in remaining:
        ordered.append(" ")
        remaining.discard(" ")
    if words:
        for prefix in trie_prefix_display_order(words):
            if prefix in remaining:
                ordered.append(prefix)
                remaining.discard(prefix)
    ordered.extend(sorted(remaining))
    return ordered


def condense_hidden_states_by_prefix(
    text: str,
    hidden_states: np.ndarray,
    output_probs: np.ndarray | None = None,
    *,
    spaced: bool = False,
    words: list[str] | None = None,
) -> CondensedView:
    """
    Average hidden states (and output probs) over timesteps sharing the same
    in-word prefix. Rows follow trie BFS order when a word vocabulary is known.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for t in range(len(text)):
        label = word_subsequent_label(text, t, spaced=spaced, words=words)
        groups[label].append(t)

    order = _prefix_condense_order(set(groups), words, spaced=spaced)
    labels: list[str] = []
    hs_rows: list[np.ndarray] = []
    prob_rows: list[np.ndarray] = []
    input_chars: list[str] = []
    repr_indices: list[int] = []
    counts: list[int] = []
    next_chars: list[str] = []
    n_text = len(text)

    for label in order:
        idxs = groups[label]
        labels.append(label)
        hs_rows.append(hidden_states[idxs].mean(axis=0))
        if output_probs is not None:
            prob_rows.append(output_probs[idxs].mean(axis=0))
        if label == " ":
            input_chars.append(" ")
        elif label:
            input_chars.append(label[-1])
        else:
            input_chars.append(text[idxs[0]])
        repr_indices.append(idxs[0])
        counts.append(len(idxs))
        targets = [text[(t + 1) % n_text] for t in idxs]
        next_chars.append(max(set(targets), key=targets.count))

    return CondensedView(
        hidden_states=np.vstack(hs_rows) if hs_rows else hidden_states[:0],
        labels=labels,
        input_chars=input_chars,
        timestep_indices=repr_indices,
        spaced=spaced,
        words=words,
        output_probs=np.vstack(prob_rows) if prob_rows else None,
        counts=counts,
        next_chars=next_chars,
    )


def _condensed_save_path(save_path: str) -> str:
    base, ext = os.path.splitext(save_path)
    if base.endswith("_condensed"):
        return save_path
    return f"{base}_condensed{ext}"


def _display_prefix_label(label: str) -> str:
    return "␣" if label == " " else label


def _condensed_plot_title(base: str, condensed: CondensedView | None) -> str:
    if condensed is None:
        return base
    n_inst = sum(condensed.counts)
    return (
        f"{base} (condensed: {len(condensed.labels)} prefixes, "
        f"avg over {n_inst} timesteps)"
    )


def corpus_segments(
    text: str,
    words: list[str] | None,
    *,
    spaced: bool,
) -> list[tuple[int, int, str]]:
    """Word segments: explicit spaces, or implicit vocabulary boundaries."""
    if spaced and " " in text:
        return space_to_space_segments(text)
    vocab = _corpus_vocab(text, words)
    if vocab:
        return [
            (start, end, text[start : end + 1])
            for start, end, _ in segment_corpus_by_words(text, vocab)
        ]
    if text:
        return [(0, len(text) - 1, text)]
    return []


def space_to_space_segments(text: str) -> list[tuple[int, int, str]]:
    """
    Inclusive timestep ranges from one space to the next (or document boundaries).

    Each segment includes both endpoint spaces when present.
    """
    n = len(text)
    if n == 0:
        return []

    space_ix = [i for i, c in enumerate(text) if c == " "]
    if not space_ix:
        return [(0, n - 1, text)]

    segments: list[tuple[int, int, str]] = []
    if space_ix[0] > 0:
        segments.append((0, space_ix[0], text[: space_ix[0] + 1]))
    for start, end in zip(space_ix, space_ix[1:]):
        segments.append((start, end, text[start : end + 1]))
    if space_ix[-1] < n - 1:
        segments.append((space_ix[-1], n - 1, text[space_ix[-1] :]))
    return segments


def segment_word_label(segment_text: str) -> str:
    """Readable label for a space-to-space path (stripped word, or ␣ for spaces only)."""
    stripped = segment_text.strip()
    return stripped if stripped else "␣"


def context_label(text, index, *, spaced: bool = False, words: list[str] | None = None):
    prefix = word_subsequent_label(text, index, spaced=spaced, words=words)
    if spaced and prefix == " ":
        return " "
    if prefix:
        return prefix
    previous = "^" if index == 0 else display_char(text[index - 1])
    current = display_char(text[index])
    return f"{previous}{current}@{index}"


def timestep_context_label(
    text, index, *, spaced: bool = False, words: list[str] | None = None,
):
    """Context string for plot labels (in-word prefix at each timestep)."""
    return word_subsequent_label(text, index, spaced=spaced, words=words)


def timestep_axis_description(
    text: str, exp_name: str | None = None, words: list[str] | None = None,
) -> str:
    if corpus_uses_word_spacing(text, exp_name):
        return "timestep (prefix after space, or ' ')"
    if _corpus_vocab(text, words):
        return "timestep (in-word prefix)"
    return "timestep (up to 3 chars)"


def infer_task_words(text: str) -> list[str] | None:
    """Best-matching word vocabulary from task.py regimes for this corpus."""
    text_chars = set(text)
    allow_space = " " in text_chars
    best_words = None
    best_char_count = None
    for words in REGIMES.values():
        regime_chars = set("".join(words))
        if allow_space:
            regime_chars.add(" ")
        if text_chars <= regime_chars:
            n = len(regime_chars)
            if best_char_count is None or n < best_char_count:
                best_words = words
                best_char_count = n
    return best_words


def original_vocabulary_title(chars, text: str | None = None) -> str:
    """Suptitle text: task word vocabulary (inferred) and model character vocab."""
    parts = []
    if text:
        words = infer_task_words(text)
        if words:
            parts.append(f"vocabulary: {', '.join(words)}")
    parts.append(f"chars: {''.join(chars)}")
    return " · ".join(parts)


def plot_hidden_states_clustermap(
    text,
    hidden_states,
    chars,
    save_path,
    *,
    exp_name: str | None = None,
    condensed: CondensedView | None = None,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
    repr_label: str = "hidden state",
    dim_label: str = "hidden unit",
):
    """Heatmap (dims × timesteps) with seaborn clustermap layout."""
    prefix_keys: list[str] | None = None
    if condensed is not None:
        hidden_states = condensed.hidden_states
        spaced = condensed.spaced
        words = condensed.words
        prefix_keys = condensed.labels
        row_labels = [_display_prefix_label(l) for l in prefix_keys]
    else:
        spaced = corpus_uses_word_spacing(text, exp_name) or spaced
        words = vocabulary_for_experiment(exp_name) if exp_name else infer_task_words(text)
        row_labels = [
            timestep_context_label(text, t, spaced=spaced, words=words) for t in range(len(text))
        ]
    n_rows, n_cols = hidden_states.shape
    if n_rows == 0:
        return
    col_labels = [f"{dim_label}{i}" for i in range(n_cols)]
    # Flip orientation: units on rows, timesteps on columns (makes long sequences readable).
    data = pd.DataFrame(hidden_states, index=row_labels, columns=col_labels).T

    grid = sns.clustermap(
        data,
        method="average",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        figsize=(max(10, n_rows * 0.24), max(6, n_cols * 0.55)),
        dendrogram_ratio=(0.12, 0.1),
        cbar=False,
        cbar_pos=None,
        xticklabels=True,
        yticklabels=True,
    )
    xlabel = timestep_axis_description(text, exp_name, words=words)
    if automaton is not None:
        if prefix_keys is not None:
            state_ids = _dfa_state_ids_for_prefixes(
                prefix_keys, automaton, spaced=spaced,
            )
        else:
            state_ids = _dfa_state_ids_at_timesteps(
                text, automaton, spaced=spaced, words=words,
            )
        col_order = grid.dendrogram_col.reordered_ind
        _color_tick_labels_by_state_ids(
            grid.ax_heatmap.get_xticklabels(), state_ids, order=col_order,
        )
        xlabel += " · tick color = min DFA state"
    grid.ax_heatmap.set_xlabel(xlabel)
    grid.ax_heatmap.set_ylabel(dim_label)
    grid.ax_heatmap.tick_params(axis="y", labelsize=8)
    grid.ax_heatmap.tick_params(axis="x", labelsize=7)
    grid.fig.suptitle(
        _condensed_plot_title(
            f"{repr_label} clustered (dims × timesteps) · {original_vocabulary_title(chars, text)}",
            condensed,
        ),
        y=1.02, fontsize=11,
    )
    grid.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"wrote {save_path}")


def prefix_tick_label(
    text: str, index: int, *, spaced: bool, words: list[str] | None = None,
) -> str:
    """Axis tick text: in-word prefix (␣ on spaces when spaced)."""
    label = prefix_annotation_label(text, index, spaced=spaced, words=words)
    return "␣" if label == " " else label


def plot_hidden_states_correlation_clustermap(
    text: str,
    hidden_states: np.ndarray,
    chars,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    words: list[str] | None = None,
    condensed: CondensedView | None = None,
    repr_label: str = "hidden state",
):
    """One clustered matrix: Pearson r between vectors at each timestep."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        spaced = condensed.spaced
        words = condensed.words
        labels = [_display_prefix_label(l) for l in condensed.labels]
        if automaton is not None:
            state_ids = [
                dfa_state_for_prefix(l, automaton, spaced=spaced) for l in condensed.labels
            ]
        else:
            state_ids = None
    else:
        labels = [
            prefix_tick_label(text, t, spaced=spaced, words=words) for t in range(len(text))
        ]
        vocab = _corpus_vocab(text, words)
        state_ids = None
        if automaton is not None:
            state_ids = [
                dfa_state_at_position(
                    text, t, automaton, spaced=spaced, vocab=vocab,
                ) for t in range(len(text))
            ]
    n = hidden_states.shape[0]
    if n < 2:
        return

    corr = np.corrcoef(hidden_states)
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)
    data = pd.DataFrame(corr, index=labels, columns=labels)

    panel = max(10.0, n * 0.2)
    grid = sns.clustermap(
        data,
        method="average",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        figsize=(panel, panel),
        dendrogram_ratio=(0.12, 0.12),
        cbar=False,
        cbar_pos=None,
        xticklabels=True,
        yticklabels=True,
    )

    if state_ids is not None:
        row_order = grid.dendrogram_row.reordered_ind
        col_order = grid.dendrogram_col.reordered_ind
        _color_tick_labels_by_state_ids(
            grid.ax_heatmap.get_yticklabels(), state_ids, order=row_order,
        )
        _color_tick_labels_by_state_ids(
            grid.ax_heatmap.get_xticklabels(), state_ids, order=col_order,
        )

    axis_label = prefix_axis_label(spaced=spaced, text=text, words=words)
    grid.ax_heatmap.set_xlabel(axis_label)
    grid.ax_heatmap.set_ylabel(axis_label)
    grid.ax_heatmap.tick_params(axis="both", labelsize=7)
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=90, ha="center")

    title = f"{repr_label} correlation"
    if automaton is not None:
        title += " · tick color = min DFA state"
    grid.fig.suptitle(
        _condensed_plot_title(
            f"{title} · {original_vocabulary_title(chars, text)}",
            condensed,
        ),
        y=1.02,
        fontsize=11,
    )
    grid.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"wrote {save_path}")


def _invalid_word_fraction(
    sampled_text: str,
    vocab: set[str],
    *,
    spaced: bool = True,
) -> float:
    from vocab_diagrams import invalid_word_fraction

    return invalid_word_fraction(sampled_text, vocab, spaced=spaced, trim_edges=True)


def plot_dfa_grouped_state_correlation(
    text: str,
    hidden_states: np.ndarray,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton,
    condensed: CondensedView | None = None,
    repr_label: str = "hidden state",
) -> None:
    """Pearson r between vectors; rows/cols grouped by min DFA state (all blocks)."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        spaced = condensed.spaced
        n = hidden_states.shape[0]
        state_ids = [
            dfa_state_for_prefix(l, automaton, spaced=spaced) for l in condensed.labels
        ]
        label_at = {
            i: _display_prefix_label(condensed.labels[i]) for i in range(n)
        }
    else:
        n = hidden_states.shape[0]
        state_ids = [
            dfa_state_at_position(
                text, t, automaton, spaced=spaced, vocab=_corpus_vocab(text),
            ) for t in range(n)
        ]
        label_at = {
            t: prefix_tick_label(text, t, spaced=spaced, words=_resolve_words(text))
            for t in range(n)
        }
    if n < 2:
        return

    by_state: dict[int, list[int]] = {}
    for t, sid in enumerate(state_ids):
        by_state.setdefault(sid, []).append(t)

    order: list[int] = []
    boundaries: list[int] = [0]
    block_labels: list[str] = []
    for sid in sorted(by_state.keys()):
        idxs = sorted(by_state[sid], key=lambda t: label_at[t])
        order.extend(idxs)
        boundaries.append(len(order))
        block_labels.append(dfa_state_label(sid, automaton))

    if len(order) < 2:
        print(f"skip {save_path}: need ≥2 timesteps")
        return

    corr = np.corrcoef(hidden_states[order])
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)

    state_colors = _state_id_colors(state_ids)

    panel = max(9.0, len(order) * 0.14)
    fig, ax = plt.subplots(figsize=(panel, panel * 0.92), constrained_layout=True)
    im = ax.imshow(
        corr,
        aspect="equal",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        origin="lower",
    )
    block_sids = sorted(by_state.keys())
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="black", lw=0.8)
        ax.axvline(b - 0.5, color="black", lw=0.8)

    tick_pos: list[float] = []
    tick_labels: list[str] = []
    for sid, lo, hi, lab in zip(block_sids, boundaries[:-1], boundaries[1:], block_labels):
        tick_pos.append((lo + hi - 1) / 2.0)
        tick_labels.append(f"q{sid}: {lab}")
    tick_fs = max(5, min(8, 120 // max(len(tick_labels), 1)))

    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs, rotation=55, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=tick_fs)
    for tick, sid in zip(ax.get_xticklabels(), block_sids):
        tick.set_color(state_colors[sid])
    for tick, sid in zip(ax.get_yticklabels(), block_sids):
        tick.set_color(state_colors[sid])

    ax.set_xlabel("min DFA state (accepted prefixes)")
    ax.set_ylabel("min DFA state (accepted prefixes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Pearson r")
    fig.suptitle(
        _condensed_plot_title(
            f"{repr_label} correlation grouped by min DFA state "
            "(diagonal = within state, off-diagonal = vs other states)",
            condensed,
        ),
        fontsize=10,
        y=1.02,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def median_mad(vals: np.ndarray) -> tuple[float, float]:
    """Median and median absolute deviation (empty → nan)."""
    if len(vals) == 0:
        return float("nan"), float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, mad


def pairwise_hidden_state_distance_groups(
    chars: list[str],
    hidden_states: np.ndarray,
    state_ids: list[int],
    position_ids: list[int | None],
    position_from_end_ids: list[int | None],
) -> dict[str, np.ndarray]:
    """L2 distances (i < j) for within/between char, position, position-from-end, DFA, all pairs."""
    n = hidden_states.shape[0]
    groups: dict[str, list[float]] = {
        "Within DFA state": [],
        "Between DFA states": [],
        "Within word position": [],
        "Between word positions": [],
        "Within position from end": [],
        "Between positions from end": [],
        "Within char": [],
        "Between chars": [],
        "All pairs": [],
    }
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(hidden_states[i] - hidden_states[j]))
            groups["All pairs"].append(dist)
            if state_ids[i] == state_ids[j]:
                groups["Within DFA state"].append(dist)
            else:
                groups["Between DFA states"].append(dist)
            if chars[i] == chars[j]:
                groups["Within char"].append(dist)
            else:
                groups["Between chars"].append(dist)
            pi, pj = position_ids[i], position_ids[j]
            if pi is not None and pj is not None:
                if pi == pj:
                    groups["Within word position"].append(dist)
                else:
                    groups["Between word positions"].append(dist)
            pei, pej = position_from_end_ids[i], position_from_end_ids[j]
            if pei is not None and pej is not None:
                if pei == pej:
                    groups["Within position from end"].append(dist)
                else:
                    groups["Between positions from end"].append(dist)
    return {k: np.asarray(v) for k, v in groups.items()}


PAIR_DISTANCE_CATEGORY_ORDER = (
    "Within DFA state",
    "Between DFA states",
    "Within word position",
    "Between word positions",
    "Within position from end",
    "Between positions from end",
    "Within char",
    "Between chars",
    "All pairs",
)

PAIR_DISTANCE_PALETTE = {
    "Within DFA state": "#4c72b0",
    "Between DFA states": "#dd8452",
    "Within word position": "#8172b3",
    "Between word positions": "#c5b0d5",
    "Within position from end": "#9372b3",
    "Between positions from end": "#b39ddb",
    "Within char": "#55a868",
    "Between chars": "#2ca02c",
    "All pairs": "#8c8c8c",
}

SEPARATION_FEATURES = (
    "dfa", "prefix", "string", "char", "position", "position_from_end",
)


@dataclass
class FeatureSeparationStats:
    features: tuple[str, ...]
    centroid_gap: dict[str, float]
    within_spread: dict[str, float]
    between_spread: dict[str, float]
    silhouette: dict[str, float]
    eta2: dict[str, float]
    pairwise_within_median: dict[str, float]
    pairwise_between_median: dict[str, float]
    shuffle_z: dict[str, float]
    shuffle_p: dict[str, float]
    n_groups: dict[str, int]
    n_points: dict[str, int]


def _label_groups(
    labels: list[Any],
    valid_mask: list[bool] | None = None,
) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if valid_mask is not None and not valid_mask[i]:
            continue
        groups[lbl].append(i)
    return {lbl: idxs for lbl, idxs in groups.items() if idxs}


def _centroid_separation_metrics(
    states: np.ndarray,
    group_map: dict[Any, list[int]],
) -> tuple[float, float, float]:
    """Group-balanced centroid gap: mean between-centroid dist minus mean within spread."""
    if len(group_map) < 2:
        return float("nan"), float("nan"), float("nan")

    centroids: dict[Any, np.ndarray] = {}
    within_spreads: list[float] = []
    for lbl, idxs in group_map.items():
        pts = states[idxs]
        centroid = pts.mean(axis=0)
        centroids[lbl] = centroid
        if len(idxs) >= 2:
            within_spreads.append(float(np.linalg.norm(pts - centroid, axis=1).mean()))
        else:
            within_spreads.append(0.0)

    within_spread = float(np.mean(within_spreads))
    labels = list(group_map.keys())
    between_dists = [
        float(np.linalg.norm(centroids[labels[i]] - centroids[labels[j]]))
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    between_spread = float(np.mean(between_dists)) if between_dists else float("nan")
    return between_spread - within_spread, within_spread, between_spread


def _mean_silhouette(
    states: np.ndarray,
    group_map: dict[Any, list[int]],
) -> float:
    if len(group_map) < 2:
        return float("nan")

    indices: list[int] = []
    cluster_ids: list[int] = []
    label_to_id = {lbl: k for k, lbl in enumerate(group_map)}
    for lbl, idxs in group_map.items():
        cid = label_to_id[lbl]
        for i in idxs:
            indices.append(i)
            cluster_ids.append(cid)

    n = len(indices)
    if n < 2:
        return float("nan")

    pts = states[indices]
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    silhouettes: list[float] = []
    cluster_ids_arr = np.array(cluster_ids)
    for i in range(n):
        same = cluster_ids_arr == cluster_ids_arr[i]
        same[i] = False
        if not same.any():
            continue
        a_i = float(dists[i, same].mean())
        other_clusters = [c for c in np.unique(cluster_ids_arr) if c != cluster_ids_arr[i]]
        if not other_clusters:
            continue
        b_i = min(float(dists[i, cluster_ids_arr == c].mean()) for c in other_clusters)
        denom = max(a_i, b_i)
        silhouettes.append((b_i - a_i) / denom if denom > 0 else 0.0)

    return float(np.mean(silhouettes)) if silhouettes else float("nan")


def _multivariate_eta2(
    states: np.ndarray,
    group_map: dict[Any, list[int]],
) -> float:
    if len(group_map) < 2:
        return float("nan")

    all_indices = [i for idxs in group_map.values() for i in idxs]
    pts = states[all_indices]
    grand_mean = pts.mean(axis=0)
    ss_total = float(((pts - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = sum(
        len(idxs) * float(((states[idxs].mean(axis=0) - grand_mean) ** 2).sum())
        for idxs in group_map.values()
    )
    return ss_between / ss_total


def _pairwise_within_between_medians(
    states: np.ndarray,
    group_map: dict[Any, list[int]],
) -> tuple[float, float]:
    indices: list[int] = []
    labels: list[Any] = []
    for lbl, idxs in group_map.items():
        for i in idxs:
            indices.append(i)
            labels.append(lbl)

    within: list[float] = []
    between: list[float] = []
    n = len(indices)
    for a in range(n):
        for b in range(a + 1, n):
            dist = float(np.linalg.norm(states[indices[a]] - states[indices[b]]))
            if labels[a] == labels[b]:
                within.append(dist)
            else:
                between.append(dist)

    within_med = float(np.median(within)) if within else float("nan")
    between_med = float(np.median(between)) if between else float("nan")
    return within_med, between_med


def _label_shuffle_centroid_gap(
    states: np.ndarray,
    labels: list[Any],
    valid_mask: list[bool] | None,
    observed_gap: float,
    *,
    n_shuffle: int = 199,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    valid_indices = [
        i for i in range(len(labels))
        if valid_mask is None or valid_mask[i]
    ]
    if len(valid_indices) < 2 or not np.isfinite(observed_gap):
        return float("nan"), float("nan")

    base_labels = [labels[i] for i in valid_indices]
    null_gaps: list[float] = []
    for _ in range(n_shuffle):
        shuffled = list(base_labels)
        rng.shuffle(shuffled)
        group_map: dict[Any, list[int]] = defaultdict(list)
        for idx, lbl in zip(valid_indices, shuffled):
            group_map[lbl].append(idx)
        gap, _, _ = _centroid_separation_metrics(states, dict(group_map))
        if np.isfinite(gap):
            null_gaps.append(gap)

    if not null_gaps:
        return float("nan"), float("nan")

    null_arr = np.asarray(null_gaps, dtype=float)
    z = float((observed_gap - null_arr.mean()) / (null_arr.std() + 1e-12))
    p = float((np.sum(null_arr >= observed_gap) + 1) / (len(null_arr) + 1))
    return z, p


def _label_shuffle_pairwise_ratio_p(
    states: np.ndarray,
    labels: list[Any],
    valid_mask: list[bool] | None,
    observed_ratio: float,
    *,
    n_shuffle: int = 199,
    rng: np.random.Generator | None = None,
) -> float:
    """One-sided p-value: fraction of shuffles with ratio <= observed (within closer than chance)."""
    rng = rng or np.random.default_rng(1)
    valid_indices = [
        i for i in range(len(labels))
        if valid_mask is None or valid_mask[i]
    ]
    if len(valid_indices) < 2 or not np.isfinite(observed_ratio):
        return float("nan")

    base_labels = [labels[i] for i in valid_indices]
    null_ratios: list[float] = []
    for _ in range(n_shuffle):
        shuffled = list(base_labels)
        rng.shuffle(shuffled)
        group_map: dict[Any, list[int]] = defaultdict(list)
        for idx, lbl in zip(valid_indices, shuffled):
            group_map[lbl].append(idx)
        w_med, b_med = _pairwise_within_between_medians(states, dict(group_map))
        if np.isfinite(w_med) and np.isfinite(b_med) and b_med > 0:
            null_ratios.append(w_med / b_med)

    if not null_ratios:
        return float("nan")

    null_arr = np.asarray(null_ratios, dtype=float)
    return float((np.sum(null_arr <= observed_ratio) + 1) / (len(null_arr) + 1))


def compute_feature_separation_stats(
    hidden_states: np.ndarray,
    labels,
    *,
    features: tuple[str, ...] = SEPARATION_FEATURES,
    n_shuffle: int = 199,
    rng: np.random.Generator | None = None,
) -> FeatureSeparationStats:
    rng = rng or np.random.default_rng(0)
    centroid_gap: dict[str, float] = {}
    within_spread: dict[str, float] = {}
    between_spread: dict[str, float] = {}
    silhouette: dict[str, float] = {}
    eta2: dict[str, float] = {}
    pairwise_within_median: dict[str, float] = {}
    pairwise_between_median: dict[str, float] = {}
    shuffle_z: dict[str, float] = {}
    shuffle_p: dict[str, float] = {}
    n_groups: dict[str, int] = {}
    n_points: dict[str, int] = {}

    for feat in features:
        vals, mask = labels.feature_values(feat)
        group_map = _label_groups(vals, mask)
        n_groups[feat] = len(group_map)
        n_points[feat] = sum(len(v) for v in group_map.values())

        gap, w_spread, b_spread = _centroid_separation_metrics(hidden_states, group_map)
        centroid_gap[feat] = gap
        within_spread[feat] = w_spread
        between_spread[feat] = b_spread
        silhouette[feat] = _mean_silhouette(hidden_states, group_map)
        eta2[feat] = _multivariate_eta2(hidden_states, group_map)
        w_med, b_med = _pairwise_within_between_medians(hidden_states, group_map)
        pairwise_within_median[feat] = w_med
        pairwise_between_median[feat] = b_med

        z, _ = _label_shuffle_centroid_gap(
            hidden_states, vals, mask, gap, n_shuffle=n_shuffle, rng=rng,
        )
        shuffle_z[feat] = z
        ratio = w_med / b_med if np.isfinite(w_med) and np.isfinite(b_med) and b_med > 0 else float("nan")
        shuffle_p[feat] = _label_shuffle_pairwise_ratio_p(
            hidden_states, vals, mask, ratio, n_shuffle=n_shuffle, rng=rng,
        )

    return FeatureSeparationStats(
        features=features,
        centroid_gap=centroid_gap,
        within_spread=within_spread,
        between_spread=between_spread,
        silhouette=silhouette,
        eta2=eta2,
        pairwise_within_median=pairwise_within_median,
        pairwise_between_median=pairwise_between_median,
        shuffle_z=shuffle_z,
        shuffle_p=shuffle_p,
        n_groups=n_groups,
        n_points=n_points,
    )


def plot_feature_separation_summary(
    text: str,
    hidden_states: np.ndarray,
    automaton: MinimizedVocabAutomaton,
    save_path: str,
    *,
    spaced: bool = False,
    words: list[str] | None = None,
    label_words: list[str] | None = None,
    condensed: CondensedView | None = None,
    output_probs: np.ndarray | None = None,
    repr_label: str = "hidden state",
    n_shuffle: int = 199,
) -> FeatureSeparationStats | None:
    from unit_selectivity import FEATURE_COLORS, FEATURE_DISPLAY, build_timestep_labels

    if condensed is None:
        condensed = condense_hidden_states_by_prefix(
            text, hidden_states, output_probs, spaced=spaced, words=words,
        )
        hidden_states = condensed.hidden_states

    if hidden_states.shape[0] < 2:
        print("feature separation: need at least 2 points")
        return None

    ts_labels = build_timestep_labels(
        text, automaton,
        spaced=spaced, words=words, label_words=label_words, condensed=condensed,
    )
    stats = compute_feature_separation_stats(
        hidden_states, ts_labels, n_shuffle=n_shuffle,
    )
    feats = list(stats.features)
    x = np.arange(len(feats))
    colors = [FEATURE_COLORS.get(f, "#888888") for f in feats]
    tick_labels = [FEATURE_DISPLAY.get(f, f) for f in feats]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    vals = [stats.centroid_gap[f] for f in feats]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0.0, color="0.3", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("between − within spread")
    ax.set_title("Centroid gap (group-balanced)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax = axes[0, 1]
    vals = [stats.silhouette[f] for f in feats]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0.0, color="0.3", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean silhouette")
    ax.set_title("Mean silhouette")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax = axes[0, 2]
    vals = [stats.eta2[f] for f in feats]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("η²")
    ax.set_title("Multivariate η²")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax = axes[1, 0]
    width = 0.36
    w_vals = [stats.pairwise_within_median[f] for f in feats]
    b_vals = [stats.pairwise_between_median[f] for f in feats]
    ax.bar(x - width / 2, w_vals, width, label="within", color="#4c72b0", alpha=0.85)
    ax.bar(x + width / 2, b_vals, width, label="between", color="#dd8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"L2 distance ({repr_label})")
    ax.set_title("Pairwise within vs between (median)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax = axes[1, 1]
    vals = [stats.shuffle_z[f] for f in feats]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0.0, color="0.3", linewidth=0.8, linestyle=":")
    ax.axhline(1.96, color="0.5", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(-1.96, color="0.5", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("z-score")
    ax.set_title("Centroid gap vs label shuffle")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax = axes[1, 2]
    ratios = []
    for f in feats:
        w_med = stats.pairwise_within_median[f]
        b_med = stats.pairwise_between_median[f]
        ratios.append(w_med / b_med if np.isfinite(w_med) and np.isfinite(b_med) and b_med > 0 else float("nan"))
    ax.bar(x, ratios, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(1.0, color="0.3", linewidth=0.8, linestyle=":")
    ymax = float(np.nanmax(ratios)) if np.any(np.isfinite(ratios)) else 1.0
    for i, (f, r) in enumerate(zip(feats, ratios)):
        p = stats.shuffle_p[f]
        if np.isfinite(p) and np.isfinite(r):
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            label = f"p={p:.3f}{stars}"
            ax.text(i, r + 0.02 * max(ymax, 0.01), label, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("within / between")
    ax.set_title("Pairwise ratio (shuffle p-value)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    title = f"Feature separation summary ({repr_label}, n={hidden_states.shape[0]} points)"
    fig.suptitle(_condensed_plot_title(title, condensed), fontsize=12, y=1.02)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")

    json_path = str(Path(save_path).with_suffix(".json"))

    def _json_float(v: float) -> float | None:
        return None if isinstance(v, float) and not np.isfinite(v) else v

    payload = {
        "features": list(stats.features),
        "centroid_gap": {k: _json_float(v) for k, v in stats.centroid_gap.items()},
        "within_spread": {k: _json_float(v) for k, v in stats.within_spread.items()},
        "between_spread": {k: _json_float(v) for k, v in stats.between_spread.items()},
        "silhouette": {k: _json_float(v) for k, v in stats.silhouette.items()},
        "eta2": {k: _json_float(v) for k, v in stats.eta2.items()},
        "pairwise_within_median": {k: _json_float(v) for k, v in stats.pairwise_within_median.items()},
        "pairwise_between_median": {k: _json_float(v) for k, v in stats.pairwise_between_median.items()},
        "shuffle_z": {k: _json_float(v) for k, v in stats.shuffle_z.items()},
        "shuffle_p": {k: _json_float(v) for k, v in stats.shuffle_p.items()},
        "n_groups": stats.n_groups,
        "n_points": stats.n_points,
        "n_shuffle": n_shuffle,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_path}")

    return stats


def plot_dfa_state_distance_comparison(
    text: str,
    hidden_states: np.ndarray,
    automaton: MinimizedVocabAutomaton,
    save_path: str,
    *,
    spaced: bool = False,
    words: list[str] | None = None,
    condensed: CondensedView | None = None,
    repr_label: str = "hidden state",
) -> None:
    """Subsampled pairwise distances + median (diamond) ± MAD; y-axis clipped at 0."""
    from vocab_diagrams import (
        dfa_state_at_position,
        dfa_state_for_prefix,
        position_from_end_at_index,
        position_in_word_at_index,
        position_in_word_for_prefix_label,
    )

    vocab = _corpus_vocab(text, words)
    if condensed is not None:
        hidden_states = condensed.hidden_states
        spaced = condensed.spaced
        compare_chars = condensed.input_chars
        state_ids = [
            dfa_state_for_prefix(l, automaton, spaced=spaced) for l in condensed.labels
        ]
        position_ids = [
            position_in_word_for_prefix_label(l) for l in condensed.labels
        ]
        position_from_end_ids = [
            position_from_end_at_index(text, idx, spaced=spaced, vocab=vocab)
            for idx in condensed.timestep_indices
        ]
    else:
        compare_chars = list(text)
        state_ids = [
            dfa_state_at_position(
                text, t, automaton, spaced=spaced, vocab=vocab,
            ) for t in range(len(text))
        ]
        position_ids = [
            position_in_word_at_index(text, t, spaced=spaced, vocab=vocab)
            for t in range(len(text))
        ]
        position_from_end_ids = [
            position_from_end_at_index(text, t, spaced=spaced, vocab=vocab)
            for t in range(len(text))
        ]
    n = hidden_states.shape[0]
    if n < 2:
        return

    by_label = pairwise_hidden_state_distance_groups(
        compare_chars, hidden_states, state_ids, position_ids,
        position_from_end_ids,
    )
    if len(by_label["Within DFA state"]) == 0 or len(by_label["Between DFA states"]) == 0:
        print("DFA distance comparison: need both within- and between-state pairs")
        return

    palette = PAIR_DISTANCE_PALETTE
    order = [
        label for label in PAIR_DISTANCE_CATEGORY_ORDER
        if len(by_label[label]) > 0
    ]
    specs = [(label, by_label[label]) for label in order]
    stats = {
        label: (*median_mad(vals), len(vals))
        for label, vals in specs
    }

    fig, ax = plt.subplots(figsize=(18.5, 5.5), constrained_layout=True)
    x = np.arange(len(order))
    rng = np.random.default_rng(0)
    max_points = 200
    for i, label in enumerate(order):
        vals = np.asarray(by_label[label], dtype=float)
        if len(vals) > max_points:
            idx = rng.choice(len(vals), size=max_points, replace=False)
            vals = vals[idx]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        color = palette[label]
        ax.scatter(
            x[i] + jitter,
            vals,
            c=color,
            alpha=0.35,
            s=14,
            linewidths=0,
            zorder=1,
        )

        med, mad, _ = stats[label]
        err_lo = min(med, mad)
        ax.errorbar(
            x[i],
            med,
            yerr=np.array([[err_lo], [mad]]),
            fmt="D",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=10,
            capthick=2.0,
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="0.15",
            markeredgewidth=1.5,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel(f"Euclidean distance ||v_i − v_j|| ({repr_label})")
    n_pairs = n * (n - 1) // 2
    title = f"Pairwise {repr_label} distance ({n_pairs} pairs, n={n} timesteps)"
    ax.set_title(_condensed_plot_title(title, condensed))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)
    within_med = stats.get("Within DFA state", (float("nan"),))[0]
    between_med = stats.get("Between DFA states", (float("nan"),))[0]
    ratio = within_med / between_med if between_med > 0 else float("inf")
    parts = [
        f"{label}: n={n_} median={m:.4f} mad={s:.4f}"
        for label, (m, s, n_) in stats.items()
    ]
    print("pairwise L2: " + " | ".join(parts) + f" | ratio within/between={ratio:.3f}")

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


from viz.dimred import (
    EMBED_METHODS,
    embed_axis_labels_2d,
    embed_axis_labels_3d,
    embed_dim_label,
    embed_save_path,
    fit_embed_2d_with_evr,
    fit_embed_3d_with_evr,
    fit_pca_2d,
    fit_pca_2d_with_evr,
    fit_pca_3d_with_evr,
    trajectories_for_embed,
    trajectories_from_segments,
)


def _embed_trajectories_for_text(
    text: str,
    hidden_states: np.ndarray,
    *,
    spaced: bool,
    words: list[str] | None,
    word_path_indices: list[list[int]] | None = None,
) -> list[np.ndarray]:
    if word_path_indices is not None:
        return trajectories_for_embed(hidden_states, word_path_indices=word_path_indices)
    segments = corpus_segments(text, list(_corpus_vocab(text, words) or []), spaced=spaced)
    return trajectories_for_embed(hidden_states, segments=segments)


def _plot_embed_variants(plot_fn, base_save_path: str, **kwargs) -> None:
    """Call a plot function for each linear embedding method (PCA and JPCA)."""
    for method in EMBED_METHODS:
        plot_fn(
            save_path=embed_save_path(base_save_path, method),
            embed_method=method,
            **kwargs,
        )


def _normalize_hidden_rows(hidden: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Unit-normalize hidden activity: û = h / ||h|| (per row or 1D vector)."""
    h = np.asarray(hidden, dtype=float)
    if h.ndim == 1:
        norm = max(float(np.linalg.norm(h)), eps)
        return h / norm
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    return h / np.maximum(norms, eps)


def _project_hidden_to_pca(
    hidden: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    *,
    normalize_activity: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project hidden state(s) into PCA space, optionally after unit normalization."""
    h = np.asarray(hidden, dtype=float)
    if normalize_activity:
        h = _normalize_hidden_rows(h, eps=eps)
    if h.ndim == 1:
        return ((h.ravel() - mean) @ components.T).reshape(1, -1)
    return (h - mean) @ components.T


def fit_ica_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FastICA to 2D (sklearn). Returns coords, mean, components."""
    from sklearn.decomposition import FastICA

    mean = np.mean(points, axis=0)
    centered = points - mean
    if centered.shape[0] < 2 or centered.shape[1] < 2:
        raise ValueError("ICA needs at least 2 samples and 2 dimensions")
    ica = FastICA(n_components=2, random_state=0, max_iter=2000, tol=1e-4)
    coords = ica.fit_transform(centered)
    return coords, mean, ica.components_


def pca_2d(points):
    """Project points to two dimensions with PCA using NumPy's SVD."""
    return fit_pca_2d(points)[0]


def reconstruct_from_pca(coords, mean, components):
    """Approximate hidden states from PC1/PC2 (other PCs set to zero)."""
    return mean + coords @ components


def project_to_fixed_pca(hidden_states, mean, components):
    """Project hidden states into a pre-fit PCA basis."""
    return (hidden_states - mean) @ components.T


def model_at_weight_snapshot(model: dict, snap_idx: int) -> dict:
    """Reconstruct model weights (and biases when saved) at a training snapshot."""
    from rnn.rnn_dyn import unpack_weight_snapshot

    hidden_size = int(model["hidden_size"])
    vocab_size = int(model["vocab_size"])
    vec = np.asarray(model["weight_snap_outgoing"][snap_idx], dtype=float)
    W_in, W_hh, W_ho = unpack_weight_snapshot(vec, hidden_size, vocab_size)
    snap_model = dict(model)
    snap_model["weights_input_to_hidden"] = W_in
    snap_model["weights_hidden_to_hidden"] = W_hh
    snap_model["weights_hidden_to_output"] = W_ho
    if "weight_snap_bias_hidden" in model:
        snap_model["bias_hidden"] = np.asarray(
            model["weight_snap_bias_hidden"][snap_idx],
        ).reshape(-1, 1)
    if "weight_snap_bias_output" in model:
        snap_model["bias_output"] = np.asarray(
            model["weight_snap_bias_output"][snap_idx],
        ).reshape(-1, 1)
    return snap_model


def _encode_frame_sequence(frame_paths: list[str], out_path: str, *, fps: int) -> str:
    """Encode PNG frames to gif (Pillow) or mp4 (ffmpeg / imageio). Returns path written."""
    import shutil
    import subprocess

    if not frame_paths:
        raise ValueError("no frames to encode")

    if out_path.endswith(".gif"):
        from PIL import Image

        images = [Image.open(fp) for fp in frame_paths]
        duration_ms = max(int(1000 / fps), 1)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        for im in images:
            im.close()
        return out_path

    mp4_path = out_path if out_path.endswith(".mp4") else f"{out_path}.mp4"
    if shutil.which("ffmpeg"):
        frame_dir = os.path.dirname(frame_paths[0])
        pattern = os.path.join(frame_dir, "frame_%04d.png")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", pattern,
                    "-pix_fmt", "yuv420p",
                    mp4_path,
                ],
                check=True,
                capture_output=True,
            )
            return mp4_path
        except subprocess.CalledProcessError:
            pass

    try:
        import imageio.v3 as iio

        frames = [iio.imread(fp) for fp in frame_paths]
        iio.imwrite(mp4_path, frames, fps=fps, codec="libx264")
        return mp4_path
    except Exception:
        pass

    from PIL import Image

    gif_path = out_path.replace(".mp4", ".gif")
    images = [Image.open(fp) for fp in frame_paths]
    duration_ms = max(int(1000 / fps), 1)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for im in images:
        im.close()
    return gif_path


def _snapshot_mean_displacements(projected_frames: list[np.ndarray]) -> np.ndarray:
    """Per-snapshot step size (index 0 is 0)."""
    projs = [np.asarray(p, dtype=float) for p in projected_frames]
    disps = np.zeros(len(projs), dtype=float)
    for t in range(1, len(projs)):
        step = projs[t] - projs[t - 1]
        disps[t] = float(np.linalg.norm(step, axis=-1).mean())
    return disps


def _displacement_weights_at_iters(
    model: dict,
    iters: np.ndarray,
    projected_frames: list[np.ndarray],
    *,
    error_weight: float = 0.35,
) -> np.ndarray:
    """Per-step weights for resampling: PCA displacement + optional word-error drop."""
    disps = _snapshot_mean_displacements(projected_frames)
    weights = disps.copy()
    if error_weight <= 0:
        return weights

    metric_iters = np.asarray(model.get("metric_iterations", []), dtype=float)
    metric_err = np.asarray(model.get("metric_word_error_frac", []), dtype=float)
    if len(metric_iters) < 2:
        return weights

    err_on_snaps = np.interp(iters.astype(float), metric_iters, metric_err)
    err_drop = np.zeros(len(iters), dtype=float)
    err_drop[1:] = np.maximum(0.0, err_on_snaps[:-1] - err_on_snaps[1:])
    err_scale = float(np.max(disps[1:])) if np.any(disps[1:] > 0) else 1.0
    err_max = float(np.max(err_drop[1:])) if np.any(err_drop[1:] > 0) else 0.0
    if err_max > 0:
        weights[1:] += error_weight * err_scale * (err_drop[1:] / err_max)
    return weights


def resample_frames_by_displacement(
    model: dict,
    iters: np.ndarray,
    projected_frames: list[np.ndarray],
    n_video_frames: int,
    *,
    error_weight: float = 0.35,
) -> list[tuple[int, float, np.ndarray]]:
    """Sample video frames uniformly in cumulative displacement (linearly interpolated)."""
    iters_f = np.asarray(iters, dtype=float)
    projs = [np.asarray(p, dtype=float) for p in projected_frames]
    if len(projs) < 2 or n_video_frames < 2:
        n = min(len(projs), max(n_video_frames, 1))
        return [
            (int(iters[i]), 100.0 * i / max(n - 1, 1), projs[i]) for i in range(n)
        ]

    weights = _displacement_weights_at_iters(
        model, iters, projected_frames, error_weight=error_weight,
    )
    cum = np.cumsum(weights)
    total = float(cum[-1])
    if total <= 0:
        return [
            (int(iters_f[i]), 100.0 * i / max(len(projs) - 1, 1), projs[i])
            for i in range(len(projs))
        ]

    out: list[tuple[int, float, np.ndarray]] = []
    for k in range(n_video_frames):
        target = (k / (n_video_frames - 1)) * total
        idx = int(np.searchsorted(cum, target, side="right")) - 1
        idx = min(max(idx, 0), len(projs) - 2)
        seg_lo, seg_hi = float(cum[idx]), float(cum[idx + 1])
        alpha = (target - seg_lo) / (seg_hi - seg_lo) if seg_hi > seg_lo else 0.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        proj = (1.0 - alpha) * projs[idx] + alpha * projs[idx + 1]
        iter_num = int(round((1.0 - alpha) * iters_f[idx] + alpha * iters_f[idx + 1]))
        geom_pct = 100.0 * target / total
        out.append((iter_num, geom_pct, proj))
    return out


def geometry_learning_end_iteration(
    iters: np.ndarray,
    projected_frames: list[np.ndarray],
    *,
    cumulative_fraction: float = 0.90,
    tail_iters: int = 30,
) -> int:
    """Last training iter to show once PCA geometry has moved most of its distance."""
    if len(projected_frames) < 2:
        return int(iters[-1])

    disps: list[float] = []
    for t in range(1, len(projected_frames)):
        step = np.asarray(projected_frames[t]) - np.asarray(projected_frames[t - 1])
        disps.append(float(np.linalg.norm(step, axis=-1).mean()))

    cum = np.cumsum(disps)
    total = float(cum[-1])
    if total <= 0:
        return int(iters[-1])

    idx = int(np.searchsorted(cum, cumulative_fraction * total))
    idx = min(idx + 1, len(iters) - 1)
    return int(iters[idx]) + int(tail_iters)


def learning_plateau_iteration(
    model: dict,
    *,
    fraction_remaining: float = 0.05,
    min_iter: int = 100,
) -> int:
    """First iteration where smoothed loss is within `fraction_remaining` of its total drop."""
    iters = np.asarray(model.get("loss_iterations", []), dtype=int)
    smooth = np.asarray(model.get("loss_smooth", []), dtype=float)
    if len(iters) < 2:
        return int(iters[-1]) if len(iters) else 0

    tail = smooth[-max(100, len(smooth) // 20):]
    finite_tail = tail[np.isfinite(tail)]
    final_loss = float(np.median(finite_tail)) if finite_tail.size else float(smooth[-1])
    initial_loss = float(smooth[0])
    drop = initial_loss - final_loss
    if drop <= 0:
        return int(iters[-1])

    threshold = final_loss + fraction_remaining * drop
    for it, loss in zip(iters, smooth):
        if int(it) < min_iter:
            continue
        if np.isfinite(loss) and loss <= threshold:
            return int(it)
    return int(iters[-1])


def write_hidden_state_pca_learning_video(
    model: dict,
    text: str,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    fps: int = 4,
    video_frames: int | None = None,
    frame_subsample: int = 1,
    dpi: int = 100,
    max_iter: int | None = None,
    annot_style: str = "leaders",
    error_weight: float = 0.35,
) -> None:
    """Animate hidden states in the final-model PCA basis over the learning phase."""
    if "weight_snap_outgoing" not in model:
        print(f"skip {save_path}: re-run training with --save-snapshots to record weight snapshots")
        return

    snaps = np.asarray(model["weight_snap_outgoing"], dtype=float)
    iters = np.asarray(model["weight_snap_iterations"], dtype=int)
    if snaps.ndim != 2 or snaps.shape[0] < 2:
        print(f"skip {save_path}: insufficient weight snapshot history")
        return

    from rnn.rnn_dyn import snapshot_vector_layout

    hidden_size = int(model["hidden_size"])
    vocab_size = int(model["vocab_size"])
    if snapshot_vector_layout(hidden_size, vocab_size, snaps.shape[1]) != "full":
        print(
            f"skip {save_path}: re-run training for full snapshots "
            "(need W_xh + W_hh in weight_snap_outgoing)",
        )
        return

    if automaton is None:
        print(f"skip {save_path}: vocabulary automaton required for DFA coloring")
        return

    final_hidden, _ = forward_pass(model, text)
    _, mean, components, evr = fit_pca_2d_with_evr(final_hidden)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0

    state_ids = _dfa_state_ids_at_timesteps(text, automaton, spaced=spaced)
    state_colors = _state_id_colors(state_ids)

    scan_cap = min(int(iters[-1]), 600)
    scan_idx = np.where(iters <= scan_cap)[0]
    projected_scan: list[np.ndarray] = []
    for snap_i in scan_idx:
        snap_model = model_at_weight_snapshot(model, int(snap_i))
        hidden_states, _ = forward_pass(snap_model, text)
        projected_scan.append(project_to_fixed_pca(hidden_states, mean, components))

    loss_plateau = learning_plateau_iteration(model)
    if max_iter is None:
        max_iter = geometry_learning_end_iteration(
            iters[scan_idx], projected_scan,
        )
    max_iter = min(int(max_iter), int(iters[-1]))

    in_range = iters <= max_iter
    frame_idx = np.where(in_range)[0]
    frame_idx = frame_idx[:: max(int(frame_subsample), 1)]
    if frame_idx.size < 2:
        print(f"skip {save_path}: no snapshots in learning window (max_iter={max_iter})")
        return

    scan_lookup = {int(i): j for j, i in enumerate(scan_idx)}
    projected_frames: list[np.ndarray] = []
    for snap_i in frame_idx:
        j = scan_lookup.get(int(snap_i))
        if j is not None:
            projected_frames.append(projected_scan[j])
        else:
            snap_model = model_at_weight_snapshot(model, int(snap_i))
            hidden_states, _ = forward_pass(snap_model, text)
            projected_frames.append(project_to_fixed_pca(hidden_states, mean, components))

    snap_iters = iters[frame_idx]
    n_out = int(video_frames) if video_frames is not None else fps * 72
    n_out = max(n_out, 2)
    timeline = resample_frames_by_displacement(
        model, snap_iters, projected_frames, n_out, error_weight=error_weight,
    )
    print(
        f"learning video: snapshots 0-{max_iter} -> {len(timeline)} frames "
        f"(displacement-paced, geometry end ~{max_iter}, loss plateau ~{loss_plateau})",
    )

    xlim, ylim = _square_data_limits(
        *[p for _, _, p in timeline], padding_frac=0.38,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    import tempfile

    frame_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rnn_pca_frames_") as tmp:
        for frame_no, (iter_num, geom_pct, projected) in enumerate(timeline):
            fig, ax = plt.subplots(figsize=(12.8, 12.8), constrained_layout=True)
            add_dfa_state_annotations(
                ax,
                text,
                projected,
                automaton,
                spaced=spaced,
                state_colors=state_colors,
                annot_style=annot_style,
                point_size=70,
                label_fontsize=11,
                leader_linewidth=1.6,
            )
            ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
            ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
            ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
            ctx = prefix_axis_label(spaced=spaced, text=text)
            ax.set_title(
                f"Hidden states in fixed final-model PCA · {ctx}\n"
                f"geometry {geom_pct:.0f}% · training iter {int(iter_num)}",
            )
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fp = os.path.join(tmp, f"frame_{frame_no:04d}.png")
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            frame_paths.append(fp)

        written = _encode_frame_sequence(frame_paths, save_path, fps=fps)
    print(f"wrote {written}")


def argmax_next_char(model, hidden_states):
    """Most likely next character index for each row of hidden_states."""
    return np.argmax(next_char_probabilities(model, hidden_states), axis=1)


def next_char_probabilities(model, hidden_states):
    """Softmax next-char distribution for each row of hidden_states."""
    if model.get("model_type") == "transformer":
        import torch
        torch_model = model["_torch_model"]
        h = torch.tensor(hidden_states, dtype=torch.float32)
        logits = torch_model.lm_head(h)
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()
    weights = model["weights_hidden_to_output"]
    bias = model["bias_output"].ravel()
    logits = hidden_states @ weights.T + bias
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def prediction_entropy(probs):
    """Shannon entropy (nats) of each row of a probability matrix."""
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def representation_label(
    model,
    *,
    prob_grid: bool = False,
    repr_name: str | None = None,
) -> str:
    """Human-readable name for the vector being PCA'd / read out."""
    if repr_name is not None:
        if prob_grid:
            return f"{repr_name} (lm_head on 2D PCA reconstruction; not full forward pass)"
        return repr_name
    if model.get("model_type") == "transformer":
        if prob_grid:
            return "block output (lm_head on 2D PCA reconstruction; not full forward pass)"
        return "transformer output (pre-lm_head, query position)"
    return "hidden state h"


def build_pca_plane_grid(
    text,
    hidden_states,
    grid_resolution=120,
    *,
    spaced: bool = False,
    prefix_labels: list[str] | None = None,
    embed_method: str = "pca",
    trajectories: list[np.ndarray] | None = None,
):
    """Embedding mesh and 2D-reconstructed hidden states on a grid covering data + labels."""
    projected, mean, components, _evr = fit_embed_2d_with_evr(
        hidden_states, method=embed_method, trajectories=trajectories,
    )
    x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
    y_min, y_max = projected[:, 1].min(), projected[:, 1].max()
    x_pad = max((x_max - x_min) * 0.12, 1e-3)
    y_pad = max((y_max - y_min) * 0.12, 1e-3)
    if prefix_labels is not None or text:
        _, _, _, label_positions = layout_trigram_labels(
            text, projected, spaced=spaced, prefix_labels=prefix_labels,
        )
        if label_positions:
            text_positions = np.array(list(label_positions.values()))
            x_min = min(x_min, text_positions[:, 0].min())
            x_max = max(x_max, text_positions[:, 0].max())
            y_min = min(y_min, text_positions[:, 1].min())
            y_max = max(y_max, text_positions[:, 1].max())
            x_pad = max(x_pad, (x_max - x_min) * 0.08)
            y_pad = max(y_pad, (y_max - y_min) * 0.08)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    xs = np.linspace(xlim[0], xlim[1], grid_resolution)
    ys = np.linspace(ylim[0], ylim[1], grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_hidden = reconstruct_from_pca(grid_coords, mean, components)
    return grid_x, grid_y, grid_hidden, projected, xlim, ylim, _evr


def _rolling_median(y: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling median; edges use available samples only."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0 or win <= 1:
        return y
    out = np.empty(n)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(y[lo:hi]))
    return out


def _tight_ylim(
    y: np.ndarray,
    *,
    pad_frac: float = 0.06,
    floor: float | None = None,
    ceiling: float | None = None,
) -> tuple[float, float]:
    """Axis limits that fit the full series with a small margin."""
    vals = np.asarray(y, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        pad = max(abs(lo) * 0.05, 1e-3)
    else:
        pad = (hi - lo) * pad_frac
    lo -= pad
    hi += pad
    if floor is not None:
        lo = max(floor, lo)
    if ceiling is not None:
        hi = min(ceiling, hi)
    if lo >= hi:
        hi = lo + max(pad, 1e-3)
    return lo, hi


def _learning_curve_series(
    model,
    *,
    smoothed: bool = False,
    sequence_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if "loss_iterations" not in model:
        return None
    iters = np.asarray(model["loss_iterations"], dtype=int)
    seq_len = sequence_length
    if seq_len is None:
        seq_len = int(model["sequence_length"]) if "sequence_length" in model else None
    if smoothed and "loss_smooth" in model:
        ce = np.asarray(model["loss_smooth"], dtype=float)
    elif "loss_window" in model:
        ce = np.asarray(model["loss_window"], dtype=float)
    elif "loss_smooth" in model:
        ce = np.asarray(model["loss_smooth"], dtype=float)
        label = "cross-entropy / char (smoothed; re-train for raw loss)"
        if seq_len and seq_len > 0:
            ce = ce / seq_len
        return iters, ce, label
    else:
        return None
    label = "cross-entropy / char"
    if seq_len and seq_len > 0:
        ce = ce / seq_len
    if "metric_val_ce" in model and "metric_iterations" in model:
        label = "train cross-entropy / char"
    return iters, ce, label


def plot_learning_curve_on_axes(
    ax,
    model,
    *,
    title: str | None = None,
    compact: bool = False,
    loss_only: bool = False,
    show_legend: bool = True,
    show_ylabel: bool = True,
    show_metric_ylabel: bool = True,
    smoothed: bool = False,
    max_iter: int | None = None,
    truncate_to_plateau: bool = False,
    plateau_tail_iters: int = 200,
    sequence_length: int | None = None,
) -> bool:
    """Plot cross-entropy (and optional rollout metric) on ax. Returns False if no history."""
    series = _learning_curve_series(model, smoothed=smoothed, sequence_length=sequence_length)
    if series is None:
        return False

    iters, ce_plot, ce_label = series
    if max_iter is None and truncate_to_plateau:
        max_iter = learning_plateau_iteration(model) + int(plateau_tail_iters)
    if max_iter is not None:
        end = int(max_iter)
        keep = iters <= end
        iters = iters[keep]
        ce_plot = ce_plot[keep]
        if iters.size == 0:
            return False
    lw = 1.0 if compact else 1.2
    fs = 7 if compact else 8
    ce_line, = ax.plot(iters, ce_plot, color="steelblue", linewidth=lw, label=ce_label)
    legend_lines = [ce_line]
    legend_labels = [ce_label]
    val_line = None
    if not loss_only and "metric_val_ce" in model and "metric_iterations" in model:
        val_iters = np.asarray(model["metric_iterations"], dtype=int)
        val_ce = np.asarray(model["metric_val_ce"], dtype=float)
        if max_iter is not None:
            keep = val_iters <= int(max_iter)
            val_iters = val_iters[keep]
            val_ce = val_ce[keep]
        if val_iters.size:
            val_line, = ax.plot(
                val_iters,
                val_ce,
                color="seagreen",
                linewidth=lw,
                alpha=0.9,
                label="val cross-entropy / char",
            )
            legend_lines.append(val_line)
            legend_labels.append(val_line.get_label())
    ax.set_xlabel("iteration", fontsize=fs)
    if show_ylabel:
        ylabel = "cross-entropy / char" if val_line is not None else ce_label
        ax.set_ylabel(ylabel, fontsize=fs)
    if title is not None:
        ax.set_title(title, fontsize=fs + 1 if compact else 10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ce_for_ylim = ce_plot
    if val_line is not None:
        ce_for_ylim = np.concatenate([ce_plot, val_ce])
    ax.set_ylim(*_tight_ylim(ce_for_ylim, floor=0.0))
    ax.tick_params(labelsize=fs)
    if max_iter is not None:
        ax.set_xlim(0, int(max_iter))

    if loss_only:
        if show_legend:
            ax.legend(loc="upper right", fontsize=fs)
        return True

    metric_line = None
    if "metric_iterations" in model and "metric_word_error_frac" in model:
        ax2 = ax.twinx()
        metric_iters = np.asarray(model["metric_iterations"], dtype=int)
        metric_pct = 100.0 * np.asarray(model["metric_word_error_frac"], dtype=float)
        if max_iter is not None:
            keep = metric_iters <= int(max_iter)
            metric_iters = metric_iters[keep]
            metric_pct = metric_pct[keep]
        metric_line, = ax2.plot(
            metric_iters,
            metric_pct,
            color="darkorange",
            linewidth=lw,
            linestyle="--",
            alpha=0.9,
            label="% chars outside vocab (avg)",
        )
    elif "metric_iterations" in model and "metric_valid_vocab_letter_frac" in model:
        ax2 = ax.twinx()
        metric_iters = np.asarray(model["metric_iterations"], dtype=int)
        metric_pct = 100.0 * (1.0 - np.asarray(model["metric_valid_vocab_letter_frac"], dtype=float))
        if max_iter is not None:
            keep = metric_iters <= int(max_iter)
            metric_iters = metric_iters[keep]
            metric_pct = metric_pct[keep]
        metric_line, = ax2.plot(
            metric_iters,
            metric_pct,
            color="darkorange",
            linewidth=lw,
            linestyle="--",
            alpha=0.9,
            label="% letters OOV",
        )

    if metric_line is not None:
        if show_metric_ylabel:
            ax2.set_ylabel(metric_line.get_label(), fontsize=fs)
        ax2.set_ylim(*_tight_ylim(metric_pct, floor=0.0, ceiling=100.0))
        ax2.tick_params(labelsize=fs)
        legend_lines.append(metric_line)
        legend_labels.append(metric_line.get_label())

    if show_legend and legend_lines:
        ax.legend(legend_lines, legend_labels, loc="upper right", fontsize=fs)
    return True


def plot_learning_curve(model, save_path, *, loss_only: bool = False):
    """Per-eval cross-entropy (raw); optional word-validity metric on twin axis."""
    if _learning_curve_series(model) is None:
        print(f"skip {save_path}: re-run training to record loss history")
        return

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    title = "Training loss" if loss_only else "Training: cross-entropy vs word-validity rollout"
    if not plot_learning_curve_on_axes(ax, model, title=title, loss_only=loss_only):
        print(f"skip {save_path}: no loss history in model bundle")
        plt.close(fig)
        return

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def _token_letter_valid_mask(text: str, vocab: set[str]) -> list[bool]:
    """Per-character mask: True if char is in a whitespace token in vocab."""
    mask = [False] * len(text)
    if not vocab or not text:
        return mask
    i = 0
    n = len(text)
    while i < n:
        if text[i] == " ":
            mask[i] = True
            i += 1
            continue
        j = i
        while j < n and text[j] != " ":
            j += 1
        token = text[i:j]
        ok = token in vocab
        for k in range(i, j):
            mask[k] = ok
        i = j
    return mask


def _char_vocab_word_mask(text: str, vocab: set[str], *, spaced: bool) -> list[bool]:
    """Per-character mask: True when the char belongs to an in-vocabulary word."""
    mask = [False] * len(text)
    if not vocab or not text:
        return mask
    if spaced:
        return _token_letter_valid_mask(text, vocab)
    for start, end, word in segment_corpus_by_words(text, vocab):
        ok = word in vocab
        for k in range(start, end + 1):
            mask[k] = ok
    return mask


SAMPLE_DISPLAY_LEN = 50
UNSPACED_WORD_SEP = "·"


def _unspaced_display_parts(
    text: str,
    vocab: set[str],
    *,
    max_len: int = SAMPLE_DISPLAY_LEN,
) -> list[tuple[str, bool | None]]:
    """Greedy word chunks for display only; None validity = separator glyph."""
    snippet = text[:max_len]
    if not snippet or not vocab:
        return [(snippet, None)]
    parts: list[tuple[str, bool | None]] = []
    for i, (start, end, word) in enumerate(segment_corpus_by_words(snippet, vocab)):
        end = min(end, len(snippet) - 1)
        if start >= len(snippet):
            break
        if i > 0:
            parts.append((UNSPACED_WORD_SEP, None))
        parts.append((snippet[start : end + 1], word in vocab))
    return parts or [(snippet, None)]


def _draw_sample_chars(
    ax,
    text: str,
    y: float,
    *,
    vocab: set[str] | None = None,
    spaced: bool = False,
    color_by_vocab: bool = True,
    show_word_separators: bool = True,
) -> None:
    snippet = text[:SAMPLE_DISPLAY_LEN]
    if not snippet:
        return
    x_step = min(0.019, 0.98 / max(len(snippet) - 1, 1))

    if vocab and not spaced and show_word_separators:
        parts = _unspaced_display_parts(snippet, vocab)
        n_glyphs = sum(len(chunk) for chunk, _ in parts)
        x_step = min(0.019, 0.98 / max(n_glyphs - 1, 1))
        x = 0.0
        for chunk, valid in parts:
            if valid is None:
                ax.text(
                    x, y, chunk,
                    transform=ax.transAxes,
                    fontfamily="monospace",
                    fontsize=9,
                    color="0.45",
                    va="center",
                    ha="left",
                )
                x += x_step * len(chunk)
                continue
            for ch in chunk:
                if color_by_vocab and valid is not None:
                    color = "#2ca02c" if valid else "#d62728"
                else:
                    color = "0.15"
                ax.text(
                    x, y, display_char(ch),
                    transform=ax.transAxes,
                    fontfamily="monospace",
                    fontsize=10,
                    color=color,
                    va="center",
                    ha="left",
                )
                x += x_step
        return

    if vocab is None or not color_by_vocab:
        ax.text(
            0.0, y, snippet,
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=10,
            color="0.15",
            va="center",
            ha="left",
        )
        return
    mask = _char_vocab_word_mask(snippet, vocab, spaced=spaced)
    for i, ch in enumerate(snippet):
        color = "#2ca02c" if mask[i] else "#d62728"
        ax.text(
            i * x_step, y, display_char(ch),
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=10,
            color=color,
            va="center",
            ha="left",
        )


def plot_sample_before_after(model, save_path: str) -> None:
    """Stochastic samples before/after training; fixed-length char snippets."""
    if "sample_before" not in model or "sample_after" not in model:
        print(f"skip {save_path}: re-run min-char-rnn.py to record samples")
        return

    vocab = set(map(str, model.get("vocab_words", [])))
    demo_snippet = str(model.get("demo_snippet", ""))[:SAMPLE_DISPLAY_LEN]
    demo_before = str(model.get("demo_before", "")) or str(model["sample_before"])
    demo_after = str(model.get("demo_after", "")) or str(model["sample_after"])
    demo_before = demo_before[:SAMPLE_DISPLAY_LEN]
    demo_after = demo_after[:SAMPLE_DISPLAY_LEN]
    spaced = " " in demo_snippet or " " in demo_after
    if model.get("model_config", {}).get("word_space") is False:
        spaced = False

    after_err = model.get("demo_word_error_frac")
    if after_err is None or not np.isfinite(after_err):
        after_err = _invalid_word_fraction(demo_after, vocab, spaced=spaced)
    after_title = f"Generated after learning — {100.0 * after_err:.1f}% invalid words"
    if "metric_word_error_frac" in model and len(model["metric_word_error_frac"]):
        train_err = float(model["metric_word_error_frac"][-1])
        after_title += f"; training metric: {100.0 * train_err:.1f}%"

    unspaced = bool(vocab) and not spaced
    rows: list[tuple[str, str, set[str] | None, bool, bool]] = []

    def _add_pair(
        base_title: str,
        snippet: str,
        word_vocab: set[str] | None,
        color_by_vocab: bool,
    ) -> None:
        if unspaced:
            rows.append((
                f"{base_title} — as model sees it",
                snippet, word_vocab, color_by_vocab, False,
            ))
            rows.append((
                f"{base_title} — segmented ({UNSPACED_WORD_SEP} = word boundary, viz only)",
                snippet, word_vocab, color_by_vocab, True,
            ))
        else:
            rows.append((base_title, snippet, word_vocab, color_by_vocab, True))

    _add_pair(
        f"Training corpus ({SAMPLE_DISPLAY_LEN} chars)",
        demo_snippet,
        vocab if unspaced else None,
        False,
    )
    _add_pair(
        f"Generated before learning ({SAMPLE_DISPLAY_LEN} chars) "
        "— green=in vocab, red=not",
        demo_before,
        vocab,
        True,
    )
    _add_pair(
        after_title + f" ({SAMPLE_DISPLAY_LEN} chars) — green=in vocab, red=not",
        demo_after,
        vocab,
        True,
    )

    row_h = 2.2 if unspaced else 3.6
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, row_h * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, (title, snippet, word_vocab, color_by_vocab, show_seps) in zip(axes, rows):
        ax.set_axis_off()
        ax.text(0.0, 0.92, title, transform=ax.transAxes, fontsize=10, va="top")
        _draw_sample_chars(
            ax, snippet, 0.35,
            vocab=word_vocab,
            spaced=spaced,
            color_by_vocab=color_by_vocab,
            show_word_separators=show_seps,
        )

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def trigram_sequence_colors(labels):
    """Stable color per unique 3-char context label (tab10, full saturation)."""
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    return {label: cmap(i) for i, label in enumerate(unique_labels)}


def _bold_categorical_colors(categories) -> dict:
    """High-contrast hex palette for text-only PCA labels."""
    palette = _DIVERGENT_WORD_PALETTE + _TRAJECTORY_STEP_PALETTE
    sorted_cats = sorted(categories, key=lambda x: (str(type(x)), str(x)))
    return {
        c: plt.matplotlib.colors.to_rgba(palette[i % len(palette)])
        for i, c in enumerate(sorted_cats)
    }


def _state_id_colors(state_ids: list[int]) -> dict[int, tuple]:
    unique = sorted(set(state_ids))
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    return {state: cmap(i) for i, state in enumerate(unique)}


def _dfa_automaton_state_colors(automaton: MinimizedVocabAutomaton) -> dict[int, tuple]:
    """Stable color for every minimized-DFA state (not only those in a viz window)."""
    all_states = sorted(automaton.dfa.states)
    cmap = plt.get_cmap("tab20", max(len(all_states), 1))
    return {state: cmap(i) for i, state in enumerate(all_states)}


def _dfa_state_ids_for_prefixes(
    prefixes: list[str],
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool,
) -> list[int]:
    return [dfa_state_for_prefix(p, automaton, spaced=spaced) for p in prefixes]


def _dfa_state_ids_at_timesteps(
    text: str,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool,
    words: list[str] | None = None,
) -> list[int]:
    vocab = _corpus_vocab(text, words)
    return [
        dfa_state_at_position(
            text, t, automaton, spaced=spaced, vocab=vocab,
        ) for t in range(len(text))
    ]


def _color_tick_labels_by_state_ids(
    ticks,
    state_ids: list[int],
    order: list[int] | None = None,
) -> None:
    """Color tick labels by min DFA state; order maps each tick to a state_ids index."""
    if not state_ids:
        return
    state_colors = _state_id_colors(state_ids)
    if order is None:
        order = list(range(min(len(ticks), len(state_ids))))
    for tick, idx in zip(ticks, order):
        if 0 <= idx < len(state_ids):
            tick.set_color(state_colors[state_ids[idx]])


def prefix_annotation_label(
    text: str, index: int, *, spaced: bool, words: list[str] | None = None,
) -> str:
    """Text on annotation boxes: in-word prefix at this timestep."""
    return word_subsequent_label(
        text, index, spaced=spaced, words=_resolve_words(text, words),
    )


def _layout_group_label_positions(
    projected, groups: dict[str, list[int]], *, fontsize: float = 9.0,
) -> dict[str, np.ndarray]:
    center = projected.mean(axis=0)
    plane = projected[:, :2]
    span = max(
        float(np.ptp(projected[:, 0])),
        float(np.ptp(projected[:, 1])),
        1e-3,
    )
    placed: list[tuple[np.ndarray, float]] = []
    label_positions: dict[str, np.ndarray] = {}

    items = list(groups.items())
    items.sort(key=lambda kv: float(np.linalg.norm(projected[kv[1]].mean(axis=0) - center)))

    for key, indices in items:
        points = projected[indices]
        centroid = points.mean(axis=0)
        centroid2 = centroid[:2]
        outward = centroid2 - center[:2]
        display = "␣" if key == " " else str(key)
        best2 = _minimal_leader_label_position(
            centroid2, outward, display,
            span=span,
            base_offset_frac=0.04,
            fontsize=fontsize,
            placed=placed,
            avoid_points=plane,
        )
        label_positions[key] = np.array([best2[0], best2[1]])
    return label_positions


def layout_prefix_labels(projected, prefix_labels: list[str]):
    """Label layout when each row already has a unique (or grouped) prefix label."""
    sequence_color = trigram_sequence_colors(prefix_labels)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(prefix_labels):
        by_label[label].append(i)
    label_positions = _layout_group_label_positions(projected, by_label)
    return prefix_labels, sequence_color, by_label, label_positions


def layout_trigram_labels(
    text, projected, *, spaced: bool = False, prefix_labels: list[str] | None = None,
):
    """Label positions and grouping for context annotations on PCA plots."""
    if prefix_labels is not None:
        return layout_prefix_labels(projected, prefix_labels)
    labels = [
        timestep_context_label(
            text, i, spaced=spaced, words=_resolve_words(text),
        ) for i in range(len(text))
    ]
    sequence_color = trigram_sequence_colors(labels)
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_sequence[label].append(i)
    label_positions = _layout_group_label_positions(projected, by_sequence)
    return labels, sequence_color, by_sequence, label_positions


CONTEXT_LABEL_FONTSIZE = 9


def _context_annotation_bbox(edge_color: str) -> dict:
    """Opaque label box readable on top of colored contour regions."""
    return dict(
        boxstyle="round,pad=0.22",
        facecolor="#ffffff",
        edgecolor=edge_color,
        linewidth=1.0,
        alpha=1.0,
    )


def _context_annotation_effects():
    """Thin outline so small labels stay legible without looking heavy."""
    return [
        path_effects.withStroke(linewidth=2.5, foreground="#ffffff"),
        path_effects.Normal(),
    ]


def _draw_annotation_groups(
    ax,
    projected,
    groups: dict,
    label_positions: dict,
    point_colors: list,
    label_text: dict,
    *,
    point_size: float = 40,
    label_fontsize: float = CONTEXT_LABEL_FONTSIZE,
    leader_linewidth: float = 0.55,
    leader_alpha: float = 0.42,
) -> list[tuple[float, float]]:
    """Scatter + leader lines + one label per group (shared by trigram / DFA modes)."""
    ax.scatter(
        projected[:, 0], projected[:, 1],
        s=point_size, c=point_colors, edgecolors="black", linewidths=0.5,
        zorder=6,
    )

    for key, indices in groups.items():
        text_pos = label_positions[key]
        color = point_colors[indices[0]]
        for point in projected[indices]:
            ax.plot(
                [text_pos[0], point[0]], [text_pos[1], point[1]],
                color="0.55", linewidth=leader_linewidth, alpha=leader_alpha,
                solid_capstyle="round", zorder=5,
            )
        ax.text(
            text_pos[0], text_pos[1], label_text[key],
            fontsize=label_fontsize, color="#1a1a1a",
            ha="center", va="center",
            bbox=_context_annotation_bbox(color),
            path_effects=_context_annotation_effects(),
            zorder=10,
        )

    return list(label_positions.values())


def _sorted_legend_categories(cat_to_color: dict) -> list:
    def sort_key(c):
        if isinstance(c, (int, np.integer)):
            return (0, int(c))
        if isinstance(c, str) and c.isdigit():
            return (0, int(c))
        return (1, str(c))

    return sorted(cat_to_color.keys(), key=sort_key)


def _dfa_compact_legend_label(state: int, automaton: MinimizedVocabAutomaton) -> str:
    prefixes = automaton.state_prefixes.get(int(state), set())
    if not prefixes:
        return f"q{state}"
    shown = sorted(prefixes, key=lambda p: (len(p), p))
    if len(shown) == 1:
        return shown[0]
    if len(shown) <= 4:
        return ", ".join(shown)
    return f"{shown[0]}, {shown[1]} …+{len(shown) - 2}"


def _add_feature_panel_legend(
    ax,
    cat_to_color: dict,
    legend_labels: dict,
    *,
    title: str | None = None,
    max_items: int = 24,
    outside: bool = False,
) -> None:
    from matplotlib.patches import Patch

    cats = _sorted_legend_categories(cat_to_color)
    truncated = len(cats) > max_items
    if truncated:
        cats = cats[:max_items]
    handles = [
        Patch(
            facecolor=cat_to_color[c],
            edgecolor="#333333",
            linewidth=0.4,
            label=legend_labels.get(c, str(c)),
        )
        for c in cats
    ]
    if truncated:
        handles.append(
            Patch(facecolor="none", edgecolor="none", label="…"),
        )
    ncol = 1
    if len(cats) > 10:
        ncol = 2
    elif len(cats) > 6:
        ncol = 1
    legend_kw: dict = {
        "handles": handles,
        "fontsize": 7,
        "title_fontsize": 8,
        "framealpha": 0.92,
        "ncol": ncol,
        "handlelength": 1.0,
        "handletextpad": 0.4,
        "borderpad": 0.35,
    }
    if title:
        legend_kw["title"] = title
    if outside:
        legend_kw.update(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        legend_kw.update(loc="upper right")
    ax.legend(**legend_kw)


def _add_dfa_state_color_legend(
    ax,
    automaton: MinimizedVocabAutomaton,
    state_colors: dict[int, tuple],
    *,
    outside: bool = False,
    compact: bool = False,
) -> None:
    handles = [
        Patch(
            facecolor=state_colors[state],
            edgecolor="#333333",
            linewidth=0.4,
            label=(
                _dfa_compact_legend_label(state, automaton)
                if compact
                else dfa_state_label(state, automaton)
            ),
        )
        for state in sorted(state_colors)
    ]
    legend_kw: dict = {
        "handles": handles,
        "title": "DFA state",
        "fontsize": 7,
        "title_fontsize": 8,
        "framealpha": 0.92,
        "handlelength": 1.0,
        "handletextpad": 0.4,
        "borderpad": 0.35,
    }
    if len(handles) > 8:
        legend_kw["ncol"] = 2
    if outside:
        legend_kw.update(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        legend_kw.update(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.legend(**legend_kw)


def add_dfa_state_annotations(
    ax,
    text,
    projected,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool,
    state_colors: dict[int, tuple] | None = None,
    show_legend: bool = False,
    point_size: float = 40,
    label_fontsize: float = CONTEXT_LABEL_FONTSIZE,
    leader_linewidth: float = 0.55,
    annot_style: str = "leaders",
    prefix_labels: list[str] | None = None,
):
    """Point color = min DFA state; annotation text = in-word prefix at timestep."""
    n = len(prefix_labels) if prefix_labels is not None else len(text)
    if prefix_labels is not None:
        prefixes = prefix_labels
        state_ids = [
            dfa_state_for_prefix(p, automaton, spaced=spaced) for p in prefixes
        ]
    else:
        state_ids = [
            dfa_state_at_position(
                text, i, automaton, spaced=spaced, vocab=_corpus_vocab(text),
            ) for i in range(n)
        ]
        prefixes = [
            prefix_annotation_label(text, i, spaced=spaced) for i in range(n)
        ]
    if state_colors is None:
        state_colors = _state_id_colors(state_ids)
    point_colors = [state_colors[s] for s in state_ids]

    by_prefix: dict[str, list[int]] = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        by_prefix[prefix].append(i)
    label_positions = _layout_group_label_positions(
        projected, by_prefix, fontsize=label_fontsize,
    )
    label_text = {p: ("␣" if p == " " else p) for p in by_prefix}

    annot_style = (annot_style or "leaders").lower()
    if annot_style == "none":
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=point_colors,
            s=point_size,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=4,
        )
        text_positions = []
    elif annot_style == "annots_only":
        # Put the prefix labels directly at each point (no box, no leader lines).
        fs = max(8, int(label_fontsize * 0.65))
        for i, prefix in enumerate(prefixes):
            label = "␣" if prefix == " " else prefix
            ax.text(
                projected[i, 0],
                projected[i, 1],
                label,
                fontsize=fs,
                color=point_colors[i],
                ha="center",
                va="center",
                zorder=10,
            )
        text_positions = projected.tolist()
    else:
        text_positions = _draw_annotation_groups(
            ax, projected, by_prefix, label_positions, point_colors, label_text,
            point_size=point_size,
            label_fontsize=label_fontsize,
            leader_linewidth=leader_linewidth,
        )
    if show_legend:
        _add_dfa_state_color_legend(ax, automaton, state_colors)
    return text_positions


def add_dfa_state_annotations_3d(
    ax,
    text,
    projected: np.ndarray,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool,
    state_colors: dict[int, tuple] | None = None,
    point_size: float = 50,
    leader_linewidth: float = 0.55,
    annot_style: str = "leaders",
    prefix_labels: list[str] | None = None,
) -> None:
    """3D scatter colored by DFA state with optional leader lines to prefix labels."""
    n = len(prefix_labels) if prefix_labels is not None else len(text)
    if prefix_labels is not None:
        prefixes = prefix_labels
        state_ids = [
            dfa_state_for_prefix(p, automaton, spaced=spaced) for p in prefixes
        ]
    else:
        state_ids = [
            dfa_state_at_position(
                text, i, automaton, spaced=spaced, vocab=_corpus_vocab(text),
            ) for i in range(n)
        ]
        prefixes = [
            prefix_annotation_label(text, i, spaced=spaced) for i in range(n)
        ]
    if state_colors is None:
        state_colors = _dfa_automaton_state_colors(automaton)
    point_colors = [state_colors[s] for s in state_ids]

    annot_style = (annot_style or "leaders").lower()
    if annot_style != "none":
        ax.scatter(
            projected[:, 0], projected[:, 1], projected[:, 2],
            s=point_size, c=point_colors, edgecolors="black", linewidths=0.4,
            depthshade=True, zorder=6,
        )
    if annot_style == "annots_only":
        fs = max(7, int(CONTEXT_LABEL_FONTSIZE * 0.65))
        for i, prefix in enumerate(prefixes):
            label = "␣" if prefix == " " else prefix
            ax.text(
                projected[i, 0], projected[i, 1], projected[i, 2],
                label, fontsize=fs, color=point_colors[i],
                ha="center", va="center", zorder=10,
            )
    elif annot_style == "leaders":
        _annotate_trajectory_labels(
            ax, projected, prefixes,
            label_colors=point_colors,
            dedupe=True,
            use_leaders=True,
            leader_linewidth=leader_linewidth,
            fontsize=8,
        )


def add_trigram_annotations(
    ax, text, projected, *, spaced: bool = False, prefix_labels: list[str] | None = None,
):
    """Context-colored points, leader lines, one label per context group."""
    labels, sequence_color, by_sequence, label_positions = layout_trigram_labels(
        text, projected, spaced=spaced, prefix_labels=prefix_labels,
    )
    label_text = {label: ("␣" if label == " " else label) for label in by_sequence}
    point_colors = [sequence_color[label] for label in labels]
    return _draw_annotation_groups(
        ax, projected, by_sequence, label_positions, point_colors, label_text
    )


def add_pca_point_annotations(
    ax,
    text,
    projected,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    show_dfa_legend: bool = False,
    annot_style: str = "leaders",
    prefix_labels: list[str] | None = None,
):
    if automaton is not None:
        return add_dfa_state_annotations(
            ax,
            text,
            projected,
            automaton,
            spaced=spaced,
            show_legend=show_dfa_legend,
            annot_style=annot_style,
            prefix_labels=prefix_labels,
        )
    return add_trigram_annotations(
        ax, text, projected, spaced=spaced, prefix_labels=prefix_labels,
    )


def _expand_limits_for_annotations(ax, projected, text_positions, base_xlim, base_ylim):
    """Union of grid limits and annotation positions."""
    all_x = [projected[:, 0].min(), projected[:, 0].max(), base_xlim[0], base_xlim[1]]
    all_y = [projected[:, 1].min(), projected[:, 1].max(), base_ylim[0], base_ylim[1]]
    if text_positions:
        all_x.extend(p[0] for p in text_positions)
        all_y.extend(p[1] for p in text_positions)
    x_pad = max((max(all_x) - min(all_x)) * 0.06, 1e-3)
    y_pad = max((max(all_y) - min(all_y)) * 0.06, 1e-3)
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)


def plot_2d_hidden_state_labels(
    text,
    hidden_states,
    chars,
    projected,
    save_path,
    title,
    xlabel,
    ylabel,
    fig_suptitle=None,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
):
    """Scatter points with one context label per group, lines to its points."""
    _ = chars
    if len(text) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    text_positions = add_pca_point_annotations(
        ax, text, projected, spaced=spaced, automaton=automaton
    )
    _expand_limits_for_annotations(
        ax, projected, text_positions,
        (projected[:, 0].min(), projected[:, 0].max()),
        (projected[:, 1].min(), projected[:, 1].max()),
    )

    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)
    if fig_suptitle:
        fig.suptitle(fig_suptitle, fontsize=11, y=1.02)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"wrote {save_path}")


def _plot_2d_hidden_state_labels_on_ax(
    ax,
    text: str,
    projected: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    annot_style: str = "leaders",
    prefix_labels: list[str] | None = None,
) -> None:
    n = len(prefix_labels) if prefix_labels is not None else len(text)
    if n == 0:
        return
    text_positions = add_pca_point_annotations(
        ax,
        text,
        projected,
        spaced=spaced,
        automaton=automaton,
        annot_style=annot_style,
        prefix_labels=prefix_labels,
    )
    _expand_limits_for_annotations(
        ax, projected, text_positions,
        (projected[:, 0].min(), projected[:, 0].max()),
        (projected[:, 1].min(), projected[:, 1].max()),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)


def _feature_point_colors_for_timesteps(
    feat: str,
    timestep_labels,
    automaton: MinimizedVocabAutomaton | None,
    n: int,
    *,
    bold: bool = False,
) -> tuple[list, dict, dict]:
    """Per-timestep RGBA colors and category maps for one analysis feature."""
    from unit_selectivity import _panel_feature_colors

    vals, mask = timestep_labels.feature_values(feat)
    visible = [j for j in range(n) if mask is None or mask[j]]
    cmap = plt.get_cmap("Set1") if bold else plt.cm.tab20
    if feat == "dfa" and automaton is not None:
        visible_states = sorted(set(int(vals[j]) for j in visible))
        if bold:
            cat_to_color = _bold_categorical_colors(visible_states)
        else:
            state_colors = _dfa_automaton_state_colors(automaton)
            cat_to_color = {int(s): state_colors[int(s)] for s in visible_states}
        from vocab_diagrams import dfa_state_label

        legend_labels = {c: dfa_state_label(int(c), automaton) for c in cat_to_color}
    else:
        if bold:
            visible_vals = sorted(set(vals[j] for j in visible), key=lambda x: (str(type(x)), str(x)))
            cat_to_color = _bold_categorical_colors(visible_vals)
            legend_labels = {c: str(c) for c in cat_to_color}
        else:
            cat_to_color, legend_labels, _ = _panel_feature_colors(
                feat, vals, visible, automaton, cmap,
            )
    point_colors: list = []
    for j in range(n):
        if mask is not None and not mask[j]:
            point_colors.append("#bbbbbb")
        elif feat == "dfa" and automaton is not None:
            point_colors.append(cat_to_color[int(vals[j])])
        else:
            point_colors.append(cat_to_color[vals[j]])
    return point_colors, cat_to_color, legend_labels


def _add_panel_color_legend(
    ax,
    feat: str,
    cat_to_color: dict,
    legend_labels: dict,
    automaton: MinimizedVocabAutomaton | None,
    *,
    feature_title: str,
    outside: bool = False,
) -> None:
    if feat == "dfa" and automaton is not None:
        _add_dfa_state_color_legend(
            ax, automaton,
            {int(c): cat_to_color[c] for c in cat_to_color},
            outside=outside,
            compact=True,
        )
    else:
        _add_feature_panel_legend(
            ax, cat_to_color, legend_labels,
            title=feature_title,
            outside=outside,
        )


def _annotate_feature_panel_within_axes(
    ax,
    projected: np.ndarray,
    labels: list[str],
    colors: list,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    fontsize: float = 7.0,
    max_labels: int = 24,
) -> None:
    """Deduplicated prefix labels at cluster centroids, clamped inside axis limits."""
    from collections import defaultdict

    buckets: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label:
            buckets[label].append(i)
    if not buckets:
        return

    xlo, xhi = xlim
    ylo, yhi = ylim
    margin_x = max((xhi - xlo) * 0.04, 1e-4)
    margin_y = max((yhi - ylo) * 0.04, 1e-4)

    def _clamp_xy(xy: np.ndarray) -> tuple[float, float]:
        return (
            float(np.clip(xy[0], xlo + margin_x, xhi - margin_x)),
            float(np.clip(xy[1], ylo + margin_y, yhi - margin_y)),
        )

    items = sorted(buckets.items(), key=lambda kv: kv[1][0])
    if len(items) > max_labels:
        items = items[:max_labels]

    for label, idxs in items:
        centroid = projected[idxs].mean(axis=0)
        x, y = _clamp_xy(centroid)
        ax.text(
            x, y, label,
            fontsize=fontsize,
            color=colors[idxs[0]],
            ha="center",
            va="center",
            zorder=5,
            clip_on=True,
        )


def _plot_2d_feature_colored_pca_panel(
    ax,
    projected: np.ndarray,
    prefix_labels: list[str],
    feat: str,
    timestep_labels,
    automaton: MinimizedVocabAutomaton | None,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    annot_style: str = "leaders",
    show_legend: bool = True,
    minimal_axes: bool = False,
    path_indices: list[list[int]] | None = None,
    faint_paths: bool = False,
) -> None:
    n = projected.shape[0]
    display_prefixes = ["␣" if p == " " else p for p in prefix_labels]
    annot_style = (annot_style or "compact").lower()
    text_only = annot_style == "annots_only"

    point_colors, cat_to_color, legend_labels = _feature_point_colors_for_timesteps(
        feat, timestep_labels, automaton, n, bold=text_only,
    )

    if faint_paths and path_indices:
        for idxs in path_indices:
            if len(idxs) < 2:
                continue
            _plot_label_gradient_path(
                ax, projected[idxs], idxs, point_colors,
                linewidth=0.85, alpha=0.50, zorder=1,
                subdivisions=10, arrow_mutation_scale=7.0,
            )

    if not text_only:
        ax.scatter(
            projected[:, 0], projected[:, 1],
            s=70, c=point_colors, edgecolors="black", linewidths=0.35, zorder=3,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if annot_style == "compact":
        _annotate_feature_panel_within_axes(
            ax, projected, display_prefixes, point_colors,
            xlim=xlim, ylim=ylim, fontsize=7,
        )
    elif annot_style == "annots_only":
        label_stroke = path_effects.withStroke(linewidth=0.6, foreground="black")
        for xy, prefix, color in zip(projected, display_prefixes, point_colors):
            ax.text(
                xy[0], xy[1], prefix,
                fontsize=4, color=color, fontweight="bold",
                ha="center", va="center", zorder=5, clip_on=True,
                path_effects=[label_stroke, path_effects.Normal()],
            )
    elif annot_style == "leaders":
        _annotate_trajectory_labels(
            ax, projected, display_prefixes,
            label_colors=point_colors,
            dedupe=True,
            use_leaders=True,
            leader_linewidth=0.45,
            fontsize=7,
            label_offset_frac=0.04,
            line_clearance_frac=0.03,
        )
        pad_x = (xlim[1] - xlim[0]) * 0.04
        pad_y = (ylim[1] - ylim[0]) * 0.04
        ax.set_xlim(xlim[0] - pad_x, xlim[1] + pad_x)
        ax.set_ylim(ylim[0] - pad_y, ylim[1] + pad_y)
    ax.set_xlabel("" if minimal_axes else xlabel, fontsize=8)
    ax.set_ylabel("" if minimal_axes else ylabel, fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if minimal_axes:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    else:
        ax.tick_params(top=False, right=False)
    if show_legend:
        _add_panel_color_legend(
            ax, feat, cat_to_color, legend_labels, automaton,
            feature_title=title,
            outside=False,
        )


def _plot_3d_feature_colored_pca_panel(
    ax,
    projected: np.ndarray,
    prefix_labels: list[str],
    feat: str,
    timestep_labels,
    automaton: MinimizedVocabAutomaton | None,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    annot_style: str = "leaders",
) -> None:
    n = projected.shape[0]
    point_colors, cat_to_color, legend_labels = _feature_point_colors_for_timesteps(
        feat, timestep_labels, automaton, n,
    )
    display_prefixes = ["␣" if p == " " else p for p in prefix_labels]
    annot_style = (annot_style or "none").lower()
    if annot_style != "annots_only":
        ax.scatter(
            projected[:, 0], projected[:, 1], projected[:, 2],
            s=55, c=point_colors, edgecolors="black", linewidths=0.35,
            depthshade=True, zorder=6,
        )
    if annot_style == "leaders":
        _annotate_trajectory_labels(
            ax, projected, display_prefixes,
            label_colors=point_colors,
            dedupe=True,
            use_leaders=True,
            leader_linewidth=0.45,
            fontsize=7,
            label_offset_frac=0.04,
        )
    elif annot_style == "annots_only":
        for i, prefix in enumerate(display_prefixes):
            ax.text(
                projected[i, 0], projected[i, 1], projected[i, 2],
                prefix, fontsize=7, color=point_colors[i],
                ha="center", va="center", zorder=10,
            )
    ax.set_xlabel(xlabel, fontsize=8, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=6)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=6)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(True, linestyle=":", alpha=0.35)
    _add_panel_color_legend(
        ax, feat, cat_to_color, legend_labels, automaton,
        feature_title=title,
        outside=True,
    )


def _feature_pca_suptitle(
    *,
    repr_name: str | None,
    words: list[str] | None,
    chars,
    condensed: CondensedView | None,
    dim_label: str = "PCA",
) -> str:
    """Compact figure title: suptitle above per-panel feature titles."""
    if words:
        vocab_part = f"{len(words)}-word vocabulary ({', '.join(words)})"
    else:
        vocab_part = original_vocabulary_title(chars)
    if repr_name:
        base = f"{repr_name} · {dim_label} · {vocab_part}"
    else:
        base = f"{dim_label} of hidden states · {vocab_part}"
    return _condensed_plot_title(base, condensed)


def _feature_pca_figure_gridspec(
    fig: plt.Figure,
    nrows: int,
    ncols: int,
    *,
    projection: str | None = None,
) -> list:
    """Subplot grid with headroom for a figure-level suptitle above panel titles."""
    gs = fig.add_gridspec(
        nrows, ncols,
        top=0.90,
        bottom=0.06,
        left=0.05,
        right=0.80 if projection == "3d" else 0.96,
        wspace=0.18,
        hspace=0.30,
    )
    axes = []
    for i in range(nrows * ncols):
        if projection == "3d":
            axes.append(fig.add_subplot(gs[i // ncols, i % ncols], projection="3d"))
        else:
            axes.append(fig.add_subplot(gs[i // ncols, i % ncols]))
    return axes


def _pca_feature_panel_context(
    text: str,
    hidden_states: np.ndarray,
    *,
    spaced: bool,
    automaton: MinimizedVocabAutomaton | None,
    words: list[str] | None,
    label_words: list[str] | None,
    condensed: CondensedView | None,
):
    """Shared PCA projection + timestep labels for feature-colored state panels."""
    from unit_selectivity import ANALYSIS_FEATURES, FEATURE_DISPLAY, build_timestep_labels

    if automaton is None:
        return None
    if condensed is not None:
        hidden_states = condensed.hidden_states
        spaced = condensed.spaced
    n = hidden_states.shape[0]
    if n < 1:
        return None
    timestep_labels = build_timestep_labels(
        text, automaton,
        spaced=spaced, words=words, label_words=label_words, condensed=condensed,
    )
    prefix_labels = timestep_labels.string
    return {
        "hidden_states": hidden_states,
        "n": n,
        "timestep_labels": timestep_labels,
        "prefix_labels": prefix_labels,
        "features": ANALYSIS_FEATURES,
        "feature_display": FEATURE_DISPLAY,
        "spaced": spaced,
    }


def plot_dimred_context_panels(
    text: str,
    hidden_states: np.ndarray,
    chars,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    annot_style: str = "leaders",
    condensed: CondensedView | None = None,
    words: list[str] | None = None,
    label_words: list[str] | None = None,
    embed_method: str = "pca",
) -> None:
    """2×2 embedding panels colored by char, position, position-from-end, and DFA state."""
    _ = chars
    ctx = _pca_feature_panel_context(
        text, hidden_states,
        spaced=spaced, automaton=automaton, words=words,
        label_words=label_words, condensed=condensed,
    )
    if ctx is None:
        print(f"skip {save_path}: need automaton and timesteps for feature-colored PCA")
        return

    hidden_states = ctx["hidden_states"]
    trajs = _embed_trajectories_for_text(
        text, hidden_states, spaced=ctx["spaced"], words=words,
    )
    pca_xy, _, _, evr = fit_embed_2d_with_evr(
        hidden_states, method=embed_method, trajectories=trajs,
    )
    xlabel, ylabel = embed_axis_labels_2d(evr, embed_method)
    pad_x = max((pca_xy[:, 0].max() - pca_xy[:, 0].min()) * 0.12, 0.08)
    pad_y = max((pca_xy[:, 1].max() - pca_xy[:, 1].min()) * 0.12, 0.08)
    xlim = (float(pca_xy[:, 0].min() - pad_x), float(pca_xy[:, 0].max() + pad_x))
    ylim = (float(pca_xy[:, 1].min() - pad_y), float(pca_xy[:, 1].max() + pad_y))

    fig = plt.figure(figsize=(22, 19))
    axes = _feature_pca_figure_gridspec(fig, 2, 2)
    for ax, feat in zip(axes, ctx["features"]):
        _plot_2d_feature_colored_pca_panel(
            ax, pca_xy, ctx["prefix_labels"], feat, ctx["timestep_labels"], automaton,
            title=ctx["feature_display"].get(feat, feat),
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            annot_style=annot_style,
        )

    fig.suptitle(
        _feature_pca_suptitle(
            repr_name=None, words=words, chars=chars, condensed=condensed,
            dim_label=embed_dim_label(embed_method),
        ),
        fontsize=13,
        y=0.97,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _pca_axis_labels(evr: np.ndarray) -> tuple[str, str, str]:
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    pc3 = 100.0 * float(evr[2]) if len(evr) > 2 else 0.0
    return (
        f"PC1 ({pc1:.1f}%)",
        f"PC2 ({pc2:.1f}%)",
        f"PC3 ({pc3:.1f}%)",
    )


def _dfa_point_colors_for_pca(
    text: str,
    *,
    n: int,
    spaced: bool,
    automaton: MinimizedVocabAutomaton,
    prefix_labels: list[str] | None,
) -> tuple[list[tuple], list[str], dict[int, tuple], list[int]]:
    if prefix_labels is not None:
        prefixes = prefix_labels
        state_ids = [
            dfa_state_for_prefix(p, automaton, spaced=spaced) for p in prefixes
        ]
    else:
        state_ids = [
            dfa_state_at_position(
                text, i, automaton, spaced=spaced, vocab=_corpus_vocab(text),
            ) for i in range(n)
        ]
        prefixes = [
            prefix_annotation_label(text, i, spaced=spaced) for i in range(n)
        ]
    state_colors = _dfa_automaton_state_colors(automaton)
    point_colors = [state_colors[s] for s in state_ids]
    return point_colors, prefixes, state_colors, state_ids


def _plot_3d_pca_scatter_with_labels(
    ax,
    projected: np.ndarray,
    labels: list[str],
    *,
    point_colors: list[tuple] | None = None,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
) -> None:
    colors = point_colors if point_colors is not None else ["C0"] * len(labels)
    ax.scatter(
        projected[:, 0], projected[:, 1], projected[:, 2],
        s=50, c=colors, edgecolors="black", linewidths=0.4, depthshade=True,
    )
    for i, label in enumerate(labels):
        disp = "␣" if label == " " else label
        ax.text(
            projected[i, 0], projected[i, 1], projected[i, 2],
            disp, fontsize=7, color="#1a1a1a",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)


def plot_dimred_context_panels_3d(
    text: str,
    hidden_states: np.ndarray,
    chars,
    save_path: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    condensed: CondensedView | None = None,
    repr_name: str = "hidden state",
    annot_style: str = "leaders",
    words: list[str] | None = None,
    label_words: list[str] | None = None,
    embed_method: str = "pca",
) -> None:
    """2×2 3D embedding panels colored by char, position, position-from-end, and DFA state."""
    _ = chars
    ctx = _pca_feature_panel_context(
        text, hidden_states,
        spaced=spaced, automaton=automaton, words=words,
        label_words=label_words, condensed=condensed,
    )
    if ctx is None:
        print(f"skip {save_path}: need automaton and timesteps for feature-colored PCA")
        return
    if ctx["n"] < 2 or ctx["hidden_states"].shape[1] < 2:
        return

    trajs = _embed_trajectories_for_text(
        text, ctx["hidden_states"], spaced=ctx["spaced"], words=words,
    )
    pca_xyz, _, _, evr = fit_embed_3d_with_evr(
        ctx["hidden_states"], method=embed_method, trajectories=trajs,
    )
    xlabel, ylabel, zlabel = embed_axis_labels_3d(evr, embed_method)

    fig = plt.figure(figsize=(24, 21))
    axes = _feature_pca_figure_gridspec(fig, 2, 2, projection="3d")
    for ax, feat in zip(axes, ctx["features"]):
        _plot_3d_feature_colored_pca_panel(
            ax, pca_xyz, ctx["prefix_labels"], feat, ctx["timestep_labels"], automaton,
            title=ctx["feature_display"].get(feat, feat),
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            annot_style=annot_style,
        )

    fig.suptitle(
        _feature_pca_suptitle(
            repr_name=repr_name, words=words, chars=chars, condensed=condensed,
            dim_label=f"3D {embed_dim_label(embed_method)}",
        ),
        fontsize=13,
        y=0.97,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_per_char_hidden_state_heatmaps(
    text,
    hidden_states,
    chars,
    save_path,
    cluster_rows=True,
    *,
    spaced: bool = False,
    condensed: CondensedView | None = None,
    automaton: MinimizedVocabAutomaton | None = None,
    repr_label: str = "hidden state",
    dim_label: str = "hidden unit",
):
    """Combined per-input-char heatmaps, rows = dims, columns = occurrences."""
    hidden_size = hidden_states.shape[1]
    groups: list[tuple] = []

    if condensed is not None:
        hidden_states = condensed.hidden_states
        spaced = condensed.spaced
        for char in chars:
            indices = [i for i, ch in enumerate(condensed.input_chars) if ch == char]
            if len(indices) < 2:
                continue
            rows = hidden_states[indices]
            prefix_keys = [condensed.labels[i] for i in indices]
            labels = [_display_prefix_label(l) for l in prefix_keys]
            title_suffix = "condensed by prefix"
            if cluster_rows and len(indices) > 2:
                order = average_linkage_cluster_order(rows)
                rows = rows[order]
                labels = [labels[i] for i in order]
                prefix_keys = [prefix_keys[i] for i in order]
                title_suffix += ", clustered"
            groups.append((char, rows, labels, title_suffix, prefix_keys))
    else:
        for char in chars:
            indices = np.array([i for i, text_char in enumerate(text) if i > 0 and text_char == char])
            if len(indices) < 2:
                continue

            rows = hidden_states[indices]
            labels = [context_label(text, int(i), spaced=spaced) for i in indices]
            prefix_keys = [
                word_subsequent_label(text, int(i), spaced=spaced) for i in indices
            ]

            if cluster_rows and len(indices) > 2:
                order = average_linkage_cluster_order(rows)
                rows = rows[order]
                labels = [labels[i] for i in order]
                prefix_keys = [prefix_keys[i] for i in order]
                title_suffix = "clustered by vector similarity"
            else:
                title_suffix = "in sequence order"

            groups.append((char, rows, labels, title_suffix, prefix_keys))

    if not groups:
        return

    n = len(groups)
    ncols = max(2, min(n, math.ceil(math.sqrt(n * 2))))
    nrows = math.ceil(n / ncols)
    max_panel_cols = max(len(labels) for _, _, labels, _, _ in groups)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(
            max(12, ncols * max_panel_cols * 0.28),
            max(3, nrows * max(2.1, hidden_size * 0.24)),
        ),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None

    for idx, (char, rows, labels, title_suffix, prefix_keys) in enumerate(groups):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.imshow(
            rows.T,
            aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
            interpolation="nearest", origin="lower",
        )
        last_image = im

        ax.set_yticks(range(hidden_size))
        ax.set_yticklabels([f"{dim_label}{i}" for i in range(hidden_size)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        if automaton is not None:
            state_ids = _dfa_state_ids_for_prefixes(
                prefix_keys, automaton, spaced=spaced,
            )
            _color_tick_labels_by_state_ids(ax.get_xticklabels(), state_ids)
        ax.set_ylabel(dim_label)
        ax.set_title(
            f"{repr_label} for input {display_char(char)!r} "
            f"({len(labels)} occurrences, {title_suffix})",
            fontsize=9,
        )
        if idx // ncols == nrows - 1:
            xlabel = f"{prefix_axis_label(spaced=spaced, text=text)} @ timestep"
            if automaton is not None:
                xlabel += " · tick color = min DFA state"
            ax.set_xlabel(xlabel)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle(
        _condensed_plot_title(
            f"{repr_label} by input character · {original_vocabulary_title(chars, text)}",
            condensed,
        ),
        y=0.995,
    )
    fig.colorbar(last_image, ax=axes, fraction=0.02, pad=0.01, label="activation")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def trigram_avoidance_points(
    text,
    projected,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    prefix_labels: list[str] | None = None,
):
    """PC coordinates to keep region letters away from (scatter + label boxes)."""
    if prefix_labels is not None:
        by_prefix: dict[str, list[int]] = defaultdict(list)
        for i, prefix in enumerate(prefix_labels):
            by_prefix[prefix].append(i)
        label_positions = _layout_group_label_positions(projected, by_prefix)
    elif automaton is not None:
        prefixes = [prefix_annotation_label(text, i, spaced=spaced) for i in range(len(text))]
        by_prefix = defaultdict(list)
        for i, prefix in enumerate(prefixes):
            by_prefix[prefix].append(i)
        label_positions = _layout_group_label_positions(projected, by_prefix)
    else:
        _, _, _, label_positions = layout_trigram_labels(text, projected, spaced=spaced)
    blocks = [projected]
    if label_positions:
        blocks.append(np.array(list(label_positions.values())))
    return np.vstack(blocks)


def region_interior_point(
    grid_x, grid_y, class_mask,
    avoid_xy=None, avoid_radius=0.0,
    xlim=None, ylim=None, edge_margin_frac=0.08,
    min_area_frac=0.02, erosion_iters=4,
):
    """Point deep inside the largest argmax blob, away from labels and plot edges."""
    if not class_mask.any():
        return None

    labeled, num_features = ndimage.label(class_mask)
    if num_features == 0:
        return None

    component_sizes = ndimage.sum(
        class_mask, labeled, index=np.arange(1, num_features + 1),
    )
    largest_label = 1 + int(np.argmax(component_sizes))
    component = labeled == largest_label

    if component.sum() < min_area_frac * class_mask.size:
        return None

    interior = component
    for _ in range(erosion_iters):
        shrunk = ndimage.binary_erosion(interior)
        if shrunk.any():
            interior = shrunk

    depth = ndimage.distance_transform_edt(interior)
    if depth.max() < 1.0:
        return None

    rows, cols = np.where(interior)
    depth_vals = depth[rows, cols]
    gx = grid_x[rows, cols]
    gy = grid_y[rows, cols]

    if avoid_xy is not None and len(avoid_xy):
        avoid = np.asarray(avoid_xy, dtype=float)
        diff = np.stack([gx, gy], axis=1)[:, None, :] - avoid[None, :, :]
        clearance = np.linalg.norm(diff, axis=2).min(axis=1) - avoid_radius
        clearance = np.maximum(clearance, 0.0)
    else:
        clearance = np.ones(len(rows), dtype=float)

    if xlim is not None and ylim is not None:
        plane_span = max(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]), 1e-3)
        margin = plane_span * edge_margin_frac
        edge_clear = np.minimum(
            np.minimum(gx - xlim[0], xlim[1] - gx),
            np.minimum(gy - ylim[0], ylim[1] - gy),
        ) - margin
        edge_clear = np.maximum(edge_clear, 0.0)
    else:
        edge_clear = np.ones(len(rows), dtype=float)

    score = depth_vals * np.sqrt(clearance + 1e-6) * edge_clear
    if score.max() <= 0:
        return None

    best = int(np.argmax(score))
    row, col = rows[best], cols[best]
    return float(grid_x[row, col]), float(grid_y[row, col])


def add_argmax_region_labels(
    ax, grid_x, grid_y, grid_pred, chars,
    avoid_xy=None, avoid_radius=0.0, xlim=None, ylim=None,
):
    """Large white letter at the interior of each argmax region."""
    stroke = path_effects.withStroke(linewidth=2.5, foreground="#1a1a1a")
    for index, char in enumerate(chars):
        mask = grid_pred == index
        if not mask.any():
            continue
        position = region_interior_point(
            grid_x, grid_y, mask,
            avoid_xy=avoid_xy, avoid_radius=avoid_radius,
            xlim=xlim, ylim=ylim,
        )
        if position is None:
            continue
        glyph = argmax_region_glyph(char)
        if glyph is None:
            continue
        ax.text(
            position[0], position[1], glyph,
            fontsize=26, color="white",
            ha="center", va="center", zorder=8,
            path_effects=[
                path_effects.withStroke(linewidth=3.5, foreground="#ffffff"),
                stroke,
            ],
        )


def plot_pca_context_labels(
    text,
    hidden_states,
    chars,
    save_path,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    condensed: CondensedView | None = None,
    words: list[str] | None = None,
    label_words: list[str] | None = None,
    annot_style: str = "leaders",
    embed_method: str = "pca",
):
    """4-panel 2D embedding colored by analysis features."""
    plot_dimred_context_panels(
        text,
        hidden_states,
        chars,
        save_path,
        spaced=spaced,
        automaton=automaton,
        annot_style=annot_style,
        condensed=condensed,
        words=words,
        label_words=label_words,
        embed_method=embed_method,
    )


def plot_pca_context_labels_3d(
    text,
    hidden_states,
    chars,
    save_path,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    condensed: CondensedView | None = None,
    repr_name: str = "hidden state",
    annot_style: str = "leaders",
    words: list[str] | None = None,
    label_words: list[str] | None = None,
    embed_method: str = "pca",
):
    """4-panel 3D embedding colored by analysis features."""
    plot_dimred_context_panels_3d(
        text,
        hidden_states,
        chars,
        save_path,
        spaced=spaced,
        automaton=automaton,
        condensed=condensed,
        repr_name=repr_name,
        annot_style=annot_style,
        words=words,
        label_words=label_words,
        embed_method=embed_method,
    )


def _word_trajectory_colors(segments: list[tuple[int, int, str]]) -> dict[str, tuple]:
    """Stable color per distinct word label across space-to-space segments."""
    words = sorted({segment_word_label(seg) for _, _, seg in segments})
    return _vocab_word_colors(words)


def _step_palette_rgba(step: int) -> tuple:
    """1-indexed in-word step → discrete divergent color."""
    idx = max(0, min(step - 1, len(_TRAJECTORY_STEP_PALETTE) - 1))
    return plt.matplotlib.colors.to_rgba(_TRAJECTORY_STEP_PALETTE[idx])


def _step_path_colors(n_points: int, cmap_name: str = "viridis") -> list:
    """Discrete divergent colors per step (step 1 = first palette color)."""
    del cmap_name  # kept for call-site compatibility
    if n_points <= 1:
        return [_step_palette_rgba(1)]
    return [_step_palette_rgba(i + 1) for i in range(n_points)]


def _plot_label_gradient_path(
    ax,
    path: np.ndarray,
    idxs: list[int],
    point_colors: list,
    *,
    linewidth: float = 0.75,
    alpha: float = 0.38,
    zorder: int = 1,
    subdivisions: int = 8,
    arrow_mutation_scale: float = 7.0,
) -> None:
    """Draw a path with each edge shaded from its start label color to its end label color."""
    if len(path) < 2:
        return
    mcolors = plt.matplotlib.colors
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        c0 = np.array(mcolors.to_rgb(point_colors[idxs[i]]))
        c1 = np.array(mcolors.to_rgb(point_colors[idxs[i + 1]]))
        ts = np.linspace(0.0, 1.0, subdivisions + 1)
        for k in range(subdivisions):
            t_a, t_b = ts[k], ts[k + 1]
            qa = p0 * (1.0 - t_a) + p1 * t_a
            qb = p0 * (1.0 - t_b) + p1 * t_b
            t_mid = 0.5 * (t_a + t_b)
            col = tuple(c0 * (1.0 - t_mid) + c1 * t_mid)
            ax.plot(
                [qa[0], qb[0]], [qa[1], qb[1]],
                color=col, linewidth=linewidth, alpha=alpha,
                solid_capstyle="round", zorder=zorder,
            )
        _midsegment_arrowhead_2d(
            ax, p0, p1,
            color=tuple(0.5 * (c0 + c1)),
            alpha=min(alpha + 0.14, 1.0),
            zorder=zorder + 1,
            mutation_scale=arrow_mutation_scale,
        )


# Trajectory word palette (user-specified).
_TRAJECTORY_WORD_PALETTE: tuple[str, ...] = (
    "#000000",  # black
    "#0066CC",  # blue
    "#DE2D26",  # red
    "#228B22",  # green
    "#FF69B4",  # hotpink
)
# Paul Tol–style qualitative palette for larger vocabs (maximally distinct hues).
_DIVERGENT_WORD_PALETTE: tuple[str, ...] = (
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#EE7733", "#0077BB", "#33BBEE", "#EE3377",
    "#CC3311", "#009988", "#BBBB44", "#AA4499", "#44AA77", "#882255",
)
# Discrete in-word step colors (high contrast; shared by bottom trajectory row).
_TRAJECTORY_STEP_PALETTE: tuple[str, ...] = (
    "#2166AC",  # blue
    "#D6604D",  # coral
    "#1A9641",  # green
    "#E08214",  # orange
    "#762A83",  # purple
)
_TRAJECTORY_RETURN_COLOR = "#9A9A9A"  # gray — space / return-to-word-start


def _is_return_to_baseline_segment(
    prev_label: str,
    next_label: str,
    *,
    word_start: bool,
) -> bool:
    """True when a segment ends at baseline or crosses a completed-word boundary."""
    if next_label in ("", "␣"):
        return True
    if word_start and prev_label not in ("", "␣"):
        return True
    return False


def _vocab_word_colors(words: list[str]) -> dict[str, tuple]:
    """Distinct color per vocabulary word."""
    unique = sorted(set(words))
    if len(unique) <= len(_TRAJECTORY_WORD_PALETTE):
        palette = [
            plt.matplotlib.colors.to_rgba(c) for c in _TRAJECTORY_WORD_PALETTE[: len(unique)]
        ]
    elif len(unique) <= len(_DIVERGENT_WORD_PALETTE):
        palette = [
            plt.matplotlib.colors.to_rgba(c)
            for c in _DIVERGENT_WORD_PALETTE[: len(unique)]
        ]
    else:
        cmap = plt.get_cmap("tab20", len(unique))
        palette = [cmap(i) for i in range(len(unique))]
    colors = {word: palette[i] for i, word in enumerate(unique)}
    colors["␣"] = plt.matplotlib.colors.to_rgba(_TRAJECTORY_RETURN_COLOR)
    colors["?"] = plt.matplotlib.colors.to_rgba(_TRAJECTORY_RETURN_COLOR)
    return colors


def _trajectory_word_color(word: str, word_colors: dict[str, tuple]) -> tuple:
    return word_colors.get(word, word_colors["?"])


def _add_trajectory_word_legend(
    fig,
    word_colors: dict[str, tuple],
    *,
    bbox_to_anchor: tuple[float, float] = (0.99, 0.5),
    include_mean: bool = True,
) -> None:
    """Word-color legend outside the plot area (replaces per-step colorbar)."""
    words = [w for w in sorted(word_colors) if w not in ("␣", "?")]
    handles = []
    if include_mean:
        handles.append(
            plt.Line2D(
                [0], [0], color=_MEAN_TRAJECTORY_COLOR, linewidth=1.1, label="mean trajectory",
            )
        )
    handles.extend(
        plt.Line2D([0], [0], color=word_colors[w], linewidth=2.2, label=w) for w in words
    )
    if not handles:
        return
    fig.legend(
        handles=handles,
        title="word",
        loc="center left",
        bbox_to_anchor=bbox_to_anchor,
        fontsize=7,
        title_fontsize=8,
        framealpha=0.92,
        ncol=1 if len(handles) <= 11 else 2,
    )


def _midsegment_arrowhead_2d(
    ax,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    color,
    alpha: float,
    zorder: int,
    mutation_scale: float = 12.0,
    head_frac: float = 0.22,
) -> None:
    """Arrowhead only, centered on the segment midpoint (no extra shaft)."""
    d = p1 - p0
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        return
    u = d / norm
    mid = 0.5 * (p0 + p1)
    head_len = norm * head_frac
    ax.annotate(
        "",
        xy=(float(mid[0]), float(mid[1])),
        xytext=(float(mid[0] - head_len * u[0]), float(mid[1] - head_len * u[1])),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=0,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0,
            alpha=alpha,
        ),
        zorder=zorder,
    )


def _midsegment_arrowhead_3d(
    ax,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    color,
    alpha: float,
    mutation_scale: float = 12.0,
    head_frac: float = 0.22,
) -> None:
    """Arrowhead only at segment midpoint (3D FancyArrowPatch projection)."""
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class _Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return float(np.min(zs))

    d = p1 - p0
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        return
    u = d / norm
    mid = 0.5 * (p0 + p1)
    head_len = norm * head_frac
    tail = mid - head_len * u
    arrow = _Arrow3D(
        [float(tail[0]), float(mid[0])],
        [float(tail[1]), float(mid[1])],
        [float(tail[2]), float(mid[2])],
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        color=color,
        lw=0,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(arrow)


def _plot_step_colored_path_arrows(
    ax,
    path: np.ndarray,
    *,
    linewidth: float = 1.6,
    alpha: float = 0.55,
    zorder: int = 2,
    cmap_name: str = "viridis",
    segment_colors: list | None = None,
    segment_linestyles: list[str] | None = None,
    segment_alphas: list[float] | None = None,
    arrow_mutation_scale: float = 12.0,
    draw_arrows: bool = True,
) -> None:
    """Trajectory lines with arrowheads centered on each segment."""
    if len(path) < 2:
        return
    if segment_colors is None:
        segment_colors = _step_path_colors(len(path), cmap_name)
    for i in range(len(path) - 1):
        color = segment_colors[i] if i < len(segment_colors) else segment_colors[-1]
        if segment_alphas is not None and i < len(segment_alphas):
            seg_alpha = segment_alphas[i]
        else:
            seg_alpha = alpha
        if seg_alpha < 0.04:
            continue
        linestyle = "-"
        if segment_linestyles is not None and i < len(segment_linestyles):
            linestyle = segment_linestyles[i]
        p0, p1 = path[i], path[i + 1]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=color, linewidth=linewidth, alpha=seg_alpha,
            linestyle=linestyle, solid_capstyle="round", zorder=zorder,
        )
        if draw_arrows:
            _midsegment_arrowhead_2d(
                ax, p0, p1,
                color=color, alpha=min(seg_alpha + 0.12, 1.0), zorder=zorder + 1,
                mutation_scale=arrow_mutation_scale,
            )


def _plot_step_colored_path_arrows_3d(
    ax,
    path: np.ndarray,
    *,
    linewidth: float = 1.6,
    alpha: float = 0.55,
    cmap_name: str = "viridis",
    segment_colors: list | None = None,
    segment_linestyles: list[str] | None = None,
    arrow_mutation_scale: float = 12.0,
) -> None:
    """3D trajectory lines with arrowheads centered on each segment."""
    if len(path) < 2:
        return
    if segment_colors is None:
        segment_colors = _step_path_colors(len(path), cmap_name)
    for i in range(len(path) - 1):
        color = segment_colors[i] if i < len(segment_colors) else segment_colors[-1]
        linestyle = "-"
        if segment_linestyles is not None and i < len(segment_linestyles):
            linestyle = segment_linestyles[i]
        p0, p1 = path[i], path[i + 1]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle,
        )
        ax.quiver(
            p0[0], p0[1], p0[2],
            p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2],
            color=color,
            alpha=min(alpha + 0.12, 1.0),
            arrow_length_ratio=0.35,
            linewidth=max(linewidth * 0.65, 0.5),
        )


def _add_trajectory_step_colorbar(
    fig,
    axes,
    n_steps: int,
    *,
    label: str = "step",
) -> None:
    if n_steps <= 1:
        return
    n_colors = min(n_steps, len(_TRAJECTORY_STEP_PALETTE))
    colors = list(_TRAJECTORY_STEP_PALETTE[:n_colors])
    cmap = ListedColormap(colors)
    bounds = np.arange(0.5, n_colors + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes, fraction=0.025, pad=0.02, shrink=0.55, label=label,
    )
    cbar.set_ticks(np.arange(1, n_colors + 1))


def _plot_colored_path_arrows(
    ax,
    path: np.ndarray,
    color,
    *,
    linewidth: float = 1.6,
    alpha: float = 0.55,
    zorder: int = 2,
    mutation_scale: float = 12.0,
    head_frac: float = 0.22,
    arrow_alpha_boost: float = 0.15,
) -> None:
    """Colored trajectory line with arrowheads centered on each segment."""
    if len(path) < 2:
        return
    ax.plot(
        path[:, 0], path[:, 1],
        color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round", zorder=zorder,
    )
    arrow_alpha = min(alpha + arrow_alpha_boost, 1.0)
    for i in range(len(path) - 1):
        _midsegment_arrowhead_2d(
            ax, path[i], path[i + 1],
            color=color, alpha=arrow_alpha, zorder=zorder + 1,
            mutation_scale=mutation_scale, head_frac=head_frac,
        )


def _plot_colored_path_arrows_3d(
    ax,
    path: np.ndarray,
    color,
    *,
    linewidth: float = 1.6,
    alpha: float = 0.55,
) -> None:
    """3D trajectory line with arrowheads at each step showing travel direction."""
    if len(path) < 2:
        return
    ax.plot(
        path[:, 0], path[:, 1], path[:, 2],
        color=color, linewidth=linewidth, alpha=alpha,
    )
    ax.quiver(
        path[:-1, 0], path[:-1, 1], path[:-1, 2],
        path[1:, 0] - path[:-1, 0],
        path[1:, 1] - path[:-1, 1],
        path[1:, 2] - path[:-1, 2],
        color=color,
        alpha=min(alpha + 0.15, 1.0),
        arrow_length_ratio=0.35,
        linewidth=max(linewidth * 0.6, 0.4),
    )


def _add_pca_prefix_labels_3d(
    ax,
    text: str,
    projected: np.ndarray,
    *,
    spaced: bool = False,
    prefix_labels: list[str] | None = None,
    fontsize: float = 6.0,
) -> None:
    """Prefix labels at PCA positions (no leader lines or scatter points)."""
    n = len(prefix_labels) if prefix_labels is not None else len(text)
    if prefix_labels is not None:
        prefixes = prefix_labels
    else:
        prefixes = [prefix_annotation_label(text, i, spaced=spaced) for i in range(n)]
    for i, prefix in enumerate(prefixes):
        if i >= len(projected):
            break
        label = "␣" if prefix == " " else prefix
        ax.text(
            projected[i, 0], projected[i, 1], projected[i, 2],
            label, fontsize=fontsize, color="#1a1a1a",
        )


def _square_data_limits(*xy_arrays: np.ndarray, padding_frac: float = 0.12):
    """Square x/y limits from trajectory data (ignore annotation label offsets)."""
    xs: list[float] = []
    ys: list[float] = []
    for arr in xy_arrays:
        if arr is None or len(arr) == 0:
            continue
        xs.extend([float(arr[:, 0].min()), float(arr[:, 0].max())])
        ys.extend([float(arr[:, 1].min()), float(arr[:, 1].max())])
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
    half *= 1.0 + padding_frac
    return (cx - half, cx + half), (cy - half, cy + half)


def _cube_data_limits(*xyz_arrays: np.ndarray, padding_frac: float = 0.12):
    """Equal x/y/z limits from trajectory data (ignore annotation label offsets)."""
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for arr in xyz_arrays:
        if arr is None or len(arr) == 0:
            continue
        for dim in range(min(3, arr.shape[1])):
            mins[dim] = min(mins[dim], float(arr[:, dim].min()))
            maxs[dim] = max(maxs[dim], float(arr[:, dim].max()))
    if not np.isfinite(mins[0]):
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    cz = 0.5 * (mins[2] + maxs[2])
    half = 0.5 * max(maxs[d] - mins[d] for d in range(3))
    half = max(half, 1e-3) * (1.0 + padding_frac)
    return (
        (cx - half, cx + half),
        (cy - half, cy + half),
        (cz - half, cz + half),
    )


def _apply_cube_limits_3d(ax, xlim, ylim, zlim) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    try:
        ax.set_box_aspect(
            (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]),
            zoom=1.0,
        )
    except AttributeError:
        pass


def _vocabulary_prefix_paths(
    words: list[str], condensed: CondensedView,
) -> list[tuple[str, list[int]]]:
    """For each vocabulary word, indices into condensed rows along its prefix chain."""
    paths: list[tuple[str, list[int]]] = []
    for word in words:
        idxs = [
            condensed.label_to_index[word[: k + 1]]
            for k in range(len(word))
            if word[: k + 1] in condensed.label_to_index
        ]
        if len(idxs) >= 2:
            paths.append((word, idxs))
    return paths


def _letter_rollout_steps(
    words: list[str],
    closed_loop_steps: int | None,
) -> int:
    """Deprecated alias — use ``_closed_loop_rollout_steps`` for closed-loop panels."""
    return _closed_loop_rollout_steps(words, closed_loop_steps, spaced=False)


def _trained_word_examples(
    segments: list[tuple[int, int, str]],
    vocab_words: list[str],
) -> list[tuple[int, int, str, str]]:
    """First corpus occurrence per vocabulary word: (start, end, segment_text, word_label)."""
    vocab_set = set(vocab_words)
    seen: set[str] = set()
    examples: list[tuple[int, int, str, str]] = []
    for start, end, seg in segments:
        word = segment_word_label(seg)
        if not word or word == "␣":
            continue
        if vocab_set and word not in vocab_set:
            continue
        if word in seen:
            continue
        seen.add(word)
        examples.append((start, end, seg, word))
    return examples


def _in_word_prefix_labels(segment_text: str, n_points: int) -> list[str]:
    """Cumulative in-word prefix labels for the first ``n_points`` characters."""
    word = segment_text.strip()
    if not word:
        return [""] * n_points
    return [word[: i + 1] for i in range(min(n_points, len(word)))]


def _plot_trained_word_examples(
    ax,
    projected: np.ndarray,
    examples: list[tuple[int, int, str, str]],
    *,
    max_word_len: int,
    annotate_fontsize: float = 9.5,
    is_3d: bool = False,
) -> list[np.ndarray]:
    """Plot one observed trajectory per vocabulary word from its first character."""
    plot_path = _plot_step_colored_path_arrows_3d if is_3d else _plot_step_colored_path_arrows
    paths: list[np.ndarray] = []
    for start, end, seg, _word in examples:
        n_pts = min(max_word_len, end - start + 1, len(seg.strip()) or (end - start + 1))
        if n_pts < 1:
            continue
        path = projected[start : start + n_pts]
        if len(path) < 1:
            continue
        paths.append(path)
        if is_3d:
            plot_path(ax, path, linewidth=1.8, alpha=0.72)
            _plot_return_to_start_segment(ax, path, is_3d=True)
        else:
            plot_path(ax, path, linewidth=1.8, alpha=0.72, zorder=2)
            _plot_return_to_start_segment(ax, path, is_3d=False, zorder=2)
        labels = _in_word_prefix_labels(seg, len(path))
        _annotate_trajectory_labels(
            ax, path, labels,
            colors=_step_path_colors(len(labels)), fontsize=annotate_fontsize,
        )
    return paths


def _plot_trained_word_examples_grid(
    fig,
    gridspec,
    projected: np.ndarray,
    examples: list[tuple[int, int, str, str]],
    *,
    max_word_len: int,
    is_3d: bool,
    xlabel: str,
    ylabel: str,
    zlabel: str | None = None,
    annotate_fontsize: float = 8.5,
) -> list[np.ndarray]:
    """One subplot per trained vocabulary word (first corpus occurrence)."""
    n_words = len(examples)
    if n_words == 0:
        return []
    ncols = min(4, max(1, n_words))
    nrows = int(math.ceil(n_words / ncols))
    plot_path = _plot_step_colored_path_arrows_3d if is_3d else _plot_step_colored_path_arrows
    paths: list[np.ndarray] = []
    for i, (start, end, seg, word) in enumerate(examples):
        row, col = divmod(i, ncols)
        if is_3d:
            ax = fig.add_subplot(gridspec[row, col], projection="3d")
        else:
            ax = fig.add_subplot(gridspec[row, col])
        n_pts = min(max_word_len, end - start + 1, len(seg.strip()) or (end - start + 1))
        if n_pts < 1:
            continue
        path = projected[start : start + n_pts]
        if len(path) < 1:
            continue
        paths.append(path)
        if is_3d:
            plot_path(ax, path, linewidth=2.0, alpha=0.85)
            _plot_return_to_start_segment(ax, path, is_3d=True)
        else:
            plot_path(ax, path, linewidth=2.0, alpha=0.85, zorder=2)
            _plot_return_to_start_segment(ax, path, is_3d=False, zorder=2)
        labels = _in_word_prefix_labels(seg, len(path))
        _annotate_trajectory_labels(
            ax, path, labels,
            colors=_step_path_colors(len(labels)), fontsize=annotate_fontsize,
        )
        ax.set_title(f"'{word}'", fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if is_3d and zlabel is not None:
            ax.set_zlabel(zlabel, fontsize=8)
        if is_3d:
            xlim, ylim, zlim = _cube_data_limits(path)
            _apply_cube_limits_3d(ax, xlim, ylim, zlim)
        else:
            xlim, ylim = _square_data_limits(path)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.35)
    return paths


def _teacher_forced_vocab_trajectories(
    model: dict,
    vocab_words: list[str],
    *,
    mean: np.ndarray,
    components: np.ndarray,
    max_word_len: int,
    normalize_activity: bool = False,
) -> list[tuple[str, np.ndarray, list[str]]]:
    """One teacher-forced trajectory per vocabulary word from its first character."""
    examples: list[tuple[str, np.ndarray, list[str]]] = []
    for word in sorted(set(vocab_words)):
        snippet = word[:max_word_len]
        if not snippet:
            continue
        hidden, _ = forward_pass(model, snippet)
        if hidden.shape[0] < 1:
            continue
        z = _project_hidden_to_pca(
            hidden, mean, components, normalize_activity=normalize_activity,
        )
        labels = [snippet[: i + 1] for i in range(len(snippet))]
        examples.append((word, z, labels))
    return examples


def _teacher_forced_vocab_cycle_pca(
    model: dict,
    vocab_words: list[str],
    *,
    mean: np.ndarray,
    components: np.ndarray,
    max_word_len: int,
    normalize_activity: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Teacher-forced states for every vocab word, concatenated into one cycle."""
    zs: list[np.ndarray] = []
    prefix_labels: list[str] = []
    word_at_step: list[str] = []
    for word in sorted(set(vocab_words)):
        snippet = word[:max_word_len]
        if not snippet:
            continue
        hidden, _ = forward_pass(model, snippet)
        if hidden.shape[0] < 1:
            continue
        z = _project_hidden_to_pca(
            hidden, mean, components, normalize_activity=normalize_activity,
        )
        zs.append(z)
        prefix_labels.extend([snippet[: i + 1] for i in range(len(snippet))])
        word_at_step.extend([word] * len(z))
    if not zs:
        return np.zeros((0, 2)), [], []
    return np.vstack(zs), prefix_labels, word_at_step


def _plot_teacher_forced_vocab_on_axis(
    ax,
    examples: list[tuple[str, np.ndarray, list[str]]],
    *,
    word_colors: dict[str, tuple],
    annotate_fontsize: float = 9.5,
    is_3d: bool = False,
) -> list[np.ndarray]:
    plot_path = _plot_colored_path_arrows_3d if is_3d else _plot_colored_path_arrows
    paths: list[np.ndarray] = []
    for word, z, labels in examples:
        if len(z) < 1:
            continue
        paths.append(z)
        color = _trajectory_word_color(word, word_colors)
        if is_3d:
            plot_path(ax, z, color, linewidth=1.8, alpha=0.72)
            _plot_return_to_start_segment(ax, z, is_3d=True)
        else:
            plot_path(ax, z, color, linewidth=1.8, alpha=0.72, zorder=2)
            _plot_return_to_start_segment(ax, z, is_3d=False, zorder=2)
        end_labels = [""] * len(z)
        end_labels[-1] = word
        _annotate_trajectory_labels(
            ax, z, end_labels,
            colors=[color] * len(end_labels), fontsize=annotate_fontsize,
        )
    avg = _position_aligned_average_trajectory(paths)
    if avg is not None:
        paths.append(avg)
        _plot_mean_trajectory_overlay(ax, avg, is_3d=is_3d)
    return paths


def _plot_teacher_forced_vocab_grid(
    fig,
    gridspec,
    examples: list[tuple[str, np.ndarray, list[str]]],
    *,
    is_3d: bool,
    xlabel: str,
    ylabel: str,
    zlabel: str | None = None,
    annotate_fontsize: float = 8.5,
) -> list[np.ndarray]:
    n_words = len(examples)
    if n_words == 0:
        return []
    ncols = min(4, max(1, n_words))
    nrows = int(math.ceil(n_words / ncols))
    plot_path = _plot_step_colored_path_arrows_3d if is_3d else _plot_step_colored_path_arrows
    paths: list[np.ndarray] = []
    for i, (word, z, labels) in enumerate(examples):
        row, col = divmod(i, ncols)
        if is_3d:
            ax = fig.add_subplot(gridspec[row, col], projection="3d")
        else:
            ax = fig.add_subplot(gridspec[row, col])
        if len(z) < 1:
            continue
        paths.append(z)
        if is_3d:
            plot_path(ax, z, linewidth=2.0, alpha=0.85)
            _plot_return_to_start_segment(ax, z, is_3d=True)
        else:
            plot_path(ax, z, linewidth=2.0, alpha=0.85, zorder=2)
            _plot_return_to_start_segment(ax, z, is_3d=False, zorder=2)
        _annotate_trajectory_labels(
            ax, z, labels,
            colors=_step_path_colors(len(labels)), fontsize=annotate_fontsize,
        )
        ax.set_title(f"'{word}'", fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if is_3d and zlabel is not None:
            ax.set_zlabel(zlabel, fontsize=8)
        if is_3d:
            xlim, ylim, zlim = _cube_data_limits(z)
            _apply_cube_limits_3d(ax, xlim, ylim, zlim)
        else:
            xlim, ylim = _square_data_limits(z)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.35)
    return paths


def _plot_letter_seed_closed_loop_on_axis(
    ax,
    model: dict,
    *,
    seed_letters: list[str],
    steps: int,
    closed_loop_seed: int,
    mean: np.ndarray,
    components: np.ndarray,
    limit_arrays: list[np.ndarray],
    vocab_words: list[str],
    word_colors: dict[str, tuple],
    spaced: bool,
    annotate: bool = True,
    annotate_fontsize: float = 9.0,
    is_3d: bool = False,
    max_rollouts: int | None = None,
    unique_word_labels: bool = False,
    average_trials: int = 0,
    normalize_activity: bool = False,
) -> None:
    """Long closed-loop rollout(s); color by segmented vocabulary word."""
    seeds = seed_letters if max_rollouts is None else seed_letters[: max(1, int(max_rollouts))]
    for seed_idx, seed_char in enumerate(seeds):
        trial_z: list[np.ndarray] = []
        trial_prefix: list[list[str]] = []
        trial_words: list[list[str]] = []
        n_trials = max(1, int(average_trials)) if average_trials else 1
        for trial in range(n_trials):
            rng = np.random.default_rng(int(closed_loop_seed) + seed_idx * 1000 + trial)
            gen_z, prefix_labels, word_at_step = _closed_loop_rollout_pca(
                model,
                seed_text=seed_char,
                steps=steps,
                rng=rng,
                mean=mean,
                components=components,
                vocab_words=vocab_words,
                spaced=spaced,
                normalize_activity=normalize_activity,
            )
            if len(gen_z) < 1:
                continue
            trial_z.append(gen_z)
            trial_prefix.append(prefix_labels)
            trial_words.append(word_at_step)
        if not trial_z:
            continue
        z_plot = trial_z[0]
        if z_plot.shape[0] < 2:
            continue
        limit_arrays.append(z_plot)
        _plot_segmented_closed_loop_rollout(
            ax, z_plot, trial_prefix[0], trial_words[0],
            word_colors=word_colors,
            vocab_words=vocab_words,
            annotate=annotate and n_trials == 1,
            annotate_fontsize=annotate_fontsize,
            is_3d=is_3d,
            linewidth=1.6,
            alpha=0.30 if n_trials > 1 else 0.82,
            unique_word_labels=unique_word_labels,
        )
        if n_trials > 1:
            avg_z = _same_length_average_trajectory(trial_z)
            avg_words = _majority_labels_per_step(trial_words)
            if avg_z is not None:
                limit_arrays.append(avg_z)
                _plot_mean_trajectory_overlay(ax, avg_z, is_3d=is_3d)
                if annotate:
                    vocab = set(vocab_words)
                    label_fn = (
                        _sparse_unique_word_end_labels if unique_word_labels else _sparse_word_end_labels
                    )
                    end_labels = label_fn(avg_words, vocab, len(avg_z))
                    _annotate_trajectory_labels(
                        ax, avg_z, end_labels,
                        fontsize=annotate_fontsize + 0.5,
                        dedupe=True,
                        word_keys=avg_words,
                        label_colors=[
                            word_colors.get(avg_words[i], word_colors["?"])
                            for i in range(len(end_labels))
                        ],
                        use_leaders=True,
                        leader_linewidth=0.5,
                    )


def _plot_letter_seed_closed_loop_grid(
    fig,
    gridspec,
    model: dict,
    *,
    seed_letters: list[str],
    steps: int,
    closed_loop_seed: int,
    mean: np.ndarray,
    components: np.ndarray,
    limit_arrays: list[np.ndarray],
    vocab_words: list[str],
    word_colors: dict[str, tuple],
    spaced: bool,
    is_3d: bool,
    xlabel: str,
    ylabel: str,
    zlabel: str | None = None,
    annotate_fontsize: float = 8.5,
) -> None:
    """One subplot per seed letter; faint sample + black mean trajectory."""
    n_letters = len(seed_letters)
    ncols = min(4, max(1, n_letters))
    nrows = int(math.ceil(n_letters / ncols))
    for i, seed_char in enumerate(seed_letters):
        row, col = divmod(i, ncols)
        if is_3d:
            ax = fig.add_subplot(gridspec[row, col], projection="3d")
        else:
            ax = fig.add_subplot(gridspec[row, col])
        letter_limits: list[np.ndarray] = []
        _plot_letter_seed_closed_loop_on_axis(
            ax, model,
            seed_letters=[seed_char],
            steps=steps,
            closed_loop_seed=closed_loop_seed,
            mean=mean,
            components=components,
            limit_arrays=letter_limits,
            vocab_words=vocab_words,
            word_colors=word_colors,
            spaced=spaced,
            annotate=True,
            annotate_fontsize=annotate_fontsize,
            is_3d=is_3d,
            unique_word_labels=True,
            average_trials=_CLOSED_LOOP_AVERAGE_TRIALS,
        )
        limit_arrays.extend(letter_limits)
        ax.set_title(
            f"'{seed_char}' · {steps} steps · mean ({_CLOSED_LOOP_AVERAGE_TRIALS} trials)",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if is_3d and zlabel is not None:
            ax.set_zlabel(zlabel, fontsize=8)
        if letter_limits:
            if is_3d:
                xlim, ylim, zlim = _cube_data_limits(*letter_limits)
                _apply_cube_limits_3d(ax, xlim, ylim, zlim)
            else:
                xlim, ylim = _square_data_limits(*letter_limits)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.35)


def _plot_trajectory_internal_panel(
    ax,
    model: dict,
    seed_letters: list[str],
    steps: int,
    mean: np.ndarray,
    components: np.ndarray,
    hidden_size: int,
    *,
    is_3d: bool,
    normalize_activity: bool = False,
) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    plot_fn = _plot_colored_path_arrows_3d if is_3d else _plot_colored_path_arrows
    seed_colors = _seed_letter_colors(seed_letters)
    for seed_char in seed_letters:
        zs = _letter_seed_no_input_trajectory_pca(
            model,
            seed_char=seed_char,
            steps=steps,
            mean=mean,
            components=components,
            hidden_size=hidden_size,
            normalize_activity=normalize_activity,
        )
        if zs.shape[0] < 2:
            continue
        paths.append(zs)
        color = seed_colors[seed_char]
        if is_3d:
            plot_fn(ax, zs, color, linewidth=1.2, alpha=0.50)
            _plot_return_to_start_segment(ax, zs, is_3d=True)
        else:
            plot_fn(ax, zs, color, linewidth=1.2, alpha=0.50, zorder=3)
            _plot_return_to_start_segment(ax, zs, is_3d=False, zorder=3)
        _annotate_trajectory_labels(
            ax, zs[:1], [seed_char],
            colors=[color], fontsize=9.0,
        )
    avg = _same_length_average_trajectory(paths)
    if avg is not None:
        paths.append(avg)
        _plot_mean_trajectory_overlay(ax, avg, is_3d=is_3d)
        if not is_3d:
            _plot_return_to_start_segment(ax, avg, is_3d=False, zorder=4)
    return paths


def _plot_trajectory_trained_panel(
    ax,
    model: dict | None,
    vocab_words: list[str],
    segments: list[tuple[int, int, str]],
    projected: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    max_word_len: int,
    *,
    word_colors: dict[str, tuple],
    is_3d: bool,
    normalize_activity: bool = False,
) -> tuple[list[np.ndarray], int]:
    if model is not None:
        cycle_z, prefix_labels, word_at_step = _teacher_forced_vocab_cycle_pca(
            model, vocab_words, mean=mean, components=components,
            max_word_len=max_word_len, normalize_activity=normalize_activity,
        )
        paths: list[np.ndarray] = []
        if len(cycle_z) >= 2:
            _plot_segmented_vocab_rollout(
                ax, cycle_z, prefix_labels, word_at_step,
                word_colors=word_colors,
                vocab_words=vocab_words,
                is_3d=is_3d,
                color_mode="word",
                max_word_len=max_word_len,
            )
            paths.append(cycle_z)
        trained_tf = _teacher_forced_vocab_trajectories(
            model, vocab_words, mean=mean, components=components,
            max_word_len=max_word_len, normalize_activity=normalize_activity,
        )
        word_paths = [z for _, z, _ in trained_tf if len(z) >= 1]
        avg = _position_aligned_average_trajectory(word_paths)
        if avg is not None:
            paths.append(avg)
            _plot_mean_trajectory_overlay(ax, avg, is_3d=is_3d)
        return paths, len(trained_tf)
    trained_examples = _trained_word_examples(segments, vocab_words)
    paths = _plot_trained_word_examples(
        ax, projected, trained_examples,
        max_word_len=max_word_len, annotate_fontsize=9.5, is_3d=is_3d,
    )
    return paths, len(trained_examples)


def _in_word_step_segment_colors(prefix_labels: list[str], max_word_len: int) -> list:
    """Discrete step colors per segment from in-word prefix length (resets each word)."""
    colors: list = []
    for i in range(len(prefix_labels) - 1):
        plen = len(prefix_labels[i + 1]) if prefix_labels[i + 1] not in ("", "␣") else 1
        colors.append(_step_palette_rgba(min(plen, max_word_len)))
    return colors


def _plot_trajectory_internal_panel_step_colored(
    ax,
    model: dict,
    seed_letters: list[str],
    steps: int,
    mean: np.ndarray,
    components: np.ndarray,
    hidden_size: int,
) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    for seed_char in seed_letters:
        zs = _letter_seed_no_input_trajectory_pca(
            model,
            seed_char=seed_char,
            steps=steps,
            mean=mean,
            components=components,
            hidden_size=hidden_size,
        )
        if zs.shape[0] < 2:
            continue
        paths.append(zs)
        _plot_step_colored_path_arrows(
            ax, zs, linewidth=1.4, alpha=0.55, zorder=3,
        )
        _plot_return_to_start_segment(ax, zs, is_3d=False, zorder=3)
        _annotate_trajectory_labels(
            ax, zs[:1], [seed_char],
            colors=_step_path_colors(1), fontsize=9.0,
        )
    avg = _same_length_average_trajectory(paths)
    if avg is not None:
        paths.append(avg)
        _plot_mean_trajectory_overlay(ax, avg, is_3d=False)
        _plot_return_to_start_segment(ax, avg, is_3d=False, zorder=4)
    return paths


def _plot_trajectory_random_internal_panel(
    ax,
    model: dict,
    hidden_seeds: list[np.ndarray],
    steps: int,
    mean: np.ndarray,
    components: np.ndarray,
    *,
    is_3d: bool,
    normalize_activity: bool = False,
) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    plot_fn = _plot_colored_path_arrows_3d if is_3d else _plot_colored_path_arrows
    seed_colors = _random_seed_colors(len(hidden_seeds))
    for i, h0 in enumerate(hidden_seeds):
        zs = _random_hidden_no_input_trajectory_pca(
            model, h0=h0, steps=steps, mean=mean, components=components,
            normalize_activity=normalize_activity,
        )
        if zs.shape[0] < 2:
            continue
        paths.append(zs)
        if is_3d:
            plot_fn(ax, zs, seed_colors[i], linewidth=1.8, alpha=0.52)
            _plot_return_to_start_segment(ax, zs, is_3d=True)
        else:
            plot_fn(ax, zs, seed_colors[i], linewidth=1.8, alpha=0.52, zorder=3)
            _plot_return_to_start_segment(ax, zs, is_3d=False, zorder=3)
    avg = _same_length_average_trajectory(paths)
    if avg is not None:
        paths.append(avg)
        _plot_mean_trajectory_overlay(ax, avg, is_3d=is_3d)
        if not is_3d:
            _plot_return_to_start_segment(ax, avg, is_3d=False, zorder=4)
    return paths


def _plot_trajectory_random_internal_panel_step_colored(
    ax,
    model: dict,
    hidden_seeds: list[np.ndarray],
    steps: int,
    mean: np.ndarray,
    components: np.ndarray,
) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    for h0 in hidden_seeds:
        zs = _random_hidden_no_input_trajectory_pca(
            model, h0=h0, steps=steps, mean=mean, components=components,
        )
        if zs.shape[0] < 2:
            continue
        paths.append(zs)
        _plot_step_colored_path_arrows(
            ax, zs, linewidth=1.6, alpha=0.55, zorder=2, draw_arrows=True,
        )
        _plot_return_to_start_segment(ax, zs, is_3d=False, zorder=2)
    return paths


def _plot_teacher_forced_vocab_step_colored_on_axis(
    ax,
    examples: list[tuple[str, np.ndarray, list[str]]],
    *,
    annotate_fontsize: float = 9.5,
) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    for _word, z, labels in examples:
        if len(z) < 1:
            continue
        paths.append(z)
        _plot_step_colored_path_arrows(
            ax, z, linewidth=1.8, alpha=0.72, zorder=2,
        )
        _annotate_trajectory_labels(
            ax, z, labels,
            colors=_step_path_colors(len(labels)), fontsize=annotate_fontsize,
        )
    avg = _position_aligned_average_trajectory(paths)
    if avg is not None:
        paths.append(avg)
        _plot_mean_trajectory_overlay(ax, avg, is_3d=False)
    return paths


def _plot_trajectory_trained_panel_step_colored(
    ax,
    model: dict | None,
    vocab_words: list[str],
    segments: list[tuple[int, int, str]],
    projected: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    max_word_len: int,
) -> tuple[list[np.ndarray], int]:
    if model is not None:
        word_colors = _vocab_word_colors(vocab_words)
        cycle_z, prefix_labels, word_at_step = _teacher_forced_vocab_cycle_pca(
            model, vocab_words, mean=mean, components=components,
            max_word_len=max_word_len,
        )
        paths: list[np.ndarray] = []
        if len(cycle_z) >= 2:
            _plot_segmented_vocab_rollout(
                ax, cycle_z, prefix_labels, word_at_step,
                word_colors=word_colors,
                vocab_words=vocab_words,
                is_3d=False,
                color_mode="step",
                max_word_len=max_word_len,
            )
            paths.append(cycle_z)
        trained_tf = _teacher_forced_vocab_trajectories(
            model, vocab_words, mean=mean, components=components,
            max_word_len=max_word_len,
        )
        word_paths = [z for _, z, _ in trained_tf if len(z) >= 1]
        avg = _position_aligned_average_trajectory(word_paths)
        if avg is not None:
            paths.append(avg)
            _plot_mean_trajectory_overlay(ax, avg, is_3d=False)
            _plot_return_to_start_segment(ax, avg, is_3d=False, zorder=4)
        return paths, len(trained_tf)
    trained_examples = _trained_word_examples(segments, vocab_words)
    paths = _plot_trained_word_examples(
        ax, projected, trained_examples,
        max_word_len=max_word_len, annotate_fontsize=9.5, is_3d=False,
    )
    return paths, len(trained_examples)


def _plot_trajectory_closed_loop_step_colored_panel(
    ax,
    model: dict,
    seed_letters: list[str],
    steps: int,
    closed_loop_seed: int,
    mean: np.ndarray,
    components: np.ndarray,
    limit_arrays: list[np.ndarray],
    *,
    vocab_words: list[str],
    spaced: bool,
    max_word_len: int,
) -> None:
    """Closed-loop rollout colored by in-word step; black mean overlay."""
    word_colors = _vocab_word_colors(vocab_words)
    seeds = seed_letters[: max(1, len(seed_letters))]
    for seed_idx, seed_char in enumerate(seeds):
        trial_z: list[np.ndarray] = []
        trial_prefix: list[list[str]] = []
        trial_words: list[list[str]] = []
        n_trials = _CLOSED_LOOP_AVERAGE_TRIALS
        for trial in range(n_trials):
            rng = np.random.default_rng(int(closed_loop_seed) + seed_idx * 1000 + trial)
            gen_z, prefix_labels, word_at_step = _closed_loop_rollout_pca(
                model,
                seed_text=seed_char,
                steps=steps,
                rng=rng,
                mean=mean,
                components=components,
                vocab_words=vocab_words,
                spaced=spaced,
            )
            if len(gen_z) < 1:
                continue
            trial_z.append(gen_z)
            trial_prefix.append(prefix_labels)
            trial_words.append(word_at_step)
        if not trial_z:
            continue
        limit_arrays.append(trial_z[0])
        _plot_segmented_vocab_rollout(
            ax, trial_z[0], trial_prefix[0], trial_words[0],
            word_colors=word_colors,
            vocab_words=vocab_words,
            is_3d=False,
            linewidth=1.6,
            alpha=0.30,
            color_mode="step",
            max_word_len=max_word_len,
        )
        avg_z = _same_length_average_trajectory(trial_z)
        if avg_z is not None:
            limit_arrays.append(avg_z)
            _plot_mean_trajectory_overlay(ax, avg_z, is_3d=False)
            _plot_return_to_start_segment(ax, avg_z, is_3d=False, zorder=4)


def _plot_trajectory_closed_loop_panel(
    ax,
    model: dict,
    seed_letters: list[str],
    steps: int,
    closed_loop_seed: int,
    mean: np.ndarray,
    components: np.ndarray,
    limit_arrays: list[np.ndarray],
    *,
    vocab_words: list[str],
    word_colors: dict[str, tuple],
    spaced: bool,
    is_3d: bool,
    max_rollouts: int | None = 1,
    unique_word_labels: bool = True,
    average_trials: int = _CLOSED_LOOP_AVERAGE_TRIALS,
    normalize_activity: bool = False,
    annotate: bool = True,
    annotate_fontsize: float = 8.0,
) -> None:
    _plot_letter_seed_closed_loop_on_axis(
        ax, model,
        seed_letters=seed_letters,
        steps=steps,
        closed_loop_seed=closed_loop_seed,
        mean=mean,
        components=components,
        limit_arrays=limit_arrays,
        vocab_words=vocab_words,
        word_colors=word_colors,
        spaced=spaced,
        annotate=annotate,
        annotate_fontsize=annotate_fontsize,
        is_3d=is_3d,
        max_rollouts=max_rollouts,
        unique_word_labels=unique_word_labels,
        average_trials=average_trials,
        normalize_activity=normalize_activity,
    )


def _word_segment_ranges(word_at_step: list[str]) -> list[tuple[int, int, str]]:
    """Contiguous index ranges where ``word_at_step`` is constant."""
    if not word_at_step:
        return []
    ranges: list[tuple[int, int, str]] = []
    start = 0
    word = word_at_step[0]
    for j in range(1, len(word_at_step)):
        if word_at_step[j] != word:
            ranges.append((start, j - 1, word))
            start = j
            word = word_at_step[j]
    ranges.append((start, len(word_at_step) - 1, word))
    return ranges


def _state_word_ages(word_at_step: list[str]) -> list[int]:
    """Per-state age in word segments (0 = newest word, 1 = previous, ...)."""
    ranges = _word_segment_ranges(word_at_step)
    ages = [10**9] * len(word_at_step)
    for word_age, (start, end, _) in enumerate(reversed(ranges)):
        for j in range(start, end + 1):
            ages[j] = word_age
    return ages


def _trail_visible_word_alphas(
    n: int,
    word_at_step: list[str],
    *,
    visible_words: int = 3,
    max_alpha: float = 0.95,
) -> list[float]:
    """Per-state alpha: only states in the last ``visible_words`` word segments."""
    alphas = [0.0] * n
    if n <= 0:
        return alphas
    tags = word_at_step[:n]
    ranges = _word_segment_ranges(tags)
    word_ages = _state_word_ages(tags)
    head = max(1, int(visible_words))
    for j in range(n):
        wa = word_ages[j]
        if wa >= head:
            continue
        seg_start, seg_end, _ = next(
            (start, end, w) for start, end, w in ranges if start <= j <= end
        )
        seg_len = seg_end - seg_start + 1
        char_age = seg_end - j
        word_factor = 1.0 - 0.35 * wa / max(head - 1, 1)
        if seg_len > 1:
            char_factor = 1.0 - 0.25 * char_age / (seg_len - 1)
        else:
            char_factor = 1.0
        alphas[j] = max(0.0, max_alpha * word_factor * char_factor)
    return alphas


def _segment_alphas_from_states(state_alphas: list[float]) -> list[float]:
    """Segment alpha follows the newer endpoint of each edge."""
    if len(state_alphas) < 2:
        return []
    return [
        state_alphas[i + 1] if state_alphas[i + 1] > state_alphas[i] else state_alphas[i]
        for i in range(len(state_alphas) - 1)
    ]


def _color_with_alpha(color, alpha: float):
    return plt.matplotlib.colors.to_rgba(color, alpha=alpha)


def _annotate_fading_prefix_vertices(
    ax,
    coords: np.ndarray,
    prefix_labels: list[str],
    state_alphas: list[float],
    *,
    word_colors: dict[str, tuple],
    word_at_step: list[str],
    fontsize: float = 10.5,
    label_offset_points: float = 14.0,
) -> None:
    """Prefix label + vertex marker at each visible state (fading with trail age)."""
    n = min(len(coords), len(prefix_labels), len(state_alphas), len(word_at_step))
    for j in range(n):
        alpha = state_alphas[j]
        if alpha < 0.04:
            continue
        label = prefix_labels[j]
        if not label:
            continue
        display = "␣" if label == " " else label
        base = word_colors.get(word_at_step[j], word_colors["?"])
        color = _color_with_alpha(base, alpha)
        ax.scatter(
            [coords[j, 0]], [coords[j, 1]],
            s=26 + 20 * alpha, c=[color], edgecolors="white",
            linewidths=0.35, zorder=8,
        )
        ax.annotate(
            display,
            (float(coords[j, 0]), float(coords[j, 1])),
            textcoords="offset points",
            xytext=(0, label_offset_points),
            ha="center",
            va="bottom",
            fontsize=max(8, int(fontsize * (0.75 + 0.25 * alpha))),
            color=color,
            fontweight="bold",
            zorder=9,
        )


def write_closed_loop_trajectory_video(
    model: dict,
    hidden_states: np.ndarray,
    save_path: str,
    *,
    vocab_words: list[str],
    spaced: bool = False,
    seed_text: str | None = None,
    steps: int | None = None,
    closed_loop_seed: int = 0,
    fps: int = 4,
    dpi: int = 120,
    hold_final_frames: int = 20,
    pause_frames_per_word: int = 5,
    trail_visible_words: int = 2,
    trail_fade_max_alpha: float = 0.95,
) -> str | None:
    """Animate one closed-loop rollout: trajectory grows as in-word prefix appears above."""
    if len(vocab_words) < 1:
        print(f"skip {save_path}: no vocabulary words")
        return None

    _, mean, components, evr = fit_pca_2d_with_evr(hidden_states)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    seed_letters = _trajectory_seed_letters(model, vocab_words)
    seed = seed_text or _closed_loop_summary_seed(vocab_words, seed_letters, spaced=spaced)
    rollout_steps = steps or _one_vocab_cycle_steps(vocab_words, spaced=spaced)
    word_colors = _vocab_word_colors(vocab_words)
    vocab = set(vocab_words)

    rng = np.random.default_rng(int(closed_loop_seed))
    hidden, generated = rnn_closed_loop_rollout(
        model, seed_text=seed, steps=rollout_steps, rng=rng,
    )
    gen_z = (hidden - mean) @ components.T
    seed_len = len(seed)
    n_states = len(gen_z)
    if n_states < 2:
        print(f"skip {save_path}: closed-loop path too short")
        return None

    prefix_labels = _rollout_prefix_labels(
        generated, seed_len, n_states, spaced=spaced, vocab=vocab,
    )
    word_at_step = _rollout_word_at_positions(
        generated, seed_len, n_states, spaced=spaced, words=vocab_words,
    )
    word_end_indices = set(_completed_word_end_indices(word_at_step, vocab))

    xlim, ylim = _square_data_limits(gen_z, padding_frac=0.14)
    word_start = _word_start_segment_flags(prefix_labels, word_at_step)
    gray = word_colors["␣"]

    import tempfile

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    frame_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="closed_loop_frames_") as tmp:
        for t in range(2, n_states + 1):
            path = gen_z[:t]
            segment_colors = [
                gray
                if _is_return_to_baseline_segment(
                    prefix_labels[i], prefix_labels[i + 1], word_start=word_start[i],
                )
                else word_colors.get(word_at_step[i + 1], word_colors["?"])
                for i in range(t - 1)
            ]
            segment_linestyles = [":" if word_start[i] else "-" for i in range(t - 1)]
            n_seg = t - 1
            state_alphas = _trail_visible_word_alphas(
                t, word_at_step,
                visible_words=trail_visible_words,
                max_alpha=trail_fade_max_alpha,
            )
            segment_alphas = _segment_alphas_from_states(state_alphas)

            fig = plt.figure(figsize=(10.5, 8.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[0.14, 0.86], hspace=0.06)
            ax_text = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])

            idx = t - 1
            prefix = prefix_labels[idx]
            if prefix == "␣":
                output_text = "␣"
            elif prefix:
                output_text = prefix
            else:
                output_text = word_at_step[idx] if word_at_step[idx] in vocab else ""
            ax_text.axis("off")
            title_color = word_colors.get(
                word_at_step[idx], word_colors["?"],
            ) if word_at_step[idx] in vocab else "#1a1a1a"
            ax_text.text(
                0.5, 0.55, output_text,
                ha="center", va="center",
                fontsize=22, fontweight="bold", family="monospace",
                color=title_color,
                transform=ax_text.transAxes,
            )
            ax_text.text(
                0.5, 0.08,
                f"step {t}/{n_states} · seed '{seed}' · closed-loop generation",
                ha="center", va="center", fontsize=10, color="#444444",
                transform=ax_text.transAxes,
            )

            _plot_step_colored_path_arrows(
                ax, path,
                linewidth=2.0, alpha=0.88, zorder=2,
                segment_colors=segment_colors,
                segment_linestyles=segment_linestyles,
                segment_alphas=segment_alphas,
            )
            ax.scatter(
                [path[-1, 0]], [path[-1, 1]],
                s=70, c="#e74c3c", zorder=10, edgecolors="white", linewidths=0.8,
            )

            _annotate_fading_prefix_vertices(
                ax, path, prefix_labels[:t], state_alphas,
                word_colors=word_colors,
                word_at_step=word_at_step[:t],
                fontsize=11.0,
                label_offset_points=16.0,
            )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.set_xlabel(f"PC1 ({pc1:.1f}%)", fontsize=11)
            ax.set_ylabel(f"PC2 ({pc2:.1f}%)", fontsize=11)

            frame_path = os.path.join(tmp, f"frame_{t - 2:04d}.png")
            fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            frame_paths.append(frame_path)
            if idx in word_end_indices and pause_frames_per_word > 0:
                frame_paths.extend([frame_path] * int(pause_frames_per_word))

        if hold_final_frames > 0 and frame_paths:
            frame_paths.extend([frame_paths[-1]] * int(hold_final_frames))

        written = _encode_frame_sequence(frame_paths, save_path, fps=fps)
        if save_path.endswith(".gif"):
            mp4_path = save_path[:-4] + ".mp4"
            mp4_written = _encode_frame_sequence(frame_paths, mp4_path, fps=fps)
            print(f"wrote {mp4_written}")
    print(f"wrote {written}")
    return written


def _style_trajectory_row_2d(
    axes: list,
    limit_arrays: list[np.ndarray],
    *,
    limit_arrays_per_axis: list[list[np.ndarray]] | None = None,
    xlabel: str,
    ylabel: str,
    model: dict | None,
    mean: np.ndarray,
    components: np.ndarray,
    fig,
    draw_vector_field: bool = True,
) -> None:
    for ax_idx, ax in enumerate(axes):
        if limit_arrays_per_axis is not None:
            arrays = (
                limit_arrays_per_axis[ax_idx]
                if ax_idx < len(limit_arrays_per_axis) else []
            )
        else:
            arrays = limit_arrays
        if not arrays:
            continue
        xlim, ylim = _square_data_limits(*arrays, padding_frac=0.12)
        if model is not None and draw_vector_field:
            grid_resolution = 26
            xs = np.linspace(xlim[0], xlim[1], grid_resolution)
            ys = np.linspace(ylim[0], ylim[1], grid_resolution)
            grid_x, grid_y = np.meshgrid(xs, ys)
            z_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            h = reconstruct_from_pca(z_grid, mean, components)
            W_hh = np.asarray(model["weights_hidden_to_hidden"])
            b_h = np.asarray(model["bias_hidden"]).ravel()
            use_relu = bool(model.get("use_relu", False))
            h_next = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)
            z_next = (h_next - mean) @ components.T
            d = z_next - z_grid
            U = d[:, 0].reshape(grid_resolution, grid_resolution)
            V = d[:, 1].reshape(grid_resolution, grid_resolution)
            ax.quiver(
                grid_x, grid_y, U, V,
                angles="xy", scale_units="xy", scale=35.0,
                width=0.0022, headwidth=3.6, headlength=4.6, headaxislength=3.6,
                color="#000000", alpha=0.18, zorder=1,
            )
        ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")


def _style_trajectory_row_3d(
    axes: list,
    limit_arrays: list[np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    fig,
) -> None:
    if not limit_arrays:
        return
    xlim, ylim, zlim = _cube_data_limits(*limit_arrays, padding_frac=0.12)
    for ax in axes:
        _apply_cube_limits_3d(ax, xlim, ylim, zlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.grid(True, linestyle=":", alpha=0.35)


def plot_space_to_space_trajectories(
    text: str,
    hidden_states: np.ndarray,
    save_path: str,
    *,
    model=None,
    free_rollout_steps: int | None = None,
    closed_loop_steps: int | None = None,
    closed_loop_seed: int = 0,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    annot_style: str = "leaders",
    condensed: CondensedView | None = None,
    words: list[str] | None = None,
    embed_method: str = "pca",
):
    """Embedding plot of every hidden-state path from one space timestep to the next.

    If `model` is provided, draw the no-input recurrent vector field in PCA
    as a faint background quiver grid.
    Closed-loop panels run one long autoregressive rollout per seed letter;
    path color and labels use greedy vocabulary word segmentation.
    """
    if condensed is not None:
        words = condensed.words or _resolve_words(text) or []
        word_paths = _vocabulary_prefix_paths(words, condensed)
        if len(word_paths) < 1:
            return
        hidden_states = condensed.hidden_states
        word_path_indices = [idxs for _word, idxs in word_paths]
        trajs = trajectories_for_embed(hidden_states, word_path_indices=word_path_indices)
        projected, mean, components, evr = fit_embed_2d_with_evr(
            hidden_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel = embed_axis_labels_2d(evr, embed_method)
        cmap = plt.get_cmap("tab20", max(len(word_paths), 1))
        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        for i, (word, idxs) in enumerate(word_paths):
            path = projected[idxs]
            color = cmap(i)
            _plot_colored_path_arrows(ax, path, color, linewidth=1.8, alpha=0.7, zorder=2)
            ax.plot([], [], color=color, linewidth=1.8, alpha=0.7, label=word)
        add_pca_point_annotations(
            ax, text, projected, spaced=condensed.spaced, automaton=automaton,
            annot_style="annots_only", prefix_labels=condensed.labels,
        )
        xlim, ylim = _square_data_limits(projected)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            _condensed_plot_title(
                f"Vocabulary word paths through trie prefixes ({len(word_paths)} words) · "
                f"{embed_dim_label(embed_method)}",
                condensed,
            )
        )
        ax.legend(title="word", loc="best", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {save_path}")
        return

    segments = corpus_segments(text, list(_corpus_vocab(text) or []), spaced=spaced)
    if len(text) < 2 or not segments:
        return

    trajs = trajectories_for_embed(hidden_states, segments=segments)
    projected2d, mean2d, components2d, evr2d = fit_embed_2d_with_evr(
        hidden_states, method=embed_method, trajectories=trajs,
    )
    xlabel2d, ylabel2d = embed_axis_labels_2d(evr2d, embed_method)
    vocab_words = _trajectory_vocabulary_words(text, words)
    max_word_len = _longest_vocabulary_word_length(vocab_words)
    if free_rollout_steps is None:
        free_rollout_steps = max_word_len
    closed_loop_rollout_steps = _closed_loop_rollout_steps(
        vocab_words, closed_loop_steps, spaced=spaced,
    )
    seed_letters = _trajectory_seed_letters(model, vocab_words) if model is not None else []
    closed_loop_summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=spaced)
    closed_loop_summary_seed = _closed_loop_summary_seed(
        vocab_words, seed_letters, spaced=spaced,
    )
    word_colors = _vocab_word_colors(vocab_words)

    if model is not None and seed_letters:
        n_random_seeds = _INTERNAL_RANDOM_HIDDEN_SEED_COUNT
        random_hidden_seeds = _random_hidden_seeds(
            hidden_states.shape[1], n_random_seeds,
            rng_seed=int(closed_loop_seed) + 7919,
            reference_states=hidden_states,
        )

        fig = plt.figure(figsize=(38, 19))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.16)
        ax_letter_2d = fig.add_subplot(gs[0, 0])
        ax_random_2d = fig.add_subplot(gs[0, 1])
        ax_train_2d = fig.add_subplot(gs[0, 2])
        ax_gen_2d = fig.add_subplot(gs[0, 3])
        ax_letter_step = fig.add_subplot(gs[1, 0])
        ax_random_step = fig.add_subplot(gs[1, 1])
        ax_train_step = fig.add_subplot(gs[1, 2])
        ax_gen_step = fig.add_subplot(gs[1, 3])

        limits_letter_2d: list[np.ndarray] = []
        limits_random_2d: list[np.ndarray] = []
        limits_train_2d: list[np.ndarray] = []
        limits_gen_2d: list[np.ndarray] = []
        limits_letter_step: list[np.ndarray] = []
        limits_random_step: list[np.ndarray] = []
        limits_train_step: list[np.ndarray] = []
        limits_gen_step: list[np.ndarray] = []

        internal_2d = _plot_trajectory_internal_panel(
            ax_letter_2d, model, seed_letters, int(free_rollout_steps),
            mean2d, components2d, hidden_states.shape[1], is_3d=False,
        )
        limits_letter_2d.extend(internal_2d)
        random_2d = _plot_trajectory_random_internal_panel(
            ax_random_2d, model, random_hidden_seeds, int(free_rollout_steps),
            mean2d, components2d, is_3d=False,
        )
        limits_random_2d.extend(random_2d)
        internal_step = _plot_trajectory_internal_panel_step_colored(
            ax_letter_step, model, seed_letters, int(free_rollout_steps),
            mean2d, components2d, hidden_states.shape[1],
        )
        limits_letter_step.extend(internal_step)
        random_step = _plot_trajectory_random_internal_panel_step_colored(
            ax_random_step, model, random_hidden_seeds, int(free_rollout_steps),
            mean2d, components2d,
        )
        limits_random_step.extend(random_step)

        trained_2d, n_trained = _plot_trajectory_trained_panel(
            ax_train_2d, model, vocab_words, segments, projected2d,
            mean2d, components2d, max_word_len, word_colors=word_colors, is_3d=False,
        )
        limits_train_2d.extend(trained_2d)
        trained_step, _ = _plot_trajectory_trained_panel_step_colored(
            ax_train_step, model, vocab_words, segments, projected2d,
            mean2d, components2d, max_word_len,
        )
        limits_train_step.extend(trained_step)

        _plot_trajectory_closed_loop_panel(
            ax_gen_2d, model, [closed_loop_summary_seed],
            closed_loop_summary_steps, closed_loop_seed, mean2d, components2d,
            limits_gen_2d, vocab_words=vocab_words, word_colors=word_colors,
            spaced=spaced, is_3d=False,
        )
        _plot_trajectory_closed_loop_step_colored_panel(
            ax_gen_step, model, [closed_loop_summary_seed],
            closed_loop_summary_steps, closed_loop_seed, mean2d, components2d,
            limits_gen_step, vocab_words=vocab_words, spaced=spaced,
            max_word_len=max_word_len,
        )

        top_axes = [ax_letter_2d, ax_random_2d, ax_train_2d, ax_gen_2d]
        bottom_axes = [ax_letter_step, ax_random_step, ax_train_step, ax_gen_step]
        per_axis_top = [
            limits_letter_2d, limits_random_2d, limits_train_2d, limits_gen_2d,
        ]
        per_axis_bottom = [
            limits_letter_step, limits_random_step, limits_train_step, limits_gen_step,
        ]
        _style_trajectory_row_2d(
            top_axes, [],
            limit_arrays_per_axis=per_axis_top,
            xlabel=xlabel2d, ylabel=ylabel2d, model=model,
            mean=mean2d, components=components2d, fig=fig,
        )
        _style_trajectory_row_2d(
            bottom_axes, [],
            limit_arrays_per_axis=per_axis_bottom,
            xlabel=xlabel2d, ylabel=ylabel2d, model=model,
            mean=mean2d, components=components2d, fig=fig,
        )
        _add_trajectory_word_legend(fig, word_colors)
        _add_trajectory_step_colorbar(
            fig, bottom_axes, max_word_len,
            label="step in word",
        )

        ax_letter_2d.set_title(
            f"Internal dynamics · letter seed\n"
            f"{len(seed_letters)} seed letters × {max_word_len} steps · black = mean"
        )
        ax_random_2d.set_title(
            f"Internal dynamics · random hidden seed\n"
            f"{n_random_seeds} uniform h₀ × {max_word_len} steps"
        )
        ax_train_2d.set_title(
            f"Trained (teacher-forced) · 2D\n"
            f"{n_trained} vocabulary words · black = mean across words"
        )
        ax_gen_2d.set_title(
            f"Closed-loop generation · 2D\n"
            f"seed '{closed_loop_summary_seed}' · {closed_loop_summary_steps} steps · "
            f"faint = sample, black = mean ({_CLOSED_LOOP_AVERAGE_TRIALS} trials)"
        )
        ax_letter_step.set_title(
            f"Internal dynamics · letter seed · step color\n"
            f"{len(seed_letters)} seed letters × {max_word_len} steps · black = mean"
        )
        ax_random_step.set_title(
            f"Internal dynamics · random hidden seed · step color\n"
            f"{n_random_seeds} uniform h₀ × {max_word_len} steps"
        )
        ax_train_step.set_title(
            f"Trained (teacher-forced) · step color\n"
            f"{n_trained} vocabulary words · color = char index in word"
        )
        ax_gen_step.set_title(
            f"Closed-loop generation · step color\n"
            f"seed '{closed_loop_summary_seed}' · {closed_loop_summary_steps} steps · "
            f"color = in-word step"
        )

        fig.subplots_adjust(left=0.03, right=0.88, bottom=0.04, top=0.96)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {save_path}")
        return

    # Single 2D row when model unavailable.
    projected = projected2d
    mean, components = mean2d, components2d
    xlabel, ylabel = xlabel2d, ylabel2d
    ncols = 1
    fig, axes = plt.subplots(1, ncols, figsize=(12, 10), constrained_layout=True)
    axes = np.atleast_1d(axes)
    ax_free = None
    ax_paths = axes[0]
    ax_gen = None

    rollout_paths: list[np.ndarray] = []
    limit_arrays: list[np.ndarray] = []

    trained_examples = _trained_word_examples(segments, vocab_words)
    trained_paths = _plot_trained_word_examples(
        ax_paths, projected, trained_examples,
        max_word_len=max_word_len, annotate_fontsize=9.5, is_3d=False,
    )
    n_trained = len(trained_examples)
    limit_arrays.extend(trained_paths)

    limit_arrays.extend(rollout_paths)
    if not limit_arrays:
        limit_arrays = [projected]
    xlim, ylim = _square_data_limits(*limit_arrays)

    panel_axes = [ax_paths]
    _add_trajectory_step_colorbar(fig, panel_axes, max_word_len)

    for ax in panel_axes:
        ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")

    ax_paths.set_title(
        f"Trained (observed) trajectories ({embed_dim_label(embed_method)})\n"
        f"{n_trained} vocabulary words · labels = in-word prefix"
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _trajectory_figure_sibling(save_path: str, suffix: str) -> str:
    """Derive a sibling output path, e.g. foo_3d.png + '_trained' -> foo_3d_trained.png."""
    p = Path(save_path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


def _save_trajectory_figure(fig, save_path: str) -> None:
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_space_to_space_trajectories_3d(
    text: str,
    hidden_states: np.ndarray,
    save_path: str,
    *,
    model=None,
    free_rollout_steps: int | None = None,
    closed_loop_steps: int | None = None,
    closed_loop_seed: int = 0,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    condensed: CondensedView | None = None,
    words: list[str] | None = None,
    embed_method: str = "pca",
):
    """3D embedding plot of hidden-state paths from one space timestep to the next."""
    if condensed is not None:
        words = condensed.words or _resolve_words(text) or []
        word_paths = _vocabulary_prefix_paths(words, condensed)
        if len(word_paths) < 1:
            return
        hidden_states = condensed.hidden_states
        word_path_indices = [idxs for _word, idxs in word_paths]
        trajs = trajectories_for_embed(hidden_states, word_path_indices=word_path_indices)
        projected, _, _, evr = fit_embed_3d_with_evr(
            hidden_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel, zlabel = embed_axis_labels_3d(evr, embed_method)
        cmap = plt.get_cmap("tab20", max(len(word_paths), 1))
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i, (word, idxs) in enumerate(word_paths):
            path = projected[idxs]
            color = cmap(i)
            _plot_colored_path_arrows_3d(ax, path, color, linewidth=1.8, alpha=0.7)
            ax.plot([], [], [], color=color, linewidth=1.8, alpha=0.7, label=word)
        _add_pca_prefix_labels_3d(
            ax, text, projected, spaced=condensed.spaced, prefix_labels=condensed.labels,
        )
        xlim, ylim, zlim = _cube_data_limits(projected)
        _apply_cube_limits_3d(ax, xlim, ylim, zlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(
            _condensed_plot_title(
                f"Vocabulary word paths through trie prefixes ({len(word_paths)} words) · "
                f"3D {embed_dim_label(embed_method)}",
                condensed,
            )
        )
        ax.legend(title="word", loc="best", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {save_path}")
        return

    segments = corpus_segments(text, list(_corpus_vocab(text) or []), spaced=spaced)
    if len(text) < 2 or not segments:
        return

    trajs = trajectories_for_embed(hidden_states, segments=segments)
    projected, mean, components, evr = fit_embed_3d_with_evr(
        hidden_states, method=embed_method, trajectories=trajs,
    )
    xlabel, ylabel, zlabel = embed_axis_labels_3d(evr, embed_method)
    vocab_words = _trajectory_vocabulary_words(text, words)
    max_word_len = _longest_vocabulary_word_length(vocab_words)
    if free_rollout_steps is None:
        free_rollout_steps = max_word_len
    closed_loop_rollout_steps = _closed_loop_rollout_steps(
        vocab_words, closed_loop_steps, spaced=spaced,
    )
    closed_loop_summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=spaced)
    seed_letters = _trajectory_seed_letters(model, vocab_words) if model is not None else []
    word_colors = _vocab_word_colors(vocab_words)

    trained_examples = _trained_word_examples(segments, vocab_words)
    if model is not None:
        trained_tf = _teacher_forced_vocab_trajectories(
            model, vocab_words, mean=mean, components=components, max_word_len=max_word_len,
        )
        n_trained = len(trained_tf)
    else:
        trained_tf = []
        n_trained = len(trained_examples)
    n_trained_rows = int(math.ceil(n_trained / min(4, max(1, n_trained))))
    n_loop_cols = min(4, max(1, len(seed_letters)))
    n_loop_rows = int(math.ceil(len(seed_letters) / n_loop_cols))

    rollout_paths: list[np.ndarray] = []

    if model is not None and seed_letters:
        # --- Figure 1: internal dynamics (main panel only) ---
        fig_main = plt.figure(figsize=(12, 10))
        ax_free = fig_main.add_subplot(111, projection="3d")
        main_limits: list[np.ndarray] = []
        for seed_char in seed_letters:
            zs = _letter_seed_no_input_trajectory_pca(
                model,
                seed_char=seed_char,
                steps=int(free_rollout_steps),
                mean=mean,
                components=components,
                hidden_size=hidden_states.shape[1],
            )
            if zs.shape[0] < 2:
                continue
            rollout_paths.append(zs)
            main_limits.append(zs)
            _plot_step_colored_path_arrows_3d(ax_free, zs, linewidth=1.2, alpha=0.45)
            _plot_return_to_start_segment(ax_free, zs, is_3d=True)
            _annotate_trajectory_labels(
                ax_free, zs, [seed_char] * len(zs),
                colors=_step_path_colors(len(zs)), fontsize=9.0,
            )
        if not main_limits:
            main_limits = [projected]
        xlim, ylim, zlim = _cube_data_limits(*main_limits)
        _apply_cube_limits_3d(ax_free, xlim, ylim, zlim)
        ax_free.set_xlabel(xlabel)
        ax_free.set_ylabel(ylabel)
        ax_free.set_zlabel(zlabel)
        ax_free.grid(True, linestyle=":", alpha=0.35)
        _add_trajectory_step_colorbar(fig_main, [ax_free], max_word_len)
        ax_free.set_title(
            f"Internal dynamics (no input)\n"
            f"{len(seed_letters)} seed letters × {max_word_len} chars",
            fontsize=12,
        )
        fig_main.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.92)
        _save_trajectory_figure(fig_main, save_path)

        # --- Figure 2: teacher-forced vocabulary words ---
        if trained_tf:
            ncols_trained = min(4, max(1, n_trained))
            fig_trained = plt.figure(figsize=(22, max(6, 2.8 * n_trained_rows)))
            trained_gs = fig_trained.add_gridspec(
                n_trained_rows, ncols_trained, hspace=0.48, wspace=0.30,
            )
            fig_trained.suptitle(
                f"Trained (teacher-forced) · {n_trained} vocabulary words",
                fontsize=13, fontweight="bold", y=0.98,
            )
            _plot_teacher_forced_vocab_grid(
                fig_trained, trained_gs, trained_tf,
                is_3d=True, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                annotate_fontsize=8.5,
            )
            fig_trained.subplots_adjust(top=0.93, bottom=0.06)
            _save_trajectory_figure(
                fig_trained, _trajectory_figure_sibling(save_path, "_trained"),
            )

        # --- Figure 3: closed-loop generation per seed letter ---
        fig_loop = plt.figure(figsize=(22, max(6, 3.0 * n_loop_rows)))
        loop_gs = fig_loop.add_gridspec(
            n_loop_rows, n_loop_cols, hspace=0.48, wspace=0.30,
        )
        fig_loop.suptitle(
            f"Closed-loop generation · {len(seed_letters)} seeds × "
            f"{closed_loop_summary_steps} autoregressive steps · segmented word color",
            fontsize=13, fontweight="bold", y=0.98,
        )
        loop_limits: list[np.ndarray] = []
        _plot_letter_seed_closed_loop_grid(
            fig_loop, loop_gs, model,
            seed_letters=seed_letters,
            steps=closed_loop_summary_steps,
            closed_loop_seed=closed_loop_seed,
            mean=mean,
            components=components,
            limit_arrays=loop_limits,
            vocab_words=vocab_words,
            word_colors=word_colors,
            spaced=spaced,
            is_3d=True,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            annotate_fontsize=8.5,
        )
        fig_loop.subplots_adjust(top=0.93, bottom=0.06)
        _save_trajectory_figure(
            fig_loop, _trajectory_figure_sibling(save_path, "_closed_loop"),
        )
        return

    # Fallback: model unavailable — single trained-only figure.
    fig = plt.figure(figsize=(12, 10))
    ax_paths = fig.add_subplot(111, projection="3d")
    limit_arrays: list[np.ndarray] = []
    if trained_examples:
        limit_arrays = _plot_trained_word_examples(
            ax_paths, projected, trained_examples,
            max_word_len=max_word_len, annotate_fontsize=9.5, is_3d=True,
        )
    if not limit_arrays:
        limit_arrays = [projected]
    xlim, ylim, zlim = _cube_data_limits(*limit_arrays)
    _apply_cube_limits_3d(ax_paths, xlim, ylim, zlim)
    ax_paths.set_xlabel(xlabel)
    ax_paths.set_ylabel(ylabel)
    ax_paths.set_zlabel(zlabel)
    ax_paths.grid(True, linestyle=":", alpha=0.35)
    ax_paths.set_title(
        f"Trained (observed) trajectories (3D {embed_dim_label(embed_method)})\n"
        f"{n_trained} vocabulary words · labels = in-word prefix",
        fontsize=12,
    )
    fig.tight_layout()
    _save_trajectory_figure(fig, save_path)


def plot_pca_vector_field(
    text: str,
    hidden_states: np.ndarray,
    model,
    save_path: str,
    *,
    grid_resolution: int = 26,
    stride: int = 1,
    scale: float = 35.0,
    condensed: CondensedView | None = None,
    embed_method: str = "pca",
) -> None:
    """Grid vector field in embedding space: z -> z' from no-input recurrent dynamics.

    We reconstruct h from each (PC1,PC2) grid point, apply one recurrent step with x=0,
    then project back to the embedding to get the vector z' - z.
    """
    if condensed is not None:
        hidden_states = condensed.hidden_states
    if hidden_states.shape[0] < 3:
        return

    trajs = trajectories_for_embed(hidden_states)
    projected, mean, components, evr = fit_embed_2d_with_evr(
        hidden_states, method=embed_method, trajectories=trajs,
    )
    z = projected

    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # z -> h (2D PCA reconstruction)
    h = reconstruct_from_pca(z_grid, mean, components)

    W_hh = np.asarray(model["weights_hidden_to_hidden"])
    b_h = np.asarray(model["bias_hidden"])
    use_relu = bool(model.get("use_relu", False))
    h_next = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)

    # h' -> z' via the same PCA projection
    z_next = (h_next - mean) @ components.T
    d = z_next - z_grid

    U = d[:, 0].reshape(grid_resolution, grid_resolution)
    V = d[:, 1].reshape(grid_resolution, grid_resolution)
    mask = np.ones_like(U, dtype=bool)
    if stride > 1:
        mask[:] = False
        mask[::stride, ::stride] = True

    fig, ax = plt.subplots(figsize=(10.5, 9.0), constrained_layout=True)
    ax.scatter(
        z[:, 0],
        z[:, 1],
        s=14,
        c="0.4",
        alpha=0.22,
        edgecolors="none",
        zorder=1,
    )
    ax.quiver(
        grid_x[mask],
        grid_y[mask],
        U[mask],
        V[mask],
        angles="xy",
        scale_units="xy",
        scale=max(scale, 1e-6),
        width=0.0026,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.0,
        color="#000000",
        alpha=0.9,
        zorder=3,
    )

    xlabel, ylabel = embed_axis_labels_2d(evr, embed_method)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        _condensed_plot_title(
            f"Vector field in {embed_dim_label(embed_method)} "
            f"(grid; no-input recurrent dynamics)",
            condensed,
        )
    )
    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_dfa_analysis(
    text,
    hidden_states,
    chars,
    words: list[str],
    save_path,
    automaton: MinimizedVocabAutomaton,
    *,
    model=None,
    spaced: bool = False,
    annot_style: str = "leaders",
    condensed: CondensedView | None = None,
    repr_name: str | None = None,
    embedding: str | None = None,
    embed_method: str = "pca",
):
    """Embedding beside the min-DFA with matching state colors."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        prefix_labels = condensed.labels
        spaced = condensed.spaced
    else:
        prefix_labels = None
    if hidden_states.shape[0] < 2:
        return

    trajs = trajectories_for_embed(hidden_states)
    projected, _, _, evr = fit_embed_2d_with_evr(
        hidden_states, method=embed_method, trajectories=trajs,
    )
    xlabel, ylabel = embed_axis_labels_2d(evr, embed_method)
    embed_title = f"2D {embed_dim_label(embed_method)}"
    if embed_method == "jpca":
        embed_subtitle = f"rotation rate: ω={float(evr[0]):.3f}" if len(evr) else ""
    else:
        embed_subtitle = (
            f"variance explained: PC1 {100.0 * float(evr[0]):.1f}%, PC2 {100.0 * float(evr[1]):.1f}%"
            if len(evr) > 1
            else ""
        )
    if prefix_labels is not None:
        state_ids = [
            dfa_state_for_prefix(p, automaton, spaced=spaced) for p in prefix_labels
        ]
    else:
        state_ids = [
            dfa_state_at_position(
                text, i, automaton, spaced=spaced, vocab=_corpus_vocab(text),
            ) for i in range(len(text))
        ]
    state_colors = _dfa_automaton_state_colors(automaton)

    fig, axes = plt.subplots(1, 2, figsize=(28, 11), constrained_layout=True)
    ax_dfa, ax_embed = axes[0], axes[1]

    draw_minimized_dfa_on_axes(ax_dfa, automaton, words, state_colors=state_colors)
    ax_dfa.set_title("Minimal DFA", fontsize=12, pad=12)

    text_positions = add_dfa_state_annotations(
        ax_embed, text, projected, automaton,
        spaced=spaced, state_colors=state_colors,
        point_size=160,
        label_fontsize=18,
        leader_linewidth=0.65,
        annot_style=annot_style,
        prefix_labels=prefix_labels,
    )
    _expand_limits_for_annotations(
        ax_embed, projected, text_positions,
        (projected[:, 0].min(), projected[:, 0].max()),
        (projected[:, 1].min(), projected[:, 1].max()),
    )
    ax_embed.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax_embed.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax_embed.set_xlabel(xlabel)
    ax_embed.set_ylabel(ylabel)
    ctx = prefix_axis_label(spaced=spaced, text=text)
    subtitle = f"\n{embed_subtitle}" if embed_subtitle else ""
    rep = representation_label(model, repr_name=repr_name) if model is not None else (
        repr_name or "hidden state h"
    )
    ax_embed.set_title(
        f"{embed_title} of {rep} (min DFA state · {ctx}){subtitle}"
    )
    ax_embed.grid(True, linestyle=":", alpha=0.35)
    ax_embed.spines["top"].set_visible(False)
    ax_embed.spines["right"].set_visible(False)
    ax_embed.tick_params(top=False, right=False)

    fig.suptitle(
        _condensed_plot_title(", ".join(words), condensed),
        fontsize=12, y=1.01,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_prediction_regions(
    model,
    text,
    hidden_states,
    chars,
    save_path,
    grid_resolution=120,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    condensed: CondensedView | None = None,
    repr_name: str | None = None,
    embed_method: str = "pca",
):
    """Embedding panels: argmax next-char regions and softmax entropy, with context labels."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        prefix_labels = condensed.labels
        spaced = condensed.spaced
    else:
        prefix_labels = None
    n_points, hidden_size = hidden_states.shape
    vocab_size = len(chars)
    if n_points < 2 or hidden_size < 1 or (len(text) == 0 and prefix_labels is None):
        return

    trajs = _embed_trajectories_for_text(text, hidden_states, spaced=spaced, words=None)
    grid_x, grid_y, grid_hidden, projected, xlim, ylim, evr = build_pca_plane_grid(
        text, hidden_states, grid_resolution, spaced=spaced, prefix_labels=prefix_labels,
        embed_method=embed_method, trajectories=trajs,
    )
    probs = next_char_probabilities(model, grid_hidden)
    grid_pred = np.argmax(probs, axis=1).reshape(grid_resolution, grid_resolution)
    grid_entropy = prediction_entropy(probs).reshape(grid_resolution, grid_resolution)
    max_entropy = float(np.log(vocab_size))
    avoid_xy = trigram_avoidance_points(
        text, projected, spaced=spaced, automaton=automaton, prefix_labels=prefix_labels,
    )
    plane_span = max(
        float(np.ptp(grid_x)),
        float(np.ptp(grid_y)),
        float(np.ptp(projected[:, 0])),
        float(np.ptp(projected[:, 1])),
        1e-3,
    )
    avoid_radius = plane_span * 0.12

    pred_cmap = plt.get_cmap("tab10", vocab_size)
    fig, axes = plt.subplots(1, 2, figsize=(24, 11), constrained_layout=True)
    panel_specs = [
        (
            axes[0],
            grid_pred,
            dict(
                levels=np.arange(-0.5, vocab_size, 1),
                cmap=pred_cmap,
                vmin=None,
                vmax=None,
            ),
            "Argmax next-char (2D-reconstructed "
            f"{representation_label(model, prob_grid=True, repr_name=repr_name)})",
        ),
        (
            axes[1],
            grid_entropy,
            dict(
                levels=20,
                cmap="magma",
                alpha=0.85,
                vmin=0.0,
                vmax=max_entropy,
            ),
            f"Prediction entropy (max = ln {vocab_size} ≈ {max_entropy:.2f} nats)",
        ),
    ]

    for ax, field, contour_kw, title in panel_specs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        im = ax.contourf(grid_x, grid_y, field, antialiased=True, zorder=1, **contour_kw)
        if ax is axes[0]:
            add_argmax_region_labels(
                ax, grid_x, grid_y, grid_pred, chars,
                avoid_xy=avoid_xy, avoid_radius=avoid_radius,
                xlim=xlim, ylim=ylim,
            )
        add_pca_point_annotations(
            ax, text, projected, spaced=spaced, automaton=automaton,
            prefix_labels=prefix_labels,
        )
        axis_x, axis_y = embed_axis_labels_2d(evr, embed_method)
        ax.set_xlabel(axis_x)
        ax.set_ylabel(axis_y)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.35)
        if ax is axes[1]:
            fig.colorbar(im, ax=ax, label="entropy (nats)", fraction=0.046, pad=0.02, shrink=0.85)

    if automaton is not None:
        pca_ctx = (
            "min DFA state (prefix since last space)" if spaced
            else "min DFA state (in-word prefix)"
        )
    else:
        pca_ctx = "prefix after space" if spaced else prefix_axis_label(spaced=spaced, text=text)
    fig.suptitle(
        _condensed_plot_title(
            f"{embed_dim_label(embed_method)} of {representation_label(model)} · {pca_ctx} · "
            f"{original_vocabulary_title(chars, text)}",
            condensed,
        ),
        fontsize=12, y=1.01,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_next_char_probability_panels(
    model,
    text,
    hidden_states,
    chars,
    save_path,
    grid_resolution=120,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    condensed: CondensedView | None = None,
    embed_method: str = "pca",
):
    """One panel per vocab char: P(next = char) over the embedding plane (from softmax)."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        prefix_labels = condensed.labels
        spaced = condensed.spaced
    else:
        prefix_labels = None
    n_points, hidden_size = hidden_states.shape
    vocab_size = len(chars)
    if n_points < 2 or hidden_size < 1:
        return

    trajs = _embed_trajectories_for_text(text, hidden_states, spaced=spaced, words=None)
    grid_x, grid_y, grid_hidden, projected, xlim, ylim, evr = build_pca_plane_grid(
        text, hidden_states, grid_resolution, spaced=spaced, prefix_labels=prefix_labels,
        embed_method=embed_method, trajectories=trajs,
    )
    probs = next_char_probabilities(model, grid_hidden)

    ncols = min(3, vocab_size)
    nrows = (vocab_size + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, 3.9 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    last_im = None

    for char_index, (ax, char) in enumerate(zip(axes, chars)):
        field = probs[:, char_index].reshape(grid_resolution, grid_resolution)
        last_im = ax.contourf(
            grid_x, grid_y, field,
            levels=np.linspace(0, 1, 21),
            cmap="viridis", vmin=0, vmax=1, antialiased=True,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"P(next = {display_char(char)!r})")
        ax.set_aspect("equal", adjustable="box")
        add_pca_point_annotations(
            ax, text, projected, spaced=spaced, automaton=automaton,
            prefix_labels=prefix_labels,
        )

    for ax in axes[vocab_size:]:
        ax.axis("off")

    axis_x, axis_y = embed_axis_labels_2d(evr, embed_method)
    axes[0].set_ylabel(axis_y)
    axes[(nrows - 1) * ncols].set_xlabel(axis_x)
    if nrows > 1:
        for row in range(1, nrows):
            axes[row * ncols].set_ylabel(axis_y)
        for col in range(1, ncols):
            bottom = min((nrows - 1) * ncols + col, vocab_size - 1)
            if bottom < vocab_size:
                axes[bottom].set_xlabel(axis_x)

    fig.colorbar(last_im, ax=axes[:vocab_size], label="probability", shrink=0.92)
    fig.suptitle(
        _condensed_plot_title(
            f"P(next char | {representation_label(model, prob_grid=True)}) · "
            f"{original_vocabulary_title(chars, text)}",
            condensed,
        ),
        fontsize=11, y=1.02,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def char_axis_labels(chars):
    """Tick labels for the vocabulary axis (readable for whitespace)."""
    return [display_char(c) for c in chars]


def symmetric_abs_vmax(*matrices):
    return float(max(np.max(np.abs(m)) for m in matrices))


def hidden_unit_labels(dale_sign, hidden_size: int) -> list[str]:
    if dale_sign is None or len(dale_sign) != hidden_size:
        return [f"h{i}" for i in range(hidden_size)]
    return [f"h{i}({'E' if s > 0 else 'I'})" for i, s in enumerate(dale_sign)]


def ei_block_boundary(dale_sign) -> int | None:
    """Index between E and I blocks (line drawn between n_E-1 and n_E)."""
    if dale_sign is None:
        return None
    n_exc = int(np.sum(np.asarray(dale_sign) > 0))
    if 0 < n_exc < len(dale_sign):
        return n_exc
    return None


def weights_for_plot(model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Return W_xh, W_hh, W_ho (and dale_sign) in E-first / I-last order."""
    from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

    W_in = np.asarray(model["weights_input_to_hidden"])
    W_rec = np.asarray(model["weights_hidden_to_hidden"])
    W_out = np.asarray(model["weights_hidden_to_output"])
    b_h = np.asarray(model["bias_hidden"])
    dale_sign = model.get("dale_sign")
    if dale_sign is not None and len(dale_sign) == W_in.shape[0]:
        dale_sign = np.asarray(dale_sign, dtype=float)
        if not dale_signs_ordered(dale_sign):
            W_in, W_rec, W_out, b_h, dale_sign = permute_hidden_by_dale(
                W_in, W_rec, W_out, b_h, dale_sign,
            )
    return W_in, W_rec, W_out, dale_sign


def _draw_ei_guides(ax, boundary: int | None, *, horizontal: bool, vertical: bool) -> None:
    if boundary is None:
        return
    if horizontal:
        ax.axhline(boundary - 0.5, color="black", lw=1.0, ls="--")
    if vertical:
        ax.axvline(boundary - 0.5, color="black", lw=1.0, ls="--")


def plot_learned_weights(model, save_path: str):
    """Input (W_xh) and hidden recurrent (W_hh); E columns red, I blue, 0 white."""
    W_in, W_rec, _W_out, dale_sign = weights_for_plot(model)
    chars = model["chars"]
    hidden_size, vocab_size = W_in.shape
    unit_labels = hidden_unit_labels(dale_sign, hidden_size)
    boundary = ei_block_boundary(dale_sign)

    # Hidden units are columns: E block (red) then I block (blue).
    W_input = W_in.T
    W_hidden = W_rec
    vmax = max(symmetric_abs_vmax(W_input, W_hidden), 1e-9)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(8, vocab_size * 0.5 + hidden_size * 0.45), max(3.5, hidden_size * 0.55)),
        constrained_layout=True,
    )
    cmap = plt.cm.RdBu_r

    im0 = axes[0].imshow(
        W_input, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[0].set_title("Input")
    axes[0].set_xlabel("hidden unit (E | I)")
    axes[0].set_ylabel("input character")
    axes[0].set_xticks(range(hidden_size))
    axes[0].set_xticklabels(unit_labels, fontsize=6, rotation=90)
    axes[0].set_yticks(range(vocab_size))
    axes[0].set_yticklabels(char_axis_labels(chars), fontsize=8)
    _draw_ei_guides(axes[0], boundary, horizontal=False, vertical=True)

    im1 = axes[1].imshow(
        W_hidden, aspect="equal", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[1].set_title("Hidden")
    axes[1].set_xlabel("source h (E | I)")
    axes[1].set_ylabel("target h (E | I)")
    axes[1].set_xticks(range(hidden_size))
    axes[1].set_xticklabels(unit_labels, fontsize=6, rotation=90)
    axes[1].set_yticks(range(hidden_size))
    axes[1].set_yticklabels(unit_labels, fontsize=6)
    _draw_ei_guides(axes[1], boundary, horizontal=True, vertical=True)

    fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.02, label="weight (E red, I blue)")
    fig.suptitle("Learned weights", y=1.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _mean_in_per_unit(W_in: np.ndarray, W_rec: np.ndarray) -> np.ndarray:
    """Mean weight over all connections into each hidden unit (input + recurrent row)."""
    hidden_size = W_in.shape[0]
    return np.array([
        np.mean(np.concatenate([W_in[i], W_rec[i]]))
        for i in range(hidden_size)
    ])


def _mean_out_per_unit(W_rec: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """Mean weight over all connections out of each hidden unit (recurrent col + readout col)."""
    hidden_size = W_rec.shape[0]
    return np.array([
        np.mean(np.concatenate([W_rec[:, j], W_out[:, j]]))
        for j in range(hidden_size)
    ])


def _extract_ei_block(
    W_in: np.ndarray,
    W_hh: np.ndarray,
    *,
    dale_sign: np.ndarray,
    layer: str,
    post: str,
    pre: str,
    vocab_size: int,
) -> np.ndarray:
    """Flatten one E/I submatrix (target row E/I × source E/I)."""
    from rnn.rnn_dyn import dale_ei_blocks

    exc, inh = dale_ei_blocks(dale_sign)
    post_idx = exc if post == "E" else inh
    pre_idx = exc if pre == "E" else inh
    if layer == "xh":
        # Input has no E/I; map pre E/I to first/second half of character alphabet.
        mid = max(vocab_size // 2, 1)
        cols = np.arange(0, mid) if pre == "E" else np.arange(mid, vocab_size)
        if len(post_idx) == 0 or len(cols) == 0:
            return np.array([])
        return W_in[np.ix_(post_idx, cols)].ravel()
    if len(post_idx) == 0 or len(pre_idx) == 0:
        return np.array([])
    return W_hh[np.ix_(post_idx, pre_idx)].ravel()


def _collect_block_weights(
    snaps: np.ndarray,
    *,
    hidden_size: int,
    vocab_size: int,
    dale_sign: np.ndarray,
    layer: str,
    post: str,
    pre: str,
) -> np.ndarray:
    """Weight trajectories for one block; shape (n_snap, n_syn), sorted by |w| range."""
    from rnn.rnn_dyn import unpack_weight_snapshot

    rows = []
    for vec in snaps:
        W_in, W_hh, _ = unpack_weight_snapshot(vec, hidden_size, vocab_size)
        block = _extract_ei_block(
            W_in, W_hh, dale_sign=dale_sign, layer=layer, post=post, pre=pre,
            vocab_size=vocab_size,
        )
        rows.append(block)

    max_len = max((s.size for s in rows), default=0)
    if max_len == 0:
        return np.zeros((len(rows), 0))
    out = np.full((len(rows), max_len), np.nan)
    for t, s in enumerate(rows):
        out[t, : s.size] = s
    spread = np.nanmax(out, axis=0) - np.nanmin(out, axis=0)
    order = np.argsort(spread)[::-1]
    return out[:, order]


def _panel_vmax(data: np.ndarray, pct: float = 99.0) -> float:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1e-3
    return max(float(np.percentile(np.abs(finite), pct)), 1e-4)


def _plot_ei_block_panel(ax, data, iters, title: str) -> object | None:
    if data.size == 0 or data.shape[1] == 0:
        ax.text(0.5, 0.5, "no synapses", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return None
    vmax = _panel_vmax(data)
    im = ax.imshow(
        data.T,
        aspect="auto",
        cmap=plt.cm.RdBu_r,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        origin="lower",
    )
    ax.set_title(f"{title}\n(n={data.shape[1]} syns)", fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("synapse (sorted by |w| range)")
    if len(iters) > 0:
        tick_idx = np.linspace(0, len(iters) - 1, min(6, len(iters)), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(iters[i]) for i in tick_idx], fontsize=7)
    return im


def plot_weight_dynamics_over_training(model, save_path: str) -> None:
    """Eight weight heatmaps over training: 4× W_xh + 4× W_hh (EE/EI/IE/II)."""
    if "weight_snap_outgoing" not in model:
        print(f"skip {save_path}: re-run training with --save-snapshots to record weight snapshots")
        return

    snaps = np.asarray(model["weight_snap_outgoing"], dtype=float)
    iters = np.asarray(model["weight_snap_iterations"], dtype=int)
    if snaps.ndim != 2 or snaps.shape[0] < 2:
        print(f"skip {save_path}: insufficient weight snapshot history")
        return

    dale_sign = model.get("dale_sign")
    if dale_sign is None or len(dale_sign) != int(model["hidden_size"]):
        print(f"skip {save_path}: Dale sign vector required for E/I blocks")
        return

    from rnn.rnn_dyn import snapshot_vector_layout

    hidden_size = int(model["hidden_size"])
    vocab_size = int(model["vocab_size"])
    layout = snapshot_vector_layout(hidden_size, vocab_size, snaps.shape[1])
    if layout == "outgoing":
        print(
            f"skip {save_path}: re-run training for full snapshots "
            "(need W_xh + W_hh in weight_snap_outgoing)",
        )
        return

    viol = np.asarray(model.get("weight_snap_violation_frac", []), dtype=float)
    dale_sign = np.asarray(dale_sign, dtype=float)

    xh_blocks = [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")]
    hh_blocks = [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")]
    # post = target row; pre = source (vocab half for xh, hidden unit for hh).
    xh_titles = [r"$W_{xh}$ EE", r"$W_{xh}$ EI", r"$W_{xh}$ IE", r"$W_{xh}$ II"]
    hh_titles = [r"$W_{hh}$ EE", r"$W_{hh}$ EI", r"$W_{hh}$ IE", r"$W_{hh}$ II"]

    block_data = []
    for post, pre in xh_blocks:
        block_data.append(
            _collect_block_weights(
                snaps,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dale_sign=dale_sign,
                layer="xh",
                post=post,
                pre=pre,
            )
        )
    for post, pre in hh_blocks:
        block_data.append(
            _collect_block_weights(
                snaps,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dale_sign=dale_sign,
                layer="hh",
                post=post,
                pre=pre,
            )
        )

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(
        r"Weight per synapse over training (per-panel scale; E red, I blue) — "
        r"$W_{xh}$: E/I row $\times$ input half; $W_{hh}$: E/I row $\times$ E/I column",
        fontsize=10,
        y=1.02,
    )

    ims = []
    for ax, data, title in zip(axes[0], block_data[:4], xh_titles):
        im = _plot_ei_block_panel(ax, data, iters, title)
        if im is not None:
            ims.append(im)
    for ax, data, title in zip(axes[1], block_data[4:], hh_titles):
        im = _plot_ei_block_panel(ax, data, iters, title)
        if im is not None:
            ims.append(im)

    if ims:
        fig.colorbar(
            ims[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
            label="weight (E red, I blue; scale varies per panel)",
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_weight_eigenspectra(model, save_path: str) -> None:
    """Spectra, pooled weight histogram, and per-unit mean |in| / |out|."""
    W_in, W_rec, W_out, dale_sign = weights_for_plot(model)
    b_h = np.asarray(model["bias_hidden"]).ravel()
    if dale_sign is not None and len(dale_sign) == W_in.shape[0]:
        from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale
        if not dale_signs_ordered(dale_sign):
            _, _, _, b_h, _ = permute_hidden_by_dale(W_in, W_rec, W_out, b_h, dale_sign)
    b_o = np.asarray(model["bias_output"]).ravel()
    hidden_size = W_in.shape[0]
    unit_labels = hidden_unit_labels(dale_sign, hidden_size)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)

    eigs = np.linalg.eigvals(W_rec)
    ax = axes[0, 0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#888888", lw=0.9, ls="--", zorder=1)
    ax.scatter(
        eigs.real, eigs.imag,
        c=np.abs(eigs), cmap="viridis", s=55, edgecolors="black", linewidths=0.4, zorder=3,
    )
    ax.axhline(0, color="lightgrey", lw=0.6)
    ax.axvline(0, color="lightgrey", lw=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(r"$W_{hh}$ eigenvalues")
    ax.grid(True, linestyle=":", alpha=0.35)

    for ax, name, W in zip(
        axes[0, 1:],
        (r"$W_{xh}$ singular values", r"$W_{ho}$ singular values"),
        (W_in, W_out),
    ):
        singular = np.linalg.svd(W, compute_uv=False)
        ax.bar(np.arange(len(singular)), singular, color="#4c72b0", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("index")
        ax.set_ylabel("σ")
        ax.set_title(name)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    all_weights = np.concatenate([
        W_in.ravel(), W_rec.ravel(), W_out.ravel(), b_h, b_o,
    ])
    ax_hist = axes[1, 0]
    bins = np.linspace(-np.max(np.abs(all_weights)), np.max(np.abs(all_weights)), 41)
    ax_hist.hist(
        all_weights, bins=bins, color="#888888", alpha=0.55,
        edgecolor="white", linewidth=0.4, density=True, label="all",
    )
    ax_hist.hist(
        W_in.ravel(), bins=bins, color="#4c72b0", alpha=0.45,
        edgecolor="white", linewidth=0.3, density=True, label=r"$W_{xh}$ (in)",
    )
    ax_hist.hist(
        W_out.ravel(), bins=bins, color="#dd8452", alpha=0.45,
        edgecolor="white", linewidth=0.3, density=True, label=r"$W_{ho}$ (out)",
    )
    ax_hist.axvline(0, color="black", lw=0.8)
    ax_hist.set_xlabel("weight value")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Weight distributions")
    ax_hist.legend(fontsize=7, framealpha=0.9)
    ax_hist.grid(True, axis="y", linestyle=":", alpha=0.35)

    mean_in = _mean_in_per_unit(W_in, W_rec)
    mean_out = _mean_out_per_unit(W_rec, W_out)
    x = np.arange(hidden_size)
    width = 0.38

    ax_in = axes[1, 1]
    ax_in.bar(x - width / 2, mean_in, width=width, color="#4c72b0", edgecolor="black", linewidth=0.3)
    ax_in.set_xticks(x)
    ax_in.set_xticklabels(unit_labels)
    ax_in.axhline(0, color="black", lw=0.8)
    ax_in.set_ylabel("mean weight")
    ax_in.set_title("Mean incoming per unit")
    ax_in.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax_out = axes[1, 2]
    ax_out.bar(x + width / 2, mean_out, width=width, color="#dd8452", edgecolor="black", linewidth=0.3)
    ax_out.set_xticks(x)
    ax_out.set_xticklabels(unit_labels)
    ax_out.axhline(0, color="black", lw=0.8)
    ax_out.set_ylabel("mean weight")
    ax_out.set_title("Mean outgoing per unit")
    ax_out.grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("Weight spectra and distributions (final model)", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_output_probs(
    text,
    output_probs,
    chars,
    save_path,
    *,
    condensed: CondensedView | None = None,
    exp_name: str | None = None,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
    words: list[str] | None = None,
):
    """Heatmap of P(next char) over time; overlay the true next char."""
    vocab_size = len(chars)
    prefix_keys: list[str] | None = None
    if condensed is not None:
        output_probs = condensed.output_probs
        if output_probs is None:
            return
        prefix_keys = condensed.labels
        x_labels = [_display_prefix_label(l) for l in prefix_keys]
        targets = condensed.next_chars
        x_axis = prefix_axis_label(
            spaced=condensed.spaced, text=text, words=condensed.words,
        )
        spaced = condensed.spaced
        words = condensed.words
    else:
        x_labels = list(text)
        targets = list(text[1:]) + [text[0]]
        x_axis = "timestep / input character"
    length = output_probs.shape[0]
    target_indices = np.array([chars.index(c) for c in targets])

    fig, ax = plt.subplots(figsize=(max(12, length * 0.15), 4))
    im = ax.imshow(
        output_probs.T,
        aspect="auto", cmap="viridis", vmin=0, vmax=1,
        interpolation="nearest", origin="lower",
    )

    ax.set_yticks(range(vocab_size))
    ax.set_yticklabels(chars)
    ax.set_xticks(range(length))
    ax.set_xticklabels(x_labels, fontsize=7)
    if automaton is not None:
        if prefix_keys is not None:
            state_ids = _dfa_state_ids_for_prefixes(
                prefix_keys, automaton, spaced=spaced,
            )
        else:
            state_ids = _dfa_state_ids_at_timesteps(
                text, automaton, spaced=spaced, words=words,
            )
        _color_tick_labels_by_state_ids(ax.get_xticklabels(), state_ids)
        x_axis += " · tick color = min DFA state"
    ax.set_xlabel(x_axis)
    ax.set_ylabel("predicted next char")
    ax.set_title(
        _condensed_plot_title(
            "P(next char | input so far)  —  red dots = actual next char",
            condensed,
        )
    )

    ax.scatter(
        np.arange(length), target_indices,
        color="red", s=18, edgecolor="white", linewidth=0.5, zorder=3,
    )

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="probability")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def resolve_paths(args):
    """Return (model_path, input_path, out_dir, model_type) from --exp or explicit paths."""
    from experiment import checkpoint_path, input_path, plots_dir, ensure_experiment_dirs

    model_type = getattr(args, "model_type", "rnn")
    seed = getattr(args, "seed", None)
    if args.exp:
        ensure_experiment_dirs(args.exp, model_type)
        return (
            str(checkpoint_path(args.exp, model_type, seed=seed)),
            str(input_path(args.exp)),
            str(plots_dir(args.exp, model_type)),
            model_type,
        )
    out = args.out_dir if args.out_dir is not None else "plots"
    if args.model.endswith(".pt"):
        model_type = "transformer"
    return args.model, args.input, out, model_type


def load_model_for_viz(path: str, model_type: str) -> dict:
    if model_type == "transformer" or path.endswith(".pt"):
        return load_transformer_model(path)
    return load_model(path)


def run_forward_pass(model: dict, text: str, model_type: str):
    if model.get("model_type") == "transformer" or model_type == "transformer":
        return transformer_forward_pass(model, text)
    return forward_pass(model, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", default=None,
                        help="experiment name under experiments/<exp>/")
    parser.add_argument("--model-type", default="rnn", choices=["rnn", "rnn_dale", "transformer"],
                        help="which model subdirectory to visualize (default: rnn)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (loads model_seed<N>.npz; regenerates corpus from seed)",
    )
    parser.add_argument("--model", default="model.npz")
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--length", type=int, default=50,
                        help="how many characters of the corpus to visualize (default: 50)")
    parser.add_argument("--out-dir", default=None,
                        help="plot output directory (default: experiments/<exp>/<model>/plots or plots)")
    parser.add_argument("--no-cluster-per-char", action="store_true",
                        help="keep per-character heatmap rows in sequence order")
    parser.add_argument(
        "--dfa-annot-style",
        default="leaders",
        choices=["leaders", "none", "annots_only"],
        help="DFA annotation style for hidden_states_pca_dfa_analysis.png",
    )
    parser.add_argument(
        "--condensed",
        action="store_true",
        help="average vectors over equivalent in-word prefixes (trie positions); "
             "writes *_condensed.png figures (RNN hidden states or transformer representations)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="transformer only: skip slow per-char / clustermap / DFA-distance plots",
    )
    parser.add_argument(
        "--trajectories-only",
        action="store_true",
        help="only word trajectory PCA plots (block_output for transformer; fast)",
    )
    parser.add_argument(
        "--learning-dynamics-video",
        action="store_true",
        help="render supplementary hidden_state_pca learning animation (slow; off by default)",
    )
    parser.add_argument(
        "--closed-loop-video",
        action="store_true",
        help="render closed-loop trajectory animation with generated text (off by default)",
    )
    args = parser.parse_args()

    timer = VizTimer()
    model_file, input_file, out_dir, model_type = resolve_paths(args)
    os.makedirs(out_dir, exist_ok=True)
    if args.exp:
        print(f"experiment: {args.exp} ({model_type}) -> {out_dir}")

    with timer.section("load_model"):
        model = load_model_for_viz(model_file, model_type)
    is_transformer = model.get("model_type") == "transformer" or model_type == "transformer"
    print(f"loaded model: hidden_size={model['hidden_size']}, "
          f"vocab_size={model['vocab_size']}, chars={''.join(model['chars'])}")

    if not is_transformer and not args.trajectories_only:
        with timer.section("weight_plots"):
            plot_learned_weights(model, save_path=str(numbered_plot_path(out_dir, "weights.png")))
            plot_weight_eigenspectra(
                model, save_path=str(numbered_plot_path(out_dir, "weights_eigenspectra.png")),
            )
            plot_weight_dynamics_over_training(
                model, str(numbered_plot_path(out_dir, "weight_dynamics_over_training.png")),
            )
    if not args.trajectories_only:
        with timer.section("learning_curve"):
            plot_learning_curve(
                model,
                save_path=str(numbered_plot_path(out_dir, "learning_curve.png")),
            )
        with timer.section("samples_before_after"):
            plot_sample_before_after(
                model,
                save_path=str(numbered_plot_path(out_dir, "samples_before_after.png")),
            )

    with timer.section("prepare_corpus"):
        if args.exp and args.seed is not None:
            from task import corpus_for_experiment

            full_text = corpus_for_experiment(args.exp, seed=args.seed)
            print(f"corpus from seed {args.seed} ({len(full_text):,} chars)")
        else:
            with open(input_file, "r") as f:
                full_text = f.read()
        spaced = corpus_uses_word_spacing(full_text, args.exp)
        words = vocabulary_for_experiment(args.exp) if args.exp else infer_task_words(full_text)
        if words and not spaced:
            from task import label_extensions_for_experiment
            from vocab_diagrams import select_analysis_window

            extensions = label_extensions_for_experiment(args.exp) if args.exp else []
            win_start, text, label_words = select_analysis_window(
                full_text, words, args.length, spaced=spaced, extensions=extensions,
            )
            if win_start:
                print(f"analysis window starts at corpus offset {win_start}")
            aux = [w for w in label_words if w not in words]
            if aux:
                print(f"label vocabulary includes {len(aux)} extra 4–5 letter words: {aux[:8]}{'...' if len(aux) > 8 else ''}")
        else:
            text = full_text[: args.length]
            label_words = None
        print(f"running forward pass over {len(text)} characters of {input_file}")

        global _VIS_WORDS, _VIS_LABEL_WORDS
        _VIS_WORDS = words
        _VIS_LABEL_WORDS = label_words if label_words is not None else words
        automaton = build_minimized_vocabulary_automaton(words) if words else None
        if automaton is not None:
            print(
                "PCA point colors: minimized DFA state after in-word prefix since last space"
                if spaced
                else "PCA point colors: minimized DFA state after in-word prefix (vocab boundaries)"
            )
        elif spaced:
            print("annotation mode: prefix after space (e.g. h, ha, hat; ' ' on spaces)")
        elif words:
            print("annotation mode: in-word prefix via vocabulary word boundaries")

        targets = list(text[1:]) + [text[0]]

    if is_transformer:
        print(
            "transformer mode: each representation gets its own plot suite "
            "under representations/<name>/"
            if not args.trajectories_only
            else "transformer mode: trajectories only (representations/block_output/)"
        )
        with timer.section("transformer_representations"):
            acts = run_transformer_visualization(
                model,
                text,
                out_dir,
                spaced=spaced,
                automaton=automaton,
                words=words,
                condensed=args.condensed,
                quick=args.quick,
                trajectories_only=args.trajectories_only,
            )
        output_probs = acts.output_probs
    elif args.trajectories_only:
        with timer.section("forward_pass"):
            hidden_states, output_probs = run_forward_pass(model, text, model_type)
        if words:
            traj_words = label_words if label_words is not None else words
            with timer.section("trajectories"):
                _plot_embed_variants(
                    plot_space_to_space_trajectories,
                    str(numbered_plot_path(out_dir, "word_trajectories_pca.png")),
                    text=text, hidden_states=hidden_states, model=model,
                    spaced=spaced, automaton=automaton,
                    annot_style=args.dfa_annot_style, words=traj_words,
                )
                _plot_embed_variants(
                    plot_space_to_space_trajectories_3d,
                    str(numbered_plot_path(out_dir, "word_trajectories_pca_3d.png")),
                    text=text, hidden_states=hidden_states, model=model,
                    spaced=spaced, automaton=automaton, words=traj_words,
                )
                if args.closed_loop_video:
                    traj_path = numbered_plot_path(out_dir, "word_trajectories_pca.png")
                    write_closed_loop_trajectory_video(
                        model,
                        hidden_states,
                        save_path=str(traj_path.parent / "closed_loop_trajectory.gif"),
                        vocab_words=traj_words,
                        spaced=spaced,
                    )
    else:
        with timer.section("forward_pass"):
            hidden_states, output_probs = run_forward_pass(model, text, model_type)
        act_label = activation_label(use_relu=bool(model.get("use_relu", False)))

        condensed_view = None
        if args.condensed:
            condensed_view = condense_hidden_states_by_prefix(
                text, hidden_states, output_probs, spaced=spaced, words=words,
            )
            print(
                f"condensed view: {len(condensed_view.labels)} unique prefixes "
                f"(avg over {sum(condensed_view.counts)} timesteps)"
            )

        def plot_path(name: str) -> str:
            path = str(numbered_plot_path(out_dir, name))
            return _condensed_save_path(path) if args.condensed else path

        cv = condensed_view

        with timer.section("activations"):
            plot_hidden_states_heatmap(
                text, hidden_states,
                save_path=plot_path("activation_heatmap.png"),
                act_label=act_label,
                condensed=cv,
                exp_name=args.exp,
                automaton=automaton,
                spaced=spaced,
                words=words,
                cluster_units=True,
            )

            plot_output_probs(
                text, output_probs, model["chars"],
                save_path=plot_path("next_char_prob_sequence_heatmap.png"),
                condensed=cv,
                exp_name=args.exp,
                automaton=automaton,
                spaced=spaced,
                words=words,
            )

            plot_per_char_hidden_state_heatmaps(
                text, hidden_states, model["chars"],
                save_path=plot_path("activation_by_input_char.png"),
                cluster_rows=not args.no_cluster_per_char,
                spaced=spaced,
                condensed=cv,
                automaton=automaton,
            )

        with timer.section("states_pca"):
            _plot_embed_variants(
                plot_pca_context_labels,
                plot_path("embedding_panels_context.png"),
                text=text, hidden_states=hidden_states, chars=model["chars"],
                spaced=spaced, automaton=automaton, condensed=cv, words=words,
                label_words=label_words if not spaced else None,
                annot_style=args.dfa_annot_style,
            )

            dfa_viz_text = text
            dfa_viz_states = hidden_states
            if automaton is not None and words:
                dfa_viz_text = build_vocabulary_coverage_text(words, spaced=spaced)
                dfa_viz_states, _ = run_forward_pass(model, dfa_viz_text, model_type)
                print(
                    f"DFA figures: teacher-forced over full vocabulary coverage "
                    f"({len(dfa_viz_text)} chars, {len(words)} words)",
                )
            _plot_embed_variants(
                plot_pca_context_labels_3d,
                plot_path("embedding_panels_context_3d.png"),
                text=dfa_viz_text, hidden_states=dfa_viz_states, chars=model["chars"],
                spaced=spaced, automaton=automaton, condensed=None,
                annot_style=args.dfa_annot_style, words=words,
                label_words=label_words if not spaced else None,
            )

            _plot_embed_variants(
                plot_pca_prediction_regions,
                plot_path("next_char_regions_pca.png"),
                model=model, text=text, hidden_states=hidden_states, chars=model["chars"],
                spaced=spaced, automaton=automaton, condensed=cv,
            )

            if automaton is not None and words:
                _plot_embed_variants(
                    plot_pca_dfa_analysis,
                    plot_path("dfa_and_embedding_pca.png"),
                    text=dfa_viz_text, hidden_states=dfa_viz_states, chars=model["chars"],
                    words=words, automaton=automaton, model=model, spaced=spaced,
                    annot_style=args.dfa_annot_style, condensed=cv,
                )

            _plot_embed_variants(
                plot_pca_next_char_probability_panels,
                plot_path("next_char_prob_panels_pca.png"),
                model=model, text=text, hidden_states=hidden_states, chars=model["chars"],
                spaced=spaced, automaton=automaton, condensed=cv,
            )

            _plot_embed_variants(
                plot_pca_vector_field,
                plot_path("vector_field_grid_pca_no_input.png"),
                text=text, hidden_states=hidden_states, model=model, condensed=cv,
            )

        if words:
            traj_words = label_words if label_words is not None else words
            with timer.section("trajectories"):
                _plot_embed_variants(
                    plot_space_to_space_trajectories,
                    plot_path("word_trajectories_pca.png"),
                    text=text, hidden_states=hidden_states, model=model,
                    spaced=spaced, automaton=automaton,
                    annot_style=args.dfa_annot_style, condensed=cv, words=traj_words,
                )
                _plot_embed_variants(
                    plot_space_to_space_trajectories_3d,
                    plot_path("word_trajectories_pca_3d.png"),
                    text=text, hidden_states=hidden_states, model=model,
                    spaced=spaced, automaton=automaton, condensed=cv, words=traj_words,
                )
                if args.closed_loop_video:
                    traj_dir = plot_path("word_trajectories_pca.png")
                    traj_dir = os.path.dirname(traj_dir)
                    write_closed_loop_trajectory_video(
                        model,
                        hidden_states,
                        save_path=os.path.join(traj_dir, "closed_loop_trajectory.gif"),
                        vocab_words=traj_words,
                        spaced=spaced,
                    )

        with timer.section("states_correlation"):
            plot_hidden_states_clustermap(
                text, hidden_states, model["chars"],
                save_path=plot_path("activation_clustered_heatmap.png"),
                exp_name=args.exp,
                condensed=cv,
                automaton=automaton,
                spaced=spaced,
            )

            plot_hidden_states_correlation_clustermap(
                text, hidden_states, model["chars"],
                save_path=plot_path("state_correlation_clustered_heatmap.png"),
                spaced=spaced,
                automaton=automaton,
                words=words,
                condensed=cv,
            )

            if automaton is not None:
                plot_dfa_grouped_state_correlation(
                    text,
                    hidden_states,
                    save_path=plot_path("state_correlation_by_dfa_state.png"),
                    spaced=spaced,
                    automaton=automaton,
                    condensed=cv,
                )
                plot_dfa_state_distance_comparison(
                    text, hidden_states, automaton,
                    save_path=plot_path("dfa_state_distance_comparison.png"),
                    spaced=spaced,
                    words=words,
                    condensed=cv,
                )
                plot_feature_separation_summary(
                    text, hidden_states, automaton,
                    save_path=plot_path("feature_separation_summary.png"),
                    spaced=spaced,
                    words=words,
                    label_words=label_words if not spaced else None,
                    condensed=cv,
                    output_probs=output_probs,
                )

        if automaton is not None:
            from unit_selectivity import plot_unit_selectivity_suite

            with timer.section("unit_selectivity"):
                plot_unit_selectivity_suite(
                    hidden_states,
                    text,
                    automaton,
                    os.path.join(out_dir, "unit_selectivity"),
                    model=model,
                    spaced=spaced,
                    words=words,
                    condensed=cv,
                    repr_label="RNN hidden state",
                    unit_labels=hidden_unit_labels(
                        model.get("dale_sign"), model["hidden_size"],
                    ),
                    label_words=label_words if not spaced else None,
                    output_probs=output_probs,
                )

        if model["hidden_size"] == 2:
            with timer.section("state_trajectories_2d"):
                plot_state_trajectory(
                    hidden_states,
                    color_by_chars=list(text) if cv is None else cv.input_chars,
                    chars=model["chars"],
                    title=f"Hidden state trajectory over {len(text)} chars (colored by INPUT char)",
                    save_path=plot_path("state_trajectory_by_input.png"),
                    condensed=cv,
                )
                plot_state_trajectory(
                    hidden_states,
                    color_by_chars=targets if cv is None else cv.next_chars,
                    chars=model["chars"],
                    title=f"Hidden state trajectory over {len(text)} chars (colored by TARGET / next char)",
                    save_path=plot_path("state_trajectory_by_target.png"),
                    condensed=cv,
                )

    correct = np.sum(np.argmax(output_probs, axis=1) ==
                     np.array([model["chars"].index(c) for c in targets]))
    print(f"top-1 next-char accuracy over the {len(text)}-char window: "
          f"{correct}/{len(text)} = {100*correct/len(text):.1f}%")

    if words and args.exp:
        with timer.section("vocab_diagrams"):
            shared = shared_dir(args.exp)
            shared.mkdir(parents=True, exist_ok=True)
            trie_path, dfa_path = write_vocabulary_diagrams(words, shared)
            print(f"wrote {trie_path}")
            print(f"wrote {dfa_path}")

    if (
        args.learning_dynamics_video
        and args.exp
        and automaton is not None
        and not args.condensed
        and not is_transformer
    ):
        dyn_dir = learning_dynamics_dir(args.exp, model_type)
        dyn_dir.mkdir(parents=True, exist_ok=True)
        try:
            with timer.section("learning_dynamics_video"):
                write_hidden_state_pca_learning_video(
                    model,
                    text,
                    save_path=str(dyn_dir / "hidden_state_pca.mp4"),
                    spaced=spaced,
                    automaton=automaton,
                )
        except Exception as exc:
            print(f"skip learning dynamics video: {exc}")

    if args.exp:
        with timer.section("cleanup"):
            remove_legacy_readme_plot_names(out_dir)
            remove_flat_plot_files(out_dir)
            remove_shared_figures_from_model_plots(out_dir, shared_dir(args.exp))
            # Drop pre-refactor learning_dynamics/ sibling of plots/.
            legacy_dyn = model_dir(args.exp, model_type) / "learning_dynamics"
            if legacy_dyn.is_dir() and legacy_dyn != Path(out_dir) / "learning_dynamics":
                import shutil
                shutil.rmtree(legacy_dyn)
                print(f"removed legacy {legacy_dyn}")

    timer.print_summary(title="Overall visualization timing")


if __name__ == "__main__":
    main()
