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

  14) learning_dynamics/hidden_state_pca.mp4 (or .gif)
       Hidden states on a fixed final-model PCA basis across weight snapshots.

Usage:
    python visualize.py --exp ten_word_overlap_s
    python visualize.py --exp ten_word_overlap --length 100
    python visualize.py --model path/to/model.npz --input path/to/input.txt --out-dir path/to/plots
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
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
    remove_legacy_readme_plot_names,
    remove_shared_figures_from_model_plots,
)
from task import REGIMES
from rnn.rnn_dyn import activation_label, no_input_hidden_step, rnn_hidden_step
from vocab_diagrams import (
    MinimizedVocabAutomaton,
    build_minimized_vocabulary_automaton,
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
    data = np.load(path, allow_pickle=False)
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
    if "loss_iterations" in data.files:
        model["loss_iterations"] = data["loss_iterations"]
        model["loss_smooth"] = data["loss_smooth"]
        model["loss_window"] = data["loss_window"]
    if "metric_iterations" in data.files:
        model["metric_iterations"] = data["metric_iterations"]
        model["metric_valid_vocab_letter_frac"] = data["metric_valid_vocab_letter_frac"]
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
    if "weight_snap_iterations" in data.files:
        model["weight_snap_iterations"] = data["weight_snap_iterations"]
        model["weight_snap_outgoing"] = data["weight_snap_outgoing"]
        model["weight_snap_violation_frac"] = data["weight_snap_violation_frac"]
    if "weight_snap_bias_hidden" in data.files:
        model["weight_snap_bias_hidden"] = data["weight_snap_bias_hidden"]
    if "weight_snap_bias_output" in data.files:
        model["weight_snap_bias_output"] = data["weight_snap_bias_output"]
    if "metric_word_error_frac" in data.files:
        model["metric_word_error_frac"] = data["metric_word_error_frac"]
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
    default_title = (
        f"Hidden state activations ({act_label} output) over the input sequence"
        if y_label == "hidden unit"
        else f"{y_label} over the input sequence"
    )
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


def _resolve_words(text: str, words: list[str] | None = None) -> list[str] | None:
    if words is not None:
        return words
    if _VIS_WORDS:
        return _VIS_WORDS
    return infer_task_words(text)


def _corpus_vocab(text: str, words: list[str] | None = None) -> set[str] | None:
    w = _resolve_words(text, words)
    return set(w) if w else None


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
    if not vocab:
        return float("nan")
    if spaced:
        tokens = [t for t in sampled_text.split(" ") if t]
    else:
        tokens = [seg[2] for seg in segment_corpus_by_words(sampled_text, vocab)]
    if not tokens:
        return float("nan")
    bad = sum(1 for t in tokens if t not in vocab)
    return bad / len(tokens)


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
    prefix_labels: list[str],
    string_labels: list[str],
) -> dict[str, np.ndarray]:
    """L2 distances (i < j) for within/between prefix, string, DFA, position, char, all pairs."""
    n = hidden_states.shape[0]
    groups: dict[str, list[float]] = {
        "Within prefix": [],
        "Between prefixes": [],
        "Within string": [],
        "Between strings": [],
        "Within DFA state": [],
        "Between DFA states": [],
        "Within word position": [],
        "Between word positions": [],
        "Within char": [],
        "Between chars": [],
        "All pairs": [],
    }
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(hidden_states[i] - hidden_states[j]))
            groups["All pairs"].append(dist)
            if prefix_labels[i] == prefix_labels[j]:
                groups["Within prefix"].append(dist)
            else:
                groups["Between prefixes"].append(dist)
            if string_labels[i] == string_labels[j]:
                groups["Within string"].append(dist)
            else:
                groups["Between strings"].append(dist)
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
    return {k: np.asarray(v) for k, v in groups.items()}


PAIR_DISTANCE_CATEGORY_ORDER = (
    "Within prefix",
    "Between prefixes",
    "Within string",
    "Between strings",
    "Within DFA state",
    "Between DFA states",
    "Within word position",
    "Between word positions",
    "Within char",
    "Between chars",
    "All pairs",
)

PAIR_DISTANCE_PALETTE = {
    "Within prefix": "#9467bd",
    "Between prefixes": "#c5b0d5",
    "Within string": "#8c564b",
    "Between strings": "#c49c94",
    "Within DFA state": "#4c72b0",
    "Between DFA states": "#dd8452",
    "Within word position": "#8172b3",
    "Between word positions": "#9372b3",
    "Within char": "#55a868",
    "Between chars": "#2ca02c",
    "All pairs": "#8c8c8c",
}


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
        string_labels = list(condensed.labels)
        prefix_labels = [
            prefix_before_from_string_label(l) for l in condensed.labels
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
        string_labels = [
            in_word_prefix_at_position(text, t, spaced=spaced, vocab=vocab)
            for t in range(len(text))
        ]
        prefix_labels = [
            in_word_prefix_before_current(text, t, spaced=spaced, vocab=vocab)
            for t in range(len(text))
        ]
    n = hidden_states.shape[0]
    if n < 2:
        return

    by_label = pairwise_hidden_state_distance_groups(
        compare_chars, hidden_states, state_ids, position_ids,
        prefix_labels, string_labels,
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


def fit_pca_2d(points):
    """PCA fit: return 2D coords, mean, and (2, D) principal axes for reconstruction."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    return coords, mean, components


def fit_pca_2d_with_evr(points):
    """PCA fit + explained variance ratio for PC1/PC2."""
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coords = centered @ components.T
    denom = float(np.sum(s * s)) if len(s) else 1.0
    evr = (s[:2] * s[:2]) / denom if denom > 0 else np.array([0.0, 0.0])
    return coords, mean, components, evr


def fit_pca_3d_with_evr(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    """Encode PNG frames to mp4 (ffmpeg) or gif (Pillow). Returns path written."""
    import shutil
    import subprocess

    if not frame_paths:
        raise ValueError("no frames to encode")

    mp4_path = out_path if out_path.endswith(".mp4") else f"{out_path}.mp4"
    if shutil.which("ffmpeg"):
        frame_dir = os.path.dirname(frame_paths[0])
        pattern = os.path.join(frame_dir, "frame_%04d.png")
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
        print(f"skip {save_path}: re-run min-char-rnn.py to record weight snapshots")
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
):
    """PCA mesh and 2D-reconstructed hidden states on a grid covering data + labels."""
    projected, mean, components = fit_pca_2d(hidden_states)
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
    return grid_x, grid_y, grid_hidden, projected, xlim, ylim


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


def plot_learning_curve(model, save_path, *, loss_only: bool = False):
    """Per-eval cross-entropy (raw); optional word-validity metric on twin axis."""
    if "loss_iterations" not in model:
        print(f"skip {save_path}: re-run training to record loss history")
        return

    iters = np.asarray(model["loss_iterations"], dtype=int)
    if "loss_window" in model:
        ce_plot = np.asarray(model["loss_window"], dtype=float)
        ce_label = "cross-entropy"
    elif "loss_smooth" in model:
        ce_plot = np.asarray(model["loss_smooth"], dtype=float)
        ce_label = "cross-entropy (smoothed; re-train for raw loss)"
    else:
        print(f"skip {save_path}: no loss history in model bundle")
        return

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ce_line, = ax.plot(iters, ce_plot, color="steelblue", linewidth=1.2, label=ce_label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy")
    title = "Training loss" if loss_only else "Training: cross-entropy vs word-validity rollout"
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_ylim(*_tight_ylim(ce_plot, floor=0.0))
    ax.legend(loc="upper right", fontsize=8)

    if loss_only:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"wrote {save_path}")
        return

    legend_lines = [ce_line]
    legend_labels = [ce_label]

    if "metric_iterations" in model and "metric_word_error_frac" in model:
        ax2 = ax.twinx()
        metric_pct = 100.0 * np.asarray(model["metric_word_error_frac"], dtype=float)
        metric_line, = ax2.plot(
            model["metric_iterations"],
            metric_pct,
            color="darkorange",
            linewidth=1.2,
            alpha=0.9,
            label="% invalid words (rollout)",
        )
        ax2.set_ylabel("% invalid words (mean stochastic rollout)")
        ax2.set_ylim(*_tight_ylim(metric_pct, floor=0.0, ceiling=100.0))
        legend_lines.append(metric_line)
        legend_labels.append(metric_line.get_label())
    elif "metric_iterations" in model and "metric_valid_vocab_letter_frac" in model:
        ax2 = ax.twinx()
        metric_pct = 100.0 * (1.0 - np.asarray(model["metric_valid_vocab_letter_frac"], dtype=float))
        metric_line, = ax2.plot(
            model["metric_iterations"],
            metric_pct,
            color="darkorange",
            linewidth=1.2,
            alpha=0.9,
            label="% letters out of vocab (rollout)",
        )
        ax2.set_ylabel("% letters out of vocab (rollout)")
        ax2.set_ylim(*_tight_ylim(metric_pct, floor=0.0, ceiling=100.0))
        legend_lines.append(metric_line)
        legend_labels.append(metric_line.get_label())

    if len(legend_lines) > 1:
        ax.legend(legend_lines, legend_labels, loc="upper right", fontsize=8)

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


def _draw_sample_chars(
    ax,
    text: str,
    y: float,
    *,
    vocab: set[str] | None = None,
    spaced: bool = False,
) -> None:
    snippet = text[:SAMPLE_DISPLAY_LEN]
    if not snippet:
        return
    n = len(snippet)
    x_step = min(0.019, 0.98 / max(n - 1, 1))
    if vocab is None:
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
    if after_err is None or not np.isfinite(after_err) or not spaced:
        after_err = _invalid_word_fraction(demo_after, vocab, spaced=spaced)
    after_title = f"Generated after learning — {100.0 * after_err:.1f}% invalid words"
    if "metric_word_error_frac" in model and len(model["metric_word_error_frac"]):
        train_err = float(model["metric_word_error_frac"][-1])
        after_title += f"; training metric: {100.0 * train_err:.1f}%"

    rows = [
        (f"Training corpus ({SAMPLE_DISPLAY_LEN} chars)", demo_snippet, None),
        (
            f"Generated before learning ({SAMPLE_DISPLAY_LEN} chars) "
            "— green=in vocab, red=not",
            demo_before,
            vocab,
        ),
        (
            after_title + f" ({SAMPLE_DISPLAY_LEN} chars) — green=in vocab, red=not",
            demo_after,
            vocab,
        ),
    ]

    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 3.6), constrained_layout=True)
    for ax, (title, snippet, word_vocab) in zip(axes, rows):
        ax.set_axis_off()
        ax.text(0.0, 0.92, title, transform=ax.transAxes, fontsize=10, va="top")
        _draw_sample_chars(
            ax, snippet, 0.35, vocab=word_vocab, spaced=spaced,
        )

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def trigram_sequence_colors(labels):
    """Stable color per unique 3-char context label (tab10, full saturation)."""
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    return {label: cmap(i) for i, label in enumerate(unique_labels)}


def _state_id_colors(state_ids: list[int]) -> dict[int, tuple]:
    unique = sorted(set(state_ids))
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    return {state: cmap(i) for i, state in enumerate(unique)}


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
    projected, groups: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    center = projected.mean(axis=0)
    span = max(
        float(np.ptp(projected[:, 0])),
        float(np.ptp(projected[:, 1])),
        1e-3,
    )
    label_offset = span * 0.14
    label_positions = {}

    def _rot(v: np.ndarray, deg: float) -> np.ndarray:
        t = np.deg2rad(deg)
        c, s = float(np.cos(t)), float(np.sin(t))
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    # Try a small set of angles so leader labels don't overlap each other.
    # (We approximate overlap using label center distance in data units.)
    angle_candidates_deg = [0, 18, -18, 36, -36, 54, -54, 72, -72, 90, -90, 120, -120, 150, -150, 180]
    min_sep = label_offset * 0.65

    # Place "harder" groups first (those nearer the center tend to collide more).
    items = list(groups.items())
    items.sort(key=lambda kv: float(np.linalg.norm(projected[kv[1]].mean(axis=0) - center)))

    for key, indices in items:
        points = projected[indices]
        centroid = points.mean(axis=0)
        outward = centroid - center
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            outward = np.array([0.0, 1.0])
        else:
            outward = outward / norm

        best = centroid + outward * label_offset
        for deg in angle_candidates_deg:
            cand = centroid + _rot(outward, deg) * label_offset
            if not label_positions:
                best = cand
                break
            if all(float(np.linalg.norm(cand - p)) >= min_sep for p in label_positions.values()):
                best = cand
                break
        label_positions[key] = best
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
    leader_linewidth: float = 1.4,
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
                color=color, linewidth=leader_linewidth, solid_capstyle="round", zorder=5,
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


def _add_dfa_state_color_legend(
    ax, automaton: MinimizedVocabAutomaton, state_colors: dict[int, tuple]
) -> None:
    handles = [
        Patch(
            facecolor=state_colors[state],
            edgecolor="#333333",
            label=dfa_state_label(state, automaton),
        )
        for state in sorted(state_colors)
    ]
    ax.legend(
        handles=handles,
        title="min DFA state",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.95,
        borderaxespad=0.0,
    )


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
    leader_linewidth: float = 1.4,
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
    label_positions = _layout_group_label_positions(projected, by_prefix)
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
    x_pad = max((max(all_x) - min(all_x)) * 0.1, 1e-3)
    y_pad = max((max(all_y) - min(all_y)) * 0.1, 1e-3)
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
) -> None:
    """2D PCA of vectors with prefix / DFA annotations."""
    _ = chars
    if condensed is not None:
        hidden_states = condensed.hidden_states
        prefix_labels = condensed.labels
        spaced = condensed.spaced
    else:
        prefix_labels = None
    n = hidden_states.shape[0]
    if n < 1:
        return

    pca_xy, _, _, evr = fit_pca_2d_with_evr(hidden_states)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0

    ctx = prefix_axis_label(spaced=spaced, text=text)
    if automaton is not None:
        scheme = f"min DFA state · {ctx}"
    else:
        scheme = f"prefix after space" if spaced else prefix_axis_label(spaced=spaced, text=text)

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    _plot_2d_hidden_state_labels_on_ax(
        ax,
        text,
        pca_xy,
        title=f"PCA (PC1 {pc1:.1f}%, PC2 {pc2:.1f}%)\n({scheme})",
        xlabel=f"PC1 ({pc1:.1f}%)",
        ylabel=f"PC2 ({pc2:.1f}%)",
        spaced=spaced,
        automaton=automaton,
        annot_style=annot_style,
        prefix_labels=prefix_labels,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)

    fig.suptitle(
        _condensed_plot_title(original_vocabulary_title(chars, text), condensed),
        fontsize=12,
        y=1.01,
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
) -> tuple[list[tuple], list[str], dict[int, tuple]]:
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
    state_colors = _state_id_colors(state_ids)
    point_colors = [state_colors[s] for s in state_ids]
    return point_colors, prefixes, state_colors


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
) -> None:
    """3D PCA scatter with prefix labels (separate figure from the 2D panel)."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        prefix_labels = condensed.labels
        spaced = condensed.spaced
    else:
        prefix_labels = None
    n = hidden_states.shape[0]
    if n < 2 or hidden_states.shape[1] < 2:
        return

    pca_xyz, _, _, evr = fit_pca_3d_with_evr(hidden_states)
    xlabel, ylabel, zlabel = _pca_axis_labels(evr)
    ctx = prefix_axis_label(spaced=spaced, text=text)
    if automaton is not None:
        scheme = f"min DFA state · {ctx}"
    else:
        scheme = f"prefix after space" if spaced else prefix_axis_label(spaced=spaced, text=text)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    if automaton is not None:
        point_colors, prefixes, state_colors = _dfa_point_colors_for_pca(
            text, n=n, spaced=spaced, automaton=automaton, prefix_labels=prefix_labels,
        )
        _plot_3d_pca_scatter_with_labels(
            ax, pca_xyz, prefixes,
            point_colors=point_colors,
            title=f"{repr_name} · 3D PCA\n({scheme})",
            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
        )
        _add_dfa_state_color_legend(ax, automaton, state_colors)
    else:
        if prefix_labels is not None:
            prefixes = prefix_labels
        else:
            prefixes = [context_label(text, i, spaced=spaced) for i in range(n)]
        _plot_3d_pca_scatter_with_labels(
            ax, pca_xyz, prefixes,
            title=f"{repr_name} · 3D PCA\n({scheme})",
            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
        )

    fig.suptitle(
        _condensed_plot_title(original_vocabulary_title(chars, text), condensed),
        fontsize=12,
        y=0.98,
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
):
    """4-panel dim-reduction comparison with shared annotations/colors."""
    plot_dimred_context_panels(
        text,
        hidden_states,
        chars,
        save_path,
        spaced=spaced,
        automaton=automaton,
        annot_style="leaders",
        condensed=condensed,
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
):
    """3D PCA scatter (companion to embedding_panels_context)."""
    plot_dimred_context_panels_3d(
        text,
        hidden_states,
        chars,
        save_path,
        spaced=spaced,
        automaton=automaton,
        condensed=condensed,
        repr_name=repr_name,
    )


def _word_trajectory_colors(segments: list[tuple[int, int, str]]) -> dict[str, tuple]:
    """Stable color per distinct word label across space-to-space segments."""
    words = sorted({segment_word_label(seg) for _, _, seg in segments})
    cmap = plt.get_cmap("tab20", max(len(words), 1))
    return {word: cmap(i) for i, word in enumerate(words)}


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


def plot_space_to_space_trajectories(
    text: str,
    hidden_states: np.ndarray,
    save_path: str,
    *,
    model=None,
    free_rollout_steps: int = 10,
    closed_loop_steps: int | None = None,
    closed_loop_seed: int = 0,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    annot_style: str = "leaders",
    condensed: CondensedView | None = None,
):
    """PCA plot of every hidden-state path from one space timestep to the next.

    If `model` is provided, draw the no-input recurrent vector field in PCA
    as a faint background quiver grid.
    """
    if condensed is not None:
        words = condensed.words or _resolve_words(text) or []
        word_paths = _vocabulary_prefix_paths(words, condensed)
        if len(word_paths) < 1:
            return
        hidden_states = condensed.hidden_states
        projected, mean, components, evr = fit_pca_2d_with_evr(hidden_states)
        pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
        pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
        cmap = plt.get_cmap("tab20", max(len(word_paths), 1))
        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        for i, (word, idxs) in enumerate(word_paths):
            path = projected[idxs]
            color = cmap(i)
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=0.7, label=word)
            ax.scatter(
                path[:, 0], path[:, 1], s=30, c=[color], edgecolors="black", linewidths=0.3, zorder=3,
            )
        add_pca_point_annotations(
            ax, text, projected, spaced=condensed.spaced, automaton=automaton,
            annot_style=annot_style, prefix_labels=condensed.labels,
        )
        xlim, ylim = _square_data_limits(projected)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
        ax.set_title(
            _condensed_plot_title(
                f"Vocabulary word paths through trie prefixes ({len(word_paths)} words)",
                condensed,
            )
        )
        ax.legend(title="word", loc="best", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {save_path}")
        return

    segments = corpus_segments(text, _resolve_words(text), spaced=spaced)
    if len(text) < 2 or not segments:
        return

    projected, mean, components, evr = fit_pca_2d_with_evr(hidden_states)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    xlabel = f"PC1 ({pc1:.1f}%)"
    ylabel = f"PC2 ({pc2:.1f}%)"
    word_colors = _word_trajectory_colors(segments)

    if closed_loop_steps is None:
        closed_loop_steps = len(text)

    ncols = 3 if model is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(22 if ncols == 2 else 30, 11), constrained_layout=True)
    axes = np.atleast_1d(axes)
    # Panel order: internal (no input), trained (observed), closed-loop (self-fed).
    ax_free = axes[0] if ncols >= 2 else None
    ax_paths = axes[1] if ncols >= 2 else axes[0]
    ax_gen = axes[2] if ncols >= 3 else None

    rollout_paths: list[np.ndarray] = []
    gen_z = projected

    # Panel: trained (observed) trajectories colored by true word segment.
    for start, end, segment_text in segments:
        path = projected[start : end + 1]
        if len(path) == 0:
            continue
        color = word_colors[segment_word_label(segment_text)]

        ax_paths.plot(
            path[:, 0], path[:, 1],
            color=color, linewidth=1.6, alpha=0.55, solid_capstyle="round", zorder=2,
        )

    # Panel 2: free dynamics rollouts from each observed hidden state (no input).
    if ax_free is not None and model is not None and free_rollout_steps > 0:
        W_hh = np.asarray(model["weights_hidden_to_hidden"])
        b_h = np.asarray(model["bias_hidden"]).ravel()

        # Map each timestep to its containing word label for coloring.
        word_at_t = [""] * len(text)
        for start, end, segment_text in segments:
            word = segment_word_label(segment_text)
            for t in range(start, end + 1):
                if 0 <= t < len(word_at_t):
                    word_at_t[t] = word

        use_relu = bool(model.get("use_relu", False))
        for t, h0 in enumerate(hidden_states):
            h = np.asarray(h0, dtype=float)
            zs = [projected[t]]  # start exactly at the observed point
            for _ in range(int(free_rollout_steps)):
                h = no_input_hidden_step(h, W_hh, b_h, use_relu=use_relu)
                z = (h - mean) @ components.T
                zs.append(z)
            zs = np.asarray(zs, dtype=float)
            if zs.shape[0] < 2:
                continue
            rollout_paths.append(zs)
            color = word_colors.get(word_at_t[t], "0.15")
            ax_free.plot(zs[:, 0], zs[:, 1], color=color, linewidth=1.0, alpha=0.22, zorder=3)

    # Panel 3: closed-loop generation (sampled; previous output fed back as input).
    if ax_gen is not None and model is not None and closed_loop_steps > 1:
        rng = np.random.default_rng(int(closed_loop_seed))
        chars = list(model["chars"])
        char_to_index = {c: i for i, c in enumerate(chars)}
        vocab_size = len(chars)

        W_xh = np.asarray(model["weights_input_to_hidden"])
        W_hh = np.asarray(model["weights_hidden_to_hidden"])
        W_ho = np.asarray(model["weights_hidden_to_output"])
        b_h = np.asarray(model["bias_hidden"]).ravel()
        b_o = np.asarray(model["bias_output"]).ravel()

        # Seed with the first character of the observed window (keeps vocab consistent).
        seed_char = text[0] if text else chars[0]
        if seed_char not in char_to_index:
            seed_char = chars[0]

        h = np.zeros((hidden_states.shape[1], 1), dtype=float)
        generated = [seed_char]
        gen_h = []
        use_relu = bool(model.get("use_relu", False))
        b_h_col = np.asarray(model["bias_hidden"])

        prev_char = seed_char
        for _ in range(int(closed_loop_steps)):
            x = np.zeros((vocab_size, 1), dtype=float)
            x[char_to_index[prev_char], 0] = 1.0
            h, _ = rnn_hidden_step(
                h, x, W_xh, W_hh, b_h_col, use_relu=use_relu,
            )
            gen_h.append(h.ravel().copy())

            logits = W_ho @ h.ravel() + b_o
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            next_ix = int(rng.choice(vocab_size, p=probs))
            next_char = chars[next_ix]
            generated.append(next_char)
            prev_char = next_char

        gen_h = np.asarray(gen_h, dtype=float)
        gen_z = (gen_h - mean) @ components.T

        # Break generated characters into word segments between spaces and color by word
        gen_text = "".join(generated[: len(gen_z)])
        gen_segments = corpus_segments(gen_text, _resolve_words(text), spaced=spaced)
        if not gen_segments:
            gen_segments = [(0, len(gen_text) - 1, gen_text)]

        for start, end, seg in gen_segments:
            if start < 0 or end < 0 or start >= len(gen_z):
                continue
            end = min(end, len(gen_z) - 1)
            path = gen_z[start : end + 1]
            if len(path) == 0:
                continue
            word = segment_word_label(seg)
            color = word_colors.get(word, (0.2, 0.2, 0.2, 1.0))

            ax_gen.plot(path[:, 0], path[:, 1], color=color, linewidth=1.3, alpha=0.40, zorder=2)
            if len(path) >= 2:
                ax_gen.scatter(
                    path[:, 0], path[:, 1],
                    s=18, c=[color] * len(path),
                    alpha=0.55, edgecolors="black", linewidths=0.25, zorder=3,
                )

    limit_arrays = [projected, gen_z]
    limit_arrays.extend(rollout_paths)
    xlim, ylim = _square_data_limits(*limit_arrays)

    if model is not None:
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

        for ax in (a for a in (ax_paths, ax_free, ax_gen) if a is not None):
            ax.quiver(
                grid_x,
                grid_y,
                U,
                V,
                angles="xy",
                scale_units="xy",
                scale=35.0,
                width=0.0022,
                headwidth=3.6,
                headlength=4.6,
                headaxislength=3.6,
                color="#000000",
                alpha=0.18,
                zorder=1,
            )

    # Observed test-window prefix labels at their trained PCA positions on every panel.
    for ax in (a for a in (ax_paths, ax_free, ax_gen) if a is not None):
        add_pca_point_annotations(
            ax,
            text,
            projected,
            spaced=spaced,
            automaton=automaton,
            annot_style=annot_style,
        )

    handles = [
        Patch(facecolor=word_colors[w], edgecolor="#333333", label=w)
        for w in sorted(word_colors)
    ]
    ax_paths.legend(
        handles=handles,
        title="word",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.95,
    )

    for ax in (a for a in (ax_paths, ax_free, ax_gen) if a is not None):
        ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")

    ax_paths.set_title(f"Trained (observed) trajectories (PCA)\n{len(segments)} segments, {len(text)} chars")
    if ax_free is not None:
        ax_free.set_title(
            f"Internal dynamics (no input)\n"
            f"{len(text)} start states × {free_rollout_steps} steps"
        )
    if ax_gen is not None:
        ax_gen.set_title(
            f"Closed-loop generation (sampled; self-fed)\n"
            f"{closed_loop_steps} steps (seed={closed_loop_seed})"
        )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


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
) -> None:
    """Grid vector field in PCA: z -> z' from no-input recurrent dynamics.

    We reconstruct h from each (PC1,PC2) grid point, apply one recurrent step with x=0,
    then project back to PCA to get the vector z' - z.
    """
    if condensed is not None:
        hidden_states = condensed.hidden_states
    if hidden_states.shape[0] < 3:
        return

    projected, mean, components, evr = fit_pca_2d_with_evr(hidden_states)
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

    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title(
        _condensed_plot_title(
            "Vector field in PCA (grid; no-input recurrent dynamics)",
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
):
    """PCA beside the min-DFA with matching state colors."""
    if condensed is not None:
        hidden_states = condensed.hidden_states
        prefix_labels = condensed.labels
        spaced = condensed.spaced
    else:
        prefix_labels = None
    if hidden_states.shape[0] < 2:
        return

    projected, _, _, evr = fit_pca_2d_with_evr(hidden_states)
    xlabel = f"PC1 ({100.0 * float(evr[0]):.1f}%)" if len(evr) > 0 else "PC1"
    ylabel = f"PC2 ({100.0 * float(evr[1]):.1f}%)" if len(evr) > 1 else "PC2"
    embed_title = "2D PCA"
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
    state_colors = _state_id_colors(state_ids)

    fig, axes = plt.subplots(1, 2, figsize=(28, 11), constrained_layout=True)
    ax_dfa, ax_embed = axes[0], axes[1]

    draw_minimized_dfa_on_axes(ax_dfa, automaton, words, state_colors=state_colors)
    ax_dfa.set_title("Minimal DFA", fontsize=12, pad=12)

    text_positions = add_dfa_state_annotations(
        ax_embed, text, projected, automaton,
        spaced=spaced, state_colors=state_colors,
        point_size=160,
        label_fontsize=18,
        leader_linewidth=2.8,
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
):
    """PCA panels: argmax next-char regions and softmax entropy, with context labels."""
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

    grid_x, grid_y, grid_hidden, projected, xlim, ylim = build_pca_plane_grid(
        text, hidden_states, grid_resolution, spaced=spaced, prefix_labels=prefix_labels,
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
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.35)
        if ax is axes[1]:
            fig.colorbar(im, ax=ax, label="entropy (nats)", fraction=0.046, pad=0.02)

    if automaton is not None:
        pca_ctx = (
            "min DFA state (prefix since last space)" if spaced
            else "min DFA state (in-word prefix)"
        )
    else:
        pca_ctx = "prefix after space" if spaced else prefix_axis_label(spaced=spaced, text=text)
    fig.suptitle(
        _condensed_plot_title(
            f"PCA of {representation_label(model)} · {pca_ctx} · "
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
):
    """One panel per vocab char: P(next = char) over the PCA plane (from softmax)."""
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

    grid_x, grid_y, grid_hidden, projected, xlim, ylim = build_pca_plane_grid(
        text, hidden_states, grid_resolution, spaced=spaced, prefix_labels=prefix_labels,
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

    axes[0].set_ylabel("PC2")
    axes[(nrows - 1) * ncols].set_xlabel("PC1")
    if nrows > 1:
        for row in range(1, nrows):
            axes[row * ncols].set_ylabel("PC2")
        for col in range(1, ncols):
            bottom = min((nrows - 1) * ncols + col, vocab_size - 1)
            if bottom < vocab_size:
                axes[bottom].set_xlabel("PC1")

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
        print(f"skip {save_path}: re-run min-char-rnn.py to record weight snapshots")
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
    model_type = getattr(args, "model_type", "rnn")
    if args.exp:
        ensure_experiment_dirs(args.exp, model_type)
        return (
            str(model_path(args.exp, model_type)),
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
    parser.add_argument("--model-type", default="rnn", choices=["rnn", "transformer"],
                        help="which model subdirectory to visualize (default: rnn)")
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

    if not is_transformer:
        with timer.section("weight_plots"):
            plot_learned_weights(model, save_path=str(numbered_plot_path(out_dir, "weights.png")))
            plot_weight_eigenspectra(
                model, save_path=str(numbered_plot_path(out_dir, "weights_eigenspectra.png")),
            )
            plot_weight_dynamics_over_training(
                model, os.path.join(out_dir, "weight_dynamics_over_training.png"),
            )
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

    with open(input_file, "r") as f:
        text = f.read()[: args.length]
    print(f"running forward pass over {len(text)} characters of {input_file}")

    spaced = corpus_uses_word_spacing(text, args.exp)
    words = vocabulary_for_experiment(args.exp) if args.exp else infer_task_words(text)
    global _VIS_WORDS
    _VIS_WORDS = words
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
            )
        output_probs = acts.output_probs
    else:
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

        plot_hidden_states_heatmap(
            text, hidden_states,
            save_path=plot_path("activation_heatmap.png"),
            act_label=act_label,
            condensed=cv,
            exp_name=args.exp,
            automaton=automaton,
            spaced=spaced,
            words=words,
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

        plot_pca_context_labels(
            text, hidden_states, model["chars"],
            save_path=plot_path("embedding_panels_context.png"),
            spaced=spaced,
            automaton=automaton,
            condensed=cv,
        )

        plot_pca_prediction_regions(
            model, text, hidden_states, model["chars"],
            save_path=plot_path("next_char_regions_pca.png"),
            spaced=spaced,
            automaton=automaton,
            condensed=cv,
        )

        if automaton is not None and words:
            plot_pca_dfa_analysis(
                text, hidden_states, model["chars"], words,
                save_path=plot_path("dfa_and_embedding_pca.png"),
                automaton=automaton,
                model=model,
                spaced=spaced,
                annot_style=args.dfa_annot_style,
                condensed=cv,
            )

        if words:
            plot_space_to_space_trajectories(
                text, hidden_states,
                save_path=plot_path("word_trajectories_pca.png"),
                model=model,
                spaced=spaced,
                automaton=automaton,
                annot_style=args.dfa_annot_style,
                condensed=cv,
            )

        plot_pca_next_char_probability_panels(
            model, text, hidden_states, model["chars"],
            save_path=plot_path("next_char_prob_panels_pca.png"),
            spaced=spaced,
            automaton=automaton,
            condensed=cv,
        )

        plot_pca_vector_field(
            text,
            hidden_states,
            model,
            plot_path("vector_field_grid_pca_no_input.png"),
            condensed=cv,
        )

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

        if model["hidden_size"] == 2:
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
        shared = shared_dir(args.exp)
        shared.mkdir(parents=True, exist_ok=True)
        trie_path, dfa_path = write_vocabulary_diagrams(words, shared)
        print(f"wrote {trie_path}")
        print(f"wrote {dfa_path}")

    if args.exp and automaton is not None and not args.condensed and not is_transformer:
        dyn_dir = learning_dynamics_dir(args.exp, model_type)
        dyn_dir.mkdir(parents=True, exist_ok=True)
        write_hidden_state_pca_learning_video(
            model,
            text,
            save_path=str(dyn_dir / "hidden_state_pca.mp4"),
            spaced=spaced,
            automaton=automaton,
        )

    if args.exp:
        remove_legacy_readme_plot_names(out_dir)
        remove_shared_figures_from_model_plots(out_dir, shared_dir(args.exp))

    timer.print_summary(title="Overall visualization timing")


if __name__ == "__main__":
    main()
