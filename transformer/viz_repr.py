"""Transformer-specific visualization: separate plots per representation type."""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from readme_figures import numbered_plot_path
from transformer.adapter import extract_transformer_activations
from vocab_diagrams import MinimizedVocabAutomaton


def _plot_path(out_dir: str, name: str, condensed: bool) -> str:
    path = str(numbered_plot_path(out_dir, name))
    if not condensed:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_condensed{ext}" if not base.endswith("_condensed") else path


def plot_embedding_heatmap(
    text: str,
    vectors: np.ndarray,
    save_path: str,
    *,
    repr_name: str,
    dim_label: str,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
) -> None:
    """Heatmap for a (T, n_embd) embedding or block vector."""
    from visualize import plot_hidden_states_heatmap

    plot_hidden_states_heatmap(
        text,
        vectors,
        save_path,
        act_label="raw",
        y_label=dim_label,
        title=f"{repr_name} at each timestep (current query position)",
        colorbar_label="value",
        automaton=automaton,
        spaced=spaced,
    )


def plot_layer_qkv_figure(
    text: str,
    layer: LayerActivations,
    layer_idx: int,
    save_path: str,
    *,
    num_heads: int,
    head_size: int,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
) -> None:
    """One figure with Q, K, V heatmaps per attention head."""
    from visualize import _color_tick_labels_by_state_ids, _dfa_state_ids_at_timesteps

    T = len(text)
    fig, axes = plt.subplots(3, num_heads, figsize=(max(12, num_heads * 3.2), 9), squeeze=False)
    qkv_rows = [
        ("Query", layer.queries),
        ("Key", layer.keys),
        ("Value", layer.values),
    ]
    for row_idx, (qkv_name, head_arrays) in enumerate(qkv_rows):
        for h in range(num_heads):
            ax = axes[row_idx, h]
            data = head_arrays[h].T
            im = ax.imshow(data, aspect="auto", cmap="RdBu_r", interpolation="nearest", origin="lower")
            ax.set_title(f"Layer {layer_idx} · head {h} · {qkv_name}")
            ax.set_yticks(range(head_size))
            ax.set_yticklabels([f"d{i}" for i in range(head_size)], fontsize=7)
            ax.set_xticks(range(T))
            ax.set_xticklabels(list(text), fontsize=6, rotation=90)
            if automaton is not None:
                state_ids = _dfa_state_ids_at_timesteps(text, automaton, spaced=spaced)
                _color_tick_labels_by_state_ids(ax.get_xticklabels(), state_ids)
            if row_idx == 2:
                ax.set_xlabel("timestep / input character")
            if h == 0:
                ax.set_ylabel(f"{qkv_name} dim")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"Per-head Q/K/V at the current query position (layer {layer_idx})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_layer_attention_figure(
    text: str,
    layer: LayerActivations,
    layer_idx: int,
    block_size: int,
    save_path: str,
    *,
    num_heads: int,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
) -> None:
    """Attention weights from the current query to prior keys (lag 0 = self)."""
    from visualize import _color_tick_labels_by_state_ids, _dfa_state_ids_at_timesteps

    T = len(text)
    ncols = min(num_heads, 4)
    nrows = math.ceil(num_heads / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(12, ncols * 3.5), nrows * 3.2), squeeze=False)
    lag_labels = [f"lag {i}" for i in range(block_size)]
    for h in range(num_heads):
        ax = axes[h // ncols, h % ncols]
        im = ax.imshow(
            layer.attention_lags[h].T,
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(f"Layer {layer_idx} · head {h}")
        ax.set_yticks(range(block_size))
        ax.set_yticklabels(lag_labels, fontsize=7)
        ax.set_xticks(range(T))
        ax.set_xticklabels(list(text), fontsize=6, rotation=90)
        if automaton is not None:
            state_ids = _dfa_state_ids_at_timesteps(text, automaton, spaced=spaced)
            _color_tick_labels_by_state_ids(ax.get_xticklabels(), state_ids)
        ax.set_xlabel("timestep / input character")
        ax.set_ylabel("attention to key lag")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="weight")
    for h in range(num_heads, nrows * ncols):
        axes[h // ncols, h % ncols].axis("off")
    fig.suptitle(
        f"Causal attention weights (layer {layer_idx}): query at timestep t attends to lag k",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def run_transformer_visualization(
    model: dict,
    text: str,
    out_dir: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    words: list[str] | None = None,
    condensed: bool = False,
) -> object:
    """Generate representation-specific plots (not RNN-style hidden-state analysis)."""
    from visualize import (
        condense_hidden_states_by_prefix,
        plot_hidden_states_clustermap,
        plot_hidden_states_correlation_clustermap,
        plot_output_probs,
        plot_pca_context_labels,
        plot_pca_dfa_analysis,
        plot_pca_next_char_probability_panels,
        plot_pca_prediction_regions,
    )

    acts = extract_transformer_activations(model, text)
    cv = None
    if condensed:
        cv = condense_hidden_states_by_prefix(
            text, acts.block_output, acts.output_probs, spaced=spaced, words=words,
        )

    plot_output_probs(
        text, acts.output_probs, model["chars"],
        save_path=_plot_path(out_dir, "next_char_prob_sequence_heatmap.png", condensed),
        condensed=cv,
        automaton=automaton,
        spaced=spaced,
        words=words,
    )

    embedding_specs = [
        ("token_embedding", acts.token_emb, "token_embedding_heatmap.png", "token emb dim"),
        ("position_embedding", acts.pos_emb, "position_embedding_heatmap.png", "position emb dim"),
        ("block_input (token + position)", acts.block_input, "block_input_heatmap.png", "block input dim"),
        ("block output (pre-lm_head)", acts.block_output, "block_output_heatmap.png", "block output dim"),
    ]
    for repr_name, vectors, fname, dim_label in embedding_specs:
        plot_embedding_heatmap(
            text, vectors, _plot_path(out_dir, fname, condensed),
            repr_name=repr_name, dim_label=dim_label,
            automaton=automaton, spaced=spaced,
        )
        pca_name = fname.replace("_heatmap.png", "_pca.png")
        plot_pca_context_labels(
            text, vectors, model["chars"],
            save_path=_plot_path(out_dir, pca_name, condensed),
            spaced=spaced, automaton=automaton, condensed=cv,
        )

    for layer_idx, layer in enumerate(acts.layers):
        plot_layer_qkv_figure(
            text, layer, layer_idx,
            _plot_path(out_dir, f"layer{layer_idx}_qkv.png", condensed),
            num_heads=acts.num_heads, head_size=acts.head_size,
            automaton=automaton, spaced=spaced,
        )
        plot_layer_attention_figure(
            text, layer, layer_idx, acts.block_size,
            _plot_path(out_dir, f"layer{layer_idx}_attention.png", condensed),
            num_heads=acts.num_heads,
            automaton=automaton, spaced=spaced,
        )
        plot_embedding_heatmap(
            text, layer.attn_input,
            _plot_path(out_dir, f"layer{layer_idx}_attn_input_heatmap.png", condensed),
            repr_name=f"layer {layer_idx} attention input (post-ln1)",
            dim_label="attn input dim",
            automaton=automaton, spaced=spaced,
        )

    if automaton is not None and words:
        plot_pca_dfa_analysis(
            text, acts.block_output, model["chars"], words,
            save_path=_plot_path(out_dir, "dfa_and_embedding_pca.png", condensed),
            automaton=automaton,
            model=model,
            spaced=spaced, condensed=cv,
        )

    plot_pca_prediction_regions(
        model, text, acts.block_output, model["chars"],
        save_path=_plot_path(out_dir, "next_char_regions_pca.png", condensed),
        spaced=spaced, automaton=automaton, condensed=cv,
    )
    plot_pca_next_char_probability_panels(
        model, text, acts.block_output, model["chars"],
        save_path=_plot_path(out_dir, "next_char_prob_panels_pca.png", condensed),
        spaced=spaced, automaton=automaton, condensed=cv,
    )

    plot_hidden_states_clustermap(
        text, acts.block_output, model["chars"],
        save_path=_plot_path(out_dir, "block_output_clustered_heatmap.png", condensed),
        condensed=cv, automaton=automaton, spaced=spaced,
    )
    plot_hidden_states_correlation_clustermap(
        text, acts.block_output, model["chars"],
        save_path=_plot_path(out_dir, "block_output_correlation_heatmap.png", condensed),
        spaced=spaced, automaton=automaton, words=words, condensed=cv,
    )

    return acts
