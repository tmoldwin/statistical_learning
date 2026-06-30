"""Transformer-specific visualization: one analysis suite per representation type.

Transformers do not have a single recurrent hidden state. Each component —
token embedding, position embedding, their sum (block input), per-layer
attention I/O, Q/K/V, and the final output vector — is analyzed separately.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from readme_figures import PLOT_BASENAME_TO_FIGURE, numbered_plot_path
from transformer.adapter import LayerActivations, TransformerActivations, extract_attention_matrix, extract_transformer_activations
from vocab_diagrams import MinimizedVocabAutomaton
from viz_timing import VizTimer


# README figures that only apply to the RNN hidden-state readout. When we
# previously ran the RNN visualization path on transformers, these landed at
# the plots root with misleading names — remove them on each transformer run.
_STALE_ROOT_PLOT_NAMES: tuple[str, ...] = tuple(
    fig.filename() for fig in PLOT_BASENAME_TO_FIGURE.values()
    if fig.number in {7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19}
) + (
    "block_input_heatmap.png",
    "block_input_pca.png",
    "block_output_heatmap.png",
    "block_output_pca.png",
    "block_output_clustered_heatmap.png",
    "block_output_correlation_heatmap.png",
    "position_embedding_heatmap.png",
    "position_embedding_pca.png",
    "token_embedding_heatmap.png",
    "token_embedding_pca.png",
    "layer0_attention.png",
    "layer0_attn_input_heatmap.png",
    "layer0_qkv.png",
    "layer1_attention.png",
    "layer1_attn_input_heatmap.png",
    "layer1_qkv.png",
)


@dataclass(frozen=True)
class RepresentationSpec:
    """One representation to analyze.

    Most fields are (T, D) aligned to corpus timesteps. Lookup tables (token /
    position embeddings) use ``lookup`` so PCA plots one point per vocab symbol
    or window position, not one point per timestep.
    """

    slug: str
    display_name: str
    dim_label: str
    vectors: np.ndarray
    is_readout: bool = False
    lookup: str | None = None  # None | "token" | "position"


def _condensed_path(path: str, condensed: bool) -> str:
    if not condensed:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_condensed{ext}" if not base.endswith("_condensed") else path


def _repr_dir(out_dir: str | Path, slug: str) -> Path:
    return Path(out_dir) / "representations" / slug


def _repr_plot_path(
    out_dir: str | Path,
    slug: str,
    plot_basename: str,
    *,
    condensed: bool = False,
) -> str:
    path = numbered_plot_path(_repr_dir(out_dir, slug), plot_basename)
    return _condensed_path(str(path), condensed)


def _attention_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "attention"


def cleanup_representation_plot_dir(slug_dir: Path) -> None:
    """Remove stale PNGs before rewriting a representation folder."""
    if not slug_dir.is_dir():
        return
    for stale in slug_dir.glob("*.png"):
        stale.unlink()


def cleanup_stale_representation_dirs(out_dir: str | Path, valid_slugs: set[str]) -> None:
    """Remove representation folders that are no longer in the spec list."""
    repr_root = Path(out_dir) / "representations"
    if not repr_root.is_dir():
        return
    for child in repr_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in valid_slugs:
            continue
        shutil.rmtree(child)
        print(f"removed stale representation dir {child}")


def cleanup_stale_attention_plots(out_dir: str | Path, n_layers: int) -> None:
    """Drop old attention figure names (multi-head grids, etc.)."""
    attn_dir = Path(out_dir) / "attention"
    if not attn_dir.is_dir():
        return
    valid = {
        *(f"layer{i}_qkv.png" for i in range(n_layers)),
        *(f"layer{i}_attention.png" for i in range(n_layers)),
        *(f"layer{i}_attention_lags.png" for i in range(n_layers)),
    }
    for stale in attn_dir.iterdir():
        if stale.is_file() and stale.name not in valid:
            stale.unlink()
            print(f"removed stale attention plot {stale}")


def cleanup_stale_transformer_plots(out_dir: str | Path) -> None:
    """Drop root-level plots from the old 'treat block output as h' approach."""
    root = Path(out_dir)
    for name in _STALE_ROOT_PLOT_NAMES:
        stale = root / name
        if stale.is_file():
            stale.unlink()
            print(f"removed stale plot {stale}")


def plot_lookup_table_heatmap(
    row_labels: list[str],
    vectors: np.ndarray,
    save_path: str,
    *,
    title: str,
    dim_label: str,
) -> None:
    """Heatmap of a lookup table (vocab × dim or positions × dim)."""
    n_rows, n_dims = vectors.shape
    fig, ax = plt.subplots(figsize=(max(8, n_dims * 0.35), max(2.5, n_rows * 0.55)))
    im = ax.imshow(
        vectors,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
        origin="lower",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f"{dim_label}{i}" for i in range(n_dims)], fontsize=7, rotation=90)
    ax.set_ylabel("lookup index")
    ax.set_xlabel(dim_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="value")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_lookup_pca(
    labels: list[str],
    vectors: np.ndarray,
    save_path: str,
    *,
    title: str,
) -> None:
    """PCA scatter with one labeled point per lookup row (not per timestep)."""
    from visualize import fit_pca_2d_with_evr

    if vectors.shape[0] < 1:
        return
    projected, _, _, evr = fit_pca_2d_with_evr(vectors)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    cmap = plt.get_cmap("tab10", max(len(labels), 1))
    for i, lab in enumerate(labels):
        disp = "␣" if lab == " " else lab
        ax.scatter(
            projected[i, 0], projected[i, 1],
            s=120, c=[cmap(i)], edgecolors="black", linewidths=0.4, zorder=3,
        )
        ax.annotate(
            repr(disp), (projected[i, 0], projected[i, 1]),
            xytext=(6, 6), textcoords="offset points", fontsize=11,
        )
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title(title)
    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_lookup_pca_3d(
    labels: list[str],
    vectors: np.ndarray,
    save_path: str,
    *,
    title: str,
) -> None:
    """3D PCA scatter with one labeled point per lookup row."""
    from visualize import fit_pca_3d_with_evr, _pca_axis_labels, _plot_3d_pca_scatter_with_labels

    if vectors.shape[0] < 2 or vectors.shape[1] < 2:
        return
    projected, _, _, evr = fit_pca_3d_with_evr(vectors)
    xlabel, ylabel, zlabel = _pca_axis_labels(evr)
    display_labels = ["␣" if lab == " " else lab for lab in labels]
    cmap = plt.get_cmap("tab10", max(len(labels), 1))
    point_colors = [cmap(i) for i in range(len(labels))]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    _plot_3d_pca_scatter_with_labels(
        ax, projected, display_labels,
        point_colors=point_colors,
        title=f"{title} (3D)",
        xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _lookup_table_from_model(model: dict, lookup: str) -> tuple[list[str], np.ndarray]:
    """Rows of a learned embedding table (not timestep samples)."""
    torch_model = model["_torch_model"]
    if lookup == "token":
        table = torch_model.token_embedding.weight.detach().cpu().numpy()
        labels = list(model["chars"])
        return labels, table
    if lookup == "position":
        table = torch_model.position_embedding_table.weight.detach().cpu().numpy()
        labels = [str(i) for i in range(table.shape[0])]
        return labels, table
    raise ValueError(f"unknown lookup type: {lookup}")


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
    """Heatmap for a (T, D) embedding or block vector at the query position."""
    from visualize import plot_hidden_states_heatmap

    plot_hidden_states_heatmap(
        text,
        vectors,
        save_path,
        act_label="raw",
        y_label=dim_label,
        title=f"{repr_name} at each timestep (query position)",
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
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
) -> None:
    """Q, K, V heatmaps for the single attention head at the query position."""
    from visualize import _color_tick_labels_by_state_ids, _dfa_state_ids_at_timesteps

    T = len(text)
    q, k, v = layer.queries[0], layer.keys[0], layer.values[0]
    head_size = q.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(max(10, head_size * 0.35), max(8, T * 0.18)), squeeze=False)
    for ax, (qkv_name, data) in zip(
        axes[:, 0],
        [("Query", q), ("Key", k), ("Value", v)],
    ):
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", interpolation="nearest", origin="upper")
        ax.set_title(f"Layer {layer_idx} · {qkv_name}")
        ax.set_yticks(range(T))
        tick_labels = ["␣" if c == " " else c for c in text]
        ax.set_yticklabels(tick_labels, fontsize=5)
        ax.set_xticks(range(head_size))
        ax.set_xticklabels([f"d{i}" for i in range(head_size)], fontsize=7, rotation=90)
        if automaton is not None:
            state_ids = _dfa_state_ids_at_timesteps(text, automaton, spaced=spaced)
            _color_tick_labels_by_state_ids(ax.get_yticklabels(), state_ids)
        ax.set_ylabel("timestep / input character")
        ax.set_xlabel(f"{qkv_name} dim")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"Q / K / V at the query position (layer {layer_idx})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_layer_attention_lags_figure(
    text: str,
    layer: LayerActivations,
    layer_idx: int,
    block_size: int,
    save_path: str,
    *,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
) -> None:
    """Causal attention weights: query at t attends to keys at lags 0..block_size-1."""
    from visualize import _color_tick_labels_by_state_ids, _dfa_state_ids_at_timesteps

    T = len(text)
    attn = layer.attention_lags[0]
    lag_labels = [f"lag {i}" for i in range(block_size)]
    fig, ax = plt.subplots(figsize=(max(12, T * 0.15), 5))
    im = ax.imshow(
        attn.T,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        origin="lower",
    )
    ax.set_title(f"Layer {layer_idx} causal attention by lag")
    ax.set_yticks(range(block_size))
    ax.set_yticklabels(lag_labels, fontsize=7)
    ax.set_xticks(range(T))
    tick_labels = ["␣" if c == " " else c for c in text]
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
    if automaton is not None:
        state_ids = _dfa_state_ids_at_timesteps(text, automaton, spaced=spaced)
        _color_tick_labels_by_state_ids(ax.get_xticklabels(), state_ids)
    ax.set_xlabel("timestep / input character")
    ax.set_ylabel("attention to key lag")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="weight")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_layer_attention_figure(
    model: dict,
    text: str,
    layer_idx: int,
    save_path: str,
    *,
    automaton: MinimizedVocabAutomaton | None = None,
    spaced: bool = False,
) -> None:
    """Standard causal attention heatmap from a single forward pass."""
    from visualize import _color_tick_labels_by_state_ids, _dfa_state_ids_at_timesteps

    attn, plot_text = extract_attention_matrix(model, text, layer_idx=layer_idx, head_idx=0)
    T = len(plot_text)
    chars = list(plot_text)
    truncated = len(text) > T

    future_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    attn_plot = np.ma.array(attn, mask=future_mask)

    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="white")

    size = max(8.0, T * 0.32)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    im = ax.imshow(
        attn_plot,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )
    tick_pos = np.arange(T)
    tick_labels = ["␣" if c == " " else c for c in chars]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=90, ha="center")
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=7)
    if automaton is not None:
        state_ids = _dfa_state_ids_at_timesteps(plot_text, automaton, spaced=spaced)
        _color_tick_labels_by_state_ids(ax.get_xticklabels(), state_ids)
        _color_tick_labels_by_state_ids(ax.get_yticklabels(), state_ids)
    title = f"Layer {layer_idx} attention (causal softmax)"
    if truncated:
        title += f"\nfirst {T} characters (block_size)"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="attention weight")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def collect_representation_specs(acts: TransformerActivations) -> list[RepresentationSpec]:
    """All (T, D) representations to analyze, in forward-pass order."""
    specs = [
        RepresentationSpec(
            "token_embedding",
            "token embedding E[token] (lookup table)",
            "token emb",
            acts.token_emb,
            lookup="token",
        ),
        RepresentationSpec(
            "position_embedding",
            "position embedding E[pos] (lookup table)",
            "pos emb",
            acts.pos_emb,
            lookup="position",
        ),
        RepresentationSpec(
            "block_input",
            "token + position embedding (pre-attention input)",
            "tok+pos",
            acts.block_input,
        ),
    ]
    for layer_idx, layer in enumerate(acts.layers):
        specs.extend([
            RepresentationSpec(
                f"layer{layer_idx}_attn_input",
                f"layer {layer_idx} attention input (post-ln1)",
                "attn in",
                layer.attn_input,
            ),
            RepresentationSpec(
                f"layer{layer_idx}_post_attn",
                f"layer {layer_idx} post-attention residual",
                "post attn",
                layer.post_attn,
            ),
            RepresentationSpec(
                f"layer{layer_idx}_post_ffwd",
                f"layer {layer_idx} post-FFN output",
                "post ffn",
                layer.post_ffwd,
            ),
            RepresentationSpec(
                f"layer{layer_idx}_query",
                f"layer {layer_idx} query",
                "q",
                layer.queries[0],
            ),
            RepresentationSpec(
                f"layer{layer_idx}_key",
                f"layer {layer_idx} key",
                "k",
                layer.keys[0],
            ),
            RepresentationSpec(
                f"layer{layer_idx}_value",
                f"layer {layer_idx} value",
                "v",
                layer.values[0],
            ),
        ])
    specs.append(RepresentationSpec(
        "block_output",
        "transformer output (pre-lm_head, query position)",
        "out",
        acts.block_output,
        is_readout=True,
    ))
    return specs


def _condense_representation(
    text: str,
    vectors: np.ndarray,
    output_probs: np.ndarray,
    *,
    spaced: bool,
    words: list[str] | None,
    condensed: bool,
):
    from visualize import condense_hidden_states_by_prefix

    if not condensed:
        return None
    return condense_hidden_states_by_prefix(
        text, vectors, output_probs, spaced=spaced, words=words,
    )


def plot_representation_suite(
    spec: RepresentationSpec,
    *,
    model: dict,
    text: str,
    out_dir: str,
    output_probs: np.ndarray,
    spaced: bool,
    automaton: MinimizedVocabAutomaton | None,
    words: list[str] | None,
    condensed: bool,
    timer: VizTimer | None = None,
    quick: bool = False,
    trajectories_only: bool = False,
) -> None:
    """RNN-parallel plot set for one transformer representation."""
    from visualize import (
        plot_dfa_grouped_state_correlation,
        plot_dfa_state_distance_comparison,
        plot_feature_separation_summary,
        plot_hidden_states_clustermap,
        plot_hidden_states_correlation_clustermap,
        plot_pca_context_labels,
        plot_pca_context_labels_3d,
        plot_pca_dfa_analysis,
        plot_pca_next_char_probability_panels,
        plot_pca_prediction_regions,
        plot_per_char_hidden_state_heatmaps,
        plot_space_to_space_trajectories,
        plot_space_to_space_trajectories_3d,
    )

    chars = model["chars"]
    name = spec.display_name
    dim = spec.dim_label

    def _plot(plot_name: str, fn, *args, **kwargs):
        if timer is None:
            return fn(*args, **kwargs)
        with timer.section(plot_name):
            return fn(*args, **kwargs)

    if spec.lookup is not None:
        row_labels, table = _lookup_table_from_model(model, spec.lookup)
        _plot(
            "lookup_table_heatmap",
            plot_lookup_table_heatmap,
            row_labels, table,
            _repr_plot_path(out_dir, spec.slug, "activation_heatmap.png", condensed=condensed),
            title=f"{name} — learned lookup table",
            dim_label=dim,
        )
        _plot(
            "lookup_pca",
            plot_lookup_pca,
            row_labels, table,
            _repr_plot_path(out_dir, spec.slug, "embedding_panels_context.png", condensed=condensed),
            title=f"{name} — one point per lookup row (PCA)",
        )
        _plot(
            "lookup_pca_3d",
            plot_lookup_pca_3d,
            row_labels, table,
            _repr_plot_path(out_dir, spec.slug, "embedding_panels_context_3d.png", condensed=condensed),
            title=f"{name} — one point per lookup row (PCA)",
        )
        if automaton is not None:
            cv = _condense_representation(
                text, spec.vectors, output_probs,
                spaced=spaced, words=words, condensed=condensed,
            )
            _plot(
                "dfa_state_distance_comparison",
                plot_dfa_state_distance_comparison,
                text, spec.vectors, automaton,
                save_path=_repr_plot_path(
                    out_dir, spec.slug, "dfa_state_distance_comparison.png", condensed=condensed,
                ),
                spaced=spaced,
                words=words,
                condensed=cv,
                repr_label=name,
            )
            _plot(
                "feature_separation_summary",
                plot_feature_separation_summary,
                text, spec.vectors, automaton,
                save_path=_repr_plot_path(
                    out_dir, spec.slug, "feature_separation_summary.png", condensed=condensed,
                ),
                spaced=spaced,
                words=words,
                condensed=cv,
                output_probs=output_probs,
                repr_label=name,
            )
        return

    cv = _condense_representation(
        text, spec.vectors, output_probs,
        spaced=spaced, words=words, condensed=condensed,
    )

    if trajectories_only:
        if not spec.is_readout or not words:
            return
        _plot(
            "word_trajectories_pca",
            plot_space_to_space_trajectories,
            text, spec.vectors,
            save_path=_repr_plot_path(out_dir, spec.slug, "word_trajectories_pca.png", condensed=condensed),
            model=None,
            spaced=spaced,
            automaton=automaton,
            condensed=cv,
            words=words,
        )
        _plot(
            "word_trajectories_pca_3d",
            plot_space_to_space_trajectories_3d,
            text, spec.vectors,
            save_path=_repr_plot_path(out_dir, spec.slug, "word_trajectories_pca_3d.png", condensed=condensed),
            model=None,
            spaced=spaced,
            automaton=automaton,
            condensed=cv,
            words=words,
        )
        return

    _plot(
        "activation_heatmap",
        plot_embedding_heatmap,
        text, spec.vectors,
        _repr_plot_path(out_dir, spec.slug, "activation_heatmap.png", condensed=condensed),
        repr_name=name, dim_label=dim,
        automaton=automaton, spaced=spaced,
    )
    _plot(
        "embedding_panels_context",
        plot_pca_context_labels,
        text, spec.vectors, chars,
        save_path=_repr_plot_path(out_dir, spec.slug, "embedding_panels_context.png", condensed=condensed),
        spaced=spaced, automaton=automaton, condensed=cv, words=words,
    )
    _plot(
        "embedding_panels_context_3d",
        plot_pca_context_labels_3d,
        text, spec.vectors, chars,
        save_path=_repr_plot_path(out_dir, spec.slug, "embedding_panels_context_3d.png", condensed=condensed),
        spaced=spaced, automaton=automaton, condensed=cv,
        repr_name=name, words=words,
    )
    if quick:
        return

    _plot(
        "activation_by_input_char",
        plot_per_char_hidden_state_heatmaps,
        text, spec.vectors, chars,
        save_path=_repr_plot_path(out_dir, spec.slug, "activation_by_input_char.png", condensed=condensed),
        spaced=spaced, condensed=cv, automaton=automaton,
        repr_label=name, dim_label=dim,
    )
    _plot(
        "activation_clustered_heatmap",
        plot_hidden_states_clustermap,
        text, spec.vectors, chars,
        save_path=_repr_plot_path(out_dir, spec.slug, "activation_clustered_heatmap.png", condensed=condensed),
        condensed=cv, automaton=automaton, spaced=spaced,
        repr_label=name, dim_label=dim,
    )
    _plot(
        "state_correlation_clustered",
        plot_hidden_states_correlation_clustermap,
        text, spec.vectors, chars,
        save_path=_repr_plot_path(out_dir, spec.slug, "state_correlation_clustered_heatmap.png", condensed=condensed),
        spaced=spaced, automaton=automaton, words=words, condensed=cv,
        repr_label=name,
    )
    if automaton is not None:
        _plot(
            "state_correlation_by_dfa_state",
            plot_dfa_grouped_state_correlation,
            text, spec.vectors,
            save_path=_repr_plot_path(
                out_dir, spec.slug, "state_correlation_by_dfa_state.png", condensed=condensed,
            ),
            spaced=spaced, automaton=automaton, condensed=cv,
            repr_label=name,
        )
        _plot(
            "dfa_state_distance_comparison",
            plot_dfa_state_distance_comparison,
            text, spec.vectors, automaton,
            save_path=_repr_plot_path(
                out_dir, spec.slug, "dfa_state_distance_comparison.png", condensed=condensed,
            ),
            spaced=spaced,
            words=words,
            condensed=cv,
            repr_label=name,
        )
        _plot(
            "feature_separation_summary",
            plot_feature_separation_summary,
            text, spec.vectors, automaton,
            save_path=_repr_plot_path(
                out_dir, spec.slug, "feature_separation_summary.png", condensed=condensed,
            ),
            spaced=spaced,
            words=words,
            condensed=cv,
            output_probs=output_probs,
            repr_label=name,
        )
        from unit_selectivity import plot_unit_selectivity_suite

        unit_dir = _repr_dir(out_dir, spec.slug) / "unit_selectivity"
        _plot(
            "unit_selectivity",
            plot_unit_selectivity_suite,
            spec.vectors,
            text,
            automaton,
            unit_dir,
            model=model,
            spaced=spaced,
            words=words,
            condensed=cv,
            repr_label=name,
            output_probs=output_probs,
            unit_labels=[f"{dim}{i}" for i in range(spec.vectors.shape[1])],
        )

    if not spec.is_readout:
        return

    if automaton is not None and words:
        _plot(
            "dfa_and_embedding_pca",
            plot_pca_dfa_analysis,
            text, spec.vectors, chars, words,
            save_path=_repr_plot_path(out_dir, spec.slug, "dfa_and_embedding_pca.png", condensed=condensed),
            automaton=automaton,
            model=model,
            spaced=spaced, condensed=cv,
            repr_name=name,
        )
    _plot(
        "next_char_regions_pca",
        plot_pca_prediction_regions,
        model, text, spec.vectors, chars,
        save_path=_repr_plot_path(out_dir, spec.slug, "next_char_regions_pca.png", condensed=condensed),
        spaced=spaced, automaton=automaton, condensed=cv,
        repr_name=name,
    )
    _plot(
        "next_char_prob_panels_pca",
        plot_pca_next_char_probability_panels,
        model, text, spec.vectors, chars,
        save_path=_repr_plot_path(out_dir, spec.slug, "next_char_prob_panels_pca.png", condensed=condensed),
        spaced=spaced, automaton=automaton, condensed=cv,
    )
    if words:
        _plot(
            "word_trajectories_pca",
            plot_space_to_space_trajectories,
            text, spec.vectors,
            save_path=_repr_plot_path(out_dir, spec.slug, "word_trajectories_pca.png", condensed=condensed),
            model=None,
            spaced=spaced,
            automaton=automaton,
            condensed=cv,
        )
        _plot(
            "word_trajectories_pca_3d",
            plot_space_to_space_trajectories_3d,
            text, spec.vectors,
            save_path=_repr_plot_path(out_dir, spec.slug, "word_trajectories_pca_3d.png", condensed=condensed),
            model=None,
            spaced=spaced,
            automaton=automaton,
            condensed=cv,
        )


def run_transformer_visualization(
    model: dict,
    text: str,
    out_dir: str,
    *,
    spaced: bool = False,
    automaton: MinimizedVocabAutomaton | None = None,
    words: list[str] | None = None,
    condensed: bool = False,
    quick: bool = False,
    trajectories_only: bool = False,
) -> TransformerActivations:
    """Generate per-representation plots mirroring the RNN visualization suite."""
    from visualize import plot_output_probs

    plot_timer = VizTimer()
    suite_timer = VizTimer()

    cleanup_stale_transformer_plots(out_dir)
    acts = extract_transformer_activations(model, text)
    if acts.num_heads != 1:
        raise ValueError(
            f"Transformer visualization expects num_heads=1 (got {acts.num_heads}). "
            "Retrain with the current defaults: python -m transformer.train --exp <name>"
        )
    specs = collect_representation_specs(acts)
    if trajectories_only:
        specs = [s for s in specs if s.is_readout]
    valid_slugs = {spec.slug for spec in specs}
    cleanup_stale_representation_dirs(out_dir, valid_slugs)
    cleanup_stale_attention_plots(out_dir, len(acts.layers))

    cv_readout = _condense_representation(
        text, acts.block_output, acts.output_probs,
        spaced=spaced, words=words, condensed=condensed,
    )
    if not trajectories_only:
        with plot_timer.section("next_char_prob_sequence"):
            plot_output_probs(
                text, acts.output_probs, model["chars"],
                save_path=_condensed_path(
                    str(numbered_plot_path(out_dir, "next_char_prob_sequence_heatmap.png")),
                    condensed,
                ),
                condensed=cv_readout,
                automaton=automaton,
                spaced=spaced,
                words=words,
            )

        attn_dir = _attention_dir(out_dir)
        attn_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx, layer in enumerate(acts.layers):
            with plot_timer.section("attention_qkv"):
                plot_layer_qkv_figure(
                    text, layer, layer_idx,
                    str(attn_dir / f"layer{layer_idx}_qkv.png"),
                    automaton=automaton, spaced=spaced,
                )
            with plot_timer.section("attention_matrix"):
                plot_layer_attention_figure(
                    model, text, layer_idx,
                    str(attn_dir / f"layer{layer_idx}_attention.png"),
                    automaton=automaton, spaced=spaced,
                )
            with plot_timer.section("attention_lags"):
                plot_layer_attention_lags_figure(
                    text, layer, layer_idx, acts.block_size,
                    str(attn_dir / f"layer{layer_idx}_attention_lags.png"),
                    automaton=automaton, spaced=spaced,
                )

    mode_note = ""
    if trajectories_only:
        mode_note = " (trajectories only: block_output; use animate_micro_curriculum_trajectories.py for closed-loop)"
    elif quick:
        mode_note = " (quick mode: heatmap + PCA only)"
    print(f"transformer: analyzing {len(specs)} separate representations{mode_note}")
    for spec in specs:
        print(f"  · {spec.slug}: {spec.display_name}  shape={spec.vectors.shape}")
        slug_dir = _repr_dir(out_dir, spec.slug)
        slug_dir.mkdir(parents=True, exist_ok=True)
        cleanup_representation_plot_dir(slug_dir)
        with suite_timer.section(spec.slug):
            plot_representation_suite(
                spec,
                model=model,
                text=text,
                out_dir=out_dir,
                output_probs=acts.output_probs,
                spaced=spaced,
                automaton=automaton,
                words=words,
                condensed=condensed,
                timer=plot_timer,
                quick=quick,
                trajectories_only=trajectories_only,
            )

    plot_timer.print_summary(title="Transformer plot types (aggregated across representations)")
    suite_timer.print_summary(title="Per-representation totals")

    return acts
