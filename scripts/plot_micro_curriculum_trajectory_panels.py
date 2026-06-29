"""Multi-panel figure: word trajectories across init seeds and curriculum regimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (
    EXPERIMENT_CONFIG,
    MICRO_CURRICULUM,
    MICRO_CURRICULUM_INIT_SEEDS,
    MODEL_TYPES,
    micro_curriculum_repr_label,
    micro_curriculum_viz_dir,
    model_path,
    spaced_experiment_name,
)
from task import REGIMES
from transformer.adapter import extract_transformer_activations
from visualize import (
    _longest_vocabulary_word_length,
    _square_data_limits,
    _trajectory_vocabulary_words,
    fit_pca_2d_with_evr,
    load_model_for_viz,
    run_forward_pass,
)

PANEL_TITLES: dict[str, str] = {
    "two_word_disjoint": "disjoint",
    "two_word_pos_overlap": "same 2nd letter",
    "two_word_prefix_branch": "shared prefix",
    "three_word_overlap": "suffix family",
    "three_word_permutation": "permutation",
    "three_word_ca_hub": "3-way ca hub",
}


def _word_hidden(
    model: dict,
    snippet: str,
    *,
    model_type: str,
) -> np.ndarray:
    if model_type == "transformer":
        return extract_transformer_activations(model, snippet).block_output
    hidden, _ = run_forward_pass(model, snippet, model_type)
    return hidden


def _isolated_word_paths(
    model: dict,
    vocab_words: list[str],
    *,
    model_type: str,
    max_word_len: int,
) -> list[tuple[str, np.ndarray]]:
    """Teacher-forced trajectories per word; PCA fit on stacked hidden states."""
    word_hidden: list[np.ndarray] = []
    words_in_order: list[str] = []
    for word in sorted(set(vocab_words)):
        snippet = word[:max_word_len]
        if not snippet:
            continue
        word_hidden.append(_word_hidden(model, snippet, model_type=model_type))
        words_in_order.append(word)
    if not word_hidden:
        return []
    stacked = np.vstack(word_hidden)
    _projected, mean, components, _evr = fit_pca_2d_with_evr(stacked)
    paths: list[tuple[str, np.ndarray]] = []
    for word, hidden in zip(words_in_order, word_hidden, strict=True):
        z = (hidden - mean) @ components.T
        paths.append((word, z))
    return paths


def _plot_word_paths(
    ax: plt.Axes,
    paths: list[tuple[str, np.ndarray]],
    *,
    cmap_name: str = "tab10",
) -> list[np.ndarray]:
    plotted: list[np.ndarray] = []
    cmap = plt.get_cmap(cmap_name, max(len(paths), 1))
    for i, (word, z) in enumerate(paths):
        if len(z) < 1:
            continue
        plotted.append(z)
        color = cmap(i)
        ax.plot(z[:, 0], z[:, 1], color=color, linewidth=1.4, alpha=0.9, zorder=2)
        ax.scatter(z[-1, 0], z[-1, 1], color=color, s=12, zorder=3)
        ax.annotate(
            word,
            (z[-1, 0], z[-1, 1]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=6,
            color=color,
            fontweight="bold",
        )
    return plotted


def _trajectory_panel(
    ax: plt.Axes,
    exp: str,
    *,
    model_type: str,
    seed: int,
    show_axis_labels: bool,
) -> bool:
    ckpt = model_path(exp, model_type, seed=seed)
    if not ckpt.is_file():
        ax.text(
            0.5, 0.5, f"missing\nseed {seed}",
            ha="center", va="center", transform=ax.transAxes, fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    cfg = EXPERIMENT_CONFIG[exp]
    regime = cfg["regime"]
    words = REGIMES[regime]
    model = load_model_for_viz(str(ckpt), model_type)
    vocab_words = _trajectory_vocabulary_words("", words)
    max_word_len = _longest_vocabulary_word_length(vocab_words)
    paths = _isolated_word_paths(
        model, vocab_words, model_type=model_type, max_word_len=max_word_len,
    )
    limit_arrays = _plot_word_paths(ax, paths) if paths else []
    if limit_arrays:
        xlim, ylim = _square_data_limits(*limit_arrays)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    if show_axis_labels:
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle=":", alpha=0.3)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-word-space",
        action="store_true",
        help="use unspaced micro-curriculum experiments (output under _ns)",
    )
    parser.add_argument(
        "--model-type",
        default="rnn",
        choices=list(MODEL_TYPES),
        help="which model checkpoints to plot (default: rnn)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(MICRO_CURRICULUM_INIT_SEEDS),
        help="initialization seeds (rows)",
    )
    args = parser.parse_args()

    spaced = not args.no_word_space
    regimes = list(MICRO_CURRICULUM)
    exps = (
        [spaced_experiment_name(r) for r in regimes]
        if spaced
        else regimes
    )
    seeds = list(args.seeds)
    out_dir = micro_curriculum_viz_dir(spaced=spaced, model_type=args.model_type, kind="trajectories")
    out_path = out_dir / "word_trajectories_by_init.png"

    has_any = any(
        model_path(exp, args.model_type, seed=seed).is_file()
        for exp in exps
        for seed in seeds
    )
    if not has_any:
        print(f"skip {out_path}: no seeded {args.model_type} checkpoints for micro curriculum")
        return

    nrows = len(seeds)
    ncols = len(regimes)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.4 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, seed in enumerate(seeds):
        for col, (regime, exp) in enumerate(zip(regimes, exps, strict=True)):
            ax = axes[row, col]
            show_axis_labels = col == 0
            _trajectory_panel(
                ax, exp,
                model_type=args.model_type,
                seed=seed,
                show_axis_labels=show_axis_labels,
            )
            if row == 0:
                words = REGIMES[regime]
                tag = PANEL_TITLES.get(regime, regime)
                ax.set_title(f"{', '.join(words)}\n({tag})", fontsize=8)
            if col == 0:
                ax.text(
                    -0.28, 0.5, f"seed {seed}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    spacing = "spaced" if spaced else "unspaced"
    repr_label = micro_curriculum_repr_label(args.model_type)
    fig.suptitle(
        f"Micro curriculum: isolated word trajectories by init "
        f"({spacing}, {args.model_type} {repr_label}, PCA)",
        fontsize=11,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
