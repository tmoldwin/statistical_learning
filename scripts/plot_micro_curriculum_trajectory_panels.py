"""Multi-panel figure: trained word trajectories for the micro curriculum."""

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
    MODEL_TYPES,
    input_path,
    micro_curriculum_repr_label,
    micro_curriculum_viz_dir,
    model_path,
    spaced_experiment_name,
)
from task import REGIMES
from transformer.adapter import extract_transformer_activations
from visualize import (
    _corpus_vocab,
    _longest_vocabulary_word_length,
    _plot_trained_word_examples,
    _square_data_limits,
    _trained_word_examples,
    _trajectory_vocabulary_words,
    corpus_segments,
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


def _trajectory_panel(
    ax: plt.Axes,
    exp: str,
    *,
    model_type: str,
    annotate_fontsize: float = 7.5,
) -> None:
    cfg = EXPERIMENT_CONFIG[exp]
    regime = cfg["regime"]
    words = REGIMES[regime]
    spaced = bool(cfg.get("word_space", False))
    text = input_path(exp).read_text(encoding="utf-8")[: cfg["viz_length"]]
    model = load_model_for_viz(str(model_path(exp, model_type)), model_type)
    if model_type == "transformer":
        hidden = extract_transformer_activations(model, text).block_output
    else:
        hidden, _ = run_forward_pass(model, text, model_type)

    projected, _mean, _components, evr = fit_pca_2d_with_evr(hidden)
    segments = corpus_segments(text, list(_corpus_vocab(text, words) or []), spaced=spaced)
    vocab_words = _trajectory_vocabulary_words(text, words)
    max_word_len = _longest_vocabulary_word_length(vocab_words)
    examples = _trained_word_examples(segments, vocab_words)
    paths = _plot_trained_word_examples(
        ax,
        projected,
        examples,
        max_word_len=max_word_len,
        annotate_fontsize=annotate_fontsize,
    )
    limit_arrays = paths if paths else [projected]
    xlim, ylim = _square_data_limits(*limit_arrays)

    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"PC1 ({pc1:.0f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({pc2:.0f}%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.axhline(0, color="lightgrey", linewidth=0.5, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.5, zorder=0)

    word_str = ", ".join(words)
    tag = PANEL_TITLES.get(regime, regime)
    ax.set_title(f"{word_str}\n({tag})", fontsize=9)


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
    args = parser.parse_args()

    spaced = not args.no_word_space
    exps = (
        [spaced_experiment_name(r) for r in MICRO_CURRICULUM]
        if spaced
        else list(MICRO_CURRICULUM)
    )
    out_dir = micro_curriculum_viz_dir(spaced=spaced, model_type=args.model_type, kind="trajectories")
    out_path = out_dir / "panels.png"
    if not any(model_path(exp, args.model_type).is_file() for exp in exps):
        print(f"skip {out_path}: no {args.model_type} checkpoints for micro curriculum")
        return

    n = len(exps)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, exp in zip(axes_flat, exps, strict=False):
        if not model_path(exp, args.model_type).is_file():
            ax.set_visible(False)
            ax.text(
                0.5, 0.5, f"missing {args.model_type} model\n{exp}",
                ha="center", va="center", transform=ax.transAxes,
            )
            continue
        _trajectory_panel(ax, exp, model_type=args.model_type)

    for ax in axes_flat[len(exps):]:
        ax.set_visible(False)

    spacing = "spaced" if spaced else "unspaced"
    repr_label = micro_curriculum_repr_label(args.model_type)
    fig.suptitle(
        f"Micro curriculum: teacher-forced word trajectories "
        f"({spacing}, {args.model_type} {repr_label}, PCA)",
        fontsize=12,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
