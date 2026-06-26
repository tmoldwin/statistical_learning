"""Static 2×4 panel figure: closed-loop trajectories for the micro curriculum."""

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
    input_path,
    model_path,
    spaced_experiment_name,
)
from task import REGIMES
from transformer.adapter import extract_transformer_activations, transformer_closed_loop_rollout
from vocab_diagrams import in_word_prefix_at_position
from visualize import (
    _annotate_trajectory_labels,
    _apply_cube_limits_3d,
    _cube_data_limits,
    _plot_step_colored_path_arrows,
    _plot_step_colored_path_arrows_3d,
    _square_data_limits,
    _vocab_word_colors,
    corpus_segments,
    fit_pca_2d_with_evr,
    fit_pca_3d_with_evr,
    load_model_for_viz,
    segment_word_label,
)

PANEL_TITLES: dict[str, str] = {
    "two_word_disjoint": "disjoint",
    "two_word_pos_overlap": "same 2nd letter",
    "two_word_prefix_branch": "shared prefix",
    "two_word_nested": "nested length",
    "three_word_overlap": "suffix family",
    "three_word_permutation": "permutation",
    "three_word_ca_hub": "3-way ca hub",
    "four_word_ca_hub": "4-way ca hub",
}

DEFAULT_STEPS: int | None = None
LOOPS_PER_WORD = 3


def _one_vocab_cycle_steps(words: list[str], *, spaced: bool) -> int:
    if not words:
        return 16
    max_len = max(len(w) for w in words)
    if spaced:
        return len(words) * (max_len + 1) + 1
    return len(words) * max_len + 1


def _default_rollout_steps(words: list[str], *, spaced: bool) -> int:
    """Three full vocabulary cycles by default."""
    return LOOPS_PER_WORD * _one_vocab_cycle_steps(words, spaced=spaced)


def _rollout_word_at_positions(
    generated: list[str],
    seed_len: int,
    n_states: int,
    *,
    spaced: bool,
    words: list[str],
) -> list[str]:
    """Vocabulary word label at each rollout timestep (including in-progress prefixes)."""
    vocab = set(words)
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
    return labels


def _teacher_forced_next_char_accuracy(model: dict, text: str, acts) -> float:
    chars = model["chars"]
    char_to_idx = {c: i for i, c in enumerate(chars)}
    probs = acts.output_probs
    n = min(len(text) - 1, len(probs))
    if n <= 0:
        return float("nan")
    correct = sum(
        int(np.argmax(probs[i])) == char_to_idx[text[i + 1]]
        for i in range(n)
        if text[i + 1] in char_to_idx
    )
    return 100.0 * correct / n


def _training_word_error_pct(model: dict) -> tuple[float, int]:
    metric_iters = model.get("metric_iterations")
    metric_err = model.get("metric_word_error_frac")
    if metric_err is not None and len(metric_err):
        step = int(metric_iters[-1]) if metric_iters is not None and len(metric_iters) else -1
        return 100.0 * float(metric_err[-1]), step
    final = model.get("final_word_error_frac")
    if final is not None:
        return 100.0 * float(final), -1
    return float("nan"), -1


def _learning_stats_line(model: dict, text: str, acts) -> str:
    acc = _teacher_forced_next_char_accuracy(model, text, acts)
    word_err, train_step = _training_word_error_pct(model)
    parts = [f"{acc:.0f}% next-char", f"{word_err:.0f}% inv words"]
    if train_step >= 0:
        parts.append(f"{train_step} train steps")
    return " · ".join(parts)


def _disable_timestep_noise(model: dict) -> None:
    model["timestep_noise_std"] = 0.0
    model["_torch_model"].timestep_noise_std = 0.0


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


def _default_rollout_seed(words: list[str], *, spaced: bool) -> str:
    if spaced:
        return " "
    return words[0][0] if words else ""


def _word_start_segment_flags(
    prefix_labels: list[str],
    word_at_step: list[str],
) -> list[bool]:
    """True when a segment lands on the first character of a vocabulary word."""
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


def _closed_loop_panel(
    ax,
    exp: str,
    *,
    steps: int | None,
    seed: str,
    rng_seed: int,
    n_components: int = 2,
    annotate_fontsize: float = 7.5,
) -> None:
    cfg = EXPERIMENT_CONFIG[exp]
    regime = cfg["regime"]
    words = REGIMES[regime]
    spaced = bool(cfg.get("word_space", False))
    vocab = set(words)
    text = input_path(exp).read_text(encoding="utf-8")[: cfg["viz_length"]]
    model = load_model_for_viz(str(model_path(exp, "transformer")), "transformer")
    _disable_timestep_noise(model)

    acts = extract_transformer_activations(model, text)
    if n_components == 3:
        _, mean, components, evr = fit_pca_3d_with_evr(acts.block_output)
    else:
        _, mean, components, evr = fit_pca_2d_with_evr(acts.block_output)
    learn_line = _learning_stats_line(model, text, acts)

    rollout_steps = steps if steps is not None else _default_rollout_steps(words, spaced=spaced)
    rng = np.random.default_rng(rng_seed)
    hidden, generated = transformer_closed_loop_rollout(
        model, seed_text=seed, steps=rollout_steps, rng=rng,
    )
    z = (hidden - mean) @ components.T
    labels = _rollout_prefix_labels(
        generated, len(seed), len(z), spaced=spaced, vocab=vocab,
    )
    word_at_step = _rollout_word_at_positions(
        generated, len(seed), len(z), spaced=spaced, words=words,
    )
    word_colors = _vocab_word_colors(words)
    word_start = _word_start_segment_flags(labels, word_at_step)
    gray = word_colors["␣"]
    segment_colors = [
        gray if word_at_step[i + 1] == "␣"
        else word_colors.get(word_at_step[i + 1], word_colors[words[0]])
        for i in range(len(z) - 1)
    ]
    segment_linestyles = [":" if is_start else "-" for is_start in word_start]

    if len(z) >= 2:
        if n_components == 3:
            _plot_step_colored_path_arrows_3d(
                ax, z,
                linewidth=1.35,
                alpha=1.0,
                segment_colors=segment_colors,
                segment_linestyles=segment_linestyles,
                arrow_mutation_scale=12.0,
            )
        else:
            _plot_step_colored_path_arrows(
                ax, z,
                linewidth=1.35,
                alpha=1.0,
                zorder=2,
                segment_colors=segment_colors,
                segment_linestyles=segment_linestyles,
                arrow_mutation_scale=12.0,
            )
        _annotate_trajectory_labels(
            ax, z, labels,
            fontsize=annotate_fontsize,
            condense_nearby=True,
            use_leaders=True,
            leader_linewidth=0.4,
        )

    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    pc3 = 100.0 * float(evr[2]) if len(evr) > 2 else 0.0

    if n_components == 3:
        xlim, ylim, zlim = _cube_data_limits(z if len(z) else acts.block_output)
        _apply_cube_limits_3d(ax, xlim, ylim, zlim)
        ax.set_xlabel(f"PC1 ({pc1:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({pc2:.0f}%)", fontsize=8)
        ax.set_zlabel(f"PC3 ({pc3:.0f}%)", fontsize=8)
    else:
        xlim, ylim = _square_data_limits(z if len(z) else acts.block_output)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC1 ({pc1:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({pc2:.0f}%)", fontsize=8)
        ax.axhline(0, color="lightgrey", linewidth=0.5, zorder=0)
        ax.axvline(0, color="lightgrey", linewidth=0.5, zorder=0)

    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.35)

    word_str = ", ".join(words)
    tag = PANEL_TITLES.get(regime, regime)
    ax.set_title(f"{learn_line}\n{word_str}\n({tag}, {rollout_steps} steps)", fontsize=9)


def _write_panels(
    *,
    exps: list[str],
    steps: int | None,
    seed_char: str | None,
    rng_seed: int,
    n_components: int,
    out_path: Path,
    suptitle: str,
) -> None:
    ncols = 4
    nrows = int(np.ceil(len(exps) / ncols))
    figsize = (4.5 * ncols, 4.2 * nrows) if n_components == 3 else (4.2 * ncols, 4.0 * nrows)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    for i, exp in enumerate(exps):
        if n_components == 3:
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        else:
            ax = fig.add_subplot(nrows, ncols, i + 1)

        if not model_path(exp, "transformer").is_file():
            ax.set_visible(False)
            ax.text(0.5, 0.5, f"missing model\n{exp}", ha="center", va="center", transform=ax.transAxes)
            continue

        cfg = EXPERIMENT_CONFIG[exp]
        words = REGIMES[cfg["regime"]]
        spaced = bool(cfg.get("word_space", False))
        seed = seed_char if seed_char is not None else _default_rollout_seed(words, spaced=spaced)
        _closed_loop_panel(
            ax, exp,
            steps=steps,
            seed=seed,
            rng_seed=rng_seed,
            n_components=n_components,
        )

    for j in range(len(exps), nrows * ncols):
        fig.add_subplot(nrows, ncols, j + 1).set_visible(False)

    dim_tag = "3D PCA" if n_components == 3 else "PCA"
    fig.suptitle(f"{suptitle} ({dim_tag})", fontsize=12, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="closed-loop rollout length (default: 3 vocabulary cycles per panel)",
    )
    parser.add_argument(
        "--seed-char",
        default=None,
        help="rollout seed text (default: space for spaced, first vocab letter otherwise)",
    )
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--no-word-space",
        action="store_true",
        help="use unspaced micro-curriculum experiments (no _s suffix)",
    )
    parser.add_argument(
        "--dims",
        choices=("2", "3", "both"),
        default="both",
        help="output 2D panels, 3D panels, or both (default: both)",
    )
    args = parser.parse_args()

    exps = list(MICRO_CURRICULUM) if args.no_word_space else [
        spaced_experiment_name(r) for r in MICRO_CURRICULUM
    ]
    spacing = "unspaced" if args.no_word_space else "spaced"
    base_title = (
        f"Micro curriculum: closed-loop trajectories ({spacing}, transformer block_output"
    )
    out_dir = REPO_ROOT / "experiments" / "micro_curriculum_validation"
    suffix = "_no_space" if args.no_word_space else ""

    if args.dims in ("2", "both"):
        _write_panels(
            exps=exps,
            steps=args.steps,
            seed_char=args.seed_char,
            rng_seed=args.rng_seed,
            n_components=2,
            out_path=out_dir / f"micro_curriculum_closed_loop_panels{suffix}.png",
            suptitle=base_title,
        )
    if args.dims in ("3", "both"):
        _write_panels(
            exps=exps,
            steps=args.steps,
            seed_char=args.seed_char,
            rng_seed=args.rng_seed,
            n_components=3,
            out_path=out_dir / f"micro_curriculum_closed_loop_panels_3d{suffix}.png",
            suptitle=base_title,
        )


if __name__ == "__main__":
    main()
