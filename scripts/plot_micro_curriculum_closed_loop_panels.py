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
    MICRO_CURRICULUM_INIT_SEEDS,
    MODEL_TYPES,
    input_path,
    micro_curriculum_repr_label,
    micro_curriculum_viz_dir,
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
    _is_return_to_baseline_segment,
    _plot_step_colored_path_arrows,
    _plot_step_colored_path_arrows_3d,
    _square_data_limits,
    _vocab_word_colors,
    corpus_segments,
    fit_pca_2d_with_evr,
    fit_pca_3d_with_evr,
    load_model_for_viz,
    rnn_closed_loop_rollout,
    run_forward_pass,
    segment_word_label,
)

PANEL_TITLES: dict[str, str] = {
    "two_word_disjoint": "disjoint",
    "two_word_pos_overlap": "same 2nd letter",
    "two_word_prefix_branch": "shared prefix",
    "three_word_overlap": "suffix family",
    "three_word_permutation": "permutation",
    "three_word_ca_hub": "3-way ca hub",
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


def _unspaced_word_at_each_index(
    text: str,
    words: list[str],
    vocab: set[str],
) -> list[str]:
    """Greedy unspaced parse: earliest vocab completion from each word start."""
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
    """Vocabulary word label at each rollout timestep (including in-progress prefixes)."""
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


def _teacher_forced_next_char_accuracy(model: dict, text: str, output_probs: np.ndarray) -> float:
    chars = model["chars"]
    char_to_idx = {c: i for i, c in enumerate(chars)}
    probs = output_probs
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


def _learning_stats_line(model: dict, text: str, output_probs: np.ndarray) -> str:
    acc = _teacher_forced_next_char_accuracy(model, text, output_probs)
    word_err, train_step = _training_word_error_pct(model)
    parts = [f"{acc:.0f}% next-char", f"{word_err:.0f}% inv words"]
    if train_step >= 0:
        parts.append(f"{train_step} train steps")
    return " · ".join(parts)


def _disable_timestep_noise(model: dict) -> None:
    model["timestep_noise_std"] = 0.0
    if model.get("model_type") == "transformer":
        model["_torch_model"].timestep_noise_std = 0.0


def _teacher_forced_hidden(
    model: dict,
    text: str,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    if model_type == "transformer":
        acts = extract_transformer_activations(model, text)
        return acts.block_output, acts.output_probs
    hidden, output_probs = run_forward_pass(model, text, model_type)
    return hidden, output_probs


def _closed_loop_rollout(
    model: dict,
    *,
    seed_text: str,
    steps: int,
    rng: np.random.Generator,
    model_type: str,
) -> tuple[np.ndarray, list[str]]:
    if model_type == "transformer":
        return transformer_closed_loop_rollout(
            model, seed_text=seed_text, steps=steps, rng=rng,
        )
    return rnn_closed_loop_rollout(
        model, seed_text=seed_text, steps=steps, rng=rng,
    )


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
    model_type: str,
    init_seed: int,
    steps: int | None,
    seed: str,
    rng_seed: int,
    n_components: int = 2,
    annotate_fontsize: float = 6.5,
    show_axis_labels: bool = True,
    show_title: bool = True,
) -> bool:
    ckpt = model_path(exp, model_type, seed=init_seed)
    if not ckpt.is_file():
        missing_msg = f"missing\nseed {init_seed}"
        if n_components == 3:
            ax.text2D(0.5, 0.5, missing_msg, transform=ax.transAxes, ha="center", va="center", fontsize=7)
        else:
            ax.text(0.5, 0.5, missing_msg, ha="center", va="center", transform=ax.transAxes, fontsize=7)
        return False

    cfg = EXPERIMENT_CONFIG[exp]
    regime = cfg["regime"]
    words = REGIMES[regime]
    spaced = bool(cfg.get("word_space", False))
    vocab = set(words)
    text = input_path(exp).read_text(encoding="utf-8")[: cfg["viz_length"]]
    model = load_model_for_viz(str(ckpt), model_type)
    _disable_timestep_noise(model)

    hidden, output_probs = _teacher_forced_hidden(model, text, model_type)
    if n_components == 3:
        _, mean, components, evr = fit_pca_3d_with_evr(hidden)
    else:
        _, mean, components, evr = fit_pca_2d_with_evr(hidden)
    learn_line = _learning_stats_line(model, text, output_probs)

    rollout_steps = steps if steps is not None else _default_rollout_steps(words, spaced=spaced)
    rng = np.random.default_rng(rng_seed)
    hidden, generated = _closed_loop_rollout(
        model, seed_text=seed, steps=rollout_steps, rng=rng, model_type=model_type,
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
        gray
        if _is_return_to_baseline_segment(
            labels[i], labels[i + 1], word_start=word_start[i],
        )
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
            dedupe=True,
            word_keys=word_at_step,
            label_colors=[
                word_colors.get(word_at_step[i], word_colors[words[0]])
                for i in range(len(labels))
            ],
            use_leaders=True,
            leader_linewidth=0.4,
        )

    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    pc3 = 100.0 * float(evr[2]) if len(evr) > 2 else 0.0

    if n_components == 3:
        xlim, ylim, zlim = _cube_data_limits(z if len(z) else hidden)
        _apply_cube_limits_3d(ax, xlim, ylim, zlim)
        if show_axis_labels:
            ax.set_xlabel(f"PC1 ({pc1:.0f}%)", fontsize=7)
            ax.set_ylabel(f"PC2 ({pc2:.0f}%)", fontsize=7)
            ax.set_zlabel(f"PC3 ({pc3:.0f}%)", fontsize=7)
    else:
        xlim, ylim = _square_data_limits(z if len(z) else hidden)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        if show_axis_labels:
            ax.set_xlabel(f"PC1 ({pc1:.0f}%)", fontsize=7)
            ax.set_ylabel(f"PC2 ({pc2:.0f}%)", fontsize=7)
            ax.axhline(0, color="lightgrey", linewidth=0.5, zorder=0)
            ax.axvline(0, color="lightgrey", linewidth=0.5, zorder=0)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle=":", alpha=0.35)

    if show_title:
        word_str = ", ".join(words)
        tag = PANEL_TITLES.get(regime, regime)
        ax.set_title(
            f"{learn_line}\n{word_str}\n({tag}, {rollout_steps} steps)",
            fontsize=8,
        )
    return True


def _write_panels(
    *,
    exps: list[str],
    regimes: list[str],
    init_seeds: list[int],
    model_type: str,
    steps: int | None,
    seed_char: str | None,
    n_components: int,
    out_path: Path,
    suptitle: str,
) -> None:
    has_any = any(
        model_path(exp, model_type, seed=s).is_file()
        for exp in exps
        for s in init_seeds
    )
    if not has_any:
        print(f"skip {out_path}: no seeded {model_type} checkpoints for micro curriculum")
        return

    nrows = len(init_seeds)
    ncols = len(exps)
    figsize = (
        (3.0 * ncols, 2.8 * nrows)
        if n_components == 3
        else (2.8 * ncols, 2.6 * nrows)
    )
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    for row, init_seed in enumerate(init_seeds):
        for col, (regime, exp) in enumerate(zip(regimes, exps, strict=True)):
            idx = row * ncols + col + 1
            if n_components == 3:
                ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
            else:
                ax = fig.add_subplot(nrows, ncols, idx)

            cfg = EXPERIMENT_CONFIG[exp]
            words = REGIMES[cfg["regime"]]
            spaced = bool(cfg.get("word_space", False))
            rollout_seed = seed_char if seed_char is not None else _default_rollout_seed(words, spaced=spaced)
            _closed_loop_panel(
                ax, exp,
                model_type=model_type,
                init_seed=init_seed,
                steps=steps,
                seed=rollout_seed,
                rng_seed=init_seed,
                n_components=n_components,
                show_axis_labels=col == 0,
                show_title=False,
            )
            if row == 0:
                word_str = ", ".join(words)
                tag = PANEL_TITLES.get(regime, regime)
                ax.set_title(f"{word_str}\n({tag})", fontsize=8)
            if col == 0:
                row_label = f"init {init_seed}"
                if n_components == 3:
                    ax.text2D(
                        -0.12, 0.5, row_label,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        -0.30, 0.5, row_label,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                    )

    dim_tag = "3D PCA" if n_components == 3 else "PCA"
    fig.suptitle(f"{suptitle} ({dim_tag})", fontsize=11, y=1.02)
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
    parser.add_argument("--rng-seed", type=int, default=None,
                        help="deprecated; rollout RNG now follows each init seed")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(MICRO_CURRICULUM_INIT_SEEDS),
        help="weight-init seeds (rows)",
    )
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
    parser.add_argument(
        "--model-type",
        default="rnn",
        choices=list(MODEL_TYPES),
        help="which model checkpoints to plot (default: rnn)",
    )
    args = parser.parse_args()

    exps = list(MICRO_CURRICULUM) if args.no_word_space else [
        spaced_experiment_name(r) for r in MICRO_CURRICULUM
    ]
    regimes = list(MICRO_CURRICULUM)
    init_seeds = list(args.seeds)
    spaced = not args.no_word_space
    spacing = "unspaced" if args.no_word_space else "spaced"
    repr_label = micro_curriculum_repr_label(args.model_type)
    base_title = (
        f"Micro curriculum: closed-loop trajectories by init "
        f"({spacing}, {args.model_type} {repr_label}"
    )
    out_dir = micro_curriculum_viz_dir(spaced=spaced, model_type=args.model_type, kind="closed_loop")

    if args.dims in ("2", "both"):
        _write_panels(
            exps=exps,
            regimes=regimes,
            init_seeds=init_seeds,
            model_type=args.model_type,
            steps=args.steps,
            seed_char=args.seed_char,
            n_components=2,
            out_path=out_dir / "closed_loop_by_init.png",
            suptitle=base_title,
        )
    if args.dims in ("3", "both"):
        _write_panels(
            exps=exps,
            regimes=regimes,
            init_seeds=init_seeds,
            model_type=args.model_type,
            steps=args.steps,
            seed_char=args.seed_char,
            n_components=3,
            out_path=out_dir / "closed_loop_by_init_3d.png",
            suptitle=base_title,
        )


if __name__ == "__main__":
    main()
