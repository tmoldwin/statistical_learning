"""Side-by-side learning-curve grid for multiple tasks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import TASKS, checkpoint_path, comparison_dir, experiment_uses_word_space, input_path
from vocab_diagrams import corpus_region_snippet
from viz.compare._data import load_task_viz_context
from viz.compare.geometry import _plot_metric_by_condition
from viz.compare.spec import ComparisonSpec
from viz.dimred import fit_embed_2d_with_evr
from visualize import (
    _char_vocab_word_mask,
    _closed_loop_summary_seed,
    _draw_sample_chars,
    _one_vocab_cycle_steps,
    _plot_letter_seed_closed_loop_on_axis,
    _square_data_limits,
    _states_after_first_word,
    _trajectory_seed_letters,
    _vocab_word_colors,
    display_char,
    load_model_for_viz,
    plot_learning_curve_on_axes,
)

_CLOSED_LOOP_INSET_TRIALS = 24

_SUCCESS_WORD_ERR_PCT = 15.0
_OVERVIEW_SAMPLE_LEN = 100
_EPOCH_SAMPLE_LEN = 80
_EPOCH_ROW_HEIGHT = 0.38
_MAX_EPOCH_ROWS = 20


def _subsample_epoch_rows(rows: list[tuple[int, float, str]], *, max_rows: int) -> list[tuple[int, float, str]]:
    if len(rows) <= max_rows:
        return rows
    n = len(rows)
    picks = {0, n - 1}
    best_i = min(range(n), key=lambda i: rows[i][1] if np.isfinite(rows[i][1]) else float("inf"))
    picks.add(best_i)
    for idx in np.linspace(0, n - 1, max_rows, dtype=int):
        picks.add(int(idx))
    ordered = sorted(picks)
    if len(ordered) > max_rows:
        ordered = [ordered[int(i)] for i in np.linspace(0, len(ordered) - 1, max_rows, dtype=int)]
    return [rows[i] for i in ordered]


def _checkpoint_training_stats(
    task: str,
    model_type: str,
    run_seed: int,
) -> dict[str, float] | None:
    ckpt = checkpoint_path(task, model_type, seed=run_seed)
    if not ckpt.is_file():
        return None
    data = np.load(ckpt)
    if "metric_word_error_frac" not in data.files:
        return None
    final_we = float(data["metric_word_error_frac"][-1]) * 100.0
    best_we = float(data["best_metric_word_error_frac"]) * 100.0
    if "metric_val_ce" in data.files and len(data["metric_val_ce"]):
        final_ce = float(data["metric_val_ce"][-1])
    elif "loss_smooth" in data.files:
        seq = float(data["sequence_length"]) if "sequence_length" in data.files else 1.0
        final_ce = float(data["loss_smooth"][-1]) / max(seq, 1.0)
    else:
        final_ce = float("nan")
    final_iter = float(data["loss_iterations"][-1]) if "loss_iterations" in data.files else float("nan")
    return {
        "final_word_err_pct": final_we,
        "best_word_err_pct": best_we,
        "final_ce": final_ce,
        "final_iter": final_iter,
    }


def plot_learning_summary(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
) -> Path:
    """Bar plots of final training outcomes by condition; error bars = SEM across seeds."""
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = tuple(spec.tasks)
    condition_labels = [spec.label_for(t) for t in tasks]
    seed_colors = {int(s): plt.cm.tab20(i % 20) for i, s in enumerate(run_seeds)}

    metric_specs = (
        ("final_word_err_pct", "final % OOV chars"),
        ("best_word_err_pct", "best % OOV chars"),
        ("final_ce", "final val CE / char"),
        ("final_iter", "training iterations"),
    )
    groups_by_key: dict[str, list[tuple[list[float], list[int]]]] = {
        key: [] for key, _ in metric_specs
    }

    for task in tasks:
        for key, _ in metric_specs:
            vals: list[float] = []
            seed_ids: list[int] = []
            for run_seed in run_seeds:
                stats = _checkpoint_training_stats(task, spec.model_type, run_seed)
                if stats is None:
                    continue
                vals.append(float(stats[key]))
                seed_ids.append(int(run_seed))
            groups_by_key[key].append((vals, seed_ids))

    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 7.0), constrained_layout=True, squeeze=False,
    )
    for ax, (key, ylabel) in zip(axes.ravel(), metric_specs):
        _plot_metric_by_condition(
            ax, groups_by_key[key], condition_labels, ylabel=ylabel, seed_colors=seed_colors,
        )
        if key == "final_word_err_pct":
            ax.axhline(_SUCCESS_WORD_ERR_PCT, color="#cc3333", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(
        f"{spec.display_title}: training summary ({spec.model_type}, n={len(run_seeds)} seeds)",
        fontsize=11,
    )
    out_dir = comparison_dir(spec.name, "learning_curves")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _add_mean_pca_trajectory_inset(ax, task: str, model_type: str, run_seed: int) -> None:
    """Upper-right inset: mean closed-loop trajectory in the model's PCA plane."""
    try:
        ctx = load_task_viz_context(task, model_type=model_type, seed=run_seed)
        fit_states, trajs = _states_after_first_word(
            ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=ctx.words,
        )
        _projected, mean, components, _evr = fit_embed_2d_with_evr(
            fit_states, method="pca", trajectories=trajs,
        )
        vocab_words = list(ctx.words)
        seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
        summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
        summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
        if vocab_words:
            summary_steps += max(len(w) for w in vocab_words)
        word_colors = _vocab_word_colors(vocab_words)

        inset = ax.inset_axes([0.56, 0.52, 0.43, 0.46])
        inset.set_facecolor((1.0, 1.0, 1.0, 0.85))
        limit_arrays: list[np.ndarray] = []
        _plot_letter_seed_closed_loop_on_axis(
            inset, ctx.model,
            seed_letters=[summary_seed],
            steps=summary_steps,
            closed_loop_seed=0,
            mean=mean,
            components=components,
            limit_arrays=limit_arrays,
            vocab_words=vocab_words,
            word_colors=word_colors,
            spaced=ctx.spaced,
            annotate=False,
            is_3d=False,
            unique_word_labels=True,
            average_trials=_CLOSED_LOOP_INSET_TRIALS,
        )
        if limit_arrays:
            xlim, ylim = _square_data_limits(*limit_arrays, padding_frac=0.10)
            inset.set_xlim(xlim)
            inset.set_ylim(ylim)
            inset.set_aspect("equal", adjustable="box")
        inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in inset.spines.values():
            spine.set_linewidth(0.6)
            spine.set_alpha(0.5)
    except Exception as exc:  # inset failure should not break the grid
        print(f"warn: no PCA inset for {task} seed {run_seed}: {exc}")


def _corpus_text(task: str, model: dict | None = None) -> str:
    corpus_path = input_path(task)
    if corpus_path.is_file():
        return corpus_path.read_text(encoding="utf-8")
    if model is not None:
        return str(model.get("demo_snippet", ""))
    return ""


def _rollout_at_training_phase(model: dict, phase: str) -> tuple[int, str]:
    """Rollout sample at start, midpoint, or end of training."""
    iters = np.asarray(model.get("metric_iterations", []), dtype=int)
    samples = model.get("metric_rollout_samples")
    if samples is None or len(iters) == 0:
        fallback = str(model.get("demo_after", "")) or str(model.get("sample_after", ""))
        return 0, fallback
    if phase == "begin":
        idx = 0
    elif phase == "end":
        idx = len(iters) - 1
    else:
        idx = len(iters) // 2
    return int(iters[idx]), str(samples[idx])


def _overview_fontsize(ncols: int, n_chars: int) -> float:
    return max(3.8, min(7.0, 520 / (max(ncols, 1) * (max(n_chars, 1) ** 0.55))))


def _draw_overview_text_line(
    ax,
    text: str,
    y: float,
    *,
    vocab: set[str],
    spaced: bool,
    fontsize: float,
) -> None:
    snippet = text[:_OVERVIEW_SAMPLE_LEN]
    n = len(snippet)
    if n == 0:
        return
    x_step = 0.98 / n
    mask = _char_vocab_word_mask(snippet, vocab, spaced=spaced) if vocab else [True] * n
    for i, ch in enumerate(snippet):
        color = "#2ca02c" if mask[i] else "#d62728"
        ax.text(
            i * x_step,
            y,
            display_char(ch),
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=fontsize,
            color=color,
            va="center",
            ha="left",
        )


def _plot_io_samples(ax, model, *, task: str, ncols: int) -> None:
    """One corpus input + example rollouts at training begin / middle / end."""
    ax.set_axis_off()
    vocab = set(map(str, model.get("vocab_words", [])))
    spaced = experiment_uses_word_space(task)

    demo_in = corpus_region_snippet(
        _corpus_text(task, model), _OVERVIEW_SAMPLE_LEN, "middle",
    )
    out_begin = _rollout_at_training_phase(model, "begin")
    out_middle = _rollout_at_training_phase(model, "middle")
    out_end = _rollout_at_training_phase(model, "end")

    if not demo_in and not out_begin[1]:
        ax.text(
            0.5, 0.5, "no samples",
            ha="center", va="center", transform=ax.transAxes, fontsize=6, color="0.5",
        )
        return

    rows: list[tuple[str, str, float]] = [
        ("in", demo_in, 0.90),
        (f"out · begin · iter {out_begin[0]} (example)", out_begin[1], 0.68),
        (f"out · middle · iter {out_middle[0]} (example)", out_middle[1], 0.46),
        (f"out · end · iter {out_end[0]} (example)", out_end[1], 0.22),
    ]
    snippets = [s for _, s, _ in rows if s]
    fs = _overview_fontsize(ncols, max((len(s) for s in snippets), default=1))
    label_fs = max(fs - 0.3, 3.8)

    for label, snippet, y in rows:
        if not snippet:
            continue
        ax.text(0.0, y + 0.10, label, transform=ax.transAxes, fontsize=label_fs, va="bottom", color="0.40")
        if vocab:
            _draw_overview_text_line(ax, snippet, y, vocab=vocab, spaced=spaced, fontsize=fs)
        else:
            ax.text(
                0.0, y, snippet[:_OVERVIEW_SAMPLE_LEN], transform=ax.transAxes,
                fontfamily="monospace", fontsize=fs, va="center",
            )


def _epoch_rollout_rows(model: dict) -> list[tuple[int, float, str]]:
    iters = np.asarray(model.get("metric_iterations", []), dtype=int)
    samples = model.get("metric_rollout_samples")
    if samples is None or len(samples) == 0:
        return []
    errs = np.asarray(model.get("metric_word_error_frac", []), dtype=float)
    rows: list[tuple[int, float, str]] = []
    for i, it in enumerate(iters):
        sample = str(samples[i]) if i < len(samples) else ""
        err = float(errs[i]) * 100.0 if i < len(errs) else float("nan")
        rows.append((int(it), err, sample))
    return rows


def plot_epoch_rollouts(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
) -> list[Path]:
    """Per task/seed: rollout strings at each metric eval under learning_curves/epochs/."""
    run_seeds = seeds if seeds is not None else spec.seeds
    out_dir = comparison_dir(spec.name, "learning_curves") / "epochs"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for task in spec.tasks:
        for run_seed in run_seeds:
            ckpt = checkpoint_path(task, spec.model_type, seed=run_seed)
            out_path = out_dir / f"{task}_seed{run_seed}.png"
            if not ckpt.is_file():
                continue
            model = load_model_for_viz(str(ckpt), spec.model_type)
            spaced = experiment_uses_word_space(task)
            rows = _epoch_rollout_rows(model)
            if not rows:
                continue
            full_count = len(rows)
            rows = _subsample_epoch_rows(rows, max_rows=_MAX_EPOCH_ROWS)

            vocab = set(map(str, model.get("vocab_words", [])))
            n_rows = len(rows)
            fig_h = max(2.5, 0.55 + _EPOCH_ROW_HEIGHT * n_rows)
            fig, axes = plt.subplots(n_rows, 1, figsize=(14.0, fig_h), constrained_layout=True)
            if n_rows == 1:
                axes = [axes]

            for ax, (it, err_pct, sample) in zip(axes, rows, strict=True):
                ax.set_axis_off()
                err_note = f"avg {err_pct:.1f}% OOV chars" if np.isfinite(err_pct) else ""
                ax.text(0.0, 0.92, f"iter {it}  {err_note}", transform=ax.transAxes, fontsize=8, va="top", color="0.35")
                text = sample[:_EPOCH_SAMPLE_LEN]
                _draw_sample_chars(
                    ax, text, 0.35,
                    vocab=vocab if vocab else None,
                    spaced=spaced,
                    color_by_vocab=bool(vocab),
                    show_word_separators=bool(vocab) and not spaced,
                )

            label = spec.label_for(task)
            subsample_note = f" ({full_count} evals, showing {n_rows})" if full_count > n_rows else ""
            fig.suptitle(
                f"{label} · seed {run_seed}: example rollouts by iteration{subsample_note}",
                fontsize=11,
            )
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(out_path)
            print(f"wrote {out_path}")

    return paths


def plot_learning_curves(
    spec: ComparisonSpec,
    *,
    truncate_to_plateau: bool = False,
    seeds: tuple[int, ...] | None = None,
    write_epoch_rollouts: bool = True,
) -> Path:
    """Grid: rows = tasks (conditions), columns = RNG seeds."""
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = list(spec.tasks)
    out_dir = comparison_dir(spec.name, "learning_curves")
    out_path = out_dir / "overview.png"

    if not any(
        checkpoint_path(t, spec.model_type, seed=s).is_file()
        for t in tasks
        for s in run_seeds
    ):
        raise FileNotFoundError(f"no {spec.model_type} checkpoints for comparison {spec.name!r}")

    ncols = len(run_seeds)
    nrows = len(tasks)
    row_h = 4.4 if ncols <= 10 else 4.0

    fig = plt.figure(figsize=(4.4 * ncols, row_h * nrows), constrained_layout=True)
    outer_gs = fig.add_gridspec(nrows, ncols, hspace=0.38, wspace=0.18)

    for row_idx, task in enumerate(tasks):
        for col_idx, run_seed in enumerate(run_seeds):
            inner = outer_gs[row_idx, col_idx].subgridspec(
                2, 1, height_ratios=[2.3, 1.15], hspace=0.04,
            )
            ax = fig.add_subplot(inner[0])
            ax_samples = fig.add_subplot(inner[1])
            show_ylabel = col_idx == 0
            show_metric_ylabel = col_idx == ncols - 1
            show_legend = row_idx == 0 and col_idx == 0

            ckpt = checkpoint_path(task, spec.model_type, seed=run_seed)
            if not ckpt.is_file():
                ax.set_visible(False)
                ax_samples.set_visible(False)
                ax.text(
                    0.5, 0.5, f"missing\nseed {run_seed}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                )
                continue
            model = load_model_for_viz(str(ckpt), spec.model_type)
            seq_len = int(TASKS[task].get("sequence_length", 1))
            if row_idx == 0:
                ax.set_title(f"seed {run_seed}", fontsize=9, fontweight="bold")
            if not plot_learning_curve_on_axes(
                ax,
                model,
                title=spec.label_for(task) if col_idx == 0 else "",
                compact=True,
                smoothed=True,
                truncate_to_plateau=truncate_to_plateau,
                show_legend=show_legend,
                show_ylabel=show_ylabel,
                show_metric_ylabel=show_metric_ylabel,
                sequence_length=seq_len,
            ):
                ax.text(
                    0.5, 0.5, f"no loss history",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                )
            else:
                if show_legend:
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.set_loc("center right")
                _add_mean_pca_trajectory_inset(ax, task, spec.model_type, run_seed)
            _plot_io_samples(ax_samples, model, task=task, ncols=ncols)

    fig.suptitle(
        f"{spec.display_title}: training ({spec.model_type}) · orange = mean % OOV chars, stochastic rollouts from corpus prompts",
        fontsize=12,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if write_epoch_rollouts:
        plot_epoch_rollouts(spec, seeds=run_seeds)
    return out_path
