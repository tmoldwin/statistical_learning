"""Plot teacher-forced trajectories: one panel per vocabulary word."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import TASKS, comparison_dir, plots_dir
from viz.compare._data import load_task_viz_context
from viz.compare.spec import COMPARISON_PRESETS
from viz.dimred import (
    embed_axis_labels_2d,
    embed_axis_labels_3d,
    embed_dim_label,
    embed_save_path,
    fit_embed_2d_with_evr,
    fit_embed_3d_with_evr,
)
from visualize import (
    _embed_trajectories_for_text,
    _longest_vocabulary_word_length,
    _plot_teacher_forced_vocab_grid,
    _teacher_forced_vocab_trajectories,
)


def plot_word_trajectories(
    task: str,
    *,
    seed: int,
    model_type: str = "rnn",
    dimensions: int = 2,
    embed_method: str = "pca",
    outfile: Path | None = None,
) -> Path:
    """One subplot per vocab word (teacher-forced from first character)."""
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")

    ctx = load_task_viz_context(task, model_type=model_type, seed=seed)
    vocab_words = list(ctx.words)
    max_word_len = _longest_vocabulary_word_length(vocab_words)
    trajs = _embed_trajectories_for_text(
        ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=vocab_words,
    )

    is_3d = dimensions == 3
    if is_3d:
        _, mean, components, evr = fit_embed_3d_with_evr(
            ctx.hidden_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel, zlabel = embed_axis_labels_3d(evr, embed_method)
    else:
        _, mean, components, evr = fit_embed_2d_with_evr(
            ctx.hidden_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel = embed_axis_labels_2d(evr, embed_method)
        zlabel = None

    trained_tf = _teacher_forced_vocab_trajectories(
        ctx.model, vocab_words, mean=mean, components=components, max_word_len=max_word_len,
    )
    n_trained = len(trained_tf)
    ncols = min(4, max(1, n_trained))
    nrows = int(math.ceil(n_trained / ncols))

    fig = plt.figure(figsize=(5.5 * ncols, 3.2 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.48, wspace=0.30)
    fig.suptitle(
        f"{task} · seed {seed} · teacher-forced "
        f"({n_trained} words · {dimensions}D {embed_dim_label(embed_method)})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    _plot_teacher_forced_vocab_grid(
        fig, gs, trained_tf,
        is_3d=is_3d, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
    )
    fig.subplots_adjust(top=0.93, bottom=0.06)

    if outfile is None:
        sub = "3d" if is_3d else "2d"
        base = f"word_trajectories_{sub}_trained.png"
        out_dir = plots_dir(task, model_type) / "trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / embed_save_path(base, embed_method)
    else:
        out_path = Path(outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS.keys()))
    parser.add_argument("--preset", choices=sorted(COMPARISON_PRESETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--model-type", default="rnn", choices=["rnn", "rnn_dale"])
    parser.add_argument("--dimensions", type=int, default=2, choices=[2, 3])
    parser.add_argument("--embed-method", default="pca", choices=["pca", "jpca"])
    parser.add_argument(
        "--comparison-out",
        action="store_true",
        help="write under experiments/comparisons/<preset>/trajectories/",
    )
    parser.add_argument("--outfile")
    args = parser.parse_args()

    if not args.task and not args.preset:
        parser.error("provide --task or --preset")

    if args.preset:
        spec = COMPARISON_PRESETS[args.preset]
        tasks = list(spec.tasks)
        model_type = spec.model_type
    else:
        tasks = [args.task]
        model_type = args.model_type

    seeds = tuple(args.seeds) if args.seeds else (args.seed,)

    for task in tasks:
        for seed in seeds:
            outfile: Path | None = None
            if args.outfile:
                outfile = Path(args.outfile)
            elif args.comparison_out and args.preset:
                spec = COMPARISON_PRESETS[args.preset]
                label = task.replace("_ns", "").replace("_", "-")
                sub = "3d" if args.dimensions == 3 else "2d"
                name = embed_save_path(
                    f"word_trajectories_{label}_s{seed}_{sub}.png",
                    args.embed_method,
                )
                outfile = comparison_dir(spec.name, "trajectories") / name

            path = plot_word_trajectories(
                task,
                seed=seed,
                model_type=model_type,
                dimensions=args.dimensions,
                embed_method=args.embed_method,
                outfile=outfile,
            )
            print(path)


if __name__ == "__main__":
    main()
