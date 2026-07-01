"""Side-by-side PCA state plots across init seeds (rows = color feature, cols = seeds)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import TASKS, common_seeds, comparison_dir, plots_dir
from task import corpus_for_experiment, label_extensions_for_experiment
from unit_selectivity import FEATURE_DISPLAY
from viz.compare._data import load_task_viz_context
from viz.compare.spec import COMPARISON_PRESETS
from viz.compare.state_clusters import cluster_counts_hidden_and_pca, format_cluster_counts
from viz.dimred import fit_embed_2d_with_evr
from vocab_diagrams import build_minimized_vocabulary_automaton, select_analysis_window
from visualize import (
    _embed_trajectories_for_text,
    _pca_feature_panel_context,
    _plot_2d_feature_colored_pca_panel,
)

_COLOR_ROWS = ("dfa", "position", "char")

_COMPACT_ROW_LABELS = {"dfa": "DFA", "position": "position", "char": "char"}
_TINY_ROW_LABELS = {"dfa": "DFA", "position": "pos", "char": "chr"}


def _seed_grid_label_sizes(n_cols: int) -> tuple[float, float, dict[str, str], float]:
    """Return (head_fs, row_fs, row_labels, hspace) scaled to column count."""
    if n_cols <= 8:
        return 9.0, 8.0, FEATURE_DISPLAY, 0.18
    if n_cols <= 12:
        return 7.0, 7.0, _COMPACT_ROW_LABELS, 0.28
    return 6.0, 6.0, _TINY_ROW_LABELS, 0.34


def _label_words_for_task(task: str, *, seed: int, spaced: bool, words: list[str]) -> list[str] | None:
    if not words or spaced:
        return None
    cfg = TASKS[task]
    full_text = corpus_for_experiment(task, seed=seed)
    extensions = label_extensions_for_experiment(task)
    _, _, label_words = select_analysis_window(
        full_text, words, int(cfg.get("viz_length", 50)),
        spaced=spaced, extensions=extensions,
    )
    return label_words


def _seed_panel_data(
    task: str,
    *,
    seed: int,
    model_type: str = "rnn",
    embed_method: str = "pca",
) -> dict | None:
    ctx = load_task_viz_context(task, model_type=model_type, seed=seed)
    words = list(ctx.words)
    if not words:
        return None
    automaton = build_minimized_vocabulary_automaton(words)
    label_words = _label_words_for_task(task, seed=seed, spaced=ctx.spaced, words=words)
    panel_ctx = _pca_feature_panel_context(
        ctx.text, ctx.hidden_states,
        spaced=ctx.spaced, automaton=automaton, words=words,
        label_words=label_words, condensed=None,
    )
    if panel_ctx is None:
        return None

    trajs = _embed_trajectories_for_text(
        ctx.text, panel_ctx["hidden_states"], spaced=ctx.spaced, words=words,
    )
    pca_xy, _, _, _ = fit_embed_2d_with_evr(
        panel_ctx["hidden_states"], method=embed_method, trajectories=trajs,
    )
    pad_x = max((pca_xy[:, 0].max() - pca_xy[:, 0].min()) * 0.06, 0.05)
    pad_y = max((pca_xy[:, 1].max() - pca_xy[:, 1].min()) * 0.06, 0.05)
    xlim = (float(pca_xy[:, 0].min() - pad_x), float(pca_xy[:, 0].max() + pad_x))
    ylim = (float(pca_xy[:, 1].min() - pad_y), float(pca_xy[:, 1].max() + pad_y))
    k_h, k_pca = cluster_counts_hidden_and_pca(panel_ctx["hidden_states"], pca_xy)
    n_steps = panel_ctx["n"]
    path_indices = [list(range(n_steps))] if n_steps >= 2 else []
    return {
        "pca_xy": pca_xy,
        "panel_ctx": panel_ctx,
        "automaton": automaton,
        "xlim": xlim,
        "ylim": ylim,
        "k_h": k_h,
        "k_pca": k_pca,
        "path_indices": path_indices,
    }


def plot_state_pca_seed_comparison(
    task: str,
    *,
    seeds: tuple[int, ...],
    model_type: str = "rnn",
    embed_method: str = "pca",
    annot_style: str = "annots_only",
    outfile: Path | None = None,
) -> Path:
    """Grid: rows = DFA / position / char coloring; columns = seeds."""
    seed_data: dict[int, dict] = {}
    for seed in seeds:
        data = _seed_panel_data(task, seed=seed, model_type=model_type, embed_method=embed_method)
        if data is not None:
            seed_data[seed] = data

    if not seed_data:
        raise RuntimeError(f"no valid panels for task {task!r}")

    run_seeds = [s for s in seeds if s in seed_data]
    n_rows = len(_COLOR_ROWS)
    n_cols = len(run_seeds)

    panel_w = 3.8 if n_cols <= 8 else (2.2 if n_cols <= 14 else 1.55)
    panel_h = 2.0 if n_cols <= 8 else 1.35
    head_fs, row_fs, row_labels, hspace = _seed_grid_label_sizes(n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_w * n_cols + 0.4, panel_h * n_rows + 0.5),
        squeeze=False,
        gridspec_kw={"hspace": hspace, "wspace": 0.10},
    )

    for row_idx, feat in enumerate(_COLOR_ROWS):
        for col_idx, seed in enumerate(run_seeds):
            ax = axes[row_idx, col_idx]
            data = seed_data[seed]
            ctx = data["panel_ctx"]
            _plot_2d_feature_colored_pca_panel(
                ax, data["pca_xy"], ctx["prefix_labels"], feat, ctx["timestep_labels"],
                data["automaton"],
                title="",
                xlabel="",
                ylabel="",
                xlim=data["xlim"],
                ylim=data["ylim"],
                annot_style=annot_style,
                show_legend=col_idx == n_cols - 1,
                minimal_axes=True,
                path_indices=data["path_indices"],
                faint_paths=True,
            )
            if col_idx < n_cols - 1:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
            if row_idx == 0:
                counts = format_cluster_counts(data["k_h"], data["k_pca"])
                head = f"s{seed}\n{counts}" if n_cols > 10 else f"s{seed}  {counts}"
                ax.set_title(head, fontsize=head_fs, fontweight="bold", pad=2)
            if col_idx == 0:
                ax.set_ylabel(
                    row_labels.get(feat, feat),
                    fontsize=row_fs,
                    fontweight="bold",
                )

    fig.suptitle(
        f"{task}: PCA hidden states by coloring × seed ({embed_method.upper()}, n={len(run_seeds)})",
        fontsize=12,
        y=0.98,
    )

    if outfile is None:
        out_dir = plots_dir(task, model_type) / "states"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"state_pca_seed_comparison_{embed_method}.png"
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
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--model-type", default="rnn", choices=["rnn", "rnn_dale"])
    parser.add_argument("--embed-method", default="pca", choices=["pca", "jpca"])
    parser.add_argument("--annot-style", default="annots_only", choices=["compact", "leaders", "annots_only"])
    parser.add_argument("--comparison-out", action="store_true")
    args = parser.parse_args()

    if not args.task and not args.preset:
        parser.error("provide --task or --preset")

    if args.preset:
        spec = COMPARISON_PRESETS[args.preset]
        tasks = list(spec.tasks)
        model_type = spec.model_type
        seeds = tuple(args.seeds) if args.seeds else common_seeds(spec.tasks, spec.model_type)
    else:
        tasks = [args.task]
        model_type = args.model_type
        seeds = tuple(args.seeds) if args.seeds else (42,)

    for task in tasks:
        outfile: Path | None = None
        if args.comparison_out and args.preset:
            spec = COMPARISON_PRESETS[args.preset]
            label = task.replace("_ns", "").replace("_", "-")
            name = f"state_pca_seeds_{label}_{args.embed_method}.png"
            outfile = comparison_dir(spec.name, "trajectories") / name
        print(plot_state_pca_seed_comparison(
            task,
            seeds=seeds,
            model_type=model_type,
            embed_method=args.embed_method,
            annot_style=args.annot_style,
            outfile=outfile,
        ))


if __name__ == "__main__":
    main()
