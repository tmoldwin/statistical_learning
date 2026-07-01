"""Plot untrained vs trained closed-loop trajectories in a chosen PCA basis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import comparison_dir
from viz.compare._data import load_task_viz_context
from viz.compare.spec import COMPARISON_PRESETS
from viz.dimred import fit_pca_2d_with_evr
from visualize import (
    _closed_loop_summary_seed,
    _one_vocab_cycle_steps,
    _same_length_average_trajectory,
    _square_data_limits,
    _trajectory_seed_letters,
    rnn_closed_loop_rollout,
    run_forward_pass,
)


def _initial_model_like(final_model: dict, *, seed: int) -> dict:
    """Recreate min_char_rnn.py's non-Dale initial weights for a saved model."""
    rng = np.random.default_rng(seed)
    hidden_size = int(final_model["hidden_size"])
    vocab_size = int(final_model["vocab_size"])
    model = dict(final_model)
    model["weights_input_to_hidden"] = rng.standard_normal((hidden_size, vocab_size)) * 0.01
    model["weights_hidden_to_hidden"] = rng.standard_normal((hidden_size, hidden_size)) * 0.01
    model["weights_hidden_to_output"] = rng.standard_normal((vocab_size, hidden_size)) * 0.01
    model["bias_hidden"] = np.zeros((hidden_size, 1))
    model["bias_output"] = np.zeros((vocab_size, 1))
    return model


def _mean_closed_loop(model: dict, *, seed_text: str, steps: int, n_trials: int = 8) -> np.ndarray:
    trials: list[np.ndarray] = []
    for trial in range(n_trials):
        hidden, _ = rnn_closed_loop_rollout(
            model,
            seed_text=seed_text,
            steps=steps,
            rng=np.random.default_rng(trial),
        )
        if len(hidden) >= 2:
            trials.append(hidden)
    out = _same_length_average_trajectory(trials)
    if out is None:
        return np.empty((0, int(model["hidden_size"])))
    return out


def plot_init_vs_final(
    preset: str,
    *,
    seeds: tuple[int, ...],
    basis: str = "initial",
    outfile: str = "init_vs_final_closed_loop_2d.png",
) -> Path:
    spec = COMPARISON_PRESETS[preset]
    tasks = list(spec.tasks)
    n_panel_rows = len(tasks) * 2

    fig, axes = plt.subplots(
        n_panel_rows,
        len(seeds),
        figsize=(1.15 * len(seeds) + 0.5, 1.25 * n_panel_rows + 0.35),
        squeeze=False,
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )

    for row_idx, task in enumerate(tasks):
        for col_idx, seed in enumerate(seeds):
            ax_init = axes[row_idx * 2, col_idx]
            ax_final = axes[row_idx * 2 + 1, col_idx]
            if row_idx == 0:
                ax_init.set_title(f"s{seed}", fontsize=8, fontweight="bold", pad=3)
            if col_idx == 0:
                ax_init.set_ylabel(f"{spec.label_for(task)}\ninitial", fontsize=8, fontweight="bold")
                ax_final.set_ylabel("trained", fontsize=8, fontweight="bold")

            try:
                ctx = load_task_viz_context(task, model_type=spec.model_type, seed=seed)
            except FileNotFoundError:
                ax_init.axis("off")
                ax_final.axis("off")
                continue

            vocab_words = list(ctx.words)
            seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
            summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
            summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)

            init_model = _initial_model_like(ctx.model, seed=seed)
            init_hidden_states, _init_probs = run_forward_pass(init_model, ctx.text, spec.model_type)
            basis_states = init_hidden_states if basis == "initial" else ctx.hidden_states
            _, pca_mean, pca_components, evr = fit_pca_2d_with_evr(basis_states)
            final_loop = _mean_closed_loop(ctx.model, seed_text=summary_seed, steps=summary_steps)
            init_loop = _mean_closed_loop(
                init_model,
                seed_text=summary_seed,
                steps=summary_steps,
            )
            if len(final_loop) < 2 or len(init_loop) < 2:
                ax_init.axis("off")
                ax_final.axis("off")
                continue

            final_pc = (final_loop - pca_mean) @ pca_components.T
            init_pc = (init_loop - pca_mean) @ pca_components.T

            for ax, path, color in (
                (ax_init, init_pc, "#777777"),
                (ax_final, final_pc, "#2255aa"),
            ):
                ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.1, alpha=0.95)
                ax.scatter(path[0, 0], path[0, 1], s=9, color="#22aa22", zorder=3)
                ax.scatter(path[-1, 0], path[-1, 1], s=9, color="#cc3333", zorder=3)
                xlim, ylim = _square_data_limits(path, padding_frac=0.18)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=5, length=2)
                ax.grid(True, linestyle=":", alpha=0.25)

            init_span = float(np.linalg.norm(init_pc.max(axis=0) - init_pc.min(axis=0)))
            final_span = float(np.linalg.norm(final_pc.max(axis=0) - final_pc.min(axis=0)))
            span_ratio = init_span / final_span if final_span > 1e-12 else float("nan")
            ax_init.set_xlabel(f"init span {span_ratio:.2f}× final", fontsize=5)
            ax_final.set_xlabel(f"PC1 {evr[0]*100:.0f}% / PC2 {evr[1]*100:.0f}%", fontsize=5)

    fig.suptitle(
        f"{spec.display_title}: initial vs trained closed loops ({basis} PCA basis, separate zooms)",
        fontsize=10,
    )
    out_dir = comparison_dir(spec.name, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="sixteen_word_lengths_ns", choices=sorted(COMPARISON_PRESETS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--basis", default="initial", choices=["initial", "final"])
    parser.add_argument("--outfile", default=None)
    args = parser.parse_args()
    outfile = args.outfile or f"init_vs_final_closed_loop_2d_{args.basis}_pca.png"
    print(plot_init_vs_final(
        args.preset,
        seeds=tuple(args.seeds),
        basis=args.basis,
        outfile=outfile,
    ))


if __name__ == "__main__":
    main()
