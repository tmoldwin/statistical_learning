"""Animated closed-loop trajectories (with noise) for the micro curriculum."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import EXPERIMENT_CONFIG, MICRO_CURRICULUM, input_path, model_path, spaced_experiment_name
from transformer.adapter import extract_transformer_activations, transformer_closed_loop_rollout
from visualize import (
    _plot_step_colored_path_arrows,
    _square_data_limits,
    fit_pca_2d_with_evr,
    load_model_for_viz,
)

DEFAULT_STEPS = 80
DEFAULT_FPS = 12


def _pca_basis(model_dict: dict, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acts = extract_transformer_activations(model_dict, text)
    _, mean, components, evr = fit_pca_2d_with_evr(acts.block_output)
    return mean, components, evr


def _write_gif(frame_paths: list[str], out_path: Path, *, fps: int) -> str:
    from PIL import Image

    frames = [Image.open(fp) for fp in frame_paths]
    duration_ms = max(1, int(1000 / max(fps, 1)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    for im in frames:
        im.close()
    return str(out_path)


def _write_rollout_animation(
    z: np.ndarray,
    generated: list[str],
    out_path: Path,
    *,
    title: str,
    fps: int = DEFAULT_FPS,
) -> str:
    if len(z) < 2:
        print(f"skip animation {out_path}: path too short")
        return ""

    xlim, ylim = _square_data_limits(z)
    trail = max(12, len(z) // 6)

    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp)
        for t in range(2, len(z) + 1):
            fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
            start = max(0, t - trail)
            path = z[start:t]
            _plot_step_colored_path_arrows(ax, path, linewidth=2.0, alpha=0.85, zorder=2)
            ax.scatter([z[t - 1, 0]], [z[t - 1, 1]], s=48, c="#e74c3c", zorder=4, edgecolors="white", linewidths=0.6)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.set_title(f"{title}\nstep {t}/{len(z)}  text: {''.join(generated[:t])[-24:]}", fontsize=9)
            frame_path = frame_dir / f"frame_{t - 2:04d}.png"
            fig.savefig(frame_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        frame_paths = sorted(str(p) for p in frame_dir.glob("frame_*.png"))
        gif_path = out_path.with_suffix(".gif")
        written = _write_gif(frame_paths, gif_path, fps=fps)
        print(f"wrote {written}")
        return written


def _write_rollout_static(
    z: np.ndarray,
    generated: list[str],
    out_path: Path,
    *,
    title: str,
) -> None:
    if len(z) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    _plot_step_colored_path_arrows(ax, z, linewidth=1.8, alpha=0.8, zorder=2)
    ax.scatter(z[0, 0], z[0, 1], s=40, c="#2ecc71", zorder=4, label="start")
    ax.scatter(z[-1, 0], z[-1, 1], s=40, c="#e74c3c", zorder=4, label="end")
    xlim, ylim = _square_data_limits(z)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{title}\n{len(z)} steps · {''.join(generated)[:40]}", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def animate_experiment(
    exp: str,
    *,
    steps: int = DEFAULT_STEPS,
    fps: int = DEFAULT_FPS,
    seed: str = " ",
) -> None:
    cfg = EXPERIMENT_CONFIG[exp]
    regime = cfg["regime"]
    text = input_path(exp).read_text(encoding="utf-8")[: cfg["viz_length"]]
    model = load_model_for_viz(str(model_path(exp, "transformer")), "transformer")
    noise_std = float(model.get("timestep_noise_std", 0.0))

    mean, components, evr = _pca_basis(model, text)
    rng = np.random.default_rng(42)
    hidden, generated = transformer_closed_loop_rollout(
        model, seed_text=seed, steps=steps, rng=rng,
    )
    z = (hidden - mean) @ components.T
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    title = f"{regime} · closed-loop (σ={noise_std:g}) · PC1 {pc1:.0f}% PC2 {pc2:.0f}%"

    out_dir = REPO_ROOT / "experiments" / exp / "transformer" / "plots" / "dynamics"
    _write_rollout_static(z, generated, out_dir / "closed_loop_trajectory.png", title=title)
    _write_rollout_animation(z, generated, out_dir / "closed_loop_trajectory.mp4", title=title, fps=fps)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("experiments", nargs="*", help="optional subset, e.g. two_word_disjoint")
    args = parser.parse_args()

    exps = [spaced_experiment_name(r) for r in MICRO_CURRICULUM]
    if args.experiments:
        exps = [e for e in exps if any(o.replace("_s", "") in e for o in args.experiments)]

    print(f"Closed-loop animations: {args.steps} steps, {args.fps} fps, noise from model config")
    for exp in exps:
        if not model_path(exp, "transformer").is_file():
            print(f"skip {exp}: no checkpoint")
            continue
        print(f"\n=== {exp} ===")
        animate_experiment(exp, steps=args.steps, fps=args.fps)

    print("\nOutputs: experiments/<exp>/transformer/plots/dynamics/closed_loop_trajectory.gif")


if __name__ == "__main__":
    main()
