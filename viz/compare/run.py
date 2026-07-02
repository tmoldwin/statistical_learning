"""Dispatch comparison figure kinds for a ComparisonSpec."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from viz.compare.learning_curves import plot_learning_curves
from viz.compare.geometry import write_trajectory_geometry
from viz.compare.shape_quantification import plot_shape_quantification
from viz.compare.spec import ComparisonSpec
from viz.compare.trajectories import plot_closed_loop_trajectories
from viz.dimred import EMBED_METHODS

ComparisonFn = Callable[..., Path | list[Path]]


def _plot_closed_loop_both(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
    **kwargs: object,
) -> list[Path]:
    paths: list[Path] = []
    for method in EMBED_METHODS:
        paths.append(plot_closed_loop_trajectories(spec, dimensions=2, seeds=seeds, embed_method=method))
        paths.append(plot_closed_loop_trajectories(spec, dimensions=3, seeds=seeds, embed_method=method))
    return paths


def _plot_closed_loop_2d_all(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
    **kwargs: object,
) -> list[Path]:
    return [
        plot_closed_loop_trajectories(spec, dimensions=2, seeds=seeds, embed_method=method)
        for method in EMBED_METHODS
    ]


def _plot_closed_loop_3d_all(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
    **kwargs: object,
) -> list[Path]:
    return [
        plot_closed_loop_trajectories(spec, dimensions=3, seeds=seeds, embed_method=method)
        for method in EMBED_METHODS
    ]


COMPARISON_KINDS: dict[str, ComparisonFn] = {
    "learning_curves": plot_learning_curves,
    "trajectory_geometry": write_trajectory_geometry,
    "shape_quantification": plot_shape_quantification,
    "closed_loop_trajectories": _plot_closed_loop_both,
    "closed_loop_trajectories_2d": _plot_closed_loop_2d_all,
    "closed_loop_trajectories_3d": _plot_closed_loop_3d_all,
}


def run_comparison(
    spec: ComparisonSpec,
    kinds: Iterable[str],
    *,
    seeds: tuple[int, ...] | None = None,
    truncate_to_plateau: bool = False,
) -> list[Path]:
    run_seeds = seeds if seeds is not None else spec.seeds
    outputs: list[Path] = []
    for kind in kinds:
        if kind not in COMPARISON_KINDS:
            raise ValueError(
                f"unknown comparison kind {kind!r}; choose from {sorted(COMPARISON_KINDS)}"
            )
        fn = COMPARISON_KINDS[kind]
        if fn is plot_learning_curves:
            out = fn(spec, truncate_to_plateau=truncate_to_plateau, seeds=run_seeds)
            outputs.append(out)
            print(f"wrote {out}")
        elif fn is write_trajectory_geometry:
            out = fn(spec, seeds=run_seeds)
            outputs.append(out)
            print(f"wrote {out}")
        elif fn is plot_shape_quantification:
            out = fn(spec, seeds=run_seeds)
            if out is not None:
                outputs.append(out)
        elif fn in (_plot_closed_loop_both, _plot_closed_loop_2d_all, _plot_closed_loop_3d_all):
            geom_path = write_trajectory_geometry(spec, seeds=run_seeds)
            outputs.append(geom_path)
            print(f"wrote {geom_path}")
            for path in fn(spec, seeds=run_seeds):
                outputs.append(path)
                print(f"wrote {path}")
        else:
            out = fn(spec, seeds=run_seeds)
            outputs.append(out)
            print(f"wrote {out}")
    return outputs
