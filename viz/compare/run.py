"""Dispatch comparison figure kinds for a ComparisonSpec."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from viz.compare.learning_curves import plot_learning_curves
from viz.compare.spec import ComparisonSpec
from viz.compare.trajectories import plot_closed_loop_trajectories

ComparisonFn = Callable[..., Path | list[Path]]


def _plot_closed_loop_both(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
    **kwargs: object,
) -> list[Path]:
    return [
        plot_closed_loop_trajectories(spec, dimensions=2, seeds=seeds),
        plot_closed_loop_trajectories(spec, dimensions=3, seeds=seeds),
    ]


COMPARISON_KINDS: dict[str, ComparisonFn] = {
    "learning_curves": plot_learning_curves,
    "closed_loop_trajectories": _plot_closed_loop_both,
    "closed_loop_trajectories_2d": lambda spec, **kw: plot_closed_loop_trajectories(
        spec, dimensions=2, **kw,
    ),
    "closed_loop_trajectories_3d": lambda spec, **kw: plot_closed_loop_trajectories(
        spec, dimensions=3, **kw,
    ),
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
        elif fn is _plot_closed_loop_both:
            for path in fn(spec, seeds=run_seeds):
                outputs.append(path)
                print(f"wrote {path}")
        else:
            out = fn(spec, seeds=run_seeds)
            outputs.append(out)
            print(f"wrote {out}")
    return outputs
