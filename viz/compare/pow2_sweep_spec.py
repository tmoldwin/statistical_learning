"""Pow2 sweep axis definitions shared by training and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import vocab_sweep_pow2 as _pow2
import vocab_sweep_pow2_h100 as _h100


@dataclass(frozen=True)
class Pow2SweepSpec:
    comparison_name: str
    word_counts: tuple[int, ...]
    lengths: tuple[int | str, ...]
    default_seeds: tuple[int, ...]
    seed_comparison_seeds: tuple[int, ...]
    task_name: Callable[[int, int | str], str]
    iter_cells: Callable[[], Iterable[tuple[int, int | str]]]
    build_vocab: Callable[[int, int | str], list[str]]
    length_label: Callable[[int | str], str]


POW2_SWEEP_SPEC_NS = Pow2SweepSpec(
    comparison_name="word_count_pow2_sweep_ns",
    word_counts=_pow2.POW2_WORD_COUNTS,
    lengths=_pow2.POW2_LENGTHS,
    default_seeds=_pow2.POW2_DEFAULT_SEEDS,
    seed_comparison_seeds=_pow2.POW2_SEED_COMPARISON_SEEDS,
    task_name=_pow2.task_name,
    iter_cells=_pow2.iter_pow2_sweep_cells,
    build_vocab=_pow2.build_vocab,
    length_label=_pow2.length_label,
)

POW2_SWEEP_SPEC_H100 = Pow2SweepSpec(
    comparison_name=_h100.POW2_H100_COMPARISON,
    word_counts=_h100.POW2_H100_WORD_COUNTS,
    lengths=_h100.POW2_H100_LENGTHS,
    default_seeds=_h100.POW2_H100_DEFAULT_SEEDS,
    seed_comparison_seeds=_h100.POW2_H100_SEED_COMPARISON_SEEDS,
    task_name=_h100.task_name,
    iter_cells=_h100.iter_pow2_h100_sweep_cells,
    build_vocab=_h100.build_vocab,
    length_label=_h100.length_label,
)
