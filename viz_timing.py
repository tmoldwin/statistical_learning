"""Lightweight timers for visualization profiling."""

from __future__ import annotations

import time
from contextlib import contextmanager


class VizTimer:
    """Accumulate named section durations and print a ranked summary."""

    def __init__(self) -> None:
        self._times: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self._times[name] = self._times.get(name, 0.0) + seconds

    @property
    def total(self) -> float:
        return sum(self._times.values())

    @property
    def wall_clock(self) -> float:
        return time.perf_counter() - self._started

    def merge(self, other: VizTimer) -> None:
        for name, seconds in other._times.items():
            self.add(name, seconds)

    def print_summary(self, *, title: str = "Visualization timing", top_n: int = 20) -> None:
        if not self._times:
            return
        ranked = sorted(self._times.items(), key=lambda item: item[1], reverse=True)
        total = sum(seconds for _, seconds in ranked)
        width = max(len(name) for name, _ in ranked)
        print(f"\n{title} ({total:.1f}s total)")
        for name, seconds in ranked[:top_n]:
            pct = 100.0 * seconds / total if total else 0.0
            print(f"  {seconds:6.2f}s ({pct:4.0f}%)  {name}")
        if len(ranked) > top_n:
            rest = sum(seconds for _, seconds in ranked[top_n:])
            print(f"  {rest:6.2f}s       ... {len(ranked) - top_n} more sections")
        untracked = self.wall_clock - total
        if untracked >= 0.05:
            print(f"  {untracked:6.2f}s       (untracked: imports, first matplotlib draw, etc.)")
        print(f"  {self.wall_clock:6.2f}s       wall clock")
