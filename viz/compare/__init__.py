"""Cross-task comparison figure specs and runners."""

from viz.compare.spec import COMPARISON_PRESETS, ComparisonSpec

__all__ = [
    "COMPARISON_KINDS",
    "COMPARISON_PRESETS",
    "ComparisonSpec",
    "run_comparison",
]


def __getattr__(name: str):
    if name in ("COMPARISON_KINDS", "run_comparison"):
        from viz.compare.run import COMPARISON_KINDS, run_comparison
        return COMPARISON_KINDS if name == "COMPARISON_KINDS" else run_comparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
