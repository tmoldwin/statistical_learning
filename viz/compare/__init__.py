"""Cross-task comparison figure specs and runners."""

from viz.compare.run import COMPARISON_KINDS, run_comparison
from viz.compare.spec import COMPARISON_PRESETS, ComparisonSpec

__all__ = [
    "COMPARISON_KINDS",
    "COMPARISON_PRESETS",
    "ComparisonSpec",
    "run_comparison",
]
