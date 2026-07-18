"""Compute pooled unit (or PC) selectivity across the mixed-vocab DFA sweep.

Outputs:
- Histogram of SI conditioned on DFA difficulty bins (10, 20, 30, ...).
- Regression plot: mean SI vs minimized DFA state count.

Use ``--pcs`` for the same analysis on principal-component scores instead of
individual hidden units (PCA fit on condensed prefixes).
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viz.compare.mixed_dfa_unit_selectivity import (
    run_mixed_dfa_pc_selectivity_analysis,
    run_mixed_dfa_unit_selectivity_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="force SI recomputation")
    parser.add_argument("--max-tasks", type=int, default=None, help="debug: limit number of runs")
    parser.add_argument(
        "--pcs",
        action="store_true",
        help="analyze principal components instead of individual neurons",
    )
    args = parser.parse_args()

    if args.pcs:
        run_mixed_dfa_pc_selectivity_analysis(
            recompute=args.recompute,
            max_tasks=args.max_tasks,
        )
    else:
        run_mixed_dfa_unit_selectivity_analysis(
            recompute=args.recompute,
            max_tasks=args.max_tasks,
        )


if __name__ == "__main__":
    main()

