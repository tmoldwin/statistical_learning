"""Compute pooled unit selectivity across the mixed-vocab DFA sweep.

Outputs:
- Histogram of SI conditioned on DFA difficulty bins (10, 20, 30, ...).
- Regression plot: mean SI vs minimized DFA state count.
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viz.compare.mixed_dfa_unit_selectivity import run_mixed_dfa_unit_selectivity_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="force SI recomputation")
    parser.add_argument("--max-tasks", type=int, default=None, help="debug: limit number of runs")
    args = parser.parse_args()

    run_mixed_dfa_unit_selectivity_analysis(
        recompute=args.recompute,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()

