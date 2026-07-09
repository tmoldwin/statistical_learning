"""Pow2 h100 sweep (L1-6, max 32 words): plan, train, plot."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pow2_sweep_driver import main_h100

if __name__ == "__main__":
    main_h100()