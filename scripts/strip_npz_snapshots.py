"""Strip weight-snapshot history from saved model .npz files.

Weight snapshots (weight_snap_*) dominate model file size (up to ~2.3 GB per
file for h500 models) and are only needed for learning-dynamics videos, which
can be regenerated on demand by retraining with --save-snapshots. Everything
else (final weights, loss curves, metrics, demo samples, config) is kept.

Rewrites files in place (via a temp file + atomic replace). Idempotent: files
without snapshot keys are left untouched. Files that fail to load (e.g. a
training run is mid-write) are skipped with a warning.

Usage:
    python scripts/strip_npz_snapshots.py            # strip everything under experiments/
    python scripts/strip_npz_snapshots.py --dry-run  # report only
    python scripts/strip_npz_snapshots.py path/to/dir_or_file.npz ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT_PREFIX = "weight_snap_"


def strip_file(path: Path, dry_run: bool = False) -> tuple[int, int] | None:
    """Return (bytes_before, bytes_after), or None if skipped/untouched."""
    size_before = path.stat().st_size
    try:
        with np.load(path, allow_pickle=False) as data:
            snap_keys = [k for k in data.files if k.startswith(SNAPSHOT_PREFIX)]
            if not snap_keys:
                return None
            kept = {k: data[k] for k in data.files if not k.startswith(SNAPSHOT_PREFIX)}
    except Exception as exc:
        print(f"SKIP (unreadable, maybe mid-write): {path} ({exc})")
        return None

    if dry_run:
        print(f"would strip {len(snap_keys)} snapshot arrays: {path} "
              f"({size_before / 1e6:,.1f} MB)")
        return (size_before, size_before)

    # Must end in .npz or np.savez appends the extension itself.
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp_path, **kept)
    os.replace(tmp_path, path)
    size_after = path.stat().st_size
    print(f"stripped: {path}  {size_before / 1e6:,.1f} MB -> {size_after / 1e6:,.1f} MB")
    return (size_before, size_after)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*",
        help="npz files or directories to process (default: experiments/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without rewriting")
    args = parser.parse_args()

    roots = [Path(p) for p in args.paths] if args.paths else [REPO_ROOT / "experiments"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(sorted(root.rglob("*.npz")))

    total_before = 0
    total_after = 0
    n_stripped = 0
    for path in files:
        result = strip_file(path, dry_run=args.dry_run)
        if result is not None:
            total_before += result[0]
            total_after += result[1]
            n_stripped += 1

    verb = "would strip" if args.dry_run else "stripped"
    print(f"\n{verb} {n_stripped}/{len(files)} files: "
          f"{total_before / 1e9:,.2f} GB -> {total_after / 1e9:,.2f} GB")


if __name__ == "__main__":
    main()
