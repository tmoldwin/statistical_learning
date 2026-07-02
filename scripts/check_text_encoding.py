"""Fail if tracked text sources contain UTF-16 null bytes or a UTF-8 BOM."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini",
    ".ps1", ".sh", ".gitignore", ".gitattributes", ".editorconfig",
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_SUFFIXES:
            continue
        yield path


def main() -> int:
    files = list(iter_text_files(REPO_ROOT))
    bad: list[str] = []
    for path in files:
        data = path.read_bytes()
        rel = path.relative_to(REPO_ROOT)
        if data.startswith(b"\xef\xbb\xbf"):
            bad.append(f"{rel}: UTF-8 BOM")
        if b"\x00" in data:
            bad.append(f"{rel}: null bytes (likely UTF-16 corruption)")
    if bad:
        print("encoding check FAILED:", file=sys.stderr)
        for line in bad:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"encoding check OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())