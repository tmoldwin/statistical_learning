"""Render trie + min-DFA for each micro-curriculum regime and print branching summary."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import MICRO_CURRICULUM, spaced_experiment_name
from task import REGIMES
from vocab_diagrams import (
    build_minimized_vocabulary_automaton,
    build_trie,
    minimize_dfa,
    trie_to_dfa,
    write_vocabulary_diagrams,
)


def _viable_words(prefix: str, words: list[str]) -> set[str]:
    if prefix in ("", "ε"):
        return set(words)
    return {w for w in words if w.startswith(prefix)}


def _ambiguous_prefixes(words: list[str]) -> list[tuple[str, int]]:
    """Prefixes where more than one vocabulary word remains viable."""
    automaton = build_minimized_vocabulary_automaton(words)
    ambiguous: list[tuple[str, int]] = []
    seen: set[str] = set()
    for prefixes in automaton.state_prefixes.values():
        viable: set[str] = set()
        for prefix in prefixes:
            viable |= _viable_words(prefix, words)
        if len(viable) > 1:
            label = min(prefixes, key=lambda s: (len(s), s)) if prefixes else "?"
            if label not in seen:
                seen.add(label)
                ambiguous.append((label, len(viable)))
    return sorted(ambiguous, key=lambda x: (len(x[0]), x[0]))


def main() -> None:
    rows: list[str] = []

    for regime in MICRO_CURRICULUM:
        words = REGIMES[regime]
        exp_name = spaced_experiment_name(regime)
        regime_dir = REPO_ROOT / "experiments" / exp_name / "shared"
        write_vocabulary_diagrams(words, regime_dir)

        root = build_trie(words)
        dfa, _old_to_new = minimize_dfa(trie_to_dfa(root))
        n_states = dfa._n
        ambiguous = _ambiguous_prefixes(words)
        amb_str = ", ".join(f"{p}({n})" for p, n in ambiguous) if ambiguous else "none"
        rows.append(
            f"{regime:28}  words={words!r:40}  dfa_states={n_states}  ambiguous={amb_str}"
        )

    print("Micro curriculum DFA validation\n" + "-" * 100)
    for row in rows:
        print(row)
    print(f"\nWrote trie/DFA SVGs under experiments/<regime>/shared/ for each regime.")


if __name__ == "__main__":
    main()
