"""Parametric suffix-family vocabularies for word-count × length sweeps."""

from __future__ import annotations

import math
from typing import Iterable

SWEEP_WORD_COUNTS: tuple[int, ...] = (4, 16, 50, 100)
SWEEP_LENGTHS: tuple[int, ...] = (3, 4, 5, 6, 7)
SWEEP_DEFAULT_SEEDS: tuple[int, ...] = tuple(range(1, 16))  # 15 seeds

# Single-char prefixes for short words; longer prefixes for longer suffixes.
_PREFIXES_1 = list("bcdfghjklmnpqrstvwxyz")
_PREFIXES_2 = [
    "ba", "be", "bi", "bo", "bu", "ca", "co", "cu", "da", "de", "di", "do", "du",
    "fa", "fe", "fi", "fo", "fu", "ga", "ge", "gi", "go", "gu", "ha", "he", "hi",
    "ho", "hu", "la", "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "mu", "na",
    "ne", "ni", "no", "nu", "pa", "pe", "pi", "po", "pu", "ra", "re", "ri", "ro",
    "ru", "sa", "se", "si", "so", "su", "ta", "te", "ti", "to", "tu", "wa", "we",
    "wi", "wo", "wr", "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl",
    "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw", "ch", "sh", "th",
]

_SUFFIXES: dict[int, list[str]] = {
    3: [
        "at", "et", "an", "ar", "ig", "og", "um", "ed", "it", "ot", "ab", "ad",
        "am", "ap", "ax", "eb", "en", "ep", "ib", "id", "in", "ip", "ob", "od",
        "op", "ub", "ud", "un", "up", "ut", "ag", "eg", "og", "ug", "ak", "ek",
        "ik", "ok", "uk", "al", "el", "il", "ol", "ul", "as", "es", "is", "os",
        "us", "aw", "ew", "ow",
    ],
    4: [
        "ake", "ank", "ate", "ant", "ine", "ock", "eed", "ore", "orn", "ile",
        "ine", "ame", "ade", "age", "ace", "ice", "ace", "ide", "ode", "ude",
        "all", "ell", "ill", "oll", "ull", "and", "end", "ind", "ond", "und",
        "art", "ert", "irt", "ort", "urt", "ast", "est", "ist", "ost", "ust",
        "ack", "eck", "ick", "ock", "uck", "ash", "esh", "ish", "osh", "ush",
    ],
    5: list(dict.fromkeys([
        "ight", "ound", "atch", "ream", "aint", "iver", "ress", "ling", "unch",
        "oard", "ight", "paint", "faint", "saint", "taint", "quaint", "river",
        "liver", "diver", "giver", "press", "dress", "cress", "tress", "stress",
        "sling", "cling", "fling", "swing", "sting", "bunch", "lunch", "munch",
        "punch", "crunch", "board", "hoard", "chord", "sword", "award", "light",
        "night", "right", "sight", "fight", "bound", "found", "hound", "pound",
        "round", "batch", "catch", "hatch", "match", "patch", "cream", "dream",
    ])),
    6: list(dict.fromkeys([
        "ation", "ought", "aster", "ement", "ment", "tion", "ally", "able",
        "ible", "ance", "ence", "less", "ness", "ward", "wise", "like",
        "some", "time", "work", "land", "hand", "head", "back", "side",
        "line", "room", "ship", "hood", "port", "view", "cast", "mark",
        "turn", "burn", "born", "corn", "horn", "torn", "worn", "fork",
        "pork", "walk", "talk", "silk", "milk", "risk", "disk", "task",
        "mask", "desk", "nest", "rest", "test", "west", "best", "fest",
    ])),
    7: list(dict.fromkeys([
        "ingly", "ation", "ction", "ently", "ously", "tedly", "fully", "ively",
        "ition", "ution", "ation", "ement", "antly", "ently", "ively", "ously",
        "ingly", "ation", "ction", "ently", "ously", "tedly", "fully", "ively",
        "ition", "ution", "ation", "ement", "antly", "ently", "ively", "ously",
        "ingly", "ation", "ction", "ently", "ously", "tedly", "fully", "ively",
        "ition", "ution", "ation", "ement", "antly", "ently", "ively", "ously",
    ])),
}


def regime_name(n_words: int, length: int) -> str:
    return f"sweep_w{n_words}_l{length}"


def task_name(n_words: int, length: int) -> str:
    return f"{regime_name(n_words, length)}_ns"


def _family_size(n_words: int) -> int:
    if n_words % 5 == 0:
        return 5
    if n_words % 4 == 0:
        return 4
    return 4


def _suffix_specs(length: int, *, min_families: int = 25) -> list[tuple[int, str]]:
    """(prefix_len, suffix) pairs that produce words of ``length``."""
    specs: list[tuple[int, str]] = []
    seen: set[str] = set()
    pool = _SUFFIXES.get(length, [])
    for suffix in pool:
        if suffix in seen:
            continue
        for plen in (1, 2, 3):
            if plen + len(suffix) == length:
                specs.append((plen, suffix))
                seen.add(suffix)
                break

    vowels = "aeiou"
    cons = "bcdfghjklmnpqrstvwxyz"
    idx = 0
    while len(specs) < min_families:
        for plen in (1, 2, 3):
            slen = length - plen
            if slen < 2:
                continue
            chars = []
            n = idx
            for _ in range(slen):
                chars.append(cons[n % len(cons)])
                n //= len(cons)
            chars[(idx + plen) % slen] = vowels[idx % len(vowels)]
            suffix = "".join(chars)
            if suffix not in seen:
                specs.append((plen, suffix))
                seen.add(suffix)
                idx += 1
                break
        else:
            idx += 1
        if idx > 5000:
            break
    return specs


def build_vocab(n_words: int, length: int) -> list[str]:
    """Build ``n_words`` words of fixed ``length`` using overlapping suffix families."""
    if n_words < 2:
        raise ValueError("need at least 2 words")
    group = _family_size(n_words)
    n_families = int(math.ceil(n_words / group))
    specs = _suffix_specs(length, min_families=n_families)
    if len(specs) < n_families:
        raise ValueError(f"only {len(specs)} suffix families for length {length}")
    used: set[str] = set()
    words: list[str] = []

    for fi in range(n_families):
        prefix_len, suffix = specs[fi]
        need = min(group, n_words - len(words))
        added = 0
        pools = _PREFIXES_1 if prefix_len == 1 else _PREFIXES_2
        for p in pools:
            if added >= need:
                break
            if len(p) != prefix_len:
                continue
            word = p + suffix
            if len(word) != length or word in used:
                continue
            used.add(word)
            words.append(word)
            added += 1
        if added < need:
            raise ValueError(
                f"could not build {need} words for suffix {suffix!r} "
                f"(prefix len {prefix_len}) at length {length}"
            )

    return words


def build_mixed_vocab(n_words: int, lengths: tuple[int, ...]) -> list[str]:
    """Build ``n_words`` words split evenly across fixed word lengths."""
    if n_words < 2:
        raise ValueError("need at least 2 words")
    if not lengths:
        raise ValueError("need at least one length")
    base, rem = divmod(n_words, len(lengths))
    counts = [base + (1 if i < rem else 0) for i in range(len(lengths))]
    words: list[str] = []
    for length, count in zip(lengths, counts):
        if count > 0:
            words.extend(build_vocab(count, length))
    return words


def sweep_task_config(n_words: int, length: int) -> dict[str, object]:
    """Training/viz defaults scaled to vocabulary size and word length."""
    hidden = 32 + n_words + 4 * length
    if n_words >= 16:
        hidden = max(64, hidden)
    if n_words >= 50:
        hidden = max(128, hidden)
    if n_words >= 100:
        hidden = max(192, hidden)
    hidden = min(int(hidden), 256)

    chars = max(30_000, n_words * length * 600)
    steps = min(120_000, max(15_000, n_words * 600 + length * 2500))
    if n_words >= 50:
        steps = max(steps, 80_000)
    if n_words >= 100:
        steps = max(steps, 100_000)

    viz_length = min(n_words * length + 20, 500)
    sequence_length = max(8, 2 * length + (8 if n_words >= 50 else 4))
    stall_min_iter = max(
        int(max(1_000, steps * 0.30)),
        max(1_000, int(steps * 0.08)),
    )
    stall_patience_evals = 24 if n_words < 50 else 36
    stall_min_delta = 0.001

    return {
        "regime": regime_name(n_words, length),
        "word_space": False,
        "chars": int(chars),
        "steps": int(steps),
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": max(1_000, int(steps * 0.08)),
        "viz_length": int(viz_length),
        "hidden_size": int(hidden),
        "sequence_length": int(sequence_length),
        "eval_interval": 50,
        "eval_iterations": 20,
        "metric_rollout_len": min(1000, max(300, n_words * length * 2)),
        "train_ratio": 0.9,
        "dropout": 0.25,
        "l2_lambda": 1e-4,
        "learning_rate": 0.04 if length >= 5 or n_words >= 50 else 0.1,
        "stall_patience_evals": int(stall_patience_evals),
        "stall_min_delta": float(stall_min_delta),
        "stall_min_iter": int(stall_min_iter),
        "sweep_n_words": int(n_words),
        "sweep_length": int(length),
    }


def register_sweep_regimes(regimes: dict[str, list[str]]) -> None:
    for n_words in SWEEP_WORD_COUNTS:
        for length in SWEEP_LENGTHS:
            regimes[regime_name(n_words, length)] = build_vocab(n_words, length)


def register_sweep_tasks(tasks: dict[str, dict]) -> None:
    for n_words in SWEEP_WORD_COUNTS:
        for length in SWEEP_LENGTHS:
            tasks[task_name(n_words, length)] = sweep_task_config(n_words, length)


def iter_sweep_cells() -> Iterable[tuple[int, int]]:
    for n_words in SWEEP_WORD_COUNTS:
        for length in SWEEP_LENGTHS:
            yield n_words, length


def parse_sweep_task(task: str) -> tuple[int, int] | None:
    """Return (n_words, length) for ``sweep_w{N}_l{L}_ns`` tasks."""
    if not task.startswith("sweep_w") or not task.endswith("_ns"):
        return None
    core = task.removeprefix("sweep_w").removesuffix("_ns")
    if "_l" not in core:
        return None
    n_s, l_s = core.split("_l", 1)
    try:
        return int(n_s), int(l_s)
    except ValueError:
        return None
