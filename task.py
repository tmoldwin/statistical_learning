"""
Statistical-learning task generator.

Task: sample a word uniformly from a small vocabulary, emit its characters,
repeat. Writes the resulting character stream to `input.txt` so it can be
consumed by `min-char-rnn.py` unchanged.

From: https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning

Usage:
    python task.py ten_word_overlap --chars 50000
"""

from __future__ import annotations

import argparse
import random

from experiment import (
    EXPERIMENT_CONFIG,
    experiment_regime,
    experiment_uses_word_space,
    input_path as experiment_input_path,
    spaced_experiment_name,
)

REGIMES: dict[str, list[str]] = {
    # Micro curriculum (2–4 words): isolate prefix vs DFA axes.
    "two_word_disjoint": ["cat", "mop"],
    "two_word_pos_overlap": ["cub", "nut"],
    "two_word_prefix_branch": ["can", "cat"],
    "three_word_permutation": ["ate", "eat", "tea"],
    "three_word_ca_hub": ["can", "cat", "cab"],
    # Minimal sandbox: three words sharing -at (for 2D transformer/RNN prototyping).
    "three_word_overlap": ["cat", "hat", "mat"],
    # Four-word demo (legacy): two vowels (a, e), overlapping structure.
    "four_word_overlap": ["cat", "met", "ate", "tea"],
    # Five-word paper demo: same a/e overlap family, plus *eat* (ties *ate*/*tea*).
    "five_word_overlap": ["cat", "met", "ate", "tea", "eat"],
    # 10 words, length 3; overlap on -at/-et/-ea; vowels a, e, i.
    "ten_word_overlap": [
        "cat", "hat", "mat", "rat",
        "met", "pet", "net",
        "ate", "eat", "tea",
    ],
    # 10 real 4-letter words with overlapping prefixes/suffixes.
    "ten_four_letter_overlap": [
        # same suffix -ake
        "bake", "cake", "lake", "make",
        # same suffix -ank
        "bank", "tank",
        # same prefix can-
        "cane", "cant",
        # same suffix -ate
        "late", "mate",
    ],
    # 6 words: one -at suffix family + one co- prefix family.
    "six_word_overlap": [
        "cat", "hat", "mat",
        "con", "cob", "cot",
    ],
    # 6 words: -ake suffix + -ank suffix (4-letter).
    "six_word_four_letter": [
        "bake", "cake", "lake",
        "bank", "tank", "rank",
    ],
    # 6 words: -ight suffix + -ound suffix (5-letter).
    "six_word_five_letter": [
        "light", "night", "right",
        "bound", "found", "hound",
    ],
    # 8 real 4-letter words; two suffix families (-ake, -ank).
    "eight_word_four_letter": [
        "bake", "cake", "lake", "rake",
        "bank", "tank", "rank", "sank",
    ],
    # 6 words: one -at suffix family + one si- prefix family.
    "six_word_overlap_sin": [
        "sin", "six", "sir",
        "cat", "hat", "mat",
    ],
    # 12+ real words: 3-letter base set plus overlapping 4–5 letter words.
    "twelve_word_overlap": [
        "ban",
        "rot",
        "cat", "hat", "mat",  # suffix -at
        "con", "cob", "cot",  # prefix co-
        "son",
        "din",
        "fun",
        "bun",
        # 4-letter (same character inventory)
        "that", "math", "band", "fund", "bund",
        # 5-letter
        "front", "count", "storm",
    ],
    # 16 words, lengths 3–7 (3–4 words per length); overlapping families.
    "sixteen_word_mixed": [
        "cat", "hat", "bat", "mat",
        "bake", "lake", "rank", "gate",
        "light", "bound", "dream", "steam",
        "nation", "moment",
        "lightly", "station",
    ],
    # 16 real words (primary task vocabulary).
    "sixteen_word": [
        "big", "dig", "fig", "pig",
        "bog", "dog", "fog", "log",
        "bum", "gum", "hum", "rum",
        "red", "wed",
        "fox", "box",
    ],
    # 16 real 4-letter words; four suffix families (-ake, -ank, -ate, -ant).
    "sixteen_word_four_letter": [
        "bake", "cake", "lake", "rake",
        "bank", "tank", "rank", "sank",
        "late", "mate", "rate", "gate",
        "cant", "pant", "rant", "want",
    ],
    # 16 real 5-letter words; four suffix families (-ight, -ound, -atch, -ream).
    "sixteen_word_five_letter": [
        "light", "night", "right", "sight",
        "bound", "found", "hound", "pound",
        "batch", "catch", "hatch", "match",
        "cream", "dream", "gleam", "steam",
    ],
    # 16 real 6-letter words; four suffix families (-ation, -ought, -ster, -ment).
    "sixteen_word_six_letter": [
        "nation", "ration", "action", "motion",
        "bought", "fought", "sought", "taught",
        "master", "faster", "sister", "mister",
        "moment", "cement", "talent", "patent",
    ],
    # 16 real 7-letter words; four suffix families (-ingly, -ation, -ounded, -owing).
    "sixteen_word_seven_letter": [
        "lightly", "tightly", "nightly", "rightly",
        "caution", "section", "fiction", "mention",
        "bounded", "founded", "rounded", "wounded",
        "blowing", "flowing", "growing", "showing",
    ],
    # 50 real 3-letter words; ten suffix families (5 words each).
    "fifty_word": [
        "cat", "hat", "mat", "rat", "bat",
        "met", "pet", "net", "bet", "wet",
        "can", "ban", "pan", "man", "tan",
        "car", "bar", "tar", "far", "jar",
        "big", "dig", "fig", "pig", "wig",
        "bog", "dog", "fog", "log", "hog",
        "bum", "gum", "hum", "rum", "sum",
        "bun", "fun", "sun", "run", "gun",
        "cap", "map", "tap", "nap", "lap",
        "red", "wed", "bed", "fox", "box",
    ],
    # 50 real 4-letter words; ten suffix families (5 words each).
    "fifty_word_four_letter": [
        "bake", "cake", "lake", "rake", "sake",
        "bank", "tank", "rank", "sank", "yank",
        "late", "mate", "rate", "gate", "hate",
        "cant", "pant", "rant", "want", "aunt",
        "line", "mine", "pine", "wine", "fine",
        "rock", "lock", "mock", "sock", "dock",
        "feed", "need", "seed", "weed", "reed",
        "bore", "core", "fore", "more", "wore",
        "born", "corn", "horn", "torn", "worn",
        "mile", "bile", "file", "pile", "tile",
    ],
    # 50 real 5-letter words; ten suffix families (5 words each).
    "fifty_word_five_letter": [
        "light", "night", "right", "sight", "fight",
        "bound", "found", "hound", "pound", "round",
        "batch", "catch", "hatch", "match", "patch",
        "cream", "dream", "gleam", "steam", "scream",
        "paint", "faint", "saint", "taint", "quaint",
        "river", "liver", "diver", "giver", "quiver",
        "press", "dress", "cress", "tress", "stress",
        "sling", "cling", "fling", "swing", "sting",
        "bunch", "lunch", "munch", "punch", "crunch",
        "board", "hoard", "chord", "sword", "award",
    ],
    # 16 real words: suffix/prefix overlap families (a/e vowels).
    "sixteen_word_overlap": [
        "cat", "hat", "mat", "rat",
        "met", "pet", "net",
        "can", "ban", "pan",
        "car", "bar", "tar",
        "ant", "bed", "bet",
    ],
    # 16 real words: disjoint inventory (i/o/u vowels); sanity-check vocab B.
    "sixteen_word_ig_um": [
        "big", "dig", "fig", "pig",
        "bog", "dog", "fog", "log",
        "bum", "gum", "hum", "rum",
        "red", "wed",
        "fox", "box",
    ],
    # 25 real words: extends sixteen_word_overlap with more overlap families.
    "twenty_five_word_overlap": [
        "cat", "hat", "mat", "rat",
        "met", "pet", "net",
        "can", "ban", "pan",
        "car", "bar", "tar",
        "ant", "and",
        "bed", "bet",
        "fun", "bun", "sun",
        "map", "cap",
        "tea", "oil", "run",
    ],
    # 10 words, lengths 1–5; unspaced (concatenated) regime.
    "ten_word_mixed": [
        "a",
        "at", "be",
        "cat", "hat", "mat",
        "bake", "lake",
        "plant", "slant",
    ],
}

# Extra 4–5 letter words for plot labels only (not training vocabulary).
# Must appear in the generated corpus and segment cleanly when added to the base words.
LABEL_WORD_EXTENSIONS: dict[str, list[str]] = {
    "twelve_word_overlap": [
        "that", "math", "band", "fund", "bund", "front", "count", "storm",
    ],
    "ten_word_mixed": ["bake", "lake", "plant", "slant"],
}


def label_extensions_for_experiment(name: str) -> list[str]:
    return list(LABEL_WORD_EXTENSIONS.get(name, []))


def corpus_for_experiment(exp_name: str, *, seed: int) -> str:
    """Deterministic training corpus for an experiment folder and RNG seed."""
    from experiment import TASKS, experiment_regime

    cfg = TASKS[exp_name]
    words = REGIMES[experiment_regime(exp_name)]
    return generate_sequence(
        words,
        int(cfg["chars"]),
        seed=seed,
        word_space=bool(cfg.get("word_space", False)),
    )


def generate_sequence(
    words: list[str],
    num_chars: int,
    seed: int = 0,
    *,
    word_space: bool = False,
) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < num_chars:
        if word_space and out:
            out.append(" ")
            if len(out) >= num_chars:
                break
        for ch in rng.choice(words):
            if len(out) >= num_chars:
                break
            out.append(ch)
    return "".join(out[:num_chars])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regime", nargs="?", default="ten_word_overlap",
                        choices=list(REGIMES.keys()))
    parser.add_argument("--chars", type=int, default=50,
                        help="total characters to emit (default: 50)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp", default=None,
                        help="experiment name (default: regime); writes experiments/<exp>/input.txt")
    parser.add_argument("--out", default=None,
                        help="output path (overrides --exp)")
    parser.add_argument(
        "--word-space",
        action="store_true",
        help="insert a space between sampled words (experiment name gets _s suffix)",
    )
    args = parser.parse_args()

    exp_name = args.exp or args.regime
    if args.exp and args.exp in EXPERIMENT_CONFIG:
        word_space = bool(EXPERIMENT_CONFIG[args.exp].get("word_space", False))
        regime = experiment_regime(args.exp)
    else:
        regime = args.regime
        word_space = args.word_space or experiment_uses_word_space(exp_name)
        if word_space and not exp_name.endswith("_s"):
            exp_name = spaced_experiment_name(regime)

    out_path = args.out
    if out_path is None:
        out_path = str(experiment_input_path(exp_name))
        experiment_input_path(exp_name).parent.mkdir(parents=True, exist_ok=True)

    words = REGIMES[regime]
    text = generate_sequence(words, args.chars, seed=args.seed, word_space=word_space)
    with open(out_path, "w") as f:
        f.write(text)

    vocab = sorted(set(text))
    print(f"Regime:  {regime}" + (" (word-space)" if word_space else ""))
    print(f"Exp:     {exp_name}")
    print(f"Words:   {words}")
    print(f"Vocab:   {''.join(vocab)} ({len(vocab)} symbols)")
    print(f"Wrote:   {out_path} ({len(text):,} characters)")
    print(f"Preview: {text[:80]}")


from vocab_sweep import build_mixed_vocab, register_sweep_regimes
from vocab_sweep_pow2 import register_pow2_sweep_regimes
from vocab_sweep_pow2_h100 import register_pow2_h100_sweep_regimes
from vocab_mixed_dfa import register_mixed_dfa_regimes

register_sweep_regimes(REGIMES)
register_pow2_sweep_regimes(REGIMES)
register_pow2_h100_sweep_regimes(REGIMES)
register_mixed_dfa_regimes(REGIMES)
REGIMES["thirty_two_word_mixed_345"] = build_mixed_vocab(32, (3, 4, 5))
REGIMES["sixteen_word_mixed_345"] = build_mixed_vocab(16, (3, 4, 5))


if __name__ == "__main__":
    main()
