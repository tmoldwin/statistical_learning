"""Demo setup figures: minimal-DFA contrast, and vocabulary + colored stream."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from viz.plot_layout import finalize_grid_figure, save_figure

# Demo chip palette (-ate chain + -at family).
WORD_STYLE: dict[str, tuple[str, str]] = {
    "ate": ("#EE6677", "#fdeef0"),
    "late": ("#CCBB44", "#faf6e3"),
    "plate": ("#4477AA", "#eef4fb"),
    "slate": ("#228833", "#e8f5ea"),
    "gate": ("#F58518", "#fff4e8"),
    "cat": ("#9467bd", "#f3eef8"),
    "hat": ("#66CCEE", "#e8f7fb"),
    "mat": ("#AA3377", "#f8eaf2"),
}

DEMO_WORDS: tuple[str, ...] = (
    "ate", "late", "plate", "slate", "gate", "cat", "hat", "mat",
)
DEMO_STREAM_WORDS: tuple[str, ...] = (
    "plate", "late", "gate", "cat", "hat", "mat", "ate", "slate",
    "late", "mat", "ate", "hat", "plate", "gate", "slate", "cat",
)
DEMO_STREAM = "".join(DEMO_STREAM_WORDS)

# Small illustrative vocabs only — panels must stay readable (see dfa-diagram-readability).
# Titles list the words themselves; short labels are optional prefixes.
DFA_EXAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Shared ending", ("cat", "hat")),
    ("Shared start", ("cat", "cake")),
    ("Letter overlap", ("tea", "eat")),
)


def _segment_stream(stream: str, words: Sequence[str]) -> list[tuple[str, str]]:
    """Greedy left-to-right segmentation of ``stream`` into vocabulary words."""
    vocab = set(words)
    maxlen = max(len(w) for w in vocab)
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(stream):
        matched = None
        for L in range(min(maxlen, len(stream) - i), 0, -1):
            piece = stream[i : i + L]
            if piece in vocab:
                matched = piece
                break
        if matched is None:
            raise ValueError(f"cannot segment stream at {i}: …{stream[i : i + 12]!r}")
        out.append((matched, matched))
        i += len(matched)
    return out


def _word_colors(word: str) -> tuple[str, str]:
    if word in WORD_STYLE:
        return WORD_STYLE[word]
    # Stable fallback for non-demo DFA panel labels.
    palette = list(WORD_STYLE.values())
    return palette[hash(word) % len(palette)]


def plot_dfa_examples(
    save_path: str | Path,
    *,
    dfa_examples: Sequence[tuple[str, Sequence[str]]] = DFA_EXAMPLES,
) -> Path:
    """Three small-vocab minimal DFAs in one figure (no stream)."""
    from vocab_diagrams import (
        build_minimized_vocabulary_automaton,
        draw_minimized_dfa_on_axes,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    automata = []
    for title, words in dfa_examples:
        words_list = list(words)
        aut = build_minimized_vocabulary_automaton(words_list)
        automata.append((title, words_list, aut))

    fig = plt.figure(figsize=(11.2, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.18, 3.2], hspace=0.18, wspace=0.22)

    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.set_axis_off()
    ax_banner.text(
        0.0, 0.55,
        "Different vocabularies yield different minimal DFAs",
        fontsize=11, fontweight="bold", va="center", transform=ax_banner.transAxes,
    )
    ax_banner.text(
        0.0, 0.05,
        "Same next-character task; automata differ in size and sharing of prefixes",
        fontsize=8, color="0.35", va="center", transform=ax_banner.transAxes,
    )

    for col, (short, words, aut) in enumerate(automata):
        ax = fig.add_subplot(gs[1, col])
        n_states = int(aut.dfa._n)
        draw_minimized_dfa_on_axes(
            ax, aut, words,
            compact=True,
            label_fontsize=12.0,
            node_scale=1.0,
            shortest_prefix_labels=True,
            fit_labels=True,
        )
        word_list = ", ".join(words)
        ax.set_title(
            f"{word_list}\n{short} · DFA={n_states}",
            fontsize=8.5, pad=6,
        )

    finalize_grid_figure(
        fig,
        suptitle="Minimal DFAs for three small vocabularies",
        top=0.88, bottom=0.04, left=0.04, right=0.98,
        hspace=0.18, wspace=0.18,
    )
    save_figure(fig, save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_mixed_vocab_dfa_examples(
    save_path: str | Path,
    *,
    vocabularies: Sequence[Sequence[str]] = (
        ("bat", "bake", "bank"),   # shared ba · DFA=6
        ("hat", "hate", "late"),   # hat→hate + late · DFA=8
    ),
    stream_chars: int = 48,
) -> Path:
    """Two ≤3-word 3/4-letter vocabularies with long streams + prefix DFAs."""
    from vocab_diagrams import (
        build_minimized_vocabulary_automaton,
        draw_minimized_dfa_on_axes,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for words_in in vocabularies:
        words = list(words_in)
        if len(words) > 3:
            raise ValueError(f"max 3 words per vocabulary, got {words}")
        if any(len(w) not in (3, 4) for w in words):
            raise ValueError(f"only 3–4 letter words allowed, got {words}")
        aut = build_minimized_vocabulary_automaton(words)
        n_states = int(aut.dfa._n)
        if n_states > 10:
            raise ValueError(f"DFA has {n_states} states (>10) for {words}")
        examples.append((words, aut))

    fig = plt.figure(figsize=(11.4, 6.8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[0.34, 0.66],
        hspace=0.16,
        wspace=0.01,
    )
    palette = list(WORD_STYLE.values())
    # Longer demos: cycle a fixed shuffle until we reach stream_chars.
    stream_patterns = (
        (0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 0, 2),
        (2, 0, 1, 2, 1, 0, 1, 2, 0, 2, 0, 1),
    )

    for row, (words, aut) in enumerate(examples):
        ax_vocab = fig.add_subplot(gs[row, 0])
        ax_vocab.set_axis_off()
        ax_vocab.set_xlim(0, 1)
        ax_vocab.set_ylim(0, 1)
        ax_vocab.text(
            0.03, 0.94,
            f"Vocabulary {'A' if row == 0 else 'B'} · {len(words)} words",
            fontsize=12, fontweight="bold", va="top",
        )
        ax_vocab.text(
            0.03, 0.86,
            f"lengths {', '.join(str(len(w)) for w in words)}",
            fontsize=9, color="0.35", va="top",
        )
        # Compact word chips in one row.
        chip_w = 0.22
        gap = 0.03
        x0 = 0.03
        for i, word in enumerate(words):
            stroke, fill = palette[i % len(palette)]
            x = x0 + i * (chip_w + gap)
            ax_vocab.add_patch(
                FancyBboxPatch(
                    (x, 0.74), chip_w, 0.07,
                    boxstyle="round,pad=0.008,rounding_size=0.018",
                    facecolor=fill, edgecolor=stroke, linewidth=1.1,
                    transform=ax_vocab.transAxes,
                )
            )
            ax_vocab.text(
                x + chip_w / 2, 0.775, word,
                ha="center", va="center", fontsize=9,
                color=stroke, fontfamily="monospace", fontweight="600",
                transform=ax_vocab.transAxes,
            )

        pattern = stream_patterns[row % len(stream_patterns)]
        stream_words: list[str] = []
        chars = 0
        pi = 0
        while chars < stream_chars:
            w = words[pattern[pi % len(pattern)] % len(words)]
            stream_words.append(w)
            chars += len(w)
            pi += 1
        stream = "".join(stream_words)
        ax_vocab.text(
            0.03, 0.64, "unsegmented training sequence",
            fontsize=9, fontweight="600", color="0.3",
        )
        # Wrap the long stream onto two rows.
        n = len(stream)
        cols = (n + 1) // 2
        margin = 0.03
        cell_w = 0.94 / cols
        char_i = 0
        for word in stream_words:
            stroke, fill = palette[words.index(word) % len(palette)]
            for ch in word:
                row_i = 0 if char_i < cols else 1
                col_i = char_i if char_i < cols else char_i - cols
                x = margin + col_i * cell_w
                y = 0.44 if row_i == 0 else 0.26
                ax_vocab.add_patch(
                    FancyBboxPatch(
                        (x, y), cell_w * 0.88, 0.11,
                        boxstyle="round,pad=0.002,rounding_size=0.008",
                        facecolor=fill, edgecolor=stroke, linewidth=0.65,
                        transform=ax_vocab.transAxes,
                    )
                )
                ax_vocab.text(
                    x + cell_w * 0.44, y + 0.055, ch,
                    ha="center", va="center", fontsize=7.5,
                    color=stroke, fontfamily="monospace", fontweight="600",
                    transform=ax_vocab.transAxes,
                )
                char_i += 1
        ax_vocab.text(
            0.03, 0.10, "no spaces; colors = hidden word boundaries",
            fontsize=7.5, color="0.5",
        )

        ax_dfa = fig.add_subplot(gs[row, 1])
        draw_minimized_dfa_on_axes(
            ax_dfa, aut, words,
            compact=True,
            label_fontsize=13.5,
            node_scale=1.35,
            circle_scale=1.25,
            shortest_prefix_labels=True,
            fit_labels=True,
            accept_marker="✓",
        )
        ax_dfa.set_title(
            f"minimal DFA · {int(aut.dfa._n)} states",
            fontsize=12, fontweight="bold", pad=2,
        )

    finalize_grid_figure(
        fig,
        suptitle="Same mixed-length task, different vocabulary structure",
        top=0.92, bottom=0.025, left=0.02, right=0.99,
        hspace=0.18, wspace=0.02,
    )
    save_figure(fig, save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_training_stream(
    save_path: str | Path,
    *,
    demo_words: Sequence[str] = DEMO_WORDS,
    stream: str = DEMO_STREAM,
) -> Path:
    """Demo vocabulary chips + colored unsegmented training stream."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    segments = _segment_stream(stream, demo_words)

    fig = plt.figure(figsize=(11.2, 3.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.35)

    # --- demo vocabulary chips ---
    ax_vocab = fig.add_subplot(gs[0, 0])
    ax_vocab.set_axis_off()
    ax_vocab.set_xlim(0, 1)
    ax_vocab.set_ylim(0, 1)
    ax_vocab.text(
        0.0, 0.92, "Demo vocabulary (this paper)",
        fontsize=9, fontweight="600", color="0.25", transform=ax_vocab.transAxes,
    )
    n = len(demo_words)
    chip_w = 0.12
    gap = 0.02
    x0 = 0.02
    y_chip = 0.28
    for i, word in enumerate(demo_words):
        stroke, fill = _word_colors(word)
        x = x0 + i * (chip_w + gap)
        ax_vocab.add_patch(
            FancyBboxPatch(
                (x, y_chip), chip_w, 0.42,
                boxstyle="round,pad=0.012,rounding_size=0.04",
                facecolor=fill, edgecolor=stroke, linewidth=1.4,
                transform=ax_vocab.transAxes, clip_on=False,
            )
        )
        ax_vocab.text(
            x + chip_w / 2, y_chip + 0.21, word,
            ha="center", va="center", fontsize=11, color=stroke, fontfamily="monospace",
            fontweight="600", transform=ax_vocab.transAxes,
        )
    ax_vocab.text(
        0.0, 0.02,
        "Lengths: 3 (cat, ate, tea) · 4 (cake, late) · 5 (plant)",
        fontsize=7.5, color="0.4", transform=ax_vocab.transAxes,
    )

    # --- colored unsegmented stream ---
    ax_stream = fig.add_subplot(gs[1, 0])
    ax_stream.set_axis_off()
    ax_stream.set_xlim(0, 1)
    ax_stream.set_ylim(0, 1)
    ax_stream.text(
        0.5, 0.92, "Training stream (no spaces)",
        ha="center", fontsize=9, fontweight="600", color="0.25",
        transform=ax_stream.transAxes,
    )
    n_chars = len(stream)
    margin = 0.02
    usable = 1.0 - 2 * margin
    cell = usable / n_chars
    y0 = 0.28
    h = 0.42
    for i, ch in enumerate(stream):
        pos = 0
        owner = demo_words[0]
        for w, _ in segments:
            if pos <= i < pos + len(w):
                owner = w
                break
            pos += len(w)
        stroke, fill = _word_colors(owner)
        x = margin + i * cell
        ax_stream.add_patch(
            FancyBboxPatch(
                (x, y0), cell * 0.92, h,
                boxstyle="round,pad=0.002,rounding_size=0.01",
                facecolor=fill, edgecolor=stroke, linewidth=0.9,
                transform=ax_stream.transAxes, clip_on=False,
            )
        )
        ax_stream.text(
            x + cell * 0.46, y0 + h / 2, ch,
            ha="center", va="center", fontsize=9, color=stroke,
            fontfamily="monospace", fontweight="600",
            transform=ax_stream.transAxes,
        )
    ax_stream.text(
        0.5, 0.06,
        "Unsegmented character stream · word boundaries invisible to the network "
        "(colors shown for illustration only)",
        ha="center", fontsize=7.5, color="0.4", transform=ax_stream.transAxes,
    )

    finalize_grid_figure(
        fig,
        suptitle="Demo vocabulary and the training stream",
        top=0.88, bottom=0.06, left=0.04, right=0.98,
        hspace=0.35, wspace=0.18,
    )
    save_figure(fig, save_path, dpi=160)
    plt.close(fig)
    return save_path


# Back-compat alias used by older call sites / notebooks.
def plot_corpus_stream_overview(
    save_path: str | Path,
    *,
    demo_words: Sequence[str] = DEMO_WORDS,
    stream: str = DEMO_STREAM,
    dfa_examples: Sequence[tuple[str, Sequence[str]]] = DFA_EXAMPLES,
) -> Path:
    """Deprecated combined figure; prefer ``plot_dfa_examples`` + ``plot_training_stream``."""
    _ = dfa_examples
    return plot_training_stream(save_path, demo_words=demo_words, stream=stream)
