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
    "mat", "slate", "gate", "ate", "hat", "plate", "cat", "late",
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


def _draw_colored_letter_stream(
    ax,
    stream_words: Sequence[str],
    words: Sequence[str],
    *,
    y_rows: Sequence[float],
    fontsize: float,
    margin: float = 0.03,
    tight: bool = False,
) -> None:
    """Draw an unsegmented stream as colored letters (no per-character boxes).

    ``tight`` packs letters at monospace width on a single line instead of
    stretching them to fill the axes.
    """
    palette = list(WORD_STYLE.values())
    # Same index→color mapping as the word chips (do not look up WORD_STYLE by
    # name: "hat"/"late" would steal demo-vocab colors and vanish from the stream).
    color_of = {w: palette[i % len(palette)][0] for i, w in enumerate(words)}
    stream = "".join(stream_words)
    n = len(stream)
    n_rows = max(1, len(y_rows))
    if tight:
        fig = ax.figure
        fig.canvas.draw()
        width_px = max(ax.get_window_extent().width, 1.0)
        cell_w = (fontsize * 0.60 * fig.dpi / 72.0) / width_px
        max_n = max(1, int((1.0 - 2 * margin) / cell_w))
        if n > max_n:
            kept: list[str] = []
            used = 0
            for word in stream_words:
                if used + len(word) > max_n:
                    break
                kept.append(word)
                used += len(word)
            stream_words = kept
            n = used
        n_rows = 1
        cols = max(n, 1)
    else:
        cols = max(1, (n + n_rows - 1) // n_rows)
        cell_w = (1.0 - 2 * margin) / cols
    char_i = 0
    for word in stream_words:
        color = color_of[word]
        for ch in word:
            row_i = min(char_i // cols, n_rows - 1)
            col_i = char_i % cols
            ax.text(
                margin + (col_i + 0.5) * cell_w,
                y_rows[min(row_i, len(y_rows) - 1)],
                ch,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                fontfamily="monospace",
                fontweight="700",
                transform=ax.transAxes,
            )
            char_i += 1


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
    stream_chars: int = 80,
) -> Path:
    """Two ≤3-word 3/4-letter vocabularies: chips, one-line stream, horizontal DFA."""
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

    fig = plt.figure(figsize=(12.2, 7.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.12)
    palette = list(WORD_STYLE.values())
    stream_patterns = (
        (0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 0, 2),
        (0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 2, 1),
    )

    for row, (words, aut) in enumerate(examples):
        inner = outer[row].subgridspec(3, 1, height_ratios=[0.30, 0.18, 2.55], hspace=0.04)

        ax_vocab = fig.add_subplot(inner[0])
        ax_vocab.set_axis_off()
        ax_vocab.set_xlim(0, 1)
        ax_vocab.set_ylim(0, 1)
        letter = "A" if row == 0 else "B"
        ax_vocab.text(
            0.0, 0.72,
            f"Vocabulary {letter}  ·  DFA={int(aut.dfa._n)}",
            fontsize=12, fontweight="bold", va="center",
            transform=ax_vocab.transAxes,
        )
        # Plain bold colored words under the heading (no chip boxes).
        x = 0.0
        for i, word in enumerate(words):
            stroke, _fill = palette[i % len(palette)]
            if i > 0:
                ax_vocab.text(
                    x, 0.22, " · ",
                    ha="left", va="center", fontsize=13, color="0.45",
                    transform=ax_vocab.transAxes,
                )
                x += 0.035
            t = ax_vocab.text(
                x, 0.22, word,
                ha="left", va="center", fontsize=14,
                color=stroke, fontfamily="monospace", fontweight="800",
                transform=ax_vocab.transAxes,
            )
            # Advance x roughly by rendered width in axes fraction.
            x += 0.018 * max(len(word), 3) + 0.01
            del t

        pattern = stream_patterns[row % len(stream_patterns)]
        stream_words: list[str] = []
        chars = 0
        pi = 0
        while chars < stream_chars:
            w = words[pattern[pi % len(pattern)] % len(words)]
            stream_words.append(w)
            chars += len(w)
            pi += 1

        ax_seq = fig.add_subplot(inner[1])
        ax_seq.set_axis_off()
        ax_seq.set_xlim(0, 1)
        ax_seq.set_ylim(0, 1)
        _draw_colored_letter_stream(
            ax_seq, stream_words, words,
            y_rows=(0.50,),
            fontsize=15.0,
            margin=0.01,
            tight=True,
        )

        ax_dfa = fig.add_subplot(inner[2])
        draw_minimized_dfa_on_axes(
            ax_dfa, aut, words,
            compact=True,
            label_fontsize=13.0,
            node_scale=1.0,
            circle_scale=1.0,
            shortest_prefix_labels=True,
            fit_labels=True,
            horizontal=True,
            # Larger equal nodes; pack layers tighter so circles dominate over long edges.
            fixed_radius=82.0,
            arrow_mutation_scale=5.5,
            horizontal_stretch=0.78,
        )

    finalize_grid_figure(
        fig,
        suptitle="Same mixed-length task, different vocabulary structure",
        top=0.96, bottom=0.01, left=0.02, right=0.995,
        hspace=0.08, wspace=0.04,
    )
    save_figure(fig, save_path, dpi=180, pad_inches=0.04)
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
    stream_words = [w for w, _ in segments]
    _draw_colored_letter_stream(
        ax_stream, stream_words, list(demo_words),
        y_rows=(0.58, 0.32),
        fontsize=13.0,
        margin=0.02,
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
