"""
SVG diagrams for a finite word vocabulary: trie and minimal DFA.

    python vocab_diagrams.py shared_letters
    python vocab_diagrams.py --words cat hat map --out-dir plots
    python vocab_diagrams.py ten_word_overlap --exp ten_word_overlap
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

from experiment import plots_dir as experiment_plots_dir
from task import REGIMES


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------


@dataclass
class TrieNode:
    children: dict[str, TrieNode] = field(default_factory=dict)
    terminal: bool = False


def build_trie(words: list[str]) -> TrieNode:
    root = TrieNode()
    for word in words:
        node = root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.terminal = True
    return root


def trie_alphabet(root: TrieNode) -> list[str]:
    chars: set[str] = set()

    def walk(node: TrieNode) -> None:
        for ch, child in node.children.items():
            chars.add(ch)
            walk(child)

    walk(root)
    return sorted(chars)


def trie_states(root: TrieNode) -> tuple[list[TrieNode], dict[int, int]]:
    """BFS order; map id(node) -> state index (root = 0)."""
    order: list[TrieNode] = []
    index: dict[int, int] = {}
    queue = [root]
    while queue:
        node = queue.pop(0)
        index[id(node)] = len(order)
        order.append(node)
        for ch in sorted(node.children):
            queue.append(node.children[ch])
    return order, index


def trie_prefixes(root: TrieNode) -> dict[int, str]:
    """Map id(node) -> prefix string read from the root."""
    out: dict[int, str] = {}

    def walk(node: TrieNode, prefix: str) -> None:
        out[id(node)] = prefix
        for ch in sorted(node.children):
            walk(node.children[ch], prefix + ch)

    walk(root, "")
    return out


def display_prefix(prefix: str) -> str:
    return "ε" if prefix == "" else prefix


def format_prefix_set(prefixes: set[str]) -> str:
    shown = sorted({display_prefix(p) for p in prefixes}, key=lambda s: (len(s), s))
    return "{" + ", ".join(shown) + "}"


# ---------------------------------------------------------------------------
# DFA
# ---------------------------------------------------------------------------


@dataclass
class DFA:
    """States are integers 0..n-1."""

    alphabet: list[str]
    start: int
    delta: dict[tuple[int, str], int]
    finals: set[int]

    @property
    def states(self) -> list[int]:
        return list(range(self._n))

    def __post_init__(self) -> None:
        self._n = 1 + max(
            [self.start, *self.finals, *self.delta.values()],
            default=0,
        )

    def transition(self, state: int, symbol: str) -> int | None:
        return self.delta.get((state, symbol))


def trie_to_dfa(root: TrieNode) -> DFA:
    nodes, node_index = trie_states(root)
    delta: dict[tuple[int, str], int] = {}
    finals: set[int] = set()
    for node in nodes:
        s = node_index[id(node)]
        if node.terminal:
            finals.add(s)
        for ch, child in node.children.items():
            delta[(s, ch)] = node_index[id(child)]
    return DFA(alphabet=trie_alphabet(root), start=0, delta=delta, finals=finals)


def add_dead_state(dfa: DFA) -> DFA:
    """Complete the DFA with one rejecting sink (needed for minimization)."""
    dead = dfa._n
    delta = dict(dfa.delta)
    for s in dfa.states:
        for a in dfa.alphabet:
            delta.setdefault((s, a), dead)
    for a in dfa.alphabet:
        delta[(dead, a)] = dead
    return DFA(
        alphabet=dfa.alphabet,
        start=dfa.start,
        delta=delta,
        finals=dfa.finals,
    )


def minimize_dfa(dfa: DFA) -> DFA:
    """Partition refinement (table-filling) on a complete DFA."""
    dfa = add_dead_state(dfa)
    n = dfa._n
    alphabet = dfa.alphabet
    finals = dfa.finals

    def step(state: int, symbol: str) -> int:
        return dfa.delta[(state, symbol)]

    distinguish = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (i in finals) != (j in finals):
                distinguish[i][j] = distinguish[j][i] = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if distinguish[i][j]:
                    continue
                for a in alphabet:
                    ti, tj = step(i, a), step(j, a)
                    if distinguish[min(ti, tj)][max(ti, tj)]:
                        distinguish[i][j] = distinguish[j][i] = True
                        changed = True
                        break

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if not distinguish[i][j]:
                union(i, j)

    blocks: dict[int, list[int]] = {}
    for s in range(n):
        blocks.setdefault(find(s), []).append(s)

    # Drop the dead-state block if it only contains the sink and is non-final.
    rep_of: dict[int, int] = {}
    new_states: list[int] = []
    for states in sorted(blocks.values(), key=min):
        if states == [n - 1] and (n - 1) not in finals:
            continue
        rep = min(states)
        rep_of[rep] = len(new_states)
        new_states.append(rep)

    old_to_new: dict[int, int] = {}
    for states in blocks.values():
        if len(states) == 1 and states[0] == n - 1 and (n - 1) not in finals:
            continue
        rep = min(states)
        new_id = rep_of[rep]
        for s in states:
            old_to_new[s] = new_id

    new_start = old_to_new[dfa.start]
    new_finals = {old_to_new[s] for s in finals if s in old_to_new}
    new_delta: dict[tuple[int, str], int] = {}
    seen: set[tuple[int, str]] = set()
    for (s, a), t in dfa.delta.items():
        if s not in old_to_new or t not in old_to_new:
            continue
        ns, nt = old_to_new[s], old_to_new[t]
        key = (ns, a)
        if key not in seen:
            new_delta[key] = nt
            seen.add(key)

    return DFA(
        alphabet=dfa.alphabet,
        start=new_start,
        delta=new_delta,
        finals=new_finals,
    ), old_to_new


def minimized_prefix_sets(
    root: TrieNode, old_to_new: dict[int, int]
) -> dict[int, set[str]]:
    """For each minimized state, the set of trie prefixes that collapse into it."""
    nodes, node_index = trie_states(root)
    prefixes = trie_prefixes(root)
    out: dict[int, set[str]] = {}
    for node in nodes:
        old = node_index[id(node)]
        if old not in old_to_new:
            continue
        out.setdefault(old_to_new[old], set()).add(prefixes[id(node)])
    return out


@dataclass
class MinimizedVocabAutomaton:
    """Minimal DFA for a word vocabulary plus trie prefixes per minimized state."""

    dfa: DFA
    state_prefixes: dict[int, set[str]]


def build_minimized_vocabulary_automaton(words: list[str]) -> MinimizedVocabAutomaton:
    root = build_trie(words)
    dfa, old_to_new = minimize_dfa(trie_to_dfa(root))
    return MinimizedVocabAutomaton(
        dfa=dfa,
        state_prefixes=minimized_prefix_sets(root, old_to_new),
    )


def in_word_prefix_since_last_space(text: str, index: int) -> str:
    """Characters in the current word up to and including `index` (empty on a space)."""
    if index < 0 or index >= len(text) or text[index] == " ":
        return ""
    start = index
    while start > 0 and text[start - 1] != " ":
        start -= 1
    return text[start : index + 1]


def segment_corpus_by_words(text: str, vocab: set[str]) -> list[tuple[int, int, str]]:
    """Greedy longest-match word segmentation for concatenated (unspaced) corpora."""
    if not text:
        return []
    if not vocab:
        return [(0, len(text) - 1, text)]
    words_sorted = sorted(vocab, key=len, reverse=True)
    segs: list[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for w in words_sorted:
            end = i + len(w)
            if end <= n and text[i:end] == w:
                matched = w
                break
        if matched:
            segs.append((i, i + len(matched) - 1, matched))
            i += len(matched)
        else:
            j = i + 1
            while j < n and not any(
                text.startswith(w, j) for w in words_sorted
            ):
                j += 1
            segs.append((i, j - 1, text[i:j]))
            i = j
    return segs


def word_start_at_index(text: str, index: int, vocab: set[str]) -> int | None:
    """Start index of the vocabulary word (or in-progress prefix) containing `index`."""
    if not text or index < 0 or index >= len(text):
        return None
    segs = segment_corpus_by_words(text[: index + 1], vocab)
    if not segs:
        return None
    start, end, _ = segs[-1]
    return start if start <= index <= end else None


def in_word_prefix_at_position(
    text: str,
    index: int,
    *,
    spaced: bool,
    vocab: set[str] | None = None,
) -> str:
    """
    In-word prefix for plot labels and DFA coloring.

    Spaced corpora: prefix since last space (or ' ' on a space).
    Unspaced: implicit word boundaries via vocabulary segmentation; if that
    fails, use up to 3 characters ending at `index`.
    """
    if index < 0 or index >= len(text):
        return ""
    if spaced:
        if text[index] == " ":
            return " "
        return in_word_prefix_since_last_space(text, index)
    if vocab:
        start = word_start_at_index(text, index, vocab)
        if start is not None:
            return text[start : index + 1]
    return text[max(0, index - 2) : index + 1]


def walk_dfa_prefix(dfa: DFA, prefix: str) -> int | None:
    """DFA state after reading `prefix` from the start state (None if undefined)."""
    state = dfa.start
    for ch in prefix:
        nxt = dfa.transition(state, ch)
        if nxt is None:
            return None
        state = nxt
    return state


def dfa_state_for_prefix(
    prefix: str,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool = False,
) -> int:
    """Minimized DFA state after reading `prefix` (start state on space when spaced)."""
    if spaced and prefix == " ":
        return automaton.dfa.start
    state = walk_dfa_prefix(automaton.dfa, prefix)
    return automaton.dfa.start if state is None else state


def trie_prefix_display_order(words: list[str]) -> list[str]:
    """Trie prefixes in BFS order (root empty prefix omitted)."""
    root = build_trie(words)
    states, _ = trie_states(root)
    id_to_prefix = trie_prefixes(root)
    return [
        id_to_prefix[id(node)]
        for node in states
        if id_to_prefix[id(node)]
    ]


def dfa_state_at_position(
    text: str,
    index: int,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool,
    vocab: set[str] | None = None,
) -> int:
    """
    Minimized DFA state for this position.

    Uses the in-word prefix at this timestep (after last explicit space, or at
    implicit vocabulary word boundaries when unspaced).
    """
    if spaced and text[index] == " ":
        return automaton.dfa.start
    fragment = in_word_prefix_at_position(
        text, index, spaced=spaced, vocab=vocab,
    )
    if spaced and fragment == " ":
        return automaton.dfa.start
    state = walk_dfa_prefix(automaton.dfa, fragment)
    return automaton.dfa.start if state is None else state


def dfa_state_label(state: int, automaton: MinimizedVocabAutomaton) -> str:
    prefixes = automaton.state_prefixes.get(state, set())
    if prefixes:
        return format_prefix_set(prefixes)
    return f"q{state}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass
class NodeLayout:
    x: float
    y: float


def layout_trie(root: TrieNode) -> dict[int, NodeLayout]:
    nodes, index = trie_states(root)
    positions: dict[int, NodeLayout] = {}
    next_x = 0.0

    def assign(node: TrieNode, depth: int) -> float:
        nonlocal next_x
        nid = id(node)
        if not node.children:
            x = next_x
            next_x += 1.0
        else:
            xs = [assign(child, depth + 1) for child in node.children.values()]
            x = sum(xs) / len(xs)
        positions[nid] = NodeLayout(x=x, y=float(depth))
        return x

    assign(root, 0)
    return positions


def layout_dfa(dfa: DFA, layer_gap: float = 1.0) -> dict[int, NodeLayout]:
    """Layer states by BFS distance from start; spread within each layer."""
    layers: dict[int, list[int]] = {}
    depth = {dfa.start: 0}
    queue = [dfa.start]
    while queue:
        s = queue.pop(0)
        d = depth[s]
        layers.setdefault(d, []).append(s)
        for a in dfa.alphabet:
            t = dfa.transition(s, a)
            if t is not None and t not in depth:
                depth[t] = d + 1
                queue.append(t)

    for s in dfa.states:
        if s not in depth:
            depth[s] = max(layers) + 1 if layers else 0
            layers.setdefault(depth[s], []).append(s)

    positions: dict[int, NodeLayout] = {}
    for d, states in layers.items():
        states = sorted(set(states))
        n = len(states)
        for i, s in enumerate(states):
            x = (i - (n - 1) / 2.0) if n > 1 else 0.0
            positions[s] = NodeLayout(x=x, y=d * layer_gap)
    return positions


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

MIN_NODE_R = 24
X_GAP = 100
Y_GAP = 110
MARGIN = 50
LABEL_OFFSET = 16
FONT = "ui-monospace, Consolas, monospace"
CHAR_W = 6.8
LINE_H = 13
STATE_PAD = 10
BG_COLOR = "#f4f1ea"
NODE_FILL = "#ffffff"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_header(width: float, height: float, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        "<style>"
        f"text {{ font-family: {FONT}; font-size: 13px; fill: #111; }}"
        ".edge-label { font-size: 12px; fill: #222; }"
        ".state-label { font-size: 11px; fill: #1a1a1a; }"
        ".title { font-size: 15px; font-weight: 600; fill: #111; }"
        "</style>",
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
        f'fill="{BG_COLOR}"/>',
        f'<text x="{MARGIN}" y="28" class="title">{_esc(title)}</text>',
    ]


def _gap_scale(radii: dict[int, float]) -> float:
    if not radii:
        return 1.0
    return max(1.0, max(radii.values()) / MIN_NODE_R)


def _scale_positions(
    positions: dict[int, NodeLayout],
    gap_scale: float = 1.0,
    y_offset: float = 40,
) -> dict[int, tuple[float, float]]:
    if not positions:
        return {}
    xs = [p.x for p in positions.values()]
    ys = [p.y for p in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    x_gap = X_GAP * gap_scale
    y_gap = Y_GAP * gap_scale
    out: dict[int, tuple[float, float]] = {}
    for key, p in positions.items():
        nx = MARGIN + MIN_NODE_R + 40 + (p.x - min_x) / span_x * max(span_x * x_gap, 0)
        if span_x == 0:
            nx = MARGIN + 100
        ny = y_offset + MARGIN + (p.y - min_y) / span_y * max(span_y * y_gap, 0)
        if span_y == 0:
            ny = y_offset + MARGIN + 50
        out[key] = (nx, ny)
    return out


def _scale_trie_positions(
    positions: dict[int, NodeLayout],
    gap_scale: float = 1.0,
    y_offset: float = 40,
) -> dict[int, tuple[float, float]]:
    if not positions:
        return {}
    xs = [p.x for p in positions.values()]
    ys = [p.y for p in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_gap = X_GAP * gap_scale
    y_gap = Y_GAP * gap_scale
    width = max((max_x - min_x) * x_gap, x_gap)
    out: dict[int, tuple[float, float]] = {}
    for nid, p in positions.items():
        nx = MARGIN + MIN_NODE_R + 40 + (p.x - min_x) * x_gap
        if max_x == min_x:
            nx = MARGIN + width / 2 + MIN_NODE_R
        ny = y_offset + MARGIN + (p.y - min_y) * y_gap
        out[nid] = (nx, ny)
    return out


def _wrap_label(text: str, max_chars: int = 16) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    if not (text.startswith("{") and text.endswith("}")):
        return [text]
    parts = [p.strip() for p in text[1:-1].split(",")]
    lines: list[str] = []
    chunk: list[str] = []
    for part in parts:
        trial = "{" + ", ".join(chunk + [part]) + "}"
        if chunk and len(trial) > max_chars:
            lines.append("{" + ", ".join(chunk) + ",")
            chunk = [part]
        else:
            chunk.append(part)
    if chunk:
        lines.append("{" + ", ".join(chunk) + "}")
    return lines


def _state_label_lines(prefix_set: set[str]) -> list[str]:
    """One word per line when a state keeps many lexical hypotheses."""
    shown = sorted({display_prefix(p) for p in prefix_set}, key=lambda s: (len(s), s))
    if not shown:
        return ["ε"]
    if len(shown) == 1:
        return shown
    compact = format_prefix_set(prefix_set)
    if len(shown) == 2 and len(compact) <= 16:
        return [compact]
    return shown


def _fit_state(prefix_set: set[str]) -> tuple[list[str], float]:
    """Choose line breaks and the smallest radius that still fits the label."""
    lines = _state_label_lines(prefix_set)
    line_step = LINE_H * (0.82 if len(lines) >= 5 else 0.88 if len(lines) >= 3 else 1.0)
    w = max(len(line) for line in lines) * CHAR_W
    h = len(lines) * line_step
    r = max(MIN_NODE_R, w / 2 + STATE_PAD + 2, h / 2 + STATE_PAD + 2)
    return lines, r


def _compute_radii(state_labels: dict[int, set[str]]) -> dict[int, float]:
    return {key: _fit_state(prefixes)[1] for key, prefixes in state_labels.items()}


def _state_font_size(radius: float, n_lines: int = 1) -> int:
    if n_lines >= 6:
        return 6
    if n_lines >= 4:
        return 7
    if n_lines >= 3:
        return 8
    if radius >= 55:
        return 8
    if radius >= 40:
        return 9
    return 10


def _state_line_step(n_lines: int) -> float:
    if n_lines >= 5:
        return LINE_H * 0.82
    if n_lines >= 3:
        return LINE_H * 0.88
    return LINE_H


def _draw_state(
    lines: list[str],
    cx: float,
    cy: float,
    radius: float,
    accepting: bool,
    prefix_set: set[str],
) -> None:
    wrapped, _ = _fit_state(prefix_set)
    fs = _state_font_size(radius, len(wrapped))
    line_step = _state_line_step(len(wrapped))
    if accepting:
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius + 4:.1f}" '
            f'fill="none" stroke="#111" stroke-width="1.5"/>'
        )
    lines.append(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
        f'fill="{NODE_FILL}" stroke="#111" stroke-width="1.5"/>'
    )
    if len(wrapped) == 1:
        lines.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" class="state-label" '
            f'font-size="{fs}px">{_esc(wrapped[0])}</text>'
        )
    else:
        block_h = (len(wrapped) - 1) * line_step
        y0 = cy - block_h / 2
        inner = "".join(
            f'<tspan x="{cx:.1f}" dy="{0 if i == 0 else line_step}">{_esc(line)}</tspan>'
            for i, line in enumerate(wrapped)
        )
        lines.append(
            f'<text x="{cx:.1f}" y="{y0:.1f}" text-anchor="middle" '
            f'class="state-label" font-size="{fs}px">{inner}</text>'
        )


def _edge_geometry(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r1: float,
    r2: float,
    curved: float = 0.0,
) -> tuple[str, float, float, float, float]:
    """Return (path_spec, label_x, label_y, normal_x, normal_y) after trimming ends."""
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    sx = x1 + ux * (r1 + 2)
    sy = y1 + uy * (r1 + 2)
    ex = x2 - ux * (r2 + 6)
    ey = y2 - uy * (r2 + 6)

    if abs(curved) > 1e-6:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        cpx, cpy = mx + px * curved, my + py * curved
        path = f"M {sx:.1f} {sy:.1f} Q {cpx:.1f} {cpy:.1f} {ex:.1f} {ey:.1f}"
        t = 0.5
        lx = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cpx + t**2 * ex
        ly = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cpy + t**2 * ey
        tx = 2 * (1 - t) * (cpx - sx) + 2 * t * (ex - cpx)
        ty = 2 * (1 - t) * (cpy - sy) + 2 * t * (ey - cpy)
        td = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / td, tx / td
    else:
        path = f"M {sx:.1f} {sy:.1f} L {ex:.1f} {ey:.1f}"
        lx, ly = (sx + ex) / 2, (sy + ey) / 2
        nx, ny = px, py
    return path, lx, ly, nx, ny


def _draw_edge(
    lines: list[str],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r1: float,
    r2: float,
    curved: float = 0.0,
) -> tuple[float, float, float, float] | None:
    path, lx, ly, nx, ny = _edge_geometry(x1, y1, x2, y2, r1, r2, curved)
    lines.append(
        f'<path d="{path}" fill="none" stroke="#444" stroke-width="1.2" '
        f'marker-end="url(#arrow)"/>'
    )
    return lx, ly, nx, ny


def _draw_edge_label(
    lines: list[str],
    lx: float,
    ly: float,
    nx: float,
    ny: float,
    label: str,
    side: float = 1.0,
) -> None:
    ox = lx + nx * LABEL_OFFSET * side
    oy = ly + ny * LABEL_OFFSET * side
    base = (
        f'<text x="{ox:.1f}" y="{oy:.1f}" text-anchor="middle" '
        f'dominant-baseline="central" class="edge-label"'
    )
    lines.append(
        f'{base} stroke="{BG_COLOR}" stroke-width="5" paint-order="stroke">'
        f"{_esc(label)}</text>"
    )
    lines.append(f"{base}>{_esc(label)}</text>")


def _marker_defs(lines: list[str]) -> None:
    lines.append("<defs>")
    lines.append(
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#444"/>'
        "</marker>"
    )
    lines.append("</defs>")


def _edge_curve_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r1: float,
    r2: float,
    curved: float = 0.0,
) -> tuple[
    tuple[float, float],
    tuple[float, float] | None,
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Start, optional quadratic control, end, label anchor, label normal (SVG coords)."""
    path, lx, ly, nx, ny = _edge_geometry(x1, y1, x2, y2, r1, r2, curved)
    tokens = path.replace("M", " ").replace("Q", " ").replace("L", " ").split()
    vals = [float(v) for v in tokens]
    if len(vals) == 4:
        return (vals[0], vals[1]), None, (vals[2], vals[3]), (lx, ly), (nx, ny)
    if len(vals) == 6:
        return (
            (vals[0], vals[1]),
            (vals[2], vals[3]),
            (vals[4], vals[5]),
            (lx, ly),
            (nx, ny),
        )
    raise ValueError(f"unexpected edge path: {path!r}")


def dfa_canvas_size(automaton: MinimizedVocabAutomaton) -> tuple[float, float]:
    """Width and height for a minimal-DFA diagram from label sizes and layout."""
    dfa = automaton.dfa
    state_labels = _dfa_state_label_map(automaton)
    radii = _compute_radii(state_labels)
    gap_scale = _gap_scale(radii)
    coords = _scale_positions(layout_dfa(dfa), gap_scale=gap_scale)
    return _canvas_size(coords, radii)


def _dfa_state_label_map(automaton: MinimizedVocabAutomaton) -> dict[int, set[str]]:
    dfa = automaton.dfa
    labels = {s: set(automaton.state_prefixes.get(s, set())) for s in automaton.state_prefixes}
    states = {dfa.start} | dfa.finals
    for (s, _), t in dfa.delta.items():
        states.add(s)
        states.add(t)
    for s in states:
        labels.setdefault(s, set())
    return labels


def draw_minimized_dfa_on_axes(
    ax,
    automaton: MinimizedVocabAutomaton,
    words: list[str],
    *,
    state_colors: dict[int, tuple] | None = None,
) -> None:
    """Matplotlib rendering of the same layout as `vocabulary_min_dfa.svg`."""
    from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
    from matplotlib.path import Path

    dfa = automaton.dfa
    state_labels = _dfa_state_label_map(automaton)
    radii = _compute_radii(state_labels)
    gap_scale = _gap_scale(radii)
    coords = _scale_positions(layout_dfa(dfa), gap_scale=gap_scale)
    width, height = _canvas_size(coords, radii)

    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(f"Minimal DFA · {', '.join(words)}", fontsize=12, pad=12)

    by_edge: dict[tuple[int, int], list[str]] = {}
    for (s, a), t in sorted(dfa.delta.items()):
        if s in coords and t in coords:
            by_edge.setdefault((s, t), []).append(a)

    for (s, t), labels in sorted(by_edge.items()):
        sx, sy = coords[s]
        tx, ty = coords[t]
        rs, rt = radii[s], radii[t]
        for i, label in enumerate(labels):
            curve = 26 * (i - (len(labels) - 1) / 2.0)
            start, control, end, (lx, ly), (nx, ny) = _edge_curve_points(
                sx, sy, tx, ty, rs, rt, curved=curve
            )
            if control is None:
                verts = [start, end]
                codes = [Path.MOVETO, Path.LINETO]
            else:
                verts = [start, control, end]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor="#444444",
                    linewidth=1.2,
                    zorder=1,
                )
            )
            ex, ey = end
            if control is None:
                sx0, sy0 = start
                dx, dy = ex - sx0, ey - sy0
            else:
                cx, cy = control
                dx, dy = ex - cx, ey - cy
            norm = math.hypot(dx, dy) or 1.0
            ax.add_patch(
                FancyArrowPatch(
                    (ex - dx / norm * 8, ey - dy / norm * 8),
                    (ex, ey),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color="#444444",
                    linewidth=1.2,
                    zorder=2,
                )
            )
            side = 1.0 if i % 2 == 0 else -1.0
            ox = lx + nx * LABEL_OFFSET * side
            oy = ly + ny * LABEL_OFFSET * side
            ax.text(
                ox, oy, label,
                fontsize=10, ha="center", va="center", color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG_COLOR, edgecolor="none"),
                zorder=3,
            )

    rx, ry = coords[dfa.start]
    rr = radii[dfa.start]
    ax.plot(
        [rx - rr - 36, rx - rr - 2], [ry, ry],
        color="#444444", linewidth=1.2, zorder=1,
    )
    ax.add_patch(
        FancyArrowPatch(
            (rx - rr - 10, ry),
            (rx - rr - 2, ry),
            arrowstyle="-|>",
            mutation_scale=10,
            color="#444444",
            linewidth=1.2,
            zorder=2,
        )
    )

    for s in sorted(coords):
        cx, cy = coords[s]
        r = radii[s]
        accepting = s in dfa.finals
        prefix_set = state_labels.get(s, set())
        if accepting:
            ax.add_patch(
                Circle(
                    (cx, cy), r + 4,
                    fill=False, edgecolor="#111111", linewidth=1.5, zorder=4,
                )
            )
        node_fill = state_colors.get(s, NODE_FILL) if state_colors else NODE_FILL
        ax.add_patch(
            Circle(
                (cx, cy), r,
                facecolor=node_fill, edgecolor="#111111", linewidth=1.5, zorder=5,
            )
        )
        wrapped, _ = _fit_state(prefix_set)
        fs = _state_font_size(r, len(wrapped))
        line_step = _state_line_step(len(wrapped))
        if len(wrapped) == 1:
            ax.text(
                cx, cy, wrapped[0],
                fontsize=fs, ha="center", va="center", color="#1a1a1a", zorder=6,
            )
        else:
            block_h = (len(wrapped) - 1) * line_step
            y0 = cy - block_h / 2
            for i, line in enumerate(wrapped):
                ax.text(
                    cx, y0 + i * line_step, line,
                    fontsize=fs, ha="center", va="center", color="#1a1a1a", zorder=6,
                )


def _canvas_size(
    coords: dict[int, tuple[float, float]],
    radii: dict[int, float],
) -> tuple[float, float]:
    if not coords:
        return 400.0, 300.0
    left = right = top = bottom = 0.0
    for key, (cx, cy) in coords.items():
        r = radii.get(key, MIN_NODE_R)
        left = min(left, cx - r) if left else cx - r
        right = max(right, cx + r)
        top = max(top, cy + r)
        bottom = min(bottom, cy - r) if bottom else cy - r
    width = max(right + MARGIN, 360)
    height = max(top + MARGIN + 30, 280)
    return width, height


def render_trie_svg(root: TrieNode, words: list[str]) -> str:
    nodes, _index = trie_states(root)
    prefixes = trie_prefixes(root)
    state_labels = {id(n): {prefixes[id(n)]} for n in nodes}
    radii = _compute_radii(state_labels)
    gap_scale = _gap_scale(radii)
    layout = layout_trie(root)
    coords = _scale_trie_positions(layout, gap_scale=gap_scale)
    width, height = _canvas_size(coords, radii)

    lines = _svg_header(width, height, f"Trie · {', '.join(words)}")
    _marker_defs(lines)

    edge_labels: list[tuple[float, float, float, float, str, float]] = []
    for node in nodes:
        nid = id(node)
        sx, sy = coords[nid]
        rs = radii[nid]
        children = sorted(node.children.items())
        for i, (ch, child) in enumerate(children):
            curve = 22 * (i - (len(children) - 1) / 2.0) if len(children) > 1 else 0.0
            cid = id(child)
            tx, ty = coords[cid]
            rt = radii[cid]
            geom = _draw_edge(lines, sx, sy, tx, ty, rs, rt, curved=curve)
            if geom:
                lx, ly, nx, ny = geom
                side = 1.0 if i % 2 == 0 else -1.0
                edge_labels.append((lx, ly, nx, ny, ch, side))

    rx, ry = coords[id(root)]
    rr = radii[id(root)]
    lines.append(
        f'<line x1="{rx - rr - 36:.1f}" y1="{ry:.1f}" '
        f'x2="{rx - rr - 2:.1f}" y2="{ry:.1f}" stroke="#444" '
        f'stroke-width="1.2" marker-end="url(#arrow)"/>'
    )
    edge_labels.append((rx - rr - 20, ry, 0.0, -1.0, "start", 1.0))

    for node in nodes:
        nid = id(node)
        cx, cy = coords[nid]
        _draw_state(lines, cx, cy, radii[nid], node.terminal, state_labels[nid])

    for lx, ly, nx, ny, label, side in edge_labels:
        _draw_edge_label(lines, lx, ly, nx, ny, label, side=side)

    lines.append("</svg>")
    return "\n".join(lines)


def render_dfa_svg(
    dfa: DFA,
    words: list[str],
    state_labels: dict[int, set[str]],
    *,
    minimized: bool,
) -> str:
    radii = _compute_radii(state_labels)
    gap_scale = _gap_scale(radii)
    positions = layout_dfa(dfa)
    coords = _scale_positions(positions, gap_scale=gap_scale)
    width, height = _canvas_size(coords, radii)

    kind = "Minimal DFA" if minimized else "DFA"
    lines = _svg_header(width, height, f"{kind} · {', '.join(words)}")
    _marker_defs(lines)

    by_edge: dict[tuple[int, int], list[str]] = {}
    for (s, a), t in sorted(dfa.delta.items()):
        if s in coords and t in coords:
            by_edge.setdefault((s, t), []).append(a)

    edge_labels: list[tuple[float, float, float, float, str, float]] = []
    for (s, t), labels in sorted(by_edge.items()):
        sx, sy = coords[s]
        tx, ty = coords[t]
        rs, rt = radii[s], radii[t]
        for i, label in enumerate(labels):
            curve = 26 * (i - (len(labels) - 1) / 2.0)
            geom = _draw_edge(lines, sx, sy, tx, ty, rs, rt, curved=curve)
            if geom:
                lx, ly, nx, ny = geom
                side = 1.0 if i % 2 == 0 else -1.0
                edge_labels.append((lx, ly, nx, ny, label, side))

    rx, ry = coords[dfa.start]
    rr = radii[dfa.start]
    lines.append(
        f'<line x1="{rx - rr - 36:.1f}" y1="{ry:.1f}" '
        f'x2="{rx - rr - 2:.1f}" y2="{ry:.1f}" stroke="#444" '
        f'stroke-width="1.2" marker-end="url(#arrow)"/>'
    )

    for s in sorted(coords):
        cx, cy = coords[s]
        _draw_state(lines, cx, cy, radii[s], s in dfa.finals, state_labels.get(s, set()))

    for lx, ly, nx, ny, label, side in edge_labels:
        _draw_edge_label(lines, lx, ly, nx, ny, label, side=side)

    lines.append("</svg>")
    return "\n".join(lines)


def vocabulary_for_experiment(exp_name: str) -> list[str]:
    """Word list for an experiment name (handles `<regime>_s` spaced variants)."""
    from experiment import experiment_regime

    return REGIMES[experiment_regime(exp_name)]


def write_vocabulary_diagrams_for_experiment(exp_name: str) -> tuple[Path, Path]:
    """Write trie + min-DFA SVGs under experiments/<exp_name>/plots/."""
    return write_vocabulary_diagrams(
        vocabulary_for_experiment(exp_name),
        experiment_plots_dir(exp_name),
    )


def write_vocabulary_diagrams(
    words: list[str],
    out_dir: Path,
    *,
    trie_name: str = "vocabulary_trie.svg",
    dfa_name: str = "vocabulary_min_dfa.svg",
) -> tuple[Path, Path]:
    """Build trie + minimized DFA SVGs for `words`; return output paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = build_trie(words)
    dfa, old_to_new = minimize_dfa(trie_to_dfa(root))
    min_labels = minimized_prefix_sets(root, old_to_new)

    trie_path = out_dir / trie_name
    dfa_path = out_dir / dfa_name
    trie_path.write_text(render_trie_svg(root, words), encoding="utf-8")
    dfa_path.write_text(
        render_dfa_svg(dfa, words, min_labels, minimized=True), encoding="utf-8"
    )
    return trie_path, dfa_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("regime", nargs="?", default=None,
                        choices=list(REGIMES.keys()),
                        help="task regime name (word list from task.py)")
    parser.add_argument("--words", nargs="+", default=None,
                        help="explicit word list (overrides regime)")
    parser.add_argument("--exp", default=None,
                        help="write to experiments/<exp>/plots/ (default: regime name)")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: plots/ or experiment plots/)")
    args = parser.parse_args()

    if args.words:
        words = list(args.words)
    elif args.regime:
        words = REGIMES[args.regime]
    else:
        parser.error("provide a regime name or --words")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.exp or args.regime:
        out_dir = experiment_plots_dir(args.exp or args.regime)
    else:
        out_dir = Path("plots")

    trie_path, dfa_path = write_vocabulary_diagrams(words, out_dir)
    print(f"Words: {words}")
    print(f"Wrote: {trie_path}")
    print(f"Wrote: {dfa_path}")


if __name__ == "__main__":
    main()
