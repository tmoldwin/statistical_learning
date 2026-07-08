"""Generate slideshow SVGs: character RNN training (clean layout, no overlaps)."""
# fig2_version: bilateral_mesh_v3

from __future__ import annotations

import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task import REGIMES, generate_sequence

BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#666"
ACCENT = "#4477AA"
ACCENT2 = "#F58518"
C_OUT = "#2a9d8f"
EDGE_IN = "#8eb5d6"
EDGE_OUT = "#7ec4b8"
IN_VOCAB = "#2ca02c"
OOV = "#d62728"
WORD_COLORS: list[tuple[str, str]] = [
    ("#4477AA", "#eef4fb"),
    ("#F58518", "#fff4e8"),
    ("#EE6677", "#fdeef0"),
    ("#228833", "#e8f5ea"),
    ("#CCBB44", "#faf6e3"),
    ("#66CCEE", "#e8f7fc"),
    ("#AA3377", "#f5eaf0"),
    ("#BBBBBB", "#f0f0f0"),
]
PANEL_BG = "#ffffff"
PANEL_BORDER = "#ccc8bc"
ARROW = "#555"
FONT = "ui-monospace, Consolas, 'Courier New', monospace"
SANS = "Segoe UI, Helvetica, Arial, sans-serif"
MARGIN = 56


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def sub(s: str) -> str:
    return f'<tspan baseline-shift="sub" font-size="10">{s}</tspan>'


def colored_w(name: str, color: str) -> str:
    subscript = name[1:] if name.startswith('W') else name
    return (
        f'<tspan fill="{color}">W</tspan>'
        f'<tspan baseline-shift="sub" font-size="10" fill="{color}">{subscript}</tspan>'
    )


FIG1_LAYOUT_PANELS: list[tuple[float, float, float, float]] = []


def svg_open(w: int, h: int, *, background: bool = True) -> list[str]:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '<marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">',
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{ARROW}"/></marker>',
        '<marker id="arrowhead-blue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">',
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{ACCENT}"/></marker>',
        '<marker id="arrowhead-orange" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">',
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{ACCENT2}"/></marker>',
        "</defs>",
        f'<style>text {{ font-family: {SANS}; fill: {INK}; }}</style>',
    ]
    if background:
        lines.append(f'<rect width="{w}" height="{h}" fill="{BG}"/>')
    return lines


def svg_close(lines: list[str]) -> str:
    lines.append("</svg>")
    return "\n".join(lines)


def title_block(lines: list[str], y: int, title: str, subtitle: str = "") -> int:
    lines.append(
        f'<text x="{MARGIN}" y="{y}" font-family="{SANS}" font-size="26" '
        f'font-weight="700" fill="{INK}">{esc(title)}</text>'
    )
    y += 36
    if subtitle:
        lines.append(
            f'<text x="{MARGIN}" y="{y}" font-family="{SANS}" font-size="15" '
            f'fill="{MUTED}">{esc(subtitle)}</text>'
        )
        y += 30
    return y + 20


def caption(
    lines: list[str],
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "start",
    size: int = 12,
    fill: str = MUTED,
) -> None:
    lines.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{SANS}" '
        f'font-size="{size}" font-weight="600" fill="{fill}">{esc(text)}</text>'
    )


def segment_greedy(text: str, vocab: set[str]) -> list[tuple[str, bool]]:
    words = sorted(vocab, key=len, reverse=True)
    out: list[tuple[str, bool]] = []
    i = 0
    while i < len(text):
        matched = next((w for w in words if text.startswith(w, i)), None)
        if matched:
            out.extend((ch, True) for ch in matched)
            i += len(matched)
        else:
            out.append((text[i], False))
            i += 1
    return out


def trim_in_vocab(text: str, vocab: set[str]) -> str:
    while text:
        if all(ok for _, ok in segment_greedy(text, vocab)):
            return text
        text = text[:-1]
    return text


def make_stream(words: list[str], n_chars: int, seed: int) -> str:
    return generate_sequence(words, n_chars, seed=seed, word_space=False)



def make_before_rollout(after: str, *, prefix_len: int = 3, seed: int = 99) -> str:
    prefix = after[:prefix_len]
    tail = list(after[prefix_len:])
    rng = random.Random(seed)
    rng.shuffle(tail)
    return prefix + "".join(tail)


def make_stream_words(words: list[str], n_chars: int, seed: int) -> list[str]:
    """Same sampling as generate_sequence, but returns the sampled word list."""
    rng = random.Random(seed)
    picked: list[str] = []
    out_len = 0
    while out_len < n_chars:
        w = rng.choice(words)
        picked.append(w)
        for _ in w:
            if out_len >= n_chars:
                break
            out_len += 1
    return picked


def _word_color(word: str, words: list[str]) -> tuple[str, str]:
    idx = words.index(word) if word in words else 0
    return WORD_COLORS[idx % len(WORD_COLORS)]


def _stream_items(words: list[str], picked: list[str], n_chars: int) -> list[tuple]:
    items: list[tuple] = []
    count = 0
    for w in picked:
        wi = words.index(w)
        stroke, _ = _word_color(w, words)
        for ch in w:
            if count >= n_chars:
                return items
            items.append(("ch", ch, wi, stroke))
            count += 1
    return items


def _layout_stream_rows(
    items: list[tuple],
    x: float,
    max_width: float,
    *,
    cell_w: float = 26,
    char_gap: float = 4,
    word_gap: float = 0,
) -> list[list[tuple[float, tuple]]]:
    rows: list[list[tuple[float, tuple]]] = [[]]
    cx = x
    for item in items:
        if item[0] == "gap":
            if rows[-1] and cx + word_gap > x + max_width:
                rows.append([])
                cx = x
            elif rows[-1]:
                cx += word_gap
            continue
        need_gap = char_gap if rows[-1] else 0
        if rows[-1] and cx + need_gap + cell_w > x + max_width:
            rows.append([])
            cx = x
            need_gap = 0
        cx += need_gap
        rows[-1].append((cx, item))
        cx += cell_w
    return [r for r in rows if r]


def measure_word_char_stream(
    words: list[str],
    seed: int,
    n_chars: int,
    max_width: float,
    *,
    cell_h: float = 32,
    row_gap: float = 10,
    **layout_kw,
) -> float:
    picked = make_stream_words(words, n_chars, seed)
    items = _stream_items(words, picked, n_chars)
    rows = _layout_stream_rows(items, 0.0, max_width, **layout_kw)
    if not rows:
        return 0.0
    return len(rows) * cell_h + (len(rows) - 1) * row_gap


def render_word_char_stream(
    lines: list[str],
    x: float,
    y: float,
    words: list[str],
    seed: int,
    n_chars: int,
    max_width: float,
    *,
    cell_w: float = 26,
    cell_h: float = 32,
    char_gap: float = 4,
    word_gap: float = 0,
    row_gap: float = 10,
    font_size: int = 16,
) -> float:
    picked = make_stream_words(words, n_chars, seed)
    items = _stream_items(words, picked, n_chars)
    rows = _layout_stream_rows(
        items, x, max_width, cell_w=cell_w, char_gap=char_gap, word_gap=word_gap
    )
    for ri, row in enumerate(rows):
        ry = y + ri * (cell_h + row_gap)
        for cx, item in row:
            if item[0] != "ch":
                continue
            _, ch, wi, stroke = item
            _, fill = _word_color(words[wi], words)
            lines.append(
                f'<rect x="{cx:.1f}" y="{ry:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
            )
            lines.append(
                f'<text x="{cx + cell_w / 2:.1f}" y="{ry + cell_h * 0.68:.1f}" '
                f'text-anchor="middle" font-family="{FONT}" font-size="{font_size}" '
                f'fill="{stroke}">{esc(ch)}</text>'
            )
    if not rows:
        return y
    return y + len(rows) * (cell_h + row_gap) - row_gap




def _auto_cols(max_width: float, cell_w: float, gap: float, max_cols: int) -> int:
    fit = max(1, int((max_width + gap) // (cell_w + gap)))
    return min(max_cols, fit)


def render_char_cells(
    lines: list[str],
    x: float,
    y: float,
    text: str,
    vocab: set[str],
    *,
    cell_w: float = 26,
    cell_h: float = 32,
    gap: float = 4,
    font_size: int = 16,
    max_cols: int = 36,
    max_width: float | None = None,
    row_gap: float = 10,
    uniform_color: str | None = None,
) -> float:
    """Each character in its own box. Returns bottom y."""
    parts = segment_greedy(text, vocab)
    if not parts:
        return y
    if max_width is not None:
        max_cols = _auto_cols(max_width, cell_w, gap, max_cols)
    rows: list[list[tuple[str, bool]]] = []
    for i in range(0, len(parts), max_cols):
        rows.append(parts[i : i + max_cols])
    for ri, row in enumerate(rows):
        ry = y + ri * (cell_h + row_gap)
        for ci, (ch, ok) in enumerate(row):
            cx = x + ci * (cell_w + gap)
            color = uniform_color if uniform_color else (IN_VOCAB if ok else OOV)
            lines.append(
                f'<rect x="{cx:.1f}" y="{ry:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'rx="4" fill="#ffffff" stroke="#ddd8cc" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{cx + cell_w / 2:.1f}" y="{ry + cell_h * 0.68:.1f}" '
                f'text-anchor="middle" font-family="{FONT}" font-size="{font_size}" '
                f'fill="{color}">{esc(ch)}</text>'
            )
    return y + len(rows) * (cell_h + row_gap) - row_gap




def corpus_window_highlight(
    lines: list[str],
    x: float,
    y: float,
    start: int,
    length: int,
    *,
    cell_w: float = 26,
    cell_h: float = 32,
    gap: float = 4,
    pad: float = 5,
    max_cols: int = 40,
    row_gap: float = 10,
) -> None:
    if length <= 0:
        return
    end = start + length - 1
    sr, sc = divmod(start, max_cols)
    er, ec = divmod(end, max_cols)
    x1 = x + sc * (cell_w + gap)
    y1 = y + sr * (cell_h + row_gap)
    x2 = x + ec * (cell_w + gap) + cell_w
    y2 = y + er * (cell_h + row_gap) + cell_h
    lines.append(
        f'<rect x="{x1 - pad:.1f}" y="{y1 - pad:.1f}" '
        f'width="{x2 - x1 + 2 * pad:.1f}" height="{y2 - y1 + 2 * pad:.1f}" rx="6" '
        f'fill="none" stroke="{ACCENT2}" stroke-width="2" stroke-dasharray="6 3"/>'
    )


def render_colored_line(
    lines: list[str],
    x: float,
    y: float,
    text: str,
    vocab: set[str],
    *,
    font_size: int = 15,
) -> float:
    parts = segment_greedy(text, vocab)
    if not parts:
        return y
    spans: list[tuple[str, bool]] = []
    buf = ""
    cur_ok = parts[0][1]
    for ch, ok in parts:
        if ok != cur_ok and buf:
            spans.append((buf, cur_ok))
            buf = ""
            cur_ok = ok
        buf += ch
    if buf:
        spans.append((buf, cur_ok))
    tspan_parts = []
    for s, ok in spans:
        color = IN_VOCAB if ok else OOV
        tspan_parts.append(f'<tspan style="fill:{color}">{esc(s)}</tspan>')
    lines.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{font_size}">'
        f'{"".join(tspan_parts)}</text>'
    )
    return y + font_size + 8


def _chip_rows(words: list[str], max_width: float) -> int:
    gap_x = 10
    cx = 0.0
    rows = 1
    started = False
    for w in words:
        chip_w = max(48, len(w) * 11 + 22)
        if started and cx + chip_w > max_width:
            rows += 1
            cx = chip_w + gap_x
        else:
            if started:
                cx += gap_x
            cx += chip_w
            started = True
    return rows


def word_chips(
    lines: list[str],
    x: float,
    y: float,
    words: list[str],
    max_width: float,
    *,
    colored: bool = False,
) -> float:
    chip_h = 30
    gap_x = 10
    gap_y = 10
    cx, row_y = x, y
    for w in words:
        chip_w = max(48, len(w) * 11 + 22)
        if cx + chip_w > x + max_width and cx > x:
            cx = x
            row_y += chip_h + gap_y
        stroke, fill = _word_color(w, words) if colored else (ACCENT, "#eef4fb")
        text_fill = stroke if colored else INK
        lines.append(
            f'<rect x="{cx:.0f}" y="{row_y:.0f}" width="{chip_w}" height="{chip_h}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        )
        lines.append(
            f'<text x="{cx + chip_w / 2:.1f}" y="{row_y + 20:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="14" fill="{text_fill}">{esc(w)}</text>'
        )
        cx += chip_w + gap_x
    return row_y + chip_h


def down_arrow(lines: list[str], x: float, y1: float, y2: float) -> None:
    lines.append(
        f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
        f'stroke="{ARROW}" stroke-width="2" marker-end="url(#arrowhead)"/>'
    )


def legend_char_colors(lines: list[str], x: float, y: float) -> None:
    for i, (color, label) in enumerate([(IN_VOCAB, "in-vocabulary"), (OOV, "out-of-vocabulary")]):
        lx = x + i * 200
        lines.append(f'<rect x="{lx:.0f}" y="{y - 13:.0f}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{lx + 20:.0f}" y="{y:.0f}" font-size="13" fill="{MUTED}">{label}</text>')


def build_figure1() -> str:
    global FIG1_LAYOUT_PANELS
    FIG1_LAYOUT_PANELS = []
    w = 1280
    mixed_vocab = REGIMES["ten_word_overlap"][:4] + REGIMES["six_word_four_letter"][:4]
    conditions = [
        ("8 Words · Length 3", REGIMES["ten_word_overlap"][:8], 42),
        ("4 Words · Length 5", REGIMES["six_word_five_letter"][:4], 43),
        ("8 Words · Mixed 3–4", mixed_vocab, 44),
    ]
    inner_w = w - 2 * MARGIN
    content_w = inner_w - 40
    bx = MARGIN + 20
    cx = bx + content_w / 2
    stream_chars = 37
    cell_h = 32
    stream_label_gap = 48
    stream_top_gap = 70

    blocks = []
    y = MARGIN
    for label, words, seed in conditions:
        box_y = y
        chip_start = box_y + 74
        chip_rows = _chip_rows(words, content_w)
        chip_bottom = chip_start + chip_rows * 30 + max(0, chip_rows - 1) * 10
        stream_y = chip_bottom + stream_top_gap
        stream_h = measure_word_char_stream(words, seed, stream_chars, content_w, cell_h=cell_h)
        block_h = stream_y + stream_h + 20 - box_y
        blocks.append((label, words, seed, box_y, block_h))
        y = box_y + block_h + 20
    total_h = y + MARGIN
    lines = svg_open(w, total_h)
    for label, words, seed, box_y, block_h in blocks:
        FIG1_LAYOUT_PANELS.append((bx, box_y, content_w, block_h))
        lines.append(
            f'<rect x="{MARGIN}" y="{box_y}" width="{inner_w}" height="{block_h:.0f}" rx="12" '
            f'fill="{PANEL_BG}" stroke="{PANEL_BORDER}" stroke-width="1.5"/>'
        )
        ty = box_y + 34
        lines.append(
            f'<text x="{bx:.1f}" y="{ty:.1f}" font-family="{SANS}" font-size="16" '
            f'font-weight="600" fill="{ACCENT}">{esc(label)}</text>'
        )
        ty += 26
        caption(lines, bx, ty, "Vocabulary")
        chip_start = box_y + 74
        chip_bottom = word_chips(lines, bx, chip_start, words, content_w, colored=True)
        down_arrow(lines, cx, chip_bottom + 8, chip_bottom + 28)
        caption(lines, cx, chip_bottom + stream_label_gap, "Training Stream", anchor="middle")
        render_word_char_stream(
            lines, bx, chip_bottom + stream_top_gap, words, seed, stream_chars, content_w, cell_h=cell_h,
        )
    return svg_close(lines)



def _layer_positions(lx: float, n: int, y_top: float, y_bot: float) -> list[tuple[float, float]]:
    span = y_bot - y_top
    return [(lx, y_top + span * (i + 1) / (n + 1)) for i in range(n)]


def draw_rnn_schematic(lines: list[str], x: float, y: float, w: float) -> float:
    """Input -> one recurrent hidden column (all-pairs mesh) -> output."""
    h = 270

    y_top = y + 36
    y_bot = y + h - 36
    lx_in = x + w * 0.14
    lx_h = x + w * 0.50
    lx_out = x + w * 0.78

    n_in, n_h, n_out = 5, 4, 5
    in_labels = ["a", "c", "h", "t", "m"]
    h_labels = ["h0", "h1", "h2", "h3"]
    out_labels = ["a", "c", "h", "t", "m"]
    active_in = 3

    in_pos = _layer_positions(lx_in, n_in, y_top, y_bot)
    h_pos = _layer_positions(lx_h, n_h, y_top, y_bot)
    out_pos = _layer_positions(lx_out, n_out, y_top, y_bot)

    def connect(
        src: list[tuple[float, float]],
        dst: list[tuple[float, float]],
        color: str,
        width: float,
        opacity: float,
    ) -> None:
        for sx, sy in src:
            for dx, dy in dst:
                lines.append(
                    f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{dx:.1f}" y2="{dy:.1f}" '
                    f'stroke="{color}" stroke-width="{width:.1f}" opacity="{opacity:.2f}"/>'
                )

    connect(in_pos, h_pos, EDGE_IN, 1.3, 0.85)
    connect(h_pos, out_pos, EDGE_OUT, 1.3, 0.85)

    max_bulge = 0.0
    for i in range(n_h):
        for j in range(n_h):
            if i == j:
                continue
            sx, sy = h_pos[i]
            dx, dy = h_pos[j]
            bulge = 18 + abs(i - j) * 6
            max_bulge = max(max_bulge, bulge)
            side = 1 if i < j else -1
            cx = sx + side * bulge
            cy = (sy + dy) / 2
            lines.append(
                f'<path d="M{sx:.1f},{sy:.1f} Q{cx:.1f},{cy:.1f} {dx:.1f},{dy:.1f}" '
                f'fill="none" stroke="{ACCENT2}" stroke-width="1.6" opacity="0.9"/>'
            )

    def draw_nodes(
        positions: list[tuple[float, float]],
        labels: list[str],
        r: float,
        active: int | None = None,
        fill: str = "#eef4fb",
        stroke: str = ACCENT,
    ) -> None:
        for i, (px, py) in enumerate(positions):
            if active is not None and i == active:
                node_fill, node_stroke, sw = "#cce0f5", "#225588", 2.0
            else:
                node_fill, node_stroke, sw = fill, stroke, 1.6
            lines.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="{node_fill}" '
                f'stroke="{node_stroke}" stroke-width="{sw}"/>'
            )
            lines.append(
                f'<text x="{px:.1f}" y="{py + 4:.1f}" text-anchor="middle" font-family="{FONT}" '
                f'font-size="11" fill="{INK}">{esc(labels[i])}</text>'
            )

    draw_nodes(in_pos, in_labels, 15, active=active_in, fill="#eef4fb", stroke=ACCENT)
    draw_nodes(h_pos, h_labels, 15, fill="#fff3e6", stroke=ACCENT2)
    draw_nodes(out_pos, out_labels, 15, fill="#e8f6f4", stroke=C_OUT)

    hdr_y = y + 18
    lines.append(
        f'<text x="{lx_in:.1f}" y="{hdr_y:.0f}" text-anchor="middle" font-size="13" fill="{ACCENT}">'
        f"input x{sub('t')}</text>"
    )
    lines.append(
        f'<text x="{lx_h:.1f}" y="{hdr_y:.0f}" text-anchor="middle" font-size="13" fill="{ACCENT2}">'
        f"hidden h{sub('t')}</text>"
    )
    lines.append(
        f'<text x="{lx_out:.1f}" y="{hdr_y:.0f}" text-anchor="middle" font-size="13" fill="{C_OUT}">'
        f"softmax</text>"
    )

    lines.append(
        f'<text x="{(lx_in + lx_h) / 2:.0f}" y="{y_top + 6:.0f}" text-anchor="middle" '
        f'font-size="13" font-weight="600">{colored_w("Wxh", ACCENT)}</text>'
    )
    lines.append(
        f'<text x="{lx_h + max_bulge + 12:.0f}" y="{(y_top + y_bot) / 2:.0f}" text-anchor="start" '
        f'font-size="13" font-weight="600">{colored_w("Whh", ACCENT2)}</text>'
    )
    lines.append(
        f'<text x="{(lx_h + lx_out) / 2:.0f}" y="{y_top + 6:.0f}" text-anchor="middle" '
        f'font-size="13" font-weight="600">{colored_w("Wyh", C_OUT)}</text>'
    )
    lines.append(
        f'<text x="{x + w / 2:.0f}" y="{y + h - 8:.0f}" text-anchor="middle" font-size="13" fill="{MUTED}">'
        f"h{sub('t')} = tanh({colored_w('Wxh', ACCENT)} x{sub('t')} + {colored_w('Whh', ACCENT2)} h{sub('t-1')}) "
        f"· predict next char · cross-entropy loss</text>"
    )
    return y + h


def render_bptt_panel(
    lines: list[str],
    x: float,
    y: float,
    window: str,
    w: float,
) -> float:
    n = len(window)
    cell_w, cell_h, gap = 32, 38, 6
    wx = x
    ty = y
    for i, ch in enumerate(window):
        cx = wx + i * (cell_w + gap)
        cy = ty
        lines.append(
            f'<rect x="{cx:.0f}" y="{cy:.0f}" width="{cell_w}" height="{cell_h}" rx="5" '
            f'fill="#ffffff" stroke="{ACCENT}" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{cx + cell_w / 2:.1f}" y="{cy + cell_h * 0.72:.0f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="17" fill="{ACCENT}">{esc(ch)}</text>'
        )
        ax = cx + cell_w / 2
        lines.append(
            f'<line x1="{ax:.1f}" y1="{cy + cell_h:.1f}" x2="{ax:.1f}" y2="{cy + cell_h + 18:.1f}" '
            f'stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrowhead)"/>'
        )
    ty += cell_h + 28
    targets = list(window[1:]) + ["…"]
    for i, ch in enumerate(targets):
        cx = wx + i * (cell_w + gap)
        lines.append(
            f'<rect x="{cx:.0f}" y="{ty:.0f}" width="{cell_w}" height="{cell_h}" rx="5" '
            f'fill="#ffffff" stroke="{C_OUT}" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{cx + cell_w / 2:.1f}" y="{ty + cell_h * 0.72:.0f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="17" fill="{C_OUT}">{esc(ch)}</text>'
        )
    return y + cell_h + 28 + cell_h


def build_figure2() -> str:
    w = 1280
    words = REGIMES["ten_word_overlap"]
    vocab = set(words)
    demo = trim_in_vocab(make_stream(words, 32, seed=100), vocab)[:24]
    window = demo[4:12]

    inner_w = w - 2 * MARGIN
    pad = 28
    content_w = inner_w - 2 * pad
    bx = MARGIN + pad
    panel_y = MARGIN

    panel: list[str] = []
    rnn_bottom = draw_rnn_schematic(panel, bx, panel_y, content_w)
    ty = rnn_bottom + 20
    panel.append(
        f'<line x1="{bx:.0f}" y1="{ty:.0f}" x2="{bx + content_w:.0f}" y2="{ty:.0f}" '
        f'stroke="#ddd8cc" stroke-width="1"/>'
    )
    ty += 24
    corpus_y = ty
    ty = render_char_cells(
        panel, bx, ty, demo, vocab,
        max_cols=40, max_width=content_w, cell_w=26, cell_h=32, gap=4,
        uniform_color=ACCENT,
    )
    corpus_window_highlight(
        panel, bx, corpus_y, 4, len(window),
        cell_w=26, cell_h=32, gap=4, max_cols=40,
    )
    ty += 24
    bptt_bottom = render_bptt_panel(panel, bx, ty, window, content_w)
    total_h = int(bptt_bottom + MARGIN)

    lines = svg_open(w, total_h)
    lines.extend(panel)
    return svg_close(lines)


def build_figure3() -> str:
    w = 1280
    words = REGIMES["ten_word_overlap"]
    vocab = set(words)
    rollout_len = 64
    after = trim_in_vocab(make_stream(words, rollout_len, seed=200), vocab)
    before = make_before_rollout(after)

    inner_w = w - 2 * MARGIN
    bx = MARGIN + 20
    content_w = inner_w - 40
    stream_fs = 20
    stream_line_h = stream_fs + 8
    panel_y = title_block([], 44, "Evaluation: model rollout",
        "Generation warm-started from a corpus prefix")

    ty = panel_y + 34 + 26 + 14 + 30 + 36
    ty += 24 + 22 + stream_line_h
    ty += 36 + 24 + 22 + stream_line_h + 36
    panel_h = ty - panel_y
    total_h = panel_y + panel_h + MARGIN

    lines = svg_open(w, total_h)
    title_block(lines, 44, "Evaluation: model rollout",
        "Generation warm-started from a corpus prefix")
    lines.append(
        f'<rect x="{MARGIN}" y="{panel_y}" width="{inner_w}" height="{panel_h:.0f}" rx="12" '
        f'fill="{PANEL_BG}" stroke="{PANEL_BORDER}" stroke-width="1.5"/>'
    )

    ty = panel_y + 34
    lines.append(
        f'<text x="{bx}" y="{ty}" font-size="16" font-weight="600" fill="{ACCENT}">Vocabulary</text>'
    )
    ty += 26
    ty = word_chips(lines, bx, ty, words, content_w) + 36

    def rollout_row(heading: str, sub: str, text: str) -> None:
        nonlocal ty
        lines.append(
            f'<text x="{bx}" y="{ty}" font-size="16" font-weight="600" fill="{INK}">{esc(heading)}</text>'
        )
        ty += 24
        lines.append(f'<text x="{bx}" y="{ty}" font-size="13" fill="{MUTED}">{esc(sub)}</text>')
        ty += 22
        ty = render_colored_line(lines, bx, ty, text, vocab, font_size=stream_fs)

    rollout_row("Before training", "high OOV rate", before)
    ty += 36
    rollout_row("After training", "mostly vocabulary words", after)
    legend_char_colors(lines, bx, ty + 24)
    return svg_close(lines)





def _text_bbox(el) -> tuple[float, float, float, float]:
    x = float(el.get("x", 0))
    y = float(el.get("y", 0))
    fs = float(el.get("font-size", 13))
    anchor = el.get("text-anchor", "start")
    text = "".join(el.itertext())
    w = max(len(text), 1) * fs * 0.58
    if anchor == "middle":
        x -= w / 2
    return x, y - fs * 0.9, x + w, y + fs * 0.25


def check_svg_layout(path: Path, *panels: tuple[float, float]) -> list[str]:
    """Return layout errors: text overlaps and char cells outside all panels."""
    root = ET.parse(path).getroot()
    svg_w = float(root.get("width", 1280))
    svg_h = float(root.get("height", 800))
    errors: list[str] = []
    texts: list[tuple[tuple[float, float, float, float], str]] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "text":
            t = "".join(el.itertext()).strip()
            if not t:
                continue
            texts.append((_text_bbox(el), t))
        elif tag == "rect":
            rw = el.get("width")
            rh = el.get("height")
            if not rw or not rh:
                continue
            x, y, rw_f, rh_f = (
                float(el.get("x", 0)),
                float(el.get("y", 0)),
                float(rw),
                float(rh),
            )
            if rh_f <= 36 and rw_f <= 34 and panels:  # char cell
                inside = False
                for panel in panels:
                    if len(panel) == 2:
                        px, pw = panel
                        py, ph = 0.0, svg_h
                    else:
                        px, py, pw, ph = panel
                    if (
                        x >= px - 2
                        and x + rw_f <= px + pw + 2
                        and y >= py - 2
                        and y + rh_f <= py + ph + 2
                    ):
                        inside = True
                        break
                if not inside:
                    panel_str = ", ".join(str(p) for p in panels)
                    errors.append(
                        f"char cell at x={x:.0f}, y={y:.0f} outside panels {panel_str}"
                    )
    for i, (a, ta) in enumerate(texts):
        if a[0] < -2 or a[2] > svg_w + 2 or a[3] > svg_h + 2:
            errors.append(f"text out of canvas: {ta!r}")
        for j, (b, tb) in enumerate(texts):
            if j <= i:
                continue
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            if ox > 3 and oy > 3 and ta != tb:
                errors.append(f"text overlap: {ta!r} vs {tb!r}")
    return errors


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "plots" / "training_diagram"
    out_dir.mkdir(parents=True, exist_ok=True)
    inner_w = 1280 - 2 * MARGIN
    col_w = (inner_w - 80) / 2
    right_x = MARGIN + col_w + 80
    figures = [
        ("01_corpus_generation.svg", build_figure1, MARGIN + 20, inner_w - 40),
        ("02_bptt_and_rnn.svg", build_figure2, MARGIN + 28, inner_w - 56),
        ("03_rollout_evaluation.svg", build_figure3),
    ]
    for old in ("02_bptt_feeding.svg", "03_rnn_architecture.svg", "04_rollout_evaluation.svg"):
        p = out_dir / old
        if p.exists():
            p.unlink()
    failed = False
    def _fig1_panel_specs(_svg_text: str) -> list[tuple[float, float, float, float]]:
        return list(FIG1_LAYOUT_PANELS)

    for item in figures:
        name, builder = item[0], item[1]
        panels = item[2:] if len(item) > 2 else ()
        path = out_dir / name
        svg_text = builder()
        path.write_text(svg_text, encoding="utf-8")
        if name == "01_corpus_generation.svg":
            panel_specs = _fig1_panel_specs(svg_text)
        else:
            panel_specs = [
                (panels[i], panels[i + 1]) for i in range(0, len(panels), 2)
            ]
        errs = check_svg_layout(path, *panel_specs) if panel_specs else []
        if errs:
            failed = True
            print(f"FAIL {name}:")
            for e in errs[:12]:
                print(f"  - {e}")
        else:
            print(f"OK   {name}")
    if failed:
        raise SystemExit(1)
    print("All figures passed layout check.")


if __name__ == "__main__":
    main()
