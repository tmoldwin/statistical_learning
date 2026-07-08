"""Restore figure-2 generator (single hidden column + BPTT panel). Run after any revert."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "generate_training_diagram.py"
MARKER = "# fig2_version: bilateral_mesh_v3"


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:4]:
        return raw.decode("utf-16")
    return raw.decode("utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def ensure_constants(text: str) -> str:
    if "C_OUT = " not in text:
        text = text.replace(
            'ACCENT2 = "#F58518"\n',
            'ACCENT2 = "#F58518"\nC_OUT = "#2a9d8f"\nEDGE_IN = "#8eb5d6"\nEDGE_OUT = "#7ec4b8"\n',
        )
    return text


def ensure_colored_w(text: str) -> str:
    if "def colored_w(" in text:
        return text
    block = '''

def colored_w(name: str, color: str) -> str:
    subscript = name[1:] if name.startswith("W") else name
    return (
        f\'<tspan fill="{color}">W</tspan>\'
        f\'<tspan baseline-shift="sub" font-size="10" fill="{color}">{subscript}</tspan>\'
    )

'''
    return text.replace(
        "    return f'<tspan baseline-shift=\"sub\" font-size=\"10\">{s}</tspan>'\n\n\n",
        "    return f'<tspan baseline-shift=\"sub\" font-size=\"10\">{s}</tspan>'\n" + block,
    )


def ensure_uniform_color(text: str) -> str:
    if "uniform_color: str | None = None" in text:
        return text
    text = text.replace(
        "    row_gap: float = 10,\n) -> float:",
        "    row_gap: float = 10,\n    uniform_color: str | None = None,\n) -> float:",
    )
    return text.replace(
        "            color = IN_VOCAB if ok else OOV\n",
        "            color = uniform_color if uniform_color else (IN_VOCAB if ok else OOV)\n",
        1,
    )


REPLACEMENT = '''
def _layer_positions(lx: float, n: int, y_top: float, y_bot: float) -> list[tuple[float, float]]:
    span = y_bot - y_top
    return [(lx, y_top + span * (i + 1) / (n + 1)) for i in range(n)]


def draw_rnn_schematic(lines: list[str], x: float, y: float, w: float) -> float:
    """Input -> one recurrent hidden column (all-pairs mesh) -> output."""
    h = 300
    lines.append(
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="10" '
        f'fill="#faf8f4" stroke="{ACCENT}" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{x + 16:.0f}" y="{y + 24:.0f}" '
        f'font-size="16" font-weight="600" fill="{INK}">Character-level RNN (one timestep)</text>'
    )

    y_top = y + 62
    y_bot = y + h - 52
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

    hdr_y = y + 46
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
        f'<text x="{x + w / 2:.0f}" y="{y + h - 14:.0f}" text-anchor="middle" font-size="13" fill="{MUTED}">'
        f"h{sub('t')} = tanh({colored_w('Wxh', ACCENT)} x{sub('t')} + {colored_w('Whh', ACCENT2)} h{sub('t-1')}) "
        f"\u00b7 predict next char \u00b7 cross-entropy loss</text>"
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
    pad = 16
    box_h = pad + 22 + cell_h + 28 + cell_h + pad
    lines.append(
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{box_h:.0f}" rx="10" '
        f'fill="#f5f5f5" stroke="#ccc8bc" stroke-width="1.2"/>'
    )
    ty = y + pad + 14
    lines.append(
        f'<text x="{x + pad:.0f}" y="{ty:.0f}" font-size="15" font-weight="600" fill="{ACCENT2}">'
        f"BPTT window (T = {n})</text>"
    )
    ty += 20
    wx = x + pad
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
    targets = list(window[1:]) + ["\u2026"]
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
    return y + box_h


def build_figure2() -> str:
    w = 1280
    words = REGIMES["ten_word_overlap"]
    vocab = set(words)
    demo = trim_in_vocab(make_stream(words, 32, seed=100), vocab)[:24]
    window = demo[4:12]

    header: list[str] = []
    y = title_block(
        header, 44,
        "Train the RNN (BPTT)",
        "Feed characters one at a time \u00b7 hidden state carries forward \u00b7 predict next character",
    )
    inner_w = w - 2 * MARGIN
    pad = 28
    content_w = inner_w - 2 * pad
    bx = MARGIN + pad
    panel_y = y

    panel: list[str] = []
    rnn_bottom = draw_rnn_schematic(panel, bx, panel_y + 20, content_w)
    ty = rnn_bottom + 20
    panel.append(
        f'<line x1="{bx:.0f}" y1="{ty:.0f}" x2="{bx + content_w:.0f}" y2="{ty:.0f}" '
        f'stroke="#ddd8cc" stroke-width="1"/>'
    )
    ty += 24
    ty = render_char_cells(
        panel, bx, ty, demo, vocab,
        max_cols=40, max_width=content_w, cell_w=26, cell_h=32, gap=4,
        uniform_color=ACCENT,
    ) + 24
    bptt_bottom = render_bptt_panel(panel, bx, ty, window, content_w)
    panel_h = bptt_bottom - panel_y + 28
    total_h = int(panel_y + panel_h + MARGIN)

    lines = svg_open(w, total_h)
    lines.extend(header)
    lines.append(
        f'<rect x="{MARGIN}" y="{panel_y}" width="{inner_w}" height="{panel_h:.0f}" rx="12" '
        f'fill="{PANEL_BG}" stroke="{PANEL_BORDER}" stroke-width="1.5"/>'
    )
    lines.extend(panel)
    return svg_close(lines)

'''


def patch_fig2_block(text: str) -> str:
    pattern = re.compile(
        r"def _layer_positions\(.*?def build_figure3\(\)",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise SystemExit("Could not find fig2 block to replace")
    return text[: m.start()] + REPLACEMENT.strip() + "\n\n\ndef build_figure3()" + text[m.end() :]


def add_version_guard(text: str) -> str:
    if "_assert_fig2_version" in text:
        return text
    guard = '''
def _assert_fig2_version() -> None:
    src = Path(__file__).read_text(encoding="utf-8")
    if "render_bptt_panel" not in src or "lx_prev" in src:
        raise RuntimeError(
            "generate_training_diagram.py regressed — run: python scripts/apply_fig2_fix.py"
        )


'''
    text = text.replace("def main() -> None:\n", guard + "def main() -> None:\n", 1)
    return text.replace(
        "def main() -> None:\n    out_dir = ",
        "def main() -> None:\n    _assert_fig2_version()\n    out_dir = ",
        1,
    )


def add_marker(text: str) -> str:
    if MARKER in text:
        return text
    return text.replace(
        '"""Generate slideshow SVGs: character RNN training (clean layout, no overlaps)."""\n',
        '"""Generate slideshow SVGs: character RNN training (clean layout, no overlaps)."""\n'
        f"{MARKER}\n",
        1,
    )


def normalize_utf8(path: Path) -> None:
    raw = path.read_bytes()
    if b"\x00" in raw:
        path.write_text(raw.decode("utf-16"), encoding="utf-8", newline="\n")


def main() -> None:
    normalize_utf8(Path(__file__))
    normalize_utf8(TARGET)
    text = read_text(TARGET)
    text = ensure_constants(text)
    text = ensure_colored_w(text)
    text = ensure_uniform_color(text)
    text = patch_fig2_block(text)
    text = add_version_guard(text)
    text = add_marker(text)
    write_text(TARGET, text)
    print("Patched", TARGET)
    subprocess.run([sys.executable, str(TARGET)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
