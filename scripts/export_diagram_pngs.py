"""Export trie and DFA SVGs as numbered PNGs for README embedding."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath

from experiment import shared_dir
from readme_figures import SHARED_FIGURE_NUMBERS, numbered_plot_path
from task import REGIMES
from vocab_diagrams import (
    BG_COLOR,
    LINE_H,
    MIN_NODE_R,
    _canvas_size,
    _compute_radii,
    _edge_curve_points,
    _fit_state,
    _gap_scale,
    _scale_trie_positions,
    _state_font_size,
    _state_line_step,
    build_minimized_vocabulary_automaton,
    build_trie,
    dfa_canvas_size,
    draw_minimized_dfa_on_axes,
    layout_trie,
    trie_prefixes,
    trie_states,
)


def export_trie_png(words: list[str], out_path: Path) -> None:
    root = build_trie(words)
    nodes, _index = trie_states(root)
    prefixes = trie_prefixes(root)
    state_labels = {id(n): {prefixes[id(n)]} for n in nodes}
    radii = _compute_radii(state_labels)
    gap_scale = _gap_scale(radii)
    coords = _scale_trie_positions(layout_trie(root), gap_scale=gap_scale)
    width, height = _canvas_size(coords, radii)

    fig, ax = plt.subplots(figsize=(width / 80, height / 80), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Trie · " + ", ".join(words), fontsize=12, pad=12)

    for node in nodes:
        nid = id(node)
        if nid not in coords:
            continue
        cx, cy = coords[nid]
        r = radii.get(nid, MIN_NODE_R)
        ax.add_patch(
            Circle((cx, cy), r, fill=False, edgecolor="#333", lw=1.4, zorder=5)
        )
        if node.terminal:
            ax.add_patch(
                Circle((cx, cy), r - 3, fill=False, edgecolor="#333", lw=1.0, zorder=5)
            )
        label = prefixes[nid] or "ε"
        wrapped, _ = _fit_state({label} if label != "ε" else set())
        fs = _state_font_size(r, len(wrapped))
        line_step = _state_line_step(len(wrapped))
        if len(wrapped) == 1:
            ax.text(cx, cy, wrapped[0], fontsize=fs, ha="center", va="center", zorder=6)
        else:
            y0 = cy - (len(wrapped) - 1) * line_step / 2
            for i, line in enumerate(wrapped):
                ax.text(cx, y0 + i * line_step, line, fontsize=fs, ha="center", va="center", zorder=6)

    for node in nodes:
        nid = id(node)
        if nid not in coords:
            continue
        sx, sy = coords[nid]
        rs = radii.get(nid, MIN_NODE_R)
        for ch, child in sorted(node.children.items()):
            cid = id(child)
            if cid not in coords:
                continue
            tx, ty = coords[cid]
            rt = radii.get(cid, MIN_NODE_R)
            start, control, end, _, _ = _edge_curve_points(sx, sy, tx, ty, rs, rt)
            if control is None:
                patch = FancyArrowPatch(
                    start, end, arrowstyle="->", color="#444", lw=1.2,
                    mutation_scale=12, zorder=2,
                )
            else:
                verts = [start, control, end]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                patch = FancyArrowPatch(
                    MplPath(verts, codes), arrowstyle="->", color="#444", lw=1.2,
                    mutation_scale=12, zorder=2,
                )
            ax.add_patch(patch)
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2
            ax.text(mx, my - 8, ch, fontsize=9, ha="center", va="center", color="#222", zorder=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def export_dfa_png(words: list[str], out_path: Path) -> None:
    automaton = build_minimized_vocabulary_automaton(words)
    width, height = dfa_canvas_size(automaton)
    fig, ax = plt.subplots(figsize=(width / 80, height / 80), facecolor=BG_COLOR)
    draw_minimized_dfa_on_axes(ax, automaton, words)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    exp_name = "ten_word_overlap_s"
    words = REGIMES["ten_word_overlap"]
    plots = shared_dir(exp_name)
    trie_path = numbered_plot_path(plots, "vocabulary_trie.png")
    dfa_path = numbered_plot_path(plots, "vocabulary_min_dfa.png")
    export_trie_png(words, trie_path)
    export_dfa_png(words, dfa_path)
    print(f"wrote {trie_path}")
    print(f"wrote {dfa_path}")


if __name__ == "__main__":
    main()
