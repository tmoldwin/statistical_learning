r"""Letter pyramid: pairwise transitions only (uniform random walk on the graph).

Same undirected graph as Tacikowski, Kalender, Ciliberti & Fried, Nature 2024,
Fig. 1b (six nodes; a step may only go to a graph neighbor). Letters replace
the images.

        a
       / \
      b---c
     / \ / \
    d---e---f

The learning task assumes **no** finite length-4 vocabulary and **no** walk DFA.
Training text is an i.i.d.-started, uniform-neighbor **Markov chain** on the
graph. Rollout metrics treat each legal directed edge as a length-2 "word"
(the only constraint the model needs to match).

Reference length-4 walks are still listed in ``walks()`` for comparison only.
This module builds data and diagrams; it does not train a model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Paper node numbers 1..6.
LETTERS = "abcdef"
EDGES: tuple[tuple[str, str], ...] = (
    ("a", "b"),
    ("a", "c"),
    ("b", "c"),
    ("b", "d"),
    ("b", "e"),
    ("c", "e"),
    ("c", "f"),
    ("d", "e"),
    ("e", "f"),
)
OUTER = "adf"
INNER = "bce"

# Normalized (x, y) layout for matplotlib panels (y up).
PYRAMID_NODE_XY: dict[str, tuple[float, float]] = {
    "a": (0.50, 0.92),
    "b": (0.32, 0.58),
    "c": (0.68, 0.58),
    "d": (0.16, 0.20),
    "e": (0.50, 0.20),
    "f": (0.84, 0.20),
}

# Legacy walk length (reference walks only; not used for training).
REFERENCE_WALK_LENGTH = 4
REGIME = "pyramid_letters"
TASK = "pyramid_letters"
CORPUS_CHARS = 20_000
BPTT_SEQUENCE_LENGTH = 8


def adjacency() -> dict[str, tuple[str, ...]]:
    nbrs: dict[str, set[str]] = {ch: set() for ch in LETTERS}
    for u, v in EDGES:
        if u == v:
            raise ValueError(f"self-loop is not a pyramid edge: {u}")
        nbrs[u].add(v)
        nbrs[v].add(u)
    return {ch: tuple(sorted(nbrs[ch])) for ch in LETTERS}


def legal_bigrams() -> list[str]:
    """Every directed edge u→v (both directions on the undirected graph)."""
    adj = adjacency()
    out = [u + v for u in LETTERS for v in adj[u]]
    out.sort()
    return out


def generate_markov_sequence(num_chars: int, seed: int = 0) -> str:
    """Uniform random walk: random start letter, then uniform among neighbors."""
    if num_chars <= 0:
        return ""
    rng = random.Random(seed)
    adj = adjacency()
    current = rng.choice(LETTERS)
    chars = [current]
    while len(chars) < num_chars:
        current = rng.choice(adj[current])
        chars.append(current)
    return "".join(chars)


def validate_markov_corpus(text: str) -> None:
    adj = adjacency()
    if not text or any(ch not in LETTERS for ch in text):
        raise RuntimeError("corpus must be non-empty and use only a–f")
    bad = [
        (i, text[i], text[i + 1])
        for i in range(len(text) - 1)
        if text[i + 1] not in adj[text[i]]
    ]
    if bad:
        i, u, v = bad[0]
        raise RuntimeError(f"illegal transition at {i}: {u}->{v}")


def walks(length: int = REFERENCE_WALK_LENGTH) -> list[str]:
    """Every walk of `length` letters (reference only; not the training regime)."""
    adj = adjacency()
    found: list[str] = []

    def extend(prefix: str) -> None:
        if len(prefix) == length:
            found.append(prefix)
            return
        options = LETTERS if not prefix else adj[prefix[-1]]
        for ch in options:
            extend(prefix + ch)

    extend("")
    found.sort()
    return found


def pyramid_task_config() -> dict[str, object]:
    """Task entry for the Markov pyramid corpus."""
    bigrams = legal_bigrams()
    return {
        "regime": REGIME,
        "word_space": False,
        "chars": int(CORPUS_CHARS),
        "steps": 15_000,
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": 1_000,
        "viz_length": 96,
        "demo_snippet_len": 64,
        "hidden_size": 50,
        "sequence_length": int(BPTT_SEQUENCE_LENGTH),
        "eval_interval": 50,
        "eval_iterations": 20,
        "metric_rollout_len": 200,
        "train_ratio": 0.9,
        "dropout": 0.15,
        "l2_lambda": 1e-4,
        "sweep_n_words": len(bigrams),
        "sweep_length": 2,
    }


def register_pyramid_regimes(regimes: dict[str, list[str]]) -> None:
    regimes[REGIME] = legal_bigrams()


def register_pyramid_tasks(tasks: dict[str, dict]) -> None:
    tasks[TASK] = pyramid_task_config()


def _check_graph() -> dict[str, tuple[str, ...]]:
    adj = adjacency()
    if set(adj) != set(LETTERS):
        raise RuntimeError(f"pyramid letters {sorted(adj)} != {list(LETTERS)}")
    for u, vs in adj.items():
        for v in vs:
            if u not in adj[v]:
                raise RuntimeError(f"edge {u}-{v} is not symmetric")
            if u == v:
                raise RuntimeError(f"self-loop at {u}")
    degrees = {ch: len(adj[ch]) for ch in LETTERS}
    if any(degrees[ch] != 2 for ch in OUTER) or any(degrees[ch] != 4 for ch in INNER):
        raise RuntimeError(f"unexpected degrees {degrees}")
    return adj


def _pyramid_svg() -> str:
    pos = {
        "a": (220, 56),
        "b": (140, 150),
        "c": (300, 150),
        "d": (60, 244),
        "e": (220, 244),
        "f": (380, 244),
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="440" height="320" viewBox="0 0 440 320">',
        '<rect width="440" height="320" fill="#f4f1ea"/>',
        '<text x="220" y="28" text-anchor="middle" font-family="Georgia, serif" '
        'font-size="16" fill="#222">Letter pyramid (pairwise transitions)</text>',
    ]
    for u, v in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#222" stroke-width="2"/>'
        )
    for ch, (x, y) in pos.items():
        lines.append(f'<circle cx="{x}" cy="{y}" r="22" fill="#fff" stroke="#111" stroke-width="1.6"/>')
        lines.append(
            f'<text x="{x}" y="{y + 1}" text-anchor="middle" dominant-baseline="central" '
            f'font-family="ui-monospace, Consolas, monospace" font-size="18" fill="#111">{ch}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def graph_state_id(letter: str) -> int:
    return LETTERS.index(letter)


def training_graph_states(text: str) -> list[tuple[int, str, int]]:
    """Per character: (index, letter, graph state id = current node)."""
    validate_markov_corpus(text)
    return [(i, ch, graph_state_id(ch)) for i, ch in enumerate(text)]


def _state_colors(n_states: int) -> dict[int, tuple]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10", max(n_states, 1))
    return {state: cmap(state) for state in range(n_states)}


def _write_training_graph_state_figure(
    rows: list[tuple[int, str, int]],
    out_path: Path,
    *,
    n_chars: int = 256,
    per_row: int = 32,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    shown = rows[:n_chars]
    n_rows = (len(shown) + per_row - 1) // per_row
    colors = _state_colors(len(LETTERS))
    fig_w = per_row * 0.42 + 0.6
    fig_h = n_rows * 0.55 + 0.85
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.2, per_row * 0.42)
    ax.set_ylim(-0.15, n_rows * 0.55)
    ax.axis("off")
    ax.set_aspect("equal")

    for i, (_idx, ch, state) in enumerate(shown):
        row = n_rows - 1 - (i // per_row)
        col = i % per_row
        x = col * 0.42
        y0 = row * 0.55
        rgba = colors[state]
        ax.add_patch(
            FancyBboxPatch(
                (x, y0 + 0.08),
                0.36,
                0.36,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=rgba,
                edgecolor="#222222",
                linewidth=0.5,
                mutation_aspect=1,
            )
        )
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        ax.text(
            x + 0.18,
            y0 + 0.26,
            ch,
            ha="center",
            va="center",
            fontsize=11,
            fontfamily="monospace",
            color="#111111" if lum > 0.62 else "#ffffff",
        )
    ax.set_title(
        "Graph node (current letter) along Markov training stream",
        fontsize=13,
        pad=8,
        loc="left",
    )
    fig.text(
        0.012,
        0.02,
        "Colors = pyramid node a–f. Only pairwise adjacency constrains the generative process.",
        fontsize=8,
        color="#444444",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_training_dfa_states() -> dict[str, object]:
    """Label the pyramid Markov corpus with graph-node ids (current letter)."""
    from experiment import input_path, shared_dir

    text = input_path(TASK).read_text(encoding="utf-8")
    rows = training_graph_states(text)

    out_dir = shared_dir(TASK)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "training_dfa_states.txt"
    lines = ["index\tletter\tgraph_state"]
    lines.extend(f"{idx}\t{ch}\t{state}" for idx, ch, state in rows)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_path = out_dir / "training_dfa_states.png"
    _write_training_graph_state_figure(rows, figure_path)
    return {
        "n_chars": len(rows),
        "n_graph_states": len(LETTERS),
        "table": str(table_path),
        "figure": str(figure_path),
        "preview": [
            {"index": idx, "letter": ch, "graph_state": state}
            for idx, ch, state in rows[:12]
        ],
    }


def write_dataset() -> dict[str, object]:
    """Write Markov corpus, pyramid drawing, and reference diagrams."""
    from experiment import DEFAULT_SEED, input_path, shared_dir
    from task import REGIMES
    from vocab_diagrams import write_vocabulary_diagrams_for_experiment

    adj = _check_graph()
    bigrams = legal_bigrams()
    if REGIMES.get(REGIME) != bigrams:
        raise RuntimeError(f"regime {REGIME} is not the pyramid bigram list")

    text = generate_markov_sequence(CORPUS_CHARS, seed=DEFAULT_SEED)
    validate_markov_corpus(text)
    if len(text) != CORPUS_CHARS:
        raise RuntimeError(f"corpus length {len(text)} is not {CORPUS_CHARS}")

    corpus_path = input_path(TASK)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(text, encoding="utf-8")

    trie_path, dfa_path = write_vocabulary_diagrams_for_experiment(TASK)
    out_dir = shared_dir(TASK)
    pyramid_path = out_dir / "pyramid.svg"
    pyramid_path.write_text(_pyramid_svg(), encoding="utf-8")
    bigrams_path = out_dir / "legal_bigrams.txt"
    bigrams_path.write_text("\n".join(bigrams) + "\n", encoding="utf-8")
    walks_path = out_dir / "reference_walks_length4.txt"
    ref_walks = walks()
    walks_path.write_text("\n".join(ref_walks) + "\n", encoding="utf-8")

    manifest = {
        "task": TASK,
        "regime": REGIME,
        "generative_model": "uniform_markov_random_walk_on_pyramid_graph",
        "source": "Tacikowski, Kalender, Ciliberti & Fried, Nature 2024, Fig. 1b (graph only)",
        "letters": LETTERS,
        "paper_node_index": {ch: i + 1 for i, ch in enumerate(LETTERS)},
        "outer": list(OUTER),
        "inner": list(INNER),
        "edges": [list(e) for e in EDGES],
        "adjacency": {ch: list(vs) for ch, vs in adj.items()},
        "n_legal_bigrams": len(bigrams),
        "legal_bigrams": bigrams,
        "reference_walk_length": REFERENCE_WALK_LENGTH,
        "n_reference_walks": len(ref_walks),
        "reference_walks": ref_walks,
        "bptt_sequence_length": BPTT_SEQUENCE_LENGTH,
        "corpus": {
            "path": str(corpus_path),
            "chars": len(text),
            "seed": int(DEFAULT_SEED),
            "word_space": False,
            "preview": text[:64],
        },
        "diagrams": {
            "pyramid": str(pyramid_path),
            "trie": str(trie_path),
            "min_dfa": str(dfa_path),
            "legal_bigrams": str(bigrams_path),
            "reference_walks": str(walks_path),
        },
    }
    state_info = write_training_dfa_states()
    manifest["diagrams"]["training_dfa_states"] = state_info["table"]
    manifest["diagrams"]["training_dfa_states_figure"] = state_info["figure"]
    manifest_path = out_dir / "pyramid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["diagrams"]["manifest"] = str(manifest_path)
    manifest["training_graph_states"] = state_info
    return manifest


def main() -> None:
    manifest = write_dataset()
    corpus = manifest["corpus"]
    print("Pyramid (pairwise Markov on graph):")
    print("      a")
    print("     / \\")
    print("    b---c")
    print("   / \\ / \\")
    print("  d---e---f")
    print(f"Legal directed bigrams: {manifest['n_legal_bigrams']}")
    print(f"Reference length-{manifest['reference_walk_length']} walks: {manifest['n_reference_walks']}")
    print(f"Corpus: {corpus['path']} ({corpus['chars']} chars, seed {corpus['seed']})")
    print(f"Preview: {corpus['preview']}")
    print(f"DFA (bigram trie): {manifest['diagrams']['min_dfa']}")
    states = manifest["training_graph_states"]
    print(f"Graph states on training data: {states['table']}")
    print(f"Figure: {states['figure']}")


if __name__ == "__main__":
    main()
