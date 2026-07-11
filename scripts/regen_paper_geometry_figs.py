"""Regenerate paper figures for sections 3.3–3.6 and copy into paper/figures/main."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task import (  # noqa: E402
    corpus_for_experiment,
    label_extensions_for_experiment,
)
from vocab_diagrams import (  # noqa: E402
    build_minimized_vocabulary_automaton,
    select_analysis_window,
    vocabulary_for_experiment,
)
from visualize import (  # noqa: E402
    condense_hidden_states_by_prefix,
    corpus_uses_word_spacing,
    load_model_for_viz,
    numbered_plot_path,
    plot_closed_loop_trajectory_panel,
    plot_hidden_states_clustermap,
    plot_hidden_states_heatmap,
    plot_pca_context_labels,
    resolve_paths,
    run_forward_pass,
)

EXP = "sixteen_word_four_letter_ns"
SEED = 2
PAPER_MAIN = ROOT / "paper" / "figures" / "main"


def _to_jpg(src: Path, dest: Path, *, max_w: int = 1600, quality: int = 90) -> None:
    im = Image.open(src)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h * max_w / w)), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=quality, optimize=True)
    print(f"paper <- {dest.relative_to(ROOT)} ({dest.stat().st_size // 1024} KB) from {src.name}")


def main() -> None:
    class Args:
        exp = EXP
        model_type = "rnn"
        seed = SEED
        model = "model.npz"
        input = "input.txt"
        out_dir = None

    model_file, _input_file, out_dir, model_type = resolve_paths(Args())
    out_dir = Path(out_dir)
    model = load_model_for_viz(model_file, model_type)
    full_text = corpus_for_experiment(EXP, seed=SEED)
    spaced = corpus_uses_word_spacing(full_text, EXP)
    words = vocabulary_for_experiment(EXP)
    extensions = label_extensions_for_experiment(EXP)
    _, text, label_words = select_analysis_window(
        full_text, words, 80, spaced=spaced, extensions=extensions,
    )
    hidden_states, output_probs = run_forward_pass(model, text, model_type)
    automaton = build_minimized_vocabulary_automaton(words)
    condensed = condense_hidden_states_by_prefix(
        text, hidden_states, output_probs, spaced=spaced, words=words,
    )
    act_label = "relu" if model.get("use_relu") else "tanh"

    heat = Path(numbered_plot_path(out_dir, "activation_heatmap.png"))
    plot_hidden_states_heatmap(
        text, hidden_states,
        save_path=str(heat),
        act_label=act_label,
        condensed=condensed,
        exp_name=EXP,
        automaton=automaton,
        spaced=spaced,
        words=words,
        cluster_units=True,
    )

    clustered = Path(numbered_plot_path(out_dir, "activation_clustered_heatmap.png"))
    plot_hidden_states_clustermap(
        text, hidden_states, model["chars"],
        save_path=str(clustered),
        exp_name=EXP,
        condensed=condensed,
        automaton=automaton,
        spaced=spaced,
    )

    embed = Path(numbered_plot_path(out_dir, "embedding_panels_context.png"))
    plot_pca_context_labels(
        text, hidden_states, model["chars"],
        str(embed),
        spaced=spaced,
        automaton=automaton,
        condensed=condensed,
        words=words,
        label_words=label_words,
        annot_style="leaders",
        embed_method="pca",
    )

    closed = Path(numbered_plot_path(out_dir, "word_trajectories_closed_loop.png"))
    plot_closed_loop_trajectory_panel(
        text, hidden_states, str(closed),
        model=model,
        spaced=spaced,
        words=words,
        condensed=condensed,
        automaton=automaton,
        embed_method="pca",
    )

    copies = {
        heat: "fig_activation_heatmap.jpg",
        clustered: "fig_activation_clustered.jpg",
        embed: "fig11_embedding_panels.jpg",
        closed: "fig_word_trajectories.jpg",
    }
    for src, name in copies.items():
        if not src.is_file():
            # embed_save_path may rewrite basename for method variants
            matches = sorted(out_dir.rglob(src.name))
            matches += sorted(out_dir.rglob(src.stem + "*.png"))
            src = next((p for p in matches if p.is_file() and "jpca" not in p.name.lower()), src)
        if not src.is_file():
            print(f"MISSING {name} (looked for {src})")
            continue
        _to_jpg(src, PAPER_MAIN / name)

    aliases = {
        "fig_activation_heatmap.jpg": "fig07_activation_heatmap.jpg",
        "fig_activation_clustered.jpg": "fig10_activation_clustered.jpg",
        "fig_word_trajectories.jpg": "fig16_word_trajectories.jpg",
    }
    for src_name, dest_name in aliases.items():
        src = PAPER_MAIN / src_name
        if src.is_file():
            _to_jpg(src, PAPER_MAIN / dest_name)


if __name__ == "__main__":
    main()
