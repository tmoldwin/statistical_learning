"""Restore timestep+prefix activation heatmaps, refresh decoding panel, copy to paper/figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task import corpus_for_experiment, label_extensions_for_experiment  # noqa: E402
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
    plot_hidden_states_clustermap,
    plot_hidden_states_heatmap,
    resolve_paths,
    run_forward_pass,
)
from viz.single_task_decoding import plot_aggregated_seed_decode_curves  # noqa: E402

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
    print(f"paper <- {dest.relative_to(ROOT)} ({dest.stat().st_size // 1024} KB)")


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
    _, text, _label_words = select_analysis_window(
        full_text, words, 80, spaced=spaced, extensions=extensions,
    )
    hidden_states, output_probs = run_forward_pass(model, text, model_type)
    automaton = build_minimized_vocabulary_automaton(words)
    condensed = condense_hidden_states_by_prefix(
        text, hidden_states, output_probs, spaced=spaced, words=words,
    )
    act_label = "relu" if model.get("use_relu") else "tanh"

    # 1) Timestep heatmap with actual input letters
    heat = Path(numbered_plot_path(out_dir, "activation_heatmap.png"))
    plot_hidden_states_heatmap(
        text, hidden_states,
        save_path=str(heat),
        act_label=act_label,
        condensed=None,
        exp_name=EXP,
        automaton=automaton,
        spaced=spaced,
        words=words,
        cluster_units=True,
    )

    # 2) Prefix-clustered heatmap
    clustered = Path(numbered_plot_path(out_dir, "activation_clustered_heatmap.png"))
    plot_hidden_states_clustermap(
        text, hidden_states, model["chars"],
        save_path=str(clustered),
        exp_name=EXP,
        condensed=condensed,
        automaton=automaton,
        spaced=spaced,
    )

    # 3) Decoding seed-mean (reuse existing by-seed JSON if present)
    decode_json = out_dir / "decoding" / "decoding_curves_by_seed.json"
    decode_png = out_dir / "decoding" / "decoding_curves_seed_mean.png"
    if decode_json.is_file():
        payload = json.loads(decode_json.read_text(encoding="utf-8"))
        panels = {int(k): v for k, v in payload.get("panels", payload).items() if str(k).isdigit()}
        if not panels and "seeds" in payload:
            # alternate schema: list of panels
            panels = {int(p["seed"]): p for p in payload["seeds"]}
        if panels:
            plot_aggregated_seed_decode_curves(
                panels, decode_png, task=EXP,
            )

    copies = {
        heat: "fig_activation_heatmap.jpg",
        clustered: "fig_activation_clustered.jpg",
        decode_png: "fig_decoding_seed_mean.jpg",
    }
    for src, name in copies.items():
        if not src.is_file():
            print(f"MISSING {src}")
            continue
        _to_jpg(src, PAPER_MAIN / name)
        if name == "fig_activation_heatmap.jpg":
            _to_jpg(src, PAPER_MAIN / "fig07_activation_heatmap.jpg")
        if name == "fig_activation_clustered.jpg":
            _to_jpg(src, PAPER_MAIN / "fig10_activation_clustered.jpg")
        if name == "fig_decoding_seed_mean.jpg":
            png_dest = PAPER_MAIN / "fig_decoding_seed_mean.png"
            png_dest.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
