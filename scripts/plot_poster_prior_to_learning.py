"""Poster Figs 9–10 from the demo RNN at initialization (before any update).

Same probes as the trained panels:
  Fig 9  example selective units (one unit per feature, shared corpus window)
  Fig 10 linear decoding across seeds, plus the init unit-SI density

Init weights match min_char_rnn.py (Generator(seed), scale 0.01, zero biases).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiment import TASKS, plots_dir
from unit_selectivity import (
    WORD_ANALYSIS_FEATURES,
    build_timestep_labels,
    compute_selectivity,
    compute_unit_selectivity_matrix,
    plot_example_units_combined,
)
from viz.compare._data import load_task_decoding_context, load_task_viz_context
from viz.compare.decoding import WORD_DECODING_FEATURES, compute_panel_decoding
from viz.single_task_decoding import plot_aggregated_seed_decode_curves
from visualize import (
    condense_hidden_states_by_prefix,
    hidden_unit_labels,
    run_forward_pass,
)
from vocab_diagrams import build_minimized_vocabulary_automaton

TASK = "eight_word_ate_at_demo_ns"
SEEDS = (1, 2, 3, 5, 7, 8)
# Trained poster Fig 9 window (first 28 letters of the analysis snippet).
POSTER_WINDOW = "platelategatelatematmatmatha"


def initial_model(model: dict, seed: int) -> dict:
    """Recreate the untrained weights for this checkpoint's seed."""
    rng = np.random.default_rng(int(seed))
    hidden = int(model["hidden_size"])
    vocab = int(model["vocab_size"])
    out = dict(model)
    out["weights_input_to_hidden"] = rng.standard_normal((hidden, vocab)) * 0.01
    out["weights_hidden_to_hidden"] = rng.standard_normal((hidden, hidden)) * 0.01
    out["weights_hidden_to_output"] = rng.standard_normal((vocab, hidden)) * 0.01
    out["bias_hidden"] = np.zeros((hidden, 1))
    out["bias_output"] = np.zeros((vocab, 1))
    out["timestep_noise_std"] = 0.0
    out["dale_sign"] = None
    return out


def _at_init(ctx, seed: int):
    model = initial_model(ctx.model, seed)
    hidden, _probs = run_forward_pass(model, ctx.text, "rnn")
    ctx.model = model
    ctx.hidden_states = np.asarray(hidden)
    return ctx


def _exemplar_seed() -> int:
    for seed in SEEDS:
        ctx = load_task_viz_context(TASK, seed=seed)
        if ctx.text.startswith(POSTER_WINDOW):
            return seed
    return SEEDS[0]


def _si_arrays(contexts: dict[int, object]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {feat: [] for feat in WORD_DECODING_FEATURES}
    for ctx in contexts.values():
        automaton = build_minimized_vocabulary_automaton(ctx.words)
        condensed = condense_hidden_states_by_prefix(
            ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=ctx.words,
        )
        if condensed.hidden_states.shape[0] < 2:
            continue
        labels = build_timestep_labels(
            ctx.text, automaton,
            spaced=ctx.spaced, words=ctx.words, condensed=condensed,
            model=None, activations=None,
        )
        si, *_rest = compute_unit_selectivity_matrix(
            condensed.hidden_states, labels, features=WORD_DECODING_FEATURES,
        )
        for feat in WORD_DECODING_FEATURES:
            vals = np.asarray(si.get(feat, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                buckets[feat].append(vals)
    return {
        feat: np.concatenate(parts)
        for feat, parts in buckets.items()
        if parts
    }


def _to_jpg(src: Path, dest: Path, max_w: int) -> None:
    im = Image.open(src)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h * max_w / w)), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=90, optimize=True)
    print(f"wrote {dest}")


def plot_example_units(seed: int, save_path: Path) -> None:
    ctx = _at_init(load_task_viz_context(TASK, seed=seed), seed)
    automaton = build_minimized_vocabulary_automaton(ctx.words)
    text = ctx.text
    chrono = np.asarray(ctx.hidden_states)
    chrono_labels = build_timestep_labels(
        text, automaton,
        spaced=ctx.spaced, words=ctx.words, condensed=None,
        model=ctx.model, activations=chrono,
    )
    condensed = condense_hidden_states_by_prefix(
        text, chrono, spaced=ctx.spaced, words=ctx.words,
    )
    labels = build_timestep_labels(
        text, automaton,
        spaced=ctx.spaced, words=ctx.words, condensed=condensed,
        model=ctx.model, activations=condensed.hidden_states,
    )
    result = compute_selectivity(condensed.hidden_states, labels, ctx.model, text)
    word_si, *_rest = compute_unit_selectivity_matrix(
        condensed.hidden_states, labels, features=("word",),
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_example_units_combined(
        result, chrono, chrono_labels,
        str(save_path),
        unit_labels=hidden_unit_labels(None, int(ctx.model["hidden_size"])),
        text=text,
        model=ctx.model,
        automaton=automaton,
        features=WORD_ANALYSIS_FEATURES,
        k=1,
        rank_si={**result.si, **word_si},
    )


def plot_decoding(save_path: Path) -> None:
    viz_ctx: dict[int, object] = {}
    panels: dict[int, dict] = {}
    for seed in SEEDS:
        print(f"prior-to-learning decode seed {seed}", flush=True)
        viz_ctx[seed] = _at_init(load_task_viz_context(TASK, seed=seed), seed)
        decode_ctx = _at_init(
            load_task_decoding_context(TASK, seed=seed), seed,
        )
        panels[seed] = compute_panel_decoding(
            decode_ctx,
            neuron_sampling="random",
            features=WORD_DECODING_FEATURES,
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_aggregated_seed_decode_curves(
        panels,
        save_path,
        task=TASK,
        features=WORD_DECODING_FEATURES,
        si_by_feat=_si_arrays(viz_ctx),
        load_summaries=False,
    )


def main() -> None:
    seed = _exemplar_seed()
    print(f"example-unit seed {seed} (viz_length={TASKS[TASK]['viz_length']})", flush=True)
    out = plots_dir(TASK, "rnn")
    units_png = out / "unit_selectivity" / "example_units_combined_prior.png"
    decode_png = out / "decoding" / "decoding_with_word_seed_mean_prior.png"
    plot_example_units(seed, units_png)
    plot_decoding(decode_png)
    paper = REPO / "paper" / "figures" / "demo"
    _to_jpg(units_png, paper / "fig_example_units_prior.jpg", max_w=1193)
    _to_jpg(decode_png, paper / "fig_decoding_with_word_prior.jpg", max_w=1671)


if __name__ == "__main__":
    main()
