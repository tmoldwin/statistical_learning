"""Compare DFA/position/char sensitivity gaps across the micro curriculum."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (
    MICRO_CURRICULUM,
    MODEL_TYPES,
    micro_curriculum_repr_label,
    micro_curriculum_viz_dir,
    model_path,
    plots_dir,
    spaced_experiment_name,
)
from scripts.analyze_transformer_dfa_sensitivity import (
    analyze_rnn,
    analyze_transformer,
)
from task import REGIMES

BLOCK_OUTPUT_SLUG = "block_output"
GAP_KEYS = ("dfa_gap", "char_gap", "pos_gap")
GAP_LABELS = ("DFA gap", "char gap", "pos gap")


def _curriculum_experiment(regime: str, *, spaced: bool) -> str:
    return spaced_experiment_name(regime) if spaced else regime


def _summary_row(regime: str, exp: str, block: dict) -> dict:
    return {
        "regime": regime,
        "exp": exp,
        "words": REGIMES[regime],
        **{k: block[k] for k in GAP_KEYS},
        "within_dfa_median": block["Within DFA state median"],
        "between_dfa_median": block["Between DFA states median"],
    }


def _block_output_row(rows: list[dict]) -> dict | None:
    for row in rows:
        if row["slug"] == BLOCK_OUTPUT_SLUG:
            return row
    return None


def collect_rows(*, spaced: bool, model_type: str, metric: str = "l2") -> list[dict]:
    out: list[dict] = []
    for regime in MICRO_CURRICULUM:
        exp = _curriculum_experiment(regime, spaced=spaced)
        if not model_path(exp, model_type).is_file():
            print(f"skip {exp}: no {model_type} checkpoint at {model_path(exp, model_type)}")
            continue
        if model_type == "rnn":
            block = analyze_rnn(exp, metric=metric)
            if block is None:
                continue
        else:
            rows = analyze_transformer(exp, metric=metric)
            block = _block_output_row(rows)
            if block is None:
                print(f"skip {exp}: no block_output row")
                continue
        out.append(_summary_row(regime, exp, block))
    return out


def plot_curriculum(
    rows: list[dict],
    out_path: Path,
    *,
    spaced: bool,
    model_type: str,
) -> None:
    if not rows:
        print("no rows to plot")
        return

    labels = [f"{r['regime']}\n{r['words']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    for i, (key, label) in enumerate(zip(GAP_KEYS, GAP_LABELS)):
        vals = [r[key] for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median distance gap (between − within)")
    spacing_tag = "_s" if spaced else "_ns"
    repr_label = micro_curriculum_repr_label(model_type)
    ax.set_title(
        f"{model_type} {repr_label} sensitivity across micro curriculum "
        f"({spacing_tag} regimes)",
    )
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.axhline(0, color="0.3", linewidth=0.8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-word-space",
        action="store_true",
        help="use unspaced micro-curriculum experiments (output under _ns)",
    )
    parser.add_argument(
        "--model-type",
        default="rnn",
        choices=list(MODEL_TYPES),
        help="which model checkpoints to analyze (default: rnn)",
    )
    args = parser.parse_args()
    spaced = not args.no_word_space

    rows = collect_rows(spaced=spaced, model_type=args.model_type, metric="l2")
    out_dir = micro_curriculum_viz_dir(spaced=spaced, model_type=args.model_type, kind="dfa_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "curriculum.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    plot_curriculum(
        rows,
        out_dir / "curriculum.png",
        spaced=spaced,
        model_type=args.model_type,
    )

    from scripts.analyze_transformer_dfa_sensitivity import plot, plot_heatmaps

    for regime in MICRO_CURRICULUM:
        exp = _curriculum_experiment(regime, spaced=spaced)
        if not model_path(exp, args.model_type).is_file():
            continue
        if args.model_type == "rnn":
            exp_rows_l2 = [row for row in [analyze_rnn(exp, metric="l2")] if row is not None]
            exp_rows_cos = [row for row in [analyze_rnn(exp, metric="cosine")] if row is not None]
        else:
            exp_rows_l2 = analyze_transformer(exp, metric="l2")
            exp_rows_cos = analyze_transformer(exp, metric="cosine")
        if not exp_rows_l2:
            continue
        exp_plots_dir = plots_dir(exp, args.model_type)
        exp_plots_dir.mkdir(parents=True, exist_ok=True)
        plot(exp_rows_l2, exp_plots_dir / "dfa_position_sensitivity_by_representation.png")
        if exp_rows_cos:
            plot_heatmaps(
                exp_rows_l2,
                exp_rows_cos,
                exp_plots_dir / "dfa_position_sensitivity_heatmap.png",
            )


if __name__ == "__main__":
    main()
