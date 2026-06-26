"""Compare DFA/position/char sensitivity gaps across the micro curriculum."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import MICRO_CURRICULUM, spaced_experiment_name
from scripts.analyze_transformer_dfa_sensitivity import analyze
from task import REGIMES

BLOCK_OUTPUT_SLUG = "block_output"
GAP_KEYS = ("dfa_gap", "char_gap", "pos_gap")
GAP_LABELS = ("DFA gap", "char gap", "pos gap")


def _curriculum_experiments() -> list[str]:
    return [spaced_experiment_name(regime) for regime in MICRO_CURRICULUM]


def _block_output_row(rows: list[dict]) -> dict | None:
    for row in rows:
        if row["slug"] == BLOCK_OUTPUT_SLUG:
            return row
    return None


def collect_rows(metric: str = "l2") -> list[dict]:
    out: list[dict] = []
    for regime in MICRO_CURRICULUM:
        exp = spaced_experiment_name(regime)
        model_path = REPO_ROOT / "experiments" / exp / "transformer" / "model.pt"
        if not model_path.is_file():
            print(f"skip {exp}: no transformer checkpoint at {model_path}")
            continue
        rows = analyze(exp, metric=metric)
        block = _block_output_row(rows)
        if block is None:
            print(f"skip {exp}: no block_output row")
            continue
        out.append({
            "regime": regime,
            "exp": exp,
            "words": REGIMES[regime],
            **{k: block[k] for k in GAP_KEYS},
            "within_dfa_median": block["Within DFA state median"],
            "between_dfa_median": block["Between DFA states median"],
        })
    return out


def plot_curriculum(rows: list[dict], out_path: Path) -> None:
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
    ax.set_title("Transformer block_output sensitivity across micro curriculum (_s regimes)")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.axhline(0, color="0.3", linewidth=0.8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    rows = collect_rows(metric="l2")
    out_dir = REPO_ROOT / "experiments" / "micro_curriculum_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dfa_sensitivity_curriculum.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    plot_curriculum(rows, out_dir / "dfa_sensitivity_curriculum.png")

    for regime in MICRO_CURRICULUM:
        exp = spaced_experiment_name(regime)
        exp_model = REPO_ROOT / "experiments" / exp / "transformer" / "model.pt"
        if not exp_model.is_file():
            continue
        from scripts.analyze_transformer_dfa_sensitivity import plot, plot_heatmaps

        exp_rows = analyze(exp, metric="l2")
        exp_rows_cos = analyze(exp, metric="cosine")
        plots_dir = REPO_ROOT / "experiments" / exp / "transformer" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot(exp_rows, plots_dir / "dfa_position_sensitivity_by_representation.png")
        plot_heatmaps(
            exp_rows,
            exp_rows_cos,
            plots_dir / "dfa_position_sensitivity_heatmap.png",
        )


if __name__ == "__main__":
    main()
