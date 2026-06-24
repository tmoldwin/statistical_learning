"""Summarize DFA / position / char pairwise distances across transformer reps (median + MAD)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from experiment import EXPERIMENT_CONFIG, input_path, model_path
from transformer.adapter import extract_transformer_activations
from transformer.viz_repr import collect_representation_specs
from vocab_diagrams import build_minimized_vocabulary_automaton, vocabulary_for_experiment
from visualize import (
    PAIR_DISTANCE_CATEGORY_ORDER,
    PAIR_DISTANCE_PALETTE,
    _corpus_vocab,
    dfa_state_at_position,
    load_model_for_viz,
    median_mad,
    position_in_word_at_index,
)

SHORT_NAMES = {
    "token_embedding": "token emb",
    "position_embedding": "pos emb",
    "block_input": "tok+pos",
    "layer0_attn_input": "attn in",
    "layer0_post_attn": "post attn",
    "layer0_post_ffwd": "post FFN",
    "layer0_query": "Q",
    "layer0_key": "K",
    "layer0_value": "V",
    "block_output": "output",
}

CATEGORIES = list(PAIR_DISTANCE_CATEGORY_ORDER)
PALETTE = PAIR_DISTANCE_PALETTE
CAT_SHORT = ["w DFA", "b DFA", "w pos", "b pos", "w chr", "b chr", "all"]


def _pair_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    if metric == "cosine":
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return 1.0 - float(np.dot(a, b) / (na * nb))
    raise ValueError(f"unknown metric: {metric}")


def pairwise_distance_groups(
    chars: list[str],
    vectors: np.ndarray,
    state_ids: list[int],
    position_ids: list[int | None],
    *,
    metric: str = "l2",
) -> dict[str, np.ndarray]:
    """Pairwise distances (i < j) for within/between DFA, position, char, and all pairs."""
    n = vectors.shape[0]
    groups: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}
    for i in range(n):
        for j in range(i + 1, n):
            dist = _pair_distance(vectors[i], vectors[j], metric)
            groups["All pairs"].append(dist)
            if state_ids[i] == state_ids[j]:
                groups["Within DFA state"].append(dist)
            else:
                groups["Between DFA states"].append(dist)
            if chars[i] == chars[j]:
                groups["Within char"].append(dist)
            else:
                groups["Between chars"].append(dist)
            pi, pj = position_ids[i], position_ids[j]
            if pi is not None and pj is not None:
                if pi == pj:
                    groups["Within word position"].append(dist)
                else:
                    groups["Between word positions"].append(dist)
    return {k: np.asarray(v) for k, v in groups.items()}


def _summarize_groups(
    groups: dict[str, np.ndarray],
    *,
    slug: str,
    name: str,
    metric: str,
) -> dict:
    row: dict = {"slug": slug, "name": name, "metric": metric}
    for cat in CATEGORIES:
        med, mad = median_mad(groups[cat])
        row[f"{cat} median"] = med
        row[f"{cat} mad"] = mad
        row[f"{cat} n"] = int(len(groups[cat]))
    all_med = row["All pairs median"]
    row["overall_pairwise_median"] = all_med
    row["overall_pairwise_mad"] = row["All pairs mad"]
    row["dfa_gap"] = row["Between DFA states median"] - row["Within DFA state median"]
    row["char_gap"] = row["Between chars median"] - row["Within char median"]
    row["pos_gap"] = row["Between word positions median"] - row["Within word position median"]
    row["dfa_ratio"] = (
        row["Within DFA state median"] / row["Between DFA states median"]
        if row["Between DFA states median"] > 0
        else float("nan")
    )
    return row


def analyze(exp: str = "ten_word_overlap_s", *, metric: str = "l2") -> list[dict]:
    cfg = EXPERIMENT_CONFIG[exp]
    text = input_path(exp).read_text(encoding="utf-8")[: cfg["viz_length"]]
    words = vocabulary_for_experiment(exp)
    spaced = True
    automaton = build_minimized_vocabulary_automaton(words)
    vocab = _corpus_vocab(text, words)

    model = load_model_for_viz(str(model_path(exp, "transformer")), "transformer")
    acts = extract_transformer_activations(model, text)
    specs = collect_representation_specs(acts)

    state_ids = [
        dfa_state_at_position(text, t, automaton, spaced=spaced, vocab=vocab)
        for t in range(len(text))
    ]
    pos_ids = [
        position_in_word_at_index(text, t, spaced=spaced, vocab=vocab)
        for t in range(len(text))
    ]

    rows: list[dict] = []
    for spec in specs:
        groups = pairwise_distance_groups(
            list(text), spec.vectors, state_ids, pos_ids, metric=metric,
        )
        rows.append(_summarize_groups(
            groups, slug=spec.slug, name=spec.display_name, metric=metric,
        ))
    return rows


def _normalize_to_overall_median(mat: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Divide each row by that representation's median all-pairs distance."""
    return mat / np.maximum(scales[:, None], 1e-12)


def plot_heatmaps(rows_l2: list[dict], rows_cos: list[dict], out_path: Path) -> None:
    """Heatmaps: category median / all-pairs median (bwr centered at 1.0)."""
    names = [SHORT_NAMES.get(r["slug"], r["slug"]) for r in rows_l2]

    def median_matrix(rows: list[dict]) -> np.ndarray:
        return np.array([[r[f"{cat} median"] for cat in CATEGORIES] for r in rows])

    def mad_matrix(rows: list[dict]) -> np.ndarray:
        return np.array([[r[f"{cat} mad"] for cat in CATEGORIES] for r in rows])

    def gap_matrix(rows: list[dict]) -> np.ndarray:
        return np.array([[r["dfa_gap"], r["char_gap"], r["pos_gap"]] for r in rows])

    def scales(rows: list[dict]) -> np.ndarray:
        return np.array([r["overall_pairwise_median"] for r in rows])

    raw_specs = [
        ("Euclidean (L2)", median_matrix(rows_l2), mad_matrix(rows_l2), gap_matrix(rows_l2), scales(rows_l2)),
        ("Cosine (1 − cos sim)", median_matrix(rows_cos), mad_matrix(rows_cos), gap_matrix(rows_cos), scales(rows_cos)),
    ]

    mean_norms = [
        _normalize_to_overall_median(med_raw, row_scale)
        for _, med_raw, _, _, row_scale in raw_specs
    ]
    mean_dev = max(float(np.max(np.abs(m - 1.0))) for m in mean_norms)
    mean_span = max(mean_dev, 0.2)
    mean_norm_cmap = TwoSlopeNorm(vmin=1.0 - mean_span, vcenter=1.0, vmax=1.0 + mean_span)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    gap_vmax = 1.5

    for col, (label, med_raw, mad_raw, gap_raw, row_scale) in enumerate(raw_specs):
        med_norm = mean_norms[col]
        gap_norm = _normalize_to_overall_median(gap_raw, row_scale)

        ax_mean = axes[0, col]
        im_m = ax_mean.imshow(med_norm, aspect="auto", cmap="bwr", norm=mean_norm_cmap)
        ax_mean.set_xticks(range(med_norm.shape[1]))
        ax_mean.set_xticklabels(CAT_SHORT, fontsize=8)
        ax_mean.set_yticks(range(len(names)))
        ax_mean.set_yticklabels(names, fontsize=9)
        ax_mean.set_title(f"{label}\n(category median / all-pairs median)", fontsize=10)
        for i in range(med_norm.shape[0]):
            for j in range(med_norm.shape[1]):
                ratio, raw_m, raw_s = med_norm[i, j], med_raw[i, j], mad_raw[i, j]
                txt_color = "white" if abs(ratio - 1.0) > 0.45 * mean_span else "0.15"
                ax_mean.text(
                    j, i, f"{ratio:.2f}\n({raw_m:.1f}±{raw_s:.1f})",
                    ha="center", va="center", fontsize=5, color=txt_color,
                )
        cbar_m = fig.colorbar(im_m, ax=ax_mean, fraction=0.046, pad=0.02)
        cbar_m.set_label("× all-pairs median (white = baseline)", fontsize=8)

        ax_gap = axes[1, col]
        gap_norm_cmap = TwoSlopeNorm(vmin=-gap_vmax, vcenter=0.0, vmax=gap_vmax)
        im_g = ax_gap.imshow(gap_norm, aspect="auto", cmap="bwr", norm=gap_norm_cmap)
        gap_labels = ["DFA gap", "char gap", "pos gap"]
        ax_gap.set_xticks(range(gap_norm.shape[1]))
        ax_gap.set_xticklabels(gap_labels, fontsize=9)
        ax_gap.set_yticks(range(len(names)))
        ax_gap.set_yticklabels(names, fontsize=9)
        ax_gap.set_title(f"{label} gaps (median between − within)", fontsize=10)
        for i in range(gap_norm.shape[0]):
            for j in range(gap_norm.shape[1]):
                ratio, raw = gap_norm[i, j], gap_raw[i, j]
                txt_color = "white" if abs(ratio) > 0.55 * gap_vmax else "0.15"
                ax_gap.text(
                    j, i, f"{ratio:.2f}\n({raw:.1f})",
                    ha="center", va="center", fontsize=6, color=txt_color,
                )
        cbar_g = fig.colorbar(im_g, ax=ax_gap, fraction=0.046, pad=0.02)
        cbar_g.set_label("× all-pairs median", fontsize=8)

    fig.suptitle(
        "Transformer pairwise distance sensitivity (median + MAD)\n"
        "(ten_word_overlap_s, 50 timesteps; blue < all-pairs median < red)",
        fontsize=11,
        y=1.02,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot(rows: list[dict], out_path: Path) -> None:
    names = [SHORT_NAMES.get(r["slug"], r["slug"]) for r in rows]
    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(rows))
    width = 0.11
    for i, cat in enumerate(CATEGORIES):
        meds = [r[f"{cat} median"] for r in rows]
        mads = [r[f"{cat} mad"] for r in rows]
        offset = (i - (n_cats - 1) / 2) * width
        ax.bar(
            x + offset, meds, width, yerr=mads, capsize=2,
            label=cat, color=PALETTE[cat], alpha=0.9, error_kw={"elinewidth": 1},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Pairwise distance (median ± MAD)")
    ax.set_title("Distance by representation and pair type\n(ten_word_overlap_s, 50 timesteps)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.set_ylim(bottom=0)

    ax2 = axes[1]
    bar_w = 0.22
    ax2.bar(x - bar_w, [r["dfa_gap"] for r in rows], bar_w, label="DFA gap", color="#dd8452")
    ax2.bar(x, [r["char_gap"] for r in rows], bar_w, label="Char gap", color="#55a868")
    ax2.bar(x + bar_w, [r["pos_gap"] for r in rows], bar_w, label="Pos gap", color="#8172b3")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right")
    ax2.set_ylabel("Median distance gap (between − within)")
    ax2.set_title("Sensitivity summaries")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax2.axhline(0, color="0.3", linewidth=0.8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    exp = "ten_word_overlap_s"
    rows_l2 = analyze(exp, metric="l2")
    rows_cos = analyze(exp, metric="cosine")
    out_dir = Path("experiments") / exp / "transformer" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_png = out_dir / "dfa_position_sensitivity_by_representation.png"
    plot(rows_l2, bar_png)

    heatmap_png = out_dir / "dfa_position_sensitivity_heatmap.png"
    plot_heatmaps(rows_l2, rows_cos, heatmap_png)

    json_path = bar_png.with_suffix(".json")
    json_path.write_text(json.dumps({"l2": rows_l2, "cosine": rows_cos}, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
