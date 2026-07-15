#!/usr/bin/env python3
"""Copy curated experiment plots into paper/figures/ (copy/convert only — never replot).

Replot via the canonical CLI, e.g.:
  python visualize.py --exp sixteen_word_four_letter_ns --seed 2 --only activations trajectories
  python -c "from viz.compare.trajectories import plot_dfa_and_trajectory_seed_grid as f; f('sixteen_word_four_letter_ns')"
Then sync:
  python scripts/paper_collect_figures.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "paper" / "figures"

ANCHOR = "sixteen_word_four_letter_ns"
DEMO = "six_word_mixed_demo_ns"
COMPARE = "comparisons"


@dataclass(frozen=True)
class FigureRef:
    section: str
    paper_name: str
    source: Path
    as_jpg: bool = True
    max_w: int = 1000


def _anchor(subpath: str) -> Path:
    return ROOT / "experiments" / ANCHOR / "rnn" / "plots" / subpath


def _demo(subpath: str) -> Path:
    return ROOT / "experiments" / DEMO / "rnn" / "plots" / subpath


def _demo_shared(name: str) -> Path:
    return ROOT / "experiments" / DEMO / "shared" / name


def _compare(kind: str, name: str) -> Path:
    return ROOT / "experiments" / COMPARE / "paper_seed_grids" / kind / name


def paper_manifest() -> list[FigureRef]:
    """Map draft.md figure names → experiment plot sources."""
    # Prefer comparison scripts' known outputs; fall back to older paths if needed.
    traj_by_length = ROOT / "experiments" / COMPARE / "paper_seed_grids" / "trajectories" / "closed_loop_by_length_16words.png"
    if not traj_by_length.is_file():
        traj_by_length = ROOT / "paper" / "figures" / "compare" / "fig_traj_by_length.jpg"
    traj_by_count = ROOT / "experiments" / COMPARE / "paper_seed_grids" / "trajectories" / "closed_loop_by_wordcount_4letter.png"
    if not traj_by_count.is_file():
        traj_by_count = ROOT / "paper" / "figures" / "compare" / "fig_traj_by_wordcount.jpg"

    return [
        FigureRef("demo", "fig01_trie.svg", _demo_shared("vocabulary_trie.svg"), as_jpg=False),
        FigureRef("demo", "fig01_trie.jpg",
                  ROOT / "paper" / "figures" / "demo" / "fig01_trie.jpg", as_jpg=False),
        FigureRef("demo", "fig02_dfa.svg", _demo_shared("vocabulary_min_dfa.svg"), as_jpg=False),
        FigureRef("demo", "fig02_dfa.jpg",
                  ROOT / "paper" / "figures" / "demo" / "fig02_dfa.jpg", as_jpg=False),
        FigureRef("demo", "fig03_learning_curve.jpg", _demo("training/3_learning_curve.png")),
        FigureRef("demo", "fig03_learning_with_samples.jpg",
                  _demo("training/learning_curve_with_samples.png")),
        FigureRef("demo", "fig04_corpus_stream.svg",
                  ROOT / "paper" / "figures" / "demo" / "fig04_corpus_stream.svg", as_jpg=False),
        FigureRef("demo", "fig04_corpus_stream.jpg",
                  ROOT / "paper" / "figures" / "demo" / "fig04_corpus_stream.jpg", as_jpg=False),
        FigureRef("demo", "fig04_samples.svg",
                  ROOT / "paper" / "figures" / "demo" / "fig04_samples.svg", as_jpg=False),
        FigureRef("demo", "fig04_samples.jpg",
                  ROOT / "paper" / "figures" / "demo" / "fig04_samples.jpg", as_jpg=False),
        # Demo narrative figs 3–10 (same six-word mixed-length vocab through trajectories).
        FigureRef("demo", "fig_next_char_probs.jpg",
                  _demo("activations/8_next_char_prob_sequence_condensed.png")),
        FigureRef("demo", "fig_activation_heatmap.jpg", _demo("activations/7_activation_heatmap.png")),
        FigureRef("demo", "fig_activation_clustered.jpg",
                  _demo("activations/10_activation_clustered_heatmap.png")),
        FigureRef("demo", "fig_state_correlation.jpg",
                  _demo("correlation/17_state_correlation_clustered_condensed.png")),
        FigureRef("demo", "fig_dfa_pca_geometry.jpg",
                  _demo("separation/dfa_pca_geometry_and_separation.png"),
                  max_w=1200),
        FigureRef("demo", "fig_example_units.jpg",
                  _demo("unit_selectivity/example_units_combined.png")),
        FigureRef("demo", "fig_decoding_seed_mean.jpg",
                  _demo("decoding/decoding_curves_seed_mean.png")),
        FigureRef("demo", "fig_word_trajectories.jpg",
                  _demo("trajectories/dfa_and_trajectory_by_seed.png"),
                  max_w=1600),
        # Leftovers / alt copies still useful for supplements.
        FigureRef("main", "fig11_embedding_panels.jpg",
                  _demo("embeddings/11_embedding_panels_context_condensed.png")),
        FigureRef("main", "fig20_feature_separation.jpg",
                  _demo("separation/feature_separation_summary_seed_mean.png")),
        FigureRef("main", "fig19_dfa_distance.jpg",
                  _demo("separation/19_dfa_state_distance_comparison_condensed.png")),
        FigureRef("main", "fig_word_trajectories_by_start.jpg",
                  _anchor("trajectories/closed_loop_run_seed_row.png")),
        FigureRef("compare", "fig_traj_by_length.jpg", traj_by_length,
                  as_jpg=traj_by_length.suffix.lower() != ".jpg"),
        FigureRef("compare", "fig_traj_by_wordcount.jpg", traj_by_count,
                  as_jpg=traj_by_count.suffix.lower() != ".jpg"),
        FigureRef("compare", "fig_mixed_traj_by_dfa.jpg",
                  ROOT / "experiments" / COMPARE / "mixed_vocab_dfa_ns"
                  / "trajectories" / "closed_loop_by_dfa_vocab_grid.png",
                  max_w=1400),
        FigureRef("compare", "fig_mixed_scaling_overview.jpg",
                  ROOT / "experiments" / COMPARE / "mixed_vocab_dfa_ns"
                  / "trajectories" / "scaling_overview.png",
                  max_w=1400),
        FigureRef("compare", "fig_mixed_decoding_curves.jpg",
                  ROOT / "experiments" / COMPARE / "mixed_vocab_dfa_ns"
                  / "decoding" / "decoding_curves_by_dfa.png",
                  max_w=1200),
        # Leftover metrics board (not in draft; weight scatters live on fig_weight_matrices).
        FigureRef("compare", "fig_mixed_metrics_vs_dfa.jpg",
                  ROOT / "experiments" / COMPARE / "mixed_vocab_dfa_ns"
                  / "trajectories" / "metrics_vs_dfa.png",
                  max_w=1200),
        # Standalone leftovers (sources still produced; not in main draft).
        FigureRef("compare", "fig_mixed_pc_spectra.jpg",
                  ROOT / "experiments" / COMPARE / "mixed_vocab_dfa_ns"
                  / "trajectories" / "pc_spectra_vs_dfa.png"),
        FigureRef("compare", "fig_mixed_nwords_vs_dfa.jpg",
                  ROOT / "experiments" / COMPARE / "mixed_vocab_dfa_ns"
                  / "trajectories" / "nwords_vs_dfa.png"),
        # Curated leftovers from the prior H=100 grid (sources deleted; keep paper JPG).
        FigureRef("compare", "fig_sweep_pc_spectra.jpg",
                  ROOT / "paper" / "figures" / "compare" / "fig_sweep_pc_spectra.jpg",
                  as_jpg=False),
        FigureRef("compare", "fig_sweep_metrics_scatter2d.jpg",
                  ROOT / "paper" / "figures" / "compare" / "fig_sweep_metrics_scatter2d.jpg",
                  as_jpg=False),
        FigureRef("main", "fig05_weights_init_final.jpg",
                  _anchor("weights/weight_init_vs_final.png")),
        FigureRef("main", "fig_weight_matrices_by_dfa.jpg",
                  ROOT / "experiments" / COMPARE / "mixed_vocab_dfa_ns"
                  / "trajectories" / "weight_matrices_by_dfa.png",
                  max_w=1400),
    ]


def _to_jpg(src: Path, dest: Path, *, max_w: int = 1000, quality: int = 90) -> None:
    im = Image.open(src)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h * max_w / w)), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=quality, optimize=True)


def collect(*, dry_run: bool = False) -> dict:
    refs = paper_manifest()
    manifest: list[dict] = []
    missing: list[str] = []

    for ref in refs:
        dest = PAPER_ROOT / ref.section / ref.paper_name
        # Skip self-copies (hand-authored SVGs already in paper/figures).
        same = ref.source.resolve() == dest.resolve() if ref.source.is_file() else False
        entry = {
            "section": ref.section,
            "paper_name": ref.paper_name,
            "source": str(ref.source.relative_to(ROOT)).replace("\\", "/")
            if ref.source.is_relative_to(ROOT) else str(ref.source),
            "dest": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "exists": ref.source.is_file(),
            "skipped_self": same,
        }
        manifest.append(entry)
        if not ref.source.is_file():
            missing.append(entry["source"])
            continue
        if dry_run or same:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if ref.as_jpg and ref.source.suffix.lower() in {".png", ".jpeg", ".jpg", ".webp"}:
            _to_jpg(ref.source, dest, max_w=ref.max_w)
        else:
            shutil.copy2(ref.source, dest)
        print(f"paper <- {dest.relative_to(ROOT)} from {ref.source.name}")

    summary = {
        "copied": sum(1 for e in manifest if e["exists"] and not e["skipped_self"]),
        "missing": missing,
        "figures": manifest,
    }
    if not dry_run:
        PAPER_ROOT.mkdir(parents=True, exist_ok=True)
        (PAPER_ROOT / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report missing sources only")
    args = parser.parse_args()
    summary = collect(dry_run=args.dry_run)
    print(json.dumps({"copied": summary["copied"], "missing": len(summary["missing"])}, indent=2))
    if summary["missing"]:
        print("missing:")
        for path in summary["missing"]:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
