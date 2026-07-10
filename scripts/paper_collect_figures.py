#!/usr/bin/env python3
"""Copy curated experiment plots into paper_figures/ with a manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from readme_figures import README_FIGURES

PAPER_ROOT = ROOT / "paper_figures"

ANCHOR = "sixteen_word_four_letter_ns"
DEMO = "four_word_overlap_ns"
COMPARE = "sixteen_word_345_ns"


@dataclass(frozen=True)
class FigureRef:
    section: str
    paper_name: str
    source: Path
    caption: str = ""


def _anchor_plot(subpath: str) -> Path:
    return ROOT / "experiments" / ANCHOR / "rnn" / "plots" / subpath


def _demo_plot(subpath: str) -> Path:
    return ROOT / "experiments" / DEMO / "rnn" / "plots" / subpath


def _demo_shared(name: str) -> Path:
    return ROOT / "experiments" / DEMO / "shared" / name


def _shared_plot(name: str) -> Path:
    return ROOT / "experiments" / ANCHOR / "shared" / name


def _compare_plot(kind: str, name: str) -> Path:
    return ROOT / "experiments" / "comparisons" / COMPARE / kind / name


def paper_manifest() -> list[FigureRef]:
    numbered = {fig.number: fig for fig in README_FIGURES}
    refs: list[FigureRef] = [
        FigureRef("demo", "fig01_trie.svg", _demo_shared("vocabulary_trie.svg"),
                  "Prefix trie for the four-word demo lexicon (cat, met, ate, tea)."),
        FigureRef("demo", "fig02_dfa.svg", _demo_shared("vocabulary_min_dfa.svg"),
                  "Minimal DFA for the four-word demo lexicon."),
        FigureRef("demo", "fig03_learning_curve.png",
                  _demo_plot("training/3_learning_curve.png"), numbered[3].caption),
        FigureRef("demo", "fig04_samples.png",
                  _demo_plot("training/4_samples_before_after.png"), numbered[4].caption),
        FigureRef("main", "fig05_weights.png", _anchor_plot("weights/5_weights.png"),
                  numbered[5].caption),
        FigureRef("main", "fig05a_weight_init_vs_final.png",
                  _anchor_plot("weights/weight_init_vs_final.png"),
                  "Random init vs learned input and recurrent weights."),
        FigureRef("main", "fig05b_weights_hh_clustered.png",
                  _anchor_plot("weights/weights_hh_clustered.png"),
                  "Hierarchically clustered recurrent weights $W_{hh}$."),
        FigureRef("main", "fig05c_weights_xh_clustered.png",
                  _anchor_plot("weights/weights_xh_clustered.png"),
                  "Hierarchically clustered input weights $W_{xh}$."),
        FigureRef("main", "fig05d_weight_motif_summary.png",
                  _anchor_plot("weights/weight_motif_summary.png"),
                  "Block coupling, cluster cohesion, and input-tuning entropy."),
        FigureRef("main", "fig05e_weight_structure_metrics.png",
                  _anchor_plot("weights/weight_structure_metrics.png"),
                  "Init vs final feedforward balance metrics."),
        FigureRef("main", "fig08_next_char_probs.png",
                  _anchor_plot("activations/8_next_char_prob_sequence.png"),
                  numbered[8].caption),
        FigureRef("main", "fig09_activation_by_char.png",
                  _anchor_plot("activations/9_activation_by_input_char.png"),
                  numbered[9].caption),
        FigureRef("main", "fig14_next_char_prob_panels.png",
                  _anchor_plot("readout/14_next_char_prob_panels_pca.png"),
                  numbered[14].caption),
        FigureRef("main", "fig18_corr_by_dfa.png",
                  _anchor_plot("correlation/18_state_correlation_by_dfa_state.png"),
                  numbered[18].caption),
        FigureRef("main", "fig07_activation_heatmap.png",
                  _anchor_plot("activations/7_activation_heatmap.png"), numbered[7].caption),
        FigureRef("main", "fig10_activation_clustered.png",
                  _anchor_plot("activations/10_activation_clustered_heatmap.png"),
                  numbered[10].caption),
        FigureRef("main", "fig11_embedding_panels.png",
                  _anchor_plot("embeddings/11_embedding_panels_context.png"),
                  numbered[11].caption),
        FigureRef("main", "fig12_dfa_embedding.png",
                  _anchor_plot("embeddings/12_dfa_and_embedding_pca.png"),
                  numbered[12].caption),
        FigureRef("main", "fig13_next_char_regions.png",
                  _anchor_plot("readout/13_next_char_regions_pca.png"),
                  numbered[13].caption),
        FigureRef("main", "fig16_word_trajectories.png",
                  _anchor_plot("trajectories/16_word_trajectories_pca.png"),
                  numbered[16].caption),
        FigureRef("main", "fig17_state_correlation.png",
                  _anchor_plot("correlation/17_state_correlation_clustered.png"),
                  numbered[17].caption),
        FigureRef("main", "fig19_dfa_distance.png",
                  _anchor_plot("separation/19_dfa_state_distance_comparison.png"),
                  numbered[19].caption),
        FigureRef("main", "fig20_feature_separation.png",
                  _anchor_plot("separation/20_feature_separation_summary.png"),
                  numbered[20].caption),
        FigureRef("main", "fig_decoding_curves.png",
                  _anchor_plot("decoding/decoding_curves.png"),
                  "Linear decoding of DFA state, character, and position from hidden states."),
        FigureRef("main", "fig_decoding_by_seed.png",
                  _anchor_plot("decoding/decoding_curves_by_seed.png"),
                  "Decoding curves across training seeds (realizability)."),
        FigureRef("main", "fig_unit_selectivity.png",
                  _anchor_plot("unit_selectivity/unit_selectivity_summary.png"),
                  "Hidden-unit selectivity for DFA state, character, and position."),
        FigureRef("compare", "fig_compare_learning.png",
                  _compare_plot("learning_curves", "summary.png"),
                  "Learning curves across word-length conditions."),
        FigureRef("compare", "fig_compare_separation.png",
                  _compare_plot("feature_separation", "summary.png"),
                  "Feature separation across word-length conditions."),
    ]
    return refs


def collect(*, dry_run: bool = False) -> dict:
    refs = paper_manifest()
    manifest: list[dict] = []
    missing: list[str] = []

    for ref in refs:
        dest_dir = PAPER_ROOT / ref.section
        dest = dest_dir / ref.paper_name
        entry = {
            "section": ref.section,
            "paper_name": ref.paper_name,
            "source": str(ref.source.relative_to(ROOT)).replace("\\", "/"),
            "dest": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "caption": ref.caption,
            "exists": ref.source.is_file(),
        }
        manifest.append(entry)
        if not ref.source.is_file():
            missing.append(entry["source"])
            continue
        if dry_run:
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref.source, dest)

    summary = {"copied": sum(1 for e in manifest if e["exists"]), "missing": missing, "figures": manifest}
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
