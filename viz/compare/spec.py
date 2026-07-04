"""Comparison definitions: specs and named presets."""

from __future__ import annotations

from dataclasses import dataclass, field

from experiment import DEFAULT_SEED  # re-export for presets

_COMPARISON_SEEDS = (42, 43, 44, 45, 46)

# Seeds for geometry variability studies (avoid the 40s block used elsewhere).
GEOMETRY_STATS_SEEDS = (1, 2, 3, 5, 7, 8, 11, 13, 17, 19, 23, 29, 31, 37, 53)

_SIXTEEN_WORD_LENGTH_TASKS = (
    "sixteen_word_ns",
    "sixteen_word_four_letter_ns",
    "sixteen_word_five_letter_ns",
    "sixteen_word_six_letter_ns",
    "sixteen_word_seven_letter_ns",
    "sixteen_word_mixed_ns",
)

_SIXTEEN_WORD_LENGTH_LABELS = {
    "sixteen_word_ns": "3-letter",
    "sixteen_word_four_letter_ns": "4-letter",
    "sixteen_word_five_letter_ns": "5-letter",
    "sixteen_word_six_letter_ns": "6-letter",
    "sixteen_word_seven_letter_ns": "7-letter",
    "sixteen_word_mixed_ns": "all lengths",
}

_SIXTEEN_WORD_LENGTH_TASKS_H500 = (
    "sixteen_word_ns_h500",
    "sixteen_word_four_letter_ns_h500",
    "sixteen_word_five_letter_ns_h500",
    "sixteen_word_six_letter_ns_h500",
    "sixteen_word_seven_letter_ns_h500",
    "sixteen_word_mixed_ns_h500",
)

_SIXTEEN_WORD_LENGTH_LABELS_H500 = {
    "sixteen_word_ns_h500": "3-letter",
    "sixteen_word_four_letter_ns_h500": "4-letter",
    "sixteen_word_five_letter_ns_h500": "5-letter",
    "sixteen_word_six_letter_ns_h500": "6-letter",
    "sixteen_word_seven_letter_ns_h500": "7-letter",
    "sixteen_word_mixed_ns_h500": "all lengths",
}

_FIFTY_WORD_LENGTH_TASKS = (
    "fifty_word_ns",
    "fifty_word_four_letter_ns",
    "fifty_word_five_letter_ns",
)

_FIFTY_WORD_LENGTH_LABELS = {
    "fifty_word_ns": "3-letter",
    "fifty_word_four_letter_ns": "4-letter",
    "fifty_word_five_letter_ns": "5-letter",
}

_SIX_WORD_OVERLAP_TASKS = (
    "six_word_overlap_ns",
    "six_word_four_letter_ns",
    "six_word_five_letter_ns",
)

_SIX_WORD_OVERLAP_LABELS = {
    "six_word_overlap_ns": "3-letter",
    "six_word_four_letter_ns": "4-letter",
    "six_word_five_letter_ns": "5-letter",
}

_MICRO_CURRICULUM_TASKS = (
    "two_word_disjoint_ns",
    "two_word_pos_overlap_ns",
    "two_word_prefix_branch_ns",
    "three_word_overlap_ns",
    "three_word_permutation_ns",
    "three_word_ca_hub_ns",
)

_MICRO_CURRICULUM_LABELS = {
    "two_word_disjoint_ns": "disjoint",
    "two_word_pos_overlap_ns": "same 2nd letter",
    "two_word_prefix_branch_ns": "shared prefix",
    "three_word_overlap_ns": "suffix family",
    "three_word_permutation_ns": "permutation",
    "three_word_ca_hub_ns": "3-way ca hub",
}


@dataclass
class ComparisonSpec:
    """Defines a side-by-side comparison written under experiments/comparisons/<name>/."""

    name: str
    tasks: tuple[str, ...]
    labels: dict[str, str] = field(default_factory=dict)
    title: str = ""
    model_type: str = "rnn"
    row_groups: tuple[tuple[str, ...], ...] | None = None
    seeds: tuple[int, ...] = _COMPARISON_SEEDS

    def label_for(self, task: str) -> str:
        return self.labels.get(task, task)

    @property
    def display_title(self) -> str:
        return self.title or self.name.replace("_", " ").title()

    @property
    def panel_rows(self) -> tuple[tuple[str, ...], ...]:
        if self.row_groups is not None:
            return self.row_groups
        return (self.tasks,)


COMPARISON_PRESETS: dict[str, ComparisonSpec] = {
    "six_word_overlap_ns": ComparisonSpec(
        name="six_word_overlap_ns",
        tasks=_SIX_WORD_OVERLAP_TASKS,
        labels=dict(_SIX_WORD_OVERLAP_LABELS),
        title="6-word vocabularies (no spaces): 3 / 4 / 5-letter",
        seeds=GEOMETRY_STATS_SEEDS,
    ),
    "sixteen_word_lengths_ns": ComparisonSpec(
        name="sixteen_word_lengths_ns",
        tasks=_SIXTEEN_WORD_LENGTH_TASKS,
        labels=dict(_SIXTEEN_WORD_LENGTH_LABELS),
        title="16-word vocabularies (no spaces): closed-loop trajectories",
        seeds=GEOMETRY_STATS_SEEDS,
    ),
    "sixteen_word_lengths_ns_learning": ComparisonSpec(
        name="sixteen_word_lengths_ns",
        tasks=_SIXTEEN_WORD_LENGTH_TASKS,
        labels=dict(_SIXTEEN_WORD_LENGTH_LABELS),
        title="16-word vocabularies (no spaces): training",
    ),
    "sixteen_word_lengths_ns_h500": ComparisonSpec(
        name="sixteen_word_lengths_ns_h500",
        tasks=_SIXTEEN_WORD_LENGTH_TASKS_H500,
        labels=dict(_SIXTEEN_WORD_LENGTH_LABELS_H500),
        title="16-word vocabularies (no spaces, 500 units): closed-loop trajectories",
        seeds=GEOMETRY_STATS_SEEDS,
    ),
    "fifty_word_lengths_ns": ComparisonSpec(
        name="fifty_word_lengths_ns",
        tasks=_FIFTY_WORD_LENGTH_TASKS,
        labels=dict(_FIFTY_WORD_LENGTH_LABELS),
        title="50-word vocabularies (no spaces): closed-loop trajectories",
    ),
    "micro_curriculum_ns": ComparisonSpec(
        name="micro_curriculum_ns",
        tasks=_MICRO_CURRICULUM_TASKS,
        labels=dict(_MICRO_CURRICULUM_LABELS),
        title="Micro curriculum (no spaces, 100 units): closed-loop trajectories",
        row_groups=(
            _MICRO_CURRICULUM_TASKS[:3],
            _MICRO_CURRICULUM_TASKS[3:],
        ),
        seeds=GEOMETRY_STATS_SEEDS,
    ),
}
