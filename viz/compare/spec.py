"""Comparison definitions: specs and named presets."""

from __future__ import annotations

from dataclasses import dataclass, field

from experiment import DEFAULT_SEED  # re-export for presets

_COMPARISON_SEEDS = (42, 43, 44, 45, 46)

_SIXTEEN_WORD_LENGTH_TASKS = (
    "sixteen_word_ns",
    "sixteen_word_four_letter_ns",
    "sixteen_word_five_letter_ns",
    "sixteen_word_mixed_ns",
)

_SIXTEEN_WORD_LENGTH_LABELS = {
    "sixteen_word_ns": "3-letter",
    "sixteen_word_four_letter_ns": "4-letter",
    "sixteen_word_five_letter_ns": "5-letter",
    "sixteen_word_mixed_ns": "mixed length",
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
    "sixteen_word_lengths_ns": ComparisonSpec(
        name="sixteen_word_lengths_ns",
        tasks=_SIXTEEN_WORD_LENGTH_TASKS,
        labels=dict(_SIXTEEN_WORD_LENGTH_LABELS),
        title="16-word vocabularies (no spaces): closed-loop trajectories",
    ),
    "sixteen_word_lengths_ns_learning": ComparisonSpec(
        name="sixteen_word_lengths_ns",
        tasks=_SIXTEEN_WORD_LENGTH_TASKS,
        labels=dict(_SIXTEEN_WORD_LENGTH_LABELS),
        title="16-word vocabularies (no spaces): training",
    ),
    "fifty_word_lengths_ns": ComparisonSpec(
        name="fifty_word_lengths_ns",
        tasks=_FIFTY_WORD_LENGTH_TASKS,
        labels=dict(_FIFTY_WORD_LENGTH_LABELS),
        title="50-word vocabularies (no spaces): closed-loop trajectories",
    ),
}
