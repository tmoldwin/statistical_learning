"""Cross-task comparison figure specs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComparisonSpec:
    """Defines a side-by-side comparison written under experiments/comparisons/<name>/."""

    name: str
    tasks: tuple[str, ...]
    labels: dict[str, str] = field(default_factory=dict)
    title: str = ""
    model_type: str = "rnn"
    # Optional row grouping: each inner tuple is task names sharing one panel row.
    row_groups: tuple[tuple[str, ...], ...] | None = None

    def label_for(self, task: str) -> str:
        return self.labels.get(task, task)

    @property
    def display_title(self) -> str:
        return self.title or self.name.replace("_", " ").title()
