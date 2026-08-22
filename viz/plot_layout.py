"""Matplotlib helpers to keep tick labels, titles, and annotations readable."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def rotation_for_tick_labels(labels: list[str], *, n_positions: int | None = None) -> tuple[int, str]:
    """Pick rotation and horizontal alignment for categorical x tick labels."""
    n = n_positions if n_positions is not None else len(labels)
    max_len = max((len(str(label)) for label in labels), default=0)
    if n >= 6 or (n >= 5 and max_len > 8):
        return 40, "right"
    if n >= 5 or max_len > 12:
        return 35, "right"
    if n >= 4 and max_len > 9:
        return 25, "right"
    if max_len > 14:
        return 20, "right"
    return 0, "center"


def apply_category_tick_labels(
    ax,
    labels: list[str],
    *,
    fontsize: float = 9,
    positions: list[int] | None = None,
) -> float:
    """Set x tick labels with auto rotation. Returns suggested bottom margin."""
    labels = [str(label) for label in labels]
    positions = list(range(len(labels))) if positions is None else positions
    rotation, ha = rotation_for_tick_labels(labels, n_positions=len(labels))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation, ha=ha)
    if rotation >= 35:
        return 0.24
    if rotation >= 20:
        return 0.17
    if rotation > 0:
        return 0.13
    return 0.09


def hide_x_tick_labels(ax) -> None:
    ax.tick_params(labelbottom=False)
    ax.set_xticklabels([])


def set_ylabel_multiline(ax, label: str, *, fontsize: float = 8, labelpad: float = 2.0) -> None:
    ax.set_ylabel(label, fontsize=fontsize, labelpad=labelpad)


def finalize_grid_figure(
    fig,
    *,
    suptitle: str | None = None,
    suptitle_fontsize: float = 11,
    bottom: float = 0.10,
    top: float | None = None,
    left: float | None = None,
    right: float | None = None,
    hspace: float = 0.38,
    wspace: float = 0.28,
) -> None:
    """Apply margins so ``suptitle`` never collides with panel titles.

    Default ``top`` is 0.88 when a suptitle is set (≤0.93 per plot-text-layout);
    use a lower ``top`` (≈0.78–0.84) when panel titles are multi-line.
    """
    if top is None:
        top = 0.84 if suptitle else 0.93
    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=0.98)
    kwargs = dict(top=top, bottom=bottom, hspace=hspace, wspace=wspace)
    if left is not None:
        kwargs["left"] = left
    if right is not None:
        kwargs["right"] = right
    fig.subplots_adjust(**kwargs)


def condition_bar_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Distinct qualitative colors for ``n`` condition bars."""
    if n <= 0:
        return []
    palette = (
        "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
    )
    if n <= len(palette):
        return [to_rgba(palette[i]) for i in range(n)]
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def save_figure(fig, path, *, dpi: int = 150) -> None:
    # Do not use bbox_inches="tight": it collapses the headroom reserved for
    # suptitles via subplots_adjust / gridspec and causes title overlap.
    fig.savefig(path, dpi=dpi, pad_inches=0.20)
    plt.close(fig)
