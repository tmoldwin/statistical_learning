"""Untrained pyramid RNN under Dale's law: selectivity to each input letter.

Pehlevan & Sompolinsky (2014) E/I ratio and strong-synapse J/sqrt(K) weights.
No training. Selectivity is shown as histograms, not density curves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiment import plots_dir
from rnn.rnn_dyn import sample_dale_signs
from visualize import (
    _draw_letter_pyramid_on_axes,
    _mean_activation_by_current_letter,
    average_linkage_cluster_order,
    load_model,
)
from vocab_pyramid import EDGES, PYRAMID_NODE_XY

TASK = "pyramid_letters"
SEED = 1
# Pehlevan & Sompolinsky, PLOS ONE 2014 (strong-synapse / balanced parameters).
# Paper counts are N_E=20000, N_I=5000, K=1000. Same 4:1 ratio and in-degree K,
# at a size that still fits a dense rate simulation.
SEQ_LEN = 48
K_SYN = 1000
N_E = 4_000
N_I = 1_000
E_FRACTION = N_E / (N_E + N_I)
SI_BINS = np.linspace(0.0, 1.0, 21)
# Driving forces (mV): V_E=0, V_I=-80, V_L=-65.
_VE_L = 65.0
_VI_L = 15.0
# J prefactors; synaptic weight is J/sqrt(K). tau_s cancels when E and I share it.
_J = {
    "0E": 4.846 * _VE_L,
    "0I": 3.808 * _VE_L,
    "EE": 0.462 * _VE_L,
    "IE": 1.270 * _VE_L,
    "EI": 10.500 * _VI_L,
    "II": 9.500 * _VI_L,
}
LETTERS = "abcdef"
LETTER_COLORS = {
    "a": "#E69F00",
    "b": "#56B4E9",
    "c": "#009E73",
    "d": "#CC79A7",
    "e": "#0072B2",
    "f": "#D55E00",
}


def _in_mask(n_post: int, n_pre: int, rng: np.random.Generator) -> np.ndarray:
    """Each postsynaptic cell draws K presynaptic partners, or all of them."""
    p = min(1.0, K_SYN / max(n_pre, 1))
    return rng.random((n_post, n_pre)) < p


def dale_init_model(template: dict, gain: float) -> tuple[dict, np.ndarray]:
    """Random Dale weights with a 4:1 E/I split and the paper's J/sqrt(K) ratios.

    Feedforward input is excitatory onto both populations. Recurrent synapses
    keep the presynaptic sign, with mean in-degree K from each source population.
    """
    hidden = N_E + N_I
    vocab = int(template["vocab_size"])
    rng = np.random.default_rng(SEED)
    signs = sample_dale_signs(hidden, E_FRACTION, rng)
    n_e = int(np.sum(signs > 0))
    n_i = hidden - n_e
    scale = {key: gain * value / np.sqrt(K_SYN) for key, value in _J.items()}
    w_ih = np.zeros((hidden, vocab))
    w_ih[:n_e] = np.abs(rng.standard_normal((n_e, vocab))) * scale["0E"]
    w_ih[n_e:] = np.abs(rng.standard_normal((n_i, vocab))) * scale["0I"]
    ee = np.abs(rng.standard_normal((n_e, n_e))) * scale["EE"] * _in_mask(n_e, n_e, rng)
    ie = np.abs(rng.standard_normal((n_i, n_e))) * scale["IE"] * _in_mask(n_i, n_e, rng)
    ei = -np.abs(rng.standard_normal((n_e, n_i))) * scale["EI"] * _in_mask(n_e, n_i, rng)
    ii = -np.abs(rng.standard_normal((n_i, n_i))) * scale["II"] * _in_mask(n_i, n_i, rng)
    np.fill_diagonal(ee, 0.0)
    np.fill_diagonal(ii, 0.0)
    w_hh = np.block([[ee, ei], [ie, ii]])
    w_ho = np.zeros((vocab, hidden))
    model = dict(template)
    model["hidden_size"] = hidden
    model["weights_input_to_hidden"] = w_ih
    model["weights_hidden_to_hidden"] = w_hh
    model["weights_hidden_to_output"] = w_ho
    model["bias_hidden"] = np.zeros((hidden, 1))
    model["bias_output"] = np.zeros((vocab, 1))
    model["use_relu"] = True
    model["dale_law"] = True
    model["e_fraction"] = E_FRACTION
    model["dale_sign"] = signs
    model["timestep_noise_std"] = 0.0
    return model, signs


def _steady_rates(model: dict, *, steps: int = 200, leak: float = 0.05) -> np.ndarray:
    """Rate per letter after the recurrent net settles. Shape (n_letters, H)."""
    w_ih = model["weights_input_to_hidden"]
    w_hh = model["weights_hidden_to_hidden"]
    hidden = w_hh.shape[0]
    vocab = w_ih.shape[1]
    char_index = {str(c): i for i, c in enumerate(model["chars"])}
    rates = np.zeros((len(LETTERS), hidden))
    for index, letter in enumerate(LETTERS):
        x = np.zeros(vocab)
        x[char_index[letter]] = 1.0
        h = np.zeros(hidden)
        for _ in range(steps):
            drive = w_ih @ x + w_hh @ h
            h = (1.0 - leak) * h + leak * np.maximum(drive, 0.0)
            if not np.isfinite(h).all() or float(np.max(h)) > 1e6:
                h[:] = np.nan
                break
        rates[index] = h
    return rates


def _choose_gain(template: dict) -> float:
    """Pick a gain whose letter responses stay finite and not all silent."""
    best = 0.02
    best_score = -1.0
    for gain in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        model, signs = dale_init_model(template, gain)
        rates = _steady_rates(model)
        if not np.isfinite(rates).all():
            continue
        excitatory = signs > 0
        e_mean = float(rates[:, excitatory].mean())
        i_mean = float(rates[:, ~excitatory].mean())
        if e_mean <= 1e-6:
            continue
        # Prefer a response that is neither silent nor huge.
        score = -abs(np.log10(e_mean / 0.2))
        if i_mean > 0:
            score += 0.25
        if score > best_score:
            best_score = score
            best = gain
    return best


def letter_selectivity(acts: np.ndarray, chars: np.ndarray) -> dict[str, np.ndarray]:
    """Peak-vs-rest SI in [0, 1] for each input letter.

    Same index as the trained selectivity curves: nonnegative category rates,
    (r_peak − mean of the other letters) / (r_peak + that mean). A letter that
    is not the unit's peak scores 0.
    """
    hidden = acts.shape[1]
    means = np.stack([
        acts[chars == letter].mean(axis=0) if np.any(chars == letter) else np.zeros(hidden)
        for letter in LETTERS
    ])
    rates = means - means.min(axis=0, keepdims=True)
    r_max = rates.max(axis=0)
    n_at_max = np.sum(np.abs(rates - r_max) <= 1e-15, axis=0)
    out: dict[str, np.ndarray] = {}
    for i, letter in enumerate(LETTERS):
        r_letter = rates[i]
        r_rest = np.delete(rates, i, axis=0).mean(axis=0)
        si = (r_letter - r_rest) / (r_letter + r_rest + 1e-12)
        is_peak = r_letter >= r_max - 1e-15
        si = np.where(is_peak & (r_max > 1e-12) & (n_at_max < rates.shape[0]), si, 0.0)
        out[letter] = np.clip(si, 0.0, 1.0)
    return out


def _histogram(ax, values: np.ndarray, *, color: str, label: str, filled: bool) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 1e-6)]
    if values.size == 0:
        return
    ax.hist(
        values,
        bins=SI_BINS,
        color=color,
        alpha=0.45 if filled else 1.0,
        histtype="stepfilled" if filled else "step",
        linewidth=1.6,
        label=f"{label}  n={values.size}",
    )


def plot(signs: np.ndarray, scores: dict[str, np.ndarray], save_path: Path) -> None:
    excitatory = signs > 0
    panels = (
        ("Excitatory", excitatory),
        ("Inhibitory", ~excitatory),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
    for ax, (name, mask) in zip(axes, panels):
        for letter in LETTERS:
            _histogram(ax, scores[letter][mask], color=LETTER_COLORS[letter], label=letter, filled=False)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"{name}  ·  n={int(mask.sum())}", fontsize=11)
        ax.set_xlabel("selectivity index (peak vs rest)", fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[0].set_ylabel("neurons", fontsize=9)
    axes[1].set_ylabel("neurons", fontsize=9)
    fig.suptitle(
        f"Untrained Dale RNN · {N_E} E, {N_I} I",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def overall_selectivity(rates: np.ndarray) -> np.ndarray:
    """One peak-vs-rest SI in [0, 1] per unit, over the six input letters."""
    shifted = rates - rates.min(axis=0, keepdims=True)
    r_max = shifted.max(axis=0)
    is_max = np.abs(shifted - r_max) <= 1e-15
    n_below = shifted.shape[0] - is_max.sum(axis=0)
    rest = np.where(is_max, 0.0, shifted).sum(axis=0) / np.maximum(n_below, 1)
    si = (r_max - rest) / (r_max + rest + 1e-12)
    si = np.where((r_max > 1e-12) & (n_below > 0), si, 0.0)
    return np.clip(si, 0.0, 1.0)


def plot_overall(signs: np.ndarray, scores: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name, mask, color in (
        ("excitatory", signs > 0, "#D55E00"),
        ("inhibitory", signs < 0, "#0072B2"),
    ):
        _histogram(ax, scores[mask], color=color, label=name, filled=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("selectivity index (peak vs rest)", fontsize=9)
    ax.set_ylabel("neurons", fontsize=9)
    ax.set_title("Per-unit selectivity index", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _letter_means(acts: np.ndarray, chars: np.ndarray, unit: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(len(LETTERS))
    sems = np.zeros(len(LETTERS))
    for i, letter in enumerate(LETTERS):
        vals = acts[chars == letter, unit]
        if vals.size == 0:
            continue
        means[i] = float(vals.mean())
        sems[i] = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return means, sems


def plot_example_means(
    acts: np.ndarray,
    chars: np.ndarray,
    signs: np.ndarray,
    scores: np.ndarray,
    save_path: Path,
    *,
    n_each: int = 6,
) -> None:
    chosen: list[int] = []
    for mask in (signs > 0, signs < 0):
        units = np.flatnonzero(mask)
        order = units[np.argsort(-scores[units])]
        chosen.extend(int(u) for u in order[:n_each])
    fig, axes = plt.subplots(2, n_each, figsize=(2.15 * n_each, 5.0), sharey="row")
    x = np.arange(len(LETTERS))
    colors = [LETTER_COLORS[letter] for letter in LETTERS]
    for ax, unit in zip(axes.ravel(), chosen):
        means, sems = _letter_means(acts, chars, unit)
        kind = "E" if signs[unit] > 0 else "I"
        ax.bar(x, means, yerr=sems, color=colors, capsize=2, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(list(LETTERS), fontsize=8)
        ax.set_title(f"h{unit} {kind}  SI={scores[unit]:.2f}", fontsize=9)
        ax.axhline(0.0, color="0.8", linewidth=0.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.tick_params(axis="y", labelsize=7)
    axes[0, 0].set_ylabel("mean firing", fontsize=9)
    axes[1, 0].set_ylabel("mean firing", fontsize=9)
    fig.suptitle("Example neurons · mean firing to each input letter", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pyramids(
    text: str,
    acts: np.ndarray,
    signs: np.ndarray,
    save_path: Path,
    units: list[int],
) -> None:
    from matplotlib.colors import Normalize

    mean_act = _mean_activation_by_current_letter(text, acts[:, units], LETTERS)
    order = [units[i] for i in average_linkage_cluster_order(mean_act)]
    unit_slot = {unit: slot for slot, unit in enumerate(units)}
    cmap = plt.get_cmap("Reds")
    ncols = 8
    n_units = len(order)
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.35 + 1.3, nrows * 1.45 + 0.35))
    axes_flat = np.atleast_1d(axes).ravel()
    for slot, unit in enumerate(order):
        ax = axes_flat[slot]
        kind = "E" if signs[unit] > 0 else "I"
        row = mean_act[unit_slot[unit]]
        peak = float(np.max(row))
        node_rgba: dict[str, tuple[float, float, float, float]] = {}
        for j, letter in enumerate(LETTERS):
            level = float(row[j]) / peak if peak > 0 else 0.0
            rgba = cmap(level)
            node_rgba[letter] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
        _draw_letter_pyramid_on_axes(ax, edges=EDGES, pos=PYRAMID_NODE_XY, node_face_rgba=node_rgba)
        for letter, (x, y) in PYRAMID_NODE_XY.items():
            ax.text(x, y, letter, ha="center", va="center", fontsize=6, color="#111111", zorder=4)
        ax.set_title(f"h{unit} {kind}", fontsize=8, pad=2)
    for ax in axes_flat[n_units:]:
        ax.axis("off")
    from matplotlib.cm import ScalarMappable

    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("fraction of this neuron's max", fontsize=8)
    fig.suptitle("Pyramid tuning · each pyramid scaled to its own maximum", fontsize=12, y=0.995)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_connectivity(model: dict, signs: np.ndarray, save_path: Path) -> None:
    """Signed input and recurrent weights. Rows are postsynaptic, columns presynaptic."""
    w_ih = np.asarray(model["weights_input_to_hidden"], dtype=float)
    w_hh = np.asarray(model["weights_hidden_to_hidden"], dtype=float)
    chars = [str(c) for c in model["chars"]]
    n_e = int(np.sum(signs > 0))
    boundary = n_e - 0.5
    mids = [n_e / 2.0, n_e + (w_hh.shape[0] - n_e) / 2.0]

    def _vmax(mat: np.ndarray) -> float:
        magnitudes = np.abs(mat)
        magnitudes = magnitudes[magnitudes > 0]
        if magnitudes.size == 0:
            return 1.0
        return max(float(np.percentile(magnitudes, 99.5)), 1e-9)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.6, 10.2),
        gridspec_kw={"height_ratios": [1.15, 6.2]},
        constrained_layout=True,
    )
    cmap = plt.cm.RdBu_r
    vin = _vmax(w_ih)
    im_in = axes[0].imshow(
        w_ih.T, aspect="auto", cmap=cmap, vmin=-vin, vmax=vin,
        interpolation="nearest", origin="upper",
    )
    axes[0].set_title("Input connectivity", fontsize=11)
    axes[0].set_ylabel("input letter", fontsize=9)
    axes[0].set_xlabel("postsynaptic neuron", fontsize=9)
    axes[0].set_yticks(range(len(chars)))
    axes[0].set_yticklabels(chars)
    axes[0].set_xticks(mids)
    axes[0].set_xticklabels(["E", "I"])
    axes[0].axvline(boundary, color="black", lw=0.8, ls="--")
    cbar_in = fig.colorbar(im_in, ax=axes[0], fraction=0.02, pad=0.02)
    cbar_in.set_label("weight", fontsize=8)

    vhh = _vmax(w_hh)
    im_hh = axes[1].imshow(
        w_hh, aspect="equal", cmap=cmap, vmin=-vhh, vmax=vhh,
        interpolation="nearest", origin="upper",
    )
    axes[1].set_title("Recurrent connectivity", fontsize=11)
    axes[1].set_xlabel("presynaptic neuron", fontsize=9)
    axes[1].set_ylabel("postsynaptic neuron", fontsize=9)
    axes[1].set_xticks(mids)
    axes[1].set_xticklabels(["E", "I"])
    axes[1].set_yticks(mids)
    axes[1].set_yticklabels(["E", "I"])
    axes[1].axvline(boundary, color="black", lw=0.8, ls="--")
    axes[1].axhline(boundary, color="black", lw=0.8, ls="--")
    cbar_hh = fig.colorbar(im_hh, ax=axes[1], fraction=0.046, pad=0.02)
    cbar_hh.set_label("weight", fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {save_path}")


def _preferred_order(rates: np.ndarray) -> np.ndarray:
    """Row order: preferred letter, then firing to that letter."""
    preferred = np.argmax(rates, axis=1)
    preferred_rate = rates[np.arange(rates.shape[0]), preferred]
    return np.lexsort((-preferred_rate, preferred))


def plot_sequence_firing(
    activity: np.ndarray,
    signs: np.ndarray,
    text: str,
    rates: np.ndarray,
    save_path: Path,
) -> None:
    """Firing of every neuron for a random sequence of letters."""
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.2))
    for ax, name, mask in zip(axes, ("Excitatory", "Inhibitory"), (signs > 0, signs < 0)):
        order = _preferred_order(rates[:, mask].T)
        mat = activity[:, mask][:, order].T
        peak = float(np.nanmax(mat)) if mat.size else 1.0
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="Reds",
            vmin=0.0,
            vmax=peak if peak > 0 else 1.0,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(text)))
        ax.set_xticklabels(list(text), fontsize=7)
        ax.set_xlabel("presented letter", fontsize=9)
        ax.set_yticks([])
        ax.set_ylabel("neurons, grouped by preferred letter", fontsize=9)
        ax.set_title(f"{name}  ·  n={int(mask.sum())}", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("firing", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Firing to a random letter sequence", fontsize=12)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_firing_heatmaps(rates: np.ndarray, signs: np.ndarray, save_path: Path) -> None:
    """Mean firing of every neuron to each letter. Rows grouped by preferred letter."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 7.4), sharex=True)
    for ax, name, mask in (
        (axes[0], "Excitatory", signs > 0),
        (axes[1], "Inhibitory", signs < 0),
    ):
        block = rates[:, mask].T
        preferred = np.argmax(block, axis=1)
        preferred_rate = block[np.arange(block.shape[0]), preferred]
        order = np.lexsort((-preferred_rate, preferred))
        mat = block[order]
        peak = float(mat.max()) if mat.size else 1.0
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="Reds",
            vmin=0.0,
            vmax=peak if peak > 0 else 1.0,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(LETTERS)))
        ax.set_xticklabels(list(LETTERS))
        ax.set_yticks([])
        ax.set_ylabel("neurons, grouped by preferred letter", fontsize=9)
        ax.set_title(f"{name}  ·  n={int(mask.sum())}", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("mean firing", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Mean firing to each input letter", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {save_path}")


def main() -> None:
    template = load_model(str(REPO / "experiments" / TASK / "rnn" / "model.npz"))
    gain = _choose_gain(template)
    model, signs = dale_init_model(template, gain)
    rates = _steady_rates(model)
    reps = 40
    hidden = np.vstack([np.repeat(rates[i][None, :], reps, axis=0) for i in range(len(LETTERS))])
    chars = np.array([letter for letter in LETTERS for _ in range(reps)])
    text = "".join(chars.tolist())
    n_e = int((signs > 0).sum())
    n_i = int((signs < 0).sum())
    print(f"gain {gain:g}  E {n_e}  I {n_i}  rate E {rates[:, signs > 0].mean():.4f}  I {rates[:, signs < 0].mean():.4f}")
    scores = letter_selectivity(hidden, chars)
    excitatory = signs > 0
    print(
        f"activity  E mean {hidden[:, excitatory].mean():.4f}  "
        f"frac>0 {(hidden[:, excitatory] > 0).mean():.2f}  "
        f"I mean {hidden[:, ~excitatory].mean():.5f}  "
        f"frac>0 {(hidden[:, ~excitatory] > 0).mean():.2f}"
    )
    for letter in LETTERS:
        n = int(np.sum(chars == letter))
        e_mean = float(scores[letter][excitatory].mean())
        i_mean = float(scores[letter][~excitatory].mean())
        print(f"  letter {letter}  n={n:4d}  E selectivity {e_mean:+.3f}  I {i_mean:+.3f}")
    out_dir = plots_dir(TASK, "rnn") / "unit_selectivity"
    si = overall_selectivity(rates)
    plot(signs, scores, out_dir / "selectivity_by_input_dale_init.png")
    plot_overall(signs, si, out_dir / "selectivity_overall_dale_init.png")
    plot_example_means(hidden, chars, signs, si, out_dir / "example_means_dale_init.png")
    examples: list[int] = []
    for mask in (signs > 0, signs < 0):
        pool = np.flatnonzero(mask)
        examples.extend(int(u) for u in pool[np.argsort(-si[pool])][:8])
    plot_pyramids(text, hidden, signs, out_dir / "neuron_pyramids_dale_init.png", examples)
    plot_firing_heatmaps(rates, signs, out_dir / "firing_heatmap_dale_init.png")
    rng = np.random.default_rng(SEED)
    sequence = "".join(LETTERS[i] for i in rng.integers(0, len(LETTERS), size=SEQ_LEN))
    activity = rates[np.fromiter((LETTERS.index(ch) for ch in sequence), dtype=int)]
    plot_sequence_firing(activity, signs, sequence, rates, out_dir / "firing_heatmap_sequence_dale_init.png")
    plot_connectivity(model, signs, out_dir / "connectivity_dale_init.png")


if __name__ == "__main__":
    main()
