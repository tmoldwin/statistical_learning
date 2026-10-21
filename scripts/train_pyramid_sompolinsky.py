"""Train the pyramid task on the Pehlevan & Sompolinsky (Haim Sompolinsky) Dale net.

Same 4,000 excitatory / 1,000 inhibitory initialization used for the untrained
plots. Before and after are the mean firing to each letter.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.plot_pyramid_dale_init_selectivity import dale_init_model
from visualize import _mean_activation_by_current_letter, load_model
from vocab_pyramid import LETTERS

TASK = "pyramid_letters"
GAIN = 0.02
LEAK = 0.05
SEQ = 8
MAX_STEPS = 2500
EVAL_EVERY = 100
OUT_DIR = REPO / "experiments" / TASK / "sompolinsky"


def _project(w_ih, w_hh, w_ho, signs, mask) -> None:
    col = signs.view(1, -1)
    w_ih.data.clamp_(min=0.0)
    w_hh.data.copy_(w_hh.data.abs() * col * mask)
    w_hh.data.fill_diagonal_(0.0)
    w_ho.data.copy_(w_ho.data.abs() * col)


def _window(w_ih, w_hh, w_ho, b_h, b_o, data, start, h):
    logits = []
    for t in range(SEQ):
        drive = w_ih[:, data[start + t]] + w_hh @ h + b_h
        h = F.relu(drive)
        logits.append(w_ho @ h + b_o)
    target = data[start + 1 : start + 1 + SEQ]
    return F.cross_entropy(torch.stack(logits), target), h.detach()


@torch.no_grad()
def _rollout(w_ih, w_hh, b_h, data) -> np.ndarray:
    h = torch.zeros(w_hh.shape[0])
    activity = torch.empty(data.shape[0], w_hh.shape[0])
    for t in range(data.shape[0]):
        drive = w_ih[:, data[t]] + w_hh @ h + b_h
        h = F.relu(drive)
        activity[t] = h
    return activity.numpy()


def _letter_means(text: str, activity: np.ndarray) -> np.ndarray:
    return _mean_activation_by_current_letter(text, activity, LETTERS)


def _order(mean_act: np.ndarray) -> np.ndarray:
    preferred = np.argmax(mean_act, axis=1)
    rate = mean_act[np.arange(mean_act.shape[0]), preferred]
    return np.lexsort((-rate, preferred))


def plot_before_after(before: np.ndarray, after: np.ndarray, signs: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), sharex=True)
    for row, (name, mask) in enumerate((
        ("Excitatory", signs > 0),
        ("Inhibitory", signs < 0),
    )):
        order = _order(after[mask])
        for col, (block, stage) in enumerate((
            (before[mask][order], "before"),
            (after[mask][order], "after"),
        )):
            ax = axes[row, col]
            positive = block[block > 0]
            peak = float(np.percentile(positive, 99)) if positive.size else 1.0
            im = ax.imshow(
                block, aspect="auto", cmap="Reds", vmin=0.0,
                vmax=peak if peak > 0.0 else 1.0, interpolation="nearest",
            )
            ax.set_xticks(range(len(LETTERS)))
            ax.set_xticklabels(list(LETTERS))
            ax.set_yticks([])
            ax.set_title(f"{name} · {stage}  ·  n={int(mask.sum())}", fontsize=11)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("mean firing", fontsize=8)
        axes[row, 0].set_ylabel("neurons, grouped by preferred letter", fontsize=9)
    fig.suptitle("Sompolinsky weights · mean firing before and after learning", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main() -> None:
    torch.set_num_threads(8)
    template = load_model(str(REPO / "experiments" / TASK / "rnn" / "model.npz"))
    text = (REPO / "experiments" / TASK / "input.txt").read_text(encoding="utf-8")
    char_index = {str(c): i for i, c in enumerate(template["chars"])}
    ids = np.array([char_index[ch] for ch in text], dtype=np.int64)
    split = int(len(ids) * 0.9)
    train = torch.from_numpy(ids[:split])
    val = torch.from_numpy(ids[split:])

    model, signs_np = dale_init_model(template, GAIN)
    signs = torch.tensor(signs_np, dtype=torch.float32)
    mask = torch.tensor(model["weights_hidden_to_hidden"] != 0, dtype=torch.float32)
    w_ih = torch.nn.Parameter(torch.tensor(model["weights_input_to_hidden"], dtype=torch.float32))
    w_hh = torch.nn.Parameter(torch.tensor(model["weights_hidden_to_hidden"], dtype=torch.float32))
    w_ho = torch.nn.Parameter(
        torch.abs(torch.randn(int(ids.max()) + 1, signs.numel())) * signs.view(1, -1) * 0.05
    )
    b_h = torch.nn.Parameter(torch.zeros(signs.numel()))
    b_o = torch.nn.Parameter(torch.zeros(int(ids.max()) + 1))
    opt = torch.optim.Adagrad(
        [
            {"params": [w_ih, w_hh], "lr": 5e-3},
            {"params": [w_ho, b_h, b_o], "lr": 2.5e-2},
        ],
    )

    n_train = train.numel() - SEQ - 1
    cursor = 0
    h = torch.zeros(signs.numel())
    losses: list[float] = []
    best = float("inf")
    stall = 0
    t0 = time.perf_counter()
    for step in range(MAX_STEPS):
        if cursor >= n_train:
            cursor = 0
            h = torch.zeros_like(h)
        opt.zero_grad(set_to_none=True)
        loss, h = _window(w_ih, w_hh, w_ho, b_h, b_o, train, cursor, h)
        if not torch.isfinite(loss):
            print(f"non-finite loss at step {step}", flush=True)
            break
        loss.backward()
        opt.step()
        _project(w_ih, w_hh, w_ho, signs, mask)
        losses.append(float(loss.detach()))
        cursor += SEQ
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                val_loss, _ = _window(w_ih, w_hh, w_ho, b_h, b_o, val, 0, torch.zeros_like(h))
            val_ce = float(val_loss)
            print(
                f"step {step:4d}  train {float(loss.detach()):.3f}  val {val_ce:.3f}  "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
            if val_ce < best - 0.01:
                best = val_ce
                stall = 0
            elif step >= 800 and best < 1.7:
                stall += 1
                if stall >= 8:
                    print(f"stop at step {step}: validation loss plateaued", flush=True)
                    break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "model.npz",
        weights_input_to_hidden=w_ih.detach().numpy(),
        weights_hidden_to_hidden=w_hh.detach().numpy(),
        weights_hidden_to_output=w_ho.detach().numpy(),
        bias_hidden=b_h.detach().numpy(),
        dale_sign=signs_np,
        chars=np.array(template["chars"]),
        loss=np.array(losses, dtype=np.float32),
        gain=np.array(GAIN),
    )
    print(f"saved {OUT_DIR / 'model.npz'}", flush=True)

    init_model, _ = dale_init_model(template, GAIN)
    before_act = _rollout(
        torch.tensor(init_model["weights_input_to_hidden"], dtype=torch.float32),
        torch.tensor(init_model["weights_hidden_to_hidden"], dtype=torch.float32),
        torch.zeros(signs.numel()),
        torch.from_numpy(ids),
    )
    after_act = _rollout(w_ih.detach(), w_hh.detach(), b_h.detach(), torch.from_numpy(ids))
    plot_before_after(
        _letter_means(text, before_act),
        _letter_means(text, after_act),
        signs_np,
        OUT_DIR / "plots" / "letter_heatmap_before_after.png",
    )


if __name__ == "__main__":
    main()
