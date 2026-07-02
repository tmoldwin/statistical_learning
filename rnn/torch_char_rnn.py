"""GPU-accelerated character RNN training (PyTorch). Writes the same .npz format as min_char_rnn.py."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import experiment_regime
from rnn.rollout_metrics import METRIC_RNG_BASE, stochastic_word_validity_metrics
from task import REGIMES
from vocab_diagrams import invalid_word_fraction


class VanillaCharRNN(torch.nn.Module):
    """Column-vector RNN matching min_char_rnn weight layout (H x V, H x H, V x H)."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.W_ih = torch.nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.W_hh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_ho = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.bias_h = torch.nn.Parameter(torch.zeros(hidden_size, 1))
        self.bias_o = torch.nn.Parameter(torch.zeros(vocab_size, 1))

    def forward_sequence(
        self,
        input_indices: torch.Tensor,
        target_indices: torch.Tensor,
        h0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = h0
        total_loss = torch.zeros((), device=h0.device)
        seq_len = int(input_indices.shape[0])
        for t in range(seq_len):
            x = F.one_hot(input_indices[t], self.vocab_size).to(h.dtype).unsqueeze(1)
            h = torch.tanh(self.W_ih @ x + self.W_hh @ h + self.bias_h)
            logits = self.W_ho @ h + self.bias_o
            total_loss = total_loss + F.cross_entropy(logits.squeeze(1), target_indices[t])
        return total_loss, h

    def numpy_weights(self) -> dict[str, np.ndarray]:
        return {
            "weights_input_to_hidden": self.W_ih.detach().cpu().numpy(),
            "weights_hidden_to_hidden": self.W_hh.detach().cpu().numpy(),
            "weights_hidden_to_output": self.W_ho.detach().cpu().numpy(),
            "bias_hidden": self.bias_h.detach().cpu().numpy(),
            "bias_output": self.bias_o.detach().cpu().numpy(),
        }

    def load_numpy_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.W_ih.data.copy_(torch.as_tensor(weights["weights_input_to_hidden"]))
        self.W_hh.data.copy_(torch.as_tensor(weights["weights_hidden_to_hidden"]))
        self.W_ho.data.copy_(torch.as_tensor(weights["weights_hidden_to_output"]))
        self.bias_h.data.copy_(torch.as_tensor(weights["bias_hidden"]))
        self.bias_o.data.copy_(torch.as_tensor(weights["bias_output"]))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--model", default="model.npz")
    parser.add_argument("--hidden-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--exp", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-word-error", type=float, default=None)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--save-snapshots", action="store_true")
    args = parser.parse_args()

    if args.save_snapshots:
        raise SystemExit("GPU trainer does not support --save-snapshots; use CPU min_char_rnn.py")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"device: {device}")

    text = Path(args.input).read_text(encoding="utf-8")
    unique_chars = list(set(text))
    text_length, vocab_size = len(text), len(unique_chars)
    print(f"data has {text_length} characters, {vocab_size} unique.")
    char_to_index = {ch: i for i, ch in enumerate(unique_chars)}
    index_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    text_indices = np.array([char_to_index[ch] for ch in text], dtype=np.int64)

    corpus_has_spaces = " " in text
    if args.exp:
        vocab_words = set(REGIMES[experiment_regime(args.exp)])
    elif corpus_has_spaces:
        vocab_words = set(text.split())
    else:
        vocab_words = set()
    use_word_segmentation = bool(vocab_words) and not corpus_has_spaces

    hidden_size = args.hidden_size
    sequence_length = max(1, int(args.sequence_length))
    learning_rate = float(args.learning_rate)

    early_stop_patience = 3
    min_checkpoint_iter = 8_000
    if args.exp:
        from experiment import EXPERIMENT_CONFIG
        cfg = EXPERIMENT_CONFIG.get(args.exp, {})
        early_stop_patience = int(cfg.get("early_stop_patience", early_stop_patience))
        min_checkpoint_iter = int(cfg.get("min_checkpoint_iter", min_checkpoint_iter))
        if args.target_word_error is None and cfg.get("target_word_error_frac") is not None:
            args.target_word_error = float(cfg["target_word_error_frac"])

    model = VanillaCharRNN(vocab_size, hidden_size).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    smooth_loss = -np.log(1.0 / vocab_size) * sequence_length
    loss_iterations: list[int] = []
    loss_smooth: list[float] = []
    loss_window: list[float] = []
    metric_iters: list[int] = []
    metric_valid_letter_frac: list[float] = []
    metric_word_error_frac: list[float] = []

    best_word_err = float("inf")
    best_iter = -1
    best_state: dict[str, np.ndarray] | None = None
    target_met_streak = 0
    stop_threshold = args.target_word_error if args.target_word_error is not None else 0.03
    if args.target_word_error is not None:
        print(f"early stop target: {100.0 * stop_threshold:.2f}% word error ({early_stop_patience} consecutive evals)")

    demo_seed_char = " " if " " in char_to_index else text[0]
    demo_snippet = text[:50]
    sample_before_text = ""
    demo_before = ""
    demo_rng_seed = METRIC_RNG_BASE

    data_pointer = 0
    h = torch.zeros(hidden_size, 1, device=device)
    iteration = 0

    while iteration < args.steps:
        if data_pointer + sequence_length + 1 >= text_length or iteration == 0:
            h.zero_()
            data_pointer = 0

        inp = text_indices[data_pointer: data_pointer + sequence_length]
        tgt = text_indices[data_pointer + 1: data_pointer + sequence_length + 1]
        input_t = torch.as_tensor(inp, device=device)
        target_t = torch.as_tensor(tgt, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss_t, h = model.forward_sequence(input_t, target_t, h)
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_val = float(loss_t.detach().cpu())
        if np.isfinite(loss_val):
            smooth_loss = smooth_loss * 0.995 + loss_val * 0.005
        loss_iterations.append(iteration)
        loss_smooth.append(smooth_loss)
        loss_window.append(loss_val)

        if iteration % 100 == 0:
            print(f"iter {iteration}, loss: {smooth_loss:.6f}")
            weights = model.numpy_weights()
            metric_rng = np.random.default_rng(METRIC_RNG_BASE + iteration)
            word_err, letter_frac, rollout_text = stochastic_word_validity_metrics(
                weights,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                index_to_char=index_to_char,
                seed_index=char_to_index[demo_seed_char],
                vocab=vocab_words,
                use_word_segmentation=use_word_segmentation,
                use_relu=False,
                timestep_noise_std=0.0,
                rng=metric_rng,
            )
            metric_iters.append(iteration)
            metric_valid_letter_frac.append(letter_frac)
            metric_word_error_frac.append(word_err)

            if iteration >= min_checkpoint_iter and vocab_words and np.isfinite(word_err):
                if word_err < best_word_err:
                    best_word_err = word_err
                    best_iter = iteration
                    best_state = copy.deepcopy(weights)
                if word_err <= stop_threshold:
                    target_met_streak += 1
                else:
                    target_met_streak = 0
                if target_met_streak >= early_stop_patience:
                    print(
                        f"early stop at iter {iteration}: "
                        f"word error <= {100.0 * stop_threshold:.2f}% "
                        f"for {early_stop_patience * 100} iterations",
                    )
                    break

            if iteration == 0:
                sample_before_text = rollout_text[:50]
                demo_before = rollout_text[:50]
                demo_rng_seed = METRIC_RNG_BASE

        data_pointer += sequence_length
        iteration += 1

    if best_state is not None and best_iter >= min_checkpoint_iter:
        print(
            f"using checkpoint from iter {best_iter} "
            f"({100.0 * best_word_err:.2f}% word error, best seen)",
        )
        model.load_numpy_weights(best_state)
    else:
        print(f"keeping final weights (target {100.0 * stop_threshold:.2f}% word error was not reached)")

    final_weights = model.numpy_weights()
    final_rng = np.random.default_rng(METRIC_RNG_BASE + iteration)
    final_word_err, final_letter_valid, rollout_text = stochastic_word_validity_metrics(
        final_weights,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        index_to_char=index_to_char,
        seed_index=char_to_index[demo_seed_char],
        vocab=vocab_words,
        use_word_segmentation=use_word_segmentation,
        use_relu=False,
        timestep_noise_std=0.0,
        rng=final_rng,
    )
    demo_word_error_frac = invalid_word_fraction(
        rollout_text, vocab_words, spaced=not use_word_segmentation, trim_edges=True,
    )
    print(f"final word error rate: {100.0 * final_word_err:.2f}%")

    model_out = args.model
    model_out_parent = os.path.dirname(model_out)
    if model_out_parent:
        os.makedirs(model_out_parent, exist_ok=True)

    np.savez(
        model_out,
        **final_weights,
        chars=np.array(unique_chars),
        hidden_size=np.array(hidden_size),
        vocab_size=np.array(vocab_size),
        loss_iterations=np.array(loss_iterations, dtype=np.int32),
        loss_smooth=np.array(loss_smooth, dtype=np.float64),
        loss_window=np.array(loss_window, dtype=np.float64),
        metric_iterations=np.array(metric_iters, dtype=np.int32),
        metric_valid_vocab_letter_frac=np.array(metric_valid_letter_frac, dtype=np.float64),
        metric_word_error_frac=np.array(metric_word_error_frac, dtype=np.float64),
        best_metric_iter=np.array(best_iter, dtype=np.int32),
        best_metric_word_error_frac=np.array(best_word_err, dtype=np.float64),
        vocab_words=np.array(sorted(vocab_words)),
        sample_before=np.array(sample_before_text),
        sample_after=np.array(rollout_text[:50]),
        demo_snippet=np.array(demo_snippet),
        demo_before=np.array(demo_before),
        demo_after=np.array(rollout_text[:50]),
        demo_word_error_frac=np.array(demo_word_error_frac, dtype=np.float64),
        demo_rng_seed=np.array(demo_rng_seed, dtype=np.int64),
        demo_seed_char=np.array(demo_seed_char),
        dale_law=np.array(False),
        use_relu=np.array(False),
        e_fraction=np.array(0.0),
        dale_sign=np.array([]),
        timestep_noise_std=np.array(0.0, dtype=np.float64),
        trainer=np.array("torch"),
        device=np.array(str(device)),
    )
    print(f"saved trained model to {model_out}")


if __name__ == "__main__":
    main()