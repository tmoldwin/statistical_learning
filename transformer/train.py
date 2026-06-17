"""Train a character-level transformer on a statistical-learning corpus."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (  # noqa: E402
    EXPERIMENT_CONFIG,
    TRANSFORMER_DEFAULTS,
    experiment_regime,
    input_path,
    model_config_path,
    model_dir,
    model_path,
    training_meta_path,
)
from task import REGIMES  # noqa: E402
from transformer.data_char import (  # noqa: E402
    build_char_vocab,
    decode,
    encode,
    get_batch_from_ids,
    split_train_val,
)
from transformer.model import BigramLanguageModel  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def sample_text(
    model: BigramLanguageModel,
    stoi: dict[str, int],
    itos: dict[int, str],
    n_chars: int,
    *,
    word_space: bool,
    vocab_words: set[str],
    rng: random.Random,
) -> str:
    start_ch = " " if (word_space and " " in stoi) else rng.choice(sorted(stoi))
    idx = torch.tensor([[stoi[start_ch]]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=n_chars)[0].tolist()
    text = decode(out, itos)
    return text[1:] if (word_space and text.startswith(" ")) else text


def invalid_word_fraction(text: str, vocab_words: set[str]) -> float:
    if not vocab_words:
        return float("nan")
    tokens = [t for t in text.split(" ") if t]
    if not tokens:
        return float("nan")
    bad = sum(1 for t in tokens if t not in vocab_words)
    return bad / len(tokens)


@torch.no_grad()
def evaluate_word_error(
    model: BigramLanguageModel,
    stoi: dict[str, int],
    itos: dict[int, str],
    vocab_words: set[str],
    *,
    word_space: bool,
    rollout_len: int,
    num_rollouts: int,
    rng: random.Random,
) -> tuple[float, str]:
    errs = []
    first = ""
    for r in range(num_rollouts):
        text = sample_text(
            model, stoi, itos, rollout_len,
            word_space=word_space, vocab_words=vocab_words, rng=rng,
        )
        if r == 0:
            first = text
        errs.append(invalid_word_fraction(text, vocab_words))
    return float(np.nanmean(errs)), first


def train(
    exp_name: str,
    *,
    steps: int | None = None,
    seed: int = 42,
) -> None:
    cfg = EXPERIMENT_CONFIG[exp_name]
    regime = experiment_regime(exp_name)
    words = REGIMES[regime]
    word_space = bool(cfg.get("word_space", False))
    vocab_words = set(words)

    corpus_path = input_path(exp_name)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}. Run task.py first.")

    text = corpus_path.read_text(encoding="utf-8")
    alphabet, stoi, itos = build_char_vocab(text)
    vocab_size = len(alphabet)
    ids = encode(text, stoi)
    train_ids, val_ids = split_train_val(ids, train_ratio=0.9)

    hidden_size = int(cfg.get("hidden_size", TRANSFORMER_DEFAULTS["n_embd"]))
    block_size = int(cfg.get("sequence_length", TRANSFORMER_DEFAULTS["block_size"]))
    max_steps = steps if steps is not None else int(cfg["steps"])
    num_heads = int(cfg.get("num_heads", TRANSFORMER_DEFAULTS["num_heads"]))
    head_size = int(cfg.get("head_size", hidden_size if num_heads == 1 else TRANSFORMER_DEFAULTS["head_size"]))
    if num_heads * head_size != hidden_size:
        raise ValueError(
            f"num_heads ({num_heads}) * head_size ({head_size}) must equal hidden_size ({hidden_size})"
        )

    model_cfg = {
        "vocab_size": vocab_size,
        "n_embd": hidden_size,
        "block_size": block_size,
        "num_heads": num_heads,
        "head_size": head_size,
        "n_layer": int(cfg.get("n_layer", TRANSFORMER_DEFAULTS["n_layer"])),
        "use_layernorm": bool(cfg.get("use_layernorm", TRANSFORMER_DEFAULTS["use_layernorm"])),
        "use_residual": bool(cfg.get("use_residual", TRANSFORMER_DEFAULTS["use_residual"])),
        "alphabet": alphabet,
        "chars": list(alphabet),
        "words": words,
        "word_space": word_space,
    }

    set_seed(seed)
    model = BigramLanguageModel(
        vocab_size,
        model_cfg["n_embd"],
        block_size,
        model_cfg["num_heads"],
        model_cfg["head_size"],
        use_residual=model_cfg["use_residual"],
        n_layer=model_cfg["n_layer"],
        use_layernorm=model_cfg["use_layernorm"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(TRANSFORMER_DEFAULTS["learning_rate"]),
    )

    batch_size = int(cfg.get("batch_size", TRANSFORMER_DEFAULTS["batch_size"]))
    eval_interval = int(cfg.get("eval_interval", TRANSFORMER_DEFAULTS["eval_interval"]))
    eval_iterations = int(cfg.get("eval_iterations", TRANSFORMER_DEFAULTS["eval_iterations"]))
    metric_rollout_len = int(cfg.get("metric_rollout_len", 3000))

    loss_iterations: list[int] = []
    loss_smooth: list[float] = []
    metric_iterations: list[int] = []
    metric_word_error_frac: list[float] = []

    smooth_loss = float("nan")
    demo_snippet_len = 50
    demo_snippet = text[:demo_snippet_len]

    rng = random.Random(seed + 1000)
    metric_rng = random.Random(seed + 2000)

    # Before-training sample for visualization.
    model.eval()
    _, sample_before = evaluate_word_error(
        model, stoi, itos, vocab_words,
        word_space=word_space, rollout_len=metric_rollout_len, num_rollouts=1, rng=metric_rng,
    )
    demo_before = sample_before[:demo_snippet_len]

    model.train()
    for step in range(max_steps + 1):
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            train_losses, val_losses = [], []
            for _ in range(eval_iterations):
                X, Y = get_batch_from_ids(train_ids, block_size, batch_size)
                _, loss = model(X, Y)
                train_losses.append(loss.item())
                Xv, Yv = get_batch_from_ids(val_ids, block_size, batch_size)
                _, vloss = model(Xv, Yv)
                val_losses.append(vloss.item())
            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            blend = train_loss
            smooth_loss = blend if not np.isfinite(smooth_loss) else smooth_loss * 0.995 + blend * 0.005
            loss_iterations.append(step)
            loss_smooth.append(smooth_loss)

            word_err, _ = evaluate_word_error(
                model, stoi, itos, vocab_words,
                word_space=word_space, rollout_len=metric_rollout_len, num_rollouts=3,
                rng=random.Random(seed + step),
            )
            metric_iterations.append(step)
            metric_word_error_frac.append(word_err)
            print(
                f"  step {step:5d} | loss {smooth_loss:.4f} | "
                f"word err {100 * word_err:.2f}%",
                flush=True,
            )
            model.train()

        if step == max_steps:
            break

        X, Y = get_batch_from_ids(train_ids, block_size, batch_size)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    final_word_err, sample_after = evaluate_word_error(
        model, stoi, itos, vocab_words,
        word_space=word_space, rollout_len=metric_rollout_len, num_rollouts=5,
        rng=random.Random(seed + max_steps),
    )
    demo_after = sample_after[:demo_snippet_len]

    out_dir = model_dir(exp_name, "transformer")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path(exp_name, "transformer"))

    with open(model_config_path(exp_name), "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=2)

    meta = {
        "loss_iterations": loss_iterations,
        "loss_smooth": loss_smooth,
        "metric_iterations": metric_iterations,
        "metric_word_error_frac": metric_word_error_frac,
        "sample_before": sample_before[:demo_snippet_len],
        "sample_after": sample_after[:demo_snippet_len],
        "demo_snippet": demo_snippet,
        "demo_before": demo_before,
        "demo_after": demo_after,
        "demo_word_error_frac": invalid_word_fraction(sample_after, vocab_words),
        "demo_rng_seed": seed + max_steps,
        "demo_seed_char": " " if (" " in stoi) else alphabet[0],
        "final_word_error_frac": final_word_err,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
    }
    with open(training_meta_path(exp_name), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved transformer to {model_path(exp_name, 'transformer')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENT_CONFIG.keys()))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.exp, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
