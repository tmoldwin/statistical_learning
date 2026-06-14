"""Adapter: expose transformer checkpoints through the RNN visualization interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from transformer.model import BigramLanguageModel


def load_transformer_bundle(exp_dir: Path) -> tuple[BigramLanguageModel, dict, dict]:
    """Load model.pt, model_config.json, training_meta.json from a transformer experiment dir."""
    with open(exp_dir / "model_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(exp_dir / "training_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    model = BigramLanguageModel(
        cfg["vocab_size"],
        cfg["n_embd"],
        cfg["block_size"],
        cfg["num_heads"],
        cfg["head_size"],
        use_residual=cfg.get("use_residual", True),
        n_layer=cfg.get("n_layer", 2),
        use_layernorm=cfg.get("use_layernorm", True),
    )
    state = torch.load(exp_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, meta


def load_model(path: str) -> dict:
    """Return a model dict compatible with visualize.py (transformer backend)."""
    exp_dir = Path(path).parent
    model, cfg, meta = load_transformer_bundle(exp_dir)

    chars = list(cfg.get("chars", cfg.get("alphabet", "")))
    model_dict = {
        "model_type": "transformer",
        "_torch_model": model,
        "model_config": cfg,
        "chars": chars,
        "hidden_size": int(cfg["n_embd"]),
        "vocab_size": int(cfg["vocab_size"]),
        "block_size": int(cfg["block_size"]),
        "use_relu": False,
        "dale_law": False,
    }

    for key in (
        "loss_iterations", "loss_smooth", "metric_iterations",
        "metric_word_error_frac", "sample_before", "sample_after",
        "demo_snippet", "demo_before", "demo_after", "demo_word_error_frac",
        "demo_rng_seed", "demo_seed_char",
    ):
        if key in meta:
            val = meta[key]
            if key.endswith("_iterations"):
                model_dict[key] = np.array(val, dtype=np.int32)
            elif key.endswith("_frac") or key == "loss_smooth":
                model_dict[key] = np.array(val, dtype=np.float64) if isinstance(val, list) else float(val)
            elif key == "demo_rng_seed":
                model_dict[key] = int(val)
            else:
                model_dict[key] = str(val)

    if "vocab_words" not in model_dict and "words" in cfg:
        model_dict["vocab_words"] = list(cfg["words"])

    return model_dict


@torch.no_grad()
def forward_pass(model_dict: dict, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Run transformer over text; return (hidden_states, output_probs) like the RNN."""
    torch_model: BigramLanguageModel = model_dict["_torch_model"]
    chars = model_dict["chars"]
    char_to_index = {c: i for i, c in enumerate(chars)}
    block_size = model_dict["block_size"]
    T = len(text)
    hidden_size = model_dict["hidden_size"]
    vocab_size = model_dict["vocab_size"]

    hidden_states = np.zeros((T, hidden_size), dtype=np.float64)
    output_probs = np.zeros((T, vocab_size), dtype=np.float64)

    ids = [char_to_index[ch] for ch in text]
    for t in range(T):
        start = max(0, t - block_size + 1)
        window = ids[start : t + 1]
        X = torch.tensor([window], dtype=torch.long)
        logits, hidden, _ = torch_model.features(X)
        h = hidden[0, -1, :].cpu().numpy()
        p = torch.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
        hidden_states[t] = h
        output_probs[t] = p

    return hidden_states, output_probs
