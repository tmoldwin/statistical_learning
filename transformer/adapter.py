"""Adapter: expose transformer checkpoints for representation-level analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from transformer.model import BigramLanguageModel


def load_transformer_bundle(
    exp_dir: Path,
    *,
    checkpoint: Path | None = None,
) -> tuple[BigramLanguageModel, dict, dict]:
    """Load checkpoint, model_config.json, and training meta from a transformer experiment dir."""
    exp_dir = Path(exp_dir)
    checkpoint = Path(checkpoint) if checkpoint is not None else exp_dir / "model.pt"
    if checkpoint.name.startswith("model_seed"):
        seed_suffix = checkpoint.stem.removeprefix("model_")
        meta_path = exp_dir / f"training_meta_{seed_suffix}.json"
        if not meta_path.is_file():
            meta_path = exp_dir / "training_meta.json"
    else:
        meta_path = exp_dir / "training_meta.json"

    with open(exp_dir / "model_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    model = BigramLanguageModel(
        cfg["vocab_size"],
        cfg["n_embd"],
        cfg["block_size"],
        cfg["num_heads"],
        cfg["head_size"],
        use_residual=cfg.get("use_residual", True),
        n_layer=cfg.get("n_layer", 2),
        use_layernorm=cfg.get("use_layernorm", True),
        pos_embd_dim=cfg.get("pos_embd_dim"),
        timestep_noise_std=cfg.get("timestep_noise_std", 0.0),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.timestep_noise_std = float(cfg.get("timestep_noise_std", 0.0))
    model.eval()
    return model, cfg, meta


def load_model(path: str) -> dict:
    """Return a model dict for visualize.py (transformer backend)."""
    ckpt_path = Path(path)
    exp_dir = ckpt_path.parent
    model, cfg, meta = load_transformer_bundle(exp_dir, checkpoint=ckpt_path)

    chars = list(cfg.get("chars", cfg.get("alphabet", "")))
    model_dict = {
        "model_type": "transformer",
        "_torch_model": model,
        "model_config": cfg,
        "chars": chars,
        "hidden_size": int(cfg["n_embd"]),
        "pos_embd_dim": int(cfg.get("pos_embd_dim", cfg["n_embd"])),
        "vocab_size": int(cfg["vocab_size"]),
        "block_size": int(cfg["block_size"]),
        "num_heads": int(cfg["num_heads"]),
        "head_size": int(cfg["head_size"]),
        "n_layer": int(cfg.get("n_layer", 2)),
        "use_relu": False,
        "dale_law": False,
        "timestep_noise_std": float(cfg.get("timestep_noise_std", 0.0)),
    }

    for key in (
        "loss_iterations", "loss_window", "loss_smooth", "metric_iterations",
        "metric_word_error_frac", "sample_before", "sample_after",
        "demo_snippet", "demo_before", "demo_after", "demo_word_error_frac",
        "demo_rng_seed", "demo_seed_char",
    ):
        if key in meta:
            val = meta[key]
            if key.endswith("_iterations"):
                model_dict[key] = np.array(val, dtype=np.int32)
            elif key.endswith("_frac") or key in ("loss_smooth", "loss_window"):
                model_dict[key] = np.array(val, dtype=np.float64) if isinstance(val, list) else float(val)
            elif key == "demo_rng_seed":
                model_dict[key] = int(val)
            else:
                model_dict[key] = str(val)

    if "vocab_words" not in model_dict and "words" in cfg:
        model_dict["vocab_words"] = list(cfg["words"])

    return model_dict


@dataclass
class LayerActivations:
    """Per-layer Q/K/V extracted at each corpus timestep."""

    attn_input: np.ndarray
    queries: list[np.ndarray]
    keys: list[np.ndarray]
    values: list[np.ndarray]
    attention_lags: list[np.ndarray]
    post_attn: np.ndarray
    post_ffwd: np.ndarray


@dataclass
class TransformerActivations:
    """All transformer representations aligned to corpus timesteps (T rows each)."""

    token_emb: np.ndarray
    pos_emb: np.ndarray
    block_input: np.ndarray
    block_output: np.ndarray
    output_probs: np.ndarray
    layers: list[LayerActivations] = field(default_factory=list)
    num_heads: int = 0
    head_size: int = 0
    n_embd: int = 0
    block_size: int = 0


@torch.no_grad()
def extract_transformer_activations(model_dict: dict, text: str) -> TransformerActivations:
    """Run the causal transformer and collect each representation independently.

    Unlike an RNN hidden state, a transformer has separate token embeddings,
    position embeddings, per-head Q/K/V vectors, attention weights, and a final
    block output fed to lm_head. Each is stored here as (T, ...) arrays aligned
    to corpus timesteps (the vector at the current query position in the causal window).
    """
    torch_model: BigramLanguageModel = model_dict["_torch_model"]
    chars = model_dict["chars"]
    char_to_index = {c: i for i, c in enumerate(chars)}
    block_size = model_dict["block_size"]
    n_embd = model_dict["hidden_size"]
    num_heads = model_dict["num_heads"]
    head_size = model_dict["head_size"]
    n_layer = len(torch_model.blocks) if not torch_model._legacy else 1
    T = len(text)

    token_emb = np.zeros((T, n_embd), dtype=np.float64)
    pos_emb = np.zeros((T, n_embd), dtype=np.float64)
    block_input = np.zeros((T, n_embd), dtype=np.float64)
    block_output = np.zeros((T, n_embd), dtype=np.float64)
    output_probs = np.zeros((T, model_dict["vocab_size"]), dtype=np.float64)

    layers: list[LayerActivations] = []
    for _ in range(n_layer):
        layers.append(LayerActivations(
            attn_input=np.zeros((T, n_embd), dtype=np.float64),
            queries=[np.zeros((T, head_size), dtype=np.float64) for _ in range(num_heads)],
            keys=[np.zeros((T, head_size), dtype=np.float64) for _ in range(num_heads)],
            values=[np.zeros((T, head_size), dtype=np.float64) for _ in range(num_heads)],
            attention_lags=[np.full((T, block_size), np.nan, dtype=np.float64) for _ in range(num_heads)],
            post_attn=np.zeros((T, n_embd), dtype=np.float64),
            post_ffwd=np.zeros((T, n_embd), dtype=np.float64),
        ))

    ids = [char_to_index[ch] for ch in text]
    for t in range(T):
        start = max(0, t - block_size + 1)
        window = ids[start : t + 1]
        W = len(window)
        X = torch.tensor([window], dtype=torch.long)
        acts = torch_model.forward_with_activations(X)
        q_idx = W - 1

        token_emb[t] = acts["token_emb"][0, q_idx].cpu().numpy()
        pos_emb[t] = acts["pos_emb"][0, q_idx].cpu().numpy()
        block_input[t] = acts["block_input"][0, q_idx].cpu().numpy()
        block_output[t] = acts["block_output"][0, q_idx].cpu().numpy()
        output_probs[t] = torch.softmax(acts["logits"][0, q_idx], dim=-1).cpu().numpy()

        for layer_idx, layer_acts in enumerate(acts["layers"]):
            layer = layers[layer_idx]
            layer.attn_input[t] = layer_acts["attn_input"][0, q_idx].cpu().numpy()
            layer.post_attn[t] = layer_acts["post_attn"][0, q_idx].cpu().numpy()
            layer.post_ffwd[t] = layer_acts["post_ffwd"][0, q_idx].cpu().numpy()
            for h in range(num_heads):
                layer.queries[h][t] = layer_acts["queries"][h][0, q_idx].cpu().numpy()
                layer.keys[h][t] = layer_acts["keys"][h][0, q_idx].cpu().numpy()
                layer.values[h][t] = layer_acts["values"][h][0, q_idx].cpu().numpy()
                attn_row = layer_acts["attention"][h][0, q_idx].cpu().numpy()
                for lag in range(W):
                    layer.attention_lags[h][t, lag] = attn_row[W - 1 - lag]

    return TransformerActivations(
        token_emb=token_emb,
        pos_emb=pos_emb,
        block_input=block_input,
        block_output=block_output,
        output_probs=output_probs,
        layers=layers,
        num_heads=num_heads,
        head_size=head_size,
        n_embd=n_embd,
        block_size=block_size,
    )


@torch.no_grad()
def transformer_closed_loop_rollout(
    model_dict: dict,
    *,
    seed_text: str,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Autoregressive rollout with per-step noise; returns block_output (T, n_embd) and chars."""
    torch_model: BigramLanguageModel = model_dict["_torch_model"]
    chars = model_dict["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    block_size = int(model_dict["block_size"])

    ids: list[int] = [stoi[c] for c in seed_text if c in stoi]
    if not ids:
        ids = [0]

    hidden_rows: list[np.ndarray] = []
    generated = list(seed_text)

    for _ in range(max(1, int(steps))):
        window = ids[-block_size:]
        x = torch.tensor([window], dtype=torch.long)
        acts = torch_model.forward_with_activations(x)
        hidden_rows.append(acts["block_output"][0, -1].cpu().numpy())

        logits = acts["logits"][0, -1]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        next_ix = int(rng.choice(len(chars), p=probs))
        ids.append(next_ix)
        generated.append(chars[next_ix])

    return np.asarray(hidden_rows, dtype=np.float64), generated


@torch.no_grad()
def forward_pass(model_dict: dict, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible helper: returns (block_output, output_probs)."""
    acts = extract_transformer_activations(model_dict, text)
    return acts.block_output, acts.output_probs


@torch.no_grad()
def extract_attention_matrix(
    model_dict: dict,
    text: str,
    *,
    layer_idx: int = 0,
    head_idx: int = 0,
) -> tuple[np.ndarray, str]:
    """Return causal softmax attention (T, T) from one forward pass.

    Row i is the attention distribution for query position i over keys 0..i.
    """
    torch_model: BigramLanguageModel = model_dict["_torch_model"]
    block_size = int(model_dict["block_size"])
    char_to_index = {c: i for i, c in enumerate(model_dict["chars"])}

    plot_text = text[:block_size]
    ids = [char_to_index[ch] for ch in plot_text]
    X = torch.tensor([ids], dtype=torch.long)
    acts = torch_model.forward_with_activations(X)
    attn = acts["layers"][layer_idx]["attention"][head_idx][0].cpu().numpy()
    row_sums = attn.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(f"attention rows do not softmax-normalize: {row_sums}")
    return attn, plot_text
