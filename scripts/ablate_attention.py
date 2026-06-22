"""Compare full model vs attention-ablated next-char accuracy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from experiment import input_path
from transformer.adapter import load_model, extract_attention_matrix


def ablate(exp: str, n: int = 40) -> dict:
    model_path = f"experiments/{exp}/transformer/model.pt"
    model = load_model(model_path)
    text = Path(f"experiments/{exp}/input.txt").read_text()[:n]

    ids = [model["chars"].index(c) for c in text]
    X = torch.tensor([ids])
    net = model["_torch_model"]

    def accuracy(logits: torch.Tensor) -> float:
        pred = logits[0].argmax(dim=-1).numpy()
        return sum(int(pred[t] == ids[t + 1]) for t in range(len(ids) - 1)) / (len(ids) - 1)

    with torch.no_grad():
        acts = net.forward_with_activations(X)
        acc_full = accuracy(acts["logits"])

        x = acts["token_emb"] + acts["pos_emb"]
        if net._legacy:
            raise NotImplementedError
        block = net.blocks[0]
        ln1_x = block.ln1(x)
        attn_out, _, _, _, _ = block.sa.heads[0].forward_with_qkv(ln1_x)
        post_attn = x + attn_out if net.use_residual else attn_out
        ffwd_out = block.ffwd(block.ln2(post_attn))
        post_ffwd = post_attn + ffwd_out if net.use_residual else ffwd_out
        acc_no_attn = accuracy(net.lm_head(net.ln_f(
            (x + block.ffwd(block.ln2(x))) if net.use_residual else block.ffwd(block.ln2(x))
        )))

        attn_only = block.ffwd(block.ln2(attn_out)) if not net.use_residual else None
        if attn_only is not None:
            acc_attn_only = accuracy(net.lm_head(net.ln_f(attn_only)))
        else:
            acc_attn_only = float("nan")

        acc_emb = accuracy(net.lm_head(net.ln_f(x)))

    attn, pt = extract_attention_matrix(model, text)
    last3_mass = []
    for i in range(2, len(pt)):
        row = attn[i, : i + 1]
        last3_mass.append(float(row[max(0, i - 2) : i + 1].sum()))
    mean_last3 = float(np.mean(last3_mass)) if last3_mass else 0.0

    return {
        "exp": exp,
        "n_embd": model["hidden_size"],
        "use_residual": bool(model["model_config"].get("use_residual", True)),
        "acc_full": acc_full,
        "acc_no_attn": acc_no_attn,
        "acc_emb_only": acc_emb,
        "acc_attn_only": acc_attn_only,
        "mean_last3_attn_mass": mean_last3,
    }


if __name__ == "__main__":
    for exp in ("ten_word_overlap_s", "sixteen_word_overlap_s"):
        path = Path(f"experiments/{exp}/transformer/model.pt")
        if not path.is_file():
            print(f"skip {exp}: no checkpoint")
            continue
        r = ablate(exp)
        print(f"\n=== {r['exp']} (n_embd={r['n_embd']}, residual={r['use_residual']}) ===")
        print(f"  full:      {100*r['acc_full']:.1f}%")
        print(f"  no attn:   {100*r['acc_no_attn']:.1f}%")
        print(f"  emb only:  {100*r['acc_emb_only']:.1f}%")
        print(f"  attn only: {100*r['acc_attn_only']:.1f}%")
        print(f"  attn gain: {100*(r['acc_full']-r['acc_no_attn']):.1f} pp")
        print(f"  mean mass on last-3 keys: {r['mean_last3_attn_mass']:.2f}")
