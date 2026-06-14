"""Character-level encoding and batching for transformer training."""

from __future__ import annotations

import random

import torch


def build_char_vocab(text: str) -> tuple[str, dict[str, int], dict[int, str]]:
    alphabet = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(alphabet)}
    itos = {i: ch for ch, i in stoi.items()}
    return alphabet, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(indices: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in indices)


def split_train_val(ids: list[int], train_ratio: float = 0.9) -> tuple[list[int], list[int]]:
    split = int(train_ratio * len(ids))
    return ids[:split], ids[split:]


def get_batch_from_ids(
    ids: list[int],
    block_size: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) < block_size + 1:
        raise ValueError(f"Corpus too short for block_size={block_size}")
    batch_x, batch_y = [], []
    for _ in range(batch_size):
        start = random.randint(0, len(ids) - block_size - 1)
        chunk = ids[start : start + block_size + 1]
        batch_x.append(chunk[:-1])
        batch_y.append(chunk[1:])
    return (
        torch.tensor(batch_x, dtype=torch.long),
        torch.tensor(batch_y, dtype=torch.long),
    )
