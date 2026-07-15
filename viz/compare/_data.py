"""Load per-task model + corpus context for cross-task comparison plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment import DEFAULT_SEED, TASKS, checkpoint_path
from task import corpus_for_experiment, label_extensions_for_experiment
from vocab_diagrams import select_analysis_window, vocabulary_for_experiment
from visualize import corpus_uses_word_spacing, load_model_for_viz, run_forward_pass


@dataclass
class TaskVizContext:
    task: str
    seed: int
    model: dict
    text: str
    words: list[str]
    spaced: bool
    hidden_states: np.ndarray


def load_task_viz_context(
    task: str,
    *,
    model_type: str = "rnn",
    seed: int | None = None,
    text_chars: int | None = None,
    checkpoint: Path | str | None = None,
) -> TaskVizContext:
    run_seed = DEFAULT_SEED if seed is None else seed
    ckpt = Path(checkpoint) if checkpoint is not None else checkpoint_path(task, model_type, seed=run_seed)
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing {model_type} checkpoint for task {task!r}: {ckpt}")

    model = load_model_for_viz(str(ckpt), model_type)
    words = vocabulary_for_experiment(task)
    cfg = TASKS[task]
    length = int(text_chars if text_chars is not None else cfg.get("viz_length", 50))

    full_text = corpus_for_experiment(task, seed=run_seed)
    spaced = corpus_uses_word_spacing(full_text, task)
    length = min(length, len(full_text))

    if words and not spaced:
        extensions = label_extensions_for_experiment(task)
        _win_start, text, _label_words = select_analysis_window(
            full_text, words, length, spaced=spaced, extensions=extensions,
        )
    else:
        text = full_text[:length]

    hidden_states, _output_probs = run_forward_pass(model, text, model_type)
    return TaskVizContext(
        task=task,
        seed=run_seed,
        model=model,
        text=text,
        words=words,
        spaced=spaced,
        hidden_states=np.asarray(hidden_states),
    )


def load_task_decoding_context(
    task: str,
    *,
    model_type: str = "rnn",
    seed: int | None = None,
) -> TaskVizContext:
    """Longer rollout than trajectory viz so decoding curves reach higher k."""
    cfg = TASKS[task]
    text_chars = int(cfg.get("metric_rollout_len", cfg.get("viz_length", 50)))
    return load_task_viz_context(
        task, model_type=model_type, seed=seed, text_chars=text_chars,
    )
