"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Per-timestep tensors are column vectors at time t:
  inputs_one_hot[t]    shape (vocab_size, 1)   -- one-hot encoding of the input char
  hidden_states[t]     shape (hidden_size, 1)  -- hidden state
  output_logits[t]     shape (vocab_size, 1)   -- unnormalized log-probabilities
  output_probs[t]      shape (vocab_size, 1)   -- softmax probabilities over the next char

Recurrence (forward):
  hidden_states[t] = tanh(
      weights_input_to_hidden  @ inputs_one_hot[t]    +
      weights_hidden_to_hidden @ hidden_states[t - 1] +
      bias_hidden
  )
  output_logits[t] = weights_hidden_to_output @ hidden_states[t] + bias_output
  output_probs[t]  = softmax(output_logits[t])
  loss_at_t        = -log(output_probs[t][target_index])
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnn.rnn_dyn import (
    activation_label,
    adagrad_step,
    flatten_weight_snapshot,
    hidden_activation,
    hidden_activation_backward,
    inject_timestep_noise,
    dale_violation_fraction,
    init_dale_weights,
    recurrent_pre_activation,
    rnn_hidden_step,
    sample_dale_signs,
    stable_softmax,
)
from experiment import experiment_regime
from task import REGIMES
from vocab_diagrams import invalid_word_fraction, segment_corpus_by_words

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=2000,
                    help='training iterations (default: 2000)')
parser.add_argument('--input', default='input.txt',
                    help='training corpus path (default: input.txt)')
parser.add_argument('--model', default='model.npz',
                    help='where to save trained weights (default: model.npz)')
parser.add_argument('--hidden-size', type=int, default=2,
                    help='number of recurrent units (default: 2)')
parser.add_argument('--dale', action='store_true',
                    help="enforce Dale's law on outgoing synapses (+/- per neuron)")
parser.add_argument('--e-fraction', type=float, default=0.8,
                    help='fraction of excitatory neurons when --dale (default: 0.8)')
parser.add_argument('--sequence-length', type=int, default=25,
                    help='BPTT window length in characters (default: 25)')
parser.add_argument('--learning-rate', type=float, default=None,
                    help='SGD step size (default: 0.1 tanh, 0.025 Dale)')
parser.add_argument('--exp', default=None,
                    help='experiment name; loads word list for unspaced word-error metrics')
parser.add_argument('--noise-std', type=float, default=None,
                    help='Gaussian noise std added to hidden state every timestep (train + sample)')
parser.add_argument('--seed', type=int, default=42,
                    help='RNG seed for weight initialization (default: 42)')
args = parser.parse_args()

# ----- data I/O ---------------------------------------------------------------
text = open(args.input, 'r').read()                 # entire training corpus as one string
unique_chars = list(set(text))                       # character vocabulary (order is arbitrary)
text_length, vocab_size = len(text), len(unique_chars)
print('data has %d characters, %d unique.' % (text_length, vocab_size))
char_to_index = { char: i for i, char in enumerate(unique_chars) }  # str -> int id
index_to_char = { i: char for i, char in enumerate(unique_chars) }  # int id -> str

# For word-space corpora, infer vocabulary from whitespace tokens; for unspaced
# experiment corpora, load the regime word list via --exp.
corpus_has_spaces = " " in text
if args.exp:
    vocab_words = set(REGIMES[experiment_regime(args.exp)])
elif corpus_has_spaces:
    vocab_words = set(text.split())
else:
    vocab_words = set()
use_word_segmentation = bool(vocab_words) and not corpus_has_spaces

# ----- hyperparameters --------------------------------------------------------
hidden_size = args.hidden_size  # number of recurrent units in the hidden layer
sequence_length = max(1, int(args.sequence_length))
dale_law = bool(args.dale)
learning_rate = args.learning_rate if args.learning_rate is not None else (
    2.5e-2 if dale_law else 1e-1
)
use_relu = dale_law        # Dale's law: nonnegative activity, fixed outgoing sign
e_fraction = float(args.e_fraction)
if args.noise_std is not None:
    timestep_noise_std = float(args.noise_std)
elif args.exp:
    from experiment import EXPERIMENT_CONFIG
    timestep_noise_std = float(EXPERIMENT_CONFIG.get(args.exp, {}).get("timestep_noise_std", 0.0))
else:
    timestep_noise_std = 0.0
noise_rng = np.random.default_rng(args.seed + 1)
init_rng = np.random.default_rng(args.seed)

# ----- model parameters -------------------------------------------------------
dale_sign = None
if dale_law:
    dale_sign = sample_dale_signs(hidden_size, e_fraction, init_rng)
    # Smaller hidden layers need slightly larger init to escape the uniform regime.
    dale_init_scale = 0.01 if hidden_size <= 32 else 0.005
    (weights_input_to_hidden,
     weights_hidden_to_hidden,
     weights_hidden_to_output) = init_dale_weights(
        hidden_size, vocab_size, dale_sign, scale=dale_init_scale, rng=init_rng,
    )
    n_exc = int(np.sum(dale_sign > 0))
    exc_range = f"h0..h{n_exc - 1}" if n_exc > 0 else "(none)"
    inh_range = f"h{n_exc}..h{hidden_size - 1}" if n_exc < hidden_size else "(none)"
    print(f"Dale's law: {n_exc} excitatory ({exc_range}), "
          f"{hidden_size - n_exc} inhibitory ({inh_range}), "
          f"E fraction {n_exc / hidden_size:.2f}, activation={activation_label(use_relu=use_relu)}")
else:
    weights_input_to_hidden = init_rng.standard_normal((hidden_size, vocab_size)) * 0.01
    weights_hidden_to_hidden = init_rng.standard_normal((hidden_size, hidden_size)) * 0.01
    weights_hidden_to_output = init_rng.standard_normal((vocab_size, hidden_size)) * 0.01

if timestep_noise_std > 0:
    print(f"timestep noise std: {timestep_noise_std}")

bias_hidden = np.zeros((hidden_size, 1))
bias_output = np.zeros((vocab_size, 1))


def compute_loss_and_gradients(input_indices, target_indices, previous_hidden_state):
  """
  Forward + backward pass over ONE backprop-through-time window of length `sequence_length`.

  Args:
    input_indices:         list[int] of length sequence_length -- input char ids at each timestep
    target_indices:        list[int] of length sequence_length -- next-char ids the model should predict
    previous_hidden_state: ndarray (hidden_size, 1) -- hidden state inherited from the previous window
                           (this is what makes the recurrence carry information across windows)

  Returns:
    loss (float):   summed cross-entropy over the predictions in this window
    grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
    grad_bias_hidden, grad_bias_output: gradients of `loss` w.r.t. each parameter (same shape as the param)
    last hidden state (hidden_size, 1): hidden_states[sequence_length - 1], used to seed the next window
  """
  # Per-timestep caches kept around so we can reuse the activations during backprop.
  # Indexed by t (the time step within this window). hidden_states[-1] holds the carried-over state.
  inputs_one_hot, hidden_states, pre_activations, output_logits, output_probs = {}, {}, {}, {}, {}
  hidden_states[-1] = np.copy(previous_hidden_state)
  loss = 0

  # ----- forward pass: t = 0, 1, ..., sequence_length - 1 ---------------------
  for t in range(len(input_indices)):
    # One-hot encode the input char so weights_input_to_hidden @ inputs_one_hot[t]
    # effectively selects column input_indices[t] of weights_input_to_hidden.
    inputs_one_hot[t] = np.zeros((vocab_size, 1))
    inputs_one_hot[t][input_indices[t]] = 1

    pre_activations[t] = recurrent_pre_activation(
        inputs_one_hot[t], hidden_states[t - 1],
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
    )
    hidden_states[t] = hidden_activation(pre_activations[t], use_relu=use_relu)
    hidden_states[t] = inject_timestep_noise(hidden_states[t], timestep_noise_std, noise_rng)

    # Read out logits, then softmax to get a probability distribution over the vocab.
    output_logits[t] = np.dot(weights_hidden_to_output, hidden_states[t]) + bias_output
    output_probs[t] = stable_softmax(output_logits[t]).reshape(-1, 1)

    # Cross-entropy: penalize the negative log-probability the model assigned to the correct next char.
    # If the model is surprised (prob of target is small), the loss is large; if prob ~ 1, loss ~ 0.
    p = float(output_probs[t][target_indices[t], 0])
    loss += -np.log(max(p, 1e-12))

  # ----- backward pass: BPTT, t = sequence_length - 1, ..., 0 -----------------
  # We need d(loss)/d(param) for every learnable parameter. Each parameter is shared
  # across all timesteps, so its gradient is the *sum* of contributions from every t.
  # We accumulate those sums into these buffers, starting from zero.
  grad_weights_input_to_hidden  = np.zeros_like(weights_input_to_hidden)   # shape (hidden_size, vocab_size)
  grad_weights_hidden_to_hidden = np.zeros_like(weights_hidden_to_hidden)  # shape (hidden_size, hidden_size)
  grad_weights_hidden_to_output = np.zeros_like(weights_hidden_to_output)  # shape (vocab_size, hidden_size)
  grad_bias_hidden = np.zeros_like(bias_hidden)                            # shape (hidden_size, 1)
  grad_bias_output = np.zeros_like(bias_output)                            # shape (vocab_size, 1)

  # grad_hidden_next carries d(loss)/d(hidden_states[t+1]) backwards across timesteps.
  # At the last timestep there's no "future" contribution yet, so we start with zeros
  # and accumulate as we unroll backward.
  grad_hidden_next = np.zeros_like(hidden_states[0])

  for t in reversed(range(len(input_indices))):
    # ---- (1) gradient w.r.t. the output logits -----------------------------
    # For softmax + cross-entropy, the gradient w.r.t. the logits simplifies beautifully to:
    #     d(loss)/d(output_logits[t]) = output_probs[t] - one_hot(target_indices[t])
    # See http://cs231n.github.io/neural-networks-case-study/#grad for the derivation.
    grad_output = np.copy(output_probs[t])
    grad_output[target_indices[t]] -= 1              # subtract one-hot(target) from the prob vector

    # ---- (2) push that gradient into weights_hidden_to_output and bias_output ----
    # output_logits[t] = weights_hidden_to_output @ hidden_states[t] + bias_output
    #   => grad_weights_hidden_to_output += grad_output @ hidden_states[t].T   (outer product)
    #      grad_bias_output              += grad_output
    grad_weights_hidden_to_output += np.dot(grad_output, hidden_states[t].T)
    grad_bias_output              += grad_output

    # ---- (3) backprop into the hidden state hidden_states[t] ---------------
    # hidden_states[t] affects the loss two ways:
    #   (a) directly through output_logits[t]
    #         contribution: weights_hidden_to_output.T @ grad_output
    #   (b) indirectly through hidden_states[t+1] (via the recurrence)
    #         contribution carried in grad_hidden_next
    # Sum both -> the *total* gradient flowing back into hidden_states[t].
    grad_hidden = np.dot(weights_hidden_to_output.T, grad_output) + grad_hidden_next

    # ---- (4) backprop through the hidden nonlinearity ----------------------
    grad_hidden_raw = hidden_activation_backward(
        grad_hidden, pre_activations[t], hidden_states[t], use_relu=use_relu,
    )

    # ---- (5) push that into the pre-activation parameters ------------------
    # pre_activation = weights_input_to_hidden  @ inputs_one_hot[t]
    #                + weights_hidden_to_hidden @ hidden_states[t-1]
    #                + bias_hidden
    # so:
    #   grad_bias_hidden               += grad_hidden_raw
    #   grad_weights_input_to_hidden   += grad_hidden_raw @ inputs_one_hot[t].T   (outer product)
    #   grad_weights_hidden_to_hidden  += grad_hidden_raw @ hidden_states[t-1].T  (outer product)
    grad_bias_hidden              += grad_hidden_raw
    grad_weights_input_to_hidden  += np.dot(grad_hidden_raw, inputs_one_hot[t].T)
    grad_weights_hidden_to_hidden += np.dot(grad_hidden_raw, hidden_states[t-1].T)

    # ---- (6) carry the gradient into the previous timestep -----------------
    # hidden_states[t-1] influences hidden_states[t] only through
    #     pre_activation = ... + weights_hidden_to_hidden @ hidden_states[t-1] + ...
    # so this iteration's contribution to d(loss)/d(hidden_states[t-1]) is:
    #     weights_hidden_to_hidden.T @ grad_hidden_raw
    # Stash it so the next loop iteration (which handles t-1) can add it to its grad_hidden.
    grad_hidden_next = np.dot(weights_hidden_to_hidden.T, grad_hidden_raw)

  # Clip gradients elementwise to [-5, 5] to mitigate exploding gradients,
  # a well-known pathology of vanilla RNNs when backpropagating through many timesteps.
  for grad in [grad_weights_input_to_hidden, grad_weights_hidden_to_hidden,
               grad_weights_hidden_to_output, grad_bias_hidden, grad_bias_output]:
    np.clip(grad, -5, 5, out=grad)

  return (loss,
          grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
          grad_bias_hidden, grad_bias_output,
          hidden_states[len(input_indices) - 1])


def sample(
    hidden_state,
    seed_index,
    num_chars_to_sample,
    *,
    rng: np.random.Generator | None = None,
):
  """
  Stochastic generation: draw each next char from the model's softmax (never argmax).
  """
  rng = rng or np.random.default_rng()
  input_one_hot = np.zeros((vocab_size, 1))
  input_one_hot[seed_index] = 1
  sampled_indices = []
  for _ in range(num_chars_to_sample):
    hidden_state, _ = rnn_hidden_step(
        hidden_state, input_one_hot,
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
        use_relu=use_relu,
        timestep_noise_std=timestep_noise_std,
        noise_rng=noise_rng,
    )
    logits = np.dot(weights_hidden_to_output, hidden_state) + bias_output
    probs = stable_softmax(logits)
    next_char_index = int(rng.choice(range(vocab_size), p=probs.ravel()))
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[next_char_index] = 1
    sampled_indices.append(next_char_index)
  return sampled_indices


def argmax_sample(hidden_state, seed_index, num_chars_to_sample):
  """Deterministic sampling: take argmax at each step (for stable metrics)."""
  input_one_hot = np.zeros((vocab_size, 1))
  input_one_hot[seed_index] = 1
  sampled_indices = []
  for t in range(num_chars_to_sample):
    hidden_state, _ = rnn_hidden_step(
        hidden_state, input_one_hot,
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
        use_relu=use_relu,
        timestep_noise_std=timestep_noise_std,
        noise_rng=noise_rng,
    )
    logits = np.dot(weights_hidden_to_output, hidden_state) + bias_output
    probs = stable_softmax(logits)
    next_char_index = int(np.argmax(probs))
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[next_char_index] = 1
    sampled_indices.append(next_char_index)
  return sampled_indices


def argmax_sample_with_prompt(prompt_text: str, num_chars_to_sample: int):
  """
  Deterministic sampling, conditioned on a prompt sequence.
  We "teacher-force" the prompt by feeding its characters as the next inputs,
  then continue with argmax sampling for `num_chars_to_sample` more characters.
  Returns indices for the continuation only (not including the prompt itself).
  """
  if not prompt_text:
    return []
  hidden_state = np.zeros((hidden_size, 1))
  input_one_hot = np.zeros((vocab_size, 1))
  input_one_hot[char_to_index[prompt_text[0]]] = 1

  # Consume prompt with teacher forcing.
  for ch_next in prompt_text[1:]:
    hidden_state, _ = rnn_hidden_step(
        hidden_state, input_one_hot,
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
        use_relu=use_relu,
        timestep_noise_std=timestep_noise_std,
        noise_rng=noise_rng,
    )
    # Advance input to the true next prompt char.
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[char_to_index[ch_next]] = 1

  # Generate continuation.
  sampled_indices = []
  for _ in range(num_chars_to_sample):
    hidden_state, _ = rnn_hidden_step(
        hidden_state, input_one_hot,
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
        use_relu=use_relu,
        timestep_noise_std=timestep_noise_std,
        noise_rng=noise_rng,
    )
    logits = np.dot(weights_hidden_to_output, hidden_state) + bias_output
    probs = stable_softmax(logits)
    next_char_index = int(np.argmax(probs))
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[next_char_index] = 1
    sampled_indices.append(next_char_index)
  return sampled_indices


def sample_with_prompt(prompt_text: str, num_chars_to_sample: int, *, rng: np.random.Generator):
  """
  Stochastic sampling, conditioned on a prompt sequence (teacher-forced prompt).
  Returns indices for the continuation only (not including the prompt itself).
  """
  if not prompt_text:
    return []
  hidden_state = np.zeros((hidden_size, 1))
  input_one_hot = np.zeros((vocab_size, 1))
  input_one_hot[char_to_index[prompt_text[0]]] = 1

  # Consume prompt with teacher forcing.
  for ch_next in prompt_text[1:]:
    hidden_state, _ = rnn_hidden_step(
        hidden_state, input_one_hot,
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
        use_relu=use_relu,
        timestep_noise_std=timestep_noise_std,
        noise_rng=noise_rng,
    )
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[char_to_index[ch_next]] = 1

  sampled_indices = []
  for _ in range(num_chars_to_sample):
    hidden_state, _ = rnn_hidden_step(
        hidden_state, input_one_hot,
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
        use_relu=use_relu,
        timestep_noise_std=timestep_noise_std,
        noise_rng=noise_rng,
    )
    logits = np.dot(weights_hidden_to_output, hidden_state) + bias_output
    probs = stable_softmax(logits)
    next_char_index = int(rng.choice(range(vocab_size), p=probs.ravel()))
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[next_char_index] = 1
    sampled_indices.append(next_char_index)
  return sampled_indices


def sample_from_seed_char(seed_char: str, num_chars_to_sample: int, *, rng: np.random.Generator):
  """Stochastic sampling from zero state, seeded by a single character."""
  if seed_char not in char_to_index:
    seed_char = text[0]
  hidden_state = np.zeros((hidden_size, 1))
  input_one_hot = np.zeros((vocab_size, 1))
  input_one_hot[char_to_index[seed_char]] = 1
  sampled_indices = []
  for _ in range(num_chars_to_sample):
    hidden_state, _ = rnn_hidden_step(
        hidden_state, input_one_hot,
        weights_input_to_hidden, weights_hidden_to_hidden, bias_hidden,
        use_relu=use_relu,
        timestep_noise_std=timestep_noise_std,
        noise_rng=noise_rng,
    )
    logits = np.dot(weights_hidden_to_output, hidden_state) + bias_output
    probs = stable_softmax(logits)
    next_char_index = int(rng.choice(range(vocab_size), p=probs.ravel()))
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[next_char_index] = 1
    sampled_indices.append(next_char_index)
  return sampled_indices


def stochastic_word_validity_metrics(
    seed_index: int,
    vocab: set[str],
    *,
    rng: np.random.Generator,
) -> tuple[float, float, str]:
    """Mean invalid-word rate and in-vocab letter frac over several long rollouts."""
    word_errs: list[float] = []
    letter_fracs: list[float] = []
    first_text = ""
    for r in range(METRIC_NUM_ROLLOUTS):
        h0 = np.zeros((hidden_size, 1))
        indices = sample(h0, seed_index, METRIC_ROLLOUT_LEN, rng=rng)
        text = "".join(index_to_char[i] for i in indices)
        if r == 0:
            first_text = text
        word_errs.append(invalid_word_fraction(text, vocab))
        letter_fracs.append(valid_vocab_letter_fraction(text, vocab))
    word_err = float(np.nanmean(word_errs))
    letter_frac = float(np.nanmean(letter_fracs))
    return word_err, letter_frac, first_text


def invalid_word_fraction(sampled_text: str, vocab: set[str]) -> float:
    """Fraction of words (split or segmented) not in the training vocabulary."""
    from vocab_diagrams import invalid_word_fraction as _invalid_word_fraction

    return _invalid_word_fraction(
        sampled_text, vocab, spaced=not use_word_segmentation, trim_edges=True,
    )


def valid_vocab_letter_fraction(sampled_text: str, vocab: set[str]) -> float:
  """Fraction of letters that belong to in-vocabulary words."""
  if not vocab:
    return float("nan")
  if use_word_segmentation:
    tokens = [seg[2] for seg in segment_corpus_by_words(sampled_text, vocab)]
  else:
    tokens = [t for t in sampled_text.split(" ") if t]
  total_letters = sum(len(token) for token in tokens)
  valid_letters = sum(len(token) for token in tokens if token in vocab)
  return (valid_letters / total_letters) if total_letters > 0 else float("nan")


# ----- training loop ----------------------------------------------------------
iteration, data_pointer = 0, 0

# Adagrad accumulators: same shape as each parameter. They keep a running sum of squared
# gradients, which is used to give each parameter its own adaptive (per-coordinate) learning rate.
mem_weights_input_to_hidden  = np.zeros_like(weights_input_to_hidden)
mem_weights_hidden_to_hidden = np.zeros_like(weights_hidden_to_hidden)
mem_weights_hidden_to_output = np.zeros_like(weights_hidden_to_output)
mem_bias_hidden = np.zeros_like(bias_hidden)
mem_bias_output = np.zeros_like(bias_output)

# A reasonable starting value: -log(1 / vocab_size) * sequence_length is the expected
# cross-entropy if the model is uniform over the vocab and we sum over the BPTT window.
# We track an exponential moving average so the printed loss isn't noisy window-to-window.
smooth_loss = -np.log(1.0 / vocab_size) * sequence_length
max_iterations = args.steps
loss_iterations = []
loss_smooth = []
loss_window = []

# Extra metric logged during training (not optimized directly): fraction of letters
# that land inside an in-vocabulary word in a short model rollout.
metric_iters = []
metric_valid_letter_frac = []
metric_word_error_frac = []

weight_snap_iters: list[int] = []
weight_snap_outgoing: list[np.ndarray] = []
weight_snap_bias_hidden: list[np.ndarray] = []
weight_snap_bias_output: list[np.ndarray] = []
weight_snap_violation_frac: list[float] = []
WEIGHT_SNAP_EVERY = 100
WEIGHT_SNAP_ULTRA_UNTIL = 300
WEIGHT_SNAP_DENSE_EVERY = 3
WEIGHT_SNAP_DENSE_UNTIL = 1_200
METRIC_ROLLOUT_LEN = 3_000
METRIC_NUM_ROLLOUTS = 5
METRIC_RNG_BASE = 42


def _should_record_weight_snapshot(iteration: int) -> bool:
    if iteration % WEIGHT_SNAP_EVERY == 0:
        return True
    if iteration <= WEIGHT_SNAP_ULTRA_UNTIL:
        return True
    return (
        iteration <= WEIGHT_SNAP_DENSE_UNTIL
        and iteration % WEIGHT_SNAP_DENSE_EVERY == 0
    )


def _append_weight_snapshot(iteration: int) -> None:
    weight_snap_iters.append(iteration)
    weight_snap_outgoing.append(
        flatten_weight_snapshot(
            weights_input_to_hidden,
            weights_hidden_to_hidden,
            weights_hidden_to_output,
        )
    )
    weight_snap_bias_hidden.append(np.copy(bias_hidden))
    weight_snap_bias_output.append(np.copy(bias_output))
    if dale_law:
        weight_snap_violation_frac.append(
            dale_violation_fraction(
                weights_input_to_hidden,
                weights_hidden_to_hidden,
                weights_hidden_to_output,
                dale_sign,
            )
        )
    else:
        weight_snap_violation_frac.append(0.0)


# Ignore early rollouts before real learning (checkpoint only on 0% invalid-word rate).
MIN_CHECKPOINT_ITER = 8_000

best_word_err = float("inf")
best_valid_letter_frac = -1.0
best_iter = -1
best_state: dict[str, np.ndarray] | None = None
zero_word_err_streak = 0


def snapshot_params() -> dict[str, np.ndarray]:
    return {
        "weights_input_to_hidden": np.copy(weights_input_to_hidden),
        "weights_hidden_to_hidden": np.copy(weights_hidden_to_hidden),
        "weights_hidden_to_output": np.copy(weights_hidden_to_output),
        "bias_hidden": np.copy(bias_hidden),
        "bias_output": np.copy(bias_output),
    }


def restore_params(state: dict[str, np.ndarray]) -> None:
    global weights_input_to_hidden, weights_hidden_to_hidden, weights_hidden_to_output
    global bias_hidden, bias_output
    weights_input_to_hidden = state["weights_input_to_hidden"]
    weights_hidden_to_hidden = state["weights_hidden_to_hidden"]
    weights_hidden_to_output = state["weights_hidden_to_output"]
    bias_hidden = state["bias_hidden"]
    bias_output = state["bias_output"]

# Store a short "before vs after" deterministic sample for visualization.
sample_before_text = None
sample_after_text = None

# Fixed-length snippets for samples_before_after.png (same length as viz window).
DEMO_SNIPPET_LEN = 50
demo_snippet = text[:DEMO_SNIPPET_LEN]
demo_before = None
demo_after = None
demo_word_error_frac = float("nan")
demo_rng_seed = 0
demo_seed_char = " " if (" " in char_to_index) else text[0]

while iteration < max_iterations:
  if iteration == 0:
    _append_weight_snapshot(0)

  # Step the data pointer through the corpus in chunks of `sequence_length`.
  # If we run off the end (or we're on iteration 0), reset the hidden state and wrap to the start.
  if data_pointer + sequence_length + 1 >= len(text) or iteration == 0:
    previous_hidden_state = np.zeros((hidden_size, 1))   # reset RNN memory across wraps
    data_pointer = 0

  # Inputs and targets are the same window, shifted by one char.
  # For each t in [0, sequence_length), the model sees text[data_pointer + t]
  # and must predict text[data_pointer + t + 1].
  input_indices  = [char_to_index[char] for char in text[data_pointer    : data_pointer + sequence_length    ]]
  target_indices = [char_to_index[char] for char in text[data_pointer + 1: data_pointer + sequence_length + 1]]

  # Every 100 iterations, draw a short sample from the model to see what it's learning.
  if iteration % 100 == 0:
    sampled_indices = sample(previous_hidden_state, input_indices[0], 50)
    sampled_text = ''.join(index_to_char[i] for i in sampled_indices)
    print('----\n %s \n----' % (sampled_text,))

    metric_seed = char_to_index[demo_seed_char]
    metric_rng = np.random.default_rng(METRIC_RNG_BASE + iteration)
    word_err, letter_frac, rollout_text = stochastic_word_validity_metrics(
        metric_seed, vocab_words, rng=metric_rng,
    )
    metric_iters.append(iteration)
    metric_valid_letter_frac.append(letter_frac)
    metric_word_error_frac.append(word_err)

    if (
        iteration >= MIN_CHECKPOINT_ITER
        and vocab_words
        and np.isfinite(word_err)
    ):
        if word_err <= 1e-12:
            best_word_err = 0.0
            best_valid_letter_frac = letter_frac
            best_iter = iteration
            best_state = copy.deepcopy(snapshot_params())
        if iteration >= MIN_CHECKPOINT_ITER and word_err <= 1e-12:
            zero_word_err_streak += 1
        else:
            zero_word_err_streak = 0
        if zero_word_err_streak >= 3:
            print(
                f"early stop at iter {iteration}: "
                "0 invalid vocabulary words for 300 iterations",
            )
            break

    if iteration == 0:
      sample_before_text = rollout_text[:DEMO_SNIPPET_LEN]
      demo_rng_seed = METRIC_RNG_BASE
      demo_before = rollout_text[:DEMO_SNIPPET_LEN]

  # Forward + backward over the window. previous_hidden_state is updated to the last
  # hidden state of *this* window, so the next iteration continues the recurrence smoothly.
  (loss,
   grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
   grad_bias_hidden, grad_bias_output,
   previous_hidden_state) = compute_loss_and_gradients(input_indices, target_indices, previous_hidden_state)

  # Exponential moving average of the loss for smoother printing.
  if np.isfinite(loss):
    smooth_loss = smooth_loss * 0.995 + loss * 0.005
  loss_iterations.append(iteration)
  loss_smooth.append(smooth_loss)
  loss_window.append(loss)
  if iteration % 100 == 0:
    print('iter %d, loss: %f' % (iteration, smooth_loss))

  # ----- Adagrad parameter update --------------------------------------------
  effective_lr = learning_rate
  if dale_law and iteration > 0:
      effective_lr = learning_rate * (0.9998 ** (iteration / 100.0))
  adagrad_step(
      weights_input_to_hidden, grad_weights_input_to_hidden, mem_weights_input_to_hidden,
      effective_lr,
      dale_sign=dale_sign if dale_law else None,
      dale_axis="row",
  )
  adagrad_step(
      weights_hidden_to_hidden, grad_weights_hidden_to_hidden, mem_weights_hidden_to_hidden,
      effective_lr,
      dale_sign=dale_sign if dale_law else None,
      dale_axis="col",
  )
  adagrad_step(
      weights_hidden_to_output, grad_weights_hidden_to_output, mem_weights_hidden_to_output,
      effective_lr,
      dale_sign=dale_sign if dale_law else None,
      dale_axis="col",
  )
  adagrad_step(bias_hidden, grad_bias_hidden, mem_bias_hidden, effective_lr)
  adagrad_step(bias_output, grad_bias_output, mem_bias_output, effective_lr)

  data_pointer += sequence_length
  iteration += 1
  if iteration > 0 and _should_record_weight_snapshot(iteration):
    _append_weight_snapshot(iteration)

# Final sample after training finishes, plus the last smoothed loss.
sampled_indices = sample(previous_hidden_state, char_to_index[text[0]], 50)
sampled_text = ''.join(index_to_char[i] for i in sampled_indices)
print('----\n %s \n----' % (sampled_text,))
print('iter %d, loss: %f (done)' % (iteration, smooth_loss))

if (
    best_state is not None
    and best_iter >= MIN_CHECKPOINT_ITER
    and best_word_err <= 1e-12
):
    print(f"using checkpoint from iter {best_iter} (0% invalid vocabulary words)")
    restore_params(best_state)
else:
    print("keeping final weights (no 0% invalid-word checkpoint was reached)")

final_rng = np.random.default_rng(METRIC_RNG_BASE + iteration)
final_word_err, final_letter_valid, rollout_text = stochastic_word_validity_metrics(
    char_to_index[demo_seed_char], vocab_words, rng=final_rng,
)
sample_after_text = rollout_text[:DEMO_SNIPPET_LEN]
demo_rng_seed = METRIC_RNG_BASE + iteration
demo_after = rollout_text[:DEMO_SNIPPET_LEN]
demo_word_error_frac = invalid_word_fraction(rollout_text, vocab_words)

print(
    f"final word error rate (mean over {METRIC_NUM_ROLLOUTS} rollouts × "
    f"{METRIC_ROLLOUT_LEN} chars, stochastic): {100.0 * final_word_err:.2f}%",
)
print(
    f"final demo rollout ({len(rollout_text)} chars, seed {demo_rng_seed}): "
    f"{100.0 * demo_word_error_frac:.2f}% invalid words",
)
print(f"final in-vocab letter fraction: {100.0 * final_letter_valid:.2f}%")
if dale_law:
    viol = dale_violation_fraction(
        weights_input_to_hidden,
        weights_hidden_to_hidden,
        weights_hidden_to_output,
        dale_sign,
    )
    print(f"final Dale violation fraction (all constrained synapses): {100.0 * viol:.4f}%")

# Save trained parameters and vocab so we can inspect/visualize the model later.
model_out = args.model
model_out_parent = os.path.dirname(model_out)
if model_out_parent:
    os.makedirs(model_out_parent, exist_ok=True)
np.savez(
    model_out,
    weights_input_to_hidden=weights_input_to_hidden,
    weights_hidden_to_hidden=weights_hidden_to_hidden,
    weights_hidden_to_output=weights_hidden_to_output,
    bias_hidden=bias_hidden,
    bias_output=bias_output,
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
    weight_snap_iterations=np.array(weight_snap_iters, dtype=np.int32),
    weight_snap_outgoing=np.array(weight_snap_outgoing, dtype=np.float64),
    weight_snap_bias_hidden=np.array(weight_snap_bias_hidden, dtype=np.float64),
    weight_snap_bias_output=np.array(weight_snap_bias_output, dtype=np.float64),
    weight_snap_violation_frac=np.array(weight_snap_violation_frac, dtype=np.float64),
    vocab_words=np.array(sorted(vocab_words)),
    sample_before=np.array(sample_before_text if sample_before_text is not None else ""),
    sample_after=np.array(sample_after_text if sample_after_text is not None else ""),
    demo_snippet=np.array(demo_snippet if demo_snippet is not None else ""),
    demo_before=np.array(demo_before if demo_before is not None else ""),
    demo_after=np.array(demo_after if demo_after is not None else ""),
    demo_word_error_frac=np.array(demo_word_error_frac, dtype=np.float64),
    demo_rng_seed=np.array(demo_rng_seed, dtype=np.int64),
    demo_seed_char=np.array(demo_seed_char),
    dale_law=np.array(dale_law),
    use_relu=np.array(use_relu),
    e_fraction=np.array(e_fraction),
    dale_sign=np.array(dale_sign if dale_sign is not None else []),
    timestep_noise_std=np.array(timestep_noise_std, dtype=np.float64),
)
print(f'saved trained model to {model_out}')
