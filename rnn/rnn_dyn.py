"""Shared RNN dynamics: activations and Dale's-law weight constraints."""

from __future__ import annotations

import numpy as np

PRE_ACTIVATION_CLIP = 50.0
DEFAULT_DALE_SOFT_EPS = 0.02


def hidden_activation(pre_activation: np.ndarray, *, use_relu: bool) -> np.ndarray:
    if use_relu:
        pre = np.clip(pre_activation, -PRE_ACTIVATION_CLIP, PRE_ACTIVATION_CLIP)
        return np.maximum(0.0, pre)
    return np.tanh(pre_activation)


def hidden_activation_backward(
    grad_hidden: np.ndarray,
    pre_activation: np.ndarray,
    hidden_state: np.ndarray,
    *,
    use_relu: bool,
) -> np.ndarray:
    if use_relu:
        return grad_hidden * (pre_activation > 0)
    return grad_hidden * (1.0 - hidden_state * hidden_state)


def recurrent_pre_activation(
    input_one_hot: np.ndarray,
    hidden_state: np.ndarray,
    weights_input_to_hidden: np.ndarray,
    weights_hidden_to_hidden: np.ndarray,
    bias_hidden: np.ndarray,
) -> np.ndarray:
    return (
        weights_input_to_hidden @ input_one_hot
        + weights_hidden_to_hidden @ hidden_state
        + bias_hidden
    )


def sample_dale_signs(hidden_size: int, e_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Per-neuron Dale sign; hidden indices 0..n_E-1 excitatory, n_E..H-1 inhibitory."""
    del rng  # order is fixed (E block then I block), not randomized
    e_fraction = float(np.clip(e_fraction, 0.0, 1.0))
    n_exc = int(round(hidden_size * e_fraction))
    n_exc = min(max(n_exc, 0), hidden_size)
    signs = np.full(hidden_size, -1.0, dtype=np.float64)
    signs[:n_exc] = 1.0
    return signs


def dale_signs_ordered(dale_sign: np.ndarray) -> bool:
    """True if E units occupy 0..n_E-1 and I units the rest."""
    n_exc = int(np.sum(dale_sign > 0))
    if n_exc == 0 or n_exc == len(dale_sign):
        return True
    return bool(np.all(dale_sign[:n_exc] > 0) and np.all(dale_sign[n_exc:] < 0))


def permute_hidden_by_dale(
    weights_input_to_hidden: np.ndarray,
    weights_hidden_to_hidden: np.ndarray,
    weights_hidden_to_output: np.ndarray,
    bias_hidden: np.ndarray,
    dale_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reorder hidden units E-first, I-last (for legacy checkpoints)."""
    if dale_signs_ordered(dale_sign):
        return (
            weights_input_to_hidden,
            weights_hidden_to_hidden,
            weights_hidden_to_output,
            bias_hidden,
            dale_sign,
        )
    perm = np.concatenate([np.flatnonzero(dale_sign > 0), np.flatnonzero(dale_sign <= 0)])
    return (
        weights_input_to_hidden[perm],
        weights_hidden_to_hidden[perm][:, perm],
        weights_hidden_to_output[:, perm],
        bias_hidden[perm],
        dale_sign[perm],
    )


def init_dale_weights(
    hidden_size: int,
    vocab_size: int,
    dale_sign: np.ndarray,
    *,
    scale: float = 0.01,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Init W_xh rows, W_hh/W_ho columns with magnitudes × neuron Dale sign."""
    rng = rng or np.random.default_rng()
    row_sign = dale_sign.reshape(-1, 1)
    col_sign = dale_sign.reshape(1, -1)
    weights_input_to_hidden = (
        np.abs(rng.standard_normal((hidden_size, vocab_size))) * scale * row_sign
    )
    weights_hidden_to_hidden = (
        np.abs(rng.standard_normal((hidden_size, hidden_size))) * scale * col_sign
    )
    weights_hidden_to_output = (
        np.abs(rng.standard_normal((vocab_size, hidden_size))) * scale * col_sign
    )
    enforce_dale_weights(
        weights_input_to_hidden,
        weights_hidden_to_hidden,
        weights_hidden_to_output,
        dale_sign,
    )
    return weights_input_to_hidden, weights_hidden_to_hidden, weights_hidden_to_output


def init_dale_outgoing_weights(
    hidden_size: int,
    vocab_size: int,
    dale_sign: np.ndarray,
    *,
    scale: float = 0.01,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias for init_dale_weights."""
    return init_dale_weights(hidden_size, vocab_size, dale_sign, scale=scale, rng=rng)


def enforce_dale_weights(
    weights_input_to_hidden: np.ndarray,
    weights_hidden_to_hidden: np.ndarray,
    weights_hidden_to_output: np.ndarray,
    dale_sign: np.ndarray,
) -> None:
    """Hard projection (init / diagnostics). Training uses soft Dale in adagrad_step."""
    row_sign = dale_sign.reshape(-1, 1)
    col_sign = dale_sign.reshape(1, -1)
    weights_input_to_hidden[:] = np.abs(weights_input_to_hidden) * row_sign
    weights_hidden_to_hidden[:] = np.abs(weights_hidden_to_hidden) * col_sign
    weights_hidden_to_output[:] = np.abs(weights_hidden_to_output) * col_sign


def enforce_dale_outgoing(
    weights_hidden_to_hidden: np.ndarray,
    weights_hidden_to_output: np.ndarray,
    dale_sign: np.ndarray,
) -> None:
    """Hard projection on recurrent/output columns only."""
    col_sign = dale_sign.reshape(1, -1)
    weights_hidden_to_hidden[:] = np.abs(weights_hidden_to_hidden) * col_sign
    weights_hidden_to_output[:] = np.abs(weights_hidden_to_output) * col_sign


def soft_dale_step_scale(
    weights: np.ndarray,
    delta: np.ndarray,
    dale_sign: np.ndarray,
    *,
    axis: str = "col",
    soft_eps: float = DEFAULT_DALE_SOFT_EPS,
) -> np.ndarray:
    """Scale Adagrad updates on Dale-constrained rows (W_xh) or columns (W_hh, W_ho)."""
    if axis == "col":
        signs = dale_sign.reshape(1, -1)
    elif axis == "row":
        signs = dale_sign.reshape(-1, 1)
    else:
        raise ValueError(f"dale axis must be 'row' or 'col', got {axis!r}")
    margin = weights * signs
    base = np.tanh(margin / soft_eps)
    opposed = delta * signs < 0
    step = np.where(
        opposed,
        np.tanh(margin / (np.abs(delta) + soft_eps)),
        1.0,
    )
    return base * step


def adagrad_step(
    param: np.ndarray,
    grad: np.ndarray,
    mem: np.ndarray,
    learning_rate: float,
    *,
    dale_sign: np.ndarray | None = None,
    dale_axis: str = "col",
    soft_eps: float = DEFAULT_DALE_SOFT_EPS,
) -> None:
    """In-place Adagrad; optional soft Dale on rows (W_xh) or columns (W_hh, W_ho)."""
    mem += grad * grad
    delta = -learning_rate * grad / np.sqrt(mem + 1e-8)
    if dale_sign is not None:
        scale = soft_dale_step_scale(
            param, delta, dale_sign, axis=dale_axis, soft_eps=soft_eps,
        )
        param += delta * scale
    else:
        param += delta


def dale_violation_fraction(
    weights_input_to_hidden: np.ndarray,
    weights_hidden_to_hidden: np.ndarray,
    weights_hidden_to_output: np.ndarray,
    dale_sign: np.ndarray,
) -> float:
    """Fraction of Dale-scoped synapses with weight × neuron_sign <= 0."""
    row_sign = dale_sign.reshape(-1, 1)
    col_sign = dale_sign.reshape(1, -1)
    margins = np.concatenate([
        (weights_input_to_hidden * row_sign).ravel(),
        (weights_hidden_to_hidden * col_sign).ravel(),
        (weights_hidden_to_output * col_sign).ravel(),
    ])
    if margins.size == 0:
        return 0.0
    return float(np.mean(margins <= 0))


def outgoing_dale_violation_fraction(
    weights_hidden_to_hidden: np.ndarray,
    weights_hidden_to_output: np.ndarray,
    dale_sign: np.ndarray,
) -> float:
    """Backward-compatible alias (outgoing only)."""
    col_sign = dale_sign.reshape(1, -1)
    margins = np.concatenate(
        [(weights_hidden_to_hidden * col_sign).ravel(), (weights_hidden_to_output * col_sign).ravel()]
    )
    if margins.size == 0:
        return 0.0
    return float(np.mean(margins <= 0))


def flatten_outgoing_weights(
    weights_hidden_to_hidden: np.ndarray,
    weights_hidden_to_output: np.ndarray,
) -> np.ndarray:
    """One value per outgoing synapse: for each neuron j, Whh[:,j] then Who[:,j]."""
    hidden_size = weights_hidden_to_hidden.shape[0]
    pieces = []
    for j in range(hidden_size):
        pieces.append(weights_hidden_to_hidden[:, j])
        pieces.append(weights_hidden_to_output[:, j])
    return np.concatenate(pieces)


def flatten_weight_snapshot(
    weights_input_to_hidden: np.ndarray,
    weights_hidden_to_hidden: np.ndarray,
    weights_hidden_to_output: np.ndarray,
) -> np.ndarray:
    """Flat vector for dynamics plots: W_xh rows then outgoing columns per hidden unit."""
    hidden_size = weights_hidden_to_hidden.shape[0]
    pieces = [weights_input_to_hidden[i] for i in range(hidden_size)]
    for j in range(hidden_size):
        pieces.append(weights_hidden_to_hidden[:, j])
        pieces.append(weights_hidden_to_output[:, j])
    return np.concatenate(pieces)


def snapshot_vector_layout(
    hidden_size: int,
    vocab_size: int,
    flat_len: int,
) -> str:
    """'full' (W_xh + outgoing) or 'outgoing' (legacy checkpoints)."""
    full = hidden_size * (vocab_size + hidden_size + vocab_size)
    outgoing = hidden_size * (hidden_size + vocab_size)
    if flat_len == full:
        return "full"
    if flat_len == outgoing:
        return "outgoing"
    return "unknown"


def unpack_weight_snapshot(
    vec: np.ndarray,
    hidden_size: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct W_xh, W_hh, W_ho from a flattened training snapshot."""
    vec = np.asarray(vec, dtype=float).ravel()
    layout = snapshot_vector_layout(hidden_size, vocab_size, vec.size)
    W_in = np.zeros((hidden_size, vocab_size))
    W_hh = np.zeros((hidden_size, hidden_size))
    W_ho = np.zeros((vocab_size, hidden_size))
    idx = 0
    if layout == "full":
        for i in range(hidden_size):
            W_in[i] = vec[idx : idx + vocab_size]
            idx += vocab_size
    for j in range(hidden_size):
        W_hh[:, j] = vec[idx : idx + hidden_size]
        idx += hidden_size
        W_ho[:, j] = vec[idx : idx + vocab_size]
        idx += vocab_size
    return W_in, W_hh, W_ho


def dale_ei_blocks(
    dale_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Excitatory and inhibitory unit index arrays (E-first ordering)."""
    sign = np.asarray(dale_sign, dtype=float).ravel()
    exc = np.flatnonzero(sign > 0)
    inh = np.flatnonzero(sign <= 0)
    return exc, inh


def activation_label(*, use_relu: bool) -> str:
    return "relu" if use_relu else "tanh"


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax with log-sum-exp stabilization."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def no_input_hidden_step(
    hidden: np.ndarray,
    weights_hidden_to_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    *,
    use_relu: bool,
) -> np.ndarray:
    """Recurrent step with zero input; 1D in -> 1D out, (N,D) in -> (N,D) out."""
    W_hh = np.asarray(weights_hidden_to_hidden)
    b = np.asarray(bias_hidden, dtype=float).ravel()
    h_in = np.asarray(hidden, dtype=float)
    if h_in.ndim == 2 and h_in.shape[0] > 1:
        pre = h_in @ W_hh.T + b
        return hidden_activation(pre, use_relu=use_relu)
    h = h_in.reshape(-1, 1)
    pre = W_hh @ h + b.reshape(-1, 1)
    return hidden_activation(pre, use_relu=use_relu).ravel()


def rnn_hidden_step(
    hidden_state: np.ndarray,
    input_one_hot: np.ndarray,
    weights_input_to_hidden: np.ndarray,
    weights_hidden_to_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    *,
    use_relu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """One recurrent update; returns (hidden_state, pre_activation)."""
    pre = recurrent_pre_activation(
        input_one_hot,
        hidden_state,
        weights_input_to_hidden,
        weights_hidden_to_hidden,
        bias_hidden,
    )
    return hidden_activation(pre, use_relu=use_relu), pre
