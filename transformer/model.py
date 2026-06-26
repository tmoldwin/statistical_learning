"""Transformer LM and components."""
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Single self-attention head with causal mask and scaling."""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        out, weight, _, _, _ = self.forward_with_qkv(x)
        return out, weight

    def forward_with_qkv(self, x):
        """Return attention output, weights, and per-position Q/K/V vectors."""
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight / (self.head_size ** 0.5)
        weight = weight.masked_fill(self.tril[:T, :T].to(x.device) == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        out = weight @ v
        return out, weight, q, k, v


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, num_heads: int, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])

    def forward(self, x):
        outs, weights = zip(*[h(x) for h in self.heads])
        return torch.cat(outs, dim=-1), weights

    def forward_with_qkv(self, x):
        outs, weights, qs, ks, vs = zip(*(h.forward_with_qkv(x) for h in self.heads))
        return torch.cat(outs, dim=-1), weights, list(qs), list(ks), list(vs)


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, ffwd_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffwd_mult * n_embd),
            nn.ReLU(),
            nn.Linear(ffwd_mult * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN transformer block: x -> attn -> ffwd, with optional residuals/LayerNorm."""

    def __init__(self, num_heads, n_embd, head_size, block_size, use_residual=True, use_layernorm=True):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, n_embd, head_size, block_size)
        self.ffwd = FeedForward(n_embd, ffwd_mult=16)
        self.use_residual = use_residual
        self.ln1 = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()

    def forward(self, x):
        post_ffwd, wei, _, _, _, _, _ = self.forward_with_activations(x)
        return post_ffwd, wei

    def forward_with_activations(self, x):
        """Run one block and expose Q/K/V, attention, and intermediate states."""
        ln1_x = self.ln1(x)
        attn_out, wei, qs, ks, vs = self.sa.forward_with_qkv(ln1_x)
        post_attn = x + attn_out if self.use_residual else attn_out
        ffwd_out = self.ffwd(self.ln2(post_attn))
        post_ffwd = post_attn + ffwd_out if self.use_residual else ffwd_out
        return post_ffwd, wei, qs, ks, vs, ln1_x, post_attn


class BigramLanguageModel(nn.Module):
    """
    - token_embedding: maps token id -> embedding vector (n_embd)
    - lm_head: maps attention output -> logits over vocab
    - use_residual: if False, disable residual connections (no x+attn, no x+ffwd)
    """
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        num_heads: int,
        head_size: int,
        use_residual: bool = True,
        n_layer: int = 1,
        use_layernorm: bool = False,
        pos_embd_dim: int | None = None,
        timestep_noise_std: float = 0.0,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.timestep_noise_std = float(timestep_noise_std)
        self.pos_embd_dim = n_embd if pos_embd_dim is None else pos_embd_dim
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # (vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, self.pos_embd_dim)
        self.pos_proj = (
            nn.Identity()
            if self.pos_embd_dim == n_embd
            else nn.Linear(self.pos_embd_dim, n_embd, bias=False)
        )
        self.block_size = block_size
        self.use_residual = use_residual
        self.n_layer = n_layer
        self.use_layernorm = use_layernorm
        # Legacy single-layer path (default): keep self.sa_heads / self.ffwd so the
        # integer-task plotting scripts that reach into these attributes keep working.
        self._legacy = (n_layer == 1 and not use_layernorm)
        if self._legacy:
            self.sa_heads = MultiHeadAttention(num_heads, n_embd, head_size, block_size)
            self.ffwd = FeedForward(n_embd, ffwd_mult=16)
        else:
            self.blocks = nn.ModuleList(
                [Block(num_heads, n_embd, head_size, block_size, use_residual, use_layernorm) for _ in range(n_layer)]
            )
            self.ln_f = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _inject_timestep_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add isotropic Gaussian noise at every (batch, time) position."""
        if self.timestep_noise_std <= 0:
            return x
        return x + torch.randn_like(x) * self.timestep_noise_std

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_wei: bool = False):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)  # (B,T,C)
        positions = torch.arange(T, device=idx.device) % self.block_size
        pos_emb = self.pos_proj(self.position_embedding_table(positions))  # (T, n_embd)

        x = self._inject_timestep_noise(token_emb + pos_emb)  # (B,T,n_embd)

        if self._legacy:
            attn_out, wei = self.sa_heads(x)  # (B,T,n_embd)
            if self.use_residual:
                x = x + attn_out
                x = x + self.ffwd(x)
            else:
                # No residuals: just pass through attention then FFN
                x = attn_out
                x = self.ffwd(x)
        else:
            wei = None
            for block in self.blocks:
                x, wei = block(x)
            x = self.ln_f(x)

        logits = self.lm_head(x)  # (B,T,vocab)

        loss = None
        if targets is not None:
            Bt, Tt, Cc = logits.shape
            loss = F.cross_entropy(logits.view(Bt * Tt, Cc), targets.view(Bt * Tt))

        if return_wei:
            return logits, loss, wei
        return logits, loss

    @torch.no_grad()
    def forward_with_activations(self, idx: torch.Tensor):
        """Expose every representation used in the causal forward pass.

        Returns a dict with:
          token_emb, pos_emb, block_input: (B, T, n_embd)
          layers: list of per-layer dicts with keys queries, keys, values (per-head
                  lists of (B,T,head_size)), attention (per-head (B,T,T)), attn_input
                  (ln1 input), post_attn, post_ffwd
          block_output: (B, T, n_embd) after final LayerNorm, fed to lm_head
          logits: (B, T, vocab)
        """
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        positions = torch.arange(T, device=idx.device) % self.block_size
        pos_emb = self.pos_proj(self.position_embedding_table(positions)).unsqueeze(0).expand(B, -1, -1)
        block_input = self._inject_timestep_noise(token_emb + pos_emb)
        x = block_input

        layers = []
        if self._legacy:
            ln1_x = x
            attn_out, attn, qs, ks, vs = self.sa_heads.forward_with_qkv(ln1_x)
            post_attn = x + attn_out if self.use_residual else attn_out
            ffwd_out = self.ffwd(post_attn)
            post_ffwd = post_attn + ffwd_out if self.use_residual else ffwd_out
            layers.append({
                "attn_input": ln1_x,
                "queries": qs,
                "keys": ks,
                "values": vs,
                "attention": list(attn),
                "post_attn": post_attn,
                "post_ffwd": post_ffwd,
            })
            x = post_ffwd
        else:
            for block in self.blocks:
                post_ffwd, attn, qs, ks, vs, ln1_x, post_attn = block.forward_with_activations(x)
                layers.append({
                    "attn_input": ln1_x,
                    "queries": qs,
                    "keys": ks,
                    "values": vs,
                    "attention": list(attn),
                    "post_attn": post_attn,
                    "post_ffwd": post_ffwd,
                })
                x = post_ffwd

        if not self._legacy:
            x = self.ln_f(x)

        logits = self.lm_head(x)
        return {
            "token_emb": token_emb,
            "pos_emb": pos_emb,
            "block_input": block_input,
            "layers": layers,
            "block_output": x,
            "logits": logits,
        }

    @torch.no_grad()
    def features(self, idx: torch.Tensor):
        """Run a forward pass and expose internal activations for analysis.

        Returns (logits, block_output, attn) where:
          - block_output: (B, T, n_embd) final representation fed to lm_head
          - attn: tuple of per-head attention matrices (B, T, T) from the last
                  attention layer, or None if unavailable.
        """
        acts = self.forward_with_activations(idx)
        attn = None
        if acts["layers"]:
            attn = tuple(acts["layers"][-1]["attention"])
        return acts["logits"], acts["block_output"], attn

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Autoregressive generation.
        idx: (B, T)
        returns idx extended to (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)            # (B, T, vocab)
            last_logits = logits[:, -1, :]   # (B, vocab)
            probs = F.softmax(last_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)             # (B, T+1)
        return idx


# Alias for backward compatibility
TransformerLM = BigramLanguageModel
