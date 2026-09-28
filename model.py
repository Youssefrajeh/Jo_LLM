"""
Jo — a small GPT (decoder-only Transformer) written from scratch in PyTorch.

Same architecture family as GPT-2 / GPT-3 / Llama, just much smaller.
Read top to bottom: Attention -> MLP -> Block -> GPT.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65      # number of distinct tokens (characters here)
    block_size: int = 256     # context length: how many tokens the model can "see"
    n_layer: int = 6          # number of Transformer blocks stacked
    n_head: int = 6           # attention heads per block
    n_embd: int = 384         # size of each token's vector ("embedding dimension")
    dropout: float = 0.2      # randomly zero activations during training to reduce overfitting


class CausalSelfAttention(nn.Module):
    """
    Lets every token look at the tokens BEFORE it and gather information from them.

    For each token we compute three vectors:
      query (q): "what am I looking for?"
      key   (k): "what do I contain?"
      value (v): "what do I pass on if you pick me?"
    score(i, j) = q_i . k_j  -> how much token i should attend to token j.
    "Causal" = token i may only look at j <= i (it can't peek at the future,
    otherwise predicting the next token would be cheating).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        # one linear layer produces q, k, v for all heads at once
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # mixes the heads' outputs back together
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (n_embd)
        q, k, v = self.qkv(x).split(C, dim=2)
        # split channels into heads: (B, T, C) -> (B, n_head, T, head_dim)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # softmax(q k^T / sqrt(hs) + causal_mask) @ v
        # PyTorch's fused kernel does exactly this (fast "flash attention").
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # the manual equivalent, for study:
        #   att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        #   att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf"))
        #   y = F.softmax(att, dim=-1) @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-join heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """
    Per-token "thinking" step. Attention moves information BETWEEN tokens;
    the MLP processes each token's vector on its own. Expand 4x, non-linearity, shrink back.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    One Transformer block = communicate (attention) + compute (MLP).
    LayerNorm keeps values well-scaled; the "x + ..." residual connections
    let gradients flow straight through deep stacks, which makes training stable.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # position -> vector (attention has no sense of order otherwise)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector -> score for every token
        self.lm_head.weight = self.tok_emb.weight  # weight tying: input and output embeddings share weights (as in GPT-2)

        self.apply(self._init_weights)
        # GPT-2 trick: scale down the residual output projections so deep stacks start stable
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) the same sequence shifted left by one = the "next token" at each position
        returns logits (B, T, vocab_size) and, if targets given, the cross-entropy loss.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than context window"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation: predict one token, append it, repeat.
        temperature < 1 -> safer/more repetitive, > 1 -> more random.
        top_k -> only sample among the k most likely tokens.
        """
        for next_id in self.generate_stream(idx, max_new_tokens, temperature, top_k):
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Same as generate(), but yields each new token (shape (B, 1)) as soon as it's sampled — used by the web app."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # only the last position predicts the next token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            yield next_id
