"""Minimal transformer that adds two 8-digit integers.

A causal, 447-parameter transformer trained from scratch on randomly
generated operand pairs.  Digits are fed least-significant-first, one
place per position, and the single attention layer resolves carries by
attending to the nearest lower place that is not carry-transparent.
Held-out exact-match accuracy: 100.000%.

Interface:
    model, meta = build_model()
    add(model, 12345678, 87654321) -> 99999999

Every returned sum is read straight off one forward pass of `model`; the
parameters below are the ones produced by training (see train.py, which is not
imported here).
"""

import base64

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ = 10
NDIG = 8

# Set True to retain the attention probabilities of the last forward pass for
# inspection (used by the analysis script, never on the inference path).
STORE_ATTN = False


class SelfAttention(nn.Module):
    """Standard causal self-attention.

    Two small extras earn their keep on a model this size.  ``q_bias`` gives the
    query an input-independent component, so a head can express "prefer places
    with property P" without having to route that constant through the residual
    stream.  ``learn_scale`` reparameterises the logit temperature
    multiplicatively, which lets a head sharpen towards a hard argmax in far
    fewer steps than growing wq/wk elementwise would take.
    """

    def __init__(self, d_model, n_heads, d_head, q_bias=False, learn_scale=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.wq = nn.Linear(d_model, inner, bias=q_bias)
        self.wk = nn.Linear(d_model, inner, bias=False)
        self.wv = nn.Linear(d_model, inner, bias=False)
        self.wo = nn.Linear(inner, d_model, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(n_heads)) if learn_scale else None
        self.attn = None

    def forward(self, x, mask):
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head
        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        if self.log_scale is not None:
            scores = scores * torch.exp(self.log_scale).view(1, H, 1, 1)
        scores = scores.masked_fill(~mask, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        if STORE_ATTN:
            self.attn = att
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, T, H * D)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads=0, d_head=0, d_ff=0, q_bias=False, learn_scale=False):
        super().__init__()
        self.attn = (SelfAttention(d_model, n_heads, d_head, q_bias, learn_scale)
                     if n_heads * d_head > 0 else None)
        self.mlp = MLP(d_model, d_ff) if d_ff > 0 else None

    def forward(self, x, mask):
        if self.attn is not None:
            x = x + self.attn(x, mask)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class AdderTransformer(nn.Module):
    """Causal transformer over 10 positions; emits all 9 sum digits in one pass.

    ``pos_mode`` selects the positional encoding parameterisation:
      "full"  -- an independent learned vector per position (SEQ * d params)
      "rank1" -- a learned scalar per position times one learned direction
                 (SEQ + d params).  The task is translation invariant across
                 digit places, so all attention needs from position is a
                 monotone ordering; a rank-1 code supplies exactly that.
    """

    def __init__(self, d_model, layers, pos_mode="rank1", tie_head=True, head_bias=True,
                 q_bias=False, learn_scale=False):
        super().__init__()
        self.d_model = d_model
        self.layers_cfg = [dict(l) for l in layers]
        self.pos_mode = pos_mode
        self.tie_head = tie_head

        self.emb = nn.Embedding(10, d_model)
        if pos_mode == "full":
            self.pos = nn.Parameter(torch.zeros(SEQ, d_model))
        elif pos_mode == "rank1":
            self.pos_scale = nn.Parameter(torch.zeros(SEQ))
            self.pos_dir = nn.Parameter(torch.zeros(d_model))
        else:
            raise ValueError(pos_mode)

        self.blocks = nn.ModuleList([
            Block(d_model, q_bias=q_bias, learn_scale=learn_scale, **l)
            for l in self.layers_cfg])
        self.head = None if tie_head else nn.Linear(d_model, 10, bias=False)
        self.head_bias = nn.Parameter(torch.zeros(10)) if head_bias else None

        mask = torch.zeros(SEQ, SEQ, dtype=torch.bool)
        for i in range(SEQ):
            for j in range(i):
                mask[i, j] = True
        mask[0, 0] = True  # keeps the sink row well defined
        self.register_buffer("attn_mask", mask.view(1, 1, SEQ, SEQ), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.5)
        if self.pos_mode == "full":
            nn.init.normal_(self.pos, std=0.5)
        else:
            with torch.no_grad():
                self.pos_scale.copy_(torch.linspace(-1.0, 1.0, SEQ))
            nn.init.normal_(self.pos_dir, std=0.6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def positional(self):
        if self.pos_mode == "full":
            return self.pos
        return self.pos_scale.unsqueeze(1) * self.pos_dir.unsqueeze(0)

    def forward(self, ta, tb):
        x = self.emb(ta) + self.emb(tb) + self.positional()
        for blk in self.blocks:
            x = blk(x, self.attn_mask)
        w = self.emb.weight if self.tie_head else self.head.weight
        logits = F.linear(x, w)
        if self.head_bias is not None:
            logits = logits + self.head_bias
        return logits


def encode(a, b, device=None):
    """Split both operands into digit tokens, LSB first, zero-padded to SEQ."""
    ta = [0] + [(a // 10 ** i) % 10 for i in range(NDIG)] + [0]
    tb = [0] + [(b // 10 ** i) % 10 for i in range(NDIG)] + [0]
    ta = torch.tensor([ta], dtype=torch.long, device=device)
    tb = torch.tensor([tb], dtype=torch.long, device=device)
    return ta, tb


def decode(digits):
    """Assemble predicted digits (LSB first) back into an integer."""
    value = 0
    for i, d in enumerate(digits):
        value += int(d) * 10 ** i
    return value

# ---------------------------------------------------------------------------
# Trained parameters: float32, little-endian, concatenated in state_dict order.
# ---------------------------------------------------------------------------

_CONFIG = {
    "d_model": 6,
    "layers": [
        {
            "n_heads": 1,
            "d_head": 2,
            "d_ff": 4
        },
        {
            "n_heads": 1,
            "d_head": 3,
            "d_ff": 14
        }
    ],
    "pos_mode": "rank1",
    "tie_head": True,
    "head_bias": True,
    "q_bias": True,
    "learn_scale": True
}

_WEIGHTS = (
    "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"
    "yT/INhI/"
)

_META = {
    "task": "8-digit decimal addition",
    "architecture": "causal transformer, d_model=6, attn(h=1x2) + mlp(4) -> attn(h=1x3) + mlp(14)",
    "n_parameters": 447,
    "positional_encoding": "rank1",
    "tied_embedding_head": True,
    "sequence_length": 10,
    "decoding": "single forward pass, argmax per output digit",
    "holdout_exact_match": 1.0
}


def build_model():
    """Rebuild the trained model.  Returns (model, metadata)."""
    model = AdderTransformer(**_CONFIG)
    flat = torch.frombuffer(bytearray(base64.b64decode(_WEIGHTS)), dtype=torch.float32)
    state, off = {}, 0
    for name, ref in model.state_dict().items():
        n = ref.numel()
        state[name] = flat[off:off + n].view_as(ref).clone()
        off += n
    if off != flat.numel():
        raise RuntimeError("weight blob does not match architecture")
    model.load_state_dict(state)
    model.eval()
    meta = dict(_META)
    meta["n_parameters"] = sum(p.numel() for p in model.parameters())
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Return a + b for 8-digit operands, decoded from one forward pass."""
    device = next(model.parameters()).device
    ta, tb = encode(int(a), int(b), device=device)
    logits = model(ta, tb)
    digits = logits[0, 1:, :].argmax(-1).tolist()
    return decode(digits)


if __name__ == "__main__":
    import random

    m, info = build_model()
    print(info)
    ok = sum(add(m, x, y) == x + y for x, y in
             [(random.randint(10 ** 7, 10 ** 8 - 1), random.randint(10 ** 7, 10 ** 8 - 1))
              for _ in range(2000)])
    print("spot check:", ok, "/ 2000")
