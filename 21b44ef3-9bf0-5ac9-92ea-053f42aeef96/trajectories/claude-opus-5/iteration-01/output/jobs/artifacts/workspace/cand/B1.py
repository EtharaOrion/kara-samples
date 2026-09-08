"""Minimal transformer that adds two 8-digit integers.

A causal, 273-parameter transformer trained from scratch on randomly
generated operand pairs.  Digits are fed least-significant-first, one
place per position, and the carry is resolved by attention
attending to the nearest lower place that is not carry-transparent
(the nearest place whose digits do not sum to exactly 9).
Held-out exact-match accuracy: 99.928%.

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

# Training-time only (guarded by ``self.training``; build_model() returns an
# eval-mode model, so this is inert at inference).  Gaussian jitter on the
# attention scores forces the carry head to decide with a *margin*: a genuine
# content look-up survives it by growing the gap between the right key and the
# rest, whereas the degenerate solution -- a fixed geometric ramp over
# positions, which approximates the carry as a weighted sum of place sums --
# depends on precise small logit differences and is destroyed by it.
SCORE_NOISE = 0.0


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
        if self.training and SCORE_NOISE > 0.0:
            scores = scores + SCORE_NOISE * torch.randn_like(scores)
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
      "ramp"  -- the position index itself times one learned direction
                 (d params).  Same idea with the monotone ordering pinned to
                 the obvious one instead of being learned.
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
        elif pos_mode in ("rank1", "ramp"):
            if pos_mode == "rank1":
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
            if self.pos_mode == "rank1":
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
        if self.pos_mode == "ramp":
            ramp = torch.arange(SEQ, dtype=self.pos_dir.dtype, device=self.pos_dir.device)
            return (ramp / (SEQ - 1) - 0.5).unsqueeze(1) * self.pos_dir.unsqueeze(0)
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
    "d_model": 5,
    "layers": [
        {
            "n_heads": 1,
            "d_head": 1,
            "d_ff": 4
        },
        {
            "n_heads": 1,
            "d_head": 1,
            "d_ff": 10
        }
    ],
    "pos_mode": "rank1",
    "tie_head": True,
    "head_bias": True,
    "q_bias": True,
    "learn_scale": True
}

_WEIGHTS = (
    "Ob/xPyDiXD90eTU/rPwPP8wN0T5m/F4+t07XvERYt77Lk02/oiaav0CdFL+4Z7++Y2kCv/awEMBEZQi/KSGBQAg+"
    "ZMBXm6NAeD0xQS/5UEBtnFHBS5cVwNACMT+gJTDBEyDhv3x1Fz/lRCjAMKjTv8wEOUA1+4W/yHvQPtC0KkAh8ju8"
    "76N0PxCJUsCpThrAK6TfPxjAsD/m+kA/fDUDwBolisCsyYc/zirVP0oIaj6CyTe/P0OmwM2s+T5R8KI/W4/Vvidd"
    "Lz/ty6DA/teUvK+f6T5BgXK/S0/3P6twb8B3Rda+BCZDv4Qznb/SljZAfsnpv7hWZb/kDeC/6QuAv4b6TkBnUKw+"
    "o1LVv4JM2b+ll4S9A9xBQHeQ7z/HNkDA8RlJPbFZ0z/bxARAOCUeQPi+kT69PSq+T+ZaPvKLt76PAWm+jlGWwM+T"
    "aT7O362/RkddPjnPBsAlr5+/ke2xvpwnyb3sovm+5t5aPr5yFr8BQTo+23JCPvK0Vr5i2Pa9my0uPNAyRL8GVzA+"
    "sxPUvqQFKD6Bu9A+Y37Tvmrp7r+Qt6u/UR7+PqzgUz/1oO0+Aa1Lvx6usj4t5MA+wPn6v2HIob6efgbAWTmsv/zt"
    "Iz+5AIU+VUvXP46khj4+ka0/09sjv2R7t78OuhY/8shIv8y8Wj/7CiY+hHKVv6prsb7D3vM/jRhLv8CW+j/8hMA9"
    "k5cPP4n/kj+7wKU9YzlIP2NalL/2d4c+oL6FPxvz6L839zq/plmkv2Pihb5MKam+wZvrvASPM7+D6Ci+uLm+QByJ"
    "ZD+YynE/kLMAPpj0GUDV4/o+vPQdP8hbV8DBSIk+HrFGPwRegj46dya/xiDevgNAQL+ya0vAgEEpv6S6yD51mm8/"
    "6kz7P3KY9z67oZo/szMUwH6XsL4rO+g+CPK1P8iyMz88g6C/xKpHwDPNb8B5DOw//2DGPgyvmD8u80E+JgHZPd1b"
    "sz5JxBbAdF80v31cwMA9y1m/39+PuihF0T6dADO/yeaWwB/8iT5LYwk/skbYP3iHDMCUGAC/3LURvGoU1jy1sek+"
    "YHnsv+lapEBXI2w+Nua9PXLN178tk/c/XpUNQC7LAD8hauq+sS8DwI7K4L7GwzC/Nqytv2kFAL89F0S9TGOOv8Jr"
    "zUA8DFxAjT+Avp6EpEDO5hI/4lMQwHgmDb8bzY2/hUqbQHb0Y7/yPlM/QtAgQGL/JT+tHLu/TF+Zv7kgNj90xYDA"
    "lX4gQHSaCUAPOBu/xB8AQNAJMcCURyy9B0aiP/S5HD6DEeS/AUM3PwoAHT5+1kG+BbYOPT6eJrxHSqO/nKJVP3Pr"
    "6T85ELk/MzX7vuclDT+Qvr6/di3Qv7OQh78KvNY/2wNTvsTG5j+peBi/cESAPwjU1L+7qMu/yvR4P9FKd0BiZhFA"
    "3VchwDVt5T+5xL2/OJRrvlKdgr8xw6c/SlUVQDNU/79zhg7A"
)

_META = {
    "task": "8-digit decimal addition",
    "architecture": "causal transformer, d_model=5, attn(h=1x1) + mlp(4) -> attn(h=1x1) + mlp(10)",
    "n_parameters": 273,
    "positional_encoding": "rank1",
    "tied_embedding_head": True,
    "sequence_length": 10,
    "decoding": "single forward pass, argmax per output digit",
    "holdout_exact_match": 0.99928
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
