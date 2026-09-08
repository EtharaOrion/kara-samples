"""Minimal transformer that adds two 8-digit integers.

A causal, 284-parameter transformer trained from scratch on randomly
generated operand pairs.  Digits are fed least-significant-first, one
place per position, and the carry is resolved by attention
attending to the nearest lower place that is not carry-transparent
(the nearest place whose digits do not sum to exactly 9).
Held-out exact-match accuracy: 99.956%.

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
            "d_ff": 3
        },
        {
            "n_heads": 1,
            "d_head": 1,
            "d_ff": 12
        }
    ],
    "pos_mode": "rank1",
    "tie_head": True,
    "head_bias": True,
    "q_bias": True,
    "learn_scale": True
}

_WEIGHTS = (
    "ddHkv2Ej87+19NK/+impv6INf7/AtSq/3yGkvvBTGT1Rpc8+YFA0PyD4979YRLa/v9Sgv8qKID+TICW8T2GUPqc8"
    "Uz+SBpRAFuyhQZ/WgcEPgLU/nUzVQSmFGUE+MlXBsqCnwDh6Xr9t+vg/y8UbwHeO0b81Dpe+892Pv3yAFD8buPK/"
    "nsNfwEeqlz/2e9u/4l/Iv4D7gUBlKLg/LhwwwG8Hk7/z9jPAYRZXQCYZXT87VXfAigsAv2GqZ8BBlwtAK+oEP4pw"
    "mcB2m0c/ji09wP/M1z6Wgj4/QnGgwA3FAEAf9QHA3fOLv6zeHT9myYjAUiU9QBcwn79yyRDAyiJdPje5SsCQEXRA"
    "WzFAvwvOO8DT2QW/1EPXv5zRi0D/Exq/QjtNwHLTur+N1IE8crlPP3GGHb7fRgo+ud2OvuKU9L0uDYi+5XXAQGhh"
    "YL8i+bu/GZqTv1Rb7D80T64/xjQEvxtUWD+xtok+Vt8Xv9AdEr+ba1K973azPjsW0r2emrA+Vd+Dvo5slT5eWJk9"
    "ZZuAv31dq7/WHmO+8uEKv3ITMT8xpUK+t+Cyv1ye8z80xGg8/VuoPjUxcL/1ppy/3Inxvms8A0C4sTHAFRUgQHXK"
    "V78LUyA+SE0lP50R9z81BMe+dlOWv40NuL7DIbg/SWuTPsvKjz9HHgy/jTAGwPJjrb6ehQJAEapWvY7YOsA1Hrq9"
    "i1GqvUjfK746jq08s2vCPY20jMDP6rs+CVSWPxFcC0BwaAO9muBUvyAepL+LK4k/U+EnP4i3MT+m8G2/NMqfvzxv"
    "0L6KPJu//G07PwNE3b97pSjAZG7eP4k66D8oELw/LF5OvaNa4b9DLUxAwtb4v/CgAMBSvARAy6nSP4xtyL8CIRE/"
    "8xOhPsNLHD/6fYa+uoS/v2j0SECwiDVAKQBov4vUHsDfsJY+6Y91QNNnHD+Qu0S/BCioP8Upkb2AJh7ALW4VwOa6"
    "5T4RiTtAmTkTwN+w1L/5waK+KCMtvyQKJUAicT+/wpYnwOjowj/Fx6M+/UVrv2qwyj8dVMY96dBHv1cYDECAQ4BA"
    "la92wPYcL79WHi1AuX6dvxN76L4BMKi//SpyQD3jmkBgpve+VetMP2GhjL+0E/E/E425vqabJUDitvjAnoWSwP6Z"
    "5T8AigtBBqS2vaUpub6oDpFADRBowDUmZMCEG6Y/6XBdv7/XG8CuBDrAT6o9v/JeD8Dj7iu+qZ1+P5A7HEDz70e/"
    "FJVQQFVhFcAHpTU/j6AhQFsTuz8hJ+rAwMnDP72WQ0BNyTpAs2hAwC7IcD1zZpu/jGRSv3fYKL/5ZTBAmJBiv8RO"
    "WMBM28BAQgDevzeEsr6QXti8N6YpQKXyWb9AroVAU150QFyCUkDCEtK/JQgIwDSIoT+RPwPBDgEYPgMRGD8NmBtA"
    "0EoYvyB4jD/fKiG/NqJYP1hIsz1x8go/ngWrQACS77+ZMAfA1MpGQAt9ckA7IsJA2pEgwPmLhsAGN32/eem9vYlG"
    "CEAwiktAt5C3wCOdlL8="
)

_META = {
    "task": "8-digit decimal addition",
    "architecture": "causal transformer, d_model=5, attn(h=1x1) + mlp(3) -> attn(h=1x1) + mlp(12)",
    "n_parameters": 284,
    "positional_encoding": "rank1",
    "tied_embedding_head": True,
    "sequence_length": 10,
    "decoding": "single forward pass, argmax per output digit",
    "holdout_exact_match": 0.99956
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
