"""Minimal transformer that adds two 8-digit integers.

A causal, 294-parameter transformer trained from scratch on randomly
generated operand pairs.  Digits are fed least-significant-first, one
place per position, and the carry is resolved by attention
attending to the nearest lower place that is not carry-transparent
(the nearest place whose digits do not sum to exactly 9).
Held-out exact-match accuracy: 99.960%.

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
            "d_head": 2,
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
    "Uim0vwTFlL+LJIC/+mNSvygPI7+v3Om+WmmGvjYYb73MTg0+N3gsP1t8QL/Sdb8/HpBLv1uzP7+WxZO+G0WnvkkN"
    "mL6j6JJAqrAuP/C44ECewT5BHophQGuZ6sElxdFA8U8uP0kkTj/TtdS/FplXwJtbD8D9UnNAwzCmv6WbAr+zm7o9"
    "Es41wBSxk0Bwy1jAbw+gvw5VpD80ejDArrIgQPODkMCjPuG/wITWP2u31r+Ijp0/+BuawNdqAMD7G8g/rBvsvU1Y"
    "TT7JBJLA98sAwAvVmT9o/MQ/8UFDvxMRe8CxzOS/mYtPPyOEQkBQttm/40cwwK0spb9/PRQ/vO+HQPueJsCmhKA+"
    "GBBIPw4XrT8nEoJAElpAwNE1EkC/YUHAs4VzwDZlDsDW4w/ADfXQv3tauT9AeWg/73NRQHlzvL83ncy/Y3X7vz+P"
    "ysB94URAFznEQL5EesCvHFLAjiVavh2HF758gqc+ACpEPjnneT5wt0g+KD4EPcws0rw3HF++AzMLvk9ODTxOAEe+"
    "FvyOv2IAkz/75l6/7ceDPsrNcT5bADQ/OUpCvxqxIz8i768+pVyEvSJ3o78Aqpo/36tPv0zPmL5huSc/veFmP1Kd"
    "cz/44+m+Fn01vnBRWD9eK24/s85FPgQA/j0qLxo9+NEYv5vXEz9GT4k+UaLsvp52Hz/QiYs/1a/1va1SYL8c80e/"
    "CWu7PoVlsL9LxpM+eiv1P4qThD9NZ5m+Q7Cnvj3SFD/Nr8i+YH07wBqbVD1dTdq9/+RrPBaKrz2jwW68oWGYvl6f"
    "pD/RMAQ+B9eOvxQB4D6CVQnBf2pIwSGKQT9rYBfAlMeUvT5A8D+p45a/9bQeP8izMMDg7e294PoUQAWhJb/s3GS/"
    "WkEmPyuqHT/P4Ws+XyU2QJZS1b5JlCg/kWX5Pp2sDr4T0QpA7tNaPwU4ET+IH86/+EyHvwBxWj/j+hM/tghKP0xN"
    "+T5cFZc+SGM3PnLObMC1h52/ULLavphbvz/o28g+vM2QvmVUtr8GpQu/uLOKP4P8lj/+gHM/O3M/P+wP5D7fmx1A"
    "yAc9wH3QU7+Ysws/OpjYP1it4T+KY2/AyIkoPxg6EcDqeiDAa6zOv3yklb5EKiXAdafdP7hCCUAzqxpAVDKMPxQC"
    "r74X7qc/39Gpu48gjj9Rf4s/FuFOQCJaED8AWdm/iROUPbiRn78GHF4/ttE/QKu41j+ElYw/tIZqvzzn2b9yzt+/"
    "f+5Nv66YDL6w/YQ/0U7jP5yLyz+IGzXAz6FzQCETob9MboBAm3byP1i32L+Pr/jAER0yQIMGGsBCr8E/MC0awB++"
    "MD8WcSe/3MBWQK4aVr3R2QZAQwYFwEyrzL/Fuvo+PF6Av7bF4r44Zra/kASjPqHyrT8ccUO//UYFP6pI4b5m548+"
    "J9QWQGWrE792BUtAaMLRv5D8b75QuQHAX8VwwDOCe7/5iJRAkp2ZPvTJ0j+ro5i/IqdoP06+NsA2AFvAhTpFPsoX"
    "87+nnRRAZxUNQP1QLEASlYm/W9IIv+4boL7b0fs/kF4oQIpudb50etQ/SngwwEXurr+oM2G/"
)

_META = {
    "task": "8-digit decimal addition",
    "architecture": "causal transformer, d_model=5, attn(h=1x1) + mlp(4) -> attn(h=1x2) + mlp(10)",
    "n_parameters": 294,
    "positional_encoding": "rank1",
    "tied_embedding_head": True,
    "sequence_length": 10,
    "decoding": "single forward pass, argmax per output digit",
    "holdout_exact_match": 0.9996
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
