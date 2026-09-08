"""Minimal transformer that adds two 8-digit integers.

A causal, 295-parameter transformer trained from scratch on randomly
generated operand pairs.  Digits are fed least-significant-first, one
place per position, and the carry is resolved by attention
attending to the nearest lower place that is not carry-transparent
(the nearest place whose digits do not sum to exactly 9).
Held-out exact-match accuracy: 99.969%.

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
            "d_ff": 4
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
    "CwwIwvXJO79dM5m+rpM+vtidrL2NIK88+LoEPiAEfT53dcA+iecEP2TicT+JSdc+dojyvqWZ/r1FHIU/AnN6P47O"
    "rj1WMoHAPNI9wHXXQ0DR+BlAzBRMwIbrA8Dq7yVB5xcewCI0JkCu1HDAdGOhv4Bjfb9rvJi/hVnuP+zN5L3nZMi/"
    "HuTnv47P0797Xrc/ByfAP1dXvr+30R2/3Cbfv9ROiT8oz94/y3Zcv3e7iT/QMbC/43E4Px5RuT+IpZo+3yj6P4t2"
    "NL/ncrU+MPtrPzZv2T/RzgZApWLuPR8JCL5DSaQ+fiQtQPeQ6T8P5W8/S9RLv/pjl75qE0VAlweCP07b2j90zum/"
    "DfpAv1R17j8Tm04+1EsPQOKvYsBzA2m/hhjhvxcujT5Cwh1AtAzCPDGCQj/tnoE+rhuavnjLeTwwtYE/N+axwDRy"
    "278vMxi/O50xP+GcEL0J0RPAHSIJQPnJP79TKsS9mhpQvoAIpb8+exDAk0twv0lzvT+5r0y+9TQowEdzOb/38ME/"
    "Gs1SP/Ufhr5Orw8/9zSqPwKzwD8LlZ8+Q/1QvxzYuL/nf3E+Du1MPzB8CL/C84u/k1A1v1LYNL/yVYu+902NPjJC"
    "H79C2HE/ROAavrsCaL/viZM+YQ2pvyv3nL+Kp7I/VNDavsvzVb1qxFs/6KGgv4WYND/SeNS+dfyHvw5Toz9CJ3a/"
    "wuWtvvdJUj4cd1S/a2BRPzXmM7/tFr0+BAKkvlCsKL5zclk99F0+QOzmor9XHeC+37ADP1J20LyAj9u/UOi6v6TS"
    "Rj8UUdS/xNkDQPa1Ir6cJJ4/0GCfvkuvkD/Z7UC/S3rivvZYAr99cdg/ouAZP4Q7Zb8/SYU9gormP3zog78LKIhA"
    "eG6svoJ1mT8jUHm/IUnSv7mT4b4iNjM/47aZP36H3z/xbnY/tQsEQA0k6z+NB44/OLoTv7Mkmz90QSY+ntiyv1q0"
    "3j/oUADAAq+2vqgO2L+frBM/PpxHP3Oknz+LghvAfCCVvecM8z58cwM/KjYYQC+GrD58xFw/rzLdvjK0Iz98B6q/"
    "F9nUPqe/gEDP3xZAmqaOPy/LBL8PJ5O/mnDCPxouIECD8Ic/yK3wP0Y56T40CJC/3Ga7P2HuVr4Smm+9F0MtvwJ3"
    "l7+VUaO/z/kiwB6+2T5u/Mw+jKlMvqvoM8AcmaS/REPlv5pz077BtQ5AMlQ4QM8iQkDsGFvAuHe6vxBdcUCmmIBA"
    "xHBHwE/z7r+5ZrA+VqFOPxOT4j3Y3BG/bVq/v/GFyT1nVEC84FFWPwGG4b2ql4w+Jad+P67mSj/rQLC/6UG2P74Z"
    "mb5skZ8/VNWgv42EoD/aE+w+NEp2P8K04L+l/eq/iCQIQNzKkT8BFgbAqsULP35lcsBX2bs/OzGZP/DgBEBfIeY/"
    "cQPqP27ojL9wVBfAoO8lQA8dez6a0oO/j1MywFXNqr2cJYk/ZOHRvFp7HEAo/RvAE/POP1qM07+UuzPAM/pgPwWo"
    "cz+jSAvAGjZzvx09Sr84Z1M+x/3UP1UaKL8Nnr4/gzQnv4dilj8QPSs/mobIv7HCEj7XwLc/DFSPvw=="
)

_META = {
    "task": "8-digit decimal addition",
    "architecture": "causal transformer, d_model=5, attn(h=1x1) + mlp(4) -> attn(h=1x1) + mlp(12)",
    "n_parameters": 295,
    "positional_encoding": "rank1",
    "tied_embedding_head": True,
    "sequence_length": 10,
    "decoding": "single forward pass, argmax per output digit",
    "holdout_exact_match": 0.99969
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

