"""Minimal transformer that adds two 8-digit integers.

A causal, 284-parameter transformer trained from scratch on randomly
generated operand pairs.  Digits are fed least-significant-first, one
place per position, and the carry is resolved by attention
attending to the nearest lower place that is not carry-transparent
(the nearest place whose digits do not sum to exactly 9).
Held-out exact-match accuracy: 99.920%.

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
    "pos_mode": "ramp",
    "tie_head": True,
    "head_bias": True,
    "q_bias": True,
    "learn_scale": True
}

_WEIGHTS = (
    "xMmQv9u3l0DaW0zAZfTNvvmglr9s4TZAYjYHwAG3EMFK67nAleQiQUv7ukGVIA1BENQGwqJ0lUCJGxU/KXw9Pq84"
    "ob8pwkrAtlEUwDHMXUAp0gzAkLiRvlmEGb53gwnAAw9iQM2CbMBSxpS++BPVP6Go77/AZBhAQZGGwBSiNL95lhNA"
    "c8eGv/dziD+nYHvABWWvv4Jg/z8y5CQ+EIhPvaxoRcDnYe+/AB2RPw/4vD/8gYq/GLPpv0NP/b9OXII+UF4oQMjT"
    "A8CJ6lK+Yc/Mv1LUtL7F8V1AoBw9wEYSGUB//zA/iatvPji2UkC0flXAAvotQFhfIMBIvzzAcSM5wD1stb8iRTm/"
    "rO6lPiLetr+btUi+BS6yv9UMm8C9v7O/ff9twF05hUClVXbA1ZOzvxqjZ8DKJzG9xQsrOumB6byfpbI9RIZ4PsM4"
    "2b3hruA+zIFEvhlzh7th+su77RDNvXEvMD0mFKs+PdZKPwXgYb+E9kI+HvoGP245BT/RtJy+B0YpP9JjbT9BZYO+"
    "ov8lv4M1Zj9QrNG+50IsvtZF1j2Jlj8+57SFP4K3e76Ke4W/BGvKPxlC7j/9ma49usnpPh8bXT1thai+FtswPxx1"
    "YT4M3KS+UiIcvycG3z9mRvy999frPmjvnL9w+aQ/HtnMvsZ0Zr/0ZfE/2vnZP+4oAb9VRXy/g0JbP3VWHr7kyXTA"
    "zi+ZPibgJ794w/0+Yao1Pm9lnD4Te1o9mQrlvj7qiD5DKyk+cU0tPkYXkUEUUkfA+6jnv4ftF0CMsBjAougPv8yd"
    "0b/QcYk/vZS0v1R5rD9vmLc+mQ5cPzDidb+1tF0/aHkLPooDVrxDNOg/yZtVv9CKQD+7nw8/2u9APZqjI0BE4NQ+"
    "oxMAP1O7378adfG/Hm+XP4pSyD+tawk+2nupve4g6T6Ds/E+lFz/vwTj87+n+p8+W0+Wv+lw8D/APIS/QgTBv/Q+"
    "/L4Hpq4/fB/yPrC5rT89vZm/GsaFvT4g6j+fjbnAF1DYvohMg79BcaY/5AAqPxs00r8aodI+fwkRwH7BHMC3nB7A"
    "ssWcP8d/BMBUt4Y/jGAkQPd7rT89Xjq/JHo1QKwADL6bN3o/Q6RmwFdOQj9OJGdAdkv1vq2Un79bkIk+5m3pv7px"
    "kT+uWuo+DMm+PgJTMcCSSoe+Bw6ZwNdUD8AaB/i+3koKv746Qj93/6s/4J2AQJUJt78eUHRArDB/v/C9WUCxnF7A"
    "QazowG7/EUCb0YdAXAeFvwAw5j+sZ3DAspEJv91Onb5xxBtApgrkPyaLlT83/SrAN5jVvy9SOT9PcKw8g5FIPl1d"
    "xL/IKYk/sYhbPziu8L8vH+q+gvC0v8EpGb+Xccg/IUIgwBBVJEDpBas/30InvzWQNcAnhzjAGVoBQDtgyz+PrqE/"
    "GiwwP9g7wL6wOnC+XIcJwJKsgMDPe5m/NZ/bv4o3U0CDKYu/5919QNC0VMBrA0E//3WOPUmUUD/rbCNAh+o4PrGi"
    "b0BQP8TAcLEWQFJF678="
)

_META = {
    "task": "8-digit decimal addition",
    "architecture": "causal transformer, d_model=5, attn(h=1x1) + mlp(4) -> attn(h=1x2) + mlp(10)",
    "n_parameters": 284,
    "positional_encoding": "ramp",
    "tied_embedding_head": True,
    "sequence_length": 10,
    "decoding": "single forward pass, argmax per output digit",
    "holdout_exact_match": 0.9992
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
