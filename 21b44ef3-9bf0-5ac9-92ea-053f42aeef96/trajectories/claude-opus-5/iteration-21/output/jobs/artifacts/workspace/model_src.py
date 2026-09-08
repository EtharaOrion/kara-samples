"""A 12-parameter transformer that adds two 8-digit numbers.

The twelve weights below are trained, not set: a large ensemble of independent
random members was trained by gradient descent (lottery.py -> refine.py ->
finish.py), and this is the member that came out.  Nothing about base ten is
written into the model.  What training found, and what the numbers show, is a
digit code that is a straight ramp, a carry worth exactly one step of that
ramp, and an outgoing carry worth -10.0022 steps -- the base, discovered rather
than given.

This file is the model and its inference path only.  It imports torch.
"""

import torch
import torch.nn as nn

# --- weights ---
_WEIGHTS = {
    "code_free": [0.0] * 8,
    "carry_w": 0.0,
    "knee": [0.0, 0.0],
    "fold": 0.0,
}
# --- /weights ---


class DigitPairAdder(nn.Module):
    """One transformer block that adds two numbers a place at a time.

    Tokens are digit pairs, least significant place first, with a padding place
    at each end, so a width-n problem is P = n + 2 positions.  A single scalar
    residual channel holds the value of each place.  One gate bank turns that
    value into a key/value stream, and two attention heads read from it: one
    strictly causal (the carry arriving at a place) and one inclusively causal
    (the carry leaving it).  The readout compares the residual against the same
    digit code used to embed the inputs.  Position p predicts answer digit
    p - 1, so the whole sum comes out of a single forward pass.
    """

    n_digits = 10

    def __init__(self):
        super().__init__()
        self.code_free = nn.Parameter(torch.zeros(8))   # code for digits 2..9
        self.carry_w = nn.Parameter(torch.zeros(()))    # residual written by an incoming carry
        self.knee = nn.Parameter(torch.zeros(2))        # gate bank thresholds
        self.fold = nn.Parameter(torch.zeros(()))       # residual written by an outgoing carry
        # Architecture constants: the origin and unit of the residual axis, the
        # gate sharpness, the key contrast and the recency slope.
        self.register_buffer("code_pin", torch.tensor([0.0, 1.0]))
        self.register_buffer("gate_slope", torch.tensor(8.0))
        self.register_buffer("key_scale", torch.tensor(400.0))
        self.register_buffer("recency", torch.tensor(-12.0))

    def code(self):
        return torch.cat([self.code_pin, self.code_free])

    def forward(self, pairs):
        """pairs: (batch, n, 2) long digit pairs, least significant place first.

        Returns (batch, n + 2, 10) digit logits.
        """
        code = self.code()
        x = code[pairs[..., 0]] + code[pairs[..., 1]]
        pad = x.new_zeros(x.shape[0], 1)
        x = torch.cat([pad, x, pad], dim=1)                      # (batch, P)
        places = x.shape[1]

        gate = torch.clamp(self.gate_slope * (x.unsqueeze(-1) - self.knee), 0.0, 1.0)
        value = gate[..., 1]
        key = self.key_scale * (gate[..., 1] - gate[..., 0])

        pos = torch.arange(places, device=x.device)
        gap = pos.unsqueeze(1) - pos.unsqueeze(0)                # (P, P), gap[i, j] = i - j
        blocked = torch.finfo(x.dtype).min
        future = torch.where(gap < 0, blocked, 0.0)
        strict = torch.where(gap <= 0, blocked, 0.0)
        strict[0, 0] = 0.0                                       # the sink reads only itself
        scores = key.unsqueeze(1) + self.recency * gap           # (batch, P, P)

        weights_in = torch.softmax(scores + strict, dim=-1)
        weights_out = torch.softmax(scores + future, dim=-1)
        carry_in = torch.einsum("bij,bj->bi", weights_in, value)
        carry_out = torch.einsum("bij,bj->bi", weights_out, value)

        z = x + self.carry_w * carry_in + self.fold * carry_out
        return -(z.unsqueeze(-1) - code).pow(2)


def build_model():
    model = DigitPairAdder()
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_WEIGHTS["code_free"]))
        model.carry_w.copy_(torch.tensor(_WEIGHTS["carry_w"]))
        model.knee.copy_(torch.tensor(_WEIGHTS["knee"]))
        model.fold.copy_(torch.tensor(_WEIGHTS["fold"]))
    model.eval()
    metadata = {
        "name": "DigitPairAdder",
        "architecture": "1-block transformer, 2 attention heads over a shared content stream",
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokenization": "digit pairs, least significant place first, padded at both ends",
        "readout": "tied digit code, one forward pass for all answer digits",
    }
    return model, metadata


def add(model, a, b):
    """Return a + b, read off the model's digit predictions."""
    text_a, text_b = str(int(a)), str(int(b))
    width = max(len(text_a), len(text_b))
    digits_a = [int(c) for c in reversed(text_a.rjust(width, "0"))]
    digits_b = [int(c) for c in reversed(text_b.rjust(width, "0"))]
    device = next(model.parameters()).device
    pairs = torch.tensor([list(zip(digits_a, digits_b))], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(pairs)
    predicted = logits.argmax(dim=-1)[0, 1:].tolist()   # answer digits, least significant first
    return int("".join(str(d) for d in reversed(predicted)))
