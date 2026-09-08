"""Source of truth for the graded file.

`build.py` copies this file verbatim and replaces the `_WEIGHTS = None` line
with a literal dict of trained floats.  Nothing else changes, so the shipped
`submission.py` is exactly this module plus its weights.  Only `torch` is
imported.
"""

import torch
import torch.nn as nn

_WEIGHTS = {
    "code_free": [1.9969927072525024, 2.9947073459625244, 3.9941275119781494, 4.992620468139648, 5.991181373596191, 6.989222526550293, 7.987253189086914, 8.985689163208008],
    "carry_w": 1.000701665878296,
    "knee": [8.360455513000488, 9.355072021484375],
    "fold": -9.987625122070312,
}


class DigitPairAdder(nn.Module):
    """One transformer block over per-place digit-pair tokens.

    Layout.  LSB first, P = n + 2 positions.  Position 0 and position n+1 are
    (0, 0) pads; positions 1..n carry the digit pair (a_i, b_i).  The residual
    stream is one scalar per position.  A token is embedded as
    code[a_i] + code[b_i] from a single learned 10-entry table which is also
    the read-out prototype table (tied).  Position p predicts answer digit
    p-1, so the whole sum comes out of one forward pass.

    Block.  A two-unit clamp bank maps the token to one content stream that is
    read as both key and value by two heads which differ only in their causal
    mask: the strict head (j < p) delivers the carry into place p, the
    inclusive head (j <= p) delivers the carry out of place p.  Attention
    scores are the content key plus a recency bias, so routing is decided by
    what the tokens are, not by where they sit.

    Learned parameters (12): code[2..9] (8), carry_w, knee[0], knee[1], fold.

    Buffers hold no learned facts.  `code_pin` fixes the origin and the unit
    of the residual axis (the two gauge freedoms of a scalar stream); the
    gate slope, key contrast and recency bias are sharpness constants whose
    exact values the computed function does not depend on -- any slope that
    saturates the bank, any key contrast that outruns the recency spread and
    any recency strong enough to order the candidates give the same answers.
    """

    def __init__(self):
        super().__init__()
        self.code_free = nn.Parameter(torch.zeros(8))    # code[2..9]
        self.carry_w = nn.Parameter(torch.zeros(()))     # carry write-back
        self.knee = nn.Parameter(torch.zeros(2))         # bank thresholds
        self.fold = nn.Parameter(torch.zeros(()))        # mod-base fold
        self.register_buffer("code_pin", torch.tensor([0.0, 1.0]))
        self.register_buffer("gate_slope", torch.tensor(8.0))
        self.register_buffer("key_scale", torch.tensor(400.0))
        self.register_buffer("recency", torch.tensor(-12.0))

    def code(self):
        return torch.cat([self.code_pin, self.code_free])

    def forward(self, da, db):
        """da, db: (B, n) long digit tensors, least significant place first.

        Returns (B, n+1, 10) logits; entry (:, k, :) scores answer digit k.
        """
        c = self.code()
        z = c[da] + c[db]                                     # (B, n)
        pad = torch.zeros(z.shape[0], 1, dtype=z.dtype, device=z.device)
        pad = pad + c[0] + c[0]
        z = torch.cat([pad, z, pad], dim=1)                   # (B, P)

        u = torch.clamp(self.gate_slope * (z.unsqueeze(-1) - self.knee), 0.0, 1.0)
        val = u[..., 1]                                       # (B, P)
        key = self.key_scale * (u[..., 1] - u[..., 0])        # (B, P)

        pos = torch.arange(z.shape[1], device=z.device)
        gap = pos.unsqueeze(-1) - pos.unsqueeze(0)            # (P, P) query - key
        score = key.unsqueeze(1) + self.recency * gap         # (B, P, P)
        blocked = torch.full_like(gap, -1e9, dtype=score.dtype)
        strict = score + torch.where(gap > 0, 0.0, blocked)
        incl = score + torch.where(gap >= 0, 0.0, blocked)

        carry_in = torch.einsum("bpq,bq->bp", strict.softmax(-1), val)
        carry_out = torch.einsum("bpq,bq->bp", incl.softmax(-1), val)
        res = z + self.carry_w * carry_in + self.fold * carry_out

        logits = -(res.unsqueeze(-1) - c).abs()               # (B, P, 10)
        return logits[:, 1:, :]


def build_model():
    model = DigitPairAdder()
    with torch.no_grad():
        for name, value in _WEIGHTS.items():
            getattr(model, name).copy_(torch.tensor(value, dtype=torch.float32))
    model.eval()
    metadata = {
        "name": "DigitPairAdder",
        "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": "1 block: clamp bank -> 2-head self-attention (strict "
                        "and inclusive causal over one content key/value "
                        "stream) -> tied read-out on a scalar residual stream",
        "tokens": "one token per decimal place, embedded as code[a]+code[b]",
        "decode": "argmax over tied prototypes at positions 1..n+1, one forward pass",
    }
    return model, metadata


def add(model, a, b):
    """Exact sum of two non-negative integers, read off one forward pass."""
    da = [int(ch) for ch in reversed(str(int(a)))]
    db = [int(ch) for ch in reversed(str(int(b)))]
    while len(da) < len(db):
        da.append(0)
    while len(db) < len(da):
        db.append(0)
    device = next(model.parameters()).device
    ta = torch.tensor([da], dtype=torch.long, device=device)
    tb = torch.tensor([db], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ta, tb)
    digits = logits[0].argmax(dim=-1).tolist()
    text = "".join([str(d) for d in reversed(digits)]).lstrip("0")
    if text == "":
        text = "0"
    return int(text)
