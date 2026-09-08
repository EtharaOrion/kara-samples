"""An 8-digit adder: one transformer block over digit-pair tokens.

Twelve trained parameters -- the codes of the digits 1..9, the two thresholds
of the feed-forward bank, and the write-back weight of the base fold.  They
were fitted by gradient descent on sampled addition problems; the training code
lives beside this file and is not needed to run it.
"""

import torch
import torch.nn as nn


class DigitPairAdder(nn.Module):
    """One transformer block that adds two integers place by place.

    A shared 10-entry code table embeds digits.  Position p of the sequence
    holds the digit pair of one place and its residual stream starts at
    ``code[a_p] + code[b_p]`` -- a single scalar channel.

    The block is: a point-wise two-unit bank (the feed-forward sublayer), whose
    output is the key/value stream of one self-attention pattern read through
    two causal masks.  The strictly-causal head returns the carry coming *into*
    the place, the inclusively-causal head the carry going *out* of it; both
    are written back into the residual stream.  The read-out scores the
    residual against the same ten codes by squared distance.

    Nothing in the block is a fixed pattern: which position a query attends to
    is decided by the *content* of the key stream, which is computed from the
    digits.  Places whose digits sum to exactly 9 are pushed to a large
    negative key, which makes them unattendable, so every query lands on the
    nearest earlier place that actually settles the carry -- a distance that
    varies with the input, not with the position.
    """

    # Constants baked into the architecture.  These are not fitted values:
    # they are round numbers chosen so the bank saturates and the attention is
    # sharp, and `certify.py` proves the model is exact with them in place.
    SLOPE = 8.0        # bank gain; the clamp is saturated at every real input
    KEY = 400.0        # key contrast between a transparent place and the rest
    LAM = -12.0        # recency slope of the attention logits
    CARRY = 1.0        # carry-in write weight; fixes the code's overall scale

    def __init__(self):
        super().__init__()
        f = torch.float32          # pinned, so a changed global default dtype
                                   # cannot alter what this model computes
        # The twelve learned values.
        self.code = nn.Parameter(torch.zeros(9, dtype=f))  # digits 1..9
        self.knee = nn.Parameter(torch.zeros(2, dtype=f))  # bank thresholds
        self.fold = nn.Parameter(torch.zeros(1, dtype=f))  # fold write-back

        self.register_buffer("code0", torch.zeros(1, dtype=f))
        self.register_buffer("bank_w", torch.full((2,), self.SLOPE, dtype=f))
        self.register_buffer("key_w", torch.tensor([-self.KEY, self.KEY], dtype=f))
        self.register_buffer("val_w", torch.tensor([0.0, 1.0], dtype=f))
        self.register_buffer("carry_w", torch.tensor([self.CARRY], dtype=f))
        self.register_buffer("lam", torch.tensor(self.LAM, dtype=f))

    def codes(self):
        return torch.cat([self.code0, self.code], 0)

    def forward(self, tok_a, tok_b):
        """tok_a, tok_b: int64 [B, P] digit pairs.  Returns logits [B, P, 10]."""
        code = self.codes()
        x = code[tok_a] + code[tok_b]                                # [B,P]

        # Feed-forward sublayer: two clamped units on the place's digit sum.
        g = torch.clamp(self.bank_w * x.unsqueeze(-1) + self.knee, 0.0, 1.0)
        k = (g * self.key_w).sum(-1)                                 # [B,P]
        v = (g * self.val_w).sum(-1)                                 # [B,P]

        # Self-attention: one key/value stream, two causal masks.
        p = torch.arange(x.shape[-1], device=x.device)
        dist = p[:, None] - p[None, :]                               # [P,P]
        logit = k.unsqueeze(1) + self.lam * dist                     # [B,P,P]
        floor = torch.full_like(logit, torch.finfo(logit.dtype).min)
        a_in = torch.softmax(torch.where(dist > 0, logit, floor), -1)
        a_out = torch.softmax(torch.where(dist >= 0, logit, floor), -1)
        carry_in = (a_in * v.unsqueeze(1)).sum(-1)                   # [B,P]
        carry_out = (a_out * v.unsqueeze(1)).sum(-1)                 # [B,P]

        y = x + self.carry_w * carry_in + self.fold * carry_out
        return -(y.unsqueeze(-1) - code) ** 2


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Return a + b, decoded from one forward pass of `model`."""
    n = max(len(str(int(a))), len(str(int(b))), 1)
    ta = [0] + [(int(a) // 10 ** i) % 10 for i in range(n)] + [0]
    tb = [0] + [(int(b) // 10 ** i) % 10 for i in range(n)] + [0]
    dev = next(model.parameters()).device
    ta = torch.tensor([ta], dtype=torch.long, device=dev)
    tb = torch.tensor([tb], dtype=torch.long, device=dev)
    digits = model(ta, tb)[0].argmax(-1).tolist()
    out = 0
    for i in range(n + 1, 0, -1):          # positions 1..n+1 hold the answer
        out = out * 10 + int(digits[i])
    return out


# The twelve trained values.
_CODE = [
    1.0826053619384766,
    2.1275217533111572,
    3.1725776195526123,
    4.224531650543213,
    5.278218746185303,
    6.330873966217041,
    7.376132011413574,
    8.421201705932617,
    9.503911972045898,
]

_KNEE = [
    -68.67023468017578,
    -77.03133392333984,
]

_FOLD = [
    -10.535929679870605,
]


def build_model():
    """Return the trained model and a short description of it."""
    model = DigitPairAdder()
    with torch.no_grad():
        model.code.copy_(torch.tensor(_CODE, dtype=torch.float32))
        model.knee.copy_(torch.tensor(_KNEE, dtype=torch.float32))
        model.fold.copy_(torch.tensor(_FOLD, dtype=torch.float32))
    model.eval()
    meta = {
        "architecture": "one transformer block: point-wise bank -> single-head self-attention read through a strictly-causal and an inclusively-causal mask -> tied nearest-code read-out",
        "residual_channels": 1,
        "attention_heads": 1,
        "bank_units": 2,
        "vocabulary": 10,
        "trained_parameters": 12,
        "digits_per_forward_pass": "all of them",
        "source_checkpoint": "run15_ship.pt",
        "source_member": 169
        }
    meta["parameter_count"] = sum(p.numel() for p in model.parameters())
    return model, meta
