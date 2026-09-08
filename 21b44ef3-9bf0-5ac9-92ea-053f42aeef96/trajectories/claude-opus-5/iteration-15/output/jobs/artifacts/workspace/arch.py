"""Model definition.

The text between the SHIPPED markers below is copied verbatim into
`/workspace/submission.py` by `build.py`, so it must depend on nothing but
`torch`.  Training code lives in `ens.py` / `train.py` and never ships.

Layout of the computation (little-endian, one place per position):

    position p:      0        1        2      ...    n       n+1
    token   (a,b):  (0,0)  (a0,b0)  (a1,b1)  ...  (a_{n-1},b_{n-1})  (0,0)
    read-out:        --    digit0   digit1   ...   digit_{n-1}   digit_n

Position 0 is a leading pad that gives the strictly-causal head something to
attend to at the least-significant place; position n+1 is the trailing slot
where the final carry surfaces.  All n+1 answer digits come out of a single
forward pass.
"""

# --- BEGIN SHIPPED ---
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
# --- END SHIPPED ---


def build_model():
    """Present for parity with the shipped file; weights are all zero here."""
    m = DigitPairAdder()
    m.eval()
    return m, {"params": sum(p.numel() for p in m.parameters())}
