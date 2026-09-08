"""Shipping template: the fully-learned form (every value an nn.Parameter).

build.py inlines the trained weights and copies this source verbatim into
submission.py, so what is graded is exactly what was trained.
"""

import torch
import torch.nn as nn

_W = {}  # WEIGHTS


class DigitPairAdder(nn.Module):
    """Single-block transformer over per-place digit-pair tokens.

    Sequence layout, LSB first, for n-digit operands (P = n + 2 positions):
        pos 0      : (0, 0) pad
        pos 1..n   : the digit pair (a_i, b_i) of place i-1
        pos n+1    : (0, 0) pad, which receives the final carry out
    Position p predicts answer digit p-1, so one forward pass yields every
    digit of the sum.

    A place token embeds as code[a] + code[b] from a single learned 10-entry
    table that is also used as the read-out prototypes.  A clamp bank turns
    that residual into a content-dependent key/value pair; two masked heads
    read it -- strictly causal (carry into the place) and inclusively causal
    (carry out of the place).
    """

    def __init__(self, C, U):
        super().__init__()
        self.C, self.U = C, U
        self.code = nn.Parameter(torch.zeros(10, C))
        self.Wb = nn.Parameter(torch.zeros(U, C))
        self.bb = nn.Parameter(torch.zeros(U))
        self.kw = nn.Parameter(torch.zeros(U))
        self.vw = nn.Parameter(torch.zeros(U))
        self.vb = nn.Parameter(torch.zeros(()))
        self.uA = nn.Parameter(torch.zeros(C))
        self.uB = nn.Parameter(torch.zeros(C))
        self.q = nn.Parameter(torch.zeros(()))
        self.lam = nn.Parameter(torch.zeros(()))
        self.ls = nn.Parameter(torch.zeros(()))

    def forward(self, a, b):
        """a, b: int64 [B, P] digit tensors.  Returns logits [B, P, 10]."""
        code = self.code
        x = code[a] + code[b]                                   # [B,P,C]
        g = (x @ self.Wb.t() + self.bb).clamp(0.0, 1.0)          # [B,P,U]
        k = g @ self.kw                                          # [B,P]
        v = g @ self.vw + self.vb                                # [B,P]

        P = a.shape[-1]
        i = torch.arange(P, device=a.device)
        dist = (i[:, None] - i[None, :]).to(x.dtype)
        mA = dist > 0
        mA = mA.clone()
        mA[0, 0] = True          # position 0 has no predecessor; its own pad
        mB = dist >= 0

        s = self.q * k[:, None, :] + self.lam * dist             # [B,P,P]
        neg = torch.finfo(s.dtype).min
        wA = torch.softmax(s.masked_fill(~mA, neg), -1)
        wB = torch.softmax(s.masked_fill(~mB, neg), -1)
        cA = (wA * v[:, None, :]).sum(-1)                        # [B,P]
        cB = (wB * v[:, None, :]).sum(-1)                        # [B,P]

        y = x + cA[..., None] * self.uA + cB[..., None] * self.uB
        d2 = ((y[:, :, None, :] - code) ** 2).sum(-1)            # [B,P,10]
        return -self.ls * d2


def build_model():
    m = DigitPairAdder(_W["C"], _W["U"])
    sd = {k: torch.tensor(v, dtype=torch.float32).reshape(p.shape)
          for k, p in m.named_parameters() for v in [_W[k]]}
    m.load_state_dict(sd)
    m.eval()
    meta = {
        "name": "digit-pair adder",
        "n_parameters": sum(p.numel() for p in m.parameters()),
        "architecture": "1 block: clamp bank -> 2 masked self-attention heads, tied code/read-out",
        "max_digits": 32,
    }
    return m, meta


def _digits(x, n):
    return [(x // 10 ** i) % 10 for i in range(n)]


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two non-negative integers, decoded from one forward pass."""
    dev = next(model.parameters()).device
    n = max(len(str(int(a))), len(str(int(b))))
    da = [0] + _digits(int(a), n) + [0]
    db = [0] + _digits(int(b), n) + [0]
    ta = torch.tensor([da], dtype=torch.long, device=dev)
    tb = torch.tensor([db], dtype=torch.long, device=dev)
    pred = model(ta, tb).argmax(-1)[0].tolist()
    return sum(d * 10 ** i for i, d in enumerate(pred[1:]))
