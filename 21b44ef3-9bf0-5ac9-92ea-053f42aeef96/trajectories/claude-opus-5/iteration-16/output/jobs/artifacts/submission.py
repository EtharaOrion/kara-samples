"""Minimal transformer that adds two integers digit by digit.

Twelve learned parameters:
    code  (9) -- the residual value of digits 1..9 (digit 0 pins the origin);
                 the same table is used as the read-out prototypes
    knee  (2) -- the two clamp-bank thresholds
    fold  (1) -- the mod-10 write-back

Everything else is a fixed architectural constant held as a buffer: the bank
slope, the +-key/0-1-value read-off of the bank gate, the carry-in write-back
(which fixes the residual scale) and the relative-distance bias slope.

Sequence layout, least-significant digit first, P = n + 2 positions for
n-digit operands:

    pos 0      : (0, 0) pad
    pos 1..n   : the digit pair (a_i, b_i) of place i-1
    pos n+1    : (0, 0) pad, which receives the final carry out

Position p predicts answer digit p-1, so a single forward pass produces every
digit of the sum.

Mechanism the model learned: `code` is an arithmetic ramp, so the residual at a
place is proportional to a+b.  The clamp bank splits places into three classes
-- absorb (a+b <= 8), transparent (a+b == 9) and generate (a+b >= 10) -- and
writes a key that is notched far down on exactly the transparent places, making
them unattendable.  With a recency bias on the remaining places, each query
therefore lands on the nearest earlier place that is not transparent, whose
value says whether it generated a carry.  That is carry lookahead: the
strictly-causal head returns the carry into a place, the inclusively-causal head
the carry out of it, and `fold` subtracts ten times a code step when the place
carries.  Attention is genuinely content-dependent -- which places are
transparent is a function of the digits, not of position.
"""

import torch
import torch.nn as nn

_W = {
    'code': [0.9999319314956665, 1.9940857887268066, 2.990553140640259, 3.988168954849243, 4.9857892990112305, 5.983412265777588, 6.979879856109619, 7.974030017852783, 8.973965644836426],
    'knee': [-65.5589370727539, -73.29173278808594],
    'fold': -9.973384857177734,
}


class DigitPairAdder(nn.Module):

    def __init__(self):
        super().__init__()
        self.code = nn.Parameter(torch.zeros(9))
        self.knee = nn.Parameter(torch.zeros(2))
        self.fold = nn.Parameter(torch.zeros(()))
        # fixed architectural constants (not fitted)
        self.register_buffer("bank_w", torch.tensor(8.0))
        self.register_buffer("key_w", torch.tensor([-400.0, 400.0]))
        self.register_buffer("val_w", torch.tensor([0.0, 1.0]))
        self.register_buffer("carry_w", torch.tensor(1.0))
        self.register_buffer("lam", torch.tensor(-12.0))

    def prototypes(self):
        """The ten digit codes; digit 0 is pinned at the origin."""
        return torch.cat([self.code.new_zeros(1), self.code])

    def forward(self, a, b):
        """a, b: int64 [B, P] digit tensors.  Returns logits [B, P, 10]."""
        c = self.prototypes()
        x = c[a] + c[b]                                          # [B,P] residual

        g = (self.bank_w * x[..., None] + self.knee).clamp(0.0, 1.0)   # [B,P,2]
        k = g @ self.key_w                                       # [B,P] key
        v = g @ self.val_w                                       # [B,P] value

        P = a.shape[-1]
        i = torch.arange(P, device=a.device)
        dist = (i[:, None] - i[None, :]).to(x.dtype)
        mA = dist > 0
        mA = mA.clone()
        mA[0, 0] = True      # position 0 has no predecessor; attend to its own pad
        mB = dist >= 0

        s = k[:, None, :] + self.lam * dist                      # [B,P,P]
        neg = torch.finfo(s.dtype).min
        wA = torch.softmax(s.masked_fill(~mA, neg), -1)          # strictly causal
        wB = torch.softmax(s.masked_fill(~mB, neg), -1)          # inclusively causal
        carry_in = (wA * v[:, None, :]).sum(-1)                  # [B,P]
        carry_out = (wB * v[:, None, :]).sum(-1)                 # [B,P]

        y = x + self.carry_w * carry_in + self.fold * carry_out
        return -(y[..., None] - c) ** 2                          # [B,P,10]


def build_model():
    m = DigitPairAdder()
    with torch.no_grad():
        m.code.copy_(torch.tensor(_W["code"], dtype=torch.float32))
        m.knee.copy_(torch.tensor(_W["knee"], dtype=torch.float32))
        m.fold.copy_(torch.tensor(_W["fold"], dtype=torch.float32))
    m.eval()
    meta = {
        "name": "digit-pair adder",
        "n_parameters": sum(p.numel() for p in m.parameters()),
        "learned": {"code[1..9]": 9, "bank knees": 2, "mod-10 fold": 1},
        "architecture": "one block: clamp bank -> two masked self-attention "
                        "heads sharing one content-dependent key/value stream, "
                        "digit code tied as read-out prototypes",
        "max_digits": 30,
    }
    return m, meta


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two non-negative integers, decoded from one forward pass."""
    a, b = int(a), int(b)
    dev = next(model.parameters()).device
    n = max(len(str(a)), len(str(b)))
    da = [0] + [(a // 10 ** i) % 10 for i in range(n)] + [0]
    db = [0] + [(b // 10 ** i) % 10 for i in range(n)] + [0]
    ta = torch.tensor([da], dtype=torch.long, device=dev)
    tb = torch.tensor([db], dtype=torch.long, device=dev)
    digits = model(ta, tb).argmax(-1)[0].tolist()[1:]
    return sum(d * 10 ** i for i, d in enumerate(digits))
