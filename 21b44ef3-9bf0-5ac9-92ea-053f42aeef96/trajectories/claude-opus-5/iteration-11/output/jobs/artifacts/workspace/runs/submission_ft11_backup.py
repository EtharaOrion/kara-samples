"""Minimal transformer that adds two 8-digit numbers.

One attention block over per-place digit-pair tokens.  For n digit places the
sequence is n+2 long, LSB first:

    pos 0      (0, 0) pad     anchors "the carry into place 0 is zero"
    pos 1..n   (a_i, b_i)     the i-th place of the two operands
    pos n+1    (0, 0) pad     slot for the final carry-out digit

and position i predicts answer digit i-1, so a single forward pass emits all n+1
answer digits.  The residual stream is one scalar per position: a token embeds as
code[a] + code[b] from a learned 10-entry table, and the same table supplies the
read-out prototypes.

Training discovered this mechanism.  A place is "transparent" when a_i + b_i is
the special value theta (it passes an incoming carry through unchanged), and the
learned code puts every other place at least one clamp-width away from theta.  So
the bank

    upos = clamp(alpha * (x - theta), 0, 1)     uneg = clamp(alpha * (theta - x), 0, 1)

gives upos + uneg = 1 everywhere except a notch to 0 at transparent places, and
upos alone = 1 exactly at the places that generate a carry.  Used as the attention
key, the notch hides transparent places, so each position attends to the nearest
earlier place that is not transparent and reads off whether it generated a carry
-- that is the carry into this place.  The two heads share that one key/value
stream and differ only in their mask: the strictly-causal head returns the carry
in, the inclusively-causal head returns the carry out, and the carry out is what
folds the digit sum back below ten.

Parameters (11 of them): the 9 free code entries, theta, and the fold weight e2.
"""

import torch
import torch.nn as nn


class DigitPairAdder(nn.Module):
    def __init__(self):
        super().__init__()
        # --- learned ---------------------------------------------------------
        self.code_free = nn.Parameter(torch.zeros(9))
        self.theta = nn.Parameter(torch.zeros(()))
        self.e2 = nn.Parameter(torch.zeros(()))
        # --- fixed -----------------------------------------------------------
        # code_pin: origin of the 1-D residual stream (a coordinate choice).
        # e1:       scale of that stream (a coordinate choice).
        # alpha, kw, lam: shape constants with wide working bands, see NOTES.md.
        self.register_buffer("code_pin", torch.tensor([0.0]))
        self.register_buffer("e1", torch.tensor(1.0))
        self.register_buffer("alpha", torch.tensor(2.0))
        self.register_buffer("kw", torch.tensor(2000.0))
        self.register_buffer("lam", torch.tensor(-8.0))

    def code(self):
        return torch.cat([self.code_pin, self.code_free])

    def forward(self, ab):
        """ab: int64 [B, P, 2] digit pairs -> logits [B, P, 10]."""
        P = ab.shape[1]
        code = self.code()
        x = code[ab].sum(-1)                                     # [B, P]

        t = self.alpha * (x - self.theta)
        upos = t.clamp(0.0, 1.0)
        uneg = (-t).clamp(0.0, 1.0)
        key = self.kw * (upos + uneg)                            # [B, P]

        idx = torch.arange(P, device=ab.device)
        dist = (idx[:, None] - idx[None, :]).to(x.dtype)
        neg = torch.finfo(x.dtype).min / 4
        strict = idx[None, :] < idx[:, None]
        strict = strict.clone()
        strict[0, 0] = True              # row 0 has no earlier place; it reads the pad
        incl = idx[None, :] <= idx[:, None]
        base = key[:, None, :] + self.lam * dist

        val = upos[:, None, :]
        a1 = torch.softmax(base.masked_fill(~strict, neg), dim=-1)
        a2 = torch.softmax(base.masked_fill(~incl, neg), dim=-1)
        o1 = (a1 * val).sum(-1)                                  # carry in
        o2 = (a2 * val).sum(-1)                                  # carry out

        y = x + self.e1 * o1 + self.e2 * o2
        d = y[..., None] - code
        return -d * d


_CODE_FREE = [1.0868616104125977, 2.1439130306243896, 3.204136610031128, 4.264296531677246, 5.3243727684021, 6.384556293487549, 7.444823265075684, 8.499720573425293, 9.572761535644531]
_THETA = 9.112385749816895
_E2 = -10.605270385742188


def build_model():
    model = DigitPairAdder()
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_CODE_FREE))
        model.theta.copy_(torch.tensor(_THETA))
        model.e2.copy_(torch.tensor(_E2))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "digit-pair-adder",
        "n_parameters": n,
        "architecture": "1 block: clamp bank -> 2-head causal self-attention "
                        "(shared key/value) -> tied 1-D digit-code read-out",
        "digits": 8,
    }
    return model, meta


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Exact sum of two integers, decoded from one forward pass of `model`."""
    a, b = int(a), int(b)
    n = max(len(str(a)), len(str(b)), 1)
    da = [(a // 10 ** i) % 10 for i in range(n)]
    db = [(b // 10 ** i) % 10 for i in range(n)]
    dev = next(model.parameters()).device
    ab = torch.zeros(1, n + 2, 2, dtype=torch.long, device=dev)
    ab[0, 1:n + 1, 0] = torch.tensor(da, dtype=torch.long, device=dev)
    ab[0, 1:n + 1, 1] = torch.tensor(db, dtype=torch.long, device=dev)
    digits = model(ab).argmax(-1)[0, 1:n + 2].tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))
