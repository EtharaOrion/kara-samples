"""8-digit addition with a 12-parameter transformer.

Weights were produced by the training pipeline in this workspace
(two_phase.py -> reduce.py -> finetune.py) and are inlined here as plain
literals.  This file contains the model and its inference path only.
"""

_PARAMS = {
    "code_free": [0.9158806800842285, 1.8399227857589722, 2.7395288944244385, 3.6635587215423584, 4.614743709564209, 5.524682521820068, 6.449503421783447, 7.375411510467529, 8.289571762084961],
    "knee": [7.8890814781188965, 8.497309684753418],
    "fold": [-9.339500427246094],
}
_BUFFERS = {
    "code0": [0.0],
    "bank_w": [8.0, 8.0],
    "key_w": [-400.0, 400.0],
    "val_w": [0.0, 1.0],
    "val_b": [0.0],
    "carry_w": [1.0],
    "lam": [-12.0],
    "ls": [1.0],
}

import torch
import torch.nn as nn


class DigitPairAdder(nn.Module):
    """A single transformer block that adds two integers of any digit width.

    Tokenisation (LSB-first).  For n digit places the sequence has P = n + 2
    positions: a (0,0) pad, then the n digit pairs (a_i, b_i), then a trailing
    (0,0) pad that holds the final carry.  Position p predicts answer digit
    p-1, so the whole sum comes out of a single forward pass.

    Residual stream.  One scalar channel.  A token embeds as
    code[a_i] + code[b_i] from a single learned 10-entry table, which is also
    used as the read-out prototypes -- so the read-out is tied to the
    embedding.

    Attention.  A two-unit clamp bank turns the residual into a content key and
    a content value.  Both are functions of the token's own digits, so the
    attention pattern is computed from the input, not fixed: the key is high on
    places that settle the carry and low on places that merely pass it along,
    and the recency bias then routes every query to the nearest earlier place
    that settles it.  Two masks read that one key/value stream -- strictly
    causal (the carry coming in to this place) and inclusively causal (the
    carry going out of it).
    """

    def __init__(self, params=None, buffers=None):
        super().__init__()
        params = _PARAMS if params is None else params
        buffers = _BUFFERS if buffers is None else buffers
        for k, v in params.items():
            self.register_parameter(
                k, nn.Parameter(torch.tensor(v, dtype=torch.float32)))
        for k, v in buffers.items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))

    def code(self):
        return torch.cat([self.code0, self.code_free])

    def forward(self, tok):
        """tok: (B, P, 2) integer digit pairs -> (B, P, 10) digit logits."""
        code = self.code()
        x = code[tok[..., 0]] + code[tok[..., 1]]              # (B,P)
        g = (self.bank_w * (x.unsqueeze(-1) - self.knee)).clamp(0.0, 1.0)
        k = g @ self.key_w                                     # (B,P)
        v = g @ self.val_w + self.val_b                        # (B,P)

        P = x.shape[-1]
        idx = torch.arange(P, device=x.device, dtype=x.dtype)
        dist = idx[:, None] - idx[None, :]                     # p - j
        logit = k.unsqueeze(-2) + self.lam * dist              # (B,P,P)
        neg = torch.finfo(logit.dtype).min / 4
        strict = dist > 0
        strict[0, 0] = True                     # keep the unused row finite
        incl = dist >= 0
        a_in = torch.softmax(logit.masked_fill(~strict, neg), dim=-1)
        a_out = torch.softmax(logit.masked_fill(~incl, neg), dim=-1)
        c_in = (a_in * v.unsqueeze(-2)).sum(-1)                # (B,P)
        c_out = (a_out * v.unsqueeze(-2)).sum(-1)

        y = x + self.carry_w * c_in + self.fold * c_out
        return -self.ls * (y.unsqueeze(-1) - code) ** 2


def _digits(v, n):
    d = []
    for _ in range(n):
        d.append(v % 10)
        v //= 10
    return d


def build_model():
    model = DigitPairAdder()
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "DigitPairAdder",
        "n_parameters": n,
        "architecture": "1-block transformer, 1-D residual stream, 2 masked "
                        "heads over a content-dependent key/value stream",
        "tokenization": "LSB-first digit pairs, one token per decimal place",
        "decoding": "single forward pass; every position emits its own digit",
    }
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two non-negative integers, read off one forward pass."""
    a, b = int(a), int(b)
    n = max(len(str(a)), len(str(b)))
    da, db = _digits(a, n), _digits(b, n)
    dev = next(model.parameters()).device
    tok = torch.tensor([[[0, 0]] + [[da[i], db[i]] for i in range(n)] + [[0, 0]]],
                       dtype=torch.long, device=dev)
    pred = model(tok)[0].argmax(-1).tolist()[1:]        # positions 1 .. n+1
    out = 0
    for i in range(len(pred) - 1, -1, -1):
        out = out * 10 + int(pred[i])
    return out
