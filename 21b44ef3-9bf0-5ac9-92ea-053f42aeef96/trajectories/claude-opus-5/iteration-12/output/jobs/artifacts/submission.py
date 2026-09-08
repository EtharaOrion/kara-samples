"""Minimal transformer for exact 8-digit decimal addition.

12 learned parameters.  See build_model() for what each of them is.
The weights below were produced by gradient descent on synthetic addition
problems (trainer: train.py, not imported here); this file contains the model
and its inference path only.
"""

import torch
import torch.nn as nn

class DigitPairAdder(nn.Module):
    """A one-block transformer that adds two decimal integers.

    Sequence layout, least-significant place first.  Position 0 and position
    P-1 carry the pad digit pair (0, 0); positions 1..P-2 carry the operand
    digit pairs.  Position i >= 1 emits the answer digit of place i-1, so a
    single forward pass produces the whole sum including the carry digit.

    The residual stream is one scalar per position,

        x_i = code[a_i] + code[b_i],

    where ``code`` is a single learned 10-entry table used both as the token
    embedding and, tied, as the read-out prototypes.

    The block is

        u_i     = clamp(bank_w * x_i + bank_bias, 0, 1)      (2 features)
        k_i     = key_w . u_i          (attention key)
        v_i     = val_w . u_i          (attention value)
        s_ij    = k_j + dist_bias * (i - j)                  (attention logit)
        A_i     = sum_{j <  i} softmax_j(s_ij) v_j           (strict head)
        B_i     = sum_{j <= i} softmax_j(s_ij) v_j           (inclusive head)
        y_i     = x_i + carry_w * A_i + fold * B_i

    Read-out is the tied prototype distance ``logits[i, d] = -(y_i-code[d])^2``,
    which is the ordinary tied-unembedding logit ``2*code[d]*y_i - code[d]^2``
    plus a term constant in d.

    Attention is content-based: the keys are a function of the token at each
    position, so which position a query reads depends on the operand digits,
    not on the position indices alone.
    """

    def __init__(self):
        super().__init__()
        # ---- learned parameters: 12 floats, and nothing else --------------
        self.code_free = nn.Parameter(torch.zeros(9))  # code[1..9]
        self.bank_bias = nn.Parameter(torch.zeros(2))  # the two clamp knees
        self.fold = nn.Parameter(torch.zeros(()))      # write weight, head 2

        # ---- architectural constants, fixed before training --------------
        # code[0] is the origin of the read-out line; bank_w sets the clamp
        # slope (any slope steep enough to saturate gives the same function on
        # the 100 reachable digit pairs); key_w / val_w say that the two heads
        # read one key stream and one value stream off the bank; dist_bias is
        # a fixed ALiBi-style distance bias; carry_w = 1 fixes the residual
        # scale gauge (rescaling code, fold and carry_w together, with the
        # inverse rescaling of bank_w, leaves every prediction unchanged).
        self.register_buffer("code_zero", torch.zeros(1))
        self.register_buffer("bank_w", torch.tensor([-2.0, 2.0]))
        self.register_buffer("key_w", torch.tensor([-200.0, -200.0]))
        self.register_buffer("val_w", torch.tensor([0.0, 1.0]))
        self.register_buffer("dist_bias", torch.tensor(-8.0))
        self.register_buffer("carry_w", torch.tensor(1.0))

    def code(self):
        """The 10 digit prototypes; entry 0 is pinned at the origin."""
        return torch.cat([self.code_zero, self.code_free])

    def forward(self, digits_a, digits_b, tau: float = 1.0, beta: float = 1.0):
        """digits_a/digits_b: (..., P) int tensors, least-significant first.

        Returns (..., P, 10) logits.

        ``tau`` and ``beta`` are both 1.0 for the model as defined and as
        shipped -- they are the two annealing knobs the training schedule
        moves.  ``tau`` scales the attention logits, ``beta`` scales the clamp
        bank's slope about its (unchanged) knees, so that early in training
        the bank is a soft ramp with a usable gradient and by the end it is the
        declared hard clamp.
        """
        code = self.code()
        x = code[digits_a] + code[digits_b]                       # (..., P)

        u = torch.clamp((x.unsqueeze(-1) * self.bank_w + self.bank_bias) * beta, 0.0, 1.0)
        k = (u * self.key_w).sum(-1)                              # (..., P)
        v = (u * self.val_w).sum(-1)                              # (..., P)

        p = x.shape[-1]
        pos = torch.arange(p, device=x.device)
        delta = pos.unsqueeze(-1) - pos.unsqueeze(-2)             # (P, P) = i-j
        logit = (k.unsqueeze(-2) + self.dist_bias * delta) * tau  # (..., P, P)

        strict = logit.masked_fill(delta <= 0, -1e9).softmax(-1)
        incl = logit.masked_fill(delta < 0, -1e9).softmax(-1)
        head1 = (strict * v.unsqueeze(-2)).sum(-1)
        head2 = (incl * v.unsqueeze(-2)).sum(-1)

        y = x + self.carry_w * head1 + self.fold * head2
        return -(y.unsqueeze(-1) - code) ** 2


def _digits(value, places):
    out = []
    for _ in range(places):
        out.append(value % 10)
        value //= 10
    return out


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two non-negative integers, from one forward pass."""
    places = max(len(str(int(a))), len(str(int(b))))
    da = [0] + _digits(int(a), places) + [0]
    db = [0] + _digits(int(b), places) + [0]
    device = model.code_zero.device
    ta = torch.tensor([da], dtype=torch.long, device=device)
    tb = torch.tensor([db], dtype=torch.long, device=device)
    pred = model(ta, tb).argmax(-1)[0].tolist()
    total = 0
    for i in range(places + 1, 0, -1):     # positions P-1 .. 1
        total = total * 10 + int(pred[i])
    return total


_WEIGHTS = {
    "code_free": [1.1934300661087036, 2.3201112747192383, 3.456360101699829, 4.593022346496582, 5.729194164276123, 6.864260196685791, 7.9870452880859375, 9.10706901550293, 10.309967041015625],
    "bank_bias": [21.650171279907227, -18.375497817993164],
    "fold": -11.37629222869873,
}


def build_model():
    """Return (model, metadata).  All 12 learned floats are nn.Parameters."""
    model = DigitPairAdder()
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_WEIGHTS["code_free"], dtype=torch.float32))
        model.bank_bias.copy_(torch.tensor(_WEIGHTS["bank_bias"], dtype=torch.float32))
        model.fold.copy_(torch.tensor(_WEIGHTS["fold"], dtype=torch.float32))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    metadata = {
        "name": "DigitPairAdder",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "parameter_breakdown": {
            "code_free": "9 - the learned digit code, tied embedding / read-out prototypes",
            "bank_bias": "2 - the two clamp knees that split a+b into <=8 / ==9 / >=10",
            "fold": "1 - the write weight of the carry-out head (the mod-10 fold)",
        },
        "architecture": "1 block: 2-unit clamp bank -> 2 masked attention heads "
                        "(strictly causal, inclusively causal) sharing one key "
                        "and one value stream -> tied prototype read-out",
        "sequence": "P = places + 2 tokens, least-significant place first, "
                    "(0,0) pads at both ends; every answer digit comes from "
                    "one forward pass",
        "input_range": [10000000, 99999999],
        "trained": "from random initialisation on synthetic addition problems",
    }
    return model, metadata
