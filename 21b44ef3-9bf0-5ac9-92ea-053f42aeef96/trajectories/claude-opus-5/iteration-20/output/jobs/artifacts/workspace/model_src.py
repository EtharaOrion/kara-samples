"""Single source of truth for the shipped model.

`build.py` splices the region between the BEGIN/END SHIPPED markers verbatim
into `/workspace/submission.py`, prefixed by the trained weight literals.  The
placeholder weights below let this file be imported directly for testing.
"""

import torch

_WEIGHTS = {
    "code_free": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "carry_w": 1.0,
    "knee": [8.5, 9.5],
    "fold": -10.0,
}


# ---- BEGIN SHIPPED ----
class DigitPairAdder(torch.nn.Module):
    """One transformer block over per-place digit-pair tokens.

    The residual stream is a single scalar channel.  Token ``p`` carries the
    pair of digits ``(a_p, b_p)`` at place ``p`` (least significant first) and
    is embedded as ``code[a_p] + code[b_p]`` from one learned 10-entry table
    that also serves as the read-out prototypes.  A two-unit gate bank turns
    that scalar into the block's key/value stream, and two attention heads read
    the same stream through a strictly-causal and an inclusively-causal mask.

    Nothing here is specific to eight digits: the parameters are
    position-independent, so a sequence of any length runs unchanged.
    """

    def __init__(self):
        super().__init__()
        # Learned values (12).
        self.code_free = torch.nn.Parameter(torch.zeros(8))
        self.carry_w = torch.nn.Parameter(torch.zeros(()))
        self.knee = torch.nn.Parameter(torch.zeros(2))
        self.fold = torch.nn.Parameter(torch.zeros(()))
        # Architectural constants: the origin and unit of the residual axis,
        # the gate slope, the key contrast and the recency slope.
        self.register_buffer("code01", torch.tensor([0.0, 1.0]))
        self.register_buffer("bank_w", torch.tensor(8.0))
        self.register_buffer("key_w", torch.tensor(400.0))
        self.register_buffer("lam", torch.tensor(-12.0))

    @property
    def code(self):
        return torch.cat([self.code01, self.code_free])

    def forward(self, tokens):
        """tokens: (..., P, 2) long digit pairs -> (..., P, 10) digit logits."""
        code = self.code
        x = code[tokens[..., 0]] + code[tokens[..., 1]]

        gate = torch.clamp(self.bank_w * (x.unsqueeze(-1) - self.knee), 0.0, 1.0)
        value = gate[..., 1]
        key = self.key_w * (gate[..., 1] - gate[..., 0])

        p = x.shape[-1]
        pos = torch.arange(p, device=x.device)
        delta = pos.unsqueeze(1) - pos.unsqueeze(0)
        logits = key.unsqueeze(-2) + self.lam * delta
        blocked = torch.finfo(logits.dtype).min
        earlier = (delta > 0) | ((pos.unsqueeze(1) == 0) & (pos.unsqueeze(0) == 0))
        upto = delta >= 0
        w_in = torch.softmax(torch.where(earlier, logits, blocked), dim=-1)
        w_out = torch.softmax(torch.where(upto, logits, blocked), dim=-1)
        carry_in = (w_in * value.unsqueeze(-2)).sum(-1)
        carry_out = (w_out * value.unsqueeze(-2)).sum(-1)

        stream = x + self.carry_w * carry_in + self.fold * carry_out
        return -(stream.unsqueeze(-1) - code) ** 2


def build_model():
    model = DigitPairAdder()
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_WEIGHTS["code_free"]))
        model.carry_w.fill_(_WEIGHTS["carry_w"])
        model.knee.copy_(torch.tensor(_WEIGHTS["knee"]))
        model.fold.fill_(_WEIGHTS["fold"])
    model.eval()
    metadata = {
        "name": "digit-pair-adder",
        "architecture": "1-block transformer, scalar residual, 2 heads over 1 key/value stream",
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokenization": "one token per decimal place, least significant first, "
                        "plus a (0,0) pad at each end",
        "decoding": "one forward pass; position p reads out answer digit p-1",
        "operand_range": [10 ** 7, 10 ** 8 - 1],
    }
    return model, metadata


def _places(v, width):
    return [(v // 10 ** k) % 10 for k in range(width)]


def add(model, a, b):
    """Exact sum of two non-negative integers, read off one forward pass."""
    a, b = int(a), int(b)
    width = max(len(str(a)), len(str(b)))
    rows = [[0, 0]]
    rows += [[da, db] for da, db in zip(_places(a, width), _places(b, width))]
    rows += [[0, 0]]
    device = next(model.parameters()).device
    tokens = torch.tensor([rows], dtype=torch.long, device=device)
    with torch.no_grad():
        predicted = model(tokens).argmax(-1)[0].tolist()
    total = 0
    for k, digit in enumerate(predicted[1:]):
        total += int(digit) * 10 ** k
    return total
# ---- END SHIPPED ----
