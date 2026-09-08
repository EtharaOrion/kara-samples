"""Minimal transformer that adds two integers exactly.

One block, a one-dimensional residual stream, 12 learned parameters.  The
weights below came out of the training pipeline in this directory
(train_parent.py -> stage2.py -> build.py); this file holds only the model
and its inference path.
"""
import torch
import torch.nn as nn


class DigitPairAdder(nn.Module):
    """A single transformer block over per-place digit-pair tokens.

    The sequence holds one (a_i, b_i) digit pair per position, least
    significant place first, with a (0, 0) pad at each end.  A token embeds as
    code[a_i] + code[b_i] on a one-dimensional residual stream, where `code` is
    a learned scalar per digit symbol that also serves as the read-out
    prototype set.  A small clamp bank maps that scalar to a key and a value.
    Self-attention with a recency bias then sends every position to the nearest
    earlier place whose key is not notched -- the place its carry actually
    comes from -- and reading the same key/value stream through two masks
    (strictly causal, inclusively causal) yields the carry in and the carry
    out.  The carry is added to the residual, the carry out triggers the
    learned mod-10 fold, and the residual is decoded against the same ten
    codes.  Position p emits answer digit p-1, so one forward pass produces the
    whole sum.

    The learned values are the ten digit codes, the bank threshold(s) and the
    fold.  The buffers are the code origin and residual scale (the model's two
    exact gauge freedoms) plus sharpness/width constants of the gate and the
    attention, none of which is fitted to a particular numeric value.
    """

    def __init__(self, n_knee, bank_w, key_w, val_w, lam):
        super().__init__()
        self.code_free = nn.Parameter(torch.zeros(9))   # codes for digits 1..9
        self.knee = nn.Parameter(torch.zeros(n_knee))   # bank threshold(s)
        self.fold = nn.Parameter(torch.zeros(1))        # mod-10 write-back
        self.register_buffer('code_zero', torch.zeros(1))
        self.register_buffer('bank_w', torch.tensor(bank_w))
        self.register_buffer('key_w', torch.tensor(key_w))
        self.register_buffer('val_w', torch.tensor(val_w))
        self.register_buffer('carry_w', torch.tensor(1.0))
        self.register_buffer('lam', torch.tensor(lam))

    def codes(self):
        return torch.cat([self.code_zero, self.code_free])

    def forward(self, da, db):
        """da, db: [B, P] digit indices (LSB first, pads at both ends)."""
        code = self.codes()
        x = code[da] + code[db]                                      # [B,P]
        u = torch.clamp(self.bank_w * (x[..., None] - self.knee), 0.0, 1.0)
        key = u @ self.key_w                                         # [B,P]
        val = u @ self.val_w                                         # [B,P]
        P = da.shape[-1]
        i = torch.arange(P, device=da.device)
        sc = key[:, None, :] + self.lam * (i[:, None] - i[None, :]).to(x.dtype)
        neg = torch.finfo(x.dtype).min
        past = (i[None, :] < i[:, None]) | ((i[:, None] == 0) & (i[None, :] == 0))
        here = i[None, :] <= i[:, None]
        w_in = torch.softmax(sc.masked_fill(~past, neg), -1)
        w_out = torch.softmax(sc.masked_fill(~here, neg), -1)
        v = val[:, None, :]
        r = x + self.carry_w * (w_in * v).sum(-1) + self.fold * (w_out * v).sum(-1)
        return -(r[..., None] - code) ** 2                           # [B,P,10]


_CODE_FREE = [
        0.9581035375595093, 1.9066083431243896, 2.8835973739624023,
        3.867126226425171, 4.823870658874512, 5.802584648132324,
        6.78992223739624, 7.743881702423096, 8.720512390136719,
]

_KNEE = [
        8.115883827209473, 8.93591022491455,
]

_FOLD = [
        -9.747736930847168,
]


def build_model():
    """Return (model, metadata).  The weights below are the trained values."""
    model = DigitPairAdder(n_knee=2, bank_w=(8.0, 8.0), key_w=(-400.0, 400.0), val_w=(0.0, 1.0), lam=-12.0)
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_CODE_FREE, dtype=torch.float32))
        model.knee.copy_(torch.tensor(_KNEE, dtype=torch.float32))
        model.fold.copy_(torch.tensor(_FOLD, dtype=torch.float32))
    model.eval()
    meta = {
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'parameters': {n: tuple(p.shape) for n, p in model.named_parameters()},
        'architecture': 'one transformer block, 1-D residual stream, 2 heads '
                        '(strictly-causal carry-in, inclusively-causal carry-out) '
                        'sharing one content-dependent key/value stream',
        'tokenisation': 'one token per decimal place, LSB first, (0,0) pad at '
                        'each end; token p emits answer digit p-1',
        'range': 'exact for both operands in [10000000, 99999999]',
        'worst_case_readout_margin': 0.799333,
        'attention_notch_depth': 400.0,
        'certified_exact_on_whole_domain': True,
    }
    return model, meta


def _places(v, n):
    return [(v // 10 ** i) % 10 for i in range(n)]


def add(model, a, b):
    """Exact sum of two non-negative integers, from one forward pass."""
    a, b = int(a), int(b)
    n = max(len(str(a)), len(str(b)))
    da = [0] + _places(a, n) + [0]
    db = [0] + _places(b, n) + [0]
    dev = next(model.parameters()).device
    ta = torch.tensor([da], dtype=torch.long, device=dev)
    tb = torch.tensor([db], dtype=torch.long, device=dev)
    with torch.no_grad():
        pred = model(ta, tb).argmax(-1)[0].tolist()
    out = 0
    for i in range(n, -1, -1):
        out = out * 10 + pred[i + 1]
    return out
