"""The model that ships.  build.py copies the class source below verbatim into
submission.py, so this file must stay importable with only torch."""
import torch
import torch.nn as nn


# --- BEGIN SHIPPED CLASS ---
class DigitPairAdder(nn.Module):
    """A single transformer block over per-place digit-pair tokens.

    The sequence holds one (a_i, b_i) digit pair per position, least
    significant place first, with a (0, 0) pad at each end.  A token embeds as
    code[a_i] + code[b_i] on a one-dimensional residual stream, where `code` is
    a learned scalar per digit symbol that also serves as the read-out
    prototype set.  A two-unit clamp bank reads that scalar against a learned
    threshold at two fixed slopes and turns it into a key and a value: the
    value says whether the place stands at or above the threshold, and the key
    is notched wherever the two units disagree.  Self-attention with a recency
    bias then sends every position to the nearest earlier place whose key is
    not notched -- the place its carry actually comes from -- and reading the
    same key/value stream through two masks (strictly causal, inclusively
    causal) yields the carry in and the carry out.  The carry is added to the
    residual, the carry out triggers the learned mod-10 fold, and the residual
    is decoded against the same ten codes.  Position p emits answer digit p-1,
    so one forward pass produces the whole sum.

    Learned: the nine free digit codes, the bank threshold and the fold.  The
    buffers are the model's two exact gauge freedoms (the code origin and the
    residual scale) and four shape constants -- the two bank slopes, the key
    contrast and the recency bias.  With the learned weights held fixed, each
    of those four keeps the model exact over a wide band around its shipped
    value, so none of them is carrying arithmetic (see band.py).
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
# --- END SHIPPED CLASS ---


# --- BEGIN SHIPPED HELPERS ---
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
# --- END SHIPPED HELPERS ---
