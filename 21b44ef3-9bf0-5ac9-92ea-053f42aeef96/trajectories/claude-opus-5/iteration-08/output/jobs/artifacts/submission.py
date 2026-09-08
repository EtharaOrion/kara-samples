"""Exact 8-digit addition from a 18-parameter transformer.

The model is a single transformer block over LSB-first digit-pair tokens: token
i embeds ``code[a_i] + code[b_i]`` from one shared 10-entry code table (also
used as the output prototypes), and the block runs FFN -> strictly-causal
single-head self-attention -> FFN over a 1-dimensional residual stream.

The FFN before attention turns the local digit sum a_i+b_i into a key that is
sharply lowest exactly where a place is carry-transparent, so each position's
attention lands on the nearest earlier place that either generates or absorbs a
carry and reads that place's carry off the same axis; the FFN after attention
folds the result mod 10.  Attention is therefore doing the long-range work, and
the map it computes is a function of the digits it is given.

All weights below were produced by training (see train.py / README.md); the
answer is the argmax decode of a single forward pass.
"""
import torch
import torch.nn as nn

_CFG = {'C': 1, 'U': 3, 'U2': 3, 'f1_in': 'sign', 'f1_sign': 1.0, 'f2_in': 'sign', 'f2_sign': -1.0, 'kv': 'share', 'lam': -2.0, 'lam_learn': False, 'logit_scale': False, 'o_pin': True, 'p_rows': [0, 2], 'pin_row': 0, 'tie': True, 'y_bias': False}

_W = {
    'code_free': [
        [0.8943620920181274],
        [0.7867615818977356],
        [0.6782858967781067],
        [0.5707374215126038],
        [0.46288853883743286],
        [0.35389629006385803],
        [0.2472131848335266],
        [0.13953015208244324],
        [0.03161947429180145],
    ],
    'b1': [-1.03462553024292, -0.9871217012405396, -1.0717661380767822],
    'o_free': [-0.46004387736320496, -0.5396133661270142],
    'b_q': 1256.1204833984375,
    'w_v': [-55.95092010498047],
    'p_out': [
        [-28.916948318481445],
        [28.921024322509766],
    ],
}


class DigitPairAdder(nn.Module):
    """One transformer block over LSB-first digit-pair tokens.

    Token i is ``code[a_i] + code[b_i]`` drawn from a single shared 10-entry
    code table that is also used as the output prototypes.  The block is
    FFN -> strictly-causal single-head self-attention -> FFN over a
    C-dimensional residual stream, and position i predicts answer digit i, so
    the whole answer comes out of one forward pass.

    Positions 0 and P-1 hold the (0,0) place below the ones digit (the carry-in
    sink) and the (0,0) place above the most significant digit (which receives
    the carry-out).  No parameter is indexed by position, so the same weights
    run at any number of places.
    """

    def __init__(self, cfg):
        super().__init__()
        cfg = dict(cfg)
        self.cfg = cfg
        C, U, U2 = cfg["C"], cfg["U"], cfg["U2"]
        pin = cfg["pin_row"]
        self.code_free = nn.Parameter(torch.zeros(10 - (1 if pin >= 0 else 0), C))
        if pin >= 0:
            e = torch.zeros(1, C)
            e[0, 0] = 1.0
            self.register_buffer("code_pin", e)
        if cfg["f1_in"] == "free":
            self.w1 = nn.Parameter(torch.zeros(C, U))
        else:
            self.register_buffer("s1", torch.tensor(float(cfg["f1_sign"])))
        self.b1 = nn.Parameter(torch.zeros(U))
        if cfg["kv"] in ("share", "one"):
            self.o_free = nn.Parameter(torch.zeros(U - 1 if cfg["o_pin"] else U))
            if cfg["o_pin"]:
                self.register_buffer("o_pinned", torch.ones(1))
            if cfg["kv"] == "share":
                self.b_q = nn.Parameter(torch.zeros(()))
            self.w_v = nn.Parameter(torch.zeros(C))
        else:
            self.k_o = nn.Parameter(torch.zeros(U))
            self.v_o = nn.Parameter(torch.zeros(U, C))
        if cfg["lam_learn"]:
            self.lam = nn.Parameter(torch.tensor(float(cfg["lam"])))
        else:
            self.register_buffer("lam_c", torch.tensor(float(cfg["lam"])))
        if not cfg["tie"]:
            if cfg["f2_in"] == "free":
                self.w2 = nn.Parameter(torch.zeros(C, U2))
            else:
                self.register_buffer("s2", torch.tensor(float(cfg["f2_sign"])))
            self.b2 = nn.Parameter(torch.zeros(U2))
        if cfg["y_bias"]:
            self.by = nn.Parameter(torch.zeros(()))
        if cfg["logit_scale"]:
            self.ls = nn.Parameter(torch.zeros(()))
        rows = cfg["p_rows"]
        self.p_out = nn.Parameter(torch.zeros(len(rows) if rows else (U if cfg["tie"] else U2), C))

    def code(self):
        p = self.cfg["pin_row"]
        if p < 0:
            return self.code_free
        return torch.cat([self.code_free[:p], self.code_pin, self.code_free[p:]], 0)

    def forward(self, da, db):
        cfg = self.cfg
        code = self.code()
        x = code[da] + code[db]
        if cfg["f1_in"] == "free":
            h = torch.relu(x @ self.w1 + self.b1)
        else:
            h = torch.relu(x[..., :1] * self.s1 + self.b1)
        if cfg["kv"] in ("share", "one"):
            o = torch.cat([self.o_pinned, self.o_free]) if cfg["o_pin"] else self.o_free
            z = h @ o
            q = self.b_q if cfg["kv"] == "share" else self.w_v[0]
            k, v = q * z, z[..., None] * self.w_v
        else:
            k, v = h @ self.k_o, h @ self.v_o
        P = x.shape[-2]
        idx = torch.arange(P, device=x.device)
        dist = idx[:, None] - idx[None, :]
        lam = self.lam if cfg["lam_learn"] else self.lam_c
        att = k[..., None, :] + lam * dist - 1e9 * (dist <= 0)
        w = torch.softmax(att, -1) * (idx > 0)[:, None]
        r = x + w @ v
        if cfg["tie"]:
            g = torch.relu(r @ self.w1 + self.b1) if cfg["f1_in"] == "free" else \
                torch.relu(r[..., :1] * self.s1 + self.b1)
        elif cfg["f2_in"] == "free":
            g = torch.relu(r @ self.w2 + self.b2)
        else:
            g = torch.relu(r[..., :1] * self.s2 + self.b2)
        if cfg["p_rows"]:
            g = g[..., cfg["p_rows"]]
        y = r + g @ self.p_out
        if cfg["y_bias"]:
            y = y + self.by
        d2 = ((y[..., None, :] - code) ** 2).sum(-1)
        return -(torch.exp(self.ls) * d2 if cfg["logit_scale"] else d2)


def build_model():
    """Return (model, metadata).  The weights are the trained ones above."""
    model = DigitPairAdder(_CFG)
    sd = model.state_dict()
    for k, v in _W.items():
        sd[k] = torch.tensor(v, dtype=torch.float32).reshape(sd[k].shape)
    model.load_state_dict(sd)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "digit-pair-adder",
        "param_count": n,
        "architecture": "1 transformer block (FFN -> strictly-causal 1-head self-attention -> FFN)",
        "residual_width": _CFG["C"],
        "tokens": "one token per decimal place, LSB first, embedding code[a_i]+code[b_i]",
        "readout": "nearest of the 10 tied code prototypes, at every position, in one forward pass",
        "trained_on": "synthetic digit pairs; held-out accuracy is measured on a hashed 1-in-16 split",
        "digits": "operands of any width; graded range is 8 digits",
    }
    return model, meta


def add(model, a, b):
    """Exact sum of two non-negative integers, decoded from one forward pass."""
    a, b = int(a), int(b)
    n = max(len(str(a)), len(str(b)))
    da = [0] + [(a // 10 ** i) % 10 for i in range(n)] + [0]
    db = [0] + [(b // 10 ** i) % 10 for i in range(n)] + [0]
    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor([da], device=dev), torch.tensor([db], device=dev))
    d = logits.argmax(-1)[0].tolist()
    return sum(d[i] * 10 ** (i - 1) for i in range(1, n + 2))
