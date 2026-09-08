import torch
import torch.nn as nn


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
