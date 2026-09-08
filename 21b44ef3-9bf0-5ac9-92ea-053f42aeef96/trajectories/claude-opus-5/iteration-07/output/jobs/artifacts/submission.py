"""Minimal transformer that adds two 8-digit numbers.

One Macaron block (FFN -> single-head strictly-causal self-attention -> FFN)
over ten LSB-first digit-pair tokens; every position predicts its own answer
digit against a tied code table, so the whole sum comes from one forward pass.
All 70 floating-point parameters below were produced by gradient descent
(see train_ens.py / cascade.py in the training workspace).
"""

import torch
import torch.nn as nn


def default_cfg():
    """Defaults for every architecture switch.

    Switches select *structure* only (which weights exist, which are tied, and
    which residual axis a sublayer reads or writes).  Every floating-point value
    that the answer depends on is a trained ``nn.Parameter``; the handful of
    non-trained constants are 0.0 / 1.0 normalisation pins and a fixed distance
    slope, all recorded as buffers.
    """
    return dict(
        P=10,            # positions: 0 and P-1 are (0,0) pads, 1..P-2 are places
        D=3,             # residual width
        C=2,             # code dimension (code occupies residual axes [0:C])
        U1=6,            # pre-attention FFN units
        U2=4,            # post-attention FFN units
        KV=1,            # residual axis carrying key/value when kv == "id"
        f1_in="full",    # "full" | "id"   ("id": read residual axis 0, unit weight)
        f1_out="full",   # "full" | "axis" ("axis": write residual axis KV only)
        f1_tie=False,    # sum-to-zero constraint on f1 output weights
        f2_in="full",    # "full" | "id"
        f2_out="full",   # "full" | "axis"
        f2_tie=False,    # antisymmetric constraint on f2 output weights
        kv="full",       # "full" | "id"
        q="proj",        # "proj" | "const"
        wo="full",       # "full" | "axis" | "fix"
        lam="learn",     # "learn" | "fix"
        lam_fix=-1.0,    # fixed relative-distance slope when lam == "fix"
        code_fix=0,      # 0/1/2 normalisation pins on the code table
        temp=4.0,        # readout temperature (a positive scale; argmax-invariant)
    )


class DigitPairAdder(nn.Module):
    """A single Macaron transformer block over LSB-first digit-pair tokens.

    Position ``p`` embeds the pair of decimal digits at place ``p-1`` as
    ``code[a] + code[b]`` from one shared table; positions ``0`` and ``P-1``
    are ``(0, 0)`` pads.  The block is FFN -> single-head strictly-causal
    self-attention -> FFN, and every position predicts its own answer digit by
    comparing the residual against the same (tied) code table, so all nine
    digits of the sum come from one forward pass.
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = dict(default_cfg(), **(cfg or {}))
        self.cfg = cfg
        P = int(cfg["P"]); D = int(cfg["D"]); C = int(cfg["C"])
        U1 = int(cfg["U1"]); U2 = int(cfg["U2"])
        self.P, self.D, self.C, self.U1, self.U2 = P, D, C, U1, U2
        self.KV = int(cfg["KV"])
        self.temp = float(cfg["temp"])
        self.code_fix = int(cfg["code_fix"])
        self.f1_in = cfg["f1_in"]; self.f1_out = cfg["f1_out"]
        self.f2_in = cfg["f2_in"]; self.f2_out = cfg["f2_out"]
        self.f1_tie = bool(cfg["f1_tie"]); self.f2_tie = bool(cfg["f2_tie"])
        self.kv = cfg["kv"]; self.q = cfg["q"]; self.wo = cfg["wo"]
        self.lam_mode = cfg["lam"]

        # ---- code table -------------------------------------------------
        # Axis 0 may carry up to two normalisation pins: digit 0 -> 0.0 fixes
        # the origin of the code axis, digit 1 -> 1.0 fixes its unit.
        self.code_free0 = nn.Parameter(torch.zeros(10 - self.code_fix))
        if C > 1:
            self.code_rest = nn.Parameter(torch.zeros(10, C - 1))
        self.register_buffer("k_zero", torch.zeros(1), persistent=False)
        self.register_buffer("k_one", torch.ones(1), persistent=False)
        if D > C:
            self.register_buffer("code_pad", torch.zeros(10, D - C), persistent=False)

        # ---- pre-attention FFN -----------------------------------------
        if self.f1_in == "full":
            self.W1 = nn.Parameter(torch.zeros(D, U1))
        self.b1 = nn.Parameter(torch.zeros(U1))
        n_o1 = U1 - 1 if self.f1_tie else U1
        if self.f1_out == "full":
            self.O1 = nn.Parameter(torch.zeros(U1, D))
        else:
            self.o1 = nn.Parameter(torch.zeros(n_o1))

        # ---- attention ---------------------------------------------------
        if self.kv == "full":
            self.Wk = nn.Parameter(torch.zeros(D, 1))
            self.Wv = nn.Parameter(torch.zeros(D, 1))
        if self.q == "proj":
            self.Wq = nn.Parameter(torch.zeros(D, 1))
        self.bq = nn.Parameter(torch.zeros(1))
        if self.wo == "full":
            self.Wo = nn.Parameter(torch.zeros(1, D))
        elif self.wo == "axis":
            self.wo_s = nn.Parameter(torch.zeros(1))
        if self.lam_mode == "learn":
            self.lam = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("lam_c", torch.tensor(float(cfg["lam_fix"])),
                                 persistent=False)

        # ---- post-attention FFN -----------------------------------------
        if self.f2_in == "full":
            self.W2 = nn.Parameter(torch.zeros(D, U2))
        self.b2 = nn.Parameter(torch.zeros(U2))
        n_o2 = U2 // 2 if self.f2_tie else U2
        if self.f2_out == "full":
            self.O2 = nn.Parameter(torch.zeros(U2, D))
        else:
            self.o2 = nn.Parameter(torch.zeros(n_o2))

        # ---- fixed geometry ----------------------------------------------
        e0 = torch.zeros(D); e0[0] = 1.0
        ekv = torch.zeros(D); ekv[self.KV] = 1.0
        self.register_buffer("e0", e0, persistent=False)
        self.register_buffer("ekv", ekv, persistent=False)
        idx = torch.arange(P)
        dist = (idx[:, None] - idx[None, :]).float()
        allow = idx[None, :] < idx[:, None]           # strictly causal
        allow[0, 0] = True                            # position 0 has no past
        self.register_buffer("dist", dist, persistent=False)
        self.register_buffer("mask_bias",
                             torch.where(allow, torch.zeros(P, P),
                                         torch.full((P, P), -1e9)),
                             persistent=False)

    # -------------------------------------------------------------------
    def code_table(self):
        cf = self.code_fix
        if cf == 0:
            col0 = self.code_free0
        elif cf == 1:
            col0 = torch.cat([self.code_free0[:1], self.k_one, self.code_free0[1:]])
        else:
            col0 = torch.cat([self.k_zero, self.k_one, self.code_free0])
        code = col0.unsqueeze(1)
        if self.C > 1:
            code = torch.cat([code, self.code_rest], dim=1)
        if self.D > self.C:
            code = torch.cat([code, self.code_pad], dim=1)
        return code

    def o1_vec(self):
        if not self.f1_tie:
            return self.o1
        return torch.cat([self.o1, -self.o1.sum(0, keepdim=True)])

    def o2_vec(self):
        if not self.f2_tie:
            return self.o2
        return torch.cat([self.o2, -self.o2])

    def forward(self, tok):
        """tok: (..., P, 2) long digit pairs -> logits (..., P, 10)."""
        code = self.code_table()
        h = code[tok[..., 0]] + code[tok[..., 1]]

        # --- FFN 1 -------------------------------------------------------
        z = (h @ self.W1 if self.f1_in == "full" else h[..., :1]) + self.b1
        u = torch.relu(z)
        if self.f1_out == "full":
            h = h + u @ self.O1
        else:
            h = h + (u @ self.o1_vec()).unsqueeze(-1) * self.ekv

        # --- attention ---------------------------------------------------
        if self.kv == "full":
            k = h @ self.Wk
            v = h @ self.Wv
        else:
            k = h[..., self.KV:self.KV + 1]
            v = k
        q = (h @ self.Wq + self.bq) if self.q == "proj" else self.bq
        lam = self.lam if self.lam_mode == "learn" else self.lam_c
        att = torch.softmax(q * k.transpose(-2, -1) + lam * self.dist
                            + self.mask_bias, dim=-1)
        o = att @ v
        if self.wo == "full":
            h = h + o @ self.Wo
        elif self.wo == "axis":
            h = h + (o * self.wo_s) * self.e0
        else:
            h = h + o * self.e0

        # --- FFN 2 -------------------------------------------------------
        z2 = (h @ self.W2 if self.f2_in == "full" else h[..., :1]) + self.b2
        u2 = torch.relu(z2)
        if self.f2_out == "full":
            h = h + u2 @ self.O2
        else:
            h = h + (u2 @ self.o2_vec()).unsqueeze(-1) * self.e0

        # --- readout: nearest tied code prototype -------------------------
        return self.temp * (2.0 * (h @ code.t()) - (code * code).sum(-1))


def _digits(x, n):
    out = []
    for _ in range(n):
        out.append(x % 10)
        x //= 10
    return out


def add(model, a, b):
    """Exact sum of two operands, read off one forward pass of ``model``."""
    n = model.P - 2
    tok = torch.zeros(1, model.P, 2, dtype=torch.long)
    ad = _digits(int(a), n)
    bd = _digits(int(b), n)
    for i in range(n):
        tok[0, i + 1, 0] = ad[i]
        tok[0, i + 1, 1] = bd[i]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(tok)[0].argmax(-1).tolist()
    if was_training:
        model.train()
    total = 0
    place = 1
    for i in range(1, model.P):
        total += int(pred[i]) * place
        place *= 10
    return total


_CFG = {"C": 1, "D": 2, "KV": 1, "P": 10, "U1": 6, "U2": 4, "code_fix": 0, "f1_in": "full", "f1_out": "full", "f1_tie": False, "f2_in": "full", "f2_out": "full", "f2_tie": False, "kv": "full", "lam": "learn", "lam_fix": -1.0, "q": "proj", "temp": 4.0, "wo": "full"}

_WEIGHTS = {
    'code_free0': [-4.620095252990723, -3.5443129539489746, -2.4931392669677734, -1.4406852722167969, -0.3867775499820709, 0.6662102341651917, 1.720104455947876, 2.772714614868164, 3.8258233070373535, 4.8998847007751465],
    'W1': [[-0.6508687734603882, 2.178079605102539, 3.122077226638794, 0.4409233629703522, -0.46833136677742004, 2.329648494720459], [1.542449712753296, -2.450547695159912, -2.189134359359741, -0.7113538980484009, 1.3705732822418213, -2.093968152999878]],
    'b1': [0.9781532883644104, -0.6001536250114441, 1.8665046691894531, 4.04218053817749, 0.17157913744449615, -2.3626725673675537],
    'O1': [[-1.0552209615707397, 0.021594233810901642], [5.119372367858887, 1.130948781967163], [2.5083630084991455, -1.2577623128890991], [0.371221125125885, 2.800811529159546], [-2.8421664237976074, 3.708810329437256], [-8.060945510864258, -0.004179447889328003]],
    'Wk': [[-0.12146773934364319], [-0.730102002620697]],
    'Wv': [[-0.1296074241399765], [0.15593478083610535]],
    'Wq': [[-0.420671671628952], [-2.250704526901245]],
    'bq': [-8.426518440246582],
    'Wo': [[-1.0926579236984253, -0.12099690735340118]],
    'lam': [-3.9677040576934814],
    'W2': [[-2.203646659851074, 0.586539089679718, -1.2148301601409912, 0.012910326942801476], [0.8174916505813599, 0.2056450992822647, 0.9099090099334717, 1.2842756509780884]],
    'b2': [2.4441163539886475, -3.2701642513275146, 6.688234806060791, -0.6917524337768555],
    'O2': [[-0.8144339919090271, -6.028448104858398], [-1.0613439083099365, -1.1590020656585693], [2.115034818649292, 5.355284690856934], [-1.4954955577850342, 2.399385690689087]],
}


def build_model():
    """Return (model, metadata) with the trained weights loaded."""
    model = DigitPairAdder(_CFG)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in _WEIGHTS.items():
            t = torch.tensor(value, dtype=torch.float32)
            params[name].copy_(t.reshape(params[name].shape))
    model.eval()
    metadata = {"architecture": "1 Macaron transformer block: FFN -> 1-head strictly-causal self-attention with a relative distance bias -> FFN", "d_model": 2, "n_heads": 1, "n_layers": 1, "n_parameters": 70, "name": "digit-pair-adder", "sequence": "10 LSB-first digit-pair tokens, tied code table readout", "task": "exact addition of two 8-digit integers", "trained": True}
    return model, metadata
