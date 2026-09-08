"""Architecture for the digit-pair addition transformer.

The text between the BEGIN/END SUBMISSION SOURCE markers is copied verbatim
into the graded ``submission.py`` by ``build_submission.py``.  Nothing in this
file trains anything; see ``train_ens.py`` for that.
"""

# ---- BEGIN SUBMISSION SOURCE ----
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
# ---- END SUBMISSION SOURCE ----


# ======================= training-side helpers ==========================
def init_model(cfg, seed):
    """Random init used for every ensemble member (training only)."""
    g = torch.Generator().manual_seed(int(seed))
    m = DigitPairAdder(cfg)
    D = m.D

    def rnd(shape, s):
        return torch.randn(*shape, generator=g) * s

    with torch.no_grad():
        m.code_free0.copy_(rnd(m.code_free0.shape, 1.0))
        if m.C > 1:
            m.code_rest.copy_(rnd(m.code_rest.shape, 1.0))
        if m.f1_in == "full":
            m.W1.copy_(rnd(m.W1.shape, D ** -0.5))
        m.b1.copy_(rnd(m.b1.shape, 1.0))
        if m.f1_out == "full":
            m.O1.copy_(rnd(m.O1.shape, m.U1 ** -0.5))
        else:
            m.o1.copy_(rnd(m.o1.shape, 1.0))
        if m.kv == "full":
            m.Wk.copy_(rnd(m.Wk.shape, D ** -0.5))
            m.Wv.copy_(rnd(m.Wv.shape, D ** -0.5))
        if m.q == "proj":
            m.Wq.copy_(rnd(m.Wq.shape, D ** -0.5))
        m.bq.copy_(rnd(m.bq.shape, 1.0))
        if m.wo == "full":
            m.Wo.copy_(rnd(m.Wo.shape, 1.0))
        elif m.wo == "axis":
            m.wo_s.copy_(rnd(m.wo_s.shape, 1.0))
        if m.lam_mode == "learn":
            m.lam.copy_(-torch.rand(1, generator=g) * 2.0)
        if m.f2_in == "full":
            m.W2.copy_(rnd(m.W2.shape, D ** -0.5))
        m.b2.copy_(rnd(m.b2.shape, 1.0))
        if m.f2_out == "full":
            m.O2.copy_(rnd(m.O2.shape, m.U2 ** -0.5))
        else:
            m.o2.copy_(rnd(m.o2.shape, 1.0))
    return m


def n_params(model):
    return sum(p.numel() for p in model.parameters())
