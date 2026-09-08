"""Architecture for the 8-digit addition transformer.

This file is the single source of truth for the model: `build_submission.py`
copies the class below verbatim into the graded `submission.py`, and the
trainer instantiates the very same class.  Nothing here generates data or
labels -- it is model code only.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------- config ----


def default_cfg(**kw):
    """Config for the block.  Every entry either sizes a tensor or removes a
    redundant degree of freedom (a gauge) from the parameterisation."""
    cfg = dict(
        P=10,             # sequence length
        D=2,              # residual width
        C=2,              # width of the answer subspace (holds the prototypes)
        U1=4,             # width of the pre-attention FFN
        U2=0,             # width of the post-attention FFN (0 = absent)
        H=2,              # attention heads
        dh=1,             # head width
        masks=("strict", "causal"),   # per-head causal mask
        f1_id_in=False,   # pre-attention FFN reads axis 0 with unit weight
        f1_out=None,      # if set, that FFN writes only to this axis
        f2_id_in=False,
        f2_out=None,
        q_proj=True,      # queries projected from the residual (else a learned constant)
        share_q=False,    # all heads share one query
        kv_id=False,      # keys/values are read off axis D-1 with unit weight
        wo_out=None,      # if set, heads write only to this axis
        wo_fix=0,         # heads [0, wo_fix) have their write weight pinned to 1
        lam=None,         # None: learned distance slopes, else a fixed tuple
        code_fix=0,       # pinned prototypes: code[0] = 0, code[1] = e0
        logit_scale=True,
    )
    cfg.update(kw)
    return cfg


# ----------------------------------------------------------------- model ----


class DigitPairAdder(nn.Module):
    """One Macaron-style transformer block over digit-pair tokens.

    Sequence layout (least significant place first, P = 10 positions):

        position 0      a padding place (0, 0); it anchors "no carry into place 0"
        positions 1..8  place p-1 of the two operands, embedded as code[a] + code[b]
        position 9      a padding place (0, 0) that receives the final carry out

    Position p predicts the answer digit of place p-1, so a single forward pass
    emits all nine answer digits.

    Block:  x -> x + FFN1(x) -> x + Attn(x) -> [x + FFN2(x)] -> readout

    Attention is multi-head with a fixed ALiBi-style distance bias; each head has
    its own causal mask ('causal' can see the current place, 'strict' only sees
    earlier places).  Scores are content based: the keys are what the FFN made of
    each place's digit pair, so the pattern moves with the input.  The readout is
    the negative squared distance from the answer subspace of the residual stream
    to each of the ten prototypes, which are the embedding rows themselves (tied).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = c = dict(cfg)
        P, D, C, U1, U2, H, dh = (c["P"], c["D"], c["C"], c["U1"], c["U2"],
                                  c["H"], c["dh"])

        # -- digit prototypes (shared by the embedding and the readout) -------
        nfix = c["code_fix"]
        fixed = torch.zeros(nfix, C)
        if nfix >= 2:
            fixed[1, 0] = 1.0                      # sets the scale of the answer axis
        self.register_buffer("code_fixed", fixed)
        self.code_free = nn.Parameter(torch.randn(10 - nfix, C) * 0.5)

        # -- pre-attention FFN -------------------------------------------------
        if c["f1_id_in"]:
            w = torch.zeros(U1, D)
            w[:, 0] = 1.0
            self.register_buffer("f1_in", w)
        else:
            self.f1_in = nn.Parameter(torch.randn(U1, D) / D ** 0.5)
        self.f1_b = nn.Parameter(torch.zeros(U1))
        if c["f1_out"] is None:
            self.f1_out = nn.Parameter(torch.randn(U1, D) / U1 ** 0.5)
        else:
            self.f1_out = nn.Parameter(torch.randn(U1) / U1 ** 0.5)

        # -- attention ---------------------------------------------------------
        if c["q_proj"]:
            self.q_w = nn.Parameter(torch.randn(H, dh, D) / D ** 0.5)
        self.q_b = nn.Parameter(torch.zeros(1 if c["share_q"] else H, dh))
        if c["kv_id"]:
            e = torch.zeros(H, dh, D)
            e[:, :, D - 1] = 1.0
            self.register_buffer("k_w", e)
            self.register_buffer("v_w", e.clone())
        else:
            self.k_w = nn.Parameter(torch.randn(H, dh, D) / D ** 0.5)
            self.v_w = nn.Parameter(torch.randn(H, dh, D) / D ** 0.5)
        if c["wo_out"] is None:
            self.o_w = nn.Parameter(torch.randn(H, dh, D) / (H * dh) ** 0.5)
        else:
            self.register_buffer("o_fixed", torch.ones(c["wo_fix"], dh))
            self.o_w = nn.Parameter(torch.randn(H - c["wo_fix"], dh) * 0.5)
        if c["lam"] is None:
            self.lam = nn.Parameter(torch.full((H,), -1.0))
        else:
            self.register_buffer("lam", torch.tensor(list(c["lam"]), dtype=torch.float32))

        # -- post-attention FFN (optional) ------------------------------------
        if U2:
            if c["f2_id_in"]:
                w = torch.zeros(U2, D)
                w[:, 0] = 1.0
                self.register_buffer("f2_in", w)
            else:
                self.f2_in = nn.Parameter(torch.randn(U2, D) / D ** 0.5)
            self.f2_b = nn.Parameter(torch.zeros(U2))
            if c["f2_out"] is None:
                self.f2_out = nn.Parameter(torch.randn(U2, D) / U2 ** 0.5)
            else:
                self.f2_out = nn.Parameter(torch.randn(U2) / U2 ** 0.5)

        if c["logit_scale"]:
            self.ls = nn.Parameter(torch.zeros(()))

        # -- fixed geometry: distance bias and per-head causal masks ----------
        idx = torch.arange(P)
        dist = (idx[:, None] - idx[None, :]).clamp(min=0).float()
        self.register_buffer("dist", dist)
        m = torch.zeros(H, P, P)
        for h, kind in enumerate(c["masks"]):
            ok = idx[None, :] <= idx[:, None] if kind == "causal" else idx[None, :] < idx[:, None]
            if kind != "causal":
                ok = ok.clone()
                ok[0, 0] = True          # position 0 predicts nothing; keeps softmax finite
            m[h] = torch.where(ok, 0.0, float("-inf"))
        self.register_buffer("mask", m)

    # ------------------------------------------------------------ helpers --
    def code(self):
        return torch.cat([self.code_fixed, self.code_free], 0)

    def _out_mat(self, w, axis, D):
        """Scatter per-unit scalars into column `axis` of a (U, D) matrix."""
        e = torch.zeros(D, dtype=w.dtype, device=w.device)
        e = torch.cat([e[:axis], e.new_ones(1), e[axis + 1:]])
        return w[:, None] * e[None, :]

    # ------------------------------------------------------------ forward --
    def forward(self, da, db):
        """da, db: (B, P) integer digit streams.  Returns (B, P, 10) logits."""
        c = self.cfg
        D, C, H, dh = c["D"], c["C"], c["H"], c["dh"]
        code = self.code()

        x = code[da] + code[db]                                   # (B, P, C)
        if D > C:
            x = torch.cat([x, x.new_zeros(x.shape[:-1] + (D - C,))], -1)

        # pre-attention FFN
        h = torch.relu(torch.einsum("bpd,ud->bpu", x, self.f1_in) + self.f1_b)
        w1 = self.f1_out if c["f1_out"] is None else self._out_mat(self.f1_out, c["f1_out"], D)
        x = x + torch.einsum("bpu,ud->bpd", h, w1)

        # attention
        q = self.q_b[None, :, None, :].expand(x.shape[0], H, x.shape[1], dh)
        if c["q_proj"]:
            q = q + torch.einsum("bpd,hed->bhpe", x, self.q_w)
        k = torch.einsum("bpd,hed->bhpe", x, self.k_w)
        v = torch.einsum("bpd,hed->bhpe", x, self.v_w)
        s = torch.einsum("bhpe,bhqe->bhpq", q, k)
        s = s + self.lam[None, :, None, None] * self.dist[None, None] + self.mask[None]
        a = torch.softmax(s, -1)
        o = torch.einsum("bhpq,bhqe->bhpe", a, v)
        if c["wo_out"] is None:
            wo = self.o_w
        else:
            wo = self._out_mat(torch.cat([self.o_fixed, self.o_w], 0)[:, 0], c["wo_out"], D)
            wo = wo[:, None, :]
        x = x + torch.einsum("bhpe,hed->bpd", o, wo)

        # post-attention FFN
        if c["U2"]:
            h2 = torch.relu(torch.einsum("bpd,ud->bpu", x, self.f2_in) + self.f2_b)
            w2 = self.f2_out if c["f2_out"] is None else self._out_mat(self.f2_out, c["f2_out"], D)
            x = x + torch.einsum("bpu,ud->bpd", h2, w2)

        # nearest-prototype readout, tied to the embedding
        d2 = ((x[..., None, :C] - code[None, None]) ** 2).sum(-1)
        logits = -d2
        if c["logit_scale"]:
            logits = logits * torch.exp(self.ls)
        return logits


# --------------------------------------------------------------- decoding ---


def digits_of(n, k=8):
    return [(n // 10 ** i) % 10 for i in range(k)]


@torch.no_grad()
def add_batch(model, aa, bb):
    """Exact sums for lists of operands, from one forward pass each."""
    da = torch.tensor([[0] + digits_of(a) + [0] for a in aa])
    db = torch.tensor([[0] + digits_of(b) + [0] for b in bb])
    p = model(da, db).argmax(-1)[:, 1:]
    pw = torch.tensor([10 ** i for i in range(p.shape[1])])
    return (p * pw).sum(-1).tolist()
