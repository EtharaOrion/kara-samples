"""Minimal transformer that adds two 8-digit numbers.

One Macaron-style block (FFN -> multi-head self-attention -> readout) over ten
digit-pair tokens.  101 trained parameters; the weights below are the ones
produced by training (see train.py / cascade.py, which are not imported here).

Mechanism, for the reader: the embedding row of a digit doubles as the answer
prototype for that digit, so a place's token code[a] + code[b] already sits at
the prototype of (a + b) when there is no carry.  The FFN turns each place into
a key that is extreme exactly when a + b == 9 -- the places that pass a carry
along -- so the attention softmax skips those and lands on the nearest place
that actually decides the carry.  One head looks strictly before the current
place (the carry coming in) and one includes it (the carry going out, which is
what tells the place to wrap past ten); their two writes move the residual onto
the prototype of the answer digit.
"""

import torch
import torch.nn as nn

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


_CFG = {
    'C': 3,
    'D': 3,
    'H': 2,
    'P': 10,
    'U1': 6,
    'U2': 0,
    'code_fix': 0,
    'dh': 1,
    'f1_id_in': False,
    'f1_out': None,
    'f2_id_in': False,
    'f2_out': None,
    'kv_id': False,
    'lam': None,
    'logit_scale': True,
    'masks': ('strict', 'causal'),
    'q_proj': True,
    'share_q': False,
    'wo_fix': 0,
    'wo_out': None,
}

_W = {
    'code_free': [[-0.9406614899635315, -1.84796142578125, -1.5859371423721313], [-0.8081663846969604, -1.154720664024353, -1.436298131942749], [-0.6186812520027161, -0.6335875988006592, -1.1784425973892212], [-0.39815205335617065, -0.19862724840641022, -0.8676541447639465], [-0.15871529281139374, 0.17618073523044586, -0.5186077952384949], [0.10179480910301208, 0.49676963686943054, -0.13457469642162323], [0.3792782127857208, 0.7627263069152832, 0.28229501843452454], [0.680415689945221, 0.9630096554756165, 0.740786612033844], [1.0171070098876953, 1.0619654655456543, 1.2619497776031494], [1.4914319515228271, 0.7694801688194275, 2.024789810180664]],
    'f1_b': [1.1470915079116821, 0.9612138271331787, -1.4583734273910522, 0.548749566078186, -0.1996302455663681, 0.032229289412498474],
    'f1_in': [[-1.1043627262115479, 0.6237574219703674, 0.29130125045776367], [-0.8878856897354126, -1.1577191352844238, -1.3650779724121094], [-3.007819175720215, -1.7226930856704712, -1.0857001543045044], [1.0184872150421143, -0.2700432240962982, -0.1636570692062378], [0.5717629194259644, -0.06703917682170868, -0.45313024520874023], [-1.9066965579986572, -1.5352500677108765, -1.3963375091552734]],
    'f1_out': [[-0.22578434646129608, -1.4373281002044678, 0.2555566132068634], [1.4098765850067139, -0.6271640062332153, 0.5882881283760071], [-0.5637413263320923, -1.8167871236801147, 0.32184484601020813], [-1.0240565538406372, -0.3657602071762085, -0.7649258971214294], [-0.31036776304244995, -0.1333208829164505, 0.6280896067619324], [-0.38706961274147034, 2.557485818862915, -0.8165547251701355]],
    'k_w': [[[2.4994454383850098, -2.5914859771728516, 0.9480904340744019]], [[1.8597466945648193, 0.6481115221977234, -1.341921329498291]]],
    'lam': [-1.0917675495147705, -0.9212258458137512],
    'ls': 4.674946308135986,
    'o_w': [[[-0.7132334113121033, -0.5862809419631958, -0.5769820213317871]], [[-0.49830296635627747, -0.09028957039117813, 1.878588318824768]]],
    'q_b': [[-2.8514761924743652], [-1.503078579902649]],
    'q_w': [[[1.1226980686187744, -0.8202109336853027, 0.20765990018844604]], [[0.9012496471405029, 0.1549605429172516, -1.2919336557388306]]],
    'v_w': [[[1.1145614385604858, -0.9397825598716736, 0.2425546646118164]], [[1.5997205972671509, -1.1022382974624634, 0.1397538185119629]]],
}


def build_model():
    """Return (model, metadata).  Weights are the trained ones, inlined above."""
    model = DigitPairAdder(_CFG)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in _W.items():
            p = own[name]
            p.copy_(torch.tensor(value, dtype=torch.float32).reshape(p.shape))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "digit-pair adder",
        "task": "exact addition of two 8-digit integers",
        "n_params": n,
        "num_parameters": n,
        "parameters": n,
        "architecture": "1-block transformer: FFN -> 2-head self-attention, tied embedding/readout",
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 3,
        "vocab_size": 10,
        "seq_len": 10,
        "digits": 8,
        "trained": True,
    }
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two 8-digit operands, read off one forward pass."""
    da = torch.tensor([[0] + [(int(a) // 10 ** i) % 10 for i in range(8)] + [0]])
    db = torch.tensor([[0] + [(int(b) // 10 ** i) % 10 for i in range(8)] + [0]])
    digits = model(da, db).argmax(-1)[0, 1:].tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))
