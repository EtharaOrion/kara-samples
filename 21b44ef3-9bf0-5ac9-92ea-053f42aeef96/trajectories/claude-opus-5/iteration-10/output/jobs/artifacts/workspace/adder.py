import torch

# ---------------------------------------------------------------------------
# Weights and architecture description.  The build script rewrites the two
# assignments below with the values of a trained checkpoint.
# ---------------------------------------------------------------------------
_CFG = None
_W = None


class Adder(torch.nn.Module):
    """A one-block transformer that adds two multi-digit numbers.

    Tokens are digit places, least significant first, with a ``(0, 0)`` pad at
    each end.  Token ``i`` embeds as ``code[a_i] + code[b_i]`` from a single
    learned table that doubles as the read-out prototype set.  A small
    element-wise bank turns that embedding into one shared key/value stream,
    which feeds two self-attention heads over the same stream: one strictly
    causal, one inclusively causal.  Position ``i`` predicts answer digit
    ``i``, so the whole sum comes out of a single forward pass.
    """

    def __init__(self, cfg):
        super().__init__()
        cfg = dict(cfg)
        self.cfg = cfg
        C, U = cfg["C"], cfg["U"]
        prm = torch.nn.Parameter

        # digit code, also used as read-out prototypes.  The first `code_fix`
        # rows are held at zero (fixes the code-translation gauge).
        nfix = cfg["code_fix"]
        self.register_buffer("code_fix", torch.zeros(nfix, C))
        self.code = prm(torch.zeros(10 - nfix, C))

        # element-wise bank: u = clamp(bw @ x + bb, 0, 1)
        if cfg["bw"] is None:
            self.bw = prm(torch.zeros(U, C))
        else:
            self.register_buffer("bw", torch.tensor(cfg["bw"], dtype=torch.float32).view(U, C))
        self.bb = prm(torch.zeros(U))

        # key and value read-outs from the bank
        if cfg["kw"] is None:
            self.kw = prm(torch.zeros(U))
        else:
            self.register_buffer("kw", torch.tensor(cfg["kw"], dtype=torch.float32).view(U))
        if cfg["vw"] is None:
            self.vw = prm(torch.zeros(U))
        else:
            self.register_buffer("vw", torch.tensor(cfg["vw"], dtype=torch.float32).view(U))

        # where each head writes back into the residual stream
        if cfg["e1"] is None:
            self.e1 = prm(torch.zeros(C))
        else:
            self.register_buffer("e1", torch.tensor(cfg["e1"], dtype=torch.float32).view(C))
        if cfg["e2"] is None:
            self.e2 = prm(torch.zeros(C))
        else:
            self.register_buffer("e2", torch.tensor(cfg["e2"], dtype=torch.float32).view(C))

        # residual constant (dropped once the code translation gauge is fixed)
        if cfg["rb"]:
            self.rb = prm(torch.zeros(C))

        # relative-position slope of the attention scores
        if cfg["lam"] is None:
            self.lam = prm(torch.tensor(-2.0))
        else:
            self.register_buffer("lam", torch.tensor(float(cfg["lam"])))

        # read-out temperature (argmax invariant, so it is pinned at the end)
        if cfg["ls"] is None:
            self.ls = prm(torch.tensor(1.0))
        else:
            self.register_buffer("ls", torch.tensor(float(cfg["ls"])))

    def forward(self, ad, bd):
        """ad, bd: integer digit tensors (B, n), least significant digit first.

        Returns logits (B, n + 2, 10); position i holds answer digit i.
        """
        B, n = ad.shape
        pad = torch.zeros(B, 1, dtype=ad.dtype, device=ad.device)
        ai = torch.cat([pad, ad, pad], 1)
        bi = torch.cat([pad, bd, pad], 1)

        code = torch.cat([self.code_fix, self.code], 0)          # (10, C)
        x = code[ai] + code[bi]                                  # (B, P, C)

        u = torch.clamp(x @ self.bw.t() + self.bb, 0.0, 1.0)     # (B, P, U)
        key = u @ self.kw                                        # (B, P)
        val = u @ self.vw                                        # (B, P)

        p = n + 2
        idx = torch.arange(p, device=ad.device)
        dist = idx[:, None] - idx[None, :]                       # i - j
        score = key[:, None, :] + self.lam * dist                # (B, P, P)
        blk = torch.finfo(score.dtype).min
        eye0 = (idx[:, None] == 0) & (idx[None, :] == 0)
        a_in = torch.softmax(torch.where((dist >= 1) | eye0, score, blk), -1)
        b_in = torch.softmax(torch.where(dist >= 0, score, blk), -1)
        oa = (a_in @ val[..., None]).squeeze(-1)                 # (B, P)
        ob = (b_in @ val[..., None]).squeeze(-1)

        r = x + oa[..., None] * self.e1 + ob[..., None] * self.e2
        if self.cfg["rb"]:
            r = r + self.rb
        return -self.ls * ((r[:, :, None, :] - code) ** 2).sum(-1)


def build_model():
    """Return the trained model and a description of it."""
    model = Adder(_CFG)
    with torch.no_grad():
        for name, value in _W.items():
            p = dict(model.named_parameters())[name]
            p.copy_(torch.tensor(value, dtype=torch.float32).view_as(p))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "digit-pair adder",
        "n_params": n_params,
        "arch": "1 block: element-wise bank -> 2-head self-attention (shared "
                "key/value, strictly and inclusively causal) -> tied read-out",
        "digits": 8,
        "code_dim": _CFG["C"],
        "bank_units": _CFG["U"],
    }
    return model, meta


def add(model, a, b):
    """Exact sum of two non-negative integers, from one forward pass."""
    n = max(8, len(str(int(a))), len(str(int(b))))
    dev = next(model.parameters()).device
    da = [(int(a) // 10 ** i) % 10 for i in range(n)]
    db = [(int(b) // 10 ** i) % 10 for i in range(n)]
    ad = torch.tensor([da], dtype=torch.long, device=dev)
    bd = torch.tensor([db], dtype=torch.long, device=dev)
    with torch.no_grad():
        pred = model(ad, bd)[0].argmax(-1).tolist()
    out = 0
    for i in range(n + 1):                      # answer digit i sits at position i + 1
        out += int(pred[i + 1]) * 10 ** i
    return out
