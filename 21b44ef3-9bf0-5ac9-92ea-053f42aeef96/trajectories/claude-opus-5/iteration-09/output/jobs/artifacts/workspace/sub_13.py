import torch
import torch.nn as nn


def default_cfg(cfg=None):
    c = {
        "E": 1,             # ensemble axis (1 for a shipped model)
        "C": 2,             # code / residual width
        "U": 6,             # units in the pre-attention bank
        "U2": 6,            # units in the fold bank (ignored when tie)
        "tie": False,       # fold bank re-uses the pre-attention bank's W, b
        "act": "relu",      # relu | clamp | morph  (bank nonlinearity)
        "act_t": 0.0,       # morph position: 0 = relu, 1 = clamp
        "f_in": "free",     # free | sign   (input weights of the bank)
        "f2_in": "free",
        "signs": None,      # fixed +-1 input weights when f_in == "sign"
        "signs2": None,
        "key_lin": False,   # add a linear read of the residual to the key
        "val": "bank",      # bank | lin | id
        "kv_share": False,  # the value head reads the same projection as the key
        "heads": 1,         # 1 = carry-in only; 2 adds an inclusively-masked head
        "fold": True,       # keep the post-attention bank
        "q": "free",        # free | fix
        "q_val": 1.0,
        "lam": "free",      # free | fix   (relative-distance slope)
        "lam_val": -1.0,
        "wo": "free",       # free | fix   (attention write vector)
        "wo_val": 1.0,
        "ls": "free",       # free | fix   (read-out temperature)
        "ls_val": 1.0,
        "fold_out": "free",  # free | axis
        "pin_code": (),     # ((row, value), ...) rows held at a constant
        "b1": True,         # bank biases present
        "b2": True,
        "seed": 0,
    }
    if cfg:
        c.update(cfg)
    return c


class DigitAdder(nn.Module):
    """One transformer block over per-place digit-pair tokens.

    The sequence is LSB-first with a (0, 0) pad at each end: token i is
    ``code[a_i] + code[b_i]`` for a real place and position i predicts answer
    digit i - 1.  The block is FFN -> strictly-causal single-head attention
    -> FFN, and the read-out is the negative squared distance to the same code
    table used to embed the digits.

    Every parameter carries a leading ensemble axis E so that many independent
    members can be trained simultaneously; a shipped model has E == 1.
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = default_cfg(cfg)
        self.cfg = cfg
        E, C, U = cfg["E"], cfg["C"], cfg["U"]
        U2 = U if cfg["tie"] else cfg["U2"]
        self.U2 = U2
        gen = torch.Generator().manual_seed(int(cfg["seed"]))

        def rnd(*shape, s=1.0):
            return nn.Parameter(torch.randn(*shape, generator=gen) * s)

        # ---- code table: a few rows may be held at fixed constants ----------
        pins = tuple(tuple(p) for p in cfg["pin_code"])
        prow = [int(p[0]) for p in pins]
        free = [r for r in range(10) if r not in prow]
        perm = [0] * 10
        for i, r in enumerate(free):
            perm[r] = i
        for i, r in enumerate(prow):
            perm[r] = len(free) + i
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long))
        pv = torch.zeros(len(pins), C)
        for i, p in enumerate(pins):
            pv[i, 0] = float(p[1])
        self.register_buffer("pin_val", pv)
        self.code_free = rnd(E, len(free), C, s=0.8)

        # ---- pre-attention bank ---------------------------------------------
        if cfg["f_in"] == "free":
            self.W1 = rnd(E, U, C, s=1.0)
        else:
            sg = torch.tensor([float(s) for s in (cfg["signs"] or [1.0] * U)])
            self.register_buffer("W1f", sg.reshape(1, U, 1).expand(E, U, C).contiguous())
        if cfg["b1"]:
            self.b1 = rnd(E, U, s=0.5)
        self.key_w = rnd(E, U, s=0.7)
        if cfg["key_lin"]:
            self.kl = rnd(E, C, s=0.5)
        if not cfg["kv_share"]:
            if cfg["val"] == "bank":
                self.val_w = rnd(E, U, s=0.7)
            elif cfg["val"] == "lin":
                self.vl = rnd(E, C, s=0.5)

        # ---- attention scalars ----------------------------------------------
        if cfg["q"] == "free":
            self.q = nn.Parameter(torch.full((E,), 1.0))
        else:
            self.register_buffer("qf", torch.full((E,), float(cfg["q_val"])))
        if cfg["lam"] == "free":
            self.lam = nn.Parameter(torch.full((E,), -1.0))
        else:
            self.register_buffer("lamf", torch.full((E,), float(cfg["lam_val"])))
        if cfg["wo"] == "free":
            self.w_o = rnd(E, C, s=0.5)
        else:
            wo = torch.zeros(E, C)
            wo[:, 0] = float(cfg["wo_val"])
            self.register_buffer("w_of", wo)
        if cfg["heads"] == 2:
            self.w_o2 = rnd(E, C, s=0.5)

        # ---- fold bank -------------------------------------------------------
        if not cfg["fold"]:
            pass
        elif not cfg["tie"]:
            if cfg["f2_in"] == "free":
                self.W2 = rnd(E, U2, C, s=1.0)
            else:
                sg = torch.tensor([float(s) for s in (cfg["signs2"] or [1.0] * U2)])
                self.register_buffer("W2f", sg.reshape(1, U2, 1).expand(E, U2, C).contiguous())
            if cfg["b2"]:
                self.b2 = rnd(E, U2, s=0.5)
        if cfg["fold"]:
            if cfg["fold_out"] == "free":
                self.O2 = rnd(E, U2, C, s=0.5)
            else:
                self.O2a = rnd(E, U2, s=0.5)

        # ---- bank nonlinearity ----------------------------------------------
        if cfg["act"] == "morph":
            self.register_buffer("act_t", torch.tensor(float(cfg["act_t"])))

        # ---- read-out temperature -------------------------------------------
        if cfg["ls"] == "free":
            self.ls = nn.Parameter(torch.full((E,), float(cfg["ls_val"])))
        else:
            self.register_buffer("lsf", torch.full((E,), float(cfg["ls_val"])))

    # -- assembled tensors ----------------------------------------------------
    def code(self):
        E = self.code_free.shape[0]
        pv = self.pin_val.unsqueeze(0).expand(E, -1, -1)
        return torch.cat([self.code_free, pv], dim=1)[:, self.perm]

    def _w1(self):
        return self.W1 if self.cfg["f_in"] == "free" else self.W1f

    def _w2(self):
        if self.cfg["tie"]:
            return self._w1()
        return self.W2 if self.cfg["f2_in"] == "free" else self.W2f

    def _act(self, z):
        """relu, its bounded sibling clamp(z, 0, 1), or a path between them.

        clamp(z, 0, 1) == relu(z) - relu(z - 1), so relu(z) - t*relu(z - 1)
        walks continuously from one to the other as t goes 0 -> 1.  That is what
        lets a bank trained with relu be moved onto the bounded activation
        without losing the mechanism partway.
        """
        a = self.cfg["act"]
        if a == "relu":
            return torch.relu(z)
        if a == "clamp":
            return z.clamp(0.0, 1.0)
        return torch.relu(z) - self.act_t * torch.relu(z - 1.0)

    def _o2(self):
        if self.cfg["fold_out"] == "free":
            return self.O2
        out = torch.zeros(self.O2a.shape[0], self.U2, self.cfg["C"],
                          dtype=self.O2a.dtype, device=self.O2a.device)
        return out.index_copy(2, torch.zeros(1, dtype=torch.long, device=self.O2a.device),
                              self.O2a.unsqueeze(2))

    # -- forward ---------------------------------------------------------------
    def forward(self, a, b, return_parts=False):
        """a, b: (B, P) long digit sequences.  Returns logits (E, B, P, 10)."""
        cfg = self.cfg
        code = self.code()                                  # (E, 10, C)
        x = code[:, a] + code[:, b]                          # (E, B, P, C)
        P = x.shape[2]

        w1 = self._w1()
        pre = torch.einsum("ebpc,euc->ebpu", x, w1)
        if cfg["b1"]:
            pre = pre + self.b1[:, None, None, :]
        h = self._act(pre)

        k = torch.einsum("ebpu,eu->ebp", h, self.key_w)
        if cfg["key_lin"]:
            k = k + torch.einsum("ebpc,ec->ebp", x, self.kl)
        if cfg["kv_share"]:
            v = k
        elif cfg["val"] == "bank":
            v = torch.einsum("ebpu,eu->ebp", h, self.val_w)
        elif cfg["val"] == "lin":
            v = torch.einsum("ebpc,ec->ebp", x, self.vl)
        else:
            v = x[..., 0]

        idx = torch.arange(P, device=x.device)
        dist = (idx[:, None] - idx[None, :]).to(x.dtype)
        strict = idx[None, :] < idx[:, None]
        strict = strict | ((idx[:, None] == 0) & (idx[None, :] == 0))
        incl = idx[None, :] <= idx[:, None]

        q = self.q if cfg["q"] == "free" else self.qf
        lam = self.lam if cfg["lam"] == "free" else self.lamf
        base = (q[:, None, None, None] * k[:, :, None, :]
                + lam[:, None, None, None] * dist)
        w_o = self.w_o if cfg["wo"] == "free" else self.w_of
        masks = (strict,) if cfg["heads"] == 1 else (strict, incl)
        wos = (w_o,) if cfg["heads"] == 1 else (w_o, self.w_o2)
        y, attns = x, []
        for msk, wv in zip(masks, wos):
            at = torch.softmax(base + torch.where(msk, 0.0, -1e9).to(x.dtype), dim=-1)
            attns.append(at)
            o = torch.einsum("ebij,ebj->ebi", at, v)
            y = y + wv[:, None, None, :] * o[..., None]
        attn = attns[0]

        z = y
        if cfg["fold"]:
            pre2 = torch.einsum("ebpc,euc->ebpu", y, self._w2())
            if cfg["tie"]:
                if cfg["b1"]:
                    pre2 = pre2 + self.b1[:, None, None, :]
            elif cfg["b2"]:
                pre2 = pre2 + self.b2[:, None, None, :]
            z = y + torch.einsum("ebpu,euc->ebpc", torch.relu(pre2), self._o2())

        d2 = ((z[:, :, :, None, :] - code[:, None, None, :, :]) ** 2).sum(-1)
        ls = self.ls if cfg["ls"] == "free" else self.lsf
        logits = -ls[:, None, None, None] * d2
        if return_parts:
            return logits, {"x": x, "k": k, "v": v, "attn": attn,
                            "attns": attns, "y": y, "z": z, "h": h}
        return logits


# ---------------------------------------------------------------------------
# Weights below were produced by gradient descent in train.py (see the trainer
# alongside this file); nothing here is hand-set.
# ---------------------------------------------------------------------------

_CFG = {'E': 1, 'C': 1, 'U': 2, 'U2': 6, 'tie': False, 'act': 'clamp', 'act_t': 0.0, 'f_in': 'sign', 'f2_in': 'free', 'signs': [1.0, 1.0], 'signs2': None, 'key_lin': False, 'val': 'bank', 'kv_share': True, 'heads': 2, 'fold': False, 'q': 'fix', 'q_val': 1.0, 'lam': 'fix', 'lam_val': -4.0, 'wo': 'fix', 'wo_val': 1.0, 'ls': 'fix', 'ls_val': 1.0, 'fold_out': 'free', 'pin_code': [[0, 0.0], [1, 1.0]], 'b1': True, 'b2': True, 'seed': 0}

_W = {
    'code_free': [
        [
            [1.9386556148529053],
            [2.81889009475708],
            [3.7306923866271973],
            [4.6589131355285645],
            [5.573622703552246],
            [6.458117485046387],
            [7.3382415771484375],
            [8.333834648132324]
        ]
    ],
    'b1': [[-8.27690315246582, -7.518288612365723]],
    'key_w': [[64.23500061035156, -63.128387451171875]],
    'w_o2': [[-8.308281898498535]],
}


def _t(x):
    return torch.tensor(x, dtype=torch.float32)


def build_model():
    model = DigitAdder(_CFG)
    sd = model.state_dict()
    for k, v in _W.items():
        sd[k] = _t(v).reshape(sd[k].shape)
    model.load_state_dict(sd)
    model.eval()
    meta = {
        'name': 'digit-pair single-block transformer',
        'task': 'decimal addition, graded at 8 digits',
        'architecture': 'one block: clamp(.,0,1) projection -> self-attention with two heads (strictly-causal and inclusively-causal) sharing one set of keys and values, tied digit code used as both embedding and read-out',
        'residual_width': 1,
        'bank_units': 2,
        'heads': 2,
        'key_value_shared': True,
        'sequence': 'LSB-first, one token per decimal place, (0,0) pad at each end',
        'activation': 'clamp(.,0,1)',
        'learned_mechanism': 'the key is flat, notched at a place sum of 9 and flat again above it, so each head attends to the nearest earlier place that is not carry-transparent; the two masks read the carry in and the carry out of the current place',
        'trained_by': 'train.py (AdamW, ensemble seed lottery, warm-started shrink ladder)',
    }
    meta["parameters"] = sum(p.numel() for p in model.parameters())
    return model, meta


def _digits(x, n):
    d = []
    for _ in range(n):
        d.append(x % 10)
        x //= 10
    return d


@torch.no_grad()
def add(model, a, b):
    """Sum of a and b, read off one forward pass of the model."""
    n = max(len(str(int(a))), len(str(int(b))))
    da = [0] + _digits(int(a), n) + [0]
    db = [0] + _digits(int(b), n) + [0]
    ta = torch.tensor([da], dtype=torch.long)
    tb = torch.tensor([db], dtype=torch.long)
    pred = model(ta, tb)[0, 0, 1:, :].argmax(-1).tolist()
    out = 0
    for i, d in enumerate(pred):
        out += int(d) * 10 ** i
    return out
