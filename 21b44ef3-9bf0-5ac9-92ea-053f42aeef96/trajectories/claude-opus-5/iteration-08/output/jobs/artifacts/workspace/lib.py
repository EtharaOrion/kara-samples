"""Shared config / data / ensemble-forward helpers (training side only).

Nothing here is shipped: submission.py carries model_src.py's class text plus
the trained weights as float literals, and imports only torch.

`fwd` is an E-member-batched mirror of DigitPairAdder.forward; check.py asserts
the two agree numerically.
"""
import torch

# ---------------------------------------------------------------- config ----

DEFAULTS = dict(
    C=2,             # code dimension == residual width
    U=4,             # units in bank 1 (pre-attention: builds the attention key/value axis)
    U2=4,            # units in bank 2 (post-attention: the mod-10 fold); ignored when tie=True
    kv="share",      # "share": one axis z serves as key (x b_q) and value (x w_v).
                     # "one":   the same scalar serves as both, so the attention
                     #          logit a place broadcasts and the carry it hands on
                     #          are literally one projection, not two.
                     # "split": separate key and value read-outs of bank 1.
    tie=False,       # bank 2 reuses bank 1's weights, evaluated on the post-attention residual
    f1_in="free",    # "free": learned (C,U) input map.  "sign": the bank reads residual
                     # axis 0 through one fixed +-1, shared by all of its units --
                     # each unit's input magnitude has been absorbed into whatever
                     # reads the unit, which is exact and removes real parameters,
                     # but a per-unit sign would be learned information smuggled out
                     # of the parameter count, so it is not allowed to vary.
    f1_sign=None,    # +1.0 / -1.0
    f2_in="free",
    f2_sign=None,
    o_pin=False,     # pin o[0] = 1 (fixes the key-axis scale gauge)
    pin_row=-1,      # >=0: that code row is pinned to e_0 (fixes the residual scale/rotation gauge)
    lam_learn=True,  # learn the attention distance-bias slope (else fixed, ALiBi style)
    lam=-1.0,
    p_rows=None,     # optional subset of bank-2 units that feed the fold
    y_bias=False,    # training-only scalar added to the read-out.  x = code[a]+code[b]
                     # carries twice the code's offset while the read-out compares
                     # against one code entry, so the fold has to supply a constant;
                     # this gives it somewhere to live that is not a whole always-on
                     # ReLU unit.  The C=1 shift gauge then removes it (xform zerobias).
    logit_scale=False,  # training-only: exp(ls) on the read-out logits.  Dropping it
                        # rescales every logit by one positive factor, so the argmax
                        # decode -- and hence every answer -- is bit-identical.
)


def default_cfg(**kw):
    cfg = dict(DEFAULTS)
    for k, v in kw.items():
        if k not in DEFAULTS:
            raise KeyError(k)
        cfg[k] = v
    if cfg["tie"]:
        cfg["U2"] = cfg["U"]
    return cfg


def param_shapes(cfg):
    C, U, U2 = cfg["C"], cfg["U"], cfg["U2"]
    s = {"code_free": (10 - (1 if cfg["pin_row"] >= 0 else 0), C)}
    if cfg["f1_in"] == "free":
        s["w1"] = (C, U)
    s["b1"] = (U,)
    if cfg["kv"] in ("share", "one"):
        s["o_free"] = (U - 1 if cfg["o_pin"] else U,)
        if cfg["kv"] == "share":
            s["b_q"] = ()
        s["w_v"] = (C,)
    else:
        s["k_o"] = (U,)
        s["v_o"] = (U, C)
    if cfg["lam_learn"]:
        s["lam"] = ()
    if not cfg["tie"]:
        if cfg["f2_in"] == "free":
            s["w2"] = (C, U2)
        s["b2"] = (U2,)
    if cfg["y_bias"]:
        s["by"] = ()
    if cfg["logit_scale"]:
        s["ls"] = ()
    rows = cfg["p_rows"]
    s["p_out"] = (len(rows) if rows else (U if cfg["tie"] else U2), C)
    return s


def n_params(cfg):
    n = 0
    for sh in param_shapes(cfg).values():
        m = 1
        for d in sh:
            m *= d
        n += m
    return n


# ------------------------------------------------------- ensemble forward ----

def init_params(cfg, E, device, seed=0):
    """Random init.  `lam` starts near cfg["lam"]: that centre is a real prior on
    the solution, because attention can only skip a carry-transparent place if
    its key notch is deeper than the distance penalty |lam| it has to pay."""
    g = torch.Generator(device=device).manual_seed(seed)
    p = {}
    for name, sh in param_shapes(cfg).items():
        t = torch.randn((E,) + sh, generator=g, device=device)
        if name == "code_free":
            t = t * 2.0
        elif name in ("b1", "b2"):
            t = t * 2.0
        elif name == "lam":
            t = float(cfg["lam"]) * (0.5 + t.abs())
        elif name == "by":
            t = t * 0.1
        elif name == "b_q":
            t = t.abs() * 0.5 + 0.2
        else:
            t = t * 0.8
        p[name] = t.contiguous()
    return p


def full_code(p, cfg):
    code = p["code_free"]
    pin = cfg["pin_row"]
    if pin < 0:
        return code
    E, _, C = code.shape
    e = torch.zeros(E, 1, C, device=code.device, dtype=code.dtype)
    e[:, 0, 0] = 1.0
    return torch.cat([code[:, :pin], e, code[:, pin:]], 1)


def _bank(p, cfg, t, which):
    """ReLU bank applied to (E,B,P,C) activations."""
    key_w, key_b, key_s = ("w1", "b1", "f1_sign") if which == 1 else ("w2", "b2", "f2_sign")
    mode = cfg["f1_in"] if which == 1 else cfg["f2_in"]
    if mode == "free":
        pre = torch.einsum("ebpc,ecu->ebpu", t, p[key_w])
    else:
        s = torch.as_tensor(float(cfg[key_s]), device=t.device, dtype=t.dtype)
        pre = t[..., :1] * s
    return torch.relu(pre + p[key_b][:, None, None, :])


def fwd(p, cfg, da, db, want_attn=False, want_y=False):
    """Forward for E stacked members.  da/db are (B,P) int64 shared by members."""
    code = full_code(p, cfg)
    x = code[:, da] + code[:, db]                               # (E,B,P,C)
    h = _bank(p, cfg, x, 1)
    if cfg["kv"] in ("share", "one"):
        o = p["o_free"]
        if cfg["o_pin"]:
            one = torch.ones(o.shape[0], 1, device=o.device, dtype=o.dtype)
            o = torch.cat([one, o], 1)
        z = torch.einsum("ebpu,eu->ebp", h, o)
        q = p["b_q"] if cfg["kv"] == "share" else p["w_v"][:, 0]
        k = q.reshape(-1, 1, 1) * z
        v = z[..., None] * p["w_v"][:, None, None, :]
    else:
        k = torch.einsum("ebpu,eu->ebp", h, p["k_o"])
        v = torch.einsum("ebpu,euc->ebpc", h, p["v_o"])
    P = da.shape[-1]
    idx = torch.arange(P, device=da.device)
    dist = (idx[:, None] - idx[None, :]).to(x.dtype)
    lam = p["lam"] if cfg["lam_learn"] else torch.as_tensor(float(cfg["lam"]), device=x.device)
    att = k[:, :, None, :] + lam.reshape(-1, 1, 1, 1) * dist
    att = att - 1e9 * (dist <= 0)
    w = torch.softmax(att, -1) * (idx > 0).to(x.dtype)[:, None]
    r = x + torch.einsum("ebij,ebjc->ebic", w, v)
    g = _bank(p, cfg, r, 1 if cfg["tie"] else 2)
    if cfg["p_rows"]:
        g = g[..., cfg["p_rows"]]
    y = r + torch.einsum("ebpu,euc->ebpc", g, p["p_out"])
    if cfg["y_bias"]:
        y = y + p["by"].reshape(-1, 1, 1, 1)
    d2 = ((y[..., None, :] - code[:, None, None]) ** 2).sum(-1)
    logits = -(torch.exp(p["ls"]).reshape(-1, 1, 1, 1) * d2 if cfg["logit_scale"] else d2)
    if want_y:
        return logits, y
    return (logits, w, k) if want_attn else logits


# ------------------------------------------------------------------ data ----

MULT = 6364136223846793005


def heldout_mask(a, b):
    """1-in-16 hash bucket over the (a,b) digit pair: the held-out split."""
    n = a.shape[-1]
    pw = 10 ** torch.arange(n, device=a.device, dtype=torch.int64)
    key = (a * pw).sum(-1) * 100000000 + (b * pw).sum(-1)
    h = key * MULT
    return ((h >> 29) & 15) == 0


def sample_digits(B, n, device, gen, full_width=0.85, mix=(0.35, 0.40, 0.25)):
    """Digit pairs (B,n), LSB-first: a mix of uniform, carry-transparent-rich and
    maximal-carry-chain structure.  `mix` is (uniform, transparent-rich, chain).

    Transparent places (a_i+b_i == 9) are what makes the task non-local, so they
    are over-represented: a model that only ever looks one place back is right on
    uniform digits far more often than it deserves to be."""
    a = torch.randint(0, 10, (B, n), device=device, generator=gen)
    b = torch.randint(0, 10, (B, n), device=device, generator=gen)
    u = torch.rand(B, 1, device=device, generator=gen)
    r = torch.rand(B, n, device=device, generator=gen)
    c0, c1 = mix[0], mix[0] + mix[1]
    tr = (u >= c0) & (u < c1) & (r < 0.40)         # ~40% of places transparent
    ch = (u >= c1) & (r < 0.85)                    # long maximal chains
    b = torch.where(tr | ch, 9 - a, b)
    if n >= 2:
        fw = torch.rand(B, device=device, generator=gen) < full_width
        for t in (a, b):
            bump = torch.randint(1, 10, (B,), device=device, generator=gen)
            t[:, -1] = torch.where(fw & (t[:, -1] == 0), bump, t[:, -1])
    return a, b


def pad_places(a, b):
    B = a.shape[0]
    z = torch.zeros(B, 1, dtype=a.dtype, device=a.device)
    return torch.cat([z, a, z], 1), torch.cat([z, b, z], 1)


def targets(a, b):
    """Answer digit at every padded position (position 0's is unused)."""
    B, n = a.shape
    s = a + b
    out = torch.zeros(B, n + 2, dtype=torch.int64, device=a.device)
    carry = torch.zeros(B, dtype=torch.int64, device=a.device)
    for i in range(n):
        t = s[:, i] + carry
        out[:, i + 1] = t % 10
        carry = t // 10
    out[:, n + 1] = carry
    return out


def batch(B, n, device, gen, full_width=0.85, mix=(0.35, 0.40, 0.25)):
    a, b = sample_digits(B, n, device, gen, full_width, mix)
    y = targets(a, b)
    pa, pb = pad_places(a, b)
    return a, b, pa, pb, y
