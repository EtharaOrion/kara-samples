"""Single source of truth for the architecture.

Functional forward with a leading ensemble axis E so that E independent members
train simultaneously on one GPU.  The shipped nn.Module computes exactly the same
function for E == 1.

Token layout (LSB first), n operand places -> P = n + 2 positions:
    pos 0        : (0,0) pad   -- the anchor the carry chain terminates on
    pos 1..n     : (a_i, b_i) for operand place i-1
    pos n+1      : (0,0) pad   -- slot for the final carry-out digit
Position p predicts answer digit p-1; position 0 predicts 0.
"""
import torch


# ----------------------------------------------------------------------------- config
def default_cfg(**kw):
    c = dict(
        C=1,                # residual / code dimension
        U=2,                # bank units
        lam_init=-1.0,
        q_init=1.0,
        ls_init=2.0,
        code0_fixed=True,   # code[0] pinned to the origin (a coordinate choice)
        use_vb=True,        # value bias   (parent only; removed on the way down)
        use_rb=True,        # residual bias (parent only; removed on the way down)
    )
    c.update(kw)
    return c


def init_params(cfg, E, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    C, U = cfg["C"], cfg["U"]

    def n(*shape, s=1.0):
        return torch.randn(*shape, generator=g, device=device) * s

    nfree = 10 - (1 if cfg["code0_fixed"] else 0)
    p = dict(
        code=n(E, nfree, C, s=1.0),
        Bw=n(E, C, U, s=1.0),
        bb=n(E, U, s=1.0),
        kw=n(E, U, s=1.0),
        vw=n(E, U, s=1.0),
        q=torch.full((E,), cfg["q_init"], device=device),
        lam=torch.full((E,), cfg["lam_init"], device=device) + n(E, s=0.2),
        w1=n(E, C, s=1.0),
        w2=n(E, C, s=1.0),
        ls=torch.full((E,), cfg["ls_init"], device=device),
    )
    if cfg.get("use_vb", False):
        p["vb"] = n(E, s=0.3)
    if cfg.get("use_rb", False):
        p["rb"] = n(E, C, s=0.3)
    return {k: v.contiguous() for k, v in p.items()}


def full_code(p, cfg):
    """(E,10,C) code table, with row 0 pinned to the origin when code0_fixed."""
    if not cfg["code0_fixed"]:
        return p["code"]
    E, _, C = p["code"].shape
    z = p["code"].new_zeros(E, 1, C)
    return torch.cat([z, p["code"]], dim=1)


# ----------------------------------------------------------------------------- masks
_MASK_CACHE = {}


def masks(P, device, dtype):
    key = (P, device, dtype)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    i = torch.arange(P, device=device)
    rel = (i[:, None] - i[None, :]).to(dtype)          # i - j
    neg = torch.finfo(dtype).min / 4
    strict = torch.where(i[None, :] < i[:, None],
                         torch.zeros((), dtype=dtype, device=device),
                         torch.full((), neg, dtype=dtype, device=device))
    strict = strict.clone()
    strict[0, 0] = 0.0        # pos 0 has no predecessor; let it look at itself (output unused)
    incl = torch.where(i[None, :] <= i[:, None],
                       torch.zeros((), dtype=dtype, device=device),
                       torch.full((), neg, dtype=dtype, device=device))
    out = (rel, strict, incl)
    _MASK_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------- forward
def forward(p, da, db, cfg, want=None, leak=0.0, leak_sat=False):
    """p: dict of (E,...) tensors.  da, db: (B,P) int64 digits, LSB-first.

    Returns digit logits (E,B,P,10).  If `want` is a list, intermediates are stashed in it.

    `leak` is a training-only crutch: once the clamp bank saturates, the knee
    positions bb stop receiving gradient and can never be corrected.  A small leak
    outside [0,1] keeps that gradient alive.  It is annealed to exactly 0 well
    before the end of training, so the final function is the plain clamp.
    """
    code = full_code(p, cfg)                            # (E,10,C)
    x = code[:, da] + code[:, db]                       # (E,B,P,C)
    z = torch.einsum("ebpc,ecu->ebpu", x, p["Bw"]) + p["bb"][:, None, None, :]
    u = z.clamp(0.0, 1.0)                               # (E,B,P,U)
    if leak:
        e = z - u                                       # how far outside [0,1]
        if leak_sat:
            # A plain linear leak is fine while the clamp slope is O(1), but once the
            # slope is steep (the shipped bank uses 8, and x spans 0..18) e reaches
            # +-80 and even a 2% leak moves u by more than 1 -- the bank stops being a
            # bank and training optimises a different function than the one evaluated.
            # Dividing by the slope puts e back in code-step units and tanh bounds the
            # distortion by `leak`, while the gradient stays alive exactly where it is
            # needed: within a code step or so of the knee.
            sp = p["Bw"].abs().sum(1).clamp(min=1e-6)   # (E,U) code units -> z units
            u = u + leak * torch.tanh(e / sp[:, None, None, :])
        else:
            u = u + leak * e
    k = torch.einsum("ebpu,eu->ebp", u, p["kw"])
    v = torch.einsum("ebpu,eu->ebp", u, p["vw"])
    if "vb" in p:
        v = v + p["vb"][:, None, None]

    P = da.shape[1]
    rel, m_s, m_i = masks(P, x.device, x.dtype)
    base = (p["q"][:, None, None, None] * k[:, :, None, :]
            + p["lam"][:, None, None, None] * rel[None, None])          # (E,B,P,P)
    a_s = torch.softmax(base + m_s[None, None], dim=-1)
    a_i = torch.softmax(base + m_i[None, None], dim=-1)
    c_in = torch.einsum("ebij,ebj->ebi", a_s, v)
    c_out = torch.einsum("ebij,ebj->ebi", a_i, v)

    out = (x + c_in[..., None] * p["w1"][:, None, None, :]
             + c_out[..., None] * p["w2"][:, None, None, :])
    if "rb" in p:
        out = out + p["rb"][:, None, None, :]
    dlog = p["ls"][:, None, None, None] * (
        torch.einsum("ebpc,edc->ebpd", out, code)
        - 0.5 * (code * code).sum(-1)[:, None, None, :]
    )
    if want is not None:
        want.append(dict(x=x, u=u, k=k, v=v, a_s=a_s, a_i=a_i,
                         c_in=c_in, c_out=c_out, out=out))
    return dlog


def count_free(p):
    return sum(int(v.numel() // v.shape[0]) for v in p.values())
