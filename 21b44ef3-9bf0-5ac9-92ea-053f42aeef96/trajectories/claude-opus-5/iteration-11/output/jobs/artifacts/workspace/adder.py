"""Digit-pair addition transformer: a single attention block over per-place tokens.

Layout (LSB first), for n digit places:

    pos 0        : (0, 0) pad          -- anchors "carry into place 0 is zero"
    pos 1 .. n   : (a_i, b_i)          -- the i-th place of the two operands
    pos n+1      : (0, 0) pad          -- slot for the final carry-out digit

Position i predicts answer digit i-1, so one forward pass emits all n+1 answer
digits.  Everything below is written as a pure function of a parameter dict whose
tensors carry a leading ensemble axis E, so E independent members train at once
(the "seed lottery": at this size the outcome is dominated by initialisation).

The residual stream is a single scalar per position.  A token embeds as
code[a] + code[b] from one learned 10-entry table, and the same table supplies the
read-out prototypes, so the model has to discover a code in which summing two
digit embeddings is meaningful.

The bank is one learned threshold theta forming a symmetric V feature:

    upos = clamp(alpha*(x - theta), 0, 1)      # "this place is above theta"
    uneg = clamp(-alpha*(x - theta), 0, 1)     # "this place is below theta"
    n    = upos + uneg = min(1, alpha*|x-theta|)

n is the attention key (via a fixed gain) and upos is the attention value.  Both
heads share that one key/value stream and differ only in their mask: head 1 is
strictly causal, head 2 is inclusively causal.
"""

import math
import torch


# ---------------------------------------------------------------- configuration

DEFAULTS = dict(
    # which quantities are free parameters; the rest are buffers
    free_theta_neg=False,  # separate threshold for the descending clamp (parent only)
    free_alpha=True,       # bank slope
    free_e1=True,          # head-1 (carry-in) output weight
    free_ls=True,          # read-out temperature
    free_kw=True,          # attention key gain
    free_lam=True,         # attention relative-distance bias
    free_rb=True,          # residual bias on the embedding sum
    code_fix=0,            # number of leading code entries pinned to 0.0
    # buffer values used when the corresponding free_* flag is off
    alpha=2.0,
    e1=1.0,
    ls=1.0,
    kw=2000.0,
    lam=-8.0,
)


def default_cfg(**kw):
    cfg = dict(DEFAULTS)
    cfg.update(kw)
    return cfg


PARAM_ORDER = [
    "code", "rb", "theta", "theta_neg", "alpha", "e1", "e2", "ls", "kw", "lam",
]


def is_param(name, cfg):
    """Is `name` a free parameter under this config (vs a fixed buffer)?"""
    if name == "code":
        return True          # code_fix pins entries inside the tensor, see split_code
    if name == "e2":
        return True
    if name == "theta":
        return True
    return bool(cfg.get("free_" + name, False))


def init_params(cfg, E, device, seed=0, dtype=torch.float32):
    """Random init for E independent members.  Shapes all lead with E."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype)

    nfix = cfg["code_fix"]
    p = {}
    # a code that starts out as a noisy ramp of random scale would give away the
    # answer, so start from plain noise
    p["code"] = r(E, 10, scale=1.0)
    p["rb"] = r(E, scale=0.1)
    p["theta"] = r(E, scale=1.0)
    p["theta_neg"] = p["theta"].clone()
    p["alpha"] = torch.full((E,), 1.0, device=device, dtype=dtype) + r(E, scale=0.1)
    p["e1"] = r(E, scale=0.5)
    p["e2"] = r(E, scale=0.5)
    p["ls"] = torch.full((E,), 1.0, device=device, dtype=dtype)
    p["kw"] = torch.full((E,), 8.0, device=device, dtype=dtype) + r(E, scale=1.0)
    p["lam"] = torch.full((E,), -2.0, device=device, dtype=dtype) + r(E, scale=0.2)
    if nfix:
        p["code"][:, :nfix] = 0.0
    return p


def split_params(p, cfg):
    """Return (trainable dict, frozen dict) given the config."""
    train, frozen = {}, {}
    for k in PARAM_ORDER:
        if k == "theta_neg" and not cfg["free_theta_neg"]:
            frozen[k] = p[k]
            continue
        (train if is_param(k, cfg) else frozen)[k] = p[k]
    return train, frozen


def code_mask(cfg, device):
    """1.0 where a code entry is free, 0.0 where it is pinned to its fixed value."""
    m = torch.ones(10, device=device)
    m[: cfg["code_fix"]] = 0.0
    return m


def n_params(cfg):
    """Free scalar count for one member."""
    n = (10 - cfg["code_fix"])          # code table
    n += 1                              # theta
    n += 1                              # e2
    for k in ("theta_neg", "alpha", "e1", "ls", "kw", "lam", "rb"):
        if cfg.get("free_" + k, False):
            n += 1
    return n


# ------------------------------------------------------------------- the model

def geometry(P, device, dtype=torch.float32):
    """Position-independent attention geometry for a sequence of P places.

    dist[i, j]  = i - j          (relative distance, used with the lam bias)
    m_strict    = 0 where j <  i (plus j=0 for row 0 so the softmax is defined)
    m_incl      = 0 where j <= i
    """
    idx = torch.arange(P, device=device)
    dist = (idx[:, None] - idx[None, :]).to(dtype)
    strict = idx[None, :] < idx[:, None]
    strict = strict.clone()
    strict[0, 0] = True                      # row 0 sees only the pad: carry-in 0
    incl = idx[None, :] <= idx[:, None]
    neg = torch.finfo(dtype).min / 4
    m_strict = torch.where(strict, torch.zeros((), device=device, dtype=dtype),
                           torch.full((), neg, device=device, dtype=dtype))
    m_incl = torch.where(incl, torch.zeros((), device=device, dtype=dtype),
                         torch.full((), neg, device=device, dtype=dtype))
    return dist, m_strict, m_incl


def forward(p, cfg, ab, geo=None, freeze_attn=None, return_parts=False):
    """ab: int64 [B, P, 2] digit pairs.  Returns logits [E, B, P, 10]."""
    E = p["code"].shape[0]
    B, P, _ = ab.shape
    device = p["code"].device
    if geo is None:
        geo = geometry(P, device, p["code"].dtype)
    dist, m_strict, m_incl = geo

    code = p["code"]                                        # [E, 10]
    flat = ab.reshape(-1)                                   # [B*P*2]
    x = code.index_select(1, flat).view(E, B, P, 2).sum(-1)  # [E, B, P]
    x = x + p["rb"].view(E, 1, 1)

    alpha = p["alpha"].view(E, 1, 1)
    t_pos = alpha * (x - p["theta"].view(E, 1, 1))
    t_neg = alpha * (p["theta_neg"].view(E, 1, 1) - x)
    upos = t_pos.clamp(0.0, 1.0)
    uneg = t_neg.clamp(0.0, 1.0)

    key = p["kw"].view(E, 1, 1) * (upos + uneg)             # [E, B, P]
    bias = p["lam"].view(E, 1, 1, 1) * dist                 # [E, 1, P, P]
    base = key[:, :, None, :] + bias

    val = upos[:, :, None, :]
    if freeze_attn is None:
        a1 = torch.softmax(base + m_strict, dim=-1)
        a2 = torch.softmax(base + m_incl, dim=-1)
    else:                                                    # ablation: fixed maps
        a1, a2 = freeze_attn
    o1 = (a1 * val).sum(-1)
    o2 = (a2 * val).sum(-1)

    y = x + p["e1"].view(E, 1, 1) * o1 + p["e2"].view(E, 1, 1) * o2
    d = y[..., None] - code[:, None, None, :]
    logits = -p["ls"].view(E, 1, 1, 1).abs() * d * d
    if return_parts:
        return logits, dict(x=x, upos=upos, uneg=uneg, key=key, o1=o1, o2=o2, y=y,
                            a1=a1, a2=a2)
    return logits
