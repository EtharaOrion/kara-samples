"""Ensemble-batched functional forward for the digit-pair adder.

Everything here has a LEADING ensemble axis E so that E independent models train
simultaneously in one set of tensors (Adam is elementwise, so the members never
interact; gradients are clipped per member).

Token layout (LSB-first), for n digit places:
    P = n + 2 positions
    pos 0        : (0, 0) pad   -> supplies "carry-in = 0" for the ones place
    pos 1 .. n   : (a_i, b_i)   for i = 0 .. n-1
    pos n + 1    : (0, 0) pad   -> the final carry-out slot
Position p predicts answer digit p-1, so the whole sum comes out of one forward
pass.

Residual stream is C channels wide (C=2 for the cold-trained parent, C=1 after
projection).  A token embeds as code[a] + code[b] from ONE shared 10-entry table
that is also used as the read-out prototypes.
"""
import math

import torch


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def default_cfg(**kw):
    cfg = dict(
        C=2,             # residual channels
        U=2,             # clamp-bank units
        fix_bank_w=None,  # None => learned;  float => |Wb| pinned to this
        fix_key_w=None,   # None => learned;  float => key_w = (-v, +v)
        fix_val_w=None,   # None => learned;  tuple => val_w pinned
        fix_val_b=False,  # pin value bias to 0
        fix_lam=None,     # None => learned;  float => recency slope pinned
        fix_ls=None,      # None => learned;  float => read-out temperature
        fix_uA=None,      # None => learned;  float => carry write weight
        fix_code0=True,   # pin code[0] = 0
        fix_q=None,       # None => learned key/query scale; float => pinned
        act="clamp",      # bank nonlinearity: clamp | sigmoid | relu
    )
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def init_params(E, cfg, device, seed=0, s_code=1.0, s_bank=1.0, ls0=0.0):
    g = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape, s=1.0):
        return torch.randn(*shape, generator=g, device=device) * s

    C, U = cfg["C"], cfg["U"]
    p = {}
    # digit code: an arithmetic-progression-ish init helps nothing in
    # particular, so start from plain noise and let the lottery decide.
    p["code"] = rn(E, 10, C, s=s_code)
    p["Wb"] = rn(E, U, C, s=s_bank)
    p["bb"] = torch.rand(E, U, generator=g, device=device)
    p["kw"] = rn(E, U, s=1.0)
    p["vw"] = rn(E, U, s=1.0)
    p["vb"] = rn(E, 1, s=0.1)
    p["uA"] = rn(E, C, s=1.0)
    p["uB"] = rn(E, C, s=1.0)
    p["lam"] = -1.0 - 0.5 * torch.rand(E, 1, generator=g, device=device)
    p["q"] = torch.ones(E, 1, device=device) + rn(E, 1, s=0.1)
    p["log_ls"] = ls0 + rn(E, 1, s=0.2)
    return {k: v.contiguous() for k, v in p.items()}


def trainable_keys(cfg):
    ks = ["code", "Wb", "bb", "uB"]
    if cfg["fix_key_w"] is None:
        ks.append("kw")
    if cfg["fix_val_w"] is None:
        ks.append("vw")
    if not cfg["fix_val_b"]:
        ks.append("vb")
    if cfg["fix_uA"] is None:
        ks.append("uA")
    if cfg["fix_lam"] is None:
        ks.append("lam")
    if cfg["fix_q"] is None:
        ks.append("q")
    if cfg["fix_ls"] is None:
        ks.append("log_ls")
    return ks


def n_free_values(cfg):
    """Number of free scalars per ensemble member under this cfg."""
    C, U = cfg["C"], cfg["U"]
    n = 0
    n += (10 - (1 if cfg["fix_code0"] else 0)) * C      # code
    n += U * C if cfg["fix_bank_w"] is None else 0      # bank input weights
    n += U                                             # bank biases (knees)
    n += 0 if cfg["fix_key_w"] is not None else U
    n += 0 if cfg["fix_val_w"] is not None else U
    n += 0 if cfg["fix_val_b"] else 1
    n += 0 if cfg["fix_uA"] is not None else C
    n += C                                             # uB (the fold)
    n += 0 if cfg["fix_lam"] is not None else 1
    n += 0 if cfg["fix_q"] is not None else 1
    n += 0 if cfg["fix_ls"] is not None else 1
    return n


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------
def _masks(P, device):
    idx = torch.arange(P, device=device)
    dist = (idx[:, None] - idx[None, :]).float()        # p - j
    mA = dist > 0                                       # strictly causal
    mA = mA.clone()
    mA[0, 0] = True                                     # keep row 0 well-defined
    mB = dist >= 0                                      # inclusively causal
    return dist, mA, mB


_MASK_CACHE = {}


def masks(P, device):
    key = (P, str(device))
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _masks(P, device)
    return _MASK_CACHE[key]


def effective(p, cfg, device):
    """Resolve learned tensors + pinned constants into the tensors the forward
    pass actually uses."""
    E = p["code"].shape[0]
    C, U = cfg["C"], cfg["U"]
    code = p["code"]
    if cfg["fix_code0"]:
        code = torch.cat([torch.zeros_like(code[:, :1]), code[:, 1:]], dim=1)

    Wb = p["Wb"]
    if cfg["fix_bank_w"] is not None:
        # keep the learned SIGN pattern, pin the magnitude (a saturation
        # constant, not a degree of freedom: the knee bias absorbs the rest)
        Wb = torch.sign(Wb) * cfg["fix_bank_w"]

    kw = p["kw"]
    if cfg["fix_key_w"] is not None:
        v = cfg["fix_key_w"]
        kw = torch.tensor([-v, v], device=device).expand(E, U)

    vw = p["vw"]
    if cfg["fix_val_w"] is not None:
        vw = torch.tensor(list(cfg["fix_val_w"]), device=device).expand(E, U)

    vb = torch.zeros_like(p["vb"]) if cfg["fix_val_b"] else p["vb"]

    uA = p["uA"]
    if cfg["fix_uA"] is not None:
        uA = torch.full_like(p["uA"], float(cfg["fix_uA"]))

    lam = p["lam"]
    if cfg["fix_lam"] is not None:
        lam = torch.full_like(p["lam"], float(cfg["fix_lam"]))

    q = p["q"]
    if cfg["fix_q"] is not None:
        q = torch.full_like(p["q"], float(cfg["fix_q"]))

    if cfg["fix_ls"] is not None:
        ls = torch.full_like(p["log_ls"], float(cfg["fix_ls"]))
    else:
        ls = torch.exp(p["log_ls"])

    return dict(code=code, Wb=Wb, bb=p["bb"], kw=kw, vw=vw, vb=vb,
                uA=uA, uB=p["uB"], lam=lam, q=q, ls=ls)


def forward(p, cfg, tok, return_parts=False, key_content=True):
    """tok: (B, P, 2) int64.  Returns logits (E, B, P, 10)."""
    device = tok.device
    w = effective(p, cfg, device)
    code = w["code"]                                    # (E,10,C)
    B, P, _ = tok.shape

    x = code[:, tok[..., 0]] + code[:, tok[..., 1]]     # (E,B,P,C)
    e = torch.einsum("euc,ebpc->ebpu", w["Wb"], x) + w["bb"][:, None, None, :]
    a = cfg.get("act", "clamp")
    if a == "clamp":
        g = e.clamp(0.0, 1.0)                           # (E,B,P,U)
    elif a == "sigmoid":
        g = torch.sigmoid(4.0 * (e - 0.5))              # matches clamp at e=0.5
    elif a == "relu":
        g = torch.relu(e)
    else:
        raise ValueError(a)

    k = (g * w["kw"][:, None, None, :]).sum(-1)         # (E,B,P)
    if not key_content:                                 # ablation: freeze keys
        k = torch.zeros_like(k)
    v = (g * w["vw"][:, None, None, :]).sum(-1) + w["vb"][:, None, :]

    dist, mA, mB = masks(P, device)
    base = (w["q"][:, None, None, :] * k[:, :, None, :]
            + w["lam"][:, None, None, :] * dist)                # (E,B,P,P)
    NEG = torch.finfo(base.dtype).min / 4
    aA = torch.softmax(base.masked_fill(~mA, NEG), dim=-1)
    aB = torch.softmax(base.masked_fill(~mB, NEG), dim=-1)
    cA = (aA * v[:, :, None, :]).sum(-1)                # (E,B,P)
    cB = (aB * v[:, :, None, :]).sum(-1)

    y = x + cA[..., None] * w["uA"][:, None, None, :] \
          + cB[..., None] * w["uB"][:, None, None, :]   # (E,B,P,C)

    d2 = ((y[..., None, :] - code[:, None, None, :, :]) ** 2).sum(-1)
    logits = -w["ls"][:, None, None, :] * d2            # (E,B,P,10)
    if return_parts:
        return logits, dict(x=x, e=e, g=g, k=k, v=v, cA=cA, cB=cB, y=y,
                            aA=aA, aB=aB, w=w)
    return logits
