"""Warm-start transfer between two architectures of the DigitPairAdder family.

A trained parent is expanded into a canonical "full" parameterisation (every
structural switch written out as an explicit dense tensor), gauge-normalised,
projected onto the child's shape, and finally contracted back into whatever
reduced parameterisation the child uses.  The child is then *retrained*: this
is a warm start for structured pruning, not a way of writing weights by hand.

Two exact symmetries of the block are used for normalisation:
  * value rescale    Wv -> c * Wv,  Wo -> Wo / c   (attention map unchanged)
  * rescale of a residual axis the code does not occupy, which lets the
    attention write scale be pinned.

Rescaling a *code* axis is only a symmetry when the code occupies that axis
alone (C == 1): the readout compares the residual against the code table in the
same space, so scaling axis 0 by mu multiplies every logit by mu^2, which
leaves the argmax alone.  With C > 1 it would reweight the code dimensions
against each other and change the function, so `reduce_to` only normalises the
code unit for a child that has C == 1 (which every `code_fix` child does).
"""
import torch

from model_src import DigitPairAdder, default_cfg, init_model


def expand(sd, cfg):
    """Parent state dict -> canonical dense tensors."""
    D, C, U1, U2, KV = cfg["D"], cfg["C"], cfg["U1"], cfg["U2"], cfg["KV"]
    cf = cfg["code_fix"]
    f = sd["code_free0"]
    one = torch.ones(1, dtype=f.dtype)
    zero = torch.zeros(1, dtype=f.dtype)
    if cf == 0:
        col0 = f.clone()
    elif cf == 1:
        col0 = torch.cat([f[:1], one, f[1:]])
    else:
        col0 = torch.cat([zero, one, f])

    g = {}
    g["col0"] = col0
    g["rest"] = sd["code_rest"].clone() if C > 1 else torch.zeros(10, 0)

    if cfg["f1_in"] == "full":
        g["W1"] = sd["W1"].clone()
    else:
        W1 = torch.zeros(D, U1); W1[0, :] = 1.0
        g["W1"] = W1
    g["b1"] = sd["b1"].clone()
    if cfg["f1_out"] == "full":
        g["O1"] = sd["O1"].clone()
    else:
        o1 = sd["o1"]
        if cfg["f1_tie"]:
            o1 = torch.cat([o1, -o1.sum(0, keepdim=True)])
        O1 = torch.zeros(U1, D); O1[:, KV] = o1
        g["O1"] = O1

    if cfg["kv"] == "full":
        g["Wk"] = sd["Wk"].clone(); g["Wv"] = sd["Wv"].clone()
    else:
        e = torch.zeros(D, 1); e[KV, 0] = 1.0
        g["Wk"] = e.clone(); g["Wv"] = e.clone()
    g["Wq"] = sd["Wq"].clone() if cfg["q"] == "proj" else torch.zeros(D, 1)
    g["bq"] = sd["bq"].clone()
    if cfg["wo"] == "full":
        g["Wo"] = sd["Wo"].clone()
    else:
        Wo = torch.zeros(1, D)
        Wo[0, 0] = float(sd["wo_s"]) if cfg["wo"] == "axis" else 1.0
        g["Wo"] = Wo
    g["lam"] = sd["lam"].clone() if cfg["lam"] == "learn" \
        else torch.tensor([float(cfg["lam_fix"])])

    if cfg["f2_in"] == "full":
        g["W2"] = sd["W2"].clone()
    else:
        W2 = torch.zeros(D, U2); W2[0, :] = 1.0
        g["W2"] = W2
    g["b2"] = sd["b2"].clone()
    if cfg["f2_out"] == "full":
        g["O2"] = sd["O2"].clone()
    else:
        o2 = sd["o2"]
        if cfg["f2_tie"]:
            o2 = torch.cat([o2, -o2])
        O2 = torch.zeros(U2, D); O2[:, 0] = o2
        g["O2"] = O2
    return g


def scale_axis(g, ax, s):
    """Exact symmetry: rescale residual axis `ax` by s (code axes included)."""
    if s == 0:
        return g
    if ax == 0:
        g["col0"] = g["col0"] * s
    elif ax - 1 < g["rest"].shape[1]:
        g["rest"] = g["rest"].clone(); g["rest"][:, ax - 1] *= s
    g["W1"] = g["W1"].clone(); g["W1"][ax, :] /= s
    g["W2"] = g["W2"].clone(); g["W2"][ax, :] /= s
    g["Wk"] = g["Wk"].clone(); g["Wk"][ax, 0] /= s
    g["Wv"] = g["Wv"].clone(); g["Wv"][ax, 0] /= s
    g["Wq"] = g["Wq"].clone(); g["Wq"][ax, 0] /= s
    g["O1"] = g["O1"].clone(); g["O1"][:, ax] *= s
    g["O2"] = g["O2"].clone(); g["O2"][:, ax] *= s
    g["Wo"] = g["Wo"].clone(); g["Wo"][0, ax] *= s
    return g


def scale_value(g, c):
    """Exact symmetry: Wv -> c Wv, Wo -> Wo / c (attention map untouched)."""
    if c == 0:
        return g
    g["Wv"] = g["Wv"] * c
    g["Wo"] = g["Wo"] / c
    return g


def reduce_to(g, pcfg, ccfg):
    """Project the canonical parent onto the child's structure."""
    g = dict(g)
    KV = ccfg["KV"]

    # --- normalise the code unit if the child pins it -------------------
    if ccfg["code_fix"] >= 1:
        if ccfg["C"] != 1:
            raise ValueError("code_fix pins the axis-0 gauge, which is only a "
                             "symmetry when the code is one-dimensional")
        v1 = float(g["col0"][1])
        if abs(v1) > 1e-8:
            g = scale_axis(g, 0, 1.0 / v1)

    # --- pin the attention write scale ----------------------------------
    if ccfg["wo"] == "fix" and ccfg["kv"] == "id":
        prod = float(g["Wv"][KV, 0]) * float(g["Wo"][0, 0])
        if abs(prod) > 1e-8:
            g = scale_axis(g, KV, prod)
    elif ccfg["wo"] == "fix":
        c = float(g["Wo"][0, 0])
        if abs(c) > 1e-8:
            g = scale_value(g, c)

    # --- unit selection when a hidden layer shrinks ----------------------
    U1p, U1c = g["O1"].shape[0], ccfg["U1"]
    if U1c < U1p:
        keep = g["O1"].abs().sum(1).topk(U1c).indices.sort().values
        g["O1"] = g["O1"][keep]; g["b1"] = g["b1"][keep]; g["W1"] = g["W1"][:, keep]
    U2p, U2c = g["O2"].shape[0], ccfg["U2"]
    if U2c < U2p:
        keep = g["O2"].abs().sum(1).topk(U2c).indices.sort().values
        g["O2"] = g["O2"][keep]; g["b2"] = g["b2"][keep]; g["W2"] = g["W2"][:, keep]

    # --- residual width / code width -------------------------------------
    Dc, Cc = ccfg["D"], ccfg["C"]
    if Dc < g["W1"].shape[0]:
        for k in ("W1", "W2", "Wk", "Wv", "Wq"):
            g[k] = g[k][:Dc]
        for k in ("O1", "O2", "Wo"):
            g[k] = g[k][:, :Dc]
    elif Dc > g["W1"].shape[0]:
        Dp = g["W1"].shape[0]
        for k, sh in (("W1", (Dc, g["W1"].shape[1])), ("W2", (Dc, g["W2"].shape[1])),
                      ("Wk", (Dc, 1)), ("Wv", (Dc, 1)), ("Wq", (Dc, 1))):
            t = torch.zeros(sh); t[:Dp] = g[k]; g[k] = t
        for k in ("O1", "O2", "Wo"):
            t = torch.zeros(g[k].shape[0], Dc); t[:, :Dp] = g[k]; g[k] = t
    g["rest"] = g["rest"][:, :max(Cc - 1, 0)]
    return g


def contract(g, ccfg, child):
    """Canonical tensors -> the child's actual parameter dict."""
    out = {}
    D, KV = ccfg["D"], ccfg["KV"]
    cf = ccfg["code_fix"]
    col0 = g["col0"]
    if cf == 0:
        out["code_free0"] = col0.clone()
    elif cf == 1:
        out["code_free0"] = torch.cat([col0[:1], col0[2:]])
    else:
        out["code_free0"] = col0[2:].clone()
    if ccfg["C"] > 1:
        out["code_rest"] = g["rest"].clone()

    W1, b1, O1 = g["W1"], g["b1"], g["O1"]
    if ccfg["f1_in"] == "full":
        out["W1"] = W1.clone(); out["b1"] = b1.clone()
    else:
        s = W1[0, :].clone()
        s = torch.where(s.abs() < 1e-6, torch.full_like(s, 1e-6), s)
        out["b1"] = b1 / s
        O1 = O1 * s[:, None]          # relu(s*h0 + b1) = s * relu(h0 + b1/s)
    if ccfg["f1_out"] == "full":
        out["O1"] = O1.clone()
    else:
        o1 = O1[:, KV].clone()
        out["o1"] = o1[:-1] if ccfg["f1_tie"] else o1

    if ccfg["kv"] == "full":
        out["Wk"] = g["Wk"].clone(); out["Wv"] = g["Wv"].clone()
        kscale, vscale = 1.0, 1.0
    else:
        kscale = float(g["Wk"][KV, 0]); vscale = float(g["Wv"][KV, 0])
    if ccfg["q"] == "proj":
        out["Wq"] = g["Wq"] * kscale
    out["bq"] = g["bq"] * kscale
    Wo = g["Wo"] * vscale
    if ccfg["wo"] == "full":
        out["Wo"] = Wo.clone()
    elif ccfg["wo"] == "axis":
        out["wo_s"] = Wo[0, :1].clone()
    if ccfg["lam"] == "learn":
        out["lam"] = g["lam"].clone()

    W2, b2, O2 = g["W2"], g["b2"], g["O2"]
    if ccfg["f2_in"] == "full":
        out["W2"] = W2.clone(); out["b2"] = b2.clone()
    else:
        s = W2[0, :].clone()
        s = torch.where(s.abs() < 1e-6, torch.full_like(s, 1e-6), s)
        out["b2"] = b2 / s
        O2 = O2 * s[:, None]
    if ccfg["f2_out"] == "full":
        out["O2"] = O2.clone()
    else:
        o2 = O2[:, 0].clone()
        out["o2"] = o2[:o2.numel() // 2] if ccfg["f2_tie"] else o2

    fixed = {}
    for name, p in child.named_parameters():
        v = out.get(name)
        fixed[name] = p.detach().clone() if v is None or v.numel() != p.numel() \
            else v.reshape(p.shape).float()
    return fixed


def warm_start(parent_sd, parent_cfg, child_cfg, seed=0):
    pcfg = dict(default_cfg()); pcfg.update(parent_cfg)
    ccfg = dict(default_cfg()); ccfg.update(child_cfg)
    child = init_model(ccfg, seed)
    sd = {k: v.detach().cpu().float() for k, v in parent_sd.items()}
    g = expand(sd, pcfg)
    g = reduce_to(g, pcfg, ccfg)
    return contract(g, ccfg, child), ccfg


def identity_check(parent_sd, parent_cfg, child_cfg, tok):
    """Max logit difference between parent and warm-started child."""
    pcfg = dict(default_cfg()); pcfg.update(parent_cfg)
    ccfg = dict(default_cfg()); ccfg.update(child_cfg)
    pm = DigitPairAdder(pcfg)
    with torch.no_grad():
        for k, p in pm.named_parameters():
            p.copy_(parent_sd[k].detach().cpu().float().reshape(p.shape))
    sd, _ = warm_start(parent_sd, pcfg, ccfg)
    cm = DigitPairAdder(ccfg)
    with torch.no_grad():
        for k, p in cm.named_parameters():
            p.copy_(sd[k])
        a, b = pm(tok), cm(tok)
        agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
        return float((a - b).abs().max()), agree
