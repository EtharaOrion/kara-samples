"""Exact gauge rewrites, structural cuts, and the stage runner for the ladder.

Every rewrite here is checked numerically against the model it came from: a
rewrite is only allowed to move the read-out argmax on zero of a large sample.
"""
import copy

import torch

import lab
from adder import Adder


# --------------------------------------------------------------------------
def model_of(cfg, sd, dev="cpu"):
    return lab.member_model(cfg, sd, dev=dev)


def code_of(cfg, sd):
    """The full 10-row code table (pinned rows first)."""
    return torch.cat([sd["code_fix"], sd["code"]], 0) if cfg["code_fix"] else sd["code"]


@torch.no_grad()
def equiv(cfg1, sd1, cfg2, sd2, B=1 << 14, seed=11, dev=lab.DEV):
    """(argmax agreement, max |logit difference|) between two models."""
    m1, m2 = model_of(cfg1, sd1, dev), model_of(cfg2, sd2, dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    agree, tot, dmax = 0, 0, 0.0
    for n in (8, 5, 11):
        ad, bd, _, _ = lab.sample(B, n, g, dev)
        l1, l2 = m1(ad, bd), m2(ad, bd)
        agree += int((l1.argmax(-1) == l2.argmax(-1)).all(-1).sum())
        tot += ad.shape[0]
        if l1.shape == l2.shape:
            dmax = max(dmax, float((l1 - l2).abs().max()))
    return agree / tot, dmax


def _clone(cfg, sd):
    return lab.default_cfg(**copy.deepcopy(dict(cfg))), {k: v.clone() for k, v in sd.items()}


def _sync(cfg, sd):
    """Rebuild the state dict against ``Adder(cfg)``.

    A cut can turn a parameter into a buffer or change a tensor's shape; the
    buffers then have to come from the new config.  Parameters must survive the
    cut with their shape intact -- a mismatch there would silently discard
    trained weights, so it is an error.
    """
    m = Adder(cfg)
    pnames = set(dict(m.named_parameters()))
    out = {}
    for k, v in m.state_dict().items():
        if k in sd and sd[k].shape == v.shape:
            out[k] = sd[k].detach().clone()
        else:
            assert k not in pnames, f"{k}: parameter shape changed by the cut"
            out[k] = v.detach().clone()
    return cfg, out


# --------------------------------------------------------------------------
# exact gauge rewrites
# --------------------------------------------------------------------------
def gauge_rotate(cfg, sd, R=None):
    """Rotate the code space (an exact symmetry of the whole block)."""
    cfg, sd = _clone(cfg, sd)
    C = cfg["C"]
    code = torch.cat([sd["code_fix"], sd["code"]], 0) if cfg["code_fix"] else sd["code"]
    if R is None:
        M = (code - code.mean(0, keepdim=True)).double()
        _, _, V = torch.linalg.svd(M, full_matrices=True)
        R = V                                            # rows = principal directions
        if float(R[0] @ (code[9] - code[0]).double()) < 0:
            R = R.clone(); R[0] = -R[0]
        if C > 1 and float(torch.linalg.det(R)) < 0:
            R = R.clone(); R[-1] = -R[-1]
    R = R.float()
    sd["code"] = sd["code"] @ R.t()
    if cfg["code_fix"]:
        sd["code_fix"] = sd["code_fix"] @ R.t()
    if isinstance(cfg["bw"], list):
        cfg["bw"] = (torch.tensor(cfg["bw"]).view(-1, C) @ R.t()).tolist()
    else:
        sd["bw"] = sd["bw"] @ R.t()
    for k in ("e1", "e2", "rb"):
        if k in sd:
            sd[k] = R @ sd[k]
        elif isinstance(cfg.get(k), list):
            cfg[k] = (R @ torch.tensor(cfg[k]).view(C)).tolist()
    return cfg, sd


def gauge_translate(cfg, sd, t):
    """code -> code - t, absorbed by the residual constant and the bank biases."""
    cfg, sd = _clone(cfg, sd)
    t = torch.as_tensor(t, dtype=torch.float32).view(cfg["C"])
    assert cfg["rb"], "translation needs the residual constant to absorb it"
    assert cfg["code_fix"] == 0, "the pinned code rows already fix this gauge"
    bw = sd["bw"] if "bw" in sd else torch.tensor(cfg["bw"]).view(-1, cfg["C"])
    sd["code"] = sd["code"] - t
    if cfg["code_fix"]:
        sd["code_fix"] = sd["code_fix"] - t
    sd["rb"] = sd["rb"] + t
    sd["bb"] = sd["bb"] + 2.0 * (bw @ t)
    return cfg, sd


def gauge_scale(cfg, sd, mu):
    """code -> mu * code: the bank, the write directions and the read-out
    temperature absorb it exactly."""
    cfg, sd = _clone(cfg, sd)
    assert mu > 0
    for k in ("code", "code_fix"):
        sd[k] = sd[k] * mu
    if "bw" in sd:
        sd["bw"] = sd["bw"] / mu
    else:
        cfg["bw"] = (torch.tensor(cfg["bw"]) / mu).tolist()
    for k in ("e1", "e2", "rb"):
        if k in sd:
            sd[k] = sd[k] * mu
        elif isinstance(cfg.get(k), list):
            cfg[k] = (torch.tensor(cfg[k]) * mu).tolist()
    if "ls" in sd:
        sd["ls"] = sd["ls"] / (mu * mu)
    else:
        cfg["ls"] = float(cfg["ls"]) / (mu * mu)
    return _sync(cfg, sd)


def unit_scale(cfg, sd, i, s):
    """clamp(w x + b) is unchanged if (w, b) scale together; used to normalise slopes."""
    cfg, sd = _clone(cfg, sd)
    assert s > 0
    sd["bw"][i] = sd["bw"][i] * s
    sd["bb"][i] = sd["bb"][i] * s
    return cfg, sd


# --------------------------------------------------------------------------
# structural cuts (exact only when the annealed quantity has landed)
# --------------------------------------------------------------------------
def drop_axis(cfg, sd, axis):
    """Drop a code axis.  Argmax-exact when code[:, axis] is constant."""
    cfg, sd = _clone(cfg, sd)
    C = cfg["C"]
    keep = [i for i in range(C) if i != axis]
    sd["code"] = sd["code"][:, keep]
    if cfg["code_fix"]:
        sd["code_fix"] = sd["code_fix"][:, keep]
    if "bw" in sd:
        sd["bw"] = sd["bw"][:, keep]
    else:
        cfg["bw"] = torch.tensor(cfg["bw"]).view(-1, C)[:, keep].tolist()
    for k in ("e1", "e2", "rb"):
        if k in sd:
            sd[k] = sd[k][keep]
        elif isinstance(cfg.get(k), list):
            cfg[k] = torch.tensor(cfg[k]).view(C)[keep].tolist()
    cfg["C"] = C - 1
    return _sync(cfg, sd)


def drop_units(cfg, sd, keep):
    """Keep a subset of bank units.  Exact when the dropped ones have kw = vw = 0."""
    cfg, sd = _clone(cfg, sd)
    idx = torch.tensor(keep)
    for k in ("bw", "bb", "kw", "vw"):
        if k in sd:
            sd[k] = sd[k][idx]
        elif isinstance(cfg.get(k), list):
            t = torch.tensor(cfg[k])
            cfg[k] = t.view(len(cfg["bb"]) if k == "bw" else -1, -1)[idx].tolist() if k == "bw" else t[idx].tolist()
    cfg["U"] = len(keep)
    return _sync(cfg, sd)


def reorder_units(cfg, sd, order):
    return drop_units(cfg, sd, order)


def pin(cfg, sd, name, value=None):
    """Turn a parameter into a buffer holding its current (annealed) value."""
    cfg, sd = _clone(cfg, sd)
    v = sd.pop(name)
    val = v.tolist() if value is None else value
    cfg[name] = val if name not in ("lam", "ls") else float(val if value is not None else v.item())
    return _sync(cfg, sd)


def fix_code_rows(cfg, sd, nfix=1):
    """Move the leading (zeroed) code rows into a buffer."""
    cfg, sd = _clone(cfg, sd)
    assert cfg["code_fix"] == 0
    code = sd["code"]
    assert float(code[:nfix].abs().max()) == 0.0, "rows to pin have not landed on zero"
    sd["code_fix"] = code[:nfix].clone()
    sd["code"] = code[nfix:].clone()
    cfg["code_fix"] = nfix
    return _sync(cfg, sd)


def drop_rb(cfg, sd):
    cfg, sd = _clone(cfg, sd)
    sd.pop("rb")
    cfg["rb"] = False
    return _sync(cfg, sd)


# --------------------------------------------------------------------------
# stage runner
# --------------------------------------------------------------------------
def anneal_specs(params, specs):
    out = []
    for name, target, mask, s, e in specs:
        p = params[name]
        shape = p.shape[1:]
        tgt = torch.as_tensor(target, dtype=torch.float32, device=p.device)
        if tgt.dim() == 0:
            tgt = tgt.expand(shape)
        m = None if mask is None else torch.as_tensor(mask, dtype=torch.float32, device=p.device).expand(shape)
        out.append(lab.Anneal(params, name, tgt, m, s, e))
    return out


def run_stage(cfg, sd, specs=(), E=64, sigma=0.02, steps=8000, lr=0.004, seed=0,
              ns=(8, 5, 11, 3), B=1024, tag="stage", log_every=1000, pct_start=0.2,
              eval_every=0, clip=1.0, mgn=0.0, ce=1.0, tau=0.45, mnorm=True):
    """Retrain a perturbed ensemble of one parent under a set of annealed cuts."""
    ens = lab.Ens(cfg, E, seed=seed, init_sd=sd, sigma=sigma)
    ann = anneal_specs(ens.params, specs)
    lab.train(ens, steps=steps, B=B, lr=lr, clip=clip, ns=ns, seed=seed, anneals=ann,
              log_every=log_every, eval_every=eval_every, tag=tag, pct_start=pct_start,
              mgn=mgn, ce=ce, tau=tau, mnorm=mnorm)
    acc_s = lab.ens_acc(ens, n=8, batches=4, B=8192)
    acc_u = lab.ens_acc(ens, n=8, batches=4, B=8192, mix=(1.0, 0.0, 0.0))
    score = torch.minimum(acc_s, acc_u)
    order = torch.argsort(score, descending=True)
    print(f"[{tag}] best min(struct,uniform): " +
          ", ".join(f"{float(score[i]):.5f}" for i in order[:6]), flush=True)
    return ens, order, score, acc_u, acc_s
