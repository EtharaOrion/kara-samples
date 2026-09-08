"""E-batched functional version of the block in `arch.py`.

Every tensor carries a leading ensemble axis E so that E independent members
train simultaneously against a shared batch (Adam is elementwise, and grads are
clipped per member, so the members never interact).  Seed variance dominates at
this size, so a wide lottery is far cheaper than sequential restarts.

This module is training-side only; it never ships.
"""
import math
import torch

# Parent architecture: everything the shipped form freezes is learned here.
DEFAULT = dict(
    C=1,        # width of the residual stream
    U=2,        # units in the point-wise bank
    P=10,       # (unused, sequence length comes from the batch)
)


def init(cfg, E, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    C, U = cfg["C"], cfg["U"]

    def r(*shape, s=1.0):
        return torch.randn(*shape, generator=g, device=device) * s

    pr = dict(
        code=r(E, 10, C, s=0.5),
        bank_w=r(E, U, C, s=1.0),
        knee=r(E, U, s=1.0),
        key_w=r(E, U, s=1.0),
        val_w=r(E, U, s=1.0),
        val_b=r(E, 1, s=0.1),
        q=torch.ones(E, 1, device=device) + r(E, 1, s=0.1),
        lam=-torch.rand(E, 1, generator=g, device=device) * 1.5 - 0.5,
        carry_w=r(E, C, s=0.5),
        fold_w=r(E, C, s=0.5),
        rb=torch.zeros(E, C, device=device),
        ls_log=r(E, 1, s=0.1),
    )
    for v in pr.values():
        v.requires_grad_(True)
    return pr


def forward(pr, tok_a, tok_b):
    """tok_a, tok_b: int64 [B,P].  Returns logits [E,B,P,10]."""
    code = pr["code"]                                     # [E,10,C]
    x = code[:, tok_a] + code[:, tok_b]                   # [E,B,P,C]

    g = torch.einsum("ebpc,euc->ebpu", x, pr["bank_w"]) + pr["knee"][:, None, None, :]
    g = torch.clamp(g, 0.0, 1.0)
    k = torch.einsum("ebpu,eu->ebp", g, pr["key_w"]) * pr["q"][:, :, None]
    v = torch.einsum("ebpu,eu->ebp", g, pr["val_w"]) + pr["val_b"][:, :, None]

    P = x.shape[2]
    p = torch.arange(P, device=x.device)
    dist = p[:, None] - p[None, :]
    logit = k[:, :, None, :] + pr["lam"][:, :, None, None] * dist
    floor = torch.full_like(logit, -1e30)
    a_in = torch.softmax(torch.where(dist > 0, logit, floor), -1)
    a_out = torch.softmax(torch.where(dist >= 0, logit, floor), -1)
    c_in = torch.einsum("ebpj,ebj->ebp", a_in, v)
    c_out = torch.einsum("ebpj,ebj->ebp", a_out, v)

    y = (x
         + c_in[..., None] * pr["carry_w"][:, None, None, :]
         + c_out[..., None] * pr["fold_w"][:, None, None, :]
         + pr["rb"][:, None, None, :])
    d2 = ((y[:, :, :, None, :] - code[:, None, None, :, :]) ** 2).sum(-1)
    return -d2 * torch.exp(pr["ls_log"])[:, :, None, None]


def bank_z(pr, tok_a, tok_b):
    """Pre-activation of the bank units: saturated iff z <= 0 or z >= 1."""
    x = pr["code"][:, tok_a] + pr["code"][:, tok_b]
    return torch.einsum("ebpc,euc->ebpu", x, pr["bank_w"]) + pr["knee"][:, None, None, :]


def loss_and_acc(pr, tok_a, tok_b, target, keep, tau=1.0):
    """target: [B,P] answer digit for each position (position 0 ignored).
    keep: [B] bool, samples that count (held-out ones are excluded)."""
    logits = forward(pr, tok_a, tok_b)                       # [E,B,P,10]
    logits = logits[:, :, 1:, :] * tau      # tau sharpens the training loss
                                            # only; it cannot change the arg-max
    tgt = target[:, 1:]
    lp = logits - torch.logsumexp(logits, -1, keepdim=True)
    pick = lp.gather(-1, tgt.expand(lp.shape[0], -1, -1).unsqueeze(-1)).squeeze(-1)
    m = keep[None, :, None].float()
    loss = -(pick * m).sum((1, 2)) / m.sum().clamp(min=1.0) / max(pick.shape[2], 1)
    with torch.no_grad():
        ok = (logits.argmax(-1) == tgt[None]).all(-1).float()   # [E,B]
        acc = (ok * keep[None].float()).sum(1) / keep.float().sum().clamp(min=1.0)
    return loss, acc


@torch.no_grad()
def exact_acc(pr, tok_a, tok_b, target, keep, chunk=4096):
    """Whole-number exact-match accuracy per member, chunked over the batch."""
    E = pr["code"].shape[0]
    dev = pr["code"].device
    hit = torch.zeros(E, device=dev)
    tot = 0
    for i in range(0, tok_a.shape[0], chunk):
        ka = keep[i:i + chunk]
        if ka.sum() == 0:
            continue
        ta, tb = tok_a[i:i + chunk][ka], tok_b[i:i + chunk][ka]
        tg = target[i:i + chunk][ka]
        pred = forward(pr, ta, tb)[:, :, 1:, :].argmax(-1)
        hit += (pred == tg[None, :, 1:]).all(-1).float().sum(1)
        tot += int(ka.sum())
    return hit / max(tot, 1), tot


@torch.no_grad()
def acc_and_margin(pr, tok_a, tok_b, target, keep, chunk=4096):
    """Exact-match accuracy and the worst read-out margin, per member."""
    E = pr["code"].shape[0]
    dev = pr["code"].device
    hit = torch.zeros(E, device=dev)
    worst = torch.full((E,), float("inf"), device=dev)
    tot = 0
    for i in range(0, tok_a.shape[0], chunk):
        ka = keep[i:i + chunk]
        if ka.sum() == 0:
            continue
        ta, tb = tok_a[i:i + chunk][ka], tok_b[i:i + chunk][ka]
        tg = target[i:i + chunk][ka][:, 1:]
        lg = forward(pr, ta, tb)[:, :, 1:, :]
        hit += (lg.argmax(-1) == tg[None]).all(-1).float().sum(1)
        good = lg.gather(-1, tg[None].expand(E, -1, -1).unsqueeze(-1)).squeeze(-1)
        other = lg.scatter(-1, tg[None].expand(E, -1, -1).unsqueeze(-1),
                           float("-inf")).max(-1).values
        worst = torch.minimum(worst, (good - other).amin((1, 2)))
        tot += int(ka.sum())
    return hit / max(tot, 1), worst, tot


def single(pr, e):
    """Pull member e out of an ensemble as a plain dict of float64 tensors."""
    return {k: v.detach()[e].double().clone() for k, v in pr.items()}


def unsqueeze(p1):
    """Put a single member back on an E=1 ensemble axis."""
    return {k: v[None].clone() for k, v in p1.items()}


def nparams(cfg):
    C, U = cfg["C"], cfg["U"]
    return 10 * C + U * C + U + U + U + 1 + 1 + 1 + C + C + C + 1
