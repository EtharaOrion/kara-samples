"""Training machinery for the digit-pair adder.

Nothing here ships: the graded file contains the model class and its inference
path only.  This module holds the data generator, the E-way ensemble trainer
("seed lottery"), and the annealed structural cuts used to walk the model down
to its parameter floor.
"""
import copy
import json
import math
import os

import torch
from torch.func import functional_call, stack_module_state

from adder import Adder

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def default_cfg(**kw):
    cfg = dict(
        C=2,            # code dimension
        U=4,            # bank units
        code_fix=0,     # code rows pinned to zero
        bw=None,        # bank input weights (None = learned)
        kw=None,        # key read-out (None = learned)
        vw=None,        # value read-out (None = learned)
        e1=None,        # strict-head write direction
        e2=None,        # inclusive-head write direction
        rb=True,        # residual constant present
        lam=None,       # relative-position slope (None = learned)
        ls=None,        # read-out temperature (None = learned)
    )
    cfg.update(kw)
    return cfg


def n_params(cfg):
    return sum(p.numel() for p in Adder(cfg).parameters())


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _hash16(av, bv):
    h = av * 1000003 + bv * 19260817
    h = h ^ (h >> 31)
    h = h * 2246822519
    h = h ^ (h >> 27)
    h = h * 3266489917
    h = h ^ (h >> 33)
    return h & 15


def bucket_of(ad, bd):
    """1-in-16 held-out bucket of a digit-pair batch (bucket 0 = held out)."""
    n = ad.shape[1]
    pw = (10 ** torch.arange(n, device=ad.device, dtype=torch.long))
    return _hash16((ad * pw).sum(1), (bd * pw).sum(1))


def sample(B, n, gen, dev=DEV, mix=(0.35, 0.40, 0.25), full_width=True):
    """Digit pairs, answer digits, and the held-out bucket.

    Three regimes are mixed: uniform digits, per-place carry-transparent
    enrichment (a + b = 9 with probability .4), and near-maximal carry chains
    (probability .9).
    """
    a = torch.randint(0, 10, (B, n), generator=gen, device=dev)
    b = torch.randint(0, 10, (B, n), generator=gen, device=dev)
    m = torch.rand(B, 1, generator=gen, device=dev)
    p = torch.where(m < mix[0], 0.0, torch.where(m < mix[0] + mix[1], 0.4, 0.9))
    t = torch.rand(B, n, generator=gen, device=dev) < p
    b = torch.where(t, 9 - a, b)
    if full_width:                       # both operands carry the full digit width
        am = torch.randint(1, 10, (B,), generator=gen, device=dev)
        bm = torch.randint(1, 10, (B,), generator=gen, device=dev)
        at = torch.randint(1, 9, (B,), generator=gen, device=dev)
        hi = t[:, n - 1]
        a[:, n - 1] = torch.where(hi, at, am)
        b[:, n - 1] = torch.where(hi, 9 - a[:, n - 1], bm)
    s = a + b
    outs, carry = [], torch.zeros(B, dtype=torch.long, device=dev)
    for i in range(n):
        v = s[:, i] + carry
        outs.append(v % 10)
        carry = v // 10
    outs.append(carry)
    tgt = torch.stack([torch.zeros_like(carry)] + outs, 1)      # (B, n+2)
    return a, b, tgt, bucket_of(a, b)


def uniform_pairs(B, n, gen, dev=DEV):
    """Uniform full-width operands (the graded distribution)."""
    return sample(B, n, gen, dev, mix=(1.0, 0.0, 0.0))


# --------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------
def init_member(m, g, scale=1.0):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name == "code":
                p.normal_(0.0, 0.6 * scale, generator=g)
            elif name == "bb":
                p.uniform_(0.0, 1.0, generator=g)
            elif name in ("e1", "e2"):
                p.normal_(0.0, 0.5 * scale, generator=g)
            elif name == "rb":
                p.zero_()
            elif name == "lam":
                p.fill_(-2.0)
            elif name == "ls":
                p.fill_(1.0)
            else:
                p.normal_(0.0, 1.0 * scale, generator=g)
    return m


class Ens:
    """E independent members trained simultaneously on a shared batch."""

    def __init__(self, cfg, E, seed=0, init_sd=None, sigma=0.0, dev=DEV):
        self.cfg, self.E, self.dev = cfg, E, dev
        ms = []
        for i in range(E):
            g = torch.Generator(device="cpu").manual_seed(seed * 7919 + i + 1)
            m = Adder(cfg)
            if init_sd is None:
                init_member(m, g)
            else:
                m.load_state_dict(init_sd, strict=True)
                if sigma > 0 and i > 0:
                    with torch.no_grad():
                        for p in m.parameters():
                            p.add_(torch.randn(p.shape, generator=g) * sigma * p.abs().mean().clamp(min=1e-3))
            ms.append(m.to(dev))
        params, buffers = stack_module_state(ms)
        self.params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
        self.buffers = {k: v.detach().clone() for k, v in buffers.items()}
        self.base = Adder(cfg).to("meta")

        def _f(p, b, ad, bd):
            return functional_call(self.base, (p, b), (ad, bd))

        self.fwd = torch.vmap(_f, in_dims=(0, 0, None, None))

    def logits(self, ad, bd):
        return self.fwd(self.params, self.buffers, ad, bd)

    def member_sd(self, i):
        sd = {k: v[i].detach().cpu().clone() for k, v in self.params.items()}
        sd.update({k: v[i].detach().cpu().clone() for k, v in self.buffers.items()})
        return sd

    def load_member(self, i, sd):
        with torch.no_grad():
            for k, v in sd.items():
                if k in self.params:
                    self.params[k][i].copy_(v)


def member_model(cfg, sd, dev=DEV):
    m = Adder(cfg).to(dev)
    m.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
    m.eval()
    return m


# --------------------------------------------------------------------------
# loss / evaluation
# --------------------------------------------------------------------------
def nll(logits, tgt, keep):
    """Mean per-member cross entropy over kept examples."""
    E = logits.shape[0]
    lp = logits.log_softmax(-1)
    t = tgt.unsqueeze(0).expand(E, -1, -1).unsqueeze(-1)
    n = -lp.gather(-1, t).squeeze(-1)                     # (E, B, P)
    w = keep.to(n.dtype).unsqueeze(0).unsqueeze(-1)
    return (n * w).sum((1, 2)) / (w.sum() * n.shape[2])


def margins(logits, tgt, code, ls, norm=True):
    """Signed distance from each read-out to its nearest decision boundary,
    in units of the smallest prototype spacing.

    ``logits = -ls * |r - code|^2``, so ``(l[d*] - l[d]) / (2 ls |c[d*]-c[d]|)``
    is exactly the signed distance from ``r`` to the bisector of the two
    prototypes -- positive when the read-out is on the correct side.  Dividing
    by the smallest spacing makes it scale free, so unlike cross entropy it
    cannot be improved by inflating the code.
    """
    E = logits.shape[0]
    t = tgt.unsqueeze(0).expand(E, -1, -1)                       # (E, B, P)
    lt = logits.gather(-1, t.unsqueeze(-1))                      # (E, B, P, 1)
    d = torch.cdist(code, code)                                  # (E, 10, 10)
    sel = d[torch.arange(E, device=d.device)[:, None, None], t]  # (E, B, P, 10)
    m = (lt - logits) / (2.0 * ls.view(E, 1, 1, 1) * sel.clamp(min=1e-9))
    m = m.scatter(-1, t.unsqueeze(-1), torch.full_like(lt, 1e9)).amin(-1)
    if not norm:
        return m
    eye = torch.eye(10, dtype=torch.bool, device=d.device)
    gmin = d.masked_fill(eye, 1e9).amin((1, 2))                  # (E,)
    return m / gmin.view(E, 1, 1)


def margin_loss(logits, tgt, keep, code, ls, tau=0.45, beta=0.05, norm=True):
    """Hinge on the read-out margin.

    ``norm`` divides by the smallest prototype spacing, which makes the target
    scale-free.  That is the right thing when the code scale is still a free
    gauge; once ``e1`` is pinned the scale is anchored and the certificate
    compares an *absolute* margin against an absolute leakage bound, so the
    unnormalised form is what actually matters.
    """
    m = margins(logits, tgt, code, ls, norm=norm)
    pen = torch.nn.functional.softplus((tau - m) / beta) * beta
    w = keep.to(pen.dtype).unsqueeze(0).unsqueeze(-1)
    return (pen * w).sum((1, 2)) / (w.sum() * pen.shape[2])


def ens_code_ls(ens):
    src = {**ens.params, **ens.buffers}
    code = torch.cat([src["code_fix"], src["code"]], 1)
    return code, src["ls"]


@torch.no_grad()
def ens_acc(ens, n=8, batches=8, B=4096, seed=1234, held_out=True, mix=(0.35, 0.40, 0.25)):
    """Per-member exact-match accuracy on the held-out (or training) split."""
    g = torch.Generator(device=ens.dev).manual_seed(seed)
    ok = torch.zeros(ens.E, device=ens.dev)
    tot = 0
    for _ in range(batches):
        ad, bd, tgt, bk = sample(B, n, g, ens.dev, mix=mix)
        sel = (bk == 0) if held_out else (bk != 0)
        if sel.sum() == 0:
            continue
        ad, bd, tgt = ad[sel], bd[sel], tgt[sel]
        pred = ens.logits(ad, bd).argmax(-1)              # (E, b, P)
        ok += (pred[:, :, 1:] == tgt[None, :, 1:]).all(-1).sum(1).float()
        tot += ad.shape[0]
    return (ok / max(tot, 1)).cpu()


@torch.no_grad()
def model_acc(model, n=8, batches=8, B=8192, seed=99, held_out=True, mix=(0.35, 0.40, 0.25)):
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    ok = tot = 0
    for _ in range(batches):
        ad, bd, tgt, bk = sample(B, n, g, dev, mix=mix)
        if held_out is not None:
            sel = (bk == 0) if held_out else (bk != 0)
            ad, bd, tgt = ad[sel], bd[sel], tgt[sel]
        if ad.shape[0] == 0:
            continue
        pred = model(ad, bd).argmax(-1)
        ok += int((pred[:, 1:] == tgt[:, 1:]).all(-1).sum())
        tot += ad.shape[0]
    return ok / max(tot, 1), tot


# --------------------------------------------------------------------------
# annealed structural cuts
# --------------------------------------------------------------------------
class Anneal:
    """Hard cap on the distance between a parameter and its target value.

    The cap shrinks linearly to zero, so the parameter lands exactly on the
    target and can then be turned into a buffer.  Applied as a projection after
    every optimiser step: unlike a penalty (or a forward-pass interpolation)
    it cannot be traded off against the loss.
    """

    def __init__(self, params, name, target, mask=None, start=0.0, end=0.6):
        p = params[name]
        self.name, self.start, self.end = name, start, end
        tgt = torch.as_tensor(target, dtype=p.dtype, device=p.device)
        self.target = tgt.expand_as(p).contiguous()
        m = torch.ones_like(p) if mask is None else torch.as_tensor(mask, device=p.device).expand_as(p).float()
        self.mask = m
        self.eps0 = (p.detach() - self.target).abs() * m

    def apply(self, params, frac):
        p = params[self.name]
        if frac <= self.start:
            return
        r = min(1.0, (frac - self.start) / max(1e-9, self.end - self.start))
        allow = self.eps0 * (1.0 - r) + (1.0 - self.mask) * 1e30
        with torch.no_grad():
            d = (p - self.target).clamp(-allow, allow)
            p.copy_(self.target + d)


def train(ens, steps=25000, B=1024, lr=0.012, wd=0.0, clip=1.0, ns=(8,), seed=0,
          anneals=(), log_every=1000, mix=(0.35, 0.40, 0.25), pct_start=0.15,
          eval_every=0, tag="", mgn=0.0, ce=1.0, tau=0.45, mnorm=True):
    g = torch.Generator(device=ens.dev).manual_seed(seed + 555)
    opt = torch.optim.AdamW(list(ens.params.values()), lr=lr, betas=(0.9, 0.99), weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=pct_start, anneal_strategy="cos",
                                                div_factor=10.0, final_div_factor=100.0)
    E = ens.E
    hist = []
    for step in range(steps):
        n = ns[step % len(ns)]
        ad, bd, tgt, bk = sample(B, n, g, ens.dev, mix=mix)
        keep = bk != 0
        lg = ens.logits(ad, bd)
        loss_e = ce * nll(lg, tgt, keep)
        if mgn > 0.0:
            code, ls = ens_code_ls(ens)
            loss_e = loss_e + mgn * margin_loss(lg, tgt, keep, code, ls, tau=tau,
                                                norm=mnorm)
        loss_e.sum().backward()
        with torch.no_grad():
            sq = torch.zeros(E, device=ens.dev)
            for p in ens.params.values():
                if p.grad is not None:
                    sq += p.grad.reshape(E, -1).pow(2).sum(1)
            sc = (clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
            for p in ens.params.values():
                if p.grad is not None:
                    p.grad.mul_(sc.view(-1, *([1] * (p.dim() - 1))))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        frac = (step + 1) / steps
        for a in anneals:
            a.apply(ens.params, frac)
        if log_every and (step + 1) % log_every == 0:
            with torch.no_grad():
                best = loss_e.min().item()
            msg = f"[{tag}] step {step+1}/{steps} loss(min) {best:.5f}"
            if eval_every and (step + 1) % eval_every == 0:
                acc = ens_acc(ens, batches=2, B=4096)
                msg += f" acc(max) {acc.max():.4f} n>=.99 {(acc >= 0.99).sum().item()}"
                hist.append((step + 1, acc.max().item()))
            print(msg, flush=True)
    return hist


def save_ckpt(path, cfg, sd, note=""):
    torch.save({"cfg": cfg, "sd": sd, "note": note}, path)


def load_ckpt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return default_cfg(**d["cfg"]), d["sd"], d.get("note", "")
