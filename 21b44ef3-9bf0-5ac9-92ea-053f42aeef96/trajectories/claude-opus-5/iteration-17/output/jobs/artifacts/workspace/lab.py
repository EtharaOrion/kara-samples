"""Shared training / evaluation utilities for the ensemble lottery."""
import torch

import arch
import data


def ce_loss(logits, tgt):
    """logits (E,B,P,10), tgt (B,P) with -100 at ignored positions."""
    E = logits.shape[0]
    valid = tgt >= 0
    t = tgt.clamp(min=0)
    lp = torch.log_softmax(logits, dim=-1)
    picked = lp.gather(-1, t[None, :, :, None].expand(E, -1, -1, 1)).squeeze(-1)
    picked = picked * valid[None]
    return -(picked.sum(dim=(1, 2)) / valid.sum())      # (E,)


def exact_match(logits, tgt):
    """Fraction of sequences where every predicted digit is right.  (E,)"""
    valid = tgt >= 0
    pred = logits.argmax(-1)                            # (E,B,P)
    ok = (pred == tgt[None]) | (~valid)[None]
    return ok.all(-1).float().mean(-1)                  # (E,)


def margin_stats(logits, tgt):
    """Worst-case (over batch & position) gap between the correct logit and the
    best competitor.  (E,)"""
    valid = tgt >= 0
    t = tgt.clamp(min=0)
    E = logits.shape[0]
    correct = logits.gather(-1, t[None, :, :, None].expand(E, -1, -1, 1)).squeeze(-1)
    other = logits.masked_fill(
        torch.nn.functional.one_hot(t, 10)[None].bool(), float("-inf")
    ).max(-1).values
    m = correct - other
    m = m.masked_fill(~valid[None], float("inf"))
    return m.amin(dim=(1, 2))


def clip_per_member(params, keys, max_norm=1.0):
    with torch.no_grad():
        sq = None
        for k in keys:
            g = params[k].grad
            if g is None:
                continue
            s = (g ** 2).flatten(1).sum(1)
            sq = s if sq is None else sq + s
        if sq is None:
            return
        nrm = sq.sqrt()
        scale = (max_norm / (nrm + 1e-12)).clamp(max=1.0)
        for k in keys:
            g = params[k].grad
            if g is None:
                continue
            g.mul_(scale.view(-1, *([1] * (g.dim() - 1))))


def make_eval(ns, device, per_n=4096, seed=1234, regimes=(0.0, 0.4, 0.9)):
    """Held-out evaluation corners, one per place-count."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for n in ns:
        a, b = data.sample(per_n, n, device, regimes=regimes, gen=g, held_out=True)
        out.append((data.tokens(a, b), data.targets(a, b)))
    return out


@torch.no_grad()
def eval_exact(params, cfg, corners, budget=2 ** 27):
    """Mean exact-match over all corners, per member.  (E,)

    The batch chunk is sized from E and P so wide ensembles do not blow up on
    the (E, B, P, 10) read-out tensor."""
    tot = None
    cnt = 0
    E = params["code"].shape[0]
    for tok, tgt in corners:
        acc = None
        n = tok.shape[0]
        chunk = max(16, min(n, budget // max(1, E * tok.shape[1] * 10)))
        for i in range(0, n, chunk):
            lg = arch.forward(params, cfg, tok[i:i + chunk])
            em = exact_match(lg, tgt[i:i + chunk]) * (min(i + chunk, n) - i)
            acc = em if acc is None else acc + em
        acc = acc / n
        tot = acc if tot is None else tot + acc
        cnt += 1
    return tot / cnt
