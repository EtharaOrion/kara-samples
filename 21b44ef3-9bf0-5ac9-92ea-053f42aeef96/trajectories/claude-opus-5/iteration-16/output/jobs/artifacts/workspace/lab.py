"""Training / evaluation utilities for the ensemble-batched adder."""

import math
import time
import torch

import arch
import data


def logits_loss(p, a, b, tgt, mask):
    """Cross-entropy per member, computed with an explicit gather (torch's
    nll_loss kernel dominates runtime at large E)."""
    logits = arch.forward(p, a, b)                      # [E,B,P,10]
    logp = logits.log_softmax(-1)
    E = logits.shape[0]
    t = tgt[None].expand(E, -1, -1).unsqueeze(-1)
    ll = logp.gather(-1, t).squeeze(-1)                 # [E,B,P]
    m = mask[None].to(ll.dtype)
    loss = -(ll * m).sum((1, 2)) / m.sum()
    return logits, loss


def geo_margin(p, a, b, tgt, mask):
    """Read-out margin in residual units: how far the residual sits inside the
    correct prototype's Voronoi cell.  Positive means the digit is decoded
    correctly; this is exactly the quantity the whole-domain certificate bounds."""
    _, y = arch.forward(p, a, b, want_y=True)                 # [E,B,P,C]
    code = p["code"]                                          # [E,10,C]
    # +eps: the pad position sits exactly on prototype 0, and sqrt has an
    # infinite gradient there, which otherwise poisons the whole backward pass.
    dist = ((y[:, :, :, None, :] - code[:, None, None, :, :]).pow(2).sum(-1)
            + 1e-12).sqrt()
    E = dist.shape[0]
    t = tgt[None].expand(E, -1, -1).unsqueeze(-1)
    d_true = dist.gather(-1, t).squeeze(-1)                   # [E,B,P]
    d_other = dist.scatter(-1, t, float("inf")).amin(-1)
    m = 0.5 * (d_other - d_true)
    return torch.where(mask[None], m, torch.full_like(m, float("inf")))


def margin_loss(p, a, b, tgt, mask, target=0.40, kappa=12.0):
    """Push every position's read-out margin up to `target` residual units.
    Unlike cross-entropy this does not score inflating the code scale, which
    is pinned by the residual-scale gauge in the shipped model."""
    m = geo_margin(p, a, b, tgt, mask)
    pen = torch.nn.functional.softplus(kappa * (target - m)) / kappa
    pen = torch.where(torch.isinf(m), torch.zeros_like(pen), pen)
    return pen.sum((1, 2)) / mask.sum()


_PA = torch.arange(10)[:, None].expand(10, 10).reshape(-1)
_PB = torch.arange(10)[None, :].expand(10, 10).reshape(-1)


def sat_slack(p):
    """How far each clamp unit is from leaving saturation, over all 100 digit
    pairs.  Positive everywhere means the gate is exactly 0 or exactly 1 for
    every possible place, which is what makes the whole-domain certificate
    exact -- and what stops a knee from sitting on top of a carry class where
    float32 rounding could flip it."""
    dev = p["code"].device
    x = p["code"][:, _PA.to(dev)] + p["code"][:, _PB.to(dev)]      # [E,100,C]
    e = torch.einsum("euc,epc->epu", p["Wb"], x) + p["bb"][:, None, :]
    return torch.maximum(-e, e - 1.0)                               # [E,100,U]


def sat_penalty(p, target):
    return torch.relu(target - sat_slack(p)).mean((1, 2))


@torch.no_grad()
def evaluate(p, n, device, gen, n_batch=8, B=1024, held_out=True, mix=(0.35, 0.40, 0.25),
             chunk=None):
    """Exact-match accuracy per member, plus the worst read-out margin seen."""
    E = p["code"].shape[0]
    hits = torch.zeros(E, device=device)
    tot = 0
    worst = torch.full((E,), float("inf"), device=device)
    for _ in range(n_batch):
        a, b = data.batch(B, n, device, gen, held_out=held_out, mix=mix)
        ap, bp = data.pad(a, b)
        tgt, mask = data.targets(a, b)
        for lo in range(0, E, chunk or E):
            hi = lo + (chunk or E)
            sub = {k: v[lo:hi] for k, v in p.items()}
            logits = arch.forward(sub, ap, bp)
            pred = logits.argmax(-1)
            ok = ((pred == tgt[None]) | ~mask[None]).all(-1)
            hits[lo:hi] += ok.float().sum(-1)
            m = geo_margin(sub, ap, bp, tgt, mask)
            m = torch.where(torch.isinf(m), torch.full_like(m, 1e9), m)
            worst[lo:hi] = torch.minimum(worst[lo:hi], m.amin((1, 2)))
        tot += ap.shape[0]
    return hits / tot, worst


def per_member_clip(params, keys, max_norm):
    sq = None
    for k in keys:
        g = params[k].grad
        if g is None:
            continue
        s = g.reshape(g.shape[0], -1).pow(2).sum(1)
        sq = s if sq is None else sq + s
    if sq is None:
        return
    nrm = sq.sqrt()
    scale = (max_norm / (nrm + 1e-12)).clamp(max=1.0)
    for k in keys:
        g = params[k].grad
        if g is None:
            continue
        g.mul_(scale.reshape(-1, *([1] * (g.dim() - 1))))


def train(p, train_keys, steps, device, gen, lr=0.012, B=512,
          places=(8, 5, 11, 3), clip=1.0, pct_start=0.15, log_every=500,
          eval_every=1000, eval_n=8, tag="", snap=True, mix=(0.35, 0.40, 0.25),
          eval_chunk=None, grad_mask=None, objective="ce", m_target=0.40,
          score_w=0.001, eval_batches=4, sat_w=0.0, sat_target=1.5):
    """Train E members simultaneously.  Returns (params, best_snapshot, history)."""
    for k, v in p.items():
        v.requires_grad_(k in train_keys)
    tensors = [p[k] for k in train_keys]
    opt = torch.optim.AdamW(tensors, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=pct_start)
    E = p["code"].shape[0]
    best_score = torch.full((E,), -1e9, device=device)
    best = {k: v.detach().clone() for k, v in p.items()}
    hist = []
    t0 = time.time()

    def snapshot(step):
        nonlocal best_score
        acc, worst = evaluate(p, eval_n, device, gen, n_batch=eval_batches, B=512,
                              chunk=eval_chunk)
        score = acc + score_w * worst.clamp(-5, 5)
        if sat_w:
            score = score + score_w * sat_slack(p).amin((1, 2)).clamp(-5, sat_target)
        if snap:
            imp = score > best_score
            if imp.any():
                best_score = torch.where(imp, score, best_score)
                for k in p:
                    m = imp.reshape(-1, *([1] * (p[k].dim() - 1)))
                    best[k] = torch.where(m, p[k].detach(), best[k])
        return acc, worst

    snapshot(-1)   # the starting point is a candidate too

    for step in range(steps):
        n = places[step % len(places)]
        a, b = data.batch(B, n, device, gen, held_out=False, mix=mix)
        ap, bp = data.pad(a, b)
        tgt, mask = data.targets(a, b)
        if objective == "margin":
            loss = margin_loss(p, ap, bp, tgt, mask, target=m_target)
        else:
            _, loss = logits_loss(p, ap, bp, tgt, mask)
        if sat_w:
            loss = loss + sat_w * sat_penalty(p, sat_target)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        if grad_mask:
            for k, m in grad_mask.items():
                if p[k].grad is not None:
                    p[k].grad.mul_(m)
        per_member_clip(p, train_keys, clip)
        opt.step()
        sched.step()

        if (step + 1) % eval_every == 0 or step == steps - 1:
            # exact-match first, worst margin as tiebreak once a member is perfect
            acc, worst = snapshot(step)
            hist.append((step + 1, float(loss.mean()), float(acc.max()),
                         int((acc >= 0.9999).sum())))
            if (step + 1) % log_every == 0 or step == steps - 1:
                ex = acc >= 0.9999
                bw = float(worst[ex].max()) if bool(ex.any()) else float("nan")
                print(f"[{tag}] step {step+1}/{steps} loss {loss.mean():.4f} "
                      f"best_acc {acc.max():.4f} n>=0.9999 {int(ex.sum())} "
                      f"best_margin {bw:.4f} ({time.time()-t0:.0f}s)", flush=True)
    for v in p.values():
        v.requires_grad_(False)
    return p, best, hist


def stack_select(p, idx):
    return {k: v[idx].clone() for k, v in p.items()}


def repeat_members(p, E):
    return {k: v.repeat(E, *([1] * (v.dim() - 1))).clone() for k, v in p.items()}
