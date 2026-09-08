"""The wide parent: the same one-block architecture with nothing pinned.

The shipped model fixes the gate slope, the key contrast, the recency slope and
the residual axis.  Those constants are what make the final form small, but they
also make it untrainable from scratch: with a saturated gate the two thresholds
have no gradient at all, so a cold member can only ever use the thresholds it
was born with.

Here every one of those is a free weight and the residual stream can have more
than one channel, so the gate starts soft and its thresholds can actually move.
Members trained here are reduced to the shipped form by reduce.py.

Parameters per member (C channels, U gate units):
    code (10, C)  bw (U, C)  bb (U)  kw (U)  vw (U)  uA (C)  uB (C)  lam  lsr
"""

import time

import torch

import lab


def init(E, device, C=2, U=2, gen=None):
    """Random members.

    The gate weights start small so the clamp is soft and its thresholds have
    gradient -- that is the whole reason this form exists.  The key weights and
    the recency slope start a few units wide instead, because attention only
    carries usable gradient once one place actually wins the softmax.
    """
    def r(*shape, s=1.0):
        return torch.randn(E, *shape, device=device, generator=gen) * s
    p = {
        "code": r(10, C, s=1.0),
        "bw": r(U, C, s=1.0),
        "bb": r(U, s=1.0),
        "kw": r(U, s=4.0),
        "vw": r(U, s=1.0),
        "uA": r(C, s=0.5),
        "uB": r(C, s=0.5),
        "lam": -torch.rand(E, device=device, generator=gen) * 4.0 - 1.0,
        "lsr": torch.zeros(E, device=device),
    }
    return {k: v.requires_grad_(True) for k, v in p.items()}


def pad_pairs(pairs):
    """Add a (0, 0) place at each end of the token sequence."""
    B, n, _ = pairs.shape
    z = pairs.new_zeros(B, 1, 2)
    return torch.cat([z, pairs, z], dim=1)


def forward(p, pairs, square=True, temp=1.0, norm=True):
    """(E, B, P, 10) logits for shared token batch `pairs` (B, n, 2)."""
    code = p["code"]                                          # (E, 10, C)
    E, _, C = code.shape
    tok = pad_pairs(pairs)                                    # (B, P, 2)
    x = code[:, tok[..., 0]] + code[:, tok[..., 1]]           # (E, B, P, C)
    P = x.shape[2]

    g = torch.clamp(torch.einsum("ebpc,euc->ebpu", x, p["bw"])
                    + p["bb"][:, None, None, :], 0.0, 1.0)    # (E, B, P, U)
    key = torch.einsum("ebpu,eu->ebp", g, p["kw"])
    val = torch.einsum("ebpu,eu->ebp", g, p["vw"])

    gap, strict, future = lab.masks(P, x.device, x.dtype)
    scores = key.unsqueeze(2) + p["lam"].view(E, 1, 1, 1) * gap
    w_in = torch.softmax(scores + strict, dim=-1)
    w_out = torch.softmax(scores + future, dim=-1)
    cin = torch.einsum("ebij,ebj->ebi", w_in, val)
    cout = torch.einsum("ebij,ebj->ebi", w_out, val)

    z = (x + cin[..., None] * p["uA"][:, None, None, :]
         + cout[..., None] * p["uB"][:, None, None, :])       # (E, B, P, C)
    d = z.unsqueeze(3) - code[:, None, None, :, :]            # (E, B, P, 10, C)
    dist = d.pow(2).sum(-1) if square else d.abs().sum(-1)
    if norm:
        s = code.reshape(E, -1).std(dim=1).clamp_min(1e-4).view(E, 1, 1, 1)
        dist = dist / (s.pow(2) if square else s)
    return -dist * temp * torch.exp(p["lsr"].clamp(-6, 6)).view(E, 1, 1, 1)


def loss_and_acc(p, pairs, y, temp, square=False):
    logits = forward(p, pairs, square=square, temp=temp)[:, :, 1:, :]
    logp = torch.log_softmax(logits, dim=-1)
    tgt = y[None, :, :, None].expand(logp.shape[0], -1, -1, 1)
    loss = -logp.gather(-1, tgt).squeeze(-1).mean(dim=(1, 2))
    with torch.no_grad():
        ok = (logits.argmax(-1) == y[None]).all(-1).float().mean(1)
    return loss, ok


@torch.no_grad()
def accuracy(p, device, widths=(8,), split="eval", B=2048, chunk=4096):
    E = p["code"].shape[0]
    tot = torch.zeros(E, device=device)
    for w in widths:
        pairs, y = lab.batch(B, w, device, split=split)
        for i in range(0, E, chunk):
            sub = {k: v[i:i + chunk] for k, v in p.items()}
            lg = forward(sub, pairs)[:, :, 1:, :]
            tot[i:i + chunk] += (lg.argmax(-1) == y[None]).all(-1).float().mean(1)
    return tot / len(widths)


def parse_stages(text):
    """"1:0.3,2:0.2,8:0.5" -> [(max_width, fraction_of_steps), ...]."""
    out = []
    for part in text.split(","):
        w, f = part.split(":")
        out.append((int(w), float(f)))
    return out


def train(p, device, steps, lr, stages, B, temp, clip=1.0, log_every=500,
          nocarry_frac=0.0, pct_start=0.15, track_best=0, keep_best_from=0.0):
    """Width-staged training.

    Stage k trains on widths 1..w_k, sampled uniformly.  Widening only after the
    narrower problem is solved is what makes this trainable: a member that has
    not yet found a digit code gets nothing but noise from an eight-place carry
    chain, and the code is what everything else is built on.
    """
    E = p["code"].shape[0]
    opt = torch.optim.AdamW(list(p.values()), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=pct_start)
    bounds, acc_f = [], 0.0
    for w, f in stages:
        acc_f += f
        bounds.append((w, int(acc_f / sum(s[1] for s in stages) * steps)))
    best = {k: v.detach().clone() for k, v in p.items()} if track_best else None
    best_ok = torch.full((E,), -1.0, device=device)
    t0 = time.time()
    si = 0
    for step in range(steps):
        while si < len(bounds) - 1 and step >= bounds[si][1]:
            si += 1
        wmax = bounds[si][0]
        w = int(torch.randint(1, wmax + 1, (1,)).item())
        nc = max(0.0, 1.0 - step / (nocarry_frac * steps)) if nocarry_frac > 0 else 0.0
        pairs, y = lab.batch(B, w, device, split="train", nocarry=nc)
        loss, ok = loss_and_acc(p, pairs, y, temp)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        sq = sum(v.grad.reshape(E, -1).pow(2).sum(1) for v in p.values())
        sc = (clip / (sq.sqrt() + 1e-8)).clamp(max=1.0)
        for v in p.values():
            v.grad.mul_(sc.view(E, *([1] * (v.dim() - 1))))
        opt.step()
        sched.step()
        # Snapshotting only makes sense once the hardest widths are in play;
        # a member that is perfect on one-place problems is not a candidate.
        if (track_best and step >= keep_best_from * steps
                and (step % track_best == 0 or step == steps - 1) and w >= wmax):
            with torch.no_grad():
                better = ok > best_ok
                best_ok = torch.where(better, ok, best_ok)
                for k, v in p.items():
                    best[k] = torch.where(better.view(E, *([1] * (v.dim() - 1))),
                                          v.detach(), best[k])
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:6d} w<={wmax} loss {loss.mean().item():.4f} "
                  f"best {ok.max().item():.4f} n>=0.99 {(ok >= 0.99).sum().item():5d} "
                  f"{time.time() - t0:6.1f}s", flush=True)
    return best if track_best else {k: v.detach() for k, v in p.items()}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=2048)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--U", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--temp", type=float, default=20.0)
    ap.add_argument("--stages", default="1:0.25,2:0.15,3:0.15,5:0.15,8:0.30")
    ap.add_argument("--nocarry_frac", type=float, default=0.10)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/wide.pt")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed)
    p = init(a.E, dev, C=a.C, U=a.U)
    print(f"wide parent: E={a.E} C={a.C} U={a.U}")
    p = train(p, dev, a.steps, a.lr, parse_stages(a.stages), a.B, a.temp,
              nocarry_frac=a.nocarry_frac, track_best=100, keep_best_from=0.6)
    acc = accuracy(p, dev, widths=(8, 5, 3, 12))
    order = acc.argsort(descending=True)
    print("  acc quantiles", [round(acc.quantile(q).item(), 4)
                              for q in (0.5, 0.9, 0.99, 1.0)])
    print(f"  members exact on all widths: {lab.is_exact(acc).sum().item()} / {a.E}")
    keep = order[:a.keep]
    torch.save({"params": {k: v.detach()[keep].cpu() for k, v in p.items()},
                "acc": acc[keep].cpu(), "C": a.C, "U": a.U}, a.out)
    print("  wrote", a.out)
