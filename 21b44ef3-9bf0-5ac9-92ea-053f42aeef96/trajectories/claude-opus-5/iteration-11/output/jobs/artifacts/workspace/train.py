"""Train E independent members at once (stacked on a leading ensemble axis).

At eleven-ish parameters the result is dominated by initialisation, so the useful
unit of work is a few hundred simultaneous seeds rather than one careful run.
Every tensor in the parameter dict leads with E and Adam is elementwise, so this
is exactly E independent optimisations sharing one data stream.
"""

import argparse
import json
import math
import os
import time

import torch

import adder
from data import Sampler


def log_probs(logits):
    return logits - logits.logsumexp(-1, keepdim=True)


def loss_and_acc(p, cfg, ab, tgt, mask, geo, need_acc=False):
    logits = adder.forward(p, cfg, ab, geo)                       # [E, B, P, 10]
    lp = log_probs(logits)
    E = lp.shape[0]
    t = tgt[None, :, :, None].expand(E, -1, -1, 1)
    nll = -lp.gather(-1, t).squeeze(-1)                            # [E, B, P]
    m = mask[None]
    loss = (nll * m).sum((1, 2)) / m.sum().clamp(min=1)            # [E]
    acc = None
    if need_acc:
        pred = logits.argmax(-1)
        ok = ((pred == tgt[None]) | (m == 0)).all(-1).float()       # [E, B]
        rows = (mask.sum(-1) > 0).float()[None]
        acc = (ok * rows).sum(1) / rows.sum().clamp(min=1)
    return loss, acc


@torch.no_grad()
def evaluate(p, cfg, sets):
    out = []
    for ab, tgt, mask, geo in sets:
        _, acc = loss_and_acc(p, cfg, ab, tgt, mask, geo, need_acc=True)
        out.append(acc)
    return torch.stack(out)                                        # [nsets, E]


def build_eval_sets(sam, places, n_each, device):
    sets = []
    for n in places:
        parts = []
        for spec in (None, 0.4, 0.7, 0.9):
            a, b = sam.sample_digits(n_each, n, spec)
            parts.append((a, b))
        a = torch.cat([x for x, _ in parts], 0)
        b = torch.cat([y for _, y in parts], 0)
        ab, tgt, mask = sam.pack(a, b, held_out=True)
        keep = mask.sum(-1) > 0
        ab, tgt, mask = ab[keep], tgt[keep], mask[keep]
        geo = adder.geometry(n + 2, device)
        sets.append((ab, tgt, mask, geo))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--places", type=int, nargs="+", default=[8, 5, 11, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--out", type=str, default="ckpt.pt")
    ap.add_argument("--cfg", type=str, default="{}")
    ap.add_argument("--sigma", type=float, default=0.0, help="noise on a warm start")
    ap.add_argument("--keep", type=int, default=8, help="members to save")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--rand_const", action="store_true",
                    help="resample the fixed shape constants (alpha, kw, lam) from "
                         "their intended bands every step, so the trained model has "
                         "to be correct across the whole band rather than at one "
                         "tuned point")
    ap.add_argument("--band", type=float, nargs=6,
                    default=[1.5, 12.0, 500.0, 20000.0, 3.0, 16.0],
                    help="alpha_lo alpha_hi kw_lo kw_hi |lam|_lo |lam|_hi")
    ap.add_argument("--anneal", type=float, default=0.0,
                    help="fraction of the run over which the attention sharpness "
                         "constants are ramped from soft (kw0, lam0) to the fixed "
                         "inference values; a curriculum, not a fitted value")
    ap.add_argument("--kw0", type=float, default=8.0)
    ap.add_argument("--alpha_end", type=float, default=0.0,
                    help="if set, ramp the clamp slope from cfg alpha to this value")
    ap.add_argument("--lam0", type=float, default=-2.0)
    ap.add_argument("--merge_theta", type=float, default=0.0,
                    help="fraction of the run over which |theta_neg-theta| is "
                         "hard-capped down to zero")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    cfg = adder.default_cfg(**json.loads(args.cfg))
    E = args.E

    if args.init:
        ck = torch.load(args.init, map_location=device)
        src = ck["params"]
        m = src["code"].shape[0]
        idx = torch.arange(E, device=device) % m
        p = {k: v.to(device)[idx].clone() for k, v in src.items()}
        if args.sigma > 0:
            g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
            for k in p:
                scale = p[k].abs().mean().clamp(min=1e-3)
                noise = torch.randn(p[k].shape, generator=g).to(device)
                p[k] = p[k] + args.sigma * scale * noise
                p[k][:m] = src[k].to(device)          # keep the parents unperturbed
    else:
        p = adder.init_params(cfg, E, device, seed=args.seed)

    # buffers keep their configured constant value for every member
    for k in ("alpha", "e1", "ls", "kw", "lam"):
        if not cfg["free_" + k]:
            p[k] = torch.full((E,), float(cfg[k]), device=device)
    if not cfg["free_rb"]:
        p["rb"] = torch.zeros(E, device=device)
    if not cfg["free_theta_neg"]:
        p["theta_neg"] = p["theta"].clone()
    if cfg["code_fix"]:
        p["code"][:, :cfg["code_fix"]] = 0.0

    train, frozen = adder.split_params(p, cfg)
    for v in train.values():
        v.requires_grad_(True)
    cmask = adder.code_mask(cfg, device)

    opt = torch.optim.AdamW(list(train.values()), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)

    sam = Sampler(device)
    geos = {n: adder.geometry(n + 2, device) for n in args.places}
    eval_sets = build_eval_sets(sam, [8], 8192, device)

    alo, ahi, klo, khi, llo, lhi = args.band

    def loguni(lo, hi):
        u = torch.rand(E, device=device)
        return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))

    def resample():
        frozen["alpha"] = loguni(alo, ahi)
        frozen["kw"] = loguni(klo, khi)
        frozen["lam"] = -loguni(llo, lhi)

    # the value alpha must end at.  This has to happen *before* the corners are
    # built, or the run trains at alpha_end while scoring at the starting alpha.
    alpha_start = float(cfg["alpha"])
    if args.alpha_end > 0:
        cfg = dict(cfg, alpha=args.alpha_end)

    # corners of the band: a member is only kept if it works at every one of them
    corners = [(cfg["alpha"], cfg["kw"], cfg["lam"]),
               (alo, klo, -llo), (ahi, khi, -lhi), (ahi, klo, -lhi),
               (alo, khi, -llo)] if args.rand_const else [
               (cfg["alpha"], cfg["kw"], cfg["lam"])]

    def eval_all(q):
        accs = []
        for al, kv, lv in corners:
            r = dict(q)
            r["alpha"] = torch.full((E,), float(al), device=device)
            r["kw"] = torch.full((E,), float(kv), device=device)
            r["lam"] = torch.full((E,), float(lv), device=device)
            accs.append(evaluate(r, cfg, eval_sets).min(0).values)
        return torch.stack(accs).min(0).values

    def live():
        q = dict(frozen)
        q.update(train)
        if not cfg["free_theta_neg"]:
            q["theta_neg"] = q["theta"]
        return q

    def save(q, accs, path):
        order = accs.argsort(descending=True)[: args.keep]
        out = {k: v.detach()[order].clone() for k, v in q.items()}
        torch.save({"params": out, "cfg": cfg, "acc": accs[order].cpu(),
                    "n_params": adder.n_params(cfg)}, path)
        return order

    t0 = time.time()
    best = -1.0
    cap0 = None
    for step in range(args.steps):
        if args.rand_const:
            resample()
        if args.anneal > 0:
            f = min(1.0, step / (args.anneal * args.steps))
            kv = math.exp(math.log(args.kw0)
                          + f * (math.log(cfg["kw"]) - math.log(args.kw0)))
            lv = args.lam0 + f * (cfg["lam"] - args.lam0)
            frozen["kw"] = torch.full((E,), kv, device=device)
            frozen["lam"] = torch.full((E,), lv, device=device)
        if args.alpha_end > 0:
            f = min(1.0, step / (max(args.anneal, 1e-9) * args.steps))
            av = math.exp(math.log(alpha_start)
                          + f * (math.log(args.alpha_end) - math.log(alpha_start)))
            frozen["alpha"] = torch.full((E,), av, device=device)
        n = args.places[step % len(args.places)]
        ab, tgt, mask = sam.batch(args.batch, n, held_out=False)
        loss, _ = loss_and_acc(live(), cfg, ab, tgt, mask, geos[n])
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        # per-member gradient clipping to 1.0 (members must not affect each other)
        with torch.no_grad():
            sq = torch.zeros(E, device=device)
            for k, v in train.items():
                g = v.grad
                if g is None:
                    continue
                if k == "code":
                    g.mul_(cmask)
                sq += g.reshape(E, -1).pow(2).sum(1)
            scale = (1.0 / sq.sqrt().clamp(min=1.0))
            for v in train.values():
                if v.grad is not None:
                    v.grad.mul_(scale.view(-1, *([1] * (v.dim() - 1))))
        opt.step()
        sched.step()
        with torch.no_grad():
            if cfg["code_fix"]:
                train["code"][:, :cfg["code_fix"]] = 0.0
            if args.merge_theta > 0 and cfg["free_theta_neg"]:
                # hard cap on the raw gap, applied after the optimiser step: an
                # interpolation inside the forward pass is only a change of
                # variables and the gap simply grows to compensate.
                if cap0 is None:
                    cap0 = (train["theta_neg"] - train["theta"]).abs().max().item() + 1e-6
                frac = min(1.0, step / (args.merge_theta * args.steps))
                cap = cap0 * (1.0 - frac)
                d = train["theta_neg"] - train["theta"]
                train["theta_neg"].copy_(train["theta"] + d.clamp(-cap, cap))

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            if args.anneal > 0 or args.rand_const:
                fin = dict(alpha=args.alpha_end or cfg["alpha"], kw=cfg["kw"],
                           lam=cfg["lam"])
                for k, v in fin.items():
                    frozen[k] = torch.full((E,), float(v), device=device)

            mx = eval_all(live())
            top = mx.topk(min(5, E)).values
            best = max(best, top[0].item())
            print(f"step {step+1:6d}  loss {loss.mean().item():.4f}  "
                  f"top5 {[round(v.item(), 5) for v in top]}  "
                  f"n>=0.999: {(mx >= 0.999).sum().item()}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            save(live(), mx, args.out + ".part")

    if args.rand_const:
        for k, v in zip(("alpha", "kw", "lam"),
                        (cfg["alpha"], cfg["kw"], cfg["lam"])):
            frozen[k] = torch.full((E,), float(v), device=device)
    accs = eval_all(live())
    order = save(live(), accs, args.out)
    print("saved", args.out, "best", accs[order[0]].item(),
          "n_params", adder.n_params(cfg))


if __name__ == "__main__":
    main()
