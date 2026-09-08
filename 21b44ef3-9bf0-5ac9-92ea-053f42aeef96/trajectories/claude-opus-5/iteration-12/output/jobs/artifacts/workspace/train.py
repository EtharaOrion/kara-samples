"""Train the 12-parameter adder from random init.

Seed variance dominates completely at this parameter count, so we train E
independent members at once: `torch.func.stack_module_state` puts an ensemble
axis on every parameter and `vmap` runs E copies of the *same* forward pass
over a shared batch.  Members never interact - Adam is elementwise and the
gradient clip is per member - so each one is an ordinary independent training
run, just executed alongside the others.

Nothing here is imported by submission.py; the trainer only writes weights.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

import data
from adder import DigitPairAdder


class DiscoveryAdder(DigitPairAdder):
    """Ablation used only to check what the model finds on its own.

    Promotes the fixed read-in vectors and attention scales to parameters, so
    we can see whether training *discovers* the structure that the shipped
    architecture declares up front.  Never shipped.
    """

    def __init__(self):
        super().__init__()
        for name in ("key_w", "val_w", "dist_bias", "carry_w"):
            t = self._buffers.pop(name)
            setattr(self, name, nn.Parameter(t.clone()))


class FreeZeroAdder(DigitPairAdder):
    """Ablation: code[0] is a 13th parameter instead of a pinned origin.

    The shipped model pins code[0] at the origin of the read-out line.  This
    variant lets training choose it, to check that what training picks is the
    pinned value -- i.e. that pinning it costs the model nothing it would have
    had to learn differently.  Never shipped.
    """

    def __init__(self):
        super().__init__()
        t = self._buffers.pop("code_zero")
        self.code_zero = nn.Parameter(t.clone())


ARCHES = {"base": DigitPairAdder, "discovery": DiscoveryAdder, "freezero": FreeZeroAdder}


def make_members(n_ens, device, seed, scale_code, scale_bank, cls):
    members = []
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_ens):
        m = cls()
        with torch.no_grad():
            m.code_free.copy_(torch.randn(9, generator=gen) * scale_code)
            m.bank_bias.copy_(torch.randn(2, generator=gen) * scale_bank)
            m.fold.copy_(torch.randn((), generator=gen))
            if isinstance(m.code_zero, nn.Parameter):
                m.code_zero.copy_(torch.randn(1, generator=gen) * scale_code)
            if isinstance(m, DiscoveryAdder):
                m.key_w.copy_(torch.randn(2, generator=gen))
                m.val_w.copy_(torch.randn(2, generator=gen))
                m.dist_bias.copy_(-torch.randn((), generator=gen).abs())
                m.carry_w.copy_(1.0 + 0.3 * torch.randn((), generator=gen))
        members.append(m.to(device))
    return members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--pct_start", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--places", type=str, default="8,5,11,3")
    ap.add_argument("--sched", type=str, default="",
                    help="place-count curriculum, e.g. '0:1;2000:1,2;5000:2,3;9000:8,5,11,3'")
    ap.add_argument("--scale_code", type=float, default=1.0)
    ap.add_argument("--scale_bank", type=float, default=1.0)
    ap.add_argument("--ls", type=float, default=1.0, help="read-out temperature used in the loss only")
    ap.add_argument("--tau0", type=float, default=1.0, help="initial attention-logit scale")
    ap.add_argument("--beta0", type=float, default=1.0, help="initial clamp-bank slope scale")
    ap.add_argument("--beta_frac", type=float, default=0.5)
    ap.add_argument("--tau_frac", type=float, default=0.4, help="fraction of steps spent ramping tau to 1")
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--eval_n", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batch", type=int, default=16384)
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--arch", type=str, default="base", choices=sorted(ARCHES))
    ap.add_argument("--init_from", type=str, default="",
                    help="checkpoint whose members seed this run (phase-2 training)")
    ap.add_argument("--reinit", type=str, default="bank_bias",
                    help="comma-separated params to re-randomise after seeding")
    ap.add_argument("--init_jitter", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="ckpt")
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    places = [int(p) for p in args.places.split(",")]
    stages = []
    if args.sched:
        for part in args.sched.split(";"):
            at, pl = part.split(":")
            stages.append((int(at), [int(q) for q in pl.split(",")]))
        stages.sort()

    def places_at(step):
        if not stages:
            return places
        cur = stages[0][1]
        for at, pl in stages:
            if step >= at:
                cur = pl
        return cur
    cls = DiscoveryAdder if args.discovery else ARCHES[args.arch]

    members = make_members(args.ens, device, args.seed, args.scale_code, args.scale_bank, cls)
    params, buffers = stack_module_state(members)
    params = {k: v.detach().clone() for k, v in params.items()}

    if args.init_from:
        # Phase-2 training: seed every member from the members of an earlier
        # run of this same pipeline, then re-randomise the named parameters so
        # the run explores fresh basins for them.  Everything stays trained.
        src = torch.load(args.init_from, map_location=device, weights_only=False)
        sp = src["params"]
        nsrc = next(iter(sp.values())).shape[0]
        idx = torch.arange(args.ens, device=device) % nsrc
        for k in params:
            if k in sp:
                params[k] = sp[k].to(device)[idx].clone()
        g0 = torch.Generator(device=device).manual_seed(args.seed + 777)
        if args.init_jitter > 0:
            for k in params:
                params[k] = params[k] + args.init_jitter * torch.randn(
                    params[k].shape, device=device, generator=g0)
        names = [q for q in args.reinit.split(",") if q]
        if "bank_bias" in names:
            # spread the two clamp knees uniformly over the range the bank's
            # input actually takes, which is the standard way to initialise a
            # threshold: it uses the scale of the input, not the answer.
            code = torch.cat([torch.zeros(args.ens, 1, device=device), params["code_free"]], 1)
            xmax = 2 * code.max(1).values
            knee = torch.rand(args.ens, 2, device=device, generator=g0) * xmax[:, None]
            bw = buffers["bank_w"][0] if buffers["bank_w"].dim() > 1 else buffers["bank_w"]
            params["bank_bias"] = -bw.to(device) * knee
        if "fold" in names:
            params["fold"] = -20.0 * torch.rand(args.ens, device=device, generator=g0)
        if "code_free" in names:
            params["code_free"] = torch.randn(args.ens, 9, device=device, generator=g0)
    params = {k: v.contiguous().requires_grad_(True) for k, v in params.items()}
    base = cls().to("meta")

    def fwd(p, b, da, db, tau, beta):
        return functional_call(base, (p, b), (da, db), {"tau": tau, "beta": beta})

    batched = vmap(fwd, in_dims=(0, 0, None, None, None, None))

    opt = torch.optim.AdamW(list(params.values()), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=args.pct_start)
    gen = torch.Generator(device=device).manual_seed(args.seed + 12345)
    egen = torch.Generator(device=device).manual_seed(999)

    def loss_and_acc(da, db, tgt, keep, tau, beta):
        logits = batched(params, buffers, da, db, tau, beta)          # (E,B,P,10)
        # `ls` is a temperature on the read-out *loss* only.  The model's
        # prediction is argmax over -(y-code[d])^2, which is invariant to any
        # positive scale, so this changes the gradient landscape and nothing
        # about what the model computes.
        lp = (logits * args.ls).log_softmax(-1)
        pick = lp.gather(-1, tgt.expand(args.ens, -1, -1).unsqueeze(-1)).squeeze(-1)
        posw = torch.ones(tgt.shape[-1], device=tgt.device)
        posw[0] = 0.0                                            # position 0 is unused
        w = keep[None, :, None].float() * posw[None, None, :]
        loss = -(pick * w).sum((1, 2)) / w.sum().clamp(min=1)
        return loss, logits

    @torch.no_grad()
    def evaluate_big(n, total, chunk=8192):
        """Held-out accuracy over `total` samples, chunked so a large ensemble
        times a large batch never has to be materialised at once."""
        hit = torch.zeros(args.ens, device=device)
        seen = 0
        while seen < total:
            a, m = evaluate(n, chunk)
            if m == 0:                       # empty held-out slice: no signal
                continue
            hit += a * m
            seen += m
        return hit / max(seen, 1), seen

    @torch.no_grad()
    def evaluate(n, batch):
        da, db, tgt, bucket = data.sample(batch, n, device, egen, force_msb=(n > 1))
        sel = bucket == 0
        da, db, tgt = da[sel], db[sel], tgt[sel]
        logits = batched(params, buffers, da, db, 1.0, 1.0)
        pred = logits.argmax(-1)
        ok = (pred[..., 1:] == tgt[..., 1:].unsqueeze(0)).all(-1)
        return ok.float().mean(-1), sel.sum().item()

    # Per-member early stopping: a member that lands in a good basin has zero
    # gradient on its knees but not on its code, so later steps can still walk
    # it out.  Keep each member's best-ever weights, judged on the held-out
    # split only.
    best_acc = torch.zeros(args.ens, device=device)
    best_params = {k: v.detach().clone() for k, v in params.items()}

    @torch.no_grad()
    def snapshot():
        nonlocal best_acc
        accs, nheld = evaluate(args.eval_n, args.eval_batch)
        upd = accs > best_acc
        best_acc = torch.where(upd, accs, best_acc)
        for k, v in params.items():
            m = upd.reshape(-1, *([1] * (v.dim() - 1)))
            best_params[k] = torch.where(m, v.detach(), best_params[k])
        return accs, nheld

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    best = -1.0
    hist = []
    snapshot()                       # capture the random init itself
    for step in range(args.steps):
        pl = places_at(step)
        n = pl[step % len(pl)]
        if args.tau0 < 1.0:
            f = min(1.0, step / max(1, int(args.tau_frac * args.steps)))
            tau = math.exp(math.log(args.tau0) * (1.0 - f))
        else:
            tau = 1.0
        if args.beta0 < 1.0:
            fb = min(1.0, step / max(1, int(args.beta_frac * args.steps)))
            beta = math.exp(math.log(args.beta0) * (1.0 - fb))
        else:
            beta = 1.0
        da, db, tgt, bucket = data.sample(args.batch, n, device, gen, force_msb=(n > 1))
        keep = bucket != 0                                       # train on 15/16 of the space
        loss, _ = loss_and_acc(da, db, tgt, keep, tau, beta)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        with torch.no_grad():                                    # per-member grad clip
            sq = torch.zeros(args.ens, device=device)
            for v in params.values():
                g = v.grad
                sq += g.reshape(args.ens, -1).pow(2).sum(-1) if g.dim() > 1 else g.pow(2)
            scale = (1.0 / sq.sqrt().clamp(min=1e-12)).clamp(max=1.0)
            for v in params.values():
                v.grad.mul_(scale.reshape(-1, *([1] * (v.dim() - 1))))
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            accs, nheld = snapshot()
            top = best_acc.max().item()
            nperf = (best_acc > 0.9999).sum().item()
            hist.append({"step": step + 1, "best": top, "n_perfect": nperf,
                         "loss": loss.min().item(), "tau": tau})
            print(f"[{args.tag}] step {step+1:6d} n {n:2d} tau {tau:5.3f} beta {beta:5.3f} loss {loss.min().item():.5f} "
                  f"best_heldout {top:.5f} members>=99.99% {nperf:3d}/{args.ens} "
                  f"({time.time()-t0:.0f}s, held={nheld})", flush=True)
            if top > best:
                best = top
            os.makedirs(args.out, exist_ok=True)
            sel = best_acc.argsort(descending=True)[: min(args.keep, args.ens)]
            torch.save({"cfg": vars(args), "acc": best_acc[sel].cpu(),
                        "params": {k: v[sel].cpu() for k, v in best_params.items()},
                        "buffers": {k: v.detach()[0].cpu() for k, v in buffers.items()},
                        "hist": hist},
                       os.path.join(args.out, f"{args.tag}.pt"))

    snapshot()
    params = best_params                      # ship each member's best-ever weights
    accs, _ = evaluate_big(args.eval_n, 262144)
    order = accs.argsort(descending=True)
    keep_n = min(args.keep, args.ens)
    sel = order[:keep_n]
    ck = {"cfg": vars(args),
          "acc": accs[sel].cpu(),
          "params": {k: v.detach()[sel].cpu() for k, v in params.items()},
          "buffers": {k: v.detach()[0].cpu() for k, v in buffers.items()},
          "hist": hist}
    path = os.path.join(args.out, f"{args.tag}.pt")
    torch.save(ck, path)
    print(f"[{args.tag}] saved {path}; top accs {[round(a,6) for a in accs[sel][:8].tolist()]}")
    print(json.dumps({"tag": args.tag, "best": accs.max().item(),
                      "n_perfect": int((accs > 0.9999).sum().item())}))


if __name__ == "__main__":
    main()
