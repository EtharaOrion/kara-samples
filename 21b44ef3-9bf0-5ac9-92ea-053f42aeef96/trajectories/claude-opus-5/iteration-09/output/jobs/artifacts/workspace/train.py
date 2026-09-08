"""Train E independent members of DigitAdder simultaneously (seed lottery).

Every parameter carries a leading ensemble axis, so one forward pass trains E
models on the same batch.  Below ~100 parameters a cold start never finds the
carry mechanism, so small models are reached by warm-starting from a trained
parent (--init) and optionally annealing a structure to a constant (--shrink),
which is enforced as a hard cap on the raw residual after every optimiser step.
"""
import argparse
import json
import math
import os
import time

import torch

from model_src import DigitAdder, default_cfg
from data import Sampler


def parse_shrink(spec):
    """'name.idx=value' -> (name, index tuple over axes 1.., value)."""
    body, _, val = spec.partition("=")
    val = float(val) if val else 0.0
    name, _, idx = body.partition(".")
    sel = ()
    if idx:
        sel = tuple(slice(None) if t == "*" else int(t) for t in idx.split(","))
    return name, sel, val


def sel_of(p, sel):
    return (slice(None),) + sel if sel else (slice(None),)


def loss_and_acc(model, a, b, tgt):
    logits = model(a, b)[:, :, 1:, :]                     # (E,B,n+1,10)
    lp = torch.log_softmax(logits.float(), dim=-1)
    t = tgt[None, :, :, None].expand(lp.shape[0], -1, -1, 1)
    nll = -lp.gather(-1, t).squeeze(-1)
    loss = nll.mean(dim=(1, 2))                           # (E,)
    ok = (logits.argmax(-1) == tgt[None]).all(-1).float().mean(1)
    return loss, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--cfg_file", default="", help="json file overriding --cfg")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--places", default="8")
    ap.add_argument("--init", default="")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--new_sigma", type=float, default=0.05,
                    help="init scale for parameters absent from --init")
    ap.add_argument("--pick", type=int, default=-1, help="member of --init to clone")
    ap.add_argument("--shrink", default="", help="comma list of name.idx=value")
    ap.add_argument("--shrink_frac", type=float, default=0.45)
    ap.add_argument("--shrink_start", type=float, default=0.15,
                    help="fraction of training before the anneal begins")
    ap.add_argument("--freeze", default="")
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=int, default=1000)
    ap.add_argument("--probe", type=int, default=0, help="dump member 0 internals")
    ap.add_argument("--morph", default="", help="a,b: drive act_t 0->1 over this "
                    "fraction of training (needs cfg act='morph')")
    ap.add_argument("--warm", type=float, default=0.15)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    raw = open(args.cfg_file).read() if args.cfg_file else args.cfg
    cfg = default_cfg(json.loads(raw))
    cfg["E"] = args.E
    cfg["seed"] = args.seed
    model = DigitAdder(cfg).to(dev)

    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        src = ck["state"]
        for name, p in model.named_parameters():
            if name not in src:
                with torch.no_grad():
                    p.mul_(args.new_sigma)
                print(f"  [init] no source for {name}; small random init "
                      f"(scale {args.new_sigma})")
                continue
            s = src[name].to(dev)
            if args.pick >= 0:
                s = s[args.pick: args.pick + 1]
            if s.shape[1:] != p.shape[1:]:
                print(f"  [init] shape mismatch {name}: {tuple(s.shape)} vs {tuple(p.shape)}")
                continue
            rep = s.expand(args.E, *s.shape[1:]) if s.shape[0] == 1 else s[: args.E]
            with torch.no_grad():
                p.copy_(rep)
                if args.sigma > 0:
                    p[1:] += torch.randn_like(p[1:]) * args.sigma * p[1:].abs().mean().clamp(min=1e-3)

    params = [p for p in model.parameters() if p.requires_grad]
    frozen = set(args.freeze.split(",")) if args.freeze else set()
    for n_, p in model.named_parameters():
        if n_ in frozen:
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in model.parameters()) // args.E
    print(f"config {cfg}\nparameters per member: {n_par}")

    shrinks, ties, shifts = [], [], []
    if args.shrink:
        named = dict(model.named_parameters())
        for spec in args.shrink.split(";"):
            if "~" in spec or ">" in spec:        # anneal two tensors together
                sep = "~" if "~" in spec else ">"
                x, y = spec.split(sep)
                pa, pb = named[x], named[y]
                ties.append((pa, pb, (pa.data - pb.data).abs().clone(), sep))
                print(f"  [tie]    {spec}  start gap max {ties[-1][2].max().item():.4f} "
                      f"median {ties[-1][2].median().item():.4f}")
                continue
            if "@" in spec:                       # drive one row to a constant by
                body, _, v = spec.partition("=")   # translating the whole tensor,
                nm, _, j = body.partition("@")     # which leaves its spacing intact
                p, j, v = named[nm], int(j), float(v)
                shifts.append((p, j, v, (p.data[:, j] - v).abs().clone()))
                print(f"  [shift]  {spec}  start residual max "
                      f"{shifts[-1][3].max().item():.4f}")
                continue
            nm, sel, val = parse_shrink(spec)
            p = named[nm]
            s = sel_of(p, sel)
            shrinks.append((p, s, val, (p.data[s] - val).abs().clone()))
            print(f"  [shrink] {spec}  start residual max "
                  f"{shrinks[-1][3].max().item():.4f} "
                  f"median {shrinks[-1][3].median().item():.4f}")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=args.warm)
    smp = Sampler(dev)
    places = [int(x) for x in args.places.split(",")]

    best = None
    t0 = time.time()
    for step in range(args.steps):
        n = places[step % len(places)]
        a, b, tgt = smp.batch(args.batch, n=n)
        loss, _ = loss_and_acc(model, a, b, tgt)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        if args.clip > 0:                                  # per-member clipping
            sq = torch.zeros(args.E, device=dev)
            for p in params:
                if p.grad is not None:
                    sq += p.grad.reshape(args.E, -1).pow(2).sum(1)
            scale = (args.clip / (sq.sqrt() + 1e-9)).clamp(max=1.0)
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale.reshape(-1, *([1] * (p.dim() - 1))))
        opt.step()
        sched.step()
        if shrinks or ties or shifts:
            s0 = args.shrink_start * args.steps
            f = (step - s0) / max(1.0, args.shrink_frac * args.steps - s0)
            f = min(1.0, max(0.0, f))
            f = f * f * (3.0 - 2.0 * f)      # smoothstep: no jolt at either end
            with torch.no_grad():
                for p, s, val, r0 in shrinks:
                    cap = r0 * (1.0 - f)
                    p.data[s] = val + (p.data[s] - val).clamp(-cap, cap)
                for p, j, val, r0 in shifts:
                    cap = r0 * (1.0 - f)
                    cur = p.data[:, j]
                    p.data += ((val + (cur - val).clamp(-cap, cap)) - cur)[:, None]
                for pa, pb, g0, sep in ties:
                    cap = g0 * (1.0 - f)
                    if sep == ">":               # pull pa onto pb, pb untouched
                        pa.data.copy_(pb.data + (pa.data - pb.data).clamp(-cap, cap))
                    else:
                        mid = (pa.data + pb.data) / 2
                        half = ((pa.data - pb.data) / 2).clamp(-cap / 2, cap / 2)
                        pa.data.copy_(mid + half)
                        pb.data.copy_(mid - half)

        if args.morph:
            m0_, m1_ = (float(v) * args.steps for v in args.morph.split(","))
            g = min(1.0, max(0.0, (step - m0_) / max(1.0, m1_ - m0_)))
            model.act_t.fill_(g * g * (3.0 - 2.0 * g))

        if (step + 1) % args.log == 0 or step + 1 == args.steps:
            with torch.no_grad():
                accs = []
                for nn_ in places:
                    va, vb, vt = smp.batch(2048, n=nn_, split="val")
                    accs.append(loss_and_acc(model, va, vb, vt)[1])
                acc = torch.stack(accs).min(0).values
                a8 = accs[places.index(8)] if 8 in places else acc
                top = acc.argsort(descending=True)[:5]
                tag = f" act_t {model.act_t.item():.3f}" if args.morph else ""
                print(f"step {step+1:6d}{tag} loss {loss.min().item():.4f} "
                      f"best {acc.max().item():.5f} n8 {a8.max().item():.5f} "
                      f"top5 {[round(acc[i].item(),4) for i in top]} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                best = acc
                if args.probe:
                    c = model.code()[0, :, 0]
                    kw = model.key_w[0]
                    hb = torch.relu(model.b1[0]) if cfg["f_in"] == "free" else None
                    lam_ = (model.lam if cfg["lam"] == "free" else model.lamf)[0]
                    print(f"   code0 {c[0].item():+.4f} step {(c[9]-c[0]).item()/9:+.4f} "
                          f"lam {lam_.item():+.3f} wo {model.w_o[0,0].item():+.5f} "
                          f"wo2 {model.w_o2[0,0].item():+.4f} "
                          f"W1 {[round(v,3) for v in model.W1[0,:,0].tolist()]} "
                          f"b1 {[round(v,3) for v in model.b1[0].tolist()]} "
                          f"kw {[round(v,2) for v in kw.tolist()]}", flush=True)

    with torch.no_grad():
        accs = []
        for nn_ in places:
            va, vb, vt = smp.batch(16384, n=nn_, split="val")
            accs.append(loss_and_acc(model, va, vb, vt)[1])
        acc = torch.stack(accs).min(0).values
    order = acc.argsort(descending=True).cpu()
    state = {k: v.detach().cpu()[order].contiguous() for k, v in model.state_dict().items()
             if k in dict(model.named_parameters())}
    buffers = {k: v.detach().cpu() for k, v in model.state_dict().items()
               if k not in dict(model.named_parameters())}
    torch.save({"cfg": cfg, "state": state, "buffers": buffers,
                "acc": acc[order].cpu(), "n_par": n_par, "args": vars(args)}, args.out)
    print(f"saved {args.out}  n_par {n_par}  best {acc[order][0].item():.6f}  "
          f"members>=0.999: {(acc >= 0.999).sum().item()}/{args.E}")
    for p, s, val, r0 in shrinks:
        print(f"  final residual {(p.data[s]-val).abs().max().item():.3e}")
    for p, j, val, r0 in shifts:
        print(f"  final shift residual {(p.data[:, j]-val).abs().max().item():.3e}")
    for pa, pb, g0, sep in ties:
        print(f"  final tie gap  {(pa.data-pb.data).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
