"""Train E independent members simultaneously (a 'seed lottery' on one GPU).

Parameters never mix across the ensemble axis; Adam is elementwise and the gradient
clip is applied per member, so member i's trajectory is exactly what it would have
been training alone on the same data stream.
"""

import argparse, json, math, os, time
import torch

import core, data, stage


# element-wise pins: tensors that are trainable except for a few fixed entries.
# code[0] = 0 fixes the origin of the residual stream (the (0,0) pad token sits at 0);
# e[0] = 1 is the scale gauge (one carry-in equals one unit of the residual stream).
PINS = {"code": {0: 0.0}, "e": {0: 1.0}}


def apply_pins(p, pins):
    with torch.no_grad():
        for k, entries in pins.items():
            if k in p:
                for idx, val in entries.items():
                    p[k][:, idx] = val


def loss_and_acc(out, Y, reduce_loss=True, temp=1.0):
    """out [E,Bs,P,10], Y [Bs,P]. Positions 1.. are scored; position 0 is a pad."""
    E, Bs, P, _ = out.shape
    o = out[:, :, 1:, :]
    y = Y[None, :, 1:].expand(E, -1, -1)
    lp = torch.log_softmax(o * temp, dim=-1)
    nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)              # [E,Bs,P-1]
    ok = (o.argmax(-1) == y).all(-1)                                # [E,Bs] exact match
    if reduce_loss:
        return nll.mean(dim=(1, 2)), ok.float().mean(1)
    return nll, ok


@torch.no_grad()
def evaluate(p, n, device, gen, total=65536, bs=4096, held_out=True):
    E = p["code"].shape[0]
    hit = torch.zeros(E, device=device)
    seen = 0
    while seen < total:
        m = min(bs, total - seen)
        A, B, Y = data.batch(m, n, device, gen, held_out=held_out)
        out = core.forward_core(p, A, B)
        _, ok = loss_and_acc(out, Y)
        hit += ok * A.shape[0]
        seen += A.shape[0]
    return hit / seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--U", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--places", type=int, nargs="+", default=[8, 5, 11, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--freeze", type=str, default="")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--pct_start", type=float, default=0.15)
    ap.add_argument("--out", type=str, default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--loss_temp", type=float, default=1.0)
    ap.add_argument("--pin", action="store_true", help="hold code[0]=0 and e[0]=1")
    ap.add_argument("--bw_target", type=float, default=0.0,
                    help="anneal |bw| up to this constant, ending exactly there")
    ap.add_argument("--bw_from", type=float, default=0.25,
                    help="fraction of the run at which the |bw| floor starts rising")
    ap.add_argument("--bw_to", type=float, default=0.75,
                    help="fraction of the run by which the |bw| floor reaches the target")
    ap.add_argument("--reduced", action="store_true",
                    help="control run: start from the shipped constants (stage.CONST) with "
                         "them frozen, so only the 12 shipped numbers (+bw) are trainable. "
                         "Tests whether the reduced form cold-trains on its own.")
    args = ap.parse_args()

    dev = "cuda"
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    frozen = set(x for x in args.freeze.split(",") if x)
    if args.reduced:
        args.U = 2
        args.pin = True
        frozen |= set(stage.FROZEN)

    if args.init:
        st = torch.load(args.init, map_location=dev)
        src = st["p"]
        E0 = src["code"].shape[0]
        idx = torch.arange(args.E, device=dev) % E0      # tile the warm starts round-robin
        p = {k: v[idx].contiguous() for k, v in src.items()}
        if args.noise > 0:
            for k in p:
                if k in frozen:
                    continue
                sc = args.noise * p[k].abs().mean().clamp(min=1e-3)
                p[k] = p[k] + torch.randn(p[k].shape, generator=gen, device=dev) * sc
                p[k][:E0] = src[k][:E0]                  # keep the exact warm starts
    else:
        p = core.init_params(args.E, args.U, dev, gen)

    if args.reduced:
        # overwrite the frozen tensors with the exact constants the shipped file uses
        with torch.no_grad():
            for k, val in stage.CONST.items():
                t = torch.tensor(val, dtype=p[k].dtype, device=dev)
                p[k] = t.expand_as(p[k]).clone() if t.dim() == p[k].dim() - 1 \
                    else t.reshape(p[k].shape).expand_as(p[k]).clone()

    pins = PINS if args.pin else {}
    apply_pins(p, pins)
    for k in p:
        p[k] = p[k].detach().requires_grad_(k not in frozen)
    train_ps = [p[k] for k in p if k not in frozen]
    nfree = core.n_free(p, frozen) - sum(len(v) for k, v in pins.items() if k not in frozen)
    print(f"members={args.E} free params/member={nfree} frozen={sorted(frozen)} "
          f"pinned={ {k: v for k, v in pins.items()} }", flush=True)

    opt = torch.optim.AdamW(train_ps, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=args.pct_start)

    best = torch.zeros(args.E, device=dev)
    best_p = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for step in range(args.steps):
        n = args.places[step % len(args.places)]
        A, B, Y = data.batch(args.bs, n, dev, gen)
        out = core.forward_core(p, A, B)
        ls, acc = loss_and_acc(out, Y, temp=args.loss_temp)
        opt.zero_grad(set_to_none=True)
        ls.sum().backward()
        # per-member gradient clip (norm computed over that member's own slice)
        with torch.no_grad():
            sq = torch.zeros(args.E, device=dev)
            for t in train_ps:
                if t.grad is not None:
                    sq += t.grad.reshape(args.E, -1).pow(2).sum(1)
            scale = (1.0 / sq.sqrt().clamp(min=1e-12)).clamp(max=1.0)
            for t in train_ps:
                if t.grad is not None:
                    t.grad.mul_(scale.view(-1, *([1] * (t.dim() - 1))))
        opt.step()
        sched.step()
        apply_pins(p, pins)
        if args.bw_target > 0:
            # hard projection (not a forward-pass interpolation): the ramp width is
            # squeezed to exactly 1/bw_target by the end, while the knees bb stay free
            # and keep adapting for as long as any pair still sits inside a ramp.
            f = (step + 1) / args.steps
            lo = args.bw_target * min(max((f - args.bw_from) / max(args.bw_to - args.bw_from, 1e-9), 0.0), 1.0)
            with torch.no_grad():
                mag = p["bw"].abs().clamp(min=lo, max=args.bw_target)
                p["bw"].copy_(torch.sign(p["bw"]) * mag)

        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            ev = evaluate(p, 8, dev, gen, total=8192)
            with torch.no_grad():
                imp = ev > best
                if imp.any():
                    best = torch.where(imp, ev, best)
                    for k in p:
                        best_p[k] = torch.where(imp.view(-1, *([1] * (p[k].dim() - 1))),
                                                p[k].detach(), best_p[k])
            print(f"step {step+1:6d}  loss {ls.min().item():.4f}  "
                  f"trainacc {acc.max().item():.4f}  ev8 {ev.max().item():.5f}  "
                  f"best {best.max().item():.5f}  n>=.99 {(best>=0.99).sum().item()}  "
                  f"{(step+1)/(time.time()-t0):.1f} it/s", flush=True)

    # rank by a fuller held-out eval of the best snapshots
    fin = evaluate(best_p, 8, dev, gen, total=262144)
    order = torch.argsort(fin, descending=True)
    print("top members:", [(int(i), round(float(fin[i]), 6)) for i in order[:12]], flush=True)
    keep = order[:min(32, args.E)]
    out_p = {k: v[keep].detach().clone() for k, v in best_p.items()}
    torch.save({"p": out_p, "U": args.U, "frozen": sorted(frozen), "pins": pins,
                "acc": fin[keep].cpu(), "args": vars(args)}, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
