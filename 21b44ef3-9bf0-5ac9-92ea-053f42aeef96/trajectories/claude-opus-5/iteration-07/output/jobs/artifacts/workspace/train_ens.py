"""Train E independent members of one architecture simultaneously (vmap).

At these sizes seed variance dominates, so the practical unit of training is an
ensemble of independently initialised members sharing a data stream.  Nothing
here is imported by the graded submission.

Training runs on operands *longer* than the eight digits that are graded.  Every
parameter is position-independent — only the causal mask and the integer
distance matrix depend on the position count, and both are buffers — so the same
weights run over any length.  Long operands are what force the attention to do
real work: carrying across a run of `k` carry-transparent places requires
position `i` to actually select position `i-k-1`, which a fixed distance-decay
pattern cannot do for large `k` at any decay rate.
"""
import argparse, json, math, os, time
import torch
from torch.func import stack_module_state, functional_call
from torch import vmap

import data
from model_src import DigitPairAdder, default_cfg, init_model, n_params
from transfer import warm_start

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SHIP_P = 10          # graded geometry: 8 places plus two pads


def build_stack(cfg, E, seed0, init_sd=None, sigma=0.0, parent_cfg=None,
                dev="cuda"):
    models = []
    for e in range(E):
        m = init_model(cfg, seed0 + e)
        if init_sd is not None:
            t, _ = warm_start(init_sd, parent_cfg or cfg, cfg, seed0 + e)
            g = torch.Generator().manual_seed(seed0 + 100000 + e)
            with torch.no_grad():
                for name, p in m.named_parameters():
                    v = t[name]
                    if e > 0 and sigma > 0:
                        v = v + torch.randn(v.shape, generator=g) * sigma * \
                            (v.abs().mean() + 0.5)
                    p.copy_(v)
        models.append(m.to(dev))
    params, _ = stack_module_state(models)
    params = {k: v.detach().clone().requires_grad_(True)
              for k, v in params.items()}
    base = DigitPairAdder(cfg).to("meta")
    return params, base, models[0]


def buffers_for(cfg, P, dev):
    """Buffer set for a given position count (identical across members)."""
    m = DigitPairAdder(dict(cfg, P=int(P))).to(dev)
    return {k: v.detach().clone() for k, v in m.named_buffers()}


def _acc(vf, params, bufs, tok, tgt, chunk=8192):
    outs = []
    for s in range(0, tok.shape[0], chunk):
        lg = vf(params, bufs, tok[s:s + chunk])
        pr = lg[:, :, 1:, :].argmax(-1)
        outs.append((pr == tgt[None, s:s + chunk, 1:]).all(-1).float())
    return torch.cat(outs, 1).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--init", default=None)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--sched", default="onecycle", choices=["onecycle", "cos"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--places", default="8,14",
                    help="comma-separated operand lengths to train on; one is "
                         "drawn uniformly per step.  Short operands make the "
                         "code table and the mod-10 fold easy to find, long "
                         "ones force the attention to select positions.")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=8192)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--pen", default="{}",
                    help='group-lasso coefficients, e.g. {"code_rest": 0.02}; '
                         "shrinks a whole parameter group so the structure it "
                         "occupies can then be cut")
    ap.add_argument("--shrink", default="{}",
                    help='annealed structured pruning, e.g. {"code_rest": 0.7}: '
                         "the group is frozen and scaled linearly to exactly "
                         "zero over that fraction of the run, then held at zero "
                         "for the rest.  Unlike a penalty this cannot be traded "
                         "off against the loss -- training ends with the group "
                         "contributing nothing, so the model must migrate "
                         "whatever it encoded there and the cut is lossless.")
    ap.add_argument("--pen_gate", type=float, default=0.05,
                    help="members whose penalised group exceeds this norm are "
                         "not eligible to be selected")
    ap.add_argument("--pen_warm", type=int, default=0,
                    help="steps over which the penalty ramps in")
    ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--keep", type=int, default=8, help="alternates to save")
    ap.add_argument("--patience", type=int, default=10 ** 9)
    args = ap.parse_args()

    dev = "cuda"
    over = json.loads(args.cfg)
    parent_cfg, init_sd = None, None
    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        parent_cfg = dict(default_cfg()); parent_cfg.update(ck["cfg"])
        init_sd = ck["state"]
        cfg = dict(parent_cfg)
    else:
        cfg = dict(default_cfg())
    cfg.update(over)
    cfg["P"] = SHIP_P                       # checkpoints store the graded geometry

    lens = [int(x) for x in str(args.places).split(",") if x.strip()]
    L = max(lens)
    pen = json.loads(args.pen)
    shrink = json.loads(args.shrink)
    params, base, proto = build_stack(cfg, args.E, args.seed * 7919 + 1,
                                      init_sd, args.sigma, parent_cfg, dev)
    for flag, d in (("--pen", pen), ("--shrink", shrink)):
        missing = [k for k in d if k not in params]
        if missing:
            raise SystemExit(f"{flag} names absent parameters: {missing}")
    bufs_by_len = {n: buffers_for(cfg, n + 2, dev) for n in set(lens + [8])}
    buf_s, buf_l = bufs_by_len[8], bufs_by_len[L]
    npar = n_params(proto)
    print(f"cfg={json.dumps(cfg, sort_keys=True)}\n"
          f"params/member={npar}  E={args.E}  train lengths={lens}", flush=True)

    def fwd(p, b, tok):
        return functional_call(base, (p, b), (tok,))

    vf = vmap(fwd, in_dims=(0, None, None))

    # a group being annealed away is frozen: only its scale changes
    shrink_base = {k: params[k].detach().clone() for k in shrink}
    opt = torch.optim.AdamW([v for k, v in params.items() if k not in shrink],
                            lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    if args.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=args.lr * 0.02)

    gen = torch.Generator(device=dev).manual_seed(1234 + args.seed)
    egen = torch.Generator(device=dev).manual_seed(999)
    ev_u, et_u = data.sample(args.eval_n, dev, egen, mix=(1., 0., 0.),
                             split="eval", nplace=8)
    ev_c, et_c = data.sample(args.eval_n, dev, egen, mix=(0., .3, .7),
                             split="eval", nplace=8)
    ev_l, et_l = data.sample(args.eval_n, dev, egen, mix=(0., .3, .7),
                             split="eval", nplace=L)
    E = args.E
    best = (-1.0, -1, None, None)
    t0, since = time.time(), 0

    streams = {n: data.Stream(dev, gen, args.bs, split="train", nplace=n)
               for n in set(lens)}
    for step in range(args.steps):
        n = lens[step % len(lens)]
        tok, tgt = streams[n].next()
        bufs = bufs_by_len[n]
        logits = vf(params, bufs, tok)                        # (E,B,P,10)
        lp = torch.log_softmax(logits[:, :, 1:, :], dim=-1)
        nll = -lp.gather(-1, tgt[None, :, 1:, None].expand(E, -1, -1, -1))
        loss = nll.mean()
        if pen:
            ramp = min(1.0, (step + 1) / args.pen_warm) if args.pen_warm else 1.0
            for name, coef in pen.items():
                loss = loss + (ramp * coef / E) * \
                    params[name].reshape(E, -1).norm(dim=1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        live = [p for k, p in params.items() if k not in shrink]
        sq = torch.zeros(E, device=dev)
        for p in live:
            if p.grad is not None:
                sq += p.grad.reshape(E, -1).pow(2).sum(1)
        scale = (args.clip / (sq.sqrt() + 1e-8)).clamp(max=1.0)
        for p in live:
            if p.grad is not None:
                p.grad.mul_(scale.view(E, *([1] * (p.grad.dim() - 1))))
        opt.step()
        sched.step()
        with torch.no_grad():
            for name, frac in shrink.items():
                span = max(1.0, frac * args.steps)
                params[name].copy_(shrink_base[name]
                                   * max(0.0, 1.0 - (step + 1) / span))

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            with torch.no_grad():
                au = _acc(vf, params, buf_s, ev_u, et_u)
                ac = _acc(vf, params, buf_s, ev_c, et_c)
                al = _acc(vf, params, buf_l, ev_l, et_l)
                comb = torch.minimum(torch.minimum(au, ac), al)
                if pen:
                    # only members that actually gave the group up are useful:
                    # the point of the penalty is to make the group removable
                    nrm = sum(params[k].reshape(E, -1).norm(dim=1) for k in pen)
                    ok = nrm < args.pen_gate
                    if bool(ok.any()):
                        comb = torch.where(ok, comb, torch.full_like(comb, -1.0))
                order = comb.argsort(descending=True)
                v, i = float(comb[order[0]]), int(order[0])
            if v > best[0]:
                alts = [{k: params[k][int(j)].detach().cpu().clone()
                         for k in params} for j in order[:args.keep]]
                best = (v, i, alts[0], alts)
                since = 0
                torch.save({"cfg": cfg, "state": best[2], "acc": best[0],
                            "alts": best[3], "params": npar,
                            "places": L}, args.out)          # crash-safe
            else:
                since += args.eval_every
            pn = "".join(f" |{k}| {float(params[k][i].norm()):.4f}"
                         for k in list(pen) + [x for x in shrink if x not in pen])
            print(f"step {step+1:6d}  loss {float(loss):.5f}  best {v:.5f} "
                  f"(m{i}) u8 {float(au[i]):.5f} c8 {float(ac[i]):.5f} "
                  f"c{L} {float(al[i]):.5f}{pn}  top5 "
                  f"{[round(float(comb[j]),4) for j in order[:5]]}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            if v >= args.target or since >= args.patience:
                break

    torch.save({"cfg": cfg, "state": best[2], "acc": best[0], "alts": best[3],
                "params": npar, "places": L}, args.out)
    print(f"SAVED {args.out}  acc={best[0]:.6f}  params={npar}", flush=True)


if __name__ == "__main__":
    main()
