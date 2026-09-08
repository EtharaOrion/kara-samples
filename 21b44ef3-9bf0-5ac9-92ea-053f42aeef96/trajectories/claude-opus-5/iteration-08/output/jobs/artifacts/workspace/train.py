"""Seed-lottery trainer: E independent members trained simultaneously.

Every member sees the same batch but has its own parameters; Adam is elementwise
and gradients are clipped per member, so the members are fully independent runs
that happen to share the data pipeline.  Small models of this size have huge
seed variance, so training a few hundred at once is the only practical way to
find a working one.

  python train.py --name parent --cfg '{"C":2,"U":6,"U2":6}' --steps 20000
"""
import argparse, json, os, time
import torch
import lib

DEV = "cuda"
CKPT = "/workspace/ckpt"


def ce_loss(logits, y, keep):
    """logits (E,B,P,10), y (B,P), keep (B,) -> per-member mean CE over positions 1..P-1."""
    lp = torch.log_softmax(logits[:, :, 1:], -1)
    tgt = y[None, :, 1:, None].expand(lp.shape[0], -1, -1, -1)
    nll = -lp.gather(-1, tgt).squeeze(-1)                     # (E,B,P-1)
    w = keep.to(nll.dtype)[None, :, None]
    return (nll * w).sum((1, 2)) / (w.sum() * nll.shape[2])


def make_eval(n, count, device, seed, full_width=0.85, chain=False):
    g = torch.Generator(device=device).manual_seed(seed)
    A, B_, PA, PB, Y = [], [], [], [], []
    got = 0
    while got < count:
        a, b = lib.sample_digits(8192, n, device, g, full_width)
        if chain:
            r = torch.rand(8192, n, device=device, generator=g)
            b = torch.where(r < 0.85, 9 - a, b)
            a[:, -1] = torch.clamp(a[:, -1], min=1)
            b[:, -1] = torch.where(b[:, -1] == 0, 9 - a[:, -1], b[:, -1])
        m = lib.heldout_mask(a, b)
        a, b = a[m], b[m]
        if a.shape[0] == 0:
            continue
        pa, pb = lib.pad_places(a, b)
        A.append(a); B_.append(b); PA.append(pa); PB.append(pb); Y.append(lib.targets(a, b))
        got += a.shape[0]
    return (torch.cat(PA)[:count], torch.cat(PB)[:count], torch.cat(Y)[:count])


@torch.no_grad()
def evaluate(p, cfg, ev, chunk=4096):
    pa, pb, y = ev
    E = next(iter(p.values())).shape[0]
    ok = torch.zeros(E, device=DEV)
    for i in range(0, pa.shape[0], chunk):
        lg = lib.fwd(p, cfg, pa[i:i + chunk], pb[i:i + chunk])
        pred = lg.argmax(-1)[:, :, 1:]
        ok += (pred == y[None, i:i + chunk, 1:]).all(-1).sum(1).float()
    return ok / pa.shape[0]


def parse_anneal(cfg, spec):
    """Structural cuts annealed to their exact target over training rather than
    made abruptly: 'u1:5' scales bank-1 unit 5 to zero, 'lam' drives the learned
    distance slope onto the fixed one, 'tie' drives bank 2 onto bank 1, 'kv'
    drives the value read-out onto the key axis.  Freezing the schedule (rather
    than penalising) means the cut cannot be traded off against the loss.

    Zero-masks are enforced by also zeroing the masked gradients, so a masked
    weight really is on its way out.  `toward` ops have no such protection and
    must not be applied by interpolating in the forward pass alone: b1 and b2
    reach the loss only through their annealed combination, whose gap is
    s*(b1-b2), so gradient descent grows the raw gap exactly as fast as s falls,
    the live model stays perfect, and the cut model never improves at all.  They
    are enforced instead by `project` below, which caps the raw residual.

    Returns (zero_masks, toward_ops).
    """
    masks = {n: torch.zeros(s, dtype=torch.bool) for n, s in lib.param_shapes(cfg).items()}
    toward = []
    for item in [s for s in spec.split(",") if s]:
        kind, _, idx = item.partition(":")
        i = int(idx) if idx else 0
        if kind == "lam":
            assert cfg["lam_learn"]
            toward.append(("lam", lambda p, c: torch.full_like(p["lam"], float(c["lam"]))))
        elif kind == "tie":
            assert not cfg["tie"] and cfg["U2"] == cfg["U"] and cfg["f2_in"] == cfg["f1_in"]
            # symmetric: both banks move to their midpoint rather than bank 2
            # travelling to wherever bank 1 currently sits.  The knees have to
            # satisfy two windows at once -- bank 1 splits s<=8 / s=9 / s>=10 on
            # the token, the fold splits s+c<=9 / s+c>=10 on the residual -- and
            # a one-sided anneal drags bank 2's knee out of the second window on
            # the way, which breaks the fold irrecoverably.
            if cfg["f2_in"] == "free":
                toward.append(("w1", lambda p, c: 0.5 * (p["w1"] + p["w2"])))
                toward.append(("w2", lambda p, c: 0.5 * (p["w1"] + p["w2"])))
            else:
                assert float(cfg["f2_sign"]) == float(cfg["f1_sign"])
            toward.append(("b1", lambda p, c: 0.5 * (p["b1"] + p["b2"])))
            toward.append(("b2", lambda p, c: 0.5 * (p["b1"] + p["b2"])))
        elif kind == "kvs":
            assert cfg["kv"] == "share"
            # symmetric again: the key scale is ~50 and the value scale ~33, and
            # driving either onto the other alone walks the attention key gap out
            # of the distance-penalty budget before the other side can follow.
            def mid(p, c):
                return 0.5 * (p["b_q"] + p["w_v"][:, 0])
            toward.append(("b_q", mid))
            toward.append(("w_v", lambda p, c, m=mid: torch.cat(
                [m(p, c)[:, None], p["w_v"][:, 1:]], 1)))
        elif kind == "kv":
            assert cfg["kv"] == "split"

            def proj(p, c):
                k = p["k_o"]                                   # (E,U)
                kh = k / k.norm(dim=1, keepdim=True).clamp(min=1e-9)
                return torch.einsum("eu,euc->ec", kh, p["v_o"])[:, None, :] * kh[..., None]
            toward.append(("v_o", proj))
        elif kind == "u1":
            assert not cfg["o_pin"], "shrink u1 before pinning o"
            masks["b1"][i] = True
            if cfg["kv"] == "share":
                masks["o_free"][i] = True
            else:
                masks["k_o"][i] = True
                masks["v_o"][i] = True
            if cfg["f1_in"] == "free":
                masks["w1"][:, i] = True
            if cfg["tie"] and not cfg["p_rows"]:
                masks["p_out"][i] = True
        elif kind == "u2":
            assert not cfg["tie"]
            masks["b2"][i] = True
            if cfg["f2_in"] == "free":
                masks["w2"][:, i] = True
            masks["p_out"][i] = True
        elif kind == "po":
            assert not cfg["p_rows"]
            masks["p_out"][i] = True
        elif kind == "c":
            assert cfg["pin_row"] < 0, "shrink code axes before pinning a row"
            masks["code_free"][:, i] = True
            if cfg["kv"] == "share":
                masks["w_v"][i] = True
            else:
                masks["v_o"][:, i] = True
            masks["p_out"][:, i] = True
            if cfg["f1_in"] == "free":
                masks["w1"][i, :] = True
            if cfg["f2_in"] == "free" and not cfg["tie"]:
                masks["w2"][i, :] = True
        else:
            raise KeyError(kind)
    return {n: m.to(DEV) for n, m in masks.items() if m.any()}, toward


def anneal(p, cfg, masks, toward, s):
    """p_eff at schedule position s (1 = untouched, 0 = cut fully in place)."""
    if s >= 1.0:
        return p
    pe = dict(p)
    for n, m in masks.items():
        mul = torch.ones_like(p[n][0])
        mul[m] = s
        pe[n] = pe[n] * mul
    tg = [(n, tgt(pe, cfg)) for n, tgt in toward]     # all targets read the same state
    for n, v in tg:
        pe[n] = s * pe[n] + (1.0 - s) * v
    return pe


@torch.no_grad()
def project(q, scale, cfg, toward, s, lim):
    """Cap how far the annealed parameters may still sit from their cut values.

    `lim[n]` is the per-member residual norm recorded when the schedule starts;
    the residual is clipped to `s` times that, so it is driven to zero on the
    schedule and cannot be inflated to buy the loss back.  Shrinking faster than
    the schedule is allowed -- the cap is one-sided -- and the pair stays free to
    move as a whole, since only the difference from the target is constrained.
    """
    p = {n: q[n] * scale[n] for n in q}
    upd = []
    for n, tgt in toward:
        v = tgt(p, cfg)
        res = p[n] - v
        nr = res.reshape(res.shape[0], -1).norm(dim=1)
        f = (s * lim[n] / nr.clamp(min=1e-12)).clamp(max=1.0)
        upd.append((n, v + res * f.reshape(-1, *([1] * (res.dim() - 1)))))
    for n, v in upd:
        q[n].data.copy_(v / scale[n])


@torch.no_grad()
def snap(p, cfg, masks, toward):
    """Write the fully-cut values into the raw parameters, in place."""
    for n, m in masks.items():
        p[n].data[:, m] = 0.0
    for n, v in [(n, tgt(p, cfg)) for n, tgt in toward]:
        p[n].data.copy_(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--init", default="")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrink", default="")
    ap.add_argument("--shrink_start", type=float, default=0.15)
    ap.add_argument("--shrink_end", type=float, default=0.60)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--places", default="8:0.40,3:0.06,4:0.06,5:0.08,6:0.08,7:0.08,9:0.06,10:0.06,12:0.06,14:0.06")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--by_auto", action="store_true",
                    help="with y_bias: start `by` at the mean read-out contribution of "
                         "the fold rows being annealed away, so the cut has somewhere "
                         "obvious to move the constant to")
    ap.add_argument("--ls_auto", type=float, default=0.0,
                    help="with logit_scale: calibrate exp(ls) per member so the median "
                         "margin between the correct prototype and its runner-up is this "
                         "many nats.  Without it a reduced code (spacing ~0.2) gives "
                         "CE ~ log 10 at perfect accuracy and the gradient is all "
                         "temperature.")
    ap.add_argument("--relscale", type=float, default=0.0,
                    help=">0: optimise each weight relative to its own magnitude, "
                         "with this value as the absolute floor.  --lr then means a "
                         "relative step size.")
    ap.add_argument("--mix0", default="", help="e.g. '0.05,0.25,0.70': data mix at step 0, "
                    "annealed linearly to the standard mix by --mix_end of the run")
    ap.add_argument("--mix_end", type=float, default=0.35)
    args = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(args.seed)
    over = json.loads(args.cfg)
    if args.init:
        ck = torch.load(args.init, map_location=DEV, weights_only=False)
        cfg = dict(ck["cfg"]); cfg.update(over)
        cfg = lib.default_cfg(**cfg)
        src = ck["params"]
        K = next(iter(src.values())).shape[0]
        want = lib.param_shapes(cfg)
        fresh = lib.init_params(cfg, K, DEV, seed=args.seed + 3)
        for n in want:                       # params the target cfg adds (e.g. ls)
            if n not in src or tuple(src[n].shape[1:]) != tuple(want[n]):
                src[n] = fresh[n]
                print("  fresh init for", n)
        src = {n: src[n] for n in want}      # params the target cfg drops
        g = torch.Generator(device=DEV).manual_seed(args.seed + 7)
        p = {}
        for n, t in src.items():
            t = t.to(DEV)
            rep = t[torch.arange(args.E, device=DEV) % K]
            noise = torch.randn(rep.shape, generator=g, device=DEV) * args.sigma
            noise[:K] = 0                                     # keep exact copies of the parents
            # multiplicative: a reduced model's weights span several orders of
            # magnitude, and absolute noise scaled by a tensor's mean wipes out
            # its small entries
            p[n] = (rep * (1.0 + noise)).contiguous()
        print(f"warm start from {args.init} ({K} members) sigma={args.sigma}")
    else:
        cfg = lib.default_cfg(**over)
        p = lib.init_params(cfg, args.E, DEV, seed=args.seed)
    masks, toward = parse_anneal(cfg, args.shrink) if args.shrink else ({}, [])

    if args.ls_auto > 0:
        assert cfg["logit_scale"], "--ls_auto needs logit_scale in the cfg"
        with torch.no_grad():
            gcal = torch.Generator(device=DEV).manual_seed(args.seed + 7)
            _, _, pa, pb, y = lib.batch(2048, 8, DEV, gcal)
            p["ls"] = torch.zeros(args.E, device=DEV)
            d2 = -lib.fwd(p, cfg, pa, pb)                        # (E,B,P,10)
            ye = y[None, :, :, None].expand(d2.shape[0], -1, -1, -1)
            corr = d2.gather(-1, ye).squeeze(-1)
            gap = (d2.scatter(-1, ye, float("inf")).min(-1).values - corr)[:, :, 1:]
            med = gap.reshape(gap.shape[0], -1).clamp(min=1e-9).median(1).values
            p["ls"] = torch.log(args.ls_auto / med)
            print("ls_auto: median margin", [round(float(v), 4) for v in med[:4]],
                  "-> ls", [round(float(v), 3) for v in p["ls"][:4]])

    if args.by_auto:
        assert cfg["y_bias"] and "p_out" in masks, "--by_auto needs y_bias and a fold cut"
        with torch.no_grad():
            gcal = torch.Generator(device=DEV).manual_seed(args.seed + 5)
            _, _, pa, pb, _ = lib.batch(2048, 8, DEV, gcal)
            rows = masks["p_out"].any(-1).nonzero().flatten().tolist()
            p2 = dict(p)
            p2["p_out"] = p["p_out"].clone()
            p2["p_out"][:, rows] = 0.0
            dy = (lib.fwd(p, cfg, pa, pb, want_y=True)[1] -
                  lib.fwd(p2, cfg, pa, pb, want_y=True)[1])[:, :, 1:, 0].mean((1, 2))
            n = torch.randn(dy.shape, generator=gcal, device=DEV)
            n[:min(8, args.E)] = 0
            p["by"] = dy * (1.0 + 0.05 * n)
            print("by_auto: fold rows", rows, "-> by", [round(float(v), 4) for v in dy[:4]])

    names = list(p)
    # Optimise each weight relative to its own magnitude.  Adam's step is ~lr in
    # absolute units, but a reduced model's weights range over several orders of
    # magnitude (a deep attention notch needs a large key gain next to an O(1)
    # code), so one absolute lr either freezes the large weights or destroys the
    # small ones.  Train q = p / s with s fixed at the initial magnitude instead.
    if args.relscale > 0:
        scale = {n: p[n].detach().abs().clamp(min=args.relscale) for n in names}
    else:
        scale = {n: torch.ones((), device=DEV) for n in names}
    q = {n: (p[n].detach() / scale[n]).requires_grad_(True) for n in names}

    def raw():
        return {n: q[n] * scale[n] for n in names}
    print("cfg:", json.dumps(cfg))
    print("params/member:", lib.n_params(cfg))

    if masks or toward:
        print("anneal:", args.shrink, {k: int(v.sum()) for k, v in masks.items()},
              [n for n, _ in toward])

    places = [(int(k.split(":")[0]), float(k.split(":")[1])) for k in args.places.split(",")]
    pl = torch.tensor([w for _, w in places], device=DEV)
    pl = pl / pl.sum()
    plc = [n for n, _ in places]

    opt = torch.optim.AdamW([q[n] for n in names], lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.15, div_factor=10, final_div_factor=100)
    MIX1 = (0.35, 0.40, 0.25)
    mix0 = tuple(float(v) for v in args.mix0.split(",")) if args.mix0 else None
    if mix0:
        print("data mix:", mix0, "->", MIX1, f"by frac {args.mix_end}")
    gen = torch.Generator(device=DEV).manual_seed(args.seed + 1)
    ev_u = make_eval(8, 8192, DEV, 12345)
    ev_c = make_eval(8, 8192, DEV, 999, chain=True)

    # running best over the *cut* model (see the eval block below); checkpointed
    # every eval so a long run is never lost to an interrupt.
    # recorded before the first step, and enforced from the first step: letting
    # the residual float during the warm-up would just hand the schedule a bigger
    # gap to close, and warm-start noise alone can double it.
    pr0 = raw()
    lim = {n: (pr0[n] - tgt(pr0, cfg)).reshape(args.E, -1).norm(dim=1)
           for n, tgt in toward} if toward else None
    best_sc = torch.zeros(args.E, device=DEV)
    best_p = {n: torch.zeros_like(q[n]) for n in names}

    def save_best(step):
        order = best_sc.argsort(descending=True)[:args.keep]
        torch.save(dict(cfg=cfg, step=step,
                        params={n: best_p[n][order].clone().cpu() for n in names},
                        scores=[float(best_sc[i]) for i in order]),
                   f"{CKPT}/{args.name}.pt")

    t0 = time.time()
    for step in range(args.steps):
        frac = step / args.steps
        s = 1.0
        if masks or toward:
            s = 1.0 if frac <= args.shrink_start else max(
                0.0, 1.0 - (frac - args.shrink_start) / (args.shrink_end - args.shrink_start))
        pe = anneal(raw(), cfg, masks, [], s)      # `toward` is enforced by project()

        n_pl = plc[int(torch.multinomial(pl, 1).item())]
        mix = MIX1
        if mix0 is not None and frac < args.mix_end:
            w = frac / args.mix_end
            mix = tuple(w * b_ + (1 - w) * a_ for a_, b_ in zip(mix0, MIX1))
        a, b, pa, pb, y = lib.batch(args.batch, n_pl, DEV, gen, mix=mix)
        keep = ~lib.heldout_mask(a, b)
        logits = lib.fwd(pe, cfg, pa, pb)
        loss = ce_loss(logits, y, keep)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        with torch.no_grad():
            sq = None
            for n in names:
                gg = q[n].grad
                s2 = gg.reshape(gg.shape[0], -1).pow(2).sum(1)
                sq = s2 if sq is None else sq + s2
            fac = (1.0 / sq.sqrt().clamp(min=1e-12)).clamp(max=1.0)
            for n in names:
                gg = q[n].grad
                gg.mul_(fac.reshape(-1, *([1] * (gg.dim() - 1))))
                if n in masks:
                    gg[:, masks[n]] = 0.0
        opt.step()
        sched.step()
        if toward:
            project(q, scale, cfg, toward, s, lim)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            pv = anneal(raw(), cfg, masks, [], s) if (masks or toward) else raw()
            au = evaluate(pv, cfg, ev_u)
            ac = evaluate(pv, cfg, ev_c)
            sc = torch.minimum(au, ac)
            top = sc.argsort(descending=True)[:5]
            msg = ""
            if masks or toward:
                # the model that is actually wanted is the fully-cut one, so score
                # that at every checkpoint instead of only at the end: a member can
                # be a good cut model long before the schedule finishes dragging it
                # there, and the anneal can walk away from it afterwards.
                ps = anneal(raw(), cfg, masks, toward, 0.0)
                ss = torch.minimum(evaluate(ps, cfg, ev_u), evaluate(ps, cfg, ev_c))
                upd = ss > best_sc
                for n in names:
                    best_p[n][upd] = ps[n].detach()[upd]
                best_sc.copy_(torch.maximum(best_sc, ss))
                save_best(step + 1)
                gap = 0.0
                if toward:
                    pr = raw()
                    gap = max(float((pr[n] - tgt(pr, cfg)).abs().max()) for n, tgt in toward)
                msg = (f" cut_now={ss.max():.5f} cut_best={best_sc.max():.5f}"
                       f" ({int((best_sc >= 0.999).sum())}/{args.E}) gap={gap:.3f}")
            print(f"[{step+1:6d}] {time.time()-t0:6.0f}s loss={loss.mean().item():.4f} "
                  f"best={sc[top[0]]:.5f} (u={au[top[0]]:.5f} c={ac[top[0]]:.5f}) "
                  f"top5={[round(float(sc[i]), 4) for i in top]} "
                  f">=0.999: {int((sc >= 0.999).sum())}/{args.E}{msg}", flush=True)

    if masks or toward:
        save_best(args.steps)
        print("saved", f"{CKPT}/{args.name}.pt", "best-ever cut scores",
              [round(float(v), 5) for v in best_sc.sort(descending=True).values[:args.keep]])
        return
    p = {n: v.detach() for n, v in raw().items()}
    au, ac = evaluate(p, cfg, ev_u), evaluate(p, cfg, ev_c)
    sc = torch.minimum(au, ac)
    order = sc.argsort(descending=True)[:args.keep]
    out = dict(cfg=cfg,
               params={n: p[n][order].clone().cpu() for n in names},
               scores=[float(sc[i]) for i in order],
               uniform=[float(au[i]) for i in order],
               chain=[float(ac[i]) for i in order])
    path = f"{CKPT}/{args.name}.pt"
    torch.save(out, path)
    print("saved", path, "scores", [round(s, 5) for s in out["scores"]])


if __name__ == "__main__":
    main()
