"""Two-phase curriculum.

Phase 1 (n = 1 only) teaches the digit code, the mod-10 fold and the
"this place generates a carry" detector.  It cannot teach carry ROUTING,
because with a single place there is never a transparent place to skip.

Phase 2 warm-starts from a phase-1 winner, randomly re-initialises the second
bank unit (the one the n=1 task never needed) and trains on several place
counts.  The transparency notch in the key has to be discovered here.

Wiring choice: bank unit 1 drives the value stream (val_w = (0,1)); unit 0 does
not.  Which inputs each unit fires on -- i.e. both knees -- and how each unit
drives the key are learned.
"""
import argparse
import time

import torch

import arch
import data
import lab


def train(cfg, params, keys, steps, batch, lr, seed, corners, places,
          regimes=(0.0, 0.4, 0.9), pct=0.1, freeze_code_frac=0.0, log=None,
          ramp=0.0):
    dev = "cuda"
    for k in params:
        params[k].requires_grad_(k in keys)
    opt = torch.optim.AdamW([params[k] for k in keys], lr=lr, weight_decay=0.0,
                            betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                              pct_start=pct)
    gen = torch.Generator(device=dev).manual_seed(seed + 999)
    E = params["code"].shape[0]
    best = torch.full((E,), -1.0, device=dev)
    best_w = {k: params[k].detach().clone() for k in params}
    n_frozen = int(freeze_code_frac * steps)
    for step in range(steps):
        n = places[step % len(places)]
        reg = regimes
        if ramp > 0:                     # grow the transparency rate over time
            r = min(1.0, (step / steps) / ramp)
            reg = tuple(v * r for v in regimes)
        da, db = data.sample(batch, n, dev, regimes=reg, gen=gen)
        tok, tgt = data.tokens(da, db), data.targets(da, db)
        loss = lab.ce_loss(arch.forward(params, cfg, tok), tgt)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        if step < n_frozen and params["code"].grad is not None:
            params["code"].grad.zero_()
        lab.clip_per_member(params, keys, 1.0)
        opt.step()
        sch.step()
        if (step + 1) % 250 == 0 or step == steps - 1:
            acc = lab.eval_exact(params, cfg, corners)
            imp = acc > best
            if imp.any():
                for k in params:
                    best_w[k] = torch.where(
                        imp.view(-1, *([1] * (params[k].dim() - 1))),
                        params[k].detach(), best_w[k])
                best = torch.maximum(best, acc)
            if log:
                print(f"  {log} step {step+1:5d} loss {loss.mean().item():.4f} "
                      f"max {best.max().item():.4f} "
                      f"#>=.999 {(best>=0.999).sum().item()}", flush=True)
    return best, best_w


def phase1(E, steps, seed, cfg, batch=512, lr=0.012):
    dev = "cuda"
    p = arch.init_params(E, cfg, dev, seed=seed, ls0=1.0)
    corners = lab.make_eval([1], dev, per_n=2048)
    keys = arch.trainable_keys(cfg)
    return train(cfg, p, keys, steps, batch, lr, seed, corners, [1]) + (keys,)


def reinit_unit0(p, cfg, seed):
    """Re-init bank unit 0 -- the unit the n=1 task had no use for -- as the
    lottery the transparency notch has to come out of.

    Data-driven init: unit 0 copies unit 1's (already learned) input direction,
    up to a random sign and a random slope factor, and its knee is placed at
    the pre-activation value of a RANDOMLY CHOSEN one of the 100 digit pairs.
    Nothing about which pairs matter is injected -- the threshold is drawn from
    the model's own observed activations, and training refines it."""
    dev = p["Wb"].device
    E, U, C = p["Wb"].shape
    g = torch.Generator(device=dev).manual_seed(seed)
    q = {k: v.detach().clone() for k, v in p.items()}

    code = arch.effective(q, cfg, dev)["code"]                     # (E,10,C)
    da = torch.arange(10, device=dev).repeat_interleave(10)
    db = torch.arange(10, device=dev).repeat(10)
    x = code[:, da] + code[:, db]                                  # (E,100,C)
    e1 = torch.einsum("ec,epc->ep", q["Wb"][:, 1, :], x) + q["bb"][:, 1:2]

    pick = torch.randint(0, 100, (E,), device=dev, generator=g)
    t = e1[torch.arange(E, device=dev), pick]                      # (E,)
    sgn = torch.where(torch.rand(E, device=dev, generator=g) < 0.5, -1.0, 1.0)
    slope = 0.3 + 1.2 * torch.rand(E, device=dev, generator=g)
    m = (sgn * slope)[:, None]
    q["Wb"][:, 0, :] = m * q["Wb"][:, 1, :]
    q["bb"][:, 0] = 0.5 - (m.squeeze(-1) * (t - q["bb"][:, 1]))
    q["kw"][:, 0] = torch.randn(E, device=dev, generator=g)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E1", type=int, default=1024)
    ap.add_argument("--E2", type=int, default=4096)
    ap.add_argument("--p1_steps", type=int, default=2500)
    ap.add_argument("--p2_steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--lr2", type=float, default=0.004)
    ap.add_argument("--freeze", type=float, default=0.3)
    ap.add_argument("--places2", type=int, nargs="+", default=[2, 3, 5, 8])
    ap.add_argument("--out", default="parent.pt")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)

    cfg = arch.default_cfg(C=2, U=2, act="clamp", fix_val_w=(0.0, 1.0))
    print(f"free scalars / member = {arch.n_free_values(cfg)}", flush=True)

    t0 = time.time()
    b1, w1, keys = phase1(a.E1, a.p1_steps, a.seed, cfg)
    good = (b1 >= 0.9999).nonzero().flatten()
    print(f"phase1: {good.numel()}/{a.E1} members exact on n=1 "
          f"({time.time()-t0:.0f}s)", flush=True)
    if good.numel() == 0:
        raise SystemExit("phase 1 found nothing")
    torch.save(dict(params={k: v.cpu() for k, v in w1.items()}, cfg=cfg,
                    acc=b1.cpu(), good=good.cpu()), "phase1.pt")

    # replicate winners up to E2 and scatter unit-0 inits over the copies
    rep = (a.E2 + good.numel() - 1) // good.numel()
    idx = good.repeat(rep)[:a.E2]
    base = {k: v[idx].contiguous() for k, v in w1.items()}
    corners = lab.make_eval([8, 5, 11, 3], dev, per_n=1024)

    if a.grid:
        for lr2 in [0.002, 0.006, 0.015]:
            for fz in [0.0, 0.3, 0.8]:
                for pl in [[2, 3, 5, 8], [8, 5, 11, 3]]:
                    p2 = reinit_unit0(base, cfg, a.seed + 1)
                    t0 = time.time()
                    b2, _ = train(cfg, p2, keys, 2500, 512, lr2, a.seed + 3,
                                  corners, pl, freeze_code_frac=fz, pct=0.05)
                    print(f"lr2={lr2} freeze={fz} places={pl}: "
                          f"max {b2.max().item():.4f} "
                          f"#>=.99 {(b2>=0.99).sum().item():4d} "
                          f"#>=.999 {(b2>=0.999).sum().item():4d} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        return

    p2 = reinit_unit0(base, cfg, a.seed + 1)
    t0 = time.time()
    b2, w2 = train(cfg, p2, keys, a.p2_steps, 512, a.lr2, a.seed + 3, corners,
                   a.places2, freeze_code_frac=a.freeze, pct=0.05, log="p2")
    print(f"phase2: max {b2.max().item():.4f} "
          f"#>=.999 {(b2>=0.999).sum().item()} ({time.time()-t0:.0f}s)",
          flush=True)
    torch.save(dict(params={k: v.cpu() for k, v in w2.items()}, cfg=cfg,
                    acc=b2.cpu()), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
