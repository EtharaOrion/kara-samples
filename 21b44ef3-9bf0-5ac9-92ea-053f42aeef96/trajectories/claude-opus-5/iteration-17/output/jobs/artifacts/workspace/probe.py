"""Short exploratory runs to find a recipe that teaches carry ROUTING.

Single-place training already solves the code + mod-10 fold; what no plain
multi-place run has found so far is the transparency notch in the key.  These
probes vary the bank nonlinearity, the number of bank units, and the
curriculum (place count / transparency ramps, and warm starts from an n=1
phase)."""
import argparse
import time

import torch

import arch
import data
import lab


def curric(name, step, steps):
    """-> (n_places, regimes) for this step."""
    f = step / max(1, steps - 1)
    if name == "flat":
        return [8, 5, 11, 3][step % 4], (0.0, 0.4, 0.9)
    if name == "places":                      # grow the place count
        cap = 1 + int(f * 8)
        opts = [n for n in [1, 2, 3, 5, 8, 11] if n <= max(2, cap)]
        return opts[step % len(opts)], (0.0, 0.4, 0.9)
    if name == "trans":                       # grow the transparency rate
        r = min(1.0, f / 0.5)
        return [8, 5, 11, 3][step % 4], (0.0, 0.4 * r, 0.9 * r)
    if name == "both":
        cap = 1 + int(f * 8)
        opts = [n for n in [1, 2, 3, 5, 8, 11] if n <= max(2, cap)]
        r = min(1.0, f / 0.5)
        return opts[step % len(opts)], (0.0, 0.4 * r, 0.9 * r)
    raise ValueError(name)


def run(cfg, E, steps, batch, lr, seed, eval_places, cur="flat",
        init=None, reinit=(), s_code=1.0, s_bank=1.0, ls0=0.0, pct=0.1,
        keys=None):
    dev = "cuda"
    params = arch.init_params(E, cfg, dev, seed=seed, s_code=s_code,
                              s_bank=s_bank, ls0=ls0)
    if init is not None:
        for k, v in init.items():
            if k not in reinit:
                params[k] = v.detach().clone().to(dev)
    keys = keys or arch.trainable_keys(cfg)
    for k in keys:
        params[k].requires_grad_(True)
    opt = torch.optim.AdamW([params[k] for k in keys], lr=lr, weight_decay=0.0,
                            betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                              pct_start=pct)
    gen = torch.Generator(device=dev).manual_seed(seed + 999)
    corners = lab.make_eval(eval_places, dev, per_n=1024)
    best = torch.zeros(E, device=dev)
    best_w = {k: params[k].detach().clone() for k in params}
    for step in range(steps):
        n, regimes = curric(cur, step, steps)
        da, db = data.sample(batch, n, dev, regimes=regimes, gen=gen)
        tok, tgt = data.tokens(da, db), data.targets(da, db)
        loss = lab.ce_loss(arch.forward(params, cfg, tok), tgt)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        lab.clip_per_member(params, keys, 1.0)
        opt.step()
        sch.step()
        if (step + 1) % 500 == 0 or step == steps - 1:
            acc = lab.eval_exact(params, cfg, corners)
            imp = acc > best
            if imp.any():
                for k in params:
                    best_w[k] = torch.where(
                        imp.view(-1, *([1] * (params[k].dim() - 1))),
                        params[k].detach(), best_w[k])
                best = torch.maximum(best, acc)
    return best, best_w


def report(tag, best, t0):
    print(f"{tag:52s} max {best.max().item():.4f}  "
          f"#>=.99 {(best>=0.99).sum().item():3d}  "
          f"#>=.5 {(best>=0.5).sum().item():4d}  ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="act")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    EV = [8, 5, 11, 3]

    if a.mode == "act":
        for act, U in [("clamp", 2), ("sigmoid", 2), ("sigmoid", 4),
                       ("relu", 3), ("relu", 6)]:
            for cur in ["flat", "places", "trans", "both"]:
                cfg = arch.default_cfg(C=2, U=U, act=act)
                t0 = time.time()
                best, _ = run(cfg, a.E, a.steps, 512, 0.012, 0, EV, cur=cur)
                report(f"act={act} U={U} cur={cur}", best, t0)

    if a.mode == "warm":
        # phase 1: n=1 only, then warm start into multi-place
        cfg = arch.default_cfg(C=2, U=2, act="clamp")
        t0 = time.time()
        b1, w1 = run(cfg, a.E, 2000, 512, 0.012, 0, [1], cur="flat")
        report("phase1 n=1", b1, t0)
        keep = torch.argsort(b1, descending=True)[:a.E // 4]
        init = {k: v[keep].repeat(4, *([1] * (v.dim() - 1))) for k, v in w1.items()}
        for reinit in [(), ("bb",), ("bb", "kw"), ("bb", "kw", "Wb")]:
            for lr in [0.003, 0.012]:
                t0 = time.time()
                b2, _ = run(cfg, a.E, a.steps, 512, lr, 7, EV, cur="flat",
                            init=init, reinit=reinit, pct=0.02)
                report(f"phase2 reinit={reinit} lr={lr}", b2, t0)
