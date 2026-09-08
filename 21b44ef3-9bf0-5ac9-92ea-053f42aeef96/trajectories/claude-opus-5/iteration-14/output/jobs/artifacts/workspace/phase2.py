"""Phase 2: give phase-1 members a second bank unit and train the carry routing.

Phase 1 (train.py on places 1,2 with U=1) learns the digit code, the generate
threshold, the carry write and the mod-10 fold -- everything except the "transparent"
notch that makes a place with a_i+b_i == 9 unattendable, which only matters from
three places up.  Phase 2 adds one randomly initialised bank unit per member,
replicates each parent R times with different random inits for it, and trains on
wide inputs.  The added unit is initialised at zero read-off weight so the starting
function is exactly the phase-1 function.
"""
import argparse, os, time
import torch

import arch, data, train as T


def expand(p, cfg, R, seed, device):
    """Replicate each member R times and append one fresh bank unit."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = {k: v.repeat_interleave(R, dim=0).contiguous() for k, v in p.items()}
    E = q["code"].shape[0]
    C = cfg["C"]
    # Data-dependent init: put the new knee somewhere in the range the residual
    # actually takes, with a slope within a couple of octaves of the existing unit.
    # Nothing about where the knee *should* sit is supplied; it has to be found.
    with torch.no_grad():
        code = arch.full_code(q, cfg)[:, :, 0]               # (E,10)
        x_lo = 2.0 * code.min(dim=1).values.clamp(max=0.0)
        x_hi = 2.0 * code.max(dim=1).values.clamp(min=0.0)
    knee = x_lo + (x_hi - x_lo) * torch.rand(E, generator=g, device=device)
    oct_ = torch.rand(E, generator=g, device=device) * 4.0 - 2.0        # +-2 octaves
    sign = torch.where(torch.rand(E, generator=g, device=device) < 0.5, -1.0, 1.0)
    slope = sign * q["Bw"][:, 0, 0].abs() * torch.pow(2.0, oct_)
    gw = slope.reshape(E, 1, 1).expand(E, C, 1).contiguous()
    gb = (-slope * knee).reshape(E, 1)
    q["Bw"] = torch.cat([q["Bw"], gw], dim=2).contiguous()
    q["bb"] = torch.cat([q["bb"], gb], dim=1).contiguous()
    q["kw"] = torch.cat([q["kw"], torch.zeros(E, 1, device=device)], dim=1).contiguous()
    q["vw"] = torch.cat([q["vw"], torch.zeros(E, 1, device=device)], dim=1).contiguous()
    cfg = dict(cfg)
    cfg["U"] = cfg["U"] + 1
    return q, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default="ckpt/phase1.pt")
    ap.add_argument("--keep", type=int, default=64)     # parents taken from phase 1
    ap.add_argument("--R", type=int, default=32)        # random restarts per parent
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--pct_start", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--places", default="8,5,11,3")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--out", default="ckpt/phase2.pt")
    a = ap.parse_args()

    device = "cuda"
    st = torch.load(a.init_from, map_location=device)
    src, cfg = st["p"], st["cfg"]
    order = torch.argsort(st["acc"], descending=True)[:a.keep]
    src = {k: v.to(device)[order] for k, v in src.items()}
    p, cfg = expand(src, cfg, a.R, a.seed + 77, device)
    E = p["code"].shape[0]
    for v in p.values():
        v.requires_grad_(True)
    print(f"phase2: {E} members ({a.keep} parents x {a.R} restarts), "
          f"free params/member={arch.count_free(p)}, cfg={cfg}", flush=True)

    places = [int(s) for s in a.places.split(",")]
    opt = torch.optim.AdamW(list(p.values()), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                              pct_start=a.pct_start)
    gen = torch.Generator(device=device).manual_seed(a.seed + 1)
    egen = torch.Generator(device=device).manual_seed(999)

    best = torch.zeros(E, device=device)
    best_p = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for step in range(a.steps):
        n = places[step % len(places)]
        ta, tb, tgt, _, _ = data.batch(a.batch, n, device, gen)
        dlog = arch.forward(p, ta, tb, cfg)
        loss, per = T.ce_loss(dlog, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        T.per_member_clip(p, a.clip)
        opt.step()
        sch.step()
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            acc = T.exact_match(p, cfg, 8, device, egen, N=2048, chunk=512)
            upd = acc > best
            if upd.any():
                for k in p:
                    sel = upd.reshape((-1,) + (1,) * (p[k].dim() - 1))
                    best_p[k] = torch.where(sel, p[k].detach(), best_p[k])
                best = torch.maximum(best, acc)
            print(f"step {step+1:6d} loss {per.mean():.4f} acc max {acc.max():.4f} "
                  f"best {best.max():.5f}  >=0.99 {int((best>=0.99).sum()):4d}  "
                  f"==1 {int((best>=1.0).sum()):4d}  {time.time()-t0:.0f}s", flush=True)

    # final, stricter ranking of the survivors
    cand = (best >= 0.99).nonzero(as_tuple=True)[0]
    if len(cand) == 0:
        cand = torch.argsort(best, descending=True)[:64]
    sub = {k: best_p[k][cand].contiguous() for k in best_p}
    acc2 = torch.zeros(len(cand), device=device)
    for n in (3, 5, 8, 11):
        acc2 += T.exact_match(sub, cfg, n, device, egen, N=16384, chunk=2048)
    acc2 /= 4
    order = torch.argsort(acc2, descending=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(p={k: v[order][:64].cpu() for k, v in sub.items()}, cfg=cfg,
                    acc=acc2[order][:64].cpu(), args=vars(a)), a.out)
    print(f"saved {a.out}: {len(cand)} survivors, top acc {acc2[order][:10].tolist()}")


if __name__ == "__main__":
    main()
