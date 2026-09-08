"""Generic retraining stage: seed an ensemble from trained members, then train.

Used for every rung of the ladder after the cold parent:

  --project    collapse a two-channel member onto one channel first (a re-init,
               so it must be followed by real training -- which is the point)
  --set k=v    replace a parameter group by a constant
  --freeze k   hold a parameter group fixed during training
  --shipped    put the member into the exact twelve-parameter shipped form
               (exact rewrites + constant substitution) and train only the
               twelve free values under exactly the shipped constants

Snapshots are kept per member on (exact-match, then worst read-out margin), so
members that already reach 1.0 keep improving their margin instead of freezing.
"""
import argparse, json, os, time
import torch

import data, ens, reduce as red


def project(p2):
    """Two channels -> one, along the principal direction of the ten codes."""
    code = p2["code"]
    m = code.mean(0, keepdim=True)
    _, s_, v_ = torch.linalg.svd(code - m, full_matrices=False)
    u, w = v_[0], v_[1]
    perp = code @ w
    e0 = float(perp.mean())
    p1 = {k: v.clone() for k, v in p2.items()}
    p1["code"] = (code @ u)[:, None]
    p1["bank_w"] = (p2["bank_w"] @ u)[:, None]
    p1["knee"] = p2["knee"] + (p2["bank_w"] @ w) * (2.0 * e0)
    p1["carry_w"] = (p2["carry_w"] @ u).reshape(1)
    p1["fold_w"] = (p2["fold_w"] @ u).reshape(1)
    p1["rb"] = (p2["rb"] @ u).reshape(1)
    return p1, float((perp - e0).abs().max()), float(s_[1] / s_[0])


def to_shipped_form(p, ta, tb):
    p_ex, log_a = red.exact_rewrites(p, ta, tb)
    p_sub, log_b = red.substitute(p_ex, ta, tb)
    red.to_shipped(p_sub)                       # raises if the form is wrong
    return p_sub, log_a + log_b, red.diagnose(p_ex)


def spread(p1, E, sigma, dev, seed, jitter):
    g = torch.Generator(device=dev).manual_seed(seed)
    out = {}
    for k, v in p1.items():
        v = v.to(device=dev, dtype=torch.float32)
        st = v[None].repeat(E, *([1] * v.dim())).contiguous()
        if E > 1 and sigma > 0 and k in jitter:
            n = torch.randn(st.shape, generator=g, device=dev)
            st[1:] += n[1:] * (v.abs().mean().clamp(min=1e-3) * sigma)
        out[k] = st
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--members", default="", help="comma list; default = top")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--thresh", type=float, default=0.999)
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--shipped", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--freeze", default="")
    ap.add_argument("--jitter", default="code,knee,fold_w,bank_w,key_w,val_w,carry_w,rb")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--places", default="8,5,11,3")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--aux", type=float, default=0.0,
                    help="weight on the bank-saturation objective")
    ap.add_argument("--slack", type=float, default=1.0,
                    help="how far outside [0,1] the bank is asked to sit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    pr = {k: v.to(dev) for k, v in ck["params"].items()}
    acc, order = ck["acc"].to(dev), ck["order"].to(dev)
    if args.members:
        cand = [int(s) for s in args.members.split(",")]
    else:
        cand = [int(i) for i in order if float(acc[int(i)]) >= args.thresh][:args.top]
    assert cand, "no member cleared the threshold"
    print(f"seeding from {len(cand)} members: {cand}")

    places = [int(s) for s in args.places.split(",")]
    g0 = torch.Generator(device=dev).manual_seed(5150)
    probe_a, probe_b = data.sample(4096, 8, g0, dev)[:2]

    notes = []
    seeds = []
    for m in cand:
        p = ens.single(pr, m)
        note = dict(member=m)
        if args.project:
            p, sp, ratio = project(p)
            note.update(offline_spread=sp, singular_ratio=ratio)
        if args.shipped:
            p, log, diag = to_shipped_form(p, probe_a, probe_b)
            note.update(rewrite_log=log, parent_diagnosis=diag)
        seeds.append(p)
        notes.append(note)
        print(f"  member {m}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in note.items() if isinstance(v, float)))

    per = max(1, args.E // len(seeds))
    jit = set(args.jitter.split(","))
    stacked, tags = {}, []
    for m, p in zip(cand, seeds):
        e = spread(p, per, args.sigma, dev, args.seed + m, jit)
        tags += [m] * per
        for k, v in e.items():
            stacked.setdefault(k, []).append(v)
    prm = {k: torch.cat(v, 0) for k, v in stacked.items()}

    for kv in args.set:
        k, v = kv.split("=", 1)
        vals = [float(x) for x in v.split(",")]
        t = torch.tensor(vals, device=dev, dtype=torch.float32)
        prm[k] = t.reshape(1, *prm[k].shape[1:]).repeat(prm[k].shape[0],
                                                        *([1] * (prm[k].dim() - 1)))
        print(f"  set {k} = {vals}")
    frozen = set(x for x in args.freeze.split(",") if x)
    if args.shipped:
        frozen |= {"bank_w", "key_w", "val_w", "val_b", "q", "lam",
                   "carry_w", "rb", "ls_log"}
    live = []
    for k, v in prm.items():
        if k in frozen:
            v.requires_grad_(False)
        else:
            v.requires_grad_(True)
            live.append(v)
    E = prm["code"].shape[0]
    print(f"training {E} copies; free groups: "
          f"{[k for k in prm if k not in frozen]}")

    # code[0] stays pinned at zero in the shipped form: no jitter, no gradient
    pin_code0 = args.shipped
    if pin_code0:
        with torch.no_grad():
            prm["code"][:, 0, :] = 0.0

    opt = torch.optim.AdamW(live, lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps,
                                                pct_start=0.1)
    g = torch.Generator(device=dev).manual_seed(args.seed + 4242)
    eg = torch.Generator(device=dev).manual_seed(777)
    evalset = [data.heldout(8192, n, eg, dev) for n in places]

    best_a = torch.zeros(E, device=dev)
    best_m = torch.full((E,), -1e9, device=dev)
    best = {k: v.detach().clone() for k, v in prm.items()}
    t0 = time.time()
    for step in range(args.steps):
        n = places[step % len(places)]
        ta, tb, tg, keep = data.sample(args.batch, n, g, dev)
        loss, _ = ens.loss_and_acc(prm, ta, tb, tg, keep, tau=args.tau)
        if args.aux > 0:
            # ask the bank for a decisive (saturated) answer at every place;
            # this says nothing about where the thresholds should sit, only
            # that they should not sit on top of a reachable digit sum.
            z = ens.bank_z(prm, ta, tb)
            d = torch.maximum(-z, z - 1.0)
            loss = loss + args.aux * torch.relu(args.slack - d).mean((1, 2, 3))
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        with torch.no_grad():
            if pin_code0 and prm["code"].grad is not None:
                prm["code"].grad[:, 0, :] = 0.0
            sq = torch.zeros(E, device=dev)
            for v in live:
                sq += (v.grad ** 2).flatten(1).sum(1)
            sc = (1.0 / sq.sqrt().clamp(min=1e-12)).clamp(max=1.0)
            for v in live:
                v.grad *= sc.view(-1, *([1] * (v.dim() - 1)))
        opt.step()
        sched.step()

        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            with torch.no_grad():
                aa, mm = [], []
                for e in evalset:
                    a, mg, _ = ens.acc_and_margin(prm, *e)
                    aa.append(a); mm.append(mg)
                a = torch.stack(aa).min(0).values
                mg = torch.stack(mm).min(0).values
                imp = (a > best_a) | ((a >= best_a) & (mg > best_m))
                best_a = torch.where(imp, a, best_a)
                best_m = torch.where(imp, mg, best_m)
                for k, v in prm.items():
                    best[k] = torch.where(imp.view(-1, *([1] * (v.dim() - 1))),
                                          v.detach(), best[k])
            top = best_m.masked_fill(best_a < 0.9999, -1e9).max().item()
            print(f"step {step+1:6d} loss {loss.mean().item():.4f} "
                  f"acc {best_a.max().item():.6f} "
                  f"n>=0.9999 {(best_a>=0.9999).sum().item():4d} "
                  f"best margin {top:.4f} {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rank = torch.argsort(best_a * 1e6 + best_m.clamp(min=-1e5), descending=True)
    torch.save(dict(cfg=dict(C=best["code"].shape[2], U=best["knee"].shape[1]),
                    params={k: v.cpu() for k, v in best.items()},
                    acc=best_a.cpu(), margin=best_m.cpu(), order=rank.cpu(),
                    tags=tags, notes=notes, args=vars(args)), args.out)
    n_ok = int((best_a >= 0.9999).sum())
    print(f"saved {args.out}; {n_ok} members >= 0.9999")


if __name__ == "__main__":
    main()
