"""Train an ensemble of tiny adder transformers.  Weights are exported separately."""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F
import tiny


MASKS = {}


def masks_for(T, device, strict=True):
    key = (T, strict)
    if key not in MASKS:
        MASKS[key] = tiny.make_masks(device, T=T, strict=strict)
    return MASKS[key]


def evaluate(p, cfg, tab, rel, causal, device, n=200_000, bs=8192, seed=1234, stress=False):
    M = p["U"].shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    ok = torch.zeros(M, device=device)
    tot = 0
    with torch.no_grad():
        for _ in range(n // bs):
            rp = (0.0, 0.15, 0.25, 0.6) if stress else (1.0, 0.0, 0.0, 0.0)
            a, b, t = tiny.sample_batch(bs, device, gen, regime_p=rp, ndig=14)
            lg = tiny.forward(p, cfg, a, b, tab, rel, causal)
            pred = lg.argmax(-1)
            ok += (pred == t[None]).all(-1).float().sum(1)
            tot += bs
    return (ok / tot).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--head", default="linear")
    ap.add_argument("--d_model", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=4)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_qk_bias", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default="")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--noise_end", type=float, default=0.8)
    ap.add_argument("--slope0", type=float, default=2.0)
    ap.add_argument("--ndigs", default="14")
    ap.add_argument("--regime", default="0.3,0.25,0.2,0.25")
    ap.add_argument("--nonstrict", action="store_true")
    ap.add_argument("--hard_frac", type=float, default=0.0)
    ap.add_argument("--no_bk", action="store_true")
    ap.add_argument("--no_b2", action="store_true")
    ap.add_argument("--grid_slope", default="")   # e.g. "2,5,10,20": slope0 values across members
    ap.add_argument("--grid_noise", default="")   # e.g. "0,0.5,1,2": noise levels across members
    args = ap.parse_args()

    device = args.device
    cfg = tiny.Cfg(args.d_model, args.d_ff, args.head, use_qk_bias=not args.no_qk_bias,
                   use_k_bias=not args.no_bk,
                   use_b2=not args.no_b2)
    npar = tiny.param_count(cfg)
    p = tiny.init_params(cfg, args.M, device, seed=args.seed)
    p["slope"].data.fill_(args.slope0)
    gs = [float(x) for x in args.grid_slope.split(",")] if args.grid_slope else [args.slope0]
    gn = [float(x) for x in args.grid_noise.split(",")] if args.grid_noise else [args.noise]
    combos = [(a_, b_) for a_ in gs for b_ in gn]
    combos = [combos[i % len(combos)] for i in range(args.M)]
    p["slope"].data.copy_(torch.tensor([c[0] for c in combos], device=device))
    noise_vec = torch.tensor([c[1] for c in combos], device=device)
    if args.init:
        src = torch.load(args.init, map_location=device)
        for k, v in src["params"].items():
            if k in p and p[k].shape[1:] == v.shape[1:]:
                reps = [args.M] + [1] * (v.dim() - 1)
                p[k].data.copy_(v[:1].repeat(*reps) + torch.randn_like(p[k]) * 0.0)
    tab = tiny.pair_row_table(device)
    strict = not args.nonstrict
    rel, causal = tiny.make_masks(device, strict=strict)
    ndigs = [int(x) for x in args.ndigs.split(",")]
    reg = tuple(float(x) for x in args.regime.split(","))
    stream = tiny.Stream(args.bs, device, args.seed + 999, ndigs=ndigs, regime_p=reg)

    opt = torch.optim.AdamW(list(p.values()), lr=args.lr, betas=(0.9, 0.98),
                            weight_decay=0.0, fused=(device == "cuda"))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05, anneal_strategy="cos",
                                                div_factor=10.0, final_div_factor=100.0)

    ckdir = "/workspace/work/ckpt"
    os.makedirs(ckdir, exist_ok=True)
    logf = open(f"/workspace/work/log_{args.tag}.txt", "a")

    def say(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    say(f"=== {args.tag} {cfg} hard_frac={args.hard_frac} strict={strict} regime={args.regime} ndigs={ndigs} noise={args.noise} slope0={args.slope0} params={npar} M={args.M} steps={args.steps} bs={args.bs} lr={args.lr} seed={args.seed}")
    t0 = time.time()
    best = -1.0
    for step in range(1, args.steps + 1):
        a, b, t = stream.next()
        sig = noise_vec * max(0.0, 1.0 - step / (args.noise_end * args.steps))
        trel, tcausal = masks_for(a.shape[1], device, strict)
        lg = tiny.forward(p, cfg, a, b, tab, trel, tcausal, noise=sig,
                          hard=(step <= args.hard_frac * args.steps))
        loss = tiny.ce_loss(lg, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p.values()), 1.0)
        opt.step()
        sched.step()
        if args.bench and step == 200:
            torch.cuda.synchronize() if device == "cuda" else None
            say(f"bench: {(time.time()-t0)/200*1000:.1f} ms/step")
            return
        if step % 2000 == 0 or step == args.steps:
            accs = evaluate(p, cfg, tab, rel, causal, device, n=65536)
            sac = evaluate(p, cfg, tab, rel, causal, device, n=65536, stress=True)
            mi = max(range(args.M), key=lambda i: min(accs[i], sac[i]))
            sc = min(accs[mi], sac[mi])
            sc_all = [min(a_, s_) for a_, s_ in zip(accs, sac)]
            rank = sorted(range(args.M), key=lambda j: -sc_all[j])[:3]
            say(f"[{args.tag}] step {step} {time.time()-t0:.0f}s loss {loss.item():.4f} "
                f"best#{mi}(sl{combos[mi][0]:g},nz{combos[mi][1]:g}) unif {accs[mi]:.4f} stress {sac[mi]:.4f} | "
                + " ".join(f"#{j}(sl{combos[j][0]:g},nz{combos[j][1]:g})={sc_all[j]:.3f}" for j in rank))
            if sc > best:
                best = sc
                torch.save({"params": {k: v.detach().clone() for k, v in p.items()},
                            "cfg": vars(cfg), "npar": npar, "acc": accs, "sacc": sac,
                            "best_idx": mi, "step": step, "combos": combos},
                           f"{ckdir}/{args.tag}.pt")
    say(f"[{args.tag}] done best={best:.4f} params={npar} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
