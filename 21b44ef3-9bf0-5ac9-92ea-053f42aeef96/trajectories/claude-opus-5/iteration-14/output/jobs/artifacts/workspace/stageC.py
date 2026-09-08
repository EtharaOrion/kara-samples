"""Stage C: train the shipped form itself.

Stages A and B produce a full model with 24 free values per member.  Most of those
turn out to be gauge or sharpness: the code can be rescaled at will (the read-out is
a positive-homogeneous score), the query scale folds into the key weights, and the
clamp slope / key magnitude / recency slope only have to be "sharp enough".  So
instead of training a big parent and then arguing about which of its numbers were
really fitted, stage C fixes all of that and trains only what is left:

    learned (12):  code[1..9]        9   the digit embedding, tied as read-out prototypes
                   bb[0..1]          2   the two clamp knees
                   w2                1   the fold applied on a carry-out

    fixed        :  code[0] = 0            origin of the code
                    Bw   = (8, 8)          clamp slope  -- sharpness
                    kw   = (-400, +400)    key read-off -- notch depth
                    vw   = (0, 1)          the value is bank unit 1
                    q = 1, ls = 1, w1 = 1  gauge (code scale, logit scale)
                    lam  = -12             recency slope -- sharpness
                    vb = rb = 0            no offsets

Nothing in the fixed set depends on base ten or on where the knees belong.  Stage C
starts from a stage-B model's own code and its own measured half-crossings
(extract.read_off), jittered, and refines all 12 by gradient descent; the fold is
drawn over a wide band of octaves with a random sign and learned from there.

Sharpness is annealed: the clamp slope, key magnitude and recency slope start soft
and reach their shipped values at `sharp_frac` of the run, so the last part of
training optimises the shipped function exactly as it ships (leak 0, constants final).
"""
import argparse, math, os, time
import torch

import arch, data, extract
from train2 import ce_loss, evaluate, per_member_clip

SLOPE, KEY, LAM = 8.0, 400.0, -12.0
SLOPE0, KEY0, LAM0 = 1.0, 4.0, -2.0
NOCARRY = dict(mix=(1.0, 0.0, 0.0), regimes=("nocarry", "uniform", "chain"),
               full_width=False)
CFG = dict(C=1, U=2, code0_fixed=True, lam_init=LAM, q_init=1.0, ls_init=1.0)


def constants(E, device, t):
    """t in [0,1]: 0 = soft start, 1 = the shipped constants exactly."""
    slope = math.exp(math.log(SLOPE0) + t * (math.log(SLOPE) - math.log(SLOPE0)))
    key = math.exp(math.log(KEY0) + t * (math.log(KEY) - math.log(KEY0)))
    lam = LAM0 + t * (LAM - LAM0)
    o = torch.ones(E, device=device)
    return dict(Bw=torch.full((E, 1, 2), slope, device=device),
                kw=torch.stack([-key * o, key * o], 1),
                vw=torch.stack([0.0 * o, o], 1),
                vb=torch.zeros(E, device=device),
                rb=torch.zeros(E, 1, device=device),
                q=o, lam=lam * o, ls=o, w1=o.reshape(E, 1)), slope


def init(codes, knees, R, seed, device, jitter=0.15):
    """Start each member from a stage-B model's own measured quantities: its code
    rescaled to unit step (the w1 = 1 gauge) and the two points where its bank units
    actually cross one half (see extract.read_off).  Both are read off the trained
    model.  R restarts per member differ by a small jitter on the knees and by the
    fold, which is drawn over a wide band of octaves with a random sign."""
    g = torch.Generator(device=device).manual_seed(seed)
    code = codes.repeat_interleave(R, 0).contiguous()             # (E,9,1), unit step
    E = code.shape[0]
    knee = knees.repeat_interleave(R, 0).contiguous()             # (E,2)
    knee = knee + jitter * torch.randn(E, 2, generator=g, device=device)
    w2 = (torch.pow(2.0, -2.0 + 7.0 * torch.rand(E, 1, generator=g, device=device))
          * torch.where(torch.rand(E, 1, generator=g, device=device) < 0.5, -1.0, 1.0))
    return dict(code=code.contiguous(), knee=knee.contiguous(), w2=w2.contiguous())


@torch.no_grad()
def margin(p, cfg, n, device, gen, N=2048, chunk=1024):
    """Worst-case gap, over the sampled pairs, between the score of the right digit
    and the best wrong one, in code-step units (ls = 1).  Exact-match accuracy is
    blind to how close a correct answer came to being wrong; this is not, so it is
    what separates two members that are both already perfect."""
    E = p["code"].shape[0]
    worst = torch.full((E,), 1e9, device=device)
    tot = 0
    while tot < N:
        b = min(chunk, N - tot)
        ta, tb, tgt, _, _ = data.batch(b, n, device, gen, heldout=True)
        lg = arch.forward(p, ta, tb, cfg)                       # (E,b,P,10)
        t_ = tgt[None, :, :, None].expand(E, -1, -1, 1)
        right = lg.gather(-1, t_).squeeze(-1)
        wrong = lg.scatter(-1, t_, -1e30).max(-1).values
        worst = torch.minimum(worst, (right - wrong)[:, :, 1:].amin((1, 2)))
        tot += b
    return worst


def assemble(free, E, device, t, ls=1.0):
    c, slope = constants(E, device, t)
    if ls != 1.0:
        # A positive logit scale cannot change an argmax, so this leaves the function
        # the model computes untouched; it only lowers the softmax temperature the
        # loss sees, which keeps cross-entropy pushing for margin after the
        # predictions are already correct.  Evaluation and the shipped file use 1.
        c["ls"] = c["ls"] * ls
    c["code"] = free["code"]
    c["bb"] = -slope * free["knee"]        # knee is the learned quantity; bb = -slope*knee
    c["w2"] = free["w2"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default="ckpt/B21.pt")
    ap.add_argument("--codes", type=int, default=32)
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.05)        # knees and fold
    ap.add_argument("--code_lr", type=float, default=0.002)  # the code is already close
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--leak", type=float, default=0.05)
    ap.add_argument("--train_ls", type=float, default=8.0)
    ap.add_argument("--end_leak", type=float, default=0.0)
    ap.add_argument("--leak_frac", type=float, default=0.5)
    ap.add_argument("--sharp_frac", type=float, default=0.7)
    ap.add_argument("--places", default="2,3,4,6,8")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_N", type=int, default=2048)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/C.pt")
    a = ap.parse_args()
    device = "cuda"

    st = torch.load(a.init_from, map_location=device)
    src = {k: v.to(device) for k, v in st["p"].items()}
    r = extract.read_off(src, st["cfg"], device)
    k = r["knee"]
    # only members whose bank really has two separated crossings can be expressed in
    # the shipped form at all; the rest are dropped
    usable = (k[:, 1] - k[:, 0] > 0.2).nonzero(as_tuple=True)[0][:a.codes]
    print(f"usable stage-B members: {len(usable)} of {k.shape[0]}")
    free = init(r["code_unit"][usable], k[usable], a.R, a.seed + 31, device)
    E = free["code"].shape[0]
    for v in free.values():
        v.requires_grad_(True)
    print(f"stage C: {E} members ({len(usable)} stage-B members x {a.R} draws), "
          f"12 learned values each", flush=True)

    # the knees have to travel across a range of ~18 code steps while the code only
    # needs refining, so they get very different step sizes
    groups = [{"params": [free["knee"], free["w2"]], "lr": a.lr},
              {"params": [free["code"]], "lr": a.code_lr}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.code_lr],
                                              total_steps=a.steps, pct_start=0.1)
    gen = torch.Generator(device=device).manual_seed(a.seed + 1)
    egen = torch.Generator(device=device).manual_seed(12345)
    places = [int(s) for s in a.places.split(",")]

    best = torch.zeros(E, device=device)
    best_free = {k: v.detach().clone() for k, v in free.items()}
    t0 = time.time()
    for step in range(a.steps):
        t = min(1.0, step / (a.sharp_frac * a.steps))
        leak = a.end_leak + (a.leak - a.end_leak) * max(
            0.0, 1.0 - step / (a.leak_frac * a.steps))
        p = assemble(free, E, device, t, ls=a.train_ls)
        if step % 3 == 2:
            # a carry-free batch every third step.  The shipped bank only separates
            # place sums cleanly if code[a]+code[b] depends on a+b alone, and this is
            # the objective that enforces that; without it the ramp drifts.
            ta, tb, tgt, _, _ = data.batch(a.batch, 3, device, gen, **NOCARRY)
        else:
            n = places[step % len(places)]
            ta, tb, tgt, _, _ = data.batch(a.batch, n, device, gen)
        loss, per = ce_loss(
            arch.forward(p, ta, tb, CFG, leak=leak, leak_sat=True), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        per_member_clip(free, a.clip)
        opt.step()
        sch.step()
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                pe = assemble({k: v.detach() for k, v in free.items()}, E, device, 1.0)
                acc, dig = evaluate(pe, CFG, 8, device, egen, N=a.eval_N)
                mgn = margin(pe, CFG, 8, device,
                             torch.Generator(device=device).manual_seed(99), N=1024)
                # accuracy first; among members that are already exact, the wider
                # margin wins.  The bonus is capped below one eval sample so it can
                # never outrank a genuine accuracy difference.
                score = acc + (acc >= 1.0) * 0.001 * mgn.clamp(0.0, 0.5) / 0.5
            upd = score > best
            if upd.any():
                for k in free:
                    sel = upd.reshape((-1,) + (1,) * (free[k].dim() - 1))
                    best_free[k] = torch.where(sel, free[k].detach(), best_free[k])
                best = torch.maximum(best, score)
            print(f"  C {step+1:6d} t {t:.2f} leak {leak:.4f} loss {per.mean():.4f} | "
                  f"digit {dig.max():.4f} exact {acc.max():.4f} margin {mgn.max():.4f} "
                  f"best {best.max():.5f} "
                  f">=.99 {int((best>=0.99).sum()):5d} ==1 {int((best>=1.0).sum()):5d} | "
                  f"{time.time()-t0:.0f}s", flush=True)

    # rank survivors on the shipped constants across widths
    cand = (best >= 0.99).nonzero(as_tuple=True)[0]
    if len(cand) == 0:
        cand = torch.argsort(best, descending=True)[:a.keep]
    sub = {k: best_free[k][cand].contiguous() for k in best_free}
    pe = assemble(sub, len(cand), device, 1.0)
    tot = torch.zeros(len(cand), device=device)
    for n in (2, 3, 5, 8, 11, 15):
        acc, _ = evaluate(pe, CFG, n, device, egen, N=16384, chunk=2048)
        tot += acc
        print(f"  final n={n:2d}: #==1 {int((acc>=1.0).sum())}  max {acc.max():.6f}")
    tot /= 6
    order = torch.argsort(tot, descending=True)[:a.keep]
    os.makedirs("ckpt", exist_ok=True)
    torch.save(dict(free={k: v[order].cpu() for k, v in sub.items()}, cfg=CFG,
                    acc=tot[order].cpu(), args=vars(a)), a.out)
    print(f"saved {a.out}: {len(cand)} survivors, top {tot[order][:8].tolist()}")


if __name__ == "__main__":
    main()
