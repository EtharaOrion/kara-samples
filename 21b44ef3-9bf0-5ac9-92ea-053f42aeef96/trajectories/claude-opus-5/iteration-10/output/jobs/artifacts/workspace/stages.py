"""The reduction ladder.

Each stage anneals part of the block onto a value that the task leaves free --
a gauge freedom, or a quantity the answer does not depend on -- and then turns
that part into a buffer.  Nothing that the answer depends on is ever pinned:
the digit code, the two bank thresholds and the fold weight stay learned.

Run as ``python stages.py <stage> --inp ck/a.pt --out ck/b.pt``.
"""
import argparse

import torch

import certify
import cut
import lab
import look

# Three fixed scales.  None of them carries information about addition: each
# only has to sit inside a wide band for the block to work, and the value used
# is the middle of that band rather than a tuned constant.
M_KEY = 2000.0    # notch scale; anything that clears |lam| * places will do
LAM = -8.0        # relative-position slope; any moderate negative value works
SLOPE = 2.0       # bank input slope; any value steep enough to saturate works


# --------------------------------------------------------------------------
def profile(cfg, sd):
    """Mean bank output over digit pairs with s <= 8, s == 9 and s >= 10."""
    m = cut.model_of(cfg, sd)
    code = torch.cat([m.code_fix, m.code], 0).detach()
    aa, bb = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    x = code[aa.reshape(-1)] + code[bb.reshape(-1)]
    u = torch.clamp(x @ m.bw.detach().t() + m.bb.detach(), 0, 1).double()
    s = (aa + bb).reshape(-1)
    spread = max(float(u[s <= 8].std(0).max()), float(u[s >= 10].std(0).max()))
    return u[s <= 8].mean(0), u[s == 9].mean(0), u[s >= 10].mean(0), spread


def kv_targets(cfg, sd, M=M_KEY):
    """Key and value read-outs written in their canonical form.

    value: 0 where the place generates no carry, 1 where it does (its value at a
    transparent place is a don't-care -- those places are never attended).
    key:   flat, with a notch of depth M at the transparent places.  Only
    differences of keys matter, so the notch level is solved for jointly.
    """
    lo, tr, hi, spread = profile(cfg, sd)
    assert spread < 1e-3, f"bank units are not saturated (spread {spread:.2e})"
    vw = torch.linalg.solve(torch.stack([lo, hi]), torch.tensor([0.0, 1.0], dtype=torch.float64))
    A = torch.cat([torch.stack([lo, tr, hi]), -torch.ones(3, 1, dtype=torch.float64)], 1)
    kw = torch.linalg.solve(A, torch.tensor([0.0, -M, 0.0], dtype=torch.float64))[:-1]
    return kw.float(), vw.float()


def score(cfg, sd, batches=4, B=16384, seed=77):
    m = cut.model_of(cfg, sd, dev=lab.DEV)
    au, _ = lab.model_acc(m, n=8, batches=batches, B=B, seed=seed, mix=(1.0, 0.0, 0.0))
    as_, _ = lab.model_acc(m, n=8, batches=batches, B=B, seed=seed + 1)
    a11, _ = lab.model_acc(m, n=11, batches=1, B=B, seed=seed + 2)
    return min(au, as_, a11), au, as_, a11


def cert_ratio(cfg, sd):
    """How far the worst place-level margin clears the attention leakage bound."""
    try:
        c = certify.audit(cut.model_of(cfg, sd), n=8, verbose=False)
    except AssertionError:
        return 0.0
    return c["ratio"] if c["bad"] == 0 else 0.0


def finish(cfg, ens, order, cut_fn, out, keep=8, tag=""):
    """Apply the structural cut to the best members; save the best result.

    Members are ranked first on held-out accuracy and then on the certificate
    ratio, so among equally exact models the most robust one is kept.
    """
    best = None
    for i in order[:keep].tolist():
        sd = ens.member_sd(i)
        s0 = score(cfg, sd)[0]
        cfg2, sd2 = cut_fn(cfg, sd)
        ag, _ = cut.equiv(cfg, sd, cfg2, sd2)
        s, au, as_, a11 = score(cfg2, sd2)
        cr = cert_ratio(cfg2, sd2)
        print(f"  member {i:3d}: pre {s0:.6f} -> cut {s:.6f} "
              f"(uni {au:.6f} str {as_:.6f} n11 {a11:.6f})  agreement {ag:.6f}  "
              f"certificate {cr:6.1f}x", flush=True)
        if best is None or (s, cr) > (best[0], best[6]):
            best = (s, cfg2, sd2, i, au, as_, cr)
    s, cfg2, sd2, i, au, as_, cr = best
    lab.save_ckpt(out, cfg2, sd2,
                  note=f"{tag} member {i}: uniform {au:.6f} struct {as_:.6f} cert {cr:.1f}x")
    print(f"[{tag}] saved {out}  params={lab.n_params(cfg2)}  score={s:.6f}  "
          f"certificate {cr:.1f}x", flush=True)
    return cfg2, sd2


def go(a, cfg, sd, specs, cut_fn, tag):
    ens, order, sc, au, as_ = cut.run_stage(cfg, sd, specs, E=a.E, sigma=a.sigma, steps=a.steps,
                                            lr=a.lr, tag=tag, seed=a.seed, log_every=a.steps // 8,
                                            eval_every=a.steps // 4, mgn=a.mgn, ce=a.ce, tau=a.tau,
                                            mnorm=not a.mabs)
    return finish(cfg, ens, order, cut_fn, a.out, tag=tag)


# --------------------------------------------------------------------------
def stage_A(a):
    """C=2 -> C=1, and canonicalise the attention.

    The code already lies on a line; rotating onto it and zeroing the
    perpendicular coordinate makes that axis contribute the same constant to
    every read-out distance, so it can be dropped.  The position slope, the
    notch depth and the value normalisation are all free scales.
    """
    cfg, sd, _ = lab.load_ckpt(a.inp)
    cfg, sd = cut.gauge_rotate(cfg, sd)
    t = cut.code_of(cfg, sd).mean(0).clone()
    t[0] = 0.0
    cfg, sd = cut.gauge_translate(cfg, sd, t)
    ag, dm = cut.equiv(*lab.load_ckpt(a.inp)[:2], cfg, sd)
    print(f"gauge rotate+translate: agreement {ag:.6f}, max logit diff {dm:.3e}", flush=True)

    kw, vw = kv_targets(cfg, sd, M=a.M)
    print(f"canonical kw {kw.tolist()}  vw {vw.tolist()}", flush=True)
    mask = torch.zeros(10 - cfg["code_fix"], cfg["C"])
    mask[:, 1:] = 1.0
    specs = [("code", 0.0, mask, 0.0, 0.45),
             ("lam", a.lam, None, 0.05, 0.5),
             ("vw", vw, None, 0.1, 0.6),
             ("kw", kw, None, 0.2, 0.7)]
    go(a, cfg, sd, specs,
       lambda c, s: cut.pin(*cut.pin(*cut.pin(*cut.drop_axis(c, s, 1), "lam"), "kw"), "vw"), "A")


def stage_B(a):
    """Fix the code-translation gauge: code[0] = 0 and no residual constant.

    With c[0] = 0, exactness on the carry-free places forces c[i] + c[j] =
    c[i+j], so the residual constant has to vanish; it is annealed there and
    dropped along with the pinned row.
    """
    cfg, sd, _ = lab.load_ckpt(a.inp)
    cfg, sd = cut.gauge_translate(cfg, sd, cut.code_of(cfg, sd)[0].clone())
    ag, dm = cut.equiv(*lab.load_ckpt(a.inp)[:2], cfg, sd)
    print(f"gauge translate to code[0]=0: agreement {ag:.6f}, max logit diff {dm:.3e}", flush=True)
    m0 = torch.zeros(10 - cfg["code_fix"], cfg["C"])
    m0[0] = 1.0
    specs = [("code", 0.0, m0, 0.0, 0.05), ("rb", 0.0, None, 0.0, 0.6)]
    go(a, cfg, sd, specs, lambda c, s: cut.fix_code_rows(*cut.drop_rb(c, s), 1), "B")


def stage_C(a):
    """Fix the code-scale gauge with the carry write weight, and pin the bank
    slopes -- their steepness is a don't-care as long as the units saturate."""
    cfg, sd, _ = lab.load_ckpt(a.inp)
    e1 = float(sd["e1"])
    assert e1 > 0, "code orientation should make the carry weight positive"
    cfg, sd = cut.gauge_scale(cfg, sd, 1.0 / e1)
    ag, dm = cut.equiv(*lab.load_ckpt(a.inp)[:2], cfg, sd)
    print(f"gauge scale by {1/e1:.5f} (e1 -> 1): agreement {ag:.6f}, max logit diff {dm:.3e}",
          flush=True)
    specs = [("e1", torch.ones(cfg["C"]), None, 0.0, 0.01),
             ("bw", a.slope * torch.sign(sd["bw"]), None, a.bw_start, a.bw_end)]
    go(a, cfg, sd, specs, lambda c, s: cut.pin(*cut.pin(c, s, "bw"), "e1"), "C")


def stage_D(a):
    """The read-out temperature only scales the logits: pin it to 1 (exact)."""
    cfg, sd, _ = lab.load_ckpt(a.inp)
    assert float(sd["ls"]) > 0
    cfg2, sd2 = cut.pin(cfg, sd, "ls", 1.0)
    ag, _ = cut.equiv(cfg, sd, cfg2, sd2)
    print(f"pin ls=1: argmax agreement {ag:.6f}", flush=True)
    assert ag == 1.0
    s, au, as_, a11 = score(cfg2, sd2)
    lab.save_ckpt(a.out, cfg2, sd2, note=f"ls pinned: uniform {au:.6f} struct {as_:.6f}")
    print(f"[D] saved {a.out} params={lab.n_params(cfg2)} score={s:.6f}", flush=True)


def stage_R(a):
    """Re-train in place: same shape, fresh seed lottery around the parent."""
    cfg, sd, _ = lab.load_ckpt(a.inp)
    go(a, cfg, sd, [], lambda c, s: (c, s), "R")


def stage_P(a):
    """Polish: widen the worst-case read-out margin at the final shape."""
    cfg, sd, _ = lab.load_ckpt(a.inp)
    go(a, cfg, sd, [], lambda c, s: (c, s), "P")


STAGES = {"A": stage_A, "B": stage_B, "C": stage_C, "D": stage_D, "R": stage_R, "P": stage_P}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=float, default=M_KEY)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--slope", type=float, default=SLOPE)
    ap.add_argument("--mgn", type=float, default=0.0)
    ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--mabs", action="store_true",
                    help="target the absolute margin instead of the gap-normalised one")
    ap.add_argument("--bw_start", type=float, default=0.15)
    ap.add_argument("--bw_end", type=float, default=0.6)
    args = ap.parse_args()
    STAGES[args.stage](args)
    cfg, sd, note = lab.load_ckpt(args.out)
    look.show(cfg, sd, tag=f"{args.out} [{note}]")
