"""Exhaustive place-level audit of a submission.

The read-out at place ``i`` is ``r_i = x_i + oa_i e1 + ob_i e2 (+ rb)`` with
``x_i = code[a_i] + code[b_i]``.  The key carries a deep notch at the
carry-transparent places (a + b = 9) and is otherwise flat, so each head's
softmax concentrates on the nearest *non*-transparent place -- below ``i`` for
the strictly causal head, at or below ``i`` for the inclusive one.  That makes
``oa_i`` the carry into place ``i`` and ``ob_i`` the carry out of it, up to a
leakage that this script bounds.

Consequently the read-out depends on the whole input only through
``(a_i, b_i, carry_in)``: 200 cases, every one of them checked here.  If the
smallest read-out margin over those 200 cases exceeds the leakage bound, the
model is exact on *every* input of that width -- not just the sampled ones.
"""
import argparse
import math

import torch

import verify


def pieces(model):
    g = {k: v.detach().double() for k, v in
         list(model.named_parameters()) + list(model.named_buffers())}
    code = torch.cat([g["code_fix"], g["code"]], 0)
    rb = g["rb"] if "rb" in g else torch.zeros(code.shape[1], dtype=torch.float64)
    return code, g["bw"], g["bb"], g["kw"], g["vw"], g["e1"], g["e2"], rb, g["lam"], g["ls"]


def bank(model):
    """Key, value and carry class for every one of the 100 digit pairs."""
    code, bw, bb, kw, vw = pieces(model)[:5]
    a, b = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    x = code[a] + code[b]
    u = torch.clamp(x @ bw.t() + bb, 0.0, 1.0)
    s = a + b
    cls = torch.where(s <= 8, 0, torch.where(s == 9, 1, 2))     # absorb / transparent / generate
    return u, u @ kw, u @ vw, cls, x, s


def signed_margin(r, code, d):
    """Distance from r to the nearest read-out decision boundary (code units)."""
    diff = code[d][:, None, :] - code[None, :, :]               # (N, 10, C)
    num = ((r[:, None, :] - (code[d][:, None, :] + code[None, :, :]) / 2) * diff).sum(-1)
    den = diff.norm(dim=-1)
    m = torch.where(den > 1e-12, num / den.clamp(min=1e-12), torch.full_like(num, math.inf))
    return m.amin(-1)


def audit(model, n=8, verbose=True):
    say = print if verbose else (lambda *x, **k: None)
    code, bw, bb, kw, vw, e1, e2, rb, lam, ls = pieces(model)
    P = n + 2

    # ---- the bank saturates, so key and value are exact per class ----------
    u, key, val, cls, x, s = bank(model)
    sat = float(torch.minimum(u, 1 - u)[cls != 1].abs().max())
    say(f"bank saturation : max distance from {{0,1}} off the transparent places = {sat:.3e}")

    flat = key[cls != 1]
    spread = float(flat.max() - flat.min())
    notch = float(flat.min() - key[cls == 1].max())
    vdev = max(float(val[cls == 0].abs().max()), float((val[cls == 2] - 1).abs().max()))
    span = float(val.max() - val.min())
    say(f"key             : flat level {float(flat.min()):.6f} (spread {spread:.3e}), "
        f"notch depth >= {notch:.4f}")
    say(f"value           : off {{0,1}} by at most {vdev:.3e}, full range {span:.6f}")
    if notch <= 0.0:
        say("certificate     : the key has no notch -- NOT PROVEN")
        return dict(ok=False, bad=200, margin=0.0, gap=1.0, slack=math.inf,
                    notch=notch, leak=math.inf, ratio=0.0)

    # ---- attention leakage --------------------------------------------------
    # Against the nearest non-transparent place, every other non-transparent
    # place is at least one step further away, so their combined relative
    # weight is at most the geometric sum below; the transparent places are
    # suppressed by the notch even at the largest separation in the window.
    lm = float(lam)
    lk_nt = math.exp(spread) * math.exp(lm) / (1.0 - math.exp(lm))
    lk_tr = P * math.exp(-notch + abs(lm) * (P - 1))
    leak = lk_nt + lk_tr
    slack = (vdev + leak * span) * (float(e1.norm()) + float(e2.norm()))
    say(f"leakage (P={P:2d})   : non-transparent {lk_nt:.3e} + transparent {lk_tr:.3e}"
        f"  ->  read-out shift <= {slack:.6f}")

    # ---- every (digit pair, carry in) --------------------------------------
    aa, bb_ = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    aa, bb_ = aa.reshape(-1).repeat(2), bb_.reshape(-1).repeat(2)
    cin = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
    tot = aa + bb_ + cin
    cout = (tot >= 10).long()
    want = tot % 10
    r = code[aa] + code[bb_] + cin[:, None] * e1 + cout[:, None] * e2 + rb
    got = ((r[:, None, :] - code) ** 2).sum(-1).argmin(-1)
    mg = signed_margin(r, code, want)
    gap = float((code[:, None, :] - code).norm(dim=-1).masked_fill(
        torch.eye(10, dtype=torch.bool), math.inf).min())
    bad = int((got != want).sum())
    say(f"place read-out  : {200 - bad}/200 correct   worst margin {float(mg.min()):.6f} "
          f"= {float(mg.min())/gap:.4f} prototype gaps (gap {gap:.6f})")
    if bad:
        i = int(torch.nonzero(got != want)[0])
        say(f"   first failure: a={int(aa[i])} b={int(bb_[i])} cin={int(cin[i])} "
              f"-> {int(got[i])}, want {int(want[i])}")

    ok = bad == 0 and float(mg.min()) > slack
    say(f"certificate     : worst margin {float(mg.min()):.6f} "
          f"{'>' if ok else '<='} leakage {slack:.6f}  ->  "
          f"{'exact on every ' + str(n) + '-digit input' if ok else 'NOT PROVEN'}"
          f"   (ratio {float(mg.min())/max(slack,1e-30):.1f}x)")
    return dict(ok=ok, bad=bad, margin=float(mg.min()), gap=gap, slack=slack,
                notch=notch, leak=leak, ratio=float(mg.min()) / max(slack, 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    torch.set_printoptions(precision=6, sci_mode=False)
    model, meta = verify.load(a.sub).build_model()
    print(f"file            : {a.sub}   ({sum(p.numel() for p in model.parameters())} parameters)")
    return 0 if audit(model, a.n)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
