"""Rewrite a trained parent into the 11-parameter coordinates.

Two of the moves are exact symmetries of the forward map (nothing about the
function changes, only the coordinates it is written in):

  translation  code += t,  rb -= t,  theta += t
      x = code[a]+code[b]+rb shifts by 2t+(-t) = t, the read-out prototypes shift
      by t, and the bank threshold shifts by t, so every difference is preserved.
      Choosing t = -code[0] puts the origin of the residual stream at digit 0.

  scale        code *= g, rb *= g, theta *= g, e1 *= g, e2 *= g,
               alpha /= g, ls /= g**2                       (g > 0)
      every quantity is measured in units of the residual stream, so this is a
      change of unit.  Choosing g = 1/e1 makes the carry-in weight the unit.

One is an invariance of the decision rule rather than of the logits:

  ls -> 1      argmax_d -ls*(y-code_d)^2 = argmin_d (y-code_d)^2 for any ls > 0.

The rest (alpha, kw, lam, and rb -> 0 exactly) are snapped to round constants and
the model is then fine-tuned at 11 parameters, so nothing is being fitted here.
Accuracy is printed after every move so the cost of each is visible.
"""

import argparse

import torch

import adder
from data import Sampler
from train import build_eval_sets, evaluate


def acc_of(p, cfg, sets):
    return evaluate(p, cfg, sets).min(0).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--kw", type=float, default=2000.0)
    ap.add_argument("--lam", type=float, default=-8.0)
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = adder.default_cfg(**ck["cfg"])
    p = {k: v.to(device).clone() for k, v in ck["params"].items()}
    sam = Sampler(device)
    torch.manual_seed(7)
    sets = build_eval_sets(sam, [8, 5, 11], 8192, device)

    a0 = acc_of(p, cfg, sets)
    print(f"parent ({adder.n_params(cfg)}p): best {a0.max():.5f}  "
          f">=0.999: {(a0 >= 0.999).sum().item()}/{len(a0)}")

    # only "up-facing" members can be gauged to e1 = +1 with alpha > 0: the bank's
    # value output is upos specifically, so mirroring x is not a symmetry.
    up = (p["e1"] > 0) & (a0 >= 0.99)
    print(f"up-facing and >=0.99: {int(up.sum())}")
    if int(up.sum()) == 0:
        raise SystemExit("no usable member")
    idx = up.nonzero().squeeze(1)
    idx = idx[a0[idx].argsort(descending=True)][: args.keep]
    p = {k: v[idx].clone() for k, v in p.items()}

    # --- exact: translation --------------------------------------------------
    t = -p["code"][:, 0].clone()
    p["code"] += t[:, None]
    p["rb"] -= t
    p["theta"] += t
    p["theta_neg"] += t
    a1 = acc_of(p, cfg, sets)
    print(f"after translation  best {a1.max():.5f}  (exact, delta "
          f"{(a1 - a0[idx]).abs().max():.2e})")

    # --- exact: scale --------------------------------------------------------
    g = 1.0 / p["e1"].clone()
    for k in ("code", "rb", "theta", "theta_neg", "e1", "e2"):
        p[k] = p[k] * (g[:, None] if p[k].dim() > 1 else g)
    p["alpha"] = p["alpha"] / g
    p["ls"] = p["ls"] / (g * g)
    a2 = acc_of(p, cfg, sets)
    print(f"after scale        best {a2.max():.5f}  (exact, delta "
          f"{(a2 - a1).abs().max():.2e})   e1 -> {p['e1'].mean():.6f}")

    # --- decision-rule invariance: ls ---------------------------------------
    p["ls"] = torch.ones_like(p["ls"])
    a3 = acc_of(p, cfg, sets)
    print(f"after ls=1         best {a3.max():.5f}  (argmax-invariant, delta "
          f"{(a3 - a2).abs().max():.2e})")

    # --- snapped constants (fine-tuning follows) -----------------------------
    print(f"rb before snap: mean {p['rb'].abs().mean():.4g} "
          f"max {p['rb'].abs().max():.4g}")
    print(f"alpha {p['alpha'].min():.3g}..{p['alpha'].max():.3g}   "
          f"kw {p['kw'].min():.4g}..{p['kw'].max():.4g}   "
          f"lam {p['lam'].min():.3g}..{p['lam'].max():.3g}")
    for name, val in (("rb", 0.0), ("alpha", args.alpha), ("kw", args.kw),
                      ("lam", args.lam)):
        p[name] = torch.full_like(p[name], val)
        aa = acc_of(p, cfg, sets)
        print(f"after {name}={val:<8g} best {aa.max():.5f}  "
              f">=0.999: {(aa >= 0.999).sum().item()}")

    new = adder.default_cfg(code_fix=1, free_rb=False, free_alpha=False,
                            free_e1=False, free_ls=False, free_kw=False,
                            free_lam=False, free_theta_neg=False,
                            alpha=args.alpha, kw=args.kw, lam=args.lam)
    p["code"][:, 0] = 0.0
    p["theta_neg"] = p["theta"].clone()
    final = acc_of(p, new, sets)
    order = final.argsort(descending=True)
    p = {k: v[order].clone() for k, v in p.items()}
    torch.save({"params": p, "cfg": new, "acc": final[order].cpu(),
                "n_params": adder.n_params(new)}, args.out)
    print(f"saved {args.out}  n_params {adder.n_params(new)}  "
          f"best {final.max():.5f}  >=0.999: {(final >= 0.999).sum().item()}")


if __name__ == "__main__":
    main()
