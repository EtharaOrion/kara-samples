"""Rank every member of every checkpoint by how well conditioned it is.

Accuracy at one tuned alpha is a weak criterion: a member can score 1.0 at the
value it was trained at and fall over one step either side.  What matters is the
*derived* admissible band

    [ 1/d_nt , (1 - |lam|*P/kw)/d_tr ]

(see certify.py), which is a property of the learned code and threshold alone.  A
wide band means the code is close to an exact ramp and the transparent pairs sit
tightly on theta -- i.e. the arithmetic is sharp -- and it means the shipped alpha
is genuinely a don't-care rather than a tuned value.

So: compute the band, put alpha on the roundest number inside it, certify there,
and *measure* exact-match there as well, because the certificate assumes a
mechanism and the measurement does not.
"""

import argparse
import glob

import torch

import adder
from certify import certify
from data import Sampler

ROUND = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]


def nicest(lo, hi):
    """The roundest alpha strictly inside the band, preferring the middle."""
    mid = (lo * hi) ** 0.5
    inside = [v for v in ROUND if lo * 1.15 < v < hi / 1.15]
    if not inside:
        return None
    return min(inside, key=lambda v: abs(v / mid - 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=sorted(glob.glob("runs/*.pt*")))
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--n", type=int, default=1 << 14)
    args = ap.parse_args()

    device = "cuda"
    sam = Sampler(device)
    torch.manual_seed(11)
    evals = []
    for spec in (None, 0.7, 0.9):
        a, b = sam.sample_digits(args.n, 8, spec)
        ab, tgt, mask = sam.pack(a, b, held_out=True)
        keep = mask.sum(-1) > 0
        evals.append((ab[keep], tgt[keep], mask[keep]))
    geo = adder.geometry(10, device)

    rows = []
    for path in args.ckpts:
        try:
            ck = torch.load(path, map_location=device)
        except Exception:
            continue
        if "params" not in ck or adder.n_params(ck["cfg"]) != 11:
            continue
        p = ck["params"]
        for m in range(p["code"].shape[0]):
            kw, lam = float(p["kw"][m]), float(p["lam"][m])
            r0 = certify(p["code"][m], p["theta"][m], p["e2"][m], p["e1"][m],
                         float(p["alpha"][m]), kw, lam, P=args.P, verbose=False)
            al = nicest(r0["alpha_lo"], r0["alpha_hi"])
            if al is None:
                continue
            r = certify(p["code"][m], p["theta"][m], p["e2"][m], p["e1"][m],
                        al, kw, lam, P=args.P, verbose=False)
            if not r["ok"]:
                continue
            q = {k: v[m:m + 1].clone() for k, v in p.items()}
            q["alpha"] = torch.full((1,), al, device=device)
            acc = []
            for ab, tgt, mask in evals:
                lg = adder.forward(q, ck["cfg"], ab, geo)
                ok = ((lg.argmax(-1) == tgt[None]) | (mask[None] == 0)).all(-1)
                acc.append(ok.float().mean().item())
            r.update(path=path, member=m, alpha=al, acc=min(acc),
                     width=r0["alpha_hi"] / r0["alpha_lo"],
                     lo=r0["alpha_lo"], hi=r0["alpha_hi"])
            rows.append(r)

    rows = [r for r in rows if r["acc"] >= 1.0]
    rows.sort(key=lambda r: -r["width"])
    print(f"{'ckpt':28s} {'mem':>4} {'alpha':>6} {'band':>15} {'x':>6} "
          f"{'notch':>7} {'satsl':>7} {'ratio':>8} {'acc':>8}")
    for r in rows[: args.top]:
        print(f"{r['path'][-28:]:28s} {r['member']:>4} {r['alpha']:>6g} "
              f"[{r['lo']:>5.2f},{r['hi']:>7.2f}] {r['width']:>6.1f} "
              f"{r['notch_depth']:>7.4f} {r['sat_slack']:>7.3f} "
              f"{r['ratio']:>8.1f} {r['acc']:>8.5f}")
    if rows:
        b = rows[0]
        print(f"\nbest: {b['path']} member {b['member']} at alpha={b['alpha']:g}")
        ck = torch.load(b["path"], map_location=device)
        p = ck["params"]
        m = b["member"]
        certify(p["code"][m], p["theta"][m], p["e2"][m], p["e1"][m],
                b["alpha"], float(p["kw"][m]), float(p["lam"][m]), P=args.P)


if __name__ == "__main__":
    main()
