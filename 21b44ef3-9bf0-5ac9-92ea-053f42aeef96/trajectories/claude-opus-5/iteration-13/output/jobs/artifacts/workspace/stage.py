"""Take a trained parent member -> exact gauge normalisation -> substitute the
architectural constants -> emit a warm-start checkpoint for constrained retraining.

The substituted tensors become buffers (architecture, not learned facts):
    q, ls          gauge fixings (query/key scale, read-out temperature)
    code[0], rb    origin of the scalar residual stream: the (0,0) pad token sits at 0
    e[0]           scale gauge: one carry-in is one unit of the residual stream
    kw, lam        attention sharpness and recency slope (attention hyper-parameters)
    vw, vb         the value head reads bank unit 1 (a fixed 1-hot value projection)

What stays learned: code[1..9] (the digit code, 9 numbers), bb[0..1] (the two bank
knees), e[1] (the fold applied when a carry leaves a place). 12 numbers.

Note the knee reparameterisation: freezing bw changes the ramp *width*, so bb is
rescaled to hold the knee position theta = -bb/bw fixed. If no reachable residual
value lands inside either ramp, that substitution is exactly function-preserving;
`certify.py` checks it over the entire input domain rather than on a sample.
"""

import argparse
import torch

import core, reduce as R

CONST = {
    "kw":  [400.0, 400.0],
    "lam": [-8.0, -8.0],
    "vw":  [0.0, 1.0],
    "vb":  0.0,
    "rb":  0.0,
    "q":   1.0,
    "ls":  1.0,
}
# bw stays learnable here: the clamp knees only receive gradient while the ramp can
# adapt, so freezing the ramp width before retraining strands them. It is removed
# afterwards by reduce.narrow_bank(), which is exactly function-preserving.
FROZEN = ["q", "ls", "vw", "vb", "rb", "kw", "lam"]        # + code[0], e[0] via masks


def substitute(p):
    """p must already be gauge-normalised. Returns the constant-substituted params."""
    p = {k: v.clone() for k, v in p.items()}
    dt, dev = p["code"].dtype, p["code"].device
    for k, val in CONST.items():
        t = torch.tensor(val, dtype=dt, device=dev)
        p[k] = t.expand_as(p[k]).clone() if t.dim() == p[k].dim() - 1 else t.reshape(p[k].shape).clone()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="stage.pt")
    ap.add_argument("--members", type=int, nargs="+", default=None)
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cuda")
    n = st["p"]["code"].shape[0]
    cand = args.members if args.members is not None else list(range(n))
    kept = []
    for i in cand:
        if float(st["acc"][i]) < 1.0:
            continue
        p0 = R.to64(R.member(st["p"], i))
        try:
            p, _ = R.normalize(p0, verbose=False)
        except Exception as ex:
            print(f"member {i}: not exactly reducible ({ex})")
            continue
        ag = R.check_exact(p0, p, total=120000)
        if ag < 1.0:
            print(f"member {i}: gauge rewrite changed answers (agreement {ag}) -- skipped")
            continue
        ps = substitute(p)
        acc_before = R.accuracy(p, total=1 << 17)
        acc_after = R.accuracy(ps, total=1 << 17)
        print(f"member {i}: exact-agreement {ag:.6f}  acc after gauges {acc_before:.6f}"
              f"  acc after substitution {acc_after:.6f}")
        kept.append(ps)

    if not kept:
        raise SystemExit("no reducible member found")
    out = {k: torch.cat([p[k] for p in kept], 0).to(torch.float32) for k in core.SPEC}
    torch.save({"p": out, "U": 2, "frozen": FROZEN,
                "acc": torch.zeros(len(kept))}, args.out)
    print(f"saved {args.out} with {len(kept)} warm starts")


if __name__ == "__main__":
    main()
