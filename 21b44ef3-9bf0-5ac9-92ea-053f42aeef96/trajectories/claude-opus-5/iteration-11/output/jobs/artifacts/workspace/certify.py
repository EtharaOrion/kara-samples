"""A correctness certificate for a trained member, over *all* inputs.

Sampling can only ever miss something.  The model's behaviour, though, factorises:
if for every digit pair the bank is exactly saturated, then each place contributes
one of only three things to the attention (key = kw and value = 0, key = kw and
value = 1, or key ~ 0), and the whole computation is determined by the per-place
carry classes plus the residual value x of the place being read out.  So it is
enough to check

  (A) saturation: for every one of the 100 digit pairs with a+b != 9, |alpha*(x-theta)|
      is at least 1, so upos and uneg are exactly 0/1; and for a+b == 9 it is small,
      so the notch is deep;
  (B) attention leakage: bound the total softmax weight that lands anywhere other
      than the nearest eligible place, given (A);
  (C) read-out: for all 100 pairs x 2 possible carries-in, the ideal residual
      y = x + e1*c_in + e2*c_out is nearer to the right prototype than to any other
      by more than the perturbation (B) can move it.

If all three hold, every input of every length decodes correctly.  Everything is
computed in float64; the float32 gap is measured separately in verify.py.
"""

import argparse
import math

import torch


def certify(code, theta, e2, e1, alpha, kw, lam, P=10, verbose=True):
    code = code.detach().double().cpu()
    theta, e2, e1 = float(theta), float(e2), float(e1)
    alpha, kw, lam = float(alpha), float(kw), float(lam)
    out = {}

    a = torch.arange(10)
    x = code[a][:, None] + code[a][None, :]                  # [10, 10]
    s = a[:, None] + a[None, :]
    t = alpha * (x - theta)
    transp = s == 9

    # (A) saturation
    sat_slack = (t.abs() - 1.0)[~transp].min().item()        # want > 0
    sign_ok = bool((((t > 0) == (s >= 10))[~transp]).all())
    n9 = t.abs()[transp].max().item()                        # want << 1
    out["sat_slack"] = sat_slack
    out["sign_ok"] = sign_ok
    out["notch_depth"] = 1.0 - n9

    # alpha is not free to be anything: it only has to separate the two scales
    #   d_tr  = how far the transparent pairs (a+b=9) spread from theta
    #   d_nt  = how close the nearest non-transparent pair comes to theta
    # saturation needs alpha*d_nt >= 1; an open notch needs alpha*d_tr <= 1 - |lam|*P/kw.
    # So the working band is [1/d_nt, (1-|lam|*P/kw)/d_tr], non-empty exactly when the
    # code is linear enough that d_nt > d_tr.  A perfect ramp gives d_tr = 0, band open above.
    d = (x - theta).abs()
    d_tr = d[transp].max().item()
    d_nt = d[~transp].min().item()
    hi = (1.0 - abs(lam) * P / kw)
    out["d_tr"], out["d_nt"] = d_tr, d_nt
    out["alpha_lo"] = 1.0 / d_nt if d_nt > 0 else float("inf")
    out["alpha_hi"] = hi / d_tr if d_tr > 0 else float("inf")

    # (B) leakage: weight on anything but the nearest eligible place, relative to it
    r = math.exp(lam)                                        # per extra step of distance
    leak_far = r / (1 - r) * (1 - r ** P)                    # other eligible places
    # a transparent place can be nearer, so it may gain up to |lam|*P of distance bias
    # a closed notch (n9 >= 1) means transparent places are no longer hidden and
    # the mechanism is simply broken; report it rather than overflowing.
    leak_tr = P * math.exp(min(50.0, -kw * (1.0 - n9) + abs(lam) * P))
    leak = leak_far + leak_tr
    out["leak"] = leak
    out["perturb"] = (abs(e1) + abs(e2)) * leak / (1.0 + leak)

    # (C) read-out slack for every (pair, carry-in)
    worst = float("inf")
    arg = None
    for cin in (0, 1):
        tot = s + cin
        d = tot % 10
        cout = tot // 10
        y = x + e1 * cin + e2 * cout
        dist = (y[..., None] - code).abs()                   # [10, 10, 10]
        right = dist.gather(-1, d[..., None]).squeeze(-1)
        other = dist.clone()
        other.scatter_(-1, d[..., None], float("inf"))
        slack = (other.min(-1).values - right) / 2.0         # distance to the boundary
        m = slack.min().item()
        if m < worst:
            worst = m
            i = int(slack.argmin())
            arg = (i // 10, i % 10, cin)
    out["slack"] = worst
    out["worst_case"] = arg
    out["ratio"] = worst / out["perturb"] if out["perturb"] > 0 else float("inf")
    out["ok"] = bool(sat_slack > 0 and sign_ok and worst > out["perturb"])

    if verbose:
        print(f"  (A) saturation slack   {sat_slack:+.4f}  (>0 required)   "
              f"signs {'ok' if sign_ok else 'WRONG'}")
        print(f"      notch depth        {out['notch_depth']:.6f}  (1.0 is perfect)")
        print(f"      transparent spread {d_tr:.4f} vs nearest other {d_nt:.4f}"
              f"  -> alpha band [{out['alpha_lo']:.2f}, {out['alpha_hi']:.2f}]")
        print(f"  (B) attention leakage  {leak:.3e}  -> residual perturbation "
              f"{out['perturb']:.3e}")
        print(f"  (C) read-out slack     {worst:.4f} at (a,b,cin)={arg}")
        print(f"  => {'CERTIFIED' if out['ok'] else 'NOT certified'} for P={P}, "
              f"safety ratio {out['ratio']:.1f}x")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--submission")
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    if args.submission:
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission", args.submission)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        model = m.build_model()[0]
        certify(model.code().detach(), model.theta, model.e2, model.e1,
                model.alpha, model.kw, model.lam, P=args.P)
        return

    ck = torch.load(args.ckpt, map_location="cpu")
    p = ck["params"]
    cfg = ck["cfg"]
    E = p["code"].shape[0]
    rows = []
    for m in range(E):
        r = certify(p["code"][m], p["theta"][m], p["e2"][m], p["e1"][m],
                    p["alpha"][m], p["kw"][m], p["lam"][m], P=args.P, verbose=False)
        r["member"] = m
        r["acc"] = float(ck["acc"][m])
        rows.append(r)
    rows.sort(key=lambda r: (-r["ok"], -r["ratio"]))
    print(f"{'mem':>4} {'acc':>8} {'satslack':>9} {'notch':>8} {'slack':>8} "
          f"{'perturb':>10} {'ratio':>9}  ok")
    for r in rows[: args.top]:
        print(f"{r['member']:>4} {r['acc']:>8.5f} {r['sat_slack']:>9.4f} "
              f"{r['notch_depth']:>8.5f} {r['slack']:>8.4f} {r['perturb']:>10.3e} "
              f"{r['ratio']:>9.1f}  {r['ok']}")
    print("\nbest member detail:")
    b = rows[0]["member"]
    certify(p["code"][b], p["theta"][b], p["e2"][b], p["e1"][b], p["alpha"][b],
            p["kw"][b], p["lam"][b], P=args.P)


if __name__ == "__main__":
    main()
