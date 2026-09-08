"""Sweep every trained member through the exact reduction and keep the best certificate.

For each member of each parent checkpoint:
    normalize()   exact gauge rewrites  (q, ls, translation, scale, unit flip/merge)
    substitute()  architectural constants (kw, lam, vw, vb, rb) -- certified, not exact
    narrow_bank() ramp width -> +-8            (exact when the bank stays saturated)
    certify()     whole-domain proof; rank by the safety factor it reports

Members whose bank is not saturated, or whose gauge rewrite does not reproduce the
parent's answers bit-for-bit, are rejected rather than patched.
"""

import argparse, json
import torch

import reduce as R, stage as S, certify as C


def evaluate_member(p0, verbose=False):
    p, _ = R.normalize(p0, verbose=False)
    agree = R.check_exact(p0, p, total=120000)
    if agree < 1.0:
        return None, f"gauge rewrite changed answers (agreement {agree})"
    q = S.substitute(p)
    qn, exact_narrow, rep = R.narrow_bank(q, B=8.0)
    ok, info = C.certify(qn, verbose=verbose)
    info["exact_narrow"] = exact_narrow
    info["slack_before_narrow"] = rep["slack_old"]
    info["agree"] = agree
    return (qn, ok, info), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="best.pt")
    args = ap.parse_args()

    rows = []
    for ck in args.ckpts:
        st = torch.load(ck, map_location="cuda")
        n = st["p"]["code"].shape[0]
        for i in range(n):
            if float(st["acc"][i]) < 1.0:
                continue
            p0 = R.to64(R.member(st["p"], i))
            try:
                res, err = evaluate_member(p0)
            except Exception as ex:
                print(f"{ck}[{i}]: not exactly reducible ({ex})", flush=True)
                continue
            if res is None:
                print(f"{ck}[{i}]: {err}", flush=True)
                continue
            qn, ok, info = res
            print(f"{ck}[{i}]: certified={ok} slack={info['slack']:+.4f} "
                  f"delta={info['delta']:.2e} margin={info['margin']:.5f} "
                  f"safety=x{info['safety']:.4g} exact_narrow={info['exact_narrow']}",
                  flush=True)
            if ok and info["exact_narrow"]:
                # Every certified member is correct in exact arithmetic, so rank them by
                # the tightest headroom any of the three checks leaves: that is the
                # quantity float32 rounding could eat into. `slack` is the bank's
                # distance outside the clamp ramp; `margin - bound` is what the read-out
                # has left after the worst-case attention leakage.
                headroom = min(info["slack"], info["margin"] - info["bound"])
                info["headroom"] = headroom
                rows.append((headroom, ck, i, qn, info))

    if not rows:
        raise SystemExit("no member produced a certificate")
    rows.sort(key=lambda r: -r[0])
    headroom, ck, i, qn, info = rows[0]
    print(f"\nbest: {ck}[{i}]  headroom {headroom:.5f}  (slack {info['slack']:.4f}, "
          f"margin {info['margin']:.5f}, safety x{info['safety']:.4g})")
    acc = R.accuracy(qn, total=1 << 20)
    print(f"held-out accuracy (1M unseen pairs): {acc}")
    torch.save({"p": {k: v.to(torch.float32) for k, v in qn.items()},
                "U": 2, "src": [ck, i], "acc": torch.ones(1),
                "cert": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in info.items()}}, args.out)
    print("saved", args.out)
    print(f"\ncandidates certified: {len(rows)}")
    for h, c, j, _, nfo in rows[:10]:
        print(f"  headroom {h:.5f}  slack {nfo['slack']:+.4f}  margin {nfo['margin']:.5f}"
              f"  safety x{nfo['safety']:.4g}   {c}[{j}]")


if __name__ == "__main__":
    main()
