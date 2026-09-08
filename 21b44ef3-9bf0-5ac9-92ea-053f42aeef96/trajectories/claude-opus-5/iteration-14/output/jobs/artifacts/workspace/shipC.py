"""Assemble the shipped 12-parameter file from a stage-C checkpoint.

Ranks the survivors by held-out exact match, keeps the ones the whole-domain
certificate passes, and among those picks the member with the widest margins -- the
distance from the clamp knees to the nearest place sum, and the read-out margin.
Both are pure robustness measures, so this is choosing the safest of several models
that are all already exact.
"""
import argparse
import torch

import arch, build, certify, data, stageC, train2 as T


def assemble_all(free, device):
    E = free["code"].shape[0]
    p = stageC.assemble({k: v.to(device) for k, v in free.items()}, E, device, 1.0)
    return {k: v.detach() for k, v in p.items()}


def knee_slack(p, cfg, i):
    """Smallest gap between a clamp knee and any residual value x = code[a]+code[b],
    in code steps.  Large means the bank's decisions are far from any boundary."""
    with torch.no_grad():
        q = {k: v[i:i + 1] for k, v in p.items()}
        code = arch.full_code(q, cfg)[0, :, 0].double()
        A = torch.arange(10, device=code.device)
        x = (code[A][:, None] + code[A][None, :]).flatten()
        sig = float((code * torch.arange(10., device=code.device, dtype=torch.float64)).sum()
                    / 285.0)
        knee = (-q["bb"][0].double() / q["Bw"][0, 0].double())
        return float((x[:, None] - knee[None, :]).abs().min() / abs(sig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/C7.pt")
    ap.add_argument("--out", default="submission.py")
    ap.add_argument("--N", type=int, default=32768)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"

    st = torch.load(a.ckpt, map_location=device)
    p = assemble_all(st["free"], device)
    cfg = st["cfg"]
    E = p["code"].shape[0]
    g = torch.Generator(device=device).manual_seed(20250906)
    tot = torch.zeros(E, device=device)
    widths = (2, 3, 5, 8, 11, 15)      # the held-out hash needs a*10^n to fit in int64
    for n in widths:
        acc, _ = T.evaluate(p, cfg, n, device, g, N=a.N, chunk=1024)
        tot += acc
        print(f"  n={n:2d}: #exact {int((acc>=1.0).sum()):3d}/{E}   (held out)")
    tot /= len(widths)
    # wider than any training width, so the held-out split is moot -- plain random
    for n in (20, 24):
        good = torch.zeros(E, device=device)
        for _ in range(8):
            da, db = data.sample_raw(1024, n, "chain", device, g)
            da, db = data._force_full_width(da, db, g)
            ta, tb = data.to_tokens(da, db)
            pred = arch.forward(p, ta, tb, cfg).argmax(-1)
            good += (pred[:, :, 1:] == data.targets(da, db)[None, :, 1:]).all(-1).float().mean(1)
        print(f"  n={n:2d}: #exact {int((good / 8 >= 1.0).sum()):3d}/{E}   (carry-heavy)")
    cand = (tot >= 1.0).nonzero(as_tuple=True)[0]
    print(f"{len(cand)} members exact at every width tested")

    rows = []
    for i in cand.tolist():
        one = {k: v[i].cpu() for k, v in p.items()}
        rep = certify.certify(one, cfg, Pmax=26, verbose=False)
        if rep["ok"]:
            ks, rm = knee_slack(p, cfg, i), rep["readout_margin"]
            # both margins are measured in code steps and both are ideally 0.5, so the
            # honest summary is the weaker of the two: that is the perturbation the
            # model actually tolerates.  Ties break on the sum.
            rows.append((min(ks, rm), ks + rm, ks, rm, i, rep))
    rows.sort(reverse=True)
    print(f"{len(rows)} members pass the whole-domain certificate")
    for w, _, ks, rm, i, _ in rows[:8]:
        print(f"  member {i:5d}: worst margin {w:.4f} | knee slack {ks:.4f} code steps, "
              f"read-out margin {rm:.4f}")
    if not rows:
        raise SystemExit("no member passed the certificate")

    _, _, ks, rm, i, rep = rows[0]
    one = {k: v[i].cpu() for k, v in p.items()}
    certify.certify(one, cfg, Pmax=26, verbose=True)
    doc = ("Trained in this workspace in three stages (see notes.md): a carry-free "
           "curriculum fixes the digit code, the carry mechanism is then learned on "
           "the real distribution, and the 12 values below are refined in exactly the "
           "form shipped here.  certify.py proves this weight set is exact for every "
           "one of the 10^16 operand pairs, not just the sampled ones.")
    n, sz = build.emit(one, cfg, ["code", "bb", "w2"], a.out, doc)
    print(f"\nwrote {a.out}: {n} parameters, {sz} bytes  (member {i})")


if __name__ == "__main__":
    main()
