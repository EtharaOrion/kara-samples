"""Stage 2: project a trained C=2 parent onto a scalar (C=1) residual stream
and continue training there.

The parent's learned code table comes out numerically rank-1 (top singular
ratio ~1e-3), so almost all of the model already lives on one line in the
residual plane.  We project every residual-space quantity onto that line and
keep training; nothing is invented, the narrower model is initialised from the
wider trained one.

Projection (w = unit vector along the code's dominant direction):
    code1[d] = code[d].w
    x1       = x.w                    (exact for the on-line part of x)
    Wb1[u]   = Wb[u].w                (off-line part folded into the bias)
    bb1[u]   = bb[u] + 2 * Wb[u].mu_perp
    uA1      = uA.w ,  uB1 = uB.w
    kw, vw, vb, q, lam, ls unchanged
"""

import argparse
import torch

import arch
import lab


def project(p):
    """p: dict with leading axis E and C=2.  Returns C=1 params."""
    code = p["code"]                                   # [E,10,2]
    cen = code - code.mean(1, keepdim=True)
    _, _, V = torch.linalg.svd(cen, full_matrices=False)
    w = V[:, 0, :]                                     # [E,2] top right sing. vec

    # orient so the code ramp increases with the digit and the carry-in
    # write-back uA is positive (this fixes the residual-scale sign gauge)
    proj = torch.einsum("edc,ec->ed", code, w)
    slope = proj[:, 9] - proj[:, 0]
    w = w * torch.sign(slope)[:, None]
    proj = torch.einsum("edc,ec->ed", code, w)

    mu = code.mean(1)                                  # [E,2]
    mu_par = (mu * w).sum(-1, keepdim=True) * w
    mu_perp = mu - mu_par

    out = {
        "code": proj[:, :, None].clone(),
        "Wb": torch.einsum("euc,ec->eu", p["Wb"], w)[:, :, None].clone(),
        "bb": (p["bb"] + 2 * torch.einsum("euc,ec->eu", p["Wb"], mu_perp)).clone(),
        "kw": p["kw"].clone(),
        "vw": p["vw"].clone(),
        "vb": p["vb"].clone(),
        "uA": (p["uA"] * w).sum(-1, keepdim=True).clone(),
        "uB": (p["uB"] * w).sum(-1, keepdim=True).clone(),
        "q": p["q"].clone(),
        "lam": p["lam"].clone(),
        "ls": p["ls"].clone(),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/parent.pt")
    ap.add_argument("--out", default="ckpt/c1.pt")
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    dev = "cuda"
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    d = torch.load(a.ckpt, map_location=dev, weights_only=False)
    par = {k: v.to(dev) for k, v in d["params"].items()}

    p1 = project(par)
    acc0, _ = lab.evaluate(p1, 8, dev, g, n_batch=4, B=512)
    print("after projection, held-out exact match:",
          [round(float(x), 4) for x in acc0.tolist()], flush=True)

    # replicate each projected member with small init noise and retrain
    reps = {k: v.repeat_interleave(a.copies, 0).clone() for k, v in p1.items()}
    for k, v in reps.items():
        v += a.sigma * torch.randn(v.shape, generator=g, device=dev) * v.abs().mean()
    print("retraining", reps["code"].shape[0], "C=1 members", flush=True)

    reps, best, _ = lab.train(reps, list(reps.keys()), a.steps, dev, g, lr=a.lr,
                              B=512, pct_start=0.05, tag="c1", eval_chunk=128)
    acc, worst = lab.evaluate(best, 8, dev, g, n_batch=16, B=1024, chunk=128)
    order = torch.argsort(acc + 0.001 * worst.clamp(-5, 5), descending=True)
    print("top C=1 members:",
          [(round(float(acc[i]), 5), round(float(worst[i]), 3)) for i in order[:10]],
          flush=True)
    print("members >= 0.9999:", int((acc >= 0.9999).sum()), flush=True)
    keep = order[:64]
    torch.save({"params": {k: v[keep].cpu() for k, v in best.items()},
                "acc": acc[keep].cpu(), "worst": worst[keep].cpu(),
                "C": 1, "U": 2}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
