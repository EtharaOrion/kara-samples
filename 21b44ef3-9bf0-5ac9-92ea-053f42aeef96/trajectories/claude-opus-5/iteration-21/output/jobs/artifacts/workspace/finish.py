"""Stage 3: spend the residual scale, fine-tune the 12 shipped values, build.

Training so far left code[1] free.  That extra degree of freedom is not a gauge:
the gate slope is fixed at 8, so rescaling the code by s changes the gate's
sharpness in code units by 1/s.  When s <= 1 the rescale only makes an already
saturated gate sharper, which leaves every prediction untouched -- that is why
selection upstream insists on a saturated, ascending member.  The fine-tune
afterwards is what repairs any member for which that was not quite true.
"""

import argparse

import torch

import build
import lab
import refine
import train as T


def rescale(p):
    """Free-unit members (code[1] free) -> the 12-value shipped form."""
    cf = p["code_free"]
    assert cf.shape[1] == 9, "expected a free-unit member"
    s = cf[:, :1]                                   # code[1]
    return {
        "code_free": cf[:, 1:] / s,                 # digits 2..9
        "carry_w": p["carry_w"] / s[:, 0],
        "knee": p["knee"] / s,
        "fold": p["fold"] / s[:, 0],
    }


@torch.no_grad()
def select(p, device, widths=(8, 5, 12, 3), B=4096, reps=2):
    acc = T.evaluate(p, device, widths=widths, split="eval", B=B, reps=reps)
    slack = T.gate_slack(p)
    return acc, slack, acc + 0.002 * slack.clamp(-2.0, 0.0)


def main(a):
    dev = "cuda"
    torch.manual_seed(a.seed)
    ck = torch.load(a.parents)
    p = {k: v.to(dev) for k, v in ck["params"].items()}
    up = lab.code_of(p)[:, 1] > 0
    p = {k: v[up] for k, v in p.items()}
    print(f"finish: {up.sum().item()} of {up.numel()} parents ascend")
    if up.sum() == 0:
        raise SystemExit("no ascending parent to ship")

    tight = {k: v.clone() for k, v in rescale(p).items()}
    acc, slack, _ = select(tight, dev, reps=1, B=2048)
    print(f"  straight after rescale: best acc {acc.max().item():.5f}   "
          f"members exact {lab.is_exact(acc).sum().item()}")

    R = a.R
    q = {k: v.repeat_interleave(R, dim=0).clone() for k, v in tight.items()}
    E = q["code_free"].shape[0]
    with torch.no_grad():
        for k, v in q.items():
            n = torch.randn_like(v) * a.jitter
            n[::R] = 0
            v.add_(n)
    q = {k: v.requires_grad_(True) for k, v in q.items()}
    print(f"  fine-tuning {E} members in the 12-value form", flush=True)
    q = refine.train_staged(q, dev, a.steps, a.lr, refine.stages(a.stages),
                            a.B, a.temp, snap_from=0.3, pct_start=0.05)

    acc, slack, score = select(q, dev)
    order = score.argsort(descending=True)
    print(f"  members exact on every eval width: {lab.is_exact(acc).sum().item()} / {E}   "
          f"gate-saturated too: {(lab.is_exact(acc) & (slack >= 0)).sum().item()}")
    best = order[0].item()
    print(f"  best: acc {acc[best].item():.6f}  slack {slack[best].item():.4f}")
    torch.save({"params": {k: v.detach()[order[:a.keep]].cpu() for k, v in q.items()},
                "acc": acc[order[:a.keep]].cpu()}, a.out)

    vals = {k: v.detach()[best].cpu() for k, v in q.items()}
    build.emit({"code_free": vals["code_free"].tolist(),
                "carry_w": vals["carry_w"].item(),
                "knee": vals["knee"].tolist(),
                "fold": vals["fold"].item()},
               "/workspace/submission.py", "/workspace/model_src.py")
    print("  wrote /workspace/submission.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", default="/workspace/parents2.pt")
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--temp", type=float, default=8.0)
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--stages", default="3:0.2,5:0.2,8:0.35,12:0.25")
    ap.add_argument("--keep", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/final.pt")
    main(ap.parse_args())
