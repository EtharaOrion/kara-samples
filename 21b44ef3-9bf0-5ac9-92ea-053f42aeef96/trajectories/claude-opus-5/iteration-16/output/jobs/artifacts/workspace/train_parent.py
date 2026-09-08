"""Stage 1: cold-train the wide parent (C=2, U=2, everything learned).

38 free values per member.  E independent members share one data batch but have
independent parameters and independent Adam state, so this is a seed lottery.
"""

import argparse
import torch

import arch
import data
import lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/parent.pt")
    a = ap.parse_args()

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    g = torch.Generator(device=dev); g.manual_seed(a.seed)

    p = arch.init_params(a.E, C=2, U=2, device=dev, gen=g)
    print(f"members={a.E}  free values/member={sum(v[0].numel() for v in p.values())}",
          flush=True)

    p, best, hist = lab.train(p, list(p.keys()), a.steps, dev, g, lr=a.lr, B=a.B,
                              tag="parent", eval_chunk=128)

    acc, worst = lab.evaluate(best, 8, dev, g, n_batch=16, B=1024, chunk=128)
    order = torch.argsort(acc + 0.001 * worst.clamp(-5, 5), descending=True)
    print("top members (acc, worst-margin):",
          [(round(float(acc[i]), 5), round(float(worst[i]), 3)) for i in order[:10]],
          flush=True)
    print(f"members with acc >= 0.9999: {(acc >= 0.9999).sum().item()}", flush=True)

    keep = order[:32]
    torch.save({"params": {k: v[keep].cpu() for k, v in best.items()},
                "acc": acc[keep].cpu(), "worst": worst[keep].cpu(),
                "C": 2, "U": 2}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
