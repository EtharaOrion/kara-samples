"""Rank the alternates saved in a checkpoint by accuracy *and* by whether the
attention is genuinely input-dependent.

Two independent probes:

  frozen   accuracy when the attention map is replaced by its batch mean.  A
           model whose attention is a fixed pattern in disguise is unharmed by
           this; one that selects positions by content collapses.
  len16    exact accuracy on 16-place carry chains.  Attention driven only by a
           fixed distance decay cannot move a carry across a long run of
           transparent places at any decay rate, so it falls off with length.
"""
import argparse, json
import torch

import data
import probe
from model_src import DigitPairAdder, default_cfg, n_params


def build(cfg, state, P, dev):
    m = DigitPairAdder(dict(cfg, P=int(P)))
    with torch.no_grad():
        for k, p in m.named_parameters():
            p.copy_(state[k].reshape(p.shape).float())
    return m.to(dev).eval()


def score(cfg, state, dev, gen, n=40000):
    """(u8, c8, c16, nocontent_c8, content_sd_ratio) for one member.

    `nocontent` is accuracy after deleting the q.k term from the attention
    logit, leaving the model's own fixed distance pattern.  The task requires
    the attention to depend on its input, so this number must be far below the
    real one.
    """
    m8 = build(cfg, state, 10, dev)
    tu, gu = data.sample(n, dev, gen, mix=(1., 0., 0.), split="eval")
    tc, gc = data.sample(n, dev, gen, mix=(0., .3, .7), split="eval")
    u8 = probe.batched_acc(m8, tu, gu)
    c8 = probe.batched_acc(m8, tc, gc)
    noc = probe.ablate_acc(m8, tc, gc, "nocontent")

    m16 = build(cfg, state, 18, dev)
    tl, gl = data.sample(n, dev, gen, mix=(0., .3, .7), split="eval", nplace=16)
    c16 = probe.batched_acc(m16, tl, gl)

    sc, sp = probe.content_share(m8, tu[:20000])
    return u8, c8, c16, noc, sc / (sp + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--save", default=None, help="write the winner here")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(4242)

    rows = []
    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(default_cfg()); cfg.update(ck["cfg"])
        npar = n_params(DigitPairAdder(cfg))
        alts = ck.get("alts") or [ck["state"]]
        for j, st in enumerate(alts[:args.top]):
            u8, c8, c16, noc, ratio = score(cfg, st, dev, gen, args.n)
            rows.append((path, j, npar, u8, c8, c16, noc, ratio, cfg, st))
            print(f"{path} m{j}: params={npar} u8={u8:.4f} c8={c8:.4f} "
                  f"c16={c16:.4f} nocontent={noc:.4f} c/p={ratio:.2f}",
                  flush=True)

    print("\n==== ranked by min(u8, c16) ====")
    rows.sort(key=lambda r: -min(r[3], r[5]))
    for path, j, npar, u8, c8, c16, noc, ratio, _, _ in rows[:12]:
        flag = "attention load-bearing" if (c8 - noc > 0.05 and ratio > 0.5) \
            else "SUSPECT: fixed pattern"
        print(f"  {path} m{j:<2d} params={npar:4d} u8={u8:.4f} c8={c8:.4f} "
              f"c16={c16:.4f} nocontent={noc:.4f} c/p={ratio:5.2f}  {flag}")

    if args.save and rows:
        best = rows[0]
        torch.save({"cfg": best[8], "state": best[9], "acc": min(best[3], best[5]),
                    "params": best[2]}, args.save)
        print(f"\nSAVED {args.save} <- {best[0]} m{best[1]}")


if __name__ == "__main__":
    main()
