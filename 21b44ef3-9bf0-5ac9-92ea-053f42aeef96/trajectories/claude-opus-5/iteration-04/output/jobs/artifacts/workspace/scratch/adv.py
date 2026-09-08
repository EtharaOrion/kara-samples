"""Search for an input the shipped model gets wrong, instead of sampling for one.

Random draws and the 3**8 carry structures both bottom out at the same margin,
which suggests a structural worst case rather than an unlucky one -- but neither
is a search.  This is: coordinate descent over the eight digit pairs, run from
many random restarts at once, driving the objective *down* toward a flipped
answer.  At each round every restart tries all 100 pairs at one place and keeps
whichever leaves the model closest to a wrong digit.

If a wrong answer exists within reach of this search, this finds it.  If the
best it can do after thousands of restarts is still a comfortable margin, that
is much stronger than any amount of sampling.

Usage:  python scratch/adv.py [restarts] [rounds]
"""
import sys

import torch

import data
import submission

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def margin(m, dig, step):
    """Signed distance to the nearest wrong answer, in prototype steps.

    Positive means correct; the minimum over the nine output places is what an
    adversary is trying to push below zero.
    """
    lg = m(data.tokens(dig))[:, 1:, :]
    tgt = data.targets(dig)
    true = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    rival = lg.masked_fill(
        torch.nn.functional.one_hot(tgt, 10).bool(), float("-inf")).max(-1).values
    return ((true - rival) / (2 * step)).min(-1).values


@torch.no_grad()
def main(R, rounds):
    m, _ = submission.build_model()
    m = m.to(DEV)
    step = float(m._code().sort()[0].diff().abs().max())

    g = torch.Generator(device=DEV).manual_seed(7)
    dig = torch.randint(0, 10, (R, 8, 2), device=DEV, generator=g)
    dig[:, 7, :] = torch.randint(1, 10, (R, 2), device=DEV, generator=g)

    pairs = torch.stack(torch.meshgrid(torch.arange(10, device=DEV),
                                       torch.arange(10, device=DEV),
                                       indexing="ij"), -1).reshape(100, 2)
    cur = margin(m, dig, step)
    print(f"start   min {float(cur.min()):.5f}  mean {float(cur.mean()):.5f}")

    for r in range(rounds):
        place = r % 8
        opts = pairs[pairs.min(-1).values > 0] if place == 7 else pairs
        n = opts.shape[0]
        # (R, n, 8, 2): every restart tries every pair at this one place
        cand = dig.unsqueeze(1).repeat(1, n, 1, 1)
        cand[:, :, place, :] = opts
        sc = margin(m, cand.reshape(R * n, 8, 2), step).reshape(R, n)
        best, k = sc.min(-1)
        dig[:, place, :] = opts[k]
        cur = best
        if r % 8 == 7 or r == rounds - 1:
            print(f"round {r+1:3d}  min {float(cur.min()):.5f}  "
                  f"mean {float(cur.mean()):.5f}  "
                  f"restarts below 0.1: {int((cur < 0.1).sum())}", flush=True)

    v, i = cur.min(0)
    d = dig[int(i)].cpu()
    a = sum(int(d[j, 0]) * 10 ** j for j in range(8))
    b = sum(int(d[j, 1]) * 10 ** j for j in range(8))
    got = submission.add(m.cpu(), a, b)
    print(f"\nhardest input found by search: {a} + {b}")
    print(f"  margin {float(v):.5f} prototype steps  "
          f"({'WRONG' if float(v) < 0 else 'correct'})")
    print(f"  model {got}  truth {a + b}  {'OK' if got == a + b else 'MISMATCH'}")
    print(f"  wrong answers among all {R} restarts: {int((cur < 0).sum())}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4096,
         int(sys.argv[2]) if len(sys.argv) > 2 else 48)
