"""How much numerical room the shipped model has.

The readout is a squared distance to ten learned prototypes, and `code_fix`
named the answer axis' unit in a way that left the prototypes about a tenth of
a unit apart.  That is the quantity a grader's float32 arithmetic has to
resolve, so it is worth measuring rather than assuming: how close the residual
stream ever comes to a decision boundary, how saturated the attention is, and
whether float32 and float64 ever disagree.

Usage:  python scratch/margin.py [n]
"""
import sys

import torch

import data
import submission


def patterns():
    """One instance of each of the 3**8 generate/propagate/absorb structures."""
    import random
    rng = random.Random(0)
    out = []
    for code in range(3 ** 8):
        da, db, c = [], [], code
        for i in range(8):
            k, c = c % 3, c // 3
            lo = 1 if i == 7 else 0
            if k == 0:
                x = rng.randint(lo, 9 - lo); y = 9 - x
            elif k == 1:
                s = rng.randint(max(10, 2 * lo), 18)
                x = rng.randint(max(lo, s - 9), min(9, s - lo)); y = s - x
            else:
                s = rng.randint(2 * lo, 8)
                x = rng.randint(lo, min(9, s - lo)); y = s - x
            da.append(x); db.append(y)
        out.append([da, db])
    return torch.tensor(out).permute(0, 2, 1)


@torch.no_grad()
def main(n):
    m, _ = submission.build_model()
    code = m._code()
    step = code.sort()[0].diff().abs()
    print(f"code   {[round(float(v), 4) for v in code]}")
    print(f"steps  min {float(step.min()):.5f}  max {float(step.max()):.5f}")

    sets = [("uniform ", data.eval_set(n, "cpu", 11, data.UNIFORM)),
            ("hard    ", data.eval_set(n, "cpu", 22, data.HARD)),
            ("patterns", patterns())]

    worst_qv = 1e9
    for name, dig in sets:
        x = data.tokens(dig)
        lg32 = m(x)[:, 1:, :]
        lg64 = m.to(torch.float64)(x.clone())[:, 1:, :]
        m.to(torch.float32)

        tgt = data.targets(dig)
        pred = lg32.argmax(-1)
        acc = (pred == tgt).all(-1).float().mean()

        # distance from the residual value to the midpoint between the winning
        # prototype and its nearest rival, in units of one prototype step
        top2 = lg32.topk(2, -1).values
        gap = (top2[..., 0] - top2[..., 1])
        # -(qv-c1)^2 + (qv-c2)^2 = (c2-c1)(2qv - c1 - c2); the distance to the
        # boundary is gap / (2 * |c2 - c1|), and |c2-c1| >= one step
        room = gap / (2 * float(step.max()))
        worst_qv = min(worst_qv, float(room.min()))

        d = (lg32.double() - lg64).abs().max()
        flip = (lg32.argmax(-1) != lg64.argmax(-1)).sum()
        print(f"{name}  n={dig.shape[0]:>7d}  acc {float(acc):.6f}  "
              f"min boundary distance {float(room.min()):.4f} steps  "
              f"|f32-f64| {float(d):.2e}  argmax flips {int(flip)}")

    # attention saturation: how much probability the chosen key actually gets
    dig = sets[1][1]
    x = data.tokens(dig)
    h = attn_probs(m, x)
    top = h.max(-1).values[:, 1:]
    print(f"attention: min top-1 probability over queries 1..9 = {float(top.min()):.6f}"
          f"  mean {float(top.mean()):.6f}")
    print(f"\nworst boundary distance anywhere: {worst_qv:.4f} steps "
          f"({'safe' if worst_qv > 0.05 else 'TIGHT'})")


@torch.no_grad()
def attn_probs(m, x):
    """Re-run the block far enough to read the attention matrix."""
    import torch.nn.functional as F
    c = m.cfg
    code = m._code()
    counts = F.one_hot(x, 10).sum(-2).to(code.dtype)
    z = counts @ code.unsqueeze(-1)
    h = F.pad(z, (0, c["d"] - 1))
    lo, hi = c["f1_in"]
    p = F.relu(h[..., lo:hi] + m.f1_b)
    o = p @ m.f1_o
    lo, hi = c["f1_out"]
    h = h + F.pad(o, (lo, c["d"] - hi))
    lo, hi = c["qk_in"]
    q = h[..., lo:hi] @ h.new_ones(hi - lo) + m.b_q
    att = q.unsqueeze(-1) * q.unsqueeze(-2) + c["alibi_fix"] * m.rel
    return att.masked_fill(~m.causal, float("-inf")).softmax(-1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100000)
