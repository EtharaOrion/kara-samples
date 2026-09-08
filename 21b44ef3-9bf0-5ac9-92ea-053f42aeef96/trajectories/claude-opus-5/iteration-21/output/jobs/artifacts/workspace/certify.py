"""A whole-domain certificate for /workspace/submission.py.

verify.py samples.  This does not: it proves the shipped model is exactly right
on *every* pair of 8-digit operands (8.1e15 of them) by factoring the model the
way it is built.

Three facts, each checked exhaustively in float64:

  1. Saturation.  Over all 100 digit pairs, both clamp units land exactly on 0
     or 1, with margin.  So the key of a place is one of two values and its
     value is one of two values, decided purely by which carry class the place
     is in.
  2. Class map.  Those gate patterns partition the 100 pairs into exactly the
     three carry classes: g0 fires iff a + b >= 9, g1 fires iff a + b >= 10.
  3. Attention.  With keys and values reduced to two levels each, the whole
     attention computation depends on the input only through the sequence of
     carry classes.  Every such sequence up to width 8 is enumerated, and the
     worst deviation of the two heads from the true carry bits is recorded.

Then the readout is checked against the worst case: for all 100 pairs and both
carry-in values, the distance from the correct digit's decision boundary must
exceed everything the attention leak from (3) can move the residual by.
"""

import argparse
import itertools

import torch

import verify

ABSORB, TRANSPARENT, GENERATE = 0, 1, 2


def gate_table(model):
    """Gate pattern, key and value for each of the 100 digit pairs, in float64."""
    code = model.code().double()
    a = torch.arange(10).repeat_interleave(10)
    b = torch.arange(10).repeat(10)
    x = code[a] + code[b]
    t = model.gate_slope.double() * (x[:, None] - model.knee.double())
    gate = t.clamp(0.0, 1.0)
    slack = torch.maximum(-t, t - 1.0).amin()          # >= 0 means fully saturated
    return x, gate, slack, a, b


BLOCKED = -1e30    # the shipped mask is finfo(float32).min; either way the
                   # masked softmax weight is exactly zero in float64


def head_reads(classes, recency, key_scale):
    """Exact output of both heads, for a batch of carry-class sequences.

    classes: (N, P) long, position 0 and P-1 are the pads.  Returns (cin, cout),
    the strictly-causal and inclusively-causal reads of the value stream.
    """
    P = classes.shape[1]
    key = torch.where(classes == TRANSPARENT, -key_scale, key_scale.new_zeros(()))
    value = (classes == GENERATE).double()
    pos = torch.arange(P)
    gap = (pos[:, None] - pos[None, :]).double()
    strict = torch.where(gap <= 0, torch.full_like(gap, BLOCKED), torch.zeros_like(gap))
    strict[0, 0] = 0.0
    future = torch.where(gap < 0, torch.full_like(gap, BLOCKED), torch.zeros_like(gap))
    scores = key[:, None, :] + recency * gap                  # (N, P, P)
    w_in = torch.softmax(scores + strict, dim=-1)
    w_out = torch.softmax(scores + future, dim=-1)
    return (torch.einsum("nij,nj->ni", w_in, value),
            torch.einsum("nij,nj->ni", w_out, value))


def true_carries(patterns):
    """Carry into and out of each place, for a batch of class patterns (N, n)."""
    N, n = patterns.shape
    c = torch.zeros(N, dtype=torch.long)
    cin, cout = [], []
    for k in range(n):
        cin.append(c)
        col = patterns[:, k]
        c = torch.where(col == GENERATE, torch.ones_like(c),
                        torch.where(col == TRANSPARENT, c, torch.zeros_like(c)))
        cout.append(c)
    return torch.stack(cin, 1).double(), torch.stack(cout, 1).double()


def attention_error(model, widths, chunk=4096):
    """Worst deviation of either head from the true carry bit, over all patterns."""
    recency = model.recency.double()
    key_scale = model.key_scale.double()
    worst_in = worst_out = 0.0
    n_pat = 0
    for n in widths:
        pats = torch.tensor(list(itertools.product((ABSORB, TRANSPARENT, GENERATE),
                                                   repeat=n)))
        n_pat += pats.shape[0]
        for i in range(0, pats.shape[0], chunk):
            pat = pats[i:i + chunk]
            pad = torch.full((pat.shape[0], 1), ABSORB, dtype=torch.long)
            classes = torch.cat([pad, pat, pad], dim=1)
            cin, cout = head_reads(classes, recency, key_scale)
            t_in, t_out = true_carries(pat)
            zero = torch.zeros(pat.shape[0], 1, dtype=torch.float64)
            # positions 1..n are the places; position n+1 is the top answer digit,
            # which reads the carry leaving the last place and nothing else
            want_in = torch.cat([zero, t_in, t_out[:, -1:]], dim=1)
            want_out = torch.cat([zero, t_out, zero], dim=1)
            worst_in = max(worst_in, (cin - want_in).abs().max().item())
            worst_out = max(worst_out, (cout - want_out).abs().max().item())
    return worst_in, worst_out, n_pat


def readout_margin(model, x, a, b):
    """Distance from the correct digit's decision boundary, worst over all cases.

    Every (pair, carry-in) the model can ever meet: 100 pairs x 2 carry values.
    """
    code = model.code().double()
    cw = model.carry_w.double()
    fold = model.fold.double()
    worst = float("inf")
    wrong = 0
    for cin in (0, 1):
        total = a + b + cin
        cout = (total >= 10).double()
        z = x + cw * cin + fold * cout
        target = torch.remainder(total, 10)
        d = (z[:, None] - code[None, :]).abs()
        pred = d.argmin(dim=1)
        wrong += int((pred != target).sum())
        # boundary between the target digit and every other digit
        mid = (code[target][:, None] + code[None, :]) / 2.0
        gapm = (z[:, None] - mid).abs()
        gapm[torch.arange(100), target] = float("inf")
        worst = min(worst, gapm.min().item())
    return worst, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--max_width", type=int, default=8)
    args = ap.parse_args()

    mod = verify.load(args.path)
    model, _ = mod.build_model()
    print("certificate for", args.path)

    x, gate, slack, a, b = gate_table(model)
    print(f"1. gate saturation over all 100 pairs: margin {slack.item():+.4f} "
          f"(>= 0 means every unit is exactly 0 or 1)")
    assert slack.item() >= 0, "gate is not saturated; the certificate does not apply"

    total = a + b
    ok0 = torch.equal(gate[:, 0] == 1, total >= 9)
    ok1 = torch.equal(gate[:, 1] == 1, total >= 10)
    cls = gate[:, 0].long() + gate[:, 1].long()
    n_cls = [int((cls == k).sum()) for k in (0, 1, 2)]
    print(f"2. class map: g0 fires iff a+b>=9 {ok0};  g1 fires iff a+b>=10 {ok1}")
    print(f"   absorb/transparent/generate places: {n_cls} of 100 (want [45, 10, 45])")
    assert ok0 and ok1 and n_cls == [45, 10, 45], "gate does not encode the carry classes"

    widths = list(range(1, args.max_width + 1))
    e_in, e_out, n_pat = attention_error(model, widths)
    print(f"3. attention over all {n_pat} carry patterns of width 1..{args.max_width}:")
    print(f"   worst carry-in error  {e_in:.3e}")
    print(f"   worst carry-out error {e_out:.3e}")

    drift = abs(model.carry_w.item()) * e_in + abs(model.fold.item()) * e_out
    margin, wrong = readout_margin(model, x, a, b)
    print(f"4. readout over all 100 pairs x both carry-in values: "
          f"{200 - wrong}/200 correct, boundary margin {margin:.5f}")
    print(f"   worst the attention leak can move the residual: {drift:.3e}")

    ok = wrong == 0 and margin > drift
    print(f"\nCERTIFIED: {ok}   (margin exceeds drift by {margin / max(drift, 1e-300):.1f}x)")
    if ok:
        print("every pair of 8-digit operands is answered exactly.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
