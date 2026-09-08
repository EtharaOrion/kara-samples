"""Whole-domain certificate for a shipped submission file.

This does not sample.  It proves the model is exact on *every* pair of n-digit
operands, by exploiting the fact that the bank gate saturates:

  1. Check, over all 100 digit pairs, that the clamp bank is fully saturated and
     that its gate is exactly (0,0) on absorb (a+b <= 8), (1,0) on transparent
     (a+b == 9) and (1,1) on generate (a+b >= 10).  Once that holds, the key and
     the value at a place depend only on the place's carry class.

  2. Enumerate all 3^n carry-class patterns.  For each, the attention scores are
     exactly determined, so evaluate both softmaxes in float64 -- this accounts
     for attention leakage exactly rather than bounding it -- and read off the
     carry-in and carry-out each head produces at every position.

  3. For every class pattern, every position and every digit pair consistent
     with that position's class, check that the residual decodes to the correct
     answer digit, and record the worst read-out margin.

Steps 2 and 3 together cover every operand pair, because a pair's behaviour
depends only on its class pattern and its per-place digits.
"""

import argparse
import importlib.util
import itertools
import torch

ABSORB, TRANSPARENT, GENERATE = 0, 1, 2


def load(path):
    spec = importlib.util.spec_from_file_location("shipped", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, meta = mod.build_model()
    return mod, model, meta


def bank_check(model):
    """Step 1: exact saturation and the correct three-way split."""
    c = model.prototypes().double()
    a = torch.arange(10)[:, None].expand(10, 10).reshape(-1)
    b = torch.arange(10)[None, :].expand(10, 10).reshape(-1)
    x = c[a] + c[b]                                              # [100]
    e = model.bank_w.double() * x[:, None] + model.knee.double()  # [100,2]
    g = e.clamp(0.0, 1.0)

    s = a + b
    cls = torch.where(s <= 8, ABSORB, torch.where(s == 9, TRANSPARENT, GENERATE))
    want = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    ok = (g == want[cls]).all()
    # how far every unit is from leaving saturation (>0 means strictly saturated)
    slack = torch.maximum(-e, e - 1.0).min()
    return bool(ok), float(slack), c, cls


def carry_class_tables(model, cls_all, P):
    """Step 2: exact head outputs for every class pattern."""
    kw, vw = model.key_w.double(), model.val_w.double()
    gate = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    k_of = gate @ kw                       # per class
    v_of = gate @ vw
    lam = model.lam.double()

    k = k_of[cls_all]                      # [N,P]
    v = v_of[cls_all]
    i = torch.arange(P)
    dist = (i[:, None] - i[None, :]).double()
    mA = dist > 0
    mA = mA.clone(); mA[0, 0] = True
    mB = dist >= 0
    s = k[:, None, :] + lam * dist
    neg = torch.finfo(torch.float64).min
    wA = torch.softmax(s.masked_fill(~mA, neg), -1)
    wB = torch.softmax(s.masked_fill(~mB, neg), -1)
    cA = (wA * v[:, None, :]).sum(-1)
    cB = (wB * v[:, None, :]).sum(-1)
    return cA, cB


def true_carries(cls_all, P):
    """carry[:, i] = the carry into position i (position i holds place i-1)."""
    N = cls_all.shape[0]
    carry = torch.zeros(N, P + 1, dtype=torch.long)
    for i in range(1, P):
        c = cls_all[:, i]
        carry[:, i + 1] = torch.where(c == GENERATE, torch.ones_like(carry[:, i]),
                                      torch.where(c == TRANSPARENT, carry[:, i],
                                                  torch.zeros_like(carry[:, i])))
    return carry


def certify_width(model, n, code, cls_pairs, chunk=512, verbose=True):
    """Exhaustive certificate for n-digit operands.  Returns (n_checks, n_wrong,
    worst margin)."""
    P = n + 2
    pats = torch.tensor(list(itertools.product([ABSORB, TRANSPARENT, GENERATE],
                                               repeat=n)), dtype=torch.long)
    N = pats.shape[0]
    cls_all = torch.zeros(N, P, dtype=torch.long)     # pads are absorb ((0,0))
    cls_all[:, 1:n + 1] = pats

    pair_a = torch.arange(10)[:, None].expand(10, 10).reshape(-1)
    pair_b = torch.arange(10)[None, :].expand(10, 10).reshape(-1)

    carry_w = model.carry_w.double()
    fold = model.fold.double()

    total, wrong = 0, 0
    worst = float("inf")
    for lo in range(0, N, chunk):
        sub = cls_all[lo:lo + chunk]
        cA, cB = carry_class_tables(model, sub, P)
        carry = true_carries(sub, P)
        for i in range(1, P):
            base = carry_w * cA[:, i] + fold * cB[:, i]           # [M]
            cin = carry[:, i]                                     # [M]
            for cl in (ABSORB, TRANSPARENT, GENERATE):
                rows = (sub[:, i] == cl).nonzero().squeeze(1)
                if rows.numel() == 0:
                    continue
                sel = cls_pairs[cl]
                aa, bb = pair_a[sel], pair_b[sel]                  # [K]
                y = (code[aa] + code[bb])[None, :] + base[rows][:, None]   # [M,K]
                t = (aa + bb)[None, :] + cin[rows][:, None]
                t = t % 10
                d = (y[:, :, None] - code[None, None, :]).abs()    # [M,K,10]
                dt = d.gather(-1, t[..., None]).squeeze(-1)
                do = d.scatter(-1, t[..., None], float("inf")).amin(-1)
                m = 0.5 * (do - dt)
                wrong += int((m <= 0).sum())
                worst = min(worst, float(m.min()))
                total += m.numel()
    if verbose:
        print(f"  n={n:2d}  patterns {N:6d}  checks {total:9d}  wrong {wrong}  "
              f"worst margin {worst:.6f}")
    return total, wrong, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--widths", default="1,2,3,4,5,6,7,8,9")
    a = ap.parse_args()

    mod, model, meta = load(a.path)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"file {a.path}  parameters {n_par}")
    print("  code :", [round(float(x), 6) for x in model.prototypes()])
    print("  knee :", [round(float(x), 6) for x in model.knee])
    print("  fold :", round(float(model.fold), 6))

    ok, slack, code, cls = bank_check(model)
    print(f"step 1  bank saturated and split correctly: {ok}   slack {slack:.4f}")
    if not ok:
        print("CERTIFICATE FAILED at step 1")
        return 1

    cls_pairs = {c: (cls == c).nonzero().squeeze(1) for c in (0, 1, 2)}
    print("step 2+3  exhaustive over all carry-class patterns")
    allok = True
    tot_checks = 0
    for n in [int(x) for x in a.widths.split(",")]:
        t, w, m = certify_width(model, n, code, cls_pairs)
        tot_checks += t
        allok &= (w == 0)
    step = float(code[1] - code[0])
    print(f"\nCERTIFICATE {'PASSED' if allok else 'FAILED'}  "
          f"({tot_checks} checks, code step {step:.4f})")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
