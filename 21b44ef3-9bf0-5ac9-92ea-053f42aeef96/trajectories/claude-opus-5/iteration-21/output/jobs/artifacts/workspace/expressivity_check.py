"""Can this architecture express exact addition at all?

HAND-SET WEIGHTS.  Nothing here is trained and nothing here is ever shipped:
this script exists only to separate "the architecture cannot do the task" from
"training has not found it yet", which are very different problems to be stuck
on.  The shipped weights come from training/, via lottery.py -> refine.py ->
finish.py, and are written by build.py.

The reference setting is the one the design was drawn from: code[d] = d, a
carry writes +1, the gate thresholds sit either side of a + b = 9, and an
outgoing carry folds off 10.
"""

import torch

import model_src


def reference_model():
    m = model_src.DigitPairAdder()
    with torch.no_grad():
        m.code_free.copy_(torch.arange(2.0, 10.0))   # code[d] = d for d = 2..9
        m.carry_w.fill_(1.0)
        m.knee.copy_(torch.tensor([8.5, 9.5]))
        m.fold.fill_(-10.0)
    return m.eval()


def main():
    torch.manual_seed(0)
    m = reference_model()
    print("hand-set reference (NOT trained, NOT shipped)")
    for n in (1, 2, 3, 5, 8, 12):
        a = torch.randint(0, 10, (4096, n))
        b = torch.randint(0, 10, (4096, n))
        pairs = torch.stack([a, b], dim=-1)
        with torch.no_grad():
            pred = m(pairs).argmax(-1)[:, 1:]
        p10 = torch.pow(10, torch.arange(n, dtype=torch.int64))
        want = (a * p10).sum(1) + (b * p10).sum(1)
        got = (pred * torch.pow(10, torch.arange(n + 1, dtype=torch.int64))).sum(1)
        print(f"  width {n:2d}: exact {(got == want).float().mean().item():.4f}")


if __name__ == "__main__":
    main()
