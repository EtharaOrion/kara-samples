"""How close does the shipped model come to getting a digit wrong?

Accuracy on a sample only says "no mistakes seen".  This reports the smallest
gap between the winning logit and the best rival over a large stress sample, so
a near-miss shows up before an unseen pair finds it.
"""
import argparse, importlib.util
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--sub', default='/workspace/submission.py')
ap.add_argument('--n', type=int, default=2_000_000)
ap.add_argument('--bs', type=int, default=8192)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location('sub', args.sub)
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model, meta = sub.build_model()
model = model.to(dev)
print('parameters', meta['parameters'])

g = torch.Generator(device=dev); g.manual_seed(99)
NPOS, L = sub.NPOS, sub.L
worst, worst_case, wrong = float('inf'), None, 0
scale = 0.0
for done in range(0, args.n, args.bs):
    b = min(args.bs, args.n - done)
    # stress: digits drawn to make long propagate runs common
    a_d = torch.randint(0, 10, (b, NPOS), device=dev, generator=g)
    b_d = torch.randint(0, 10, (b, NPOS), device=dev, generator=g)
    mask = torch.rand((b, NPOS), device=dev, generator=g) < 0.6
    b_d = torch.where(mask, 9 - a_d, b_d)                     # force digit sums of 9
    gen = torch.rand((b, 1), device=dev, generator=g) < 0.5
    b_d[:, 0] = torch.where(gen[:, 0], 9 - a_d[:, 0] + 1, b_d[:, 0]).clamp(0, 9)
    a_d[:, NPOS - 1] = 0; b_d[:, NPOS - 1] = 0

    s = a_d + b_d
    carry = torch.zeros(b, dtype=torch.long, device=dev)
    tgt = torch.zeros(b, NPOS, dtype=torch.long, device=dev)
    for i in range(NPOS):
        t = s[:, i] + carry
        tgt[:, i] = t % 10
        carry = t // 10

    a_tok = torch.zeros(b, L, dtype=torch.long, device=dev)
    b_tok = torch.zeros(b, L, dtype=torch.long, device=dev)
    a_tok[:, 1:] = a_d; b_tok[:, 1:] = b_d
    logits = model(a_tok, b_tok)[:, 1:]
    scale = max(scale, float(logits.abs().max()))
    good = logits.gather(-1, tgt[..., None]).squeeze(-1)
    rival = logits.masked_fill(
        torch.nn.functional.one_hot(tgt, 10).bool(), float('-inf')).max(-1).values
    gap = good - rival
    wrong += int((gap <= 0).sum())
    m, idx = gap.flatten().min(0)
    if float(m) < worst:
        worst = float(m)
        r, c = divmod(int(idx), NPOS)
        worst_case = (int(''.join(str(int(d)) for d in a_d[r].flip(0))),
                      int(''.join(str(int(d)) for d in b_d[r].flip(0))), c)

print(f'{args.n} stress pairs: {wrong} wrong digits')
print(f'smallest winning margin {worst:.4f} (logits reach {scale:.1f})')
print('tightest case: a={} b={} digit {}'.format(*worst_case))
