import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer

ROOT = Path('/workspace')
DEVICE = 'cuda'
LO, HI, MOD = 10_000_000, 99_999_999, 100_000_000
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def uniform(n):
    a = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    b = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    return a, b


def structured(n):
    a, b = uniform(n)
    kind = torch.randint(0, 8, (n,), device=DEVICE)
    # Complements and near-complements exercise carries through every position.
    m = kind == 0
    aa = torch.randint(LO, HI + 1, (n,), device=DEVICE)
    delta = torch.randint(-100, 101, (n,), device=DEVICE)
    a[m] = aa[m]
    b[m] = (MOD - aa + delta).clamp(LO, HI)[m]
    # Long asymmetric suffixes of nines / zeros.
    for typ in (1, 2):
        m = kind == typ
        k = torch.randint(1, 8, (n,), device=DEVICE)
        p = POW10[k]
        prefix = torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000
        tail9 = p - 1
        x = (prefix + tail9).clamp(LO, HI)
        y = torch.randint(LO, HI + 1, (n,), device=DEVICE)
        if typ == 1:
            a[m], b[m] = x[m], y[m]
        else:
            a[m], b[m] = y[m], x[m]
    # Values close to decimal boundaries at all scales.
    m = kind == 3
    k = torch.randint(1, 9, (n,), device=DEVICE)
    p = POW10[k]
    qmax = torch.div(HI, p, rounding_mode='floor')
    q = (torch.rand(n, device=DEVICE) * qmax).long() + 1
    x = q * p + torch.randint(-10, 11, (n,), device=DEVICE)
    a[m] = x.clamp(LO, HI)[m]
    # Repeated digits.
    m = kind == 4
    d = torch.randint(1, 10, (n,), device=DEVICE)
    x = d * 11_111_111
    a[m] = x[m]
    d2 = torch.randint(1, 10, (n,), device=DEVICE)
    b[m] = (d2 * 11_111_111)[m]
    # Sparse and dense 0/9 patterns.
    m = kind == 5
    digits = torch.where(torch.rand(n, 8, device=DEVICE) < .5, 0, 9)
    digits[:, 7] = torch.randint(1, 10, (n,), device=DEVICE)
    x = (digits * POW10[:8]).sum(1)
    a[m] = x[m]
    # Near extrema.
    m = kind == 6
    low = LO + torch.randint(0, 10000, (n,), device=DEVICE)
    high = HI - torch.randint(0, 10000, (n,), device=DEVICE)
    choose = torch.rand(n, device=DEVICE) < .5
    a[m] = torch.where(choose, low, high)[m]
    b[m] = torch.where(choose, high, low)[m]
    # Deliberately unequal long carry runs.
    m = kind == 7
    k = torch.randint(2, 8, (n,), device=DEVICE)
    p = POW10[k]
    left = torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000 + (p - 1)
    right = torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000 + torch.randint(1, 10, (n,), device=DEVICE)
    a[m] = left.clamp(LO, HI)[m]
    b[m] = right.clamp(LO, HI)[m]
    return a, b


def batch(n, structured_fraction=.18):
    a, b = uniform(n)
    count = int(n * structured_fraction)
    if count:
        a[:count], b[:count] = structured(count)
    # Randomly swap to prevent slot specialization.
    swap = torch.rand(n, device=DEVICE) < .5
    return torch.where(swap, b, a), torch.where(swap, a, b)


def encode(a, b):
    n = a.numel()
    ad = (a[:, None] // POW10[:8] % 10)
    bd = (b[:, None] // POW10[:8] % 10)
    out = ((a + b)[:, None] // POW10 % 10)
    prefix = torch.empty(n, 17, dtype=torch.long, device=DEVICE)
    prefix[:, 0:16:2] = ad
    prefix[:, 1:16:2] = bd
    prefix[:, 16] = 10
    x = torch.cat((prefix, out[:, :8]), 1)
    return x, out


@torch.no_grad()
def accuracy(model, n=100000, structured_data=False, chunk=10000):
    model.eval()
    good = total = 0
    min_margin = 1e9
    while total < n:
        bs = min(chunk, n - total)
        a, b = structured(bs) if structured_data else uniform(bs)
        x, target = encode(a, b)
        tokens = x[:, :17]
        pred = []
        for _ in range(9):
            logits = model(tokens)[:, -1]
            top = logits.topk(2, dim=-1).values
            min_margin = min(min_margin, float((top[:, 0] - top[:, 1]).min()))
            digit = logits.argmax(-1)
            pred.append(digit)
            tokens = torch.cat((tokens, digit[:, None]), 1)
        pred = torch.stack(pred, 1)
        good += int((pred == target).all(1).sum())
        total += bs
    model.train()
    return good, total, min_margin


def train_steps(model, steps, lr, sf, label, batch_size=8192):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=.01)
    start = time.time()
    for step in range(1, steps + 1):
        a, b = batch(batch_size, sf)
        x, y = encode(a, b)
        logits = model(x)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == steps:
            elapsed = time.time() - start
            print(f'{label} {step}/{steps} loss={loss.item():.6f} sec={elapsed:.1f}', flush=True)
        if step % 6000 == 0 or step == steps:
            torch.save(model.state_dict(), ROOT / f'{label}_{step}.pt')


def prune(model, width):
    old = model.ff1.shape[1]
    assert width == old - 1
    score = model.ff1.norm(dim=0) * model.ff2.norm(dim=1)
    keep = [i for i in range(old) if i != int(score.argmin())]
    result = AdditionTransformer(ff_width=width).to(DEVICE)
    source = model.state_dict()
    dest = result.state_dict()
    for name in dest:
        if name == 'ff1':
            dest[name].copy_(source[name][:, keep])
        elif name == 'ff2':
            dest[name].copy_(source[name][keep, :])
        else:
            dest[name].copy_(source[name])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--steps', type=int, default=36000)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--sf', type=float, default=.18)
    parser.add_argument('--label', default='teacher')
    parser.add_argument('--prune-to', type=int)
    args = parser.parse_args()
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.set_float32_matmul_precision('high')
    model = AdditionTransformer(ff_width=args.width).to(DEVICE)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE, weights_only=True))
    if args.prune_to is not None:
        model = prune(model, args.prune_to)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    train_steps(model, args.steps, args.lr, args.sf, args.label)
    torch.save(model.state_dict(), ROOT / f'{args.label}_final.pt')
    for structured_data in (False, True):
        print('validation', structured_data, accuracy(model, 100000, structured_data), flush=True)


if __name__ == '__main__':
    main()
